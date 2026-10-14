import argparse
import json
import platform
import statistics
import time
import warnings
from pathlib import Path

import datasets
import numpy as np
import psutil
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoModelForTDT, AutoProcessor

from experiments_asr.asr_01_fleurs_subset import load_audio_from_entry, load_manifest, load_metadata, validate_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = PROJECT_ROOT / "results" / "asr_selection" / "performance"

MAX_WHISPER_SHORTFORM_S = 30.0

WARMUP_SAMPLES = 10


MODELS = {
    "parakeet": {
        "model_id": "nvidia/parakeet-tdt-0.6b-v3",
        "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    },
    "whisper": {
        "model_id": "openai/whisper-small",
        "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
    },
}


WHISPER_LANGUAGES = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS.keys(),
    )

    parser.add_argument(
        "--precision",
        required=True,
        choices=["fp32", "fp16"],
    )

    return parser.parse_args()


def get_dtype(precision: str):
    if precision == "fp32":
        return torch.float32

    if precision == "fp16":
        return torch.float16

    raise ValueError(f"Unsupported precision: {precision}")


def bytes_to_mib(value: int) -> float:
    return value / (1024**2)


def process_rss_mib() -> float:
    process = psutil.Process()

    return bytes_to_mib(process.memory_info().rss)


def calculate_model_size(model) -> dict:
    parameters = sum(parameter.numel() for parameter in model.parameters())

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    return {
        "parameters": parameters,
        "parameter_memory_mib": bytes_to_mib(parameter_bytes),
        "buffer_memory_mib": bytes_to_mib(buffer_bytes),
    }


def load_model(
    model_name: str,
    precision: str,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    config = MODELS[model_name]

    dtype = get_dtype(precision)

    print("=" * 70)
    print("ASR PERFORMANCE BENCHMARK")
    print("=" * 70)

    print(f"Candidate      : {model_name}")
    print(f"Model          : {config['model_id']}")
    print(f"Revision       : {config['revision']}")
    print(f"Precision      : {precision.upper()}")
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print()

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        config["model_id"],
        revision=config["revision"],
    )

    print("Loading model...")

    if model_name == "parakeet":
        model = AutoModelForTDT.from_pretrained(
            config["model_id"],
            revision=config["revision"],
            dtype=dtype,
        )

    elif model_name == "whisper":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            config["model_id"],
            revision=config["revision"],
            dtype=dtype,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to("cuda")
    model.eval()

    torch.cuda.synchronize()

    size_info = calculate_model_size(model)

    gpu_allocated_mib = bytes_to_mib(torch.cuda.memory_allocated())

    gpu_reserved_mib = bytes_to_mib(torch.cuda.memory_reserved())

    ram_rss_mib = process_rss_mib()

    print()
    print("Model loaded.")
    print(f"Parameters     : {size_info['parameters']:,}")
    print(f"Parameter mem  : {size_info['parameter_memory_mib']:.2f} MiB")
    print(f"GPU allocated  : {gpu_allocated_mib:.2f} MiB")
    print(f"GPU reserved   : {gpu_reserved_mib:.2f} MiB")
    print(f"Process RSS    : {ram_rss_mib:.2f} MiB")
    print()

    resource_info = {
        **size_info,
        "gpu_after_load_allocated_mib": (gpu_allocated_mib),
        "gpu_after_load_reserved_mib": (gpu_reserved_mib),
        "process_after_load_rss_mib": (ram_rss_mib),
    }

    return (
        processor,
        model,
        resource_info,
    )


def prepare_inputs(
    model_name: str,
    samples,
    sample_rate: int,
    processor,
    model,
):
    if model_name == "whisper":
        inputs = processor(
            samples,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=False,
        )
    else:
        inputs = processor(
            samples,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

    inputs = inputs.to(
        model.device,
        dtype=model.dtype,
    )

    return inputs


def generate_parakeet(
    inputs,
    model,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Using the model-agnostic default `max_length`.*",
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                return_dict_in_generate=True,
            )

    return output.sequences


def generate_whisper(
    inputs,
    model,
    language: str,
    duration_s: float,
):
    is_longform = duration_s > MAX_WHISPER_SHORTFORM_S

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            language=WHISPER_LANGUAGES[language],
            task="transcribe",
            return_timestamps=is_longform,
        )

    return generated_ids


def decode_output(
    model_name: str,
    generated_ids,
    processor,
):
    if model_name == "parakeet":
        transcriptions = processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

    elif model_name == "whisper":
        transcriptions = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return transcriptions[0].strip()


def run_single(
    model_name: str,
    entry: dict,
    metadata: dict,
    processor,
    model,
) -> dict:
    # Dataset access / audio decoding happens outside
    # the timed ASR pipeline.
    samples, sample_rate, reference = load_audio_from_entry(
        entry,
        metadata,
    )

    duration_s = len(samples) / sample_rate

    torch.cuda.synchronize()

    rss_before = process_rss_mib()

    #
    # PREPROCESSING
    #
    preprocess_start = time.perf_counter()

    inputs = prepare_inputs(
        model_name=model_name,
        samples=samples,
        sample_rate=sample_rate,
        processor=processor,
        model=model,
    )

    # Ensure host -> device transfer is complete.
    torch.cuda.synchronize()

    preprocess_end = time.perf_counter()

    rss_after_preprocess = process_rss_mib()

    #
    # INFERENCE
    #
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    inference_start = time.perf_counter()

    if model_name == "parakeet":
        generated_ids = generate_parakeet(
            inputs=inputs,
            model=model,
        )

    elif model_name == "whisper":
        generated_ids = generate_whisper(
            inputs=inputs,
            model=model,
            language=entry["language"],
            duration_s=duration_s,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    torch.cuda.synchronize()

    inference_end = time.perf_counter()

    peak_allocated_mib = bytes_to_mib(torch.cuda.max_memory_allocated())

    peak_reserved_mib = bytes_to_mib(torch.cuda.max_memory_reserved())

    rss_after_inference = process_rss_mib()

    #
    # DECODING
    #
    decode_start = time.perf_counter()

    prediction = decode_output(
        model_name=model_name,
        generated_ids=generated_ids,
        processor=processor,
    )

    decode_end = time.perf_counter()

    rss_after_decode = process_rss_mib()

    preprocess_ms = (preprocess_end - preprocess_start) * 1000.0

    inference_ms = (inference_end - inference_start) * 1000.0

    decode_ms = (decode_end - decode_start) * 1000.0

    total_ms = preprocess_ms + inference_ms + decode_ms

    total_s = total_ms / 1000.0
    inference_s = inference_ms / 1000.0

    observed_rss_mib = max(
        rss_before,
        rss_after_preprocess,
        rss_after_inference,
        rss_after_decode,
    )

    return {
        "language": entry["language"],
        "dataset_index": entry["dataset_index"],
        "sample_id": entry["sample_id"],
        "duration_s": duration_s,
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "inference_rtf": (inference_s / duration_s),
        "end_to_end_rtf": (total_s / duration_s),
        "gpu_peak_allocated_mib": (peak_allocated_mib),
        "gpu_peak_reserved_mib": (peak_reserved_mib),
        "process_observed_rss_mib": (observed_rss_mib),
        # Stored only as a sanity check.
        # We do not calculate WER in this benchmark.
        "reference": reference,
        "prediction": prediction,
    }


def run_warmup(
    model_name: str,
    entries: list[dict],
    metadata: dict,
    processor,
    model,
):
    print(f"Warm-up        : {WARMUP_SAMPLES} utterances")

    for entry in tqdm(
        entries[:WARMUP_SAMPLES],
        desc="Warm-up",
        unit="utterance",
    ):
        run_single(
            model_name=model_name,
            entry=entry,
            metadata=metadata,
            processor=processor,
            model=model,
        )

    torch.cuda.synchronize()

    print()


def metric_stats(
    values: list[float],
) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "min": min(values),
        "max": max(values),
    }


def summarize_records(
    records: list[dict],
) -> dict:
    latency_keys = [
        "preprocess_ms",
        "inference_ms",
        "decode_ms",
        "total_ms",
        "inference_rtf",
        "end_to_end_rtf",
    ]

    summary = {}

    for key in latency_keys:
        summary[key] = metric_stats([record[key] for record in records])

    summary["gpu_peak_allocated_mib"] = {"max": max(record["gpu_peak_allocated_mib"] for record in records)}

    summary["gpu_peak_reserved_mib"] = {"max": max(record["gpu_peak_reserved_mib"] for record in records)}

    summary["process_observed_rss_mib"] = {"max": max(record["process_observed_rss_mib"] for record in records)}

    return summary


def build_summary(
    args,
    records: list[dict],
    metadata: dict,
    resource_info: dict,
):
    by_language = {}

    for language in metadata["languages"]:
        language_records = [record for record in records if record["language"] == language]

        by_language[language] = summarize_records(language_records)

    return {
        "experiment": {
            "model_name": args.model,
            "model_id": MODELS[args.model]["model_id"],
            "model_revision": MODELS[args.model]["revision"],
            "precision": args.precision,
            "samples": len(records),
            "warmup_samples": (WARMUP_SAMPLES),
            "timing_scope": ("preprocess + inference + decode; dataset I/O excluded"),
        },
        "dataset": {
            "name": metadata["dataset"],
            "revision": metadata["dataset_revision"],
            "manifest_sha256": metadata["manifest_sha256"],
        },
        "environment": {
            "python": (platform.python_version()),
            "torch": torch.__version__,
            "transformers": (transformers.__version__),
            "datasets": (datasets.__version__),
            "numpy": np.__version__,
            "cuda_runtime": (torch.version.cuda),
            "gpu": (torch.cuda.get_device_name(0)),
        },
        "model_resources": (resource_info),
        "overall": summarize_records(records),
        "languages": by_language,
    }


def print_metric(
    name: str,
    stats: dict,
    unit: str,
):
    print(
        f"{name:20} | "
        f"mean {stats['mean']:9.3f} {unit} | "
        f"median {stats['median']:9.3f} | "
        f"std {stats['std']:9.3f} | "
        f"p95 {stats['p95']:9.3f}"
    )


def print_summary(
    summary: dict,
):
    overall = summary["overall"]
    resources = summary["model_resources"]

    print()
    print("=" * 100)
    print("PERFORMANCE RESULTS")
    print("=" * 100)

    print_metric(
        "Preprocess",
        overall["preprocess_ms"],
        "ms",
    )

    print_metric(
        "Inference",
        overall["inference_ms"],
        "ms",
    )

    print_metric(
        "Decode",
        overall["decode_ms"],
        "ms",
    )

    print_metric(
        "Total latency",
        overall["total_ms"],
        "ms",
    )

    print_metric(
        "Inference RTF",
        overall["inference_rtf"],
        "",
    )

    print_metric(
        "End-to-end RTF",
        overall["end_to_end_rtf"],
        "",
    )

    print("-" * 100)

    print(f"Model parameters        : {resources['parameters']:,}")

    print(f"Parameter memory        : {resources['parameter_memory_mib']:.2f} MiB")

    print(f"GPU after load allocated: {resources['gpu_after_load_allocated_mib']:.2f} MiB")

    print(f"GPU after load reserved : {resources['gpu_after_load_reserved_mib']:.2f} MiB")

    print(f"GPU peak allocated      : {overall['gpu_peak_allocated_mib']['max']:.2f} MiB")

    print(f"GPU peak reserved       : {overall['gpu_peak_reserved_mib']['max']:.2f} MiB")

    print(f"Process RSS after load  : {resources['process_after_load_rss_mib']:.2f} MiB")

    print(f"Max observed process RSS: {overall['process_observed_rss_mib']['max']:.2f} MiB")

    print("=" * 100)


def main():
    args = parse_args()

    metadata = load_metadata()
    entries = load_manifest()

    validate_manifest(
        entries,
        metadata,
    )

    processor, model, resource_info = load_model(
        model_name=args.model,
        precision=args.precision,
    )

    run_warmup(
        model_name=args.model,
        entries=entries,
        metadata=metadata,
        processor=processor,
        model=model,
    )

    records = []

    print(f"Benchmark      : {len(entries)} utterances")
    print()

    for entry in tqdm(
        entries,
        desc="Benchmark",
        unit="utterance",
    ):
        record = run_single(
            model_name=args.model,
            entry=entry,
            metadata=metadata,
            processor=processor,
            model=model,
        )

        records.append(record)

    summary = build_summary(
        args=args,
        records=records,
        metadata=metadata,
        resource_info=resource_info,
    )

    output_dir = RESULTS_ROOT / args.model / args.precision

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records_path = output_dir / "fleurs_performance.jsonl"

    summary_path = output_dir / "fleurs_performance_summary.json"

    with records_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(summary)

    print()
    print(f"Per-sample results: {records_path}")

    print(f"Summary           : {summary_path}")


if __name__ == "__main__":
    main()
