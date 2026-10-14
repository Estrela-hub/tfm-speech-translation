from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import psutil
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore


# ---------------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda"
DTYPE = torch.float16

EXPECTED_SAMPLE_COUNT = 200

EXPECTED_MANIFEST_SHA256 = "b2e3f73365232c700a1246b7bf4726eba2401f743a6a644066c5f35d4c470399"

EXPECTED_DATASET_REVISION = "71abf77d8b7beb5cfef59898d6b24d92ab7654fc"

MAX_GENERATION_LENGTH = 512

WARMUP_RUNS = 10


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

M2M100_MODEL_ID = "facebook/m2m100_418M"
M2M100_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGUAGES = ["EN", "ES", "FR", "DE", "PT"]

FLORES_LANGUAGE_CODES = {
    "EN": "eng_Latn",
    "ES": "spa_Latn",
    "FR": "fra_Latn",
    "DE": "deu_Latn",
    "PT": "por_Latn",
}

NLLB_LANGUAGE_CODES = {
    "EN": "eng_Latn",
    "ES": "spa_Latn",
    "FR": "fra_Latn",
    "DE": "deu_Latn",
    "PT": "por_Latn",
}

M2M100_LANGUAGE_CODES = {
    "EN": "en",
    "ES": "es",
    "FR": "fr",
    "DE": "de",
    "PT": "pt",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "mt_selection" / "flores_dev_200.jsonl"

DATASET_METADATA_PATH = PROJECT_ROOT / "data" / "mt_selection" / "flores_dev_200_metadata.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "performance_fp16"

SAMPLES_CSV_PATH = RESULTS_DIR / "performance_samples.csv"
DIRECTIONS_CSV_PATH = RESULTS_DIR / "performance_by_direction.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "performance_summary.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bytes_to_mb(value: int | float) -> float:
    return value / (1024**2)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of an empty list.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100.0

    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    fraction = position - lower_index

    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def process_rss_mb() -> float:
    process = psutil.Process()
    return bytes_to_mb(process.memory_info().rss)


def parameter_memory_mb(model) -> float:
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    return bytes_to_mb(total_bytes)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_and_validate_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found:\n{MANIFEST_PATH}")

    if not DATASET_METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found:\n{DATASET_METADATA_PATH}")

    actual_sha256 = calculate_sha256(MANIFEST_PATH)

    if actual_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest SHA256 mismatch.\nExpected: {EXPECTED_MANIFEST_SHA256}\nActual  : {actual_sha256}"
        )

    with DATASET_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    actual_revision = metadata.get("dataset_revision")

    if actual_revision != EXPECTED_DATASET_REVISION:
        raise RuntimeError(
            f"Dataset revision mismatch.\nExpected: {EXPECTED_DATASET_REVISION}\nActual  : {actual_revision}"
        )

    records = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLE_COUNT} records, found {len(records)}.")

    return records


# ---------------------------------------------------------------------------
# Model-specific language handling
# ---------------------------------------------------------------------------


def set_source_language(
    model_name: str,
    tokenizer,
    source_language: str,
) -> None:
    if model_name == "NLLB":
        tokenizer.src_lang = NLLB_LANGUAGE_CODES[source_language]

    elif model_name == "M2M100":
        tokenizer.src_lang = M2M100_LANGUAGE_CODES[source_language]

    else:
        raise RuntimeError(f"Unknown model: {model_name}")


def get_target_token_id(
    model_name: str,
    tokenizer,
    target_language: str,
) -> int:
    if model_name == "NLLB":
        token_id = tokenizer.convert_tokens_to_ids(NLLB_LANGUAGE_CODES[target_language])

        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"Unknown NLLB language token for {target_language}.")

        return token_id

    if model_name == "M2M100":
        return tokenizer.get_lang_id(M2M100_LANGUAGE_CODES[target_language])

    raise RuntimeError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Single inference
# ---------------------------------------------------------------------------


def run_translation(
    model_name: str,
    model,
    tokenizer,
    source_text: str,
    source_language: str,
    target_language: str,
) -> dict:
    set_source_language(
        model_name,
        tokenizer,
        source_language,
    )

    total_start = time.perf_counter()

    # ---------------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------------

    preprocessing_start = time.perf_counter()

    inputs = tokenizer(
        source_text,
        return_tensors="pt",
    )

    inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}

    preprocessing_end = time.perf_counter()

    source_tokens = int(inputs["input_ids"].shape[-1])

    target_token_id = get_target_token_id(
        model_name,
        tokenizer,
        target_language,
    )

    # ---------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------

    torch.cuda.synchronize()

    generation_start = time.perf_counter()

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_token_id,
        num_beams=1,
        do_sample=False,
        max_length=MAX_GENERATION_LENGTH,
    )

    torch.cuda.synchronize()

    generation_end = time.perf_counter()

    # ---------------------------------------------------------------
    # Decode
    # ---------------------------------------------------------------

    decoding_start = time.perf_counter()

    hypothesis = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

    decoding_end = time.perf_counter()

    total_end = decoding_end

    generated_token_count = max(
        int(generated_tokens.shape[-1]) - 1,
        0,
    )

    preprocessing_ms = (preprocessing_end - preprocessing_start) * 1000.0

    generation_ms = (generation_end - generation_start) * 1000.0

    decoding_ms = (decoding_end - decoding_start) * 1000.0

    total_ms = (total_end - total_start) * 1000.0

    if generation_ms > 0:
        generated_tokens_per_second = generated_token_count / (generation_ms / 1000.0)
    else:
        generated_tokens_per_second = 0.0

    return {
        "preprocessing_ms": preprocessing_ms,
        "generation_ms": generation_ms,
        "decoding_ms": decoding_ms,
        "total_ms": total_ms,
        "source_tokens": source_tokens,
        "generated_tokens": generated_token_count,
        "generated_tokens_per_second": generated_tokens_per_second,
        "hypothesis": hypothesis,
    }


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def warmup_model(
    model_name: str,
    model,
    tokenizer,
    records: list[dict],
) -> None:
    print(f"Warm-up runs: {WARMUP_RUNS}")

    source_language = "EN"
    target_language = "PT"

    source_code = FLORES_LANGUAGE_CODES[source_language]

    with torch.inference_mode():
        for run_index in range(WARMUP_RUNS):
            record = records[run_index % len(records)]

            run_translation(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                source_text=record[source_code],
                source_language=source_language,
                target_language=target_language,
            )

    torch.cuda.synchronize()

    print("Warm-up completed.")


# ---------------------------------------------------------------------------
# Direction summary
# ---------------------------------------------------------------------------


def summarize_direction(
    model_name: str,
    source_language: str,
    target_language: str,
    samples: list[dict],
) -> dict:
    preprocessing = [row["preprocessing_ms"] for row in samples]

    generation = [row["generation_ms"] for row in samples]

    decoding = [row["decoding_ms"] for row in samples]

    total = [row["total_ms"] for row in samples]

    tokens_per_second = [row["generated_tokens_per_second"] for row in samples]

    return {
        "model": model_name,
        "source_language": source_language,
        "target_language": target_language,
        "samples": len(samples),
        "mean_preprocessing_ms": statistics.mean(preprocessing),
        "mean_generation_ms": statistics.mean(generation),
        "mean_decoding_ms": statistics.mean(decoding),
        "mean_total_ms": statistics.mean(total),
        "median_total_ms": statistics.median(total),
        "p95_total_ms": percentile(total, 95),
        "mean_generated_tokens_per_second": statistics.mean(tokens_per_second),
    }


# ---------------------------------------------------------------------------
# Model benchmark
# ---------------------------------------------------------------------------


def benchmark_model(
    model_name: str,
    model_id: str,
    revision: str,
    records: list[dict],
    sample_csv_writer,
    direction_csv_writer,
) -> dict:
    print()
    print("=" * 80)
    print(f"MODEL: {model_name} FP16")
    print("=" * 80)

    print(f"Model ID : {model_id}")
    print(f"Revision : {revision}")
    print("Loading tokenizer from local cache...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
    )

    print("Loading model directly in FP16 from local cache...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=DTYPE,
        use_safetensors=False,
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()

    torch.cuda.synchronize()

    model_device = str(next(model.parameters()).device)

    model_dtype = next(model.parameters()).dtype

    if model_device != "cuda:0":
        raise RuntimeError(f"{model_name} is not on cuda:0.")

    if model_dtype != torch.float16:
        raise RuntimeError(f"{model_name} is not FP16: {model_dtype}")

    parameters = sum(parameter.numel() for parameter in model.parameters())

    parameter_mb = parameter_memory_mb(model)

    print(f"Device           : {model_device}")
    print(f"Dtype            : {model_dtype}")
    print(f"Parameters       : {parameters:,}")
    print(f"Parameter memory : {parameter_mb:.2f} MB")

    # ---------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------

    warmup_model(
        model_name,
        model,
        tokenizer,
        records,
    )

    gc.collect()
    torch.cuda.synchronize()

    cuda_allocated_after_warmup_mb = bytes_to_mb(torch.cuda.memory_allocated())

    cuda_reserved_after_warmup_mb = bytes_to_mb(torch.cuda.memory_reserved())

    process_rss_after_warmup_mb = process_rss_mb()

    torch.cuda.reset_peak_memory_stats()

    max_observed_process_rss_mb = process_rss_after_warmup_mb

    print(f"CUDA allocated after warm-up: {cuda_allocated_after_warmup_mb:.2f} MB")

    print(f"CUDA reserved after warm-up : {cuda_reserved_after_warmup_mb:.2f} MB")

    print(f"Process RSS after warm-up   : {process_rss_after_warmup_mb:.2f} MB")

    # ---------------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------------

    direction_summaries = []

    all_total_latencies = []
    all_generation_latencies = []
    all_tokens_per_second = []

    total_directions = len(LANGUAGES) * (len(LANGUAGES) - 1)

    direction_number = 0

    with torch.inference_mode():
        for source_language in LANGUAGES:
            for target_language in LANGUAGES:
                if source_language == target_language:
                    continue

                direction_number += 1

                print()
                print(f"[{direction_number:02d}/{total_directions}] {source_language} -> {target_language}")

                source_code = FLORES_LANGUAGE_CODES[source_language]

                direction_samples = []

                for sample_index, record in enumerate(
                    records,
                    start=1,
                ):
                    result = run_translation(
                        model_name=model_name,
                        model=model,
                        tokenizer=tokenizer,
                        source_text=record[source_code],
                        source_language=source_language,
                        target_language=target_language,
                    )

                    if not result["hypothesis"].strip():
                        raise RuntimeError(
                            f"Empty FP16 translation produced by "
                            f"{model_name} for ID {record['id']} "
                            f"{source_language}->{target_language}."
                        )

                    current_rss_mb = process_rss_mb()

                    max_observed_process_rss_mb = max(
                        max_observed_process_rss_mb,
                        current_rss_mb,
                    )

                    sample_record = {
                        "model": model_name,
                        "id": record["id"],
                        "source_language": source_language,
                        "target_language": target_language,
                        "source_tokens": result["source_tokens"],
                        "generated_tokens": result["generated_tokens"],
                        "preprocessing_ms": result["preprocessing_ms"],
                        "generation_ms": result["generation_ms"],
                        "decoding_ms": result["decoding_ms"],
                        "total_ms": result["total_ms"],
                        "generated_tokens_per_second": (result["generated_tokens_per_second"]),
                    }

                    sample_csv_writer.writerow(sample_record)

                    direction_samples.append(sample_record)

                    all_total_latencies.append(result["total_ms"])

                    all_generation_latencies.append(result["generation_ms"])

                    all_tokens_per_second.append(result["generated_tokens_per_second"])

                    if sample_index % 50 == 0 or sample_index == len(records):
                        print(f"  Benchmarked: {sample_index}/{len(records)}")

                direction_summary = summarize_direction(
                    model_name=model_name,
                    source_language=source_language,
                    target_language=target_language,
                    samples=direction_samples,
                )

                direction_summaries.append(direction_summary)

                direction_csv_writer.writerow(direction_summary)

                print(f"  Mean total : {direction_summary['mean_total_ms']:.2f} ms")

                print(f"  Median     : {direction_summary['median_total_ms']:.2f} ms")

                print(f"  P95        : {direction_summary['p95_total_ms']:.2f} ms")

    torch.cuda.synchronize()

    peak_cuda_allocated_mb = bytes_to_mb(torch.cuda.max_memory_allocated())

    peak_cuda_reserved_mb = bytes_to_mb(torch.cuda.max_memory_reserved())

    macro_mean_total_ms = statistics.mean(row["mean_total_ms"] for row in direction_summaries)

    macro_median_total_ms = statistics.mean(row["median_total_ms"] for row in direction_summaries)

    macro_p95_total_ms = statistics.mean(row["p95_total_ms"] for row in direction_summaries)

    overall_mean_total_ms = statistics.mean(all_total_latencies)

    overall_median_total_ms = statistics.median(all_total_latencies)

    overall_p95_total_ms = percentile(
        all_total_latencies,
        95,
    )

    overall_mean_generation_ms = statistics.mean(all_generation_latencies)

    overall_mean_tokens_per_second = statistics.mean(all_tokens_per_second)

    print()
    print("-" * 80)
    print(f"{model_name} FP16 PERFORMANCE SUMMARY")
    print("-" * 80)

    print(f"Mean total latency    : {overall_mean_total_ms:.2f} ms")

    print(f"Median total latency  : {overall_median_total_ms:.2f} ms")

    print(f"P95 total latency     : {overall_p95_total_ms:.2f} ms")

    print(f"Mean generation       : {overall_mean_generation_ms:.2f} ms")

    print(f"Mean generated tok/s  : {overall_mean_tokens_per_second:.2f}")

    print(f"CUDA allocated steady : {cuda_allocated_after_warmup_mb:.2f} MB")

    print(f"Peak CUDA allocated   : {peak_cuda_allocated_mb:.2f} MB")

    print(f"Peak CUDA reserved    : {peak_cuda_reserved_mb:.2f} MB")

    print(f"Peak observed RSS     : {max_observed_process_rss_mb:.2f} MB")

    result = {
        "model": model_name,
        "model_id": model_id,
        "revision": revision,
        "parameters": parameters,
        "parameter_memory_mb": parameter_mb,
        "device": model_device,
        "dtype": str(model_dtype),
        "warmup_runs": WARMUP_RUNS,
        "cuda_allocated_after_warmup_mb": (cuda_allocated_after_warmup_mb),
        "cuda_reserved_after_warmup_mb": (cuda_reserved_after_warmup_mb),
        "peak_cuda_allocated_mb": (peak_cuda_allocated_mb),
        "peak_cuda_reserved_mb": (peak_cuda_reserved_mb),
        "process_rss_after_warmup_mb": (process_rss_after_warmup_mb),
        "peak_observed_process_rss_mb": (max_observed_process_rss_mb),
        "overall_mean_total_ms": (overall_mean_total_ms),
        "overall_median_total_ms": (overall_median_total_ms),
        "overall_p95_total_ms": (overall_p95_total_ms),
        "overall_mean_generation_ms": (overall_mean_generation_ms),
        "overall_mean_generated_tokens_per_second": (overall_mean_tokens_per_second),
        "macro_mean_total_ms": (macro_mean_total_ms),
        "macro_median_total_ms": (macro_median_total_ms),
        "macro_p95_total_ms": (macro_p95_total_ms),
        "directions": direction_summaries,
    }

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("MT FP16 PERFORMANCE BENCHMARK")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"Transformers : {transformers.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    print()
    print("Validating fixed FLORES manifest...")

    records = load_and_validate_manifest()

    print(f"Samples : {len(records)}")
    print(f"SHA256  : {EXPECTED_MANIFEST_SHA256}")
    print("Manifest status: PASS")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_fieldnames = [
        "model",
        "id",
        "source_language",
        "target_language",
        "source_tokens",
        "generated_tokens",
        "preprocessing_ms",
        "generation_ms",
        "decoding_ms",
        "total_ms",
        "generated_tokens_per_second",
    ]

    direction_fieldnames = [
        "model",
        "source_language",
        "target_language",
        "samples",
        "mean_preprocessing_ms",
        "mean_generation_ms",
        "mean_decoding_ms",
        "mean_total_ms",
        "median_total_ms",
        "p95_total_ms",
        "mean_generated_tokens_per_second",
    ]

    results = []

    with (
        SAMPLES_CSV_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as sample_csv_file,
        DIRECTIONS_CSV_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as direction_csv_file,
    ):
        sample_writer = csv.DictWriter(
            sample_csv_file,
            fieldnames=sample_fieldnames,
        )

        direction_writer = csv.DictWriter(
            direction_csv_file,
            fieldnames=direction_fieldnames,
        )

        sample_writer.writeheader()
        direction_writer.writeheader()

        nllb_result = benchmark_model(
            model_name="NLLB",
            model_id=NLLB_MODEL_ID,
            revision=NLLB_REVISION,
            records=records,
            sample_csv_writer=sample_writer,
            direction_csv_writer=direction_writer,
        )

        sample_csv_file.flush()
        direction_csv_file.flush()

        results.append(nllb_result)

        m2m100_result = benchmark_model(
            model_name="M2M100",
            model_id=M2M100_MODEL_ID,
            revision=M2M100_REVISION,
            records=records,
            sample_csv_writer=sample_writer,
            direction_csv_writer=direction_writer,
        )

        sample_csv_file.flush()
        direction_csv_file.flush()

        results.append(m2m100_result)

    summary = {
        "experiment": "MT model selection - FP16 performance",
        "dataset": {
            "dataset_id": "facebook/flores",
            "revision": EXPECTED_DATASET_REVISION,
            "split": "dev",
            "samples": EXPECTED_SAMPLE_COUNT,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "translation_directions": len(LANGUAGES) * (len(LANGUAGES) - 1),
        },
        "generation": {
            "batch_size": 1,
            "num_beams": 1,
            "do_sample": False,
            "max_length": MAX_GENERATION_LENGTH,
            "dtype": "float16",
            "warmup_runs": WARMUP_RUNS,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "models": results,
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("=" * 80)
    print("FINAL FP16 PERFORMANCE COMPARISON")
    print("=" * 80)

    for result in results:
        print()
        print(result["model"])

        print(f"  Mean latency       : {result['overall_mean_total_ms']:.2f} ms")

        print(f"  Median latency     : {result['overall_median_total_ms']:.2f} ms")

        print(f"  P95 latency        : {result['overall_p95_total_ms']:.2f} ms")

        print(f"  Generated tok/s    : {result['overall_mean_generated_tokens_per_second']:.2f}")

        print(f"  Parameter memory   : {result['parameter_memory_mb']:.2f} MB")

        print(f"  CUDA steady        : {result['cuda_allocated_after_warmup_mb']:.2f} MB")

        print(f"  CUDA peak          : {result['peak_cuda_allocated_mb']:.2f} MB")

        print(f"  Process RSS peak   : {result['peak_observed_process_rss_mb']:.2f} MB")

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Samples    : {SAMPLES_CSV_PATH}")
    print(f"Directions : {DIRECTIONS_CSV_PATH}")
    print(f"Summary    : {SUMMARY_JSON_PATH}")

    print()
    print("=" * 80)
    print("MT FP16 PERFORMANCE BENCHMARK PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
