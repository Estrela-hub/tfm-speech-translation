from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import ctranslate2
import nemo
import onnxruntime as ort
import torch
import transformers

from s2s_pipeline import (
    ASR_DTYPE,
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    MT_COMPUTE_TYPE,
    MT_MODEL_ID,
    MT_MODEL_REVISION,
    NLLB_LANGUAGE_CODES,
    PIPER_VOICES,
    PROJECT_ROOT,
    SUPPORTED_LANGUAGES,
    load_asr,
    load_input_wav,
    load_mt,
    load_tts,
    print_memory,
    process_job,
    warm_up_runtime,
)

DEFAULT_WARMUP_RUNS = 3
DEFAULT_MEASURED_RUNS = 20


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Steady-state benchmark of the final persistent Speech-to-Speech runtime.")
    )

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Input WAV: 16 kHz / mono / PCM16.",
    )

    parser.add_argument(
        "--source",
        choices=SUPPORTED_LANGUAGES,
        required=True,
    )

    parser.add_argument(
        "--target",
        choices=SUPPORTED_LANGUAGES,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_MEASURED_RUNS,
    )

    return parser.parse_args()


def percentile(
    values: list[float],
    fraction: float,
) -> float:
    ordered = sorted(values)

    if not ordered:
        raise ValueError("Cannot calculate percentile of an empty list.")

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * fraction

    lower = int(position)

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize(
    values: list[float],
) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(
            values,
            0.95,
        ),
        "stdev": (statistics.stdev(values) if len(values) > 1 else 0.0),
    }


def summarize_results(
    results: list[dict],
) -> dict:
    fields = [
        "input_load_ms",
        "asr_ms",
        "asr_rtf",
        "mt_ms",
        "tts_ms",
        "output_write_ms",
        "compute_total_ms",
        "file_to_file_wall_ms",
    ]

    summary = {}

    for field in fields:
        summary[field] = summarize([result[field] for result in results])

    mean_compute = summary["compute_total_ms"]["mean"]

    if mean_compute > 0:
        summary["mean_stage_share_percent"] = {
            "asr": (summary["asr_ms"]["mean"] / mean_compute * 100.0),
            "mt": (summary["mt_ms"]["mean"] / mean_compute * 100.0),
            "tts": (summary["tts_ms"]["mean"] / mean_compute * 100.0),
        }

    return summary


def main() -> None:
    args = parse_arguments()

    if args.source == args.target:
        raise ValueError("Source and target languages must be different.")

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")

    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero.")

    audio_path = args.audio.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = output_dir / "s2s_benchmark.json"

    # Validate the WAV before spending several minutes
    # loading the models.
    (
        _,
        initial_audio_info,
        _,
    ) = load_input_wav(
        audio_path,
        expected_sample_rate=16000,
    )

    print("=" * 80)
    print("S2S BENCHMARK")
    print("=" * 80)

    print(f"Input          : {audio_path}")
    print(f"Input duration : {initial_audio_info['duration_s']:.3f} s")
    print(f"Source         : {args.source.upper()}")
    print(f"Target         : {args.target.upper()}")
    print(f"Warm-up runs   : {args.warmup_runs}")
    print(f"Measured runs  : {args.runs}")
    print(f"Output dir     : {output_dir}")

    # -------------------------------------------------------------------------
    # Persistent model loading
    # -------------------------------------------------------------------------

    memory = {}

    memory["before_models"] = print_memory("Before models")

    (
        asr_model,
        decoding_computer,
        asr_config,
        asr_load_ms,
    ) = load_asr()

    memory["after_asr"] = print_memory("After ASR")

    (
        mt_tokenizer,
        mt_translator,
        mt_load_ms,
    ) = load_mt(source_language=args.source)

    memory["after_mt"] = print_memory("After MT")

    (
        tts_voice,
        voice_name,
        tts_load_ms,
    ) = load_tts(target_language=args.target)

    memory["after_tts"] = print_memory("After TTS")

    load_times = {
        "asr": asr_load_ms,
        "mt": mt_load_ms,
        "tts": tts_load_ms,
        "total": (asr_load_ms + mt_load_ms + tts_load_ms),
    }

    print()
    print("All three model families are simultaneously resident.")
    print("Model coexistence: PASS")

    # -------------------------------------------------------------------------
    # Warm-up / stabilization
    # -------------------------------------------------------------------------

    warmup_results = []

    print()
    print("=" * 80)
    print("RUNTIME STABILIZATION")
    print("=" * 80)

    for run_index in range(
        1,
        args.warmup_runs + 1,
    ):
        print()
        print(f"Stabilization run {run_index}/{args.warmup_runs}")

        result = warm_up_runtime(
            warmup_audio_path=audio_path,
            asr_model=asr_model,
            decoding_computer=decoding_computer,
            asr_config=asr_config,
            mt_tokenizer=mt_tokenizer,
            mt_translator=mt_translator,
            tts_voice=tts_voice,
            target_language=args.target,
        )

        result["run"] = run_index

        warmup_results.append(result)

    memory["after_warmup"] = print_memory("After stabilization")

    # -------------------------------------------------------------------------
    # Measured runs
    # -------------------------------------------------------------------------

    results = []

    print()
    print("=" * 80)
    print("MEASURED RUNS")
    print("=" * 80)

    for run_index in range(
        1,
        args.runs + 1,
    ):
        output_path = output_dir / f"run_{run_index:02d}.wav"

        job = {
            "job_id": (f"run_{run_index:02d}"),
            "audio": audio_path,
            "output": output_path,
        }

        result = process_job(
            job_index=run_index,
            job_count=args.runs,
            job=job,
            target_language=args.target,
            asr_model=asr_model,
            decoding_computer=decoding_computer,
            asr_config=asr_config,
            mt_tokenizer=mt_tokenizer,
            mt_translator=mt_translator,
            tts_voice=tts_voice,
            voice_name=voice_name,
        )

        results.append(result)

    # -------------------------------------------------------------------------
    # Output consistency
    # -------------------------------------------------------------------------

    asr_outputs = {result["asr_text"] for result in results}

    mt_outputs = {result["translated_text"] for result in results}

    if len(asr_outputs) != 1:
        raise RuntimeError("ASR produced different hypotheses across measured runs.")

    if len(mt_outputs) != 1:
        raise RuntimeError("MT produced different translations across measured runs.")

    print()
    print("ASR output consistency: PASS")
    print("MT output consistency : PASS")

    memory["after_benchmark"] = print_memory("After benchmark")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    summary = summarize_results(results)

    memory_change_mb = memory["after_benchmark"]["system_available_mb"] - memory["after_warmup"]["system_available_mb"]

    summary["system_available_change_mb"] = memory_change_mb

    # -------------------------------------------------------------------------
    # Console result
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("S2S BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"Runs                   : {args.runs}")

    print()
    print(f"Mean ASR               : {summary['asr_ms']['mean']:.2f} ms")
    print(f"Median ASR             : {summary['asr_ms']['median']:.2f} ms")
    print(f"P95 ASR                : {summary['asr_ms']['p95']:.2f} ms")
    print(f"Mean ASR RTF           : {summary['asr_rtf']['mean']:.4f}")

    print()
    print(f"Mean MT                : {summary['mt_ms']['mean']:.2f} ms")
    print(f"Median MT              : {summary['mt_ms']['median']:.2f} ms")
    print(f"P95 MT                 : {summary['mt_ms']['p95']:.2f} ms")

    print()
    print(f"Mean TTS               : {summary['tts_ms']['mean']:.2f} ms")
    print(f"Median TTS             : {summary['tts_ms']['median']:.2f} ms")
    print(f"P95 TTS                : {summary['tts_ms']['p95']:.2f} ms")

    print()
    print(f"Mean compute/runtime   : {summary['compute_total_ms']['mean']:.2f} ms")
    print(f"Median compute/runtime : {summary['compute_total_ms']['median']:.2f} ms")
    print(f"P95 compute/runtime    : {summary['compute_total_ms']['p95']:.2f} ms")

    print()
    print(f"Mean file-to-file      : {summary['file_to_file_wall_ms']['mean']:.2f} ms")
    print(f"Median file-to-file    : {summary['file_to_file_wall_ms']['median']:.2f} ms")
    print(f"P95 file-to-file       : {summary['file_to_file_wall_ms']['p95']:.2f} ms")
    print(f"Min file-to-file       : {summary['file_to_file_wall_ms']['min']:.2f} ms")
    print(f"Max file-to-file       : {summary['file_to_file_wall_ms']['max']:.2f} ms")

    shares = summary["mean_stage_share_percent"]

    print()
    print(f"ASR compute share      : {shares['asr']:.1f}%")
    print(f"MT compute share       : {shares['mt']:.1f}%")
    print(f"TTS compute share      : {shares['tts']:.1f}%")

    print()
    print(f"System memory change   : {memory_change_mb:+.2f} MiB")
    print("(after benchmark relative to after stabilization)")

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report = {
        "experiment": ("Final Jetson persistent Speech-to-Speech benchmark"),
        "architecture": (
            "16 kHz PCM16 WAV -> "
            "full-utterance Parakeet FP16 CUDA -> "
            "NLLB CTranslate2 FP16 CUDA final-only -> "
            "Piper ONNX Runtime CPU -> WAV"
        ),
        "configuration": {
            "input_audio": str(audio_path),
            "input": initial_audio_info,
            "source_language": args.source,
            "source_code": (NLLB_LANGUAGE_CODES[args.source]),
            "target_language": args.target,
            "target_code": (NLLB_LANGUAGE_CODES[args.target]),
            "warmup_runs": (args.warmup_runs),
            "measured_runs": args.runs,
            "output_directory": str(output_dir),
        },
        "models": {
            "asr": {
                "model_id": ASR_MODEL_ID,
                "revision": (ASR_MODEL_REVISION),
                "precision": str(ASR_DTYPE),
                "mode": "full_utterance",
            },
            "mt": {
                "model_id": MT_MODEL_ID,
                "revision": (MT_MODEL_REVISION),
                "runtime": "CTranslate2",
                "compute_type": (MT_COMPUTE_TYPE),
                "policy": "final_only",
            },
            "tts": {
                "family": "Piper",
                "voice": (PIPER_VOICES[args.target]),
                "runtime": ("ONNX Runtime CPU"),
            },
        },
        "model_load_ms": load_times,
        "stabilization": warmup_results,
        "runs": results,
        "summary": summary,
        "output_consistency": {
            "asr_exact": True,
            "asr_text": (results[0]["asr_text"]),
            "mt_exact": True,
            "translated_text": (results[0]["translated_text"]),
        },
        "memory": memory,
        "memory_note": ("Jetson uses unified memory. CUDA and system memory values must not be added together."),
        "benchmark_note": ("Model loading and stabilization runs are excluded from measured steady-state statistics."),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "ctranslate2": (ctranslate2.__version__),
            "nemo": nemo.__version__,
            "onnxruntime": ort.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "target_hardware": ("NVIDIA Jetson Orin Nano 8 GB"),
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(f"Report                 : {report_path}")

    print()
    print("=" * 80)
    print("S2S BENCHMARK: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
