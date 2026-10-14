from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch

from s2s_pipeline import (
    ASR_DTYPE,
    ASR_MODEL_ID,
    ASR_MODEL_PATH,
    ASR_MODEL_REVISION,
    PROJECT_ROOT,
    load_asr,
    load_input_wav,
    run_asr_waveform,
)

DEFAULT_WARMUP_RUNS = 3
DEFAULT_MEASURED_RUNS = 20

DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "asr_benchmark.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Steady-state benchmark of the final Parakeet ASR runtime."))

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Input WAV: 16 kHz / mono / PCM16.",
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

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
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
        "p95": percentile(values, 0.95),
        "stdev": (statistics.stdev(values) if len(values) > 1 else 0.0),
    }


def main() -> None:
    args = parse_arguments()

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")

    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    audio_path = args.audio.resolve()
    report_path = args.report.resolve()

    print("=" * 80)
    print("ASR BENCHMARK")
    print("=" * 80)

    print(f"Model          : {ASR_MODEL_ID}")
    print(f"Revision       : {ASR_MODEL_REVISION}")
    print(f"Checkpoint     : {ASR_MODEL_PATH}")
    print(f"Precision      : {ASR_DTYPE}")
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Audio          : {audio_path}")
    print(f"Warm-up runs   : {args.warmup_runs}")
    print(f"Measured runs  : {args.runs}")

    (
        asr_model,
        decoding_computer,
        asr_config,
        load_ms,
    ) = load_asr()

    (
        waveform,
        audio_info,
        input_load_ms,
    ) = load_input_wav(
        audio_path,
        expected_sample_rate=(asr_config["sample_rate"]),
    )

    print()
    print(f"Input duration : {audio_info['duration_s']:.3f} s")
    print(f"Input load     : {input_load_ms:.2f} ms")
    print()
    print("Input file loading is excluded from the ASR compute benchmark.")

    # -------------------------------------------------------------------------
    # Warm-up
    # -------------------------------------------------------------------------

    warmup_results = []

    print()
    print("=" * 80)
    print("WARM-UP")
    print("=" * 80)

    for run_index in range(
        1,
        args.warmup_runs + 1,
    ):
        text, metrics = run_asr_waveform(
            model=asr_model,
            decoding_computer=decoding_computer,
            asr_config=asr_config,
            waveform=waveform,
        )

        warmup_results.append(
            {
                "run": run_index,
                "text": text,
                **metrics,
            }
        )

        print(f"Warm-up {run_index:02d}: {metrics['compute_ms']:.2f} ms | RTF {metrics['rtf']:.4f}")

    # -------------------------------------------------------------------------
    # Measured runs
    # -------------------------------------------------------------------------

    torch.cuda.reset_peak_memory_stats()

    results = []

    print()
    print("=" * 80)
    print("MEASURED RUNS")
    print("=" * 80)

    for run_index in range(
        1,
        args.runs + 1,
    ):
        text, metrics = run_asr_waveform(
            model=asr_model,
            decoding_computer=decoding_computer,
            asr_config=asr_config,
            waveform=waveform,
        )

        result = {
            "run": run_index,
            "text": text,
            **metrics,
        }

        results.append(result)

        print(f"Run {run_index:02d}: {metrics['compute_ms']:.2f} ms | RTF {metrics['rtf']:.4f}")

    # -------------------------------------------------------------------------
    # Consistency
    # -------------------------------------------------------------------------

    predictions = {result["text"] for result in results}

    if len(predictions) != 1:
        raise RuntimeError("ASR produced different hypotheses across measured runs.")

    final_text = results[0]["text"]

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    latency_values = [result["compute_ms"] for result in results]

    rtf_values = [result["rtf"] for result in results]

    summary = {
        "compute_ms": summarize(latency_values),
        "rtf": summarize(rtf_values),
    }

    memory = {
        "allocated_mb": (torch.cuda.memory_allocated() / 1024**2),
        "reserved_mb": (torch.cuda.memory_reserved() / 1024**2),
        "peak_allocated_mb": (torch.cuda.max_memory_allocated() / 1024**2),
        "peak_reserved_mb": (torch.cuda.max_memory_reserved() / 1024**2),
    }

    # -------------------------------------------------------------------------
    # Console result
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("ASR BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"Text           : {final_text}")
    print()

    print(f"Mean latency   : {summary['compute_ms']['mean']:.2f} ms")

    print(f"Median latency : {summary['compute_ms']['median']:.2f} ms")

    print(f"P95 latency    : {summary['compute_ms']['p95']:.2f} ms")

    print(f"Min latency    : {summary['compute_ms']['min']:.2f} ms")

    print(f"Max latency    : {summary['compute_ms']['max']:.2f} ms")

    print(f"Std. deviation : {summary['compute_ms']['stdev']:.2f} ms")

    print(f"Mean RTF       : {summary['rtf']['mean']:.4f}")

    print(f"P95 RTF        : {summary['rtf']['p95']:.4f}")

    print()
    print(f"Peak allocated : {memory['peak_allocated_mb']:.2f} MiB")

    print(f"Peak reserved  : {memory['peak_reserved_mb']:.2f} MiB")

    print()
    print("ASR consistency: PASS")

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report = {
        "experiment": ("Final Parakeet ASR steady-state benchmark"),
        "configuration": {
            "model_id": ASR_MODEL_ID,
            "revision": ASR_MODEL_REVISION,
            "checkpoint": str(ASR_MODEL_PATH),
            "precision": str(ASR_DTYPE),
            "mode": (asr_config["mode"]),
            "audio": str(audio_path),
            "audio_info": audio_info,
            "input_load_ms": input_load_ms,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
            "input_io_included": False,
        },
        "model_load_ms": load_ms,
        "warmup": warmup_results,
        "runs": results,
        "summary": summary,
        "memory": memory,
        "output_consistency": {
            "exact": True,
            "text": final_text,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    print(f"Report         : {report_path}")

    print()
    print("=" * 80)
    print("ASR BENCHMARK: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
