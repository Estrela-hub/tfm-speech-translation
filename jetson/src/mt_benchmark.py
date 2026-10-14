from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import ctranslate2
import torch
import transformers

from s2s_pipeline import (
    MT_COMPUTE_TYPE,
    MT_CTRANSLATE2_MODEL_PATH,
    MT_MODEL_ID,
    MT_MODEL_REVISION,
    MT_TOKENIZER_PATH,
    NLLB_LANGUAGE_CODES,
    PROJECT_ROOT,
    load_mt,
    print_memory,
    run_mt,
)

DEFAULT_WARMUP_RUNS = 3
DEFAULT_MEASURED_RUNS = 20

DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "mt_benchmark.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Steady-state benchmark of the final NLLB CTranslate2 MT runtime."))

    parser.add_argument(
        "--source",
        choices=NLLB_LANGUAGE_CODES,
        required=True,
    )

    parser.add_argument(
        "--target",
        choices=NLLB_LANGUAGE_CODES,
        required=True,
    )

    parser.add_argument(
        "--text",
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
        "p95": percentile(
            values,
            0.95,
        ),
        "stdev": (statistics.stdev(values) if len(values) > 1 else 0.0),
    }


def main() -> None:
    args = parse_arguments()

    if args.source == args.target:
        raise ValueError("Source and target languages must be different.")

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")

    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    report_path = args.report.resolve()

    source_code = NLLB_LANGUAGE_CODES[args.source]

    target_code = NLLB_LANGUAGE_CODES[args.target]

    print("=" * 80)
    print("MT BENCHMARK")
    print("=" * 80)

    print(f"Model          : {MT_MODEL_ID}")
    print(f"Revision       : {MT_MODEL_REVISION}")
    print("Runtime        : CTranslate2")
    print(f"Version        : {ctranslate2.__version__}")
    print(f"Compute type   : {MT_COMPUTE_TYPE}")
    print(f"Model path     : {MT_CTRANSLATE2_MODEL_PATH}")
    print(f"Tokenizer      : {MT_TOKENIZER_PATH}")
    print(f"Source         : {source_code}")
    print(f"Target         : {target_code}")
    print(f"Text           : {args.text}")
    print(f"Warm-up runs   : {args.warmup_runs}")
    print(f"Measured runs  : {args.runs}")

    memory_before = print_memory("Before MT")

    (
        tokenizer,
        translator,
        load_ms,
    ) = load_mt(source_language=args.source)

    memory_after_load = print_memory("After MT")

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
        text, latency_ms = run_mt(
            tokenizer=tokenizer,
            translator=translator,
            source_text=args.text,
            target_language=args.target,
        )

        warmup_results.append(
            {
                "run": run_index,
                "translation": text,
                "latency_ms": latency_ms,
            }
        )

        print(f"Warm-up {run_index:02d}: {latency_ms:.2f} ms | {text}")

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
        text, latency_ms = run_mt(
            tokenizer=tokenizer,
            translator=translator,
            source_text=args.text,
            target_language=args.target,
        )

        results.append(
            {
                "run": run_index,
                "translation": text,
                "latency_ms": latency_ms,
            }
        )

        print(f"Run {run_index:02d}: {latency_ms:.2f} ms")

    # -------------------------------------------------------------------------
    # Consistency
    # -------------------------------------------------------------------------

    translations = {result["translation"] for result in results}

    if len(translations) != 1:
        raise RuntimeError("MT produced different translations across measured runs.")

    final_translation = results[0]["translation"]

    latencies = [result["latency_ms"] for result in results]

    summary = {"latency_ms": summarize(latencies)}

    memory_after_benchmark = print_memory("After MT benchmark")

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MT BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"Translation    : {final_translation}")
    print()

    print(f"Mean latency   : {summary['latency_ms']['mean']:.2f} ms")

    print(f"Median latency : {summary['latency_ms']['median']:.2f} ms")

    print(f"P95 latency    : {summary['latency_ms']['p95']:.2f} ms")

    print(f"Min latency    : {summary['latency_ms']['min']:.2f} ms")

    print(f"Max latency    : {summary['latency_ms']['max']:.2f} ms")

    print(f"Std. deviation : {summary['latency_ms']['stdev']:.2f} ms")

    print()
    print("MT consistency : PASS")

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report = {
        "experiment": ("Final NLLB CTranslate2 steady-state benchmark"),
        "configuration": {
            "model_id": MT_MODEL_ID,
            "revision": MT_MODEL_REVISION,
            "runtime": "CTranslate2",
            "compute_type": MT_COMPUTE_TYPE,
            "model_path": str(MT_CTRANSLATE2_MODEL_PATH),
            "tokenizer_path": str(MT_TOKENIZER_PATH),
            "source_language": args.source,
            "source_code": source_code,
            "target_language": args.target,
            "target_code": target_code,
            "source_text": args.text,
            "beam_size": 1,
            "policy": "final_only",
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
            "latency_scope": ("tokenization + CTranslate2 translation + detokenization"),
        },
        "model_load_ms": load_ms,
        "warmup": warmup_results,
        "runs": results,
        "summary": summary,
        "output_consistency": {
            "exact": True,
            "translation": final_translation,
        },
        "memory": {
            "before_model": memory_before,
            "after_model": memory_after_load,
            "after_benchmark": (memory_after_benchmark),
        },
        "memory_note": ("Jetson uses unified memory. CUDA and system memory values must not be added together."),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "ctranslate2": (ctranslate2.__version__),
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
    print("MT BENCHMARK: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
