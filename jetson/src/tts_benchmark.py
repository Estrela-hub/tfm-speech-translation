from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import onnxruntime as ort

from s2s_pipeline import PIPER_VOICES, PROJECT_ROOT, load_tts, synthesize_tts_to_memory

DEFAULT_WARMUP_RUNS = 3
DEFAULT_MEASURED_RUNS = 20

DEFAULT_REPORT_PATH = PROJECT_ROOT / "results" / "tts_benchmark.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Steady-state benchmark of the final Piper ONNX Runtime CPU TTS runtime.")
    )

    parser.add_argument(
        "--language",
        choices=PIPER_VOICES,
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

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")

    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero.")

    report_path = args.report.resolve()

    print("=" * 80)
    print("TTS BENCHMARK")
    print("=" * 80)

    print(f"Language       : {args.language.upper()}")
    print(f"Voice          : {PIPER_VOICES[args.language]}")
    print("Runtime        : ONNX Runtime CPU")
    print(f"ONNX Runtime   : {ort.__version__}")
    print(f"Text           : {args.text}")
    print(f"Warm-up runs   : {args.warmup_runs}")
    print(f"Measured runs  : {args.runs}")

    (
        voice,
        voice_name,
        load_ms,
    ) = load_tts(target_language=args.language)

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
        (
            _,
            audio_info,
            synthesis_ms,
        ) = synthesize_tts_to_memory(
            voice=voice,
            text=args.text,
        )

        rtf = (synthesis_ms / 1000.0) / audio_info["duration_s"]

        warmup_results.append(
            {
                "run": run_index,
                "synthesis_ms": synthesis_ms,
                "rtf": rtf,
                "audio": audio_info,
            }
        )

        print(
            f"Warm-up {run_index:02d}: {synthesis_ms:.2f} ms | RTF {rtf:.4f} | {audio_info['duration_s']:.3f} s audio"
        )

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
        (
            _,
            audio_info,
            synthesis_ms,
        ) = synthesize_tts_to_memory(
            voice=voice,
            text=args.text,
        )

        rtf = (synthesis_ms / 1000.0) / audio_info["duration_s"]

        result = {
            "run": run_index,
            "synthesis_ms": synthesis_ms,
            "rtf": rtf,
            "audio": audio_info,
        }

        results.append(result)

        print(f"Run {run_index:02d}: {synthesis_ms:.2f} ms | RTF {rtf:.4f} | {audio_info['duration_s']:.3f} s audio")

    # -------------------------------------------------------------------------
    # Validate output format
    # -------------------------------------------------------------------------

    audio_formats = {
        (
            result["audio"]["sample_rate"],
            result["audio"]["channels"],
            result["audio"]["sample_width_bytes"],
        )
        for result in results
    }

    if len(audio_formats) != 1:
        raise RuntimeError("Piper produced inconsistent WAV formats.")

    # Piper synthesis is stochastic, therefore the
    # generated duration is not required to be identical.
    synthesis_values = [result["synthesis_ms"] for result in results]

    rtf_values = [result["rtf"] for result in results]

    duration_values = [result["audio"]["duration_s"] for result in results]

    summary = {
        "synthesis_ms": summarize(synthesis_values),
        "rtf": summarize(rtf_values),
        "audio_duration_s": summarize(duration_values),
    }

    sample_rate, channels, sample_width = next(iter(audio_formats))

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TTS BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"Voice          : {voice_name}")
    print(f"Sample rate    : {sample_rate} Hz")
    print(f"Channels       : {channels}")
    print(f"Sample width   : {sample_width} bytes")
    print()

    print(f"Mean latency   : {summary['synthesis_ms']['mean']:.2f} ms")

    print(f"Median latency : {summary['synthesis_ms']['median']:.2f} ms")

    print(f"P95 latency    : {summary['synthesis_ms']['p95']:.2f} ms")

    print(f"Min latency    : {summary['synthesis_ms']['min']:.2f} ms")

    print(f"Max latency    : {summary['synthesis_ms']['max']:.2f} ms")

    print(f"Std. deviation : {summary['synthesis_ms']['stdev']:.2f} ms")

    print(f"Mean RTF       : {summary['rtf']['mean']:.4f}")

    print(f"P95 RTF        : {summary['rtf']['p95']:.4f}")

    print()
    print("TTS WAV format consistency: PASS")

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------

    report = {
        "experiment": ("Final Piper TTS steady-state benchmark"),
        "configuration": {
            "language": args.language,
            "voice": voice_name,
            "runtime": "ONNX Runtime CPU",
            "execution_provider": ("CPUExecutionProvider"),
            "text": args.text,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
        },
        "voice_load_ms": load_ms,
        "warmup": warmup_results,
        "runs": results,
        "summary": summary,
        "output_format": {
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": (sample_width),
        },
        "note": ("Piper synthesis is stochastic. Audio duration may therefore vary slightly between runs."),
        "environment": {
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "providers": (voice.session.get_providers()),
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
    print("TTS BENCHMARK: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
