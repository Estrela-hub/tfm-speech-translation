import argparse
import json
import unicodedata
from pathlib import Path

import jiwer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = PROJECT_ROOT / "results" / "asr_selection" / "performance"

MODELS = [
    "parakeet",
    "whisper",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS,
    )

    return parser.parse_args()


def normalize_for_wer(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    text = "".join(" " if unicodedata.category(character).startswith("P") else character for character in text)

    return " ".join(text.split())


def load_records(path: Path) -> dict:
    records = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            key = (
                record["language"],
                record["dataset_index"],
            )

            if key in records:
                raise RuntimeError(f"Duplicate record: {key}")

            records[key] = record

    return records


def calculate_wer(records: list[dict]) -> float:
    references = [normalize_for_wer(record["reference"]) for record in records]

    predictions = [normalize_for_wer(record["prediction"]) for record in records]

    return jiwer.wer(
        references,
        predictions,
    )


def percentile(values, q):
    return float(
        np.percentile(
            values,
            q,
        )
    )


def latency_stats(records):
    values = [record["total_ms"] for record in records]

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": percentile(values, 95),
    }


def main():
    args = parse_args()

    fp32_path = RESULTS_ROOT / args.model / "fp32" / "fleurs_performance.jsonl"

    fp16_path = RESULTS_ROOT / args.model / "fp16" / "fleurs_performance.jsonl"

    fp32 = load_records(fp32_path)

    fp16 = load_records(fp16_path)

    if fp32.keys() != fp16.keys():
        raise RuntimeError("FP32 and FP16 do not contain the same samples.")

    keys = list(fp32.keys())

    rows = []

    raw_matches = 0
    normalized_matches = 0

    for key in keys:
        record32 = fp32[key]
        record16 = fp16[key]

        prediction32 = record32["prediction"]

        prediction16 = record16["prediction"]

        if prediction32 == prediction16:
            raw_matches += 1

        normalized32 = normalize_for_wer(prediction32)

        normalized16 = normalize_for_wer(prediction16)

        if normalized32 == normalized16:
            normalized_matches += 1

        fp32_ms = record32["total_ms"]

        fp16_ms = record16["total_ms"]

        rows.append(
            {
                "key": key,
                "language": record32["language"],
                "sample_id": record32["sample_id"],
                "duration_s": record32["duration_s"],
                "reference": record32["reference"],
                "prediction_fp32": (prediction32),
                "prediction_fp16": (prediction16),
                "fp32_ms": fp32_ms,
                "fp16_ms": fp16_ms,
                "ratio": (fp16_ms / fp32_ms),
                "prediction_changed": (normalized32 != normalized16),
            }
        )

    fp32_records = [fp32[key] for key in keys]

    fp16_records = [fp16[key] for key in keys]

    fp32_wer = calculate_wer(fp32_records)

    fp16_wer = calculate_wer(fp16_records)

    print("=" * 90)
    print(f"{args.model.upper()} FP32 vs FP16")
    print("=" * 90)

    print(f"Samples                 : {len(keys)}")

    print(f"Raw predictions equal   : {raw_matches}/{len(keys)} ({raw_matches / len(keys) * 100:.2f}%)")

    print(f"Normalized predictions  : {normalized_matches}/{len(keys)} ({normalized_matches / len(keys) * 100:.2f}%)")

    print()

    print(f"FP32 corpus WER         : {fp32_wer * 100:.2f}%")

    print(f"FP16 corpus WER         : {fp16_wer * 100:.2f}%")

    print(f"WER difference          : {(fp16_wer - fp32_wer) * 100:+.2f} pp")

    print()

    fp32_latency = latency_stats(fp32_records)

    fp16_latency = latency_stats(fp16_records)

    print("TOTAL LATENCY")
    print("-" * 90)

    print(
        f"FP32 mean / median / p95: "
        f"{fp32_latency['mean']:.2f} / "
        f"{fp32_latency['median']:.2f} / "
        f"{fp32_latency['p95']:.2f} ms"
    )

    print(
        f"FP16 mean / median / p95: "
        f"{fp16_latency['mean']:.2f} / "
        f"{fp16_latency['median']:.2f} / "
        f"{fp16_latency['p95']:.2f} ms"
    )

    print()
    print("PER LANGUAGE")
    print("-" * 90)

    languages = [
        "en",
        "es",
        "fr",
        "de",
        "pt",
    ]

    for language in languages:
        records32 = [record for record in fp32_records if record["language"] == language]

        records16 = [record for record in fp16_records if record["language"] == language]

        wer32 = calculate_wer(records32)

        wer16 = calculate_wer(records16)

        changed = sum(row["prediction_changed"] for row in rows if row["language"] == language)

        latency32 = latency_stats(records32)

        latency16 = latency_stats(records16)

        print(
            f"{language.upper()} | "
            f"WER {wer32 * 100:6.2f}% -> "
            f"{wer16 * 100:6.2f}% | "
            f"changed {changed:3}/{len(records32)} | "
            f"mean latency "
            f"{latency32['mean']:7.2f} -> "
            f"{latency16['mean']:7.2f} ms | "
            f"p95 "
            f"{latency32['p95']:7.2f} -> "
            f"{latency16['p95']:7.2f} ms"
        )

    print()
    print("TOP 15 FP16 SLOWDOWNS")
    print("-" * 90)

    slowest = sorted(
        rows,
        key=lambda row: row["ratio"],
        reverse=True,
    )[:15]

    for row in slowest:
        print(
            f"{row['language'].upper()} | "
            f"sample {row['sample_id']} | "
            f"{row['duration_s']:6.2f}s | "
            f"{row['fp32_ms']:8.2f} -> "
            f"{row['fp16_ms']:8.2f} ms | "
            f"{row['ratio']:5.2f}x | "
            f"prediction changed: "
            f"{row['prediction_changed']}"
        )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
