from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore


# ---------------------------------------------------------------------------
# Fixed experiment configuration
# ---------------------------------------------------------------------------

EXPECTED_MANIFEST_SHA256 = "b2e3f73365232c700a1246b7bf4726eba2401f743a6a644066c5f35d4c470399"

MAX_GENERATION_LENGTH = 512

# performance script stores:
# generated_tokens = generated_sequence_length - 1
MAX_RECORDED_GENERATED_TOKENS = MAX_GENERATION_LENGTH - 1

NEAR_MAX_THRESHOLD = 500

TOP_OUTLIERS_TO_PRINT = 20


LANGUAGES = ["EN", "ES", "FR", "DE", "PT"]

FLORES_LANGUAGE_CODES = {
    "EN": "eng_Latn",
    "ES": "spa_Latn",
    "FR": "fra_Latn",
    "DE": "deu_Latn",
    "PT": "por_Latn",
}

EXPECTED_SAMPLE_COUNT = 200

EXPECTED_SAMPLE_COUNT_PER_MODEL = EXPECTED_SAMPLE_COUNT * len(LANGUAGES) * (len(LANGUAGES) - 1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "mt_selection" / "flores_dev_200.jsonl"

FP32_SAMPLES_PATH = PROJECT_ROOT / "results" / "mt_selection" / "performance" / "performance_samples.csv"

FP16_SAMPLES_PATH = PROJECT_ROOT / "results" / "mt_selection" / "performance_fp16" / "performance_samples.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "fp16_analysis"

OUTLIERS_CSV_PATH = RESULTS_DIR / "fp16_outliers.csv"
DIRECTIONS_CSV_PATH = RESULTS_DIR / "fp16_direction_comparison.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "fp16_outlier_analysis.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


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


def pearson_correlation(
    x_values: list[float],
    y_values: list[float],
) -> float | None:
    if len(x_values) != len(y_values):
        raise ValueError("Correlation vectors have different lengths.")

    if len(x_values) < 2:
        return None

    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))

    x_variance = sum((x - x_mean) ** 2 for x in x_values)

    y_variance = sum((y - y_mean) ** 2 for y in y_values)

    denominator = math.sqrt(x_variance * y_variance)

    if denominator == 0:
        return None

    return numerator / denominator


def direction_order(
    source_language: str,
    target_language: str,
) -> int:
    counter = 0

    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue

            counter += 1

            if source == source_language and target == target_language:
                return counter

    raise RuntimeError(f"Unknown direction: {source_language}->{target_language}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_manifest() -> dict[int, dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found:\n{MANIFEST_PATH}")

    actual_sha256 = calculate_sha256(MANIFEST_PATH)

    if actual_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest SHA256 mismatch.\nExpected: {EXPECTED_MANIFEST_SHA256}\nActual  : {actual_sha256}"
        )

    records = {}

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            sample_id = int(record["id"])

            if sample_id in records:
                raise RuntimeError(f"Duplicate manifest ID: {sample_id}")

            records[sample_id] = record

    return records


# ---------------------------------------------------------------------------
# Benchmark CSV
# ---------------------------------------------------------------------------


def load_performance_csv(
    path: Path,
) -> dict[tuple[str, int, str, str], dict]:
    if not path.exists():
        raise FileNotFoundError(f"Performance CSV not found:\n{path}")

    rows = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for raw in reader:
            key = (
                raw["model"],
                int(raw["id"]),
                raw["source_language"],
                raw["target_language"],
            )

            if key in rows:
                raise RuntimeError(f"Duplicate performance row: {key}")

            rows[key] = {
                "model": raw["model"],
                "id": int(raw["id"]),
                "source_language": raw["source_language"],
                "target_language": raw["target_language"],
                "source_tokens": int(raw["source_tokens"]),
                "generated_tokens": int(raw["generated_tokens"]),
                "preprocessing_ms": float(raw["preprocessing_ms"]),
                "generation_ms": float(raw["generation_ms"]),
                "decoding_ms": float(raw["decoding_ms"]),
                "total_ms": float(raw["total_ms"]),
                "generated_tokens_per_second": float(raw["generated_tokens_per_second"]),
            }

    return rows


# ---------------------------------------------------------------------------
# Match FP32 / FP16
# ---------------------------------------------------------------------------


def build_comparisons(
    fp32_rows: dict,
    fp16_rows: dict,
    manifest: dict[int, dict],
) -> list[dict]:
    fp32_keys = set(fp32_rows.keys())
    fp16_keys = set(fp16_rows.keys())

    if fp32_keys != fp16_keys:
        only_fp32 = sorted(fp32_keys - fp16_keys)
        only_fp16 = sorted(fp16_keys - fp32_keys)

        raise RuntimeError(
            f"FP32 and FP16 benchmark keys do not match.\nOnly FP32: {only_fp32[:10]}\nOnly FP16: {only_fp16[:10]}"
        )

    comparisons = []

    for key in sorted(fp32_keys):
        fp32 = fp32_rows[key]
        fp16 = fp16_rows[key]

        model = fp32["model"]
        sample_id = fp32["id"]
        source_language = fp32["source_language"]
        target_language = fp32["target_language"]

        if sample_id not in manifest:
            raise RuntimeError(f"Sample ID {sample_id} not found in manifest.")

        source_code = FLORES_LANGUAGE_CODES[source_language]
        source_text = manifest[sample_id][source_code]

        fp32_tokens = fp32["generated_tokens"]
        fp16_tokens = fp16["generated_tokens"]

        token_delta = fp16_tokens - fp32_tokens

        fp32_latency = fp32["total_ms"]
        fp16_latency = fp16["total_ms"]

        latency_delta_ms = fp16_latency - fp32_latency

        if fp32_latency > 0:
            latency_ratio = fp16_latency / fp32_latency
        else:
            latency_ratio = math.inf

        comparisons.append(
            {
                "model": model,
                "id": sample_id,
                "source_language": source_language,
                "target_language": target_language,
                "direction_order": direction_order(
                    source_language,
                    target_language,
                ),
                "source": source_text,
                "fp32_generated_tokens": fp32_tokens,
                "fp16_generated_tokens": fp16_tokens,
                "token_delta": token_delta,
                "fp32_total_ms": fp32_latency,
                "fp16_total_ms": fp16_latency,
                "latency_delta_ms": latency_delta_ms,
                "latency_ratio": latency_ratio,
                "fp32_generation_ms": fp32["generation_ms"],
                "fp16_generation_ms": fp16["generation_ms"],
                "fp16_near_max_length": (fp16_tokens >= NEAR_MAX_THRESHOLD),
                "fp16_reached_max_length": (fp16_tokens >= MAX_RECORDED_GENERATED_TOKENS),
            }
        )

    return comparisons


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def token_delta_bucket(delta: int) -> str:
    if delta == 0:
        return "same"

    if delta < 0:
        return "fp16_shorter"

    if delta <= 5:
        return "fp16_+1_to_5"

    if delta <= 20:
        return "fp16_+6_to_20"

    if delta <= 100:
        return "fp16_+21_to_100"

    return "fp16_+101_or_more"


def analyze_model(
    model_name: str,
    comparisons: list[dict],
) -> dict:
    rows = [row for row in comparisons if row["model"] == model_name]

    if len(rows) != EXPECTED_SAMPLE_COUNT_PER_MODEL:
        raise RuntimeError(f"{model_name}: expected {EXPECTED_SAMPLE_COUNT_PER_MODEL} samples, found {len(rows)}.")

    latency_ratios = [row["latency_ratio"] for row in rows]

    latency_deltas = [row["latency_delta_ms"] for row in rows]

    token_deltas = [row["token_delta"] for row in rows]

    bucket_counts = defaultdict(int)

    for row in rows:
        bucket_counts[token_delta_bucket(row["token_delta"])] += 1

    same_tokens = sum(row["token_delta"] == 0 for row in rows)

    fp16_longer = sum(row["token_delta"] > 0 for row in rows)

    fp16_shorter = sum(row["token_delta"] < 0 for row in rows)

    near_max = [row for row in rows if row["fp16_near_max_length"]]

    reached_max = [row for row in rows if row["fp16_reached_max_length"]]

    ratio_1_5 = sum(row["latency_ratio"] >= 1.5 for row in rows)

    ratio_2 = sum(row["latency_ratio"] >= 2.0 for row in rows)

    ratio_3 = sum(row["latency_ratio"] >= 3.0 for row in rows)

    ratio_5 = sum(row["latency_ratio"] >= 5.0 for row in rows)

    token_latency_correlation = pearson_correlation(
        [float(value) for value in token_deltas],
        latency_deltas,
    )

    direction_groups = defaultdict(list)

    for row in rows:
        direction_groups[
            (
                row["source_language"],
                row["target_language"],
            )
        ].append(row)

    direction_summaries = []

    for (
        source_language,
        target_language,
    ), direction_rows in direction_groups.items():
        fp32_latencies = [row["fp32_total_ms"] for row in direction_rows]

        fp16_latencies = [row["fp16_total_ms"] for row in direction_rows]

        direction_ratios = [row["latency_ratio"] for row in direction_rows]

        direction_token_deltas = [row["token_delta"] for row in direction_rows]

        different_token_counts = sum(row["token_delta"] != 0 for row in direction_rows)

        ratio_2_count = sum(row["latency_ratio"] >= 2.0 for row in direction_rows)

        near_max_count = sum(row["fp16_near_max_length"] for row in direction_rows)

        direction_summaries.append(
            {
                "model": model_name,
                "direction_order": direction_order(
                    source_language,
                    target_language,
                ),
                "source_language": source_language,
                "target_language": target_language,
                "samples": len(direction_rows),
                "fp32_mean_total_ms": statistics.mean(fp32_latencies),
                "fp16_mean_total_ms": statistics.mean(fp16_latencies),
                "mean_latency_ratio": statistics.mean(direction_ratios),
                "median_latency_ratio": statistics.median(direction_ratios),
                "p95_latency_ratio": percentile(
                    direction_ratios,
                    95,
                ),
                "mean_token_delta": statistics.mean(direction_token_deltas),
                "different_token_count_samples": (different_token_counts),
                "latency_ratio_ge_2_samples": (ratio_2_count),
                "fp16_near_max_length_samples": (near_max_count),
            }
        )

    direction_summaries.sort(key=lambda row: row["direction_order"])

    top_latency_outliers = sorted(
        rows,
        key=lambda row: (
            row["latency_ratio"],
            row["latency_delta_ms"],
        ),
        reverse=True,
    )[:50]

    top_token_growth = sorted(
        rows,
        key=lambda row: (
            row["token_delta"],
            row["latency_ratio"],
        ),
        reverse=True,
    )[:50]

    return {
        "model": model_name,
        "samples": len(rows),
        "tokens": {
            "same": same_tokens,
            "same_percent": (same_tokens / len(rows) * 100.0),
            "fp16_longer": fp16_longer,
            "fp16_shorter": fp16_shorter,
            "delta_buckets": dict(bucket_counts),
            "near_max_length": len(near_max),
            "reached_max_length": len(reached_max),
            "near_max_threshold": NEAR_MAX_THRESHOLD,
            "maximum_recorded_generated_tokens": (MAX_RECORDED_GENERATED_TOKENS),
        },
        "latency": {
            "mean_ratio": statistics.mean(latency_ratios),
            "median_ratio": statistics.median(latency_ratios),
            "p95_ratio": percentile(
                latency_ratios,
                95,
            ),
            "ratio_ge_1_5": ratio_1_5,
            "ratio_ge_2": ratio_2,
            "ratio_ge_3": ratio_3,
            "ratio_ge_5": ratio_5,
            "mean_delta_ms": statistics.mean(latency_deltas),
        },
        "token_delta_vs_latency_delta_pearson": (token_latency_correlation),
        "directions": direction_summaries,
        "top_latency_outliers": top_latency_outliers,
        "top_token_growth": top_token_growth,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def is_outlier(row: dict) -> bool:
    return (
        row["latency_ratio"] >= 2.0
        or row["latency_delta_ms"] >= 500.0
        or abs(row["token_delta"]) >= 20
        or row["fp16_near_max_length"]
    )


def write_outliers_csv(
    comparisons: list[dict],
) -> int:
    outliers = [row for row in comparisons if is_outlier(row)]

    outliers.sort(
        key=lambda row: (
            row["model"],
            -row["latency_ratio"],
        )
    )

    fieldnames = [
        "model",
        "id",
        "source_language",
        "target_language",
        "direction_order",
        "fp32_generated_tokens",
        "fp16_generated_tokens",
        "token_delta",
        "fp32_total_ms",
        "fp16_total_ms",
        "latency_delta_ms",
        "latency_ratio",
        "fp16_near_max_length",
        "fp16_reached_max_length",
        "source",
    ]

    with OUTLIERS_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in outliers:
            writer.writerow({field: row[field] for field in fieldnames})

    return len(outliers)


def write_directions_csv(
    model_results: list[dict],
) -> None:
    fieldnames = [
        "model",
        "direction_order",
        "source_language",
        "target_language",
        "samples",
        "fp32_mean_total_ms",
        "fp16_mean_total_ms",
        "mean_latency_ratio",
        "median_latency_ratio",
        "p95_latency_ratio",
        "mean_token_delta",
        "different_token_count_samples",
        "latency_ratio_ge_2_samples",
        "fp16_near_max_length_samples",
    ]

    with DIRECTIONS_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in model_results:
            for direction in result["directions"]:
                writer.writerow(direction)


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------


def print_model_summary(result: dict) -> None:
    print()
    print("=" * 80)
    print(f"{result['model']} FP32 vs FP16 ANALYSIS")
    print("=" * 80)

    tokens = result["tokens"]
    latency = result["latency"]

    print(f"Samples compared           : {result['samples']}")
    print()

    print("Generated token count")
    print("-" * 80)

    print(f"Same FP32/FP16             : {tokens['same']} ({tokens['same_percent']:.2f}%)")

    print(f"FP16 longer                : {tokens['fp16_longer']}")

    print(f"FP16 shorter               : {tokens['fp16_shorter']}")

    buckets = tokens["delta_buckets"]

    for bucket_name in [
        "same",
        "fp16_shorter",
        "fp16_+1_to_5",
        "fp16_+6_to_20",
        "fp16_+21_to_100",
        "fp16_+101_or_more",
    ]:
        print(f"  {bucket_name:<24}: {buckets.get(bucket_name, 0)}")

    print(f"Near max length (>={NEAR_MAX_THRESHOLD}) : {tokens['near_max_length']}")

    print(f"Reached max ({MAX_RECORDED_GENERATED_TOKENS})       : {tokens['reached_max_length']}")

    print()
    print("Latency ratio FP16 / FP32")
    print("-" * 80)

    print(f"Mean ratio                 : {latency['mean_ratio']:.3f}x")

    print(f"Median ratio               : {latency['median_ratio']:.3f}x")

    print(f"P95 ratio                  : {latency['p95_ratio']:.3f}x")

    print(f">= 1.5x                    : {latency['ratio_ge_1_5']}")

    print(f">= 2.0x                    : {latency['ratio_ge_2']}")

    print(f">= 3.0x                    : {latency['ratio_ge_3']}")

    print(f">= 5.0x                    : {latency['ratio_ge_5']}")

    correlation = result["token_delta_vs_latency_delta_pearson"]

    if correlation is None:
        correlation_text = "undefined"
    else:
        correlation_text = f"{correlation:.4f}"

    print()
    print(f"Pearson(token delta, latency delta): {correlation_text}")

    print()
    print("Per-direction overview")
    print("-" * 80)

    for direction in result["directions"]:
        print(
            f"[{direction['direction_order']:02d}] "
            f"{direction['source_language']} -> "
            f"{direction['target_language']} | "
            f"FP32 {direction['fp32_mean_total_ms']:.1f} ms | "
            f"FP16 {direction['fp16_mean_total_ms']:.1f} ms | "
            f"ratio {direction['mean_latency_ratio']:.2f}x | "
            f"token changes "
            f"{direction['different_token_count_samples']}/"
            f"{direction['samples']} | "
            f">=2x {direction['latency_ratio_ge_2_samples']} | "
            f"near-max "
            f"{direction['fp16_near_max_length_samples']}"
        )

    print()
    print(f"Top {TOP_OUTLIERS_TO_PRINT} latency outliers")
    print("-" * 80)

    for index, row in enumerate(
        result["top_latency_outliers"][:TOP_OUTLIERS_TO_PRINT],
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"ID {row['id']} "
            f"{row['source_language']} -> "
            f"{row['target_language']} | "
            f"{row['latency_ratio']:.2f}x | "
            f"FP32 {row['fp32_total_ms']:.1f} ms | "
            f"FP16 {row['fp16_total_ms']:.1f} ms | "
            f"tokens "
            f"{row['fp32_generated_tokens']} -> "
            f"{row['fp16_generated_tokens']} "
            f"(Δ {row['token_delta']:+d})"
        )

        print(f"    Source: {row['source']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("MT FP16 OUTLIER ANALYSIS")
    print("=" * 80)

    print("This script performs no model inference.")
    print()

    print("Validating inputs...")

    expected_total_rows = EXPECTED_SAMPLE_COUNT_PER_MODEL * 2

    manifest = load_manifest()

    print(f"Manifest          : {MANIFEST_PATH}")

    print(f"Manifest SHA256   : {EXPECTED_MANIFEST_SHA256}")

    fp32_rows = load_performance_csv(FP32_SAMPLES_PATH)

    fp16_rows = load_performance_csv(FP16_SAMPLES_PATH)

    print(f"FP32 benchmark rows: {len(fp32_rows)}")

    print(f"FP16 benchmark rows: {len(fp16_rows)}")

    if len(fp32_rows) != expected_total_rows:
        raise RuntimeError(f"Expected {expected_total_rows} FP32 rows, found {len(fp32_rows)}.")

    if len(fp16_rows) != expected_total_rows:
        raise RuntimeError(f"Expected {expected_total_rows} FP16 rows, found {len(fp16_rows)}.")

    comparisons = build_comparisons(
        fp32_rows=fp32_rows,
        fp16_rows=fp16_rows,
        manifest=manifest,
    )

    print(f"Matched comparisons : {len(comparisons)}")

    print("Input validation     : PASS")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_results = []

    for model_name in ["NLLB", "M2M100"]:
        result = analyze_model(
            model_name=model_name,
            comparisons=comparisons,
        )

        model_results.append(result)

        print_model_summary(result)

    outlier_count = write_outliers_csv(comparisons)

    write_directions_csv(model_results)

    summary = {
        "experiment": ("MT FP32 vs FP16 outlier analysis"),
        "inference_performed": False,
        "inputs": {
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": (EXPECTED_MANIFEST_SHA256),
            "fp32_samples": str(FP32_SAMPLES_PATH),
            "fp16_samples": str(FP16_SAMPLES_PATH),
            "matched_samples": len(comparisons),
        },
        "generation": {
            "max_length": MAX_GENERATION_LENGTH,
            "maximum_recorded_generated_tokens": (MAX_RECORDED_GENERATED_TOKENS),
            "near_max_threshold": (NEAR_MAX_THRESHOLD),
        },
        "models": model_results,
        "outlier_definition": {
            "latency_ratio_ge": 2.0,
            "latency_delta_ms_ge": 500.0,
            "absolute_token_delta_ge": 20,
            "or_fp16_near_max_length": True,
        },
        "outliers_written": outlier_count,
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
    print("OUTPUT")
    print("=" * 80)

    print(f"Outliers   : {OUTLIERS_CSV_PATH}")

    print(f"Directions : {DIRECTIONS_CSV_PATH}")

    print(f"Summary    : {SUMMARY_JSON_PATH}")

    print(f"Outlier rows written: {outlier_count}")

    print()
    print("=" * 80)
    print("MT FP16 OUTLIER ANALYSIS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
