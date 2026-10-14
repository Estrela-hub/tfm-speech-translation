from __future__ import annotations

import csv
import gc
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =============================================================================
# Console
# =============================================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore


# =============================================================================
# Fixed experiment configuration
# =============================================================================

EXPECTED_UTTERANCES = 25
EXPECTED_SEQUENCES = 100
EXPECTED_DIRECTIONS = 20
EXPECTED_SOURCE_COMMITS = 507

MAX_GENERATION_LENGTH = 512

DEVICE = "cuda"
DTYPE = torch.float16


# =============================================================================
# NLLB
# =============================================================================

MODEL_NAME = "NLLB-200 Distilled 600M"

MODEL_ID = "facebook/nllb-200-distilled-600M"

MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


# =============================================================================
# Languages
# =============================================================================

LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "pt",
]

EXPECTED_UTTERANCES_PER_LANGUAGE = EXPECTED_UTTERANCES // len(LANGUAGES)

NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
}


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MT10_DIR = PROJECT_ROOT / "results" / "mt_selection" / "incremental_translation"

SOURCE_COMMITS_PATH = MT10_DIR / "source_commits.jsonl"

TARGET_DRAFTS_PATH = MT10_DIR / "target_drafts.jsonl"

MT10_SUMMARY_PATH = MT10_DIR / "incremental_summary.json"

MT14_SUMMARY_PATH = PROJECT_ROOT / "results" / "mt_selection" / "mt_scheduling" / "mt_scheduling_summary.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "final_only_live"

PREDICTIONS_PATH = RESULTS_DIR / "final_only_predictions.jsonl"

SEQUENCE_RESULTS_PATH = RESULTS_DIR / "final_only_by_sequence.csv"

DIRECTION_RESULTS_PATH = RESULTS_DIR / "final_only_by_direction.csv"

SUMMARY_PATH = RESULTS_DIR / "final_only_summary.json"


# =============================================================================
# Generic helpers
# =============================================================================


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100.0

    lower = int(position)

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_jsonl(
    path: Path,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path.name}, line {line_number}.") from exc

            if not isinstance(
                record,
                dict,
            ):
                raise RuntimeError(f"Expected JSON object in {path.name}, line {line_number}.")

            records.append(record)

    return records


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


# =============================================================================
# Fixed-input validation
# =============================================================================


def load_and_validate_inputs() -> tuple[
    dict[int, dict],
    dict[tuple[int, str, str], dict],
    dict,
    dict,
]:
    if not MT10_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"mt_10 summary not found:\n{MT10_SUMMARY_PATH}")

    if not MT14_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"mt_14 summary not found:\n{MT14_SUMMARY_PATH}")

    with MT10_SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        mt10_summary = json.load(file)

    with MT14_SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        mt14_summary = json.load(file)

    # ------------------------------------------------------------------
    # Validate model / precision.
    # ------------------------------------------------------------------

    mt = mt10_summary.get(
        "mt",
        {},
    )

    if mt.get("model_id") != MODEL_ID:
        raise RuntimeError("Unexpected MT model in mt_10.")

    if mt.get("revision") != MODEL_REVISION:
        raise RuntimeError("Unexpected MT revision in mt_10.")

    if mt.get("precision") != "float16":
        raise RuntimeError("mt_10 was not generated in FP16.")

    # ------------------------------------------------------------------
    # Source commits.
    # ------------------------------------------------------------------

    source_commits = load_jsonl(SOURCE_COMMITS_PATH)

    if len(source_commits) != EXPECTED_SOURCE_COMMITS:
        raise RuntimeError(
            f"Unexpected source commit count.\nExpected: {EXPECTED_SOURCE_COMMITS}\nActual  : {len(source_commits)}"
        )

    commits_by_utterance = defaultdict(list)

    for row in source_commits:
        utterance_index = int(row["utterance_index"])

        commits_by_utterance[utterance_index].append(row)

    if len(commits_by_utterance) != EXPECTED_UTTERANCES:
        raise RuntimeError("Unexpected number of source utterances.")

    final_sources = {}

    for utterance_index, events in commits_by_utterance.items():
        events.sort(key=lambda row: int(row["commit_index"]))

        final_flush_events = [row for row in events if row["is_final_flush"]]

        if len(final_flush_events) != 1:
            raise RuntimeError(f"Expected exactly one final source flush for utterance {utterance_index}.")

        final_event = events[-1]

        if not final_event["is_final_flush"]:
            raise RuntimeError(f"Final source event is not the final flush for utterance {utterance_index}.")

        source_text = str(final_event["committed_text"])

        if not source_text.strip():
            raise RuntimeError(f"Empty final source text for utterance {utterance_index}.")

        final_sources[utterance_index] = final_event

    # ------------------------------------------------------------------
    # mt_10 final targets.
    # ------------------------------------------------------------------

    target_drafts = load_jsonl(TARGET_DRAFTS_PATH)

    final_targets = {}

    for row in target_drafts:
        if not row["is_final_flush"]:
            continue

        key = (
            int(row["utterance_index"]),
            row["source_language"],
            row["target_language"],
        )

        if key in final_targets:
            raise RuntimeError(f"Duplicate final target: {key}")

        final_targets[key] = row

    if len(final_targets) != EXPECTED_SEQUENCES:
        raise RuntimeError(
            "Unexpected number of final mt_10 "
            "target translations.\n"
            f"Expected: {EXPECTED_SEQUENCES}\n"
            f"Actual  : {len(final_targets)}"
        )

    # ------------------------------------------------------------------
    # Cross-check source text.
    # ------------------------------------------------------------------

    for key, target_row in final_targets.items():
        (
            utterance_index,
            source_language,
            target_language,
        ) = key

        source_event = final_sources[utterance_index]

        if source_event["language"] != source_language:
            raise RuntimeError(f"Source language mismatch for {key}.")

        if source_language == target_language:
            raise RuntimeError(f"Same-language final target: {key}")

        if target_row["source_committed_text"] != source_event["committed_text"]:
            raise RuntimeError(f"Final source text mismatch between source_commits and target_drafts for {key}.")

    # ------------------------------------------------------------------
    # 5 utterances / source language.
    # ------------------------------------------------------------------

    language_counts = defaultdict(int)

    for source in final_sources.values():
        language_counts[source["language"]] += 1

    for language in LANGUAGES:
        if language_counts[language] != EXPECTED_UTTERANCES_PER_LANGUAGE:
            raise RuntimeError(
                f"Expected {EXPECTED_UTTERANCES_PER_LANGUAGE} source utterances for {language}, found {language_counts[language]}."
            )

    return (
        final_sources,
        final_targets,
        mt10_summary,
        mt14_summary,
    )


# =============================================================================
# Model loading
# =============================================================================


def load_nllb():
    print()
    print("=" * 80)
    print("LOADING NLLB FP16")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    print(f"CUDA before model load : {free_bytes / 1024**2:.2f} MB free / {total_bytes / 1024**2:.2f} MB total")

    print(f"Model ID : {MODEL_ID}")

    print(f"Revision : {MODEL_REVISION}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=DTYPE,
        use_safetensors=False,
        local_files_only=True,
    )

    model.to(DEVICE)

    model.eval()

    model_device = str(next(model.parameters()).device)

    model_dtype = next(model.parameters()).dtype

    if model_device != "cuda:0":
        raise RuntimeError(f"Unexpected model device: {model_device}")

    if model_dtype != torch.float16:
        raise RuntimeError(f"Unexpected model dtype: {model_dtype}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    parameter_memory_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    print(f"Device           : {model_device}")

    print(f"Dtype            : {model_dtype}")

    print(f"Parameters       : {parameter_count:,}")

    print(f"Parameter memory : {parameter_memory_bytes / 1024**2:.2f} MB")

    return (
        tokenizer,
        model,
        parameter_count,
        parameter_memory_bytes / 1024**2,
    )


# =============================================================================
# Translation
# =============================================================================


def translate(
    model,
    tokenizer,
    source_text: str,
    source_language: str,
    target_language: str,
) -> dict:
    tokenizer.src_lang = NLLB_LANGUAGE_CODES[source_language]

    preprocessing_start = time.perf_counter()

    inputs = tokenizer(
        source_text,
        return_tensors="pt",
    ).to(DEVICE)

    target_token_id = tokenizer.convert_tokens_to_ids(NLLB_LANGUAGE_CODES[target_language])

    preprocessing_end = time.perf_counter()

    torch.cuda.synchronize()

    generation_start = time.perf_counter()

    generated = model.generate(
        **inputs,
        forced_bos_token_id=(target_token_id),
        num_beams=1,
        do_sample=False,
        max_length=(MAX_GENERATION_LENGTH),
    )

    torch.cuda.synchronize()

    generation_end = time.perf_counter()

    decoding_start = time.perf_counter()

    hypothesis = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0]

    decoding_end = time.perf_counter()

    if not hypothesis.strip():
        raise RuntimeError("Empty MT hypothesis.")

    generated_tokens = max(
        int(generated.shape[-1]) - 1,
        0,
    )

    if generated_tokens >= MAX_GENERATION_LENGTH - 1:
        reached_max_length = True
    else:
        reached_max_length = False

    return {
        "hypothesis": (hypothesis),
        "source_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": (generated_tokens),
        "reached_max_length": (reached_max_length),
        "preprocessing_ms": (preprocessing_end - preprocessing_start) * 1000.0,
        "generation_ms": (generation_end - generation_start) * 1000.0,
        "decoding_ms": (decoding_end - decoding_start) * 1000.0,
        "total_ms": (decoding_end - preprocessing_start) * 1000.0,
    }


# =============================================================================
# Warm-up
# =============================================================================


def warmup_all_directions(
    model,
    tokenizer,
    final_sources: dict[
        int,
        dict,
    ],
) -> None:
    print()
    print("=" * 80)
    print("WARM-UP")
    print("=" * 80)

    first_source_by_language = {}

    for utterance_index in sorted(final_sources):
        source = final_sources[utterance_index]

        language = source["language"]

        if language not in first_source_by_language:
            first_source_by_language[language] = source

    runs = 0

    with torch.inference_mode():
        for source_language in LANGUAGES:
            source_text = first_source_by_language[source_language]["committed_text"]

            for target_language in LANGUAGES:
                if source_language == target_language:
                    continue

                translate(
                    model=model,
                    tokenizer=tokenizer,
                    source_text=(source_text),
                    source_language=(source_language),
                    target_language=(target_language),
                )

                runs += 1

    torch.cuda.synchronize()

    print(f"Warm-up translations : {runs}")

    print(f"All {EXPECTED_DIRECTIONS} directions    : PASS")


# =============================================================================
# Build balanced benchmark order
# =============================================================================


def build_benchmark_cases(
    final_sources: dict[
        int,
        dict,
    ],
    final_targets: dict[
        tuple[int, str, str],
        dict,
    ],
) -> list[dict]:
    utterances_by_language = defaultdict(list)

    for utterance_index, source in final_sources.items():
        utterances_by_language[source["language"]].append(utterance_index)

    for language in LANGUAGES:
        utterances_by_language[language].sort()

    cases = []

    # Round-robin over the five source examples.
    #
    # This avoids measuring an entire direction only
    # at the beginning or end of the run.
    for sample_round in range(EXPECTED_UTTERANCES_PER_LANGUAGE):
        for source_language in LANGUAGES:
            utterance_index = utterances_by_language[source_language][sample_round]

            source = final_sources[utterance_index]

            for target_language in LANGUAGES:
                if target_language == source_language:
                    continue

                key = (
                    utterance_index,
                    source_language,
                    target_language,
                )

                mt10_target = final_targets[key]

                cases.append(
                    {
                        "round": (sample_round + 1),
                        "utterance_index": (utterance_index),
                        "dataset_index": int(source["dataset_index"]),
                        "sample_id": (source["sample_id"]),
                        "source_language": (source_language),
                        "target_language": (target_language),
                        "source_text": (source["committed_text"]),
                        "mt10_target_text": (mt10_target["target_text"]),
                        "mt10_final_latency_ms": float(mt10_target["total_ms"]),
                    }
                )

    if len(cases) != EXPECTED_SEQUENCES:
        raise RuntimeError("Unexpected benchmark case count.")

    return cases


# =============================================================================
# Live benchmark
# =============================================================================


def run_live_benchmark(
    model,
    tokenizer,
    cases: list[dict],
) -> list[dict]:
    print()
    print("=" * 80)
    print("LIVE FINAL-ONLY BENCHMARK")
    print("=" * 80)

    results = []

    exact_matches = 0

    with torch.inference_mode():
        for index, case in enumerate(
            cases,
            start=1,
        ):
            result = translate(
                model=model,
                tokenizer=tokenizer,
                source_text=(case["source_text"]),
                source_language=(case["source_language"]),
                target_language=(case["target_language"]),
            )

            exact_match = result["hypothesis"] == case["mt10_target_text"]

            if exact_match:
                exact_matches += 1

            row = {
                "case_index": (index),
                "round": (case["round"]),
                "utterance_index": (case["utterance_index"]),
                "dataset_index": (case["dataset_index"]),
                "sample_id": (case["sample_id"]),
                "source_language": (case["source_language"]),
                "target_language": (case["target_language"]),
                "source_tokens": (result["source_tokens"]),
                "generated_tokens": (result["generated_tokens"]),
                "reached_max_length": (result["reached_max_length"]),
                "preprocessing_ms": (result["preprocessing_ms"]),
                "generation_ms": (result["generation_ms"]),
                "decoding_ms": (result["decoding_ms"]),
                "total_ms": (result["total_ms"]),
                "mt10_final_latency_ms": (case["mt10_final_latency_ms"]),
                "latency_delta_vs_mt10_ms": (result["total_ms"] - case["mt10_final_latency_ms"]),
                "exact_match_mt10": (exact_match),
                "source_text": (case["source_text"]),
                "prediction": (result["hypothesis"]),
                "mt10_prediction": (case["mt10_target_text"]),
            }

            results.append(row)

            print(
                f"[{index:03d}/{EXPECTED_SEQUENCES}] "
                f"{case['source_language'].upper()}"
                f" -> "
                f"{case['target_language'].upper()} "
                f"| utt={case['utterance_index']:02d} "
                f"| {result['total_ms']:.1f} ms "
                f"| exact={'YES' if exact_match else 'NO'}"
            )

    print()
    print(f"Exact mt_10 final matches : {exact_matches}/{EXPECTED_SEQUENCES}")

    return results


# =============================================================================
# Direction summaries
# =============================================================================


def summarize_directions(
    results: list[dict],
) -> list[dict]:
    summaries = []

    for source_language in LANGUAGES:
        for target_language in LANGUAGES:
            if source_language == target_language:
                continue

            rows = [
                row
                for row in results
                if (row["source_language"] == source_language and row["target_language"] == target_language)
            ]

            if len(rows) != EXPECTED_UTTERANCES_PER_LANGUAGE:
                raise RuntimeError(
                    f"Expected {EXPECTED_UTTERANCES_PER_LANGUAGE} measured cases for {source_language}->{target_language}, found {len(rows)}."
                )

            latencies = [float(row["total_ms"]) for row in rows]

            generation_latencies = [float(row["generation_ms"]) for row in rows]

            old_latencies = [float(row["mt10_final_latency_ms"]) for row in rows]

            exact_matches = sum(1 for row in rows if row["exact_match_mt10"])

            summaries.append(
                {
                    "source_language": (source_language),
                    "target_language": (target_language),
                    "samples": (len(rows)),
                    "mean_latency_ms": (statistics.mean(latencies)),
                    "median_latency_ms": (statistics.median(latencies)),
                    "p95_latency_ms": (
                        percentile(
                            latencies,
                            95,
                        )
                    ),
                    "mean_generation_ms": (statistics.mean(generation_latencies)),
                    "mean_mt10_final_latency_ms": (statistics.mean(old_latencies)),
                    "mean_latency_delta_vs_mt10_ms": (statistics.mean(latencies) - statistics.mean(old_latencies)),
                    "exact_matches_mt10": (exact_matches),
                }
            )

    return summaries


# =============================================================================
# mt_14 final-only reference
# =============================================================================


def get_mt14_final_only(
    mt14_summary: dict,
) -> dict:
    results = mt14_summary.get("results", [])

    for row in results:
        if row.get("scheduler") == "final_only":
            return row

    raise RuntimeError("Could not find final_only in mt_14 summary.")


# =============================================================================
# Global summary
# =============================================================================


def build_summary(
    results: list[dict],
    direction_results: list[dict],
    mt14_final_only: dict,
    parameter_count: int,
    parameter_memory_mb: float,
    peak_cuda_allocated_mb: float,
    peak_cuda_reserved_mb: float,
) -> dict:
    latencies = [float(row["total_ms"]) for row in results]

    generation_latencies = [float(row["generation_ms"]) for row in results]

    preprocessing_latencies = [float(row["preprocessing_ms"]) for row in results]

    decoding_latencies = [float(row["decoding_ms"]) for row in results]

    mt10_final_latencies = [float(row["mt10_final_latency_ms"]) for row in results]

    exact_matches = sum(1 for row in results if row["exact_match_mt10"])

    reached_max_length = sum(1 for row in results if row["reached_max_length"])

    measured_mean = statistics.mean(latencies)

    measured_median = statistics.median(latencies)

    measured_p95 = percentile(
        latencies,
        95,
    )

    mt14_mean = float(mt14_final_only["mean_final_completion_after_final_source_commit_ms"])

    mt14_p95 = float(mt14_final_only["p95_final_completion_after_final_source_commit_ms"])

    return {
        "experiment": ("Live validation of the selected final-only NLLB FP16 runtime baseline."),
        "model": {
            "name": (MODEL_NAME),
            "id": (MODEL_ID),
            "revision": (MODEL_REVISION),
            "precision": ("float16"),
            "device": ("cuda:0"),
            "parameters": (parameter_count),
            "parameter_memory_mb": (parameter_memory_mb),
            "decoding": {
                "num_beams": 1,
                "do_sample": False,
                "max_length": (MAX_GENERATION_LENGTH),
            },
        },
        "scope": {
            "source_utterances": (EXPECTED_UTTERANCES),
            "directions": (EXPECTED_DIRECTIONS),
            "translations": (EXPECTED_SEQUENCES),
            "runtime_calls_per_sequence": (1),
        },
        "validation": {
            "exact_matches_vs_mt10": (exact_matches),
            "exact_match_percent": (exact_matches / EXPECTED_SEQUENCES * 100.0),
            "reached_max_length": (reached_max_length),
            "final_source": ("Final stable2_hold1 source commit produced by mt_10."),
            "reference_target": ("Final NLLB target draft from mt_10."),
        },
        "performance": {
            "mean_total_ms": (measured_mean),
            "median_total_ms": (measured_median),
            "p95_total_ms": (measured_p95),
            "mean_generation_ms": (statistics.mean(generation_latencies)),
            "mean_preprocessing_ms": (statistics.mean(preprocessing_latencies)),
            "mean_decoding_ms": (statistics.mean(decoding_latencies)),
            "mean_mt10_final_service_ms": (statistics.mean(mt10_final_latencies)),
            "mean_delta_vs_mt10_final_service_ms": (measured_mean - statistics.mean(mt10_final_latencies)),
            "mt14_simulated_final_only_mean_ms": (mt14_mean),
            "mt14_simulated_final_only_p95_ms": (mt14_p95),
            "live_vs_mt14_mean_delta_ms": (measured_mean - mt14_mean),
            "live_vs_mt14_p95_delta_ms": (measured_p95 - mt14_p95),
            "peak_cuda_allocated_mb": (peak_cuda_allocated_mb),
            "peak_cuda_reserved_mb": (peak_cuda_reserved_mb),
        },
        "direction_results": (direction_results),
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "transformers": (transformers.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
        },
        "notes": {
            "quality": (
                "This experiment is not a new "
                "BLEU/chrF quality evaluation. "
                "Translation preservation is checked "
                "by exact comparison against the "
                "corresponding final mt_10 outputs."
            ),
            "runtime": (
                "Unlike mt_14, these are live NLLB inference measurements rather than an offline scheduling simulation."
            ),
            "deployment": ("RTX 3060 laptop development result; not a final Jetson Orin measurement."),
        },
    }


# =============================================================================
# Console summary
# =============================================================================


def print_global_summary(
    summary: dict,
) -> None:
    validation = summary["validation"]

    performance = summary["performance"]

    print()
    print("=" * 80)
    print("FINAL-ONLY LIVE SUMMARY")
    print("=" * 80)

    print(f"Translations                  : {summary['scope']['translations']}")

    print(f"Exact matches vs mt_10        : {validation['exact_matches_vs_mt10']}/{EXPECTED_SEQUENCES}")

    print(f"Reached max length            : {validation['reached_max_length']}/{EXPECTED_SEQUENCES}")

    print()
    print("Live performance")
    print("-" * 80)

    print(f"Mean total                    : {performance['mean_total_ms']:.2f} ms")

    print(f"Median total                  : {performance['median_total_ms']:.2f} ms")

    print(f"P95 total                     : {performance['p95_total_ms']:.2f} ms")

    print(f"Mean generation               : {performance['mean_generation_ms']:.2f} ms")

    print(f"Mean preprocessing            : {performance['mean_preprocessing_ms']:.2f} ms")

    print(f"Mean decoding                 : {performance['mean_decoding_ms']:.2f} ms")

    print(f"Peak CUDA allocated           : {performance['peak_cuda_allocated_mb']:.2f} MB")

    print()
    print("Comparison")
    print("-" * 80)

    print(f"mt_10 final service mean      : {performance['mean_mt10_final_service_ms']:.2f} ms")

    print(f"mt_14 simulated final mean    : {performance['mt14_simulated_final_only_mean_ms']:.2f} ms")

    print(f"Live vs mt_14 mean delta      : {performance['live_vs_mt14_mean_delta_ms']:+.2f} ms")

    print(f"mt_14 simulated final P95     : {performance['mt14_simulated_final_only_p95_ms']:.2f} ms")

    print(f"Live vs mt_14 P95 delta       : {performance['live_vs_mt14_p95_delta_ms']:+.2f} ms")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 80)
    print("MT FINAL-ONLY LIVE BENCHMARK")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")

    print(f"PyTorch      : {torch.__version__}")

    print(f"Transformers : {transformers.__version__}")

    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Fixed experiment inputs.
    # ------------------------------------------------------------------

    print()
    print("Loading fixed mt_10 / mt_14 inputs...")

    (
        final_sources,
        final_targets,
        mt10_summary,
        mt14_summary,
    ) = load_and_validate_inputs()

    print(f"Final source utterances : {len(final_sources)}")

    print(f"Final target sequences  : {len(final_targets)}")

    print("Input validation        : PASS")

    cases = build_benchmark_cases(
        final_sources=(final_sources),
        final_targets=(final_targets),
    )

    print(f"Benchmark cases          : {len(cases)}")

    # ------------------------------------------------------------------
    # Model.
    # ------------------------------------------------------------------

    (
        tokenizer,
        model,
        parameter_count,
        parameter_memory_mb,
    ) = load_nllb()

    # ------------------------------------------------------------------
    # Warm-up.
    # ------------------------------------------------------------------

    warmup_all_directions(
        model=model,
        tokenizer=tokenizer,
        final_sources=(final_sources),
    )

    gc.collect()

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Live final-only inference.
    # ------------------------------------------------------------------

    results = run_live_benchmark(
        model=model,
        tokenizer=tokenizer,
        cases=cases,
    )

    torch.cuda.synchronize()

    peak_cuda_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2

    peak_cuda_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    # ------------------------------------------------------------------
    # Summaries.
    # ------------------------------------------------------------------

    direction_results = summarize_directions(results)

    mt14_final_only = get_mt14_final_only(mt14_summary)

    summary = build_summary(
        results=results,
        direction_results=(direction_results),
        mt14_final_only=(mt14_final_only),
        parameter_count=(parameter_count),
        parameter_memory_mb=(parameter_memory_mb),
        peak_cuda_allocated_mb=(peak_cuda_allocated_mb),
        peak_cuda_reserved_mb=(peak_cuda_reserved_mb),
    )

    # ------------------------------------------------------------------
    # Outputs first, then validation gate.
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        PREDICTIONS_PATH,
        results,
    )

    write_csv(
        SEQUENCE_RESULTS_PATH,
        results,
    )

    write_csv(
        DIRECTION_RESULTS_PATH,
        direction_results,
    )

    SUMMARY_PATH.write_text(
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

    print_global_summary(summary)

    # ------------------------------------------------------------------
    # Final correctness gate.
    # ------------------------------------------------------------------

    exact_matches = summary["validation"]["exact_matches_vs_mt10"]

    if exact_matches != EXPECTED_SEQUENCES:
        raise RuntimeError(
            "Final-only live translation does "
            "not reproduce all mt_10 final "
            "translations.\n"
            f"Exact matches: "
            f"{exact_matches}/{EXPECTED_SEQUENCES}\n"
            "Outputs were written for inspection."
        )

    if summary["validation"]["reached_max_length"] != 0:
        raise RuntimeError("At least one live final-only translation reached the generation length guard.")

    print()
    print("Final translation preservation : PASS")

    print("Generation length guard        : PASS")

    # ------------------------------------------------------------------
    # Output.
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Predictions   : {PREDICTIONS_PATH}")

    print(f"Per sequence  : {SEQUENCE_RESULTS_PATH}")

    print(f"Per direction : {DIRECTION_RESULTS_PATH}")

    print(f"Summary       : {SUMMARY_PATH}")

    print()
    print("=" * 80)
    print("MT FINAL-ONLY LIVE BENCHMARK PASSED")
    print("=" * 80)

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
