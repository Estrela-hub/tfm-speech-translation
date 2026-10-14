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

DEVICE = "cuda"
DTYPE = torch.float16

EXPECTED_UTTERANCES = 25
EXPECTED_PARTIAL_HYPOTHESES = 1519

MAX_GENERATION_LENGTH = 512


# =============================================================================
# Expected stable2_hold1 baseline
# =============================================================================

EXPECTED_SAFE_UTTERANCES = 25
EXPECTED_CONFLICT_UTTERANCES = 0

EXPECTED_MEAN_FIRST_COMMIT_S = 3.75
EXPECTED_MEAN_FIRST_TEXT_TO_COMMIT_S = 0.68
EXPECTED_INCREMENTAL_COVERAGE_PERCENT = 86.1

FIRST_COMMIT_TOLERANCE_S = 0.15
FIRST_TEXT_DELAY_TOLERANCE_S = 0.15
COVERAGE_TOLERANCE_PP = 1.5


# =============================================================================
# NLLB
# =============================================================================

MODEL_NAME = "NLLB-200 Distilled 600M"

MODEL_ID = "facebook/nllb-200-distilled-600M"

MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


# =============================================================================
# Languages
# =============================================================================

LANGUAGES = ["en", "es", "fr", "de", "pt"]

TOTAL_DIRECTIONS = len(LANGUAGES) * (len(LANGUAGES) - 1)
TARGETS_PER_SOURCE = len(LANGUAGES) - 1
UTTERANCES_PER_LANGUAGE = EXPECTED_UTTERANCES // len(LANGUAGES)

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

TRACE_DIR = PROJECT_ROOT / "results" / "asr_selection" / "parakeet_partial_traces"

PARTIAL_TRACES_PATH = TRACE_DIR / "partial_traces.jsonl"

UTTERANCES_PATH = TRACE_DIR / "utterances.jsonl"

METADATA_PATH = TRACE_DIR / "metadata.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "incremental_translation"

SOURCE_COMMITS_PATH = RESULTS_DIR / "source_commits.jsonl"

TARGET_DRAFTS_PATH = RESULTS_DIR / "target_drafts.jsonl"

UTTERANCE_RESULTS_PATH = RESULTS_DIR / "incremental_by_utterance_target.csv"

DIRECTION_RESULTS_PATH = RESULTS_DIR / "incremental_by_direction.csv"

SUMMARY_PATH = RESULTS_DIR / "incremental_summary.json"


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


def load_jsonl(path: Path) -> list[dict]:
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

            if not isinstance(record, dict):
                raise RuntimeError(f"Expected JSON object in {path.name}, line {line_number}.")

            records.append(record)

    return records


def longest_common_prefix_tokens(
    first: list[str],
    second: list[str],
) -> list[str]:
    length = min(
        len(first),
        len(second),
    )

    index = 0

    while index < length and first[index] == second[index]:
        index += 1

    return first[:index]


def longest_common_prefix_many(
    sequences: list[list[str]],
) -> list[str]:
    if not sequences:
        return []

    prefix = sequences[0]

    for sequence in sequences[1:]:
        prefix = longest_common_prefix_tokens(
            prefix,
            sequence,
        )

        if not prefix:
            break

    return prefix


def is_token_prefix(
    prefix: list[str],
    sequence: list[str],
) -> bool:
    if len(prefix) > len(sequence):
        return False

    return sequence[: len(prefix)] == prefix


# =============================================================================
# Trace loading and validation
# =============================================================================


def load_trace_data() -> tuple[
    list[dict],
    list[dict],
    dict,
]:
    partial_records = load_jsonl(PARTIAL_TRACES_PATH)

    utterance_records = load_jsonl(UTTERANCES_PATH)

    if len(partial_records) != EXPECTED_PARTIAL_HYPOTHESES:
        raise RuntimeError(
            "Unexpected partial trace count.\n"
            f"Expected: {EXPECTED_PARTIAL_HYPOTHESES}\n"
            f"Actual  : {len(partial_records)}"
        )

    if len(utterance_records) != EXPECTED_UTTERANCES:
        raise RuntimeError(
            f"Unexpected utterance count.\nExpected: {EXPECTED_UTTERANCES}\nActual  : {len(utterance_records)}"
        )

    utterance_indices = [int(record["utterance_index"]) for record in utterance_records]

    if len(set(utterance_indices)) != EXPECTED_UTTERANCES:
        raise RuntimeError("utterance_index is not unique.")

    languages = {record["language"] for record in utterance_records}

    if languages != set(LANGUAGES):
        raise RuntimeError(f"Unexpected language set.\nExpected: {sorted(LANGUAGES)}\nActual  : {sorted(languages)}")

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    metadata_collection = metadata.get(
        "collection",
        {},
    )

    if metadata_collection.get("utterances") != EXPECTED_UTTERANCES:
        raise RuntimeError("Metadata utterance count mismatch.")

    if metadata_collection.get("partial_records") != EXPECTED_PARTIAL_HYPOTHESES:
        raise RuntimeError("Metadata partial count mismatch.")

    return (
        partial_records,
        utterance_records,
        metadata,
    )


# =============================================================================
# stable2_hold1 reconstruction
# =============================================================================


def reconstruct_source_commits(
    partial_records: list[dict],
    utterance_records: list[dict],
) -> tuple[
    dict[int, list[dict]],
    list[dict],
    dict,
]:
    partials_by_utterance = defaultdict(list)

    for record in partial_records:
        utterance_index = int(record["utterance_index"])
        partials_by_utterance[utterance_index].append(record)

    utterances_by_index = {int(record["utterance_index"]): record for record in utterance_records}

    commits_by_utterance = {}
    source_summaries = []

    safe_utterances = 0
    conflict_utterances = 0

    first_commit_times = []
    first_text_to_commit_delays = []

    total_final_tokens = 0
    total_incremental_tokens = 0

    macro_coverages = []

    for utterance_index in sorted(utterances_by_index):
        utterance = utterances_by_index[utterance_index]

        partials = sorted(
            partials_by_utterance[utterance_index],
            key=lambda row: int(row["chunk_index"]),
        )

        if len(partials) != int(utterance["partial_count"]):
            raise RuntimeError(f"Partial count mismatch for utterance {utterance_index}.")

        language = utterance["language"]

        for partial in partials:
            if partial["language"] != language:
                raise RuntimeError(f"Language mismatch in partial traces for utterance {utterance_index}.")

        # The raw final partial is the final source sequence
        # for the incremental experiment.
        raw_final_text = partials[-1]["partial_text"]

        if raw_final_text != utterance["final_partial"]:
            raise RuntimeError(
                f"Last partial_text does not match utterances.final_partial for utterance {utterance_index}."
            )

        final_tokens = raw_final_text.split()

        committed_tokens: list[str] = []
        previous_partial_tokens = None

        commit_events = []

        conflict_count = 0
        commit_index = 0

        # ------------------------------------------------------------------
        # Incremental stable2_hold1 commits
        # ------------------------------------------------------------------

        for partial in partials:
            current_tokens = partial["partial_text"].split()

            if previous_partial_tokens is None:
                previous_partial_tokens = current_tokens
                continue

            # A transient conflict only exists when the current ASR
            # hypothesis contradicts text that has already been committed.
            #
            # The candidate stable prefix itself may temporarily become
            # shorter than the committed prefix. That does not mean that
            # previously committed text has been retracted.
            if committed_tokens and not is_token_prefix(
                committed_tokens,
                current_tokens,
            ):
                conflict_count += 1

            stable_prefix = longest_common_prefix_tokens(
                previous_partial_tokens,
                current_tokens,
            )

            # stable2_hold1:
            # stable across the last two hypotheses, while keeping the
            # final stable token uncommitted.
            candidate_tokens = stable_prefix[:-1] if stable_prefix else []

            # Never retract committed text.
            #
            # If the new stable candidate is shorter than, or otherwise
            # incompatible with, the already committed prefix, simply do
            # not commit anything on this update.
            if not is_token_prefix(
                committed_tokens,
                candidate_tokens,
            ):
                previous_partial_tokens = current_tokens
                continue

            # Extend the committed prefix only when new stable tokens
            # became available.
            if len(candidate_tokens) > len(committed_tokens):
                old_length = len(committed_tokens)

                committed_tokens = candidate_tokens.copy()

                added_tokens = committed_tokens[old_length:]

                commit_index += 1

                commit_events.append(
                    {
                        "utterance_index": utterance_index,
                        "dataset_index": int(utterance["dataset_index"]),
                        "sample_id": utterance["sample_id"],
                        "language": language,
                        "commit_index": commit_index,
                        "chunk_index": int(partial["chunk_index"]),
                        "audio_available_s": float(partial["audio_available_s"]),
                        "is_final_flush": False,
                        "added_token_count": len(added_tokens),
                        "committed_token_count": len(committed_tokens),
                        "added_text": " ".join(added_tokens),
                        "committed_text": " ".join(committed_tokens),
                    }
                )

            previous_partial_tokens = current_tokens

        # ------------------------------------------------------------------
        # State before final utterance flush
        # ------------------------------------------------------------------

        incremental_token_count = len(committed_tokens)

        # Preserve this state because "safe" refers to whether the text
        # committed incrementally remained a prefix of the final ASR
        # hypothesis. The final flush itself must not make an unsafe
        # sequence appear safe.
        pre_flush_committed_tokens = committed_tokens.copy()

        is_safe = is_token_prefix(
            pre_flush_committed_tokens,
            final_tokens,
        )

        if is_safe:
            safe_utterances += 1

        # ------------------------------------------------------------------
        # Final flush
        # ------------------------------------------------------------------

        if not is_token_prefix(
            committed_tokens,
            final_tokens,
        ):
            # The final ASR hypothesis contradicts something that had
            # already been committed.
            conflict_count += 1

        else:
            if len(final_tokens) > len(committed_tokens):
                old_length = len(committed_tokens)

                committed_tokens = final_tokens.copy()

                added_tokens = committed_tokens[old_length:]

                commit_index += 1

                commit_events.append(
                    {
                        "utterance_index": utterance_index,
                        "dataset_index": int(utterance["dataset_index"]),
                        "sample_id": utterance["sample_id"],
                        "language": language,
                        "commit_index": commit_index,
                        "chunk_index": int(partials[-1]["chunk_index"]),
                        "audio_available_s": float(utterance["last_partial_audio_s"]),
                        "is_final_flush": True,
                        "added_token_count": len(added_tokens),
                        "committed_token_count": len(committed_tokens),
                        "added_text": " ".join(added_tokens),
                        "committed_text": " ".join(committed_tokens),
                    }
                )

        if conflict_count > 0:
            conflict_utterances += 1

        if not commit_events:
            raise RuntimeError(f"No source commits for utterance {utterance_index}.")

        # ------------------------------------------------------------------
        # Per-utterance metrics
        # ------------------------------------------------------------------

        first_commit_s = float(commit_events[0]["audio_available_s"])

        first_non_empty_s = float(utterance["first_non_empty_audio_s"])

        first_commit_times.append(first_commit_s)

        first_text_to_commit_delays.append(first_commit_s - first_non_empty_s)

        final_token_count = len(final_tokens)

        if final_token_count > 0:
            coverage_percent = incremental_token_count / final_token_count * 100.0
        else:
            coverage_percent = 0.0

        macro_coverages.append(coverage_percent)

        total_final_tokens += final_token_count

        total_incremental_tokens += incremental_token_count

        commits_by_utterance[utterance_index] = commit_events

        source_summaries.append(
            {
                "utterance_index": utterance_index,
                "dataset_index": int(utterance["dataset_index"]),
                "sample_id": utterance["sample_id"],
                "language": language,
                "duration_s": float(utterance["duration_s"]),
                "partial_count": len(partials),
                "commit_count": len(commit_events),
                "non_final_commit_count": sum(1 for event in commit_events if not event["is_final_flush"]),
                "conflict_count": conflict_count,
                "safe": is_safe,
                "first_non_empty_audio_s": (first_non_empty_s),
                "first_commit_audio_s": (first_commit_s),
                "first_text_to_commit_delay_s": (first_commit_s - first_non_empty_s),
                "final_token_count": (final_token_count),
                "incremental_token_count": (incremental_token_count),
                "incremental_coverage_percent": (coverage_percent),
                "final_source_text": " ".join(final_tokens),
            }
        )

    # ----------------------------------------------------------------------
    # Global stable2_hold1 metrics
    # ----------------------------------------------------------------------

    mean_first_commit_s = statistics.mean(first_commit_times)

    mean_first_text_delay_s = statistics.mean(first_text_to_commit_delays)

    macro_coverage_percent = statistics.mean(macro_coverages)

    micro_coverage_percent = total_incremental_tokens / total_final_tokens * 100.0 if total_final_tokens > 0 else 0.0

    policy_summary = {
        "policy": "stable2_hold1",
        "utterances": EXPECTED_UTTERANCES,
        "safe_utterances": safe_utterances,
        "conflict_utterances": (conflict_utterances),
        "mean_first_commit_s": (mean_first_commit_s),
        "mean_first_text_to_commit_s": (mean_first_text_delay_s),
        "macro_incremental_coverage_percent": (macro_coverage_percent),
        "micro_incremental_coverage_percent": (micro_coverage_percent),
        "total_source_commits": sum(len(events) for events in commits_by_utterance.values()),
    }

    return (
        commits_by_utterance,
        source_summaries,
        policy_summary,
    )


# =============================================================================
# Commit-policy gate
# =============================================================================


def validate_commit_policy(
    policy_summary: dict,
) -> None:
    print()
    print("=" * 80)
    print("STABLE2_HOLD1 RECONSTRUCTION")
    print("=" * 80)

    print(f"Safe utterances             : {policy_summary['safe_utterances']}/{EXPECTED_UTTERANCES}")

    print(f"Conflict utterances         : {policy_summary['conflict_utterances']}/{EXPECTED_UTTERANCES}")

    print(f"Mean first stable commit    : {policy_summary['mean_first_commit_s']:.2f} s")

    print(f"First text -> first commit  : {policy_summary['mean_first_text_to_commit_s']:.2f} s")

    print(f"Macro incremental coverage  : {policy_summary['macro_incremental_coverage_percent']:.1f}%")

    print(f"Micro incremental coverage  : {policy_summary['micro_incremental_coverage_percent']:.1f}%")

    print(f"Total source commit events  : {policy_summary['total_source_commits']}")

    if policy_summary["safe_utterances"] != EXPECTED_SAFE_UTTERANCES:
        raise RuntimeError(
            "stable2_hold1 reconstruction does not reproduce "
            f"{EXPECTED_SAFE_UTTERANCES}/{EXPECTED_UTTERANCES} safe utterances. "
            "MT inference will not be started."
        )

    if policy_summary["conflict_utterances"] != EXPECTED_CONFLICT_UTTERANCES:
        raise RuntimeError(
            "stable2_hold1 reconstruction produced unexpected conflicts. MT inference will not be started."
        )

    if abs(policy_summary["mean_first_commit_s"] - EXPECTED_MEAN_FIRST_COMMIT_S) > FIRST_COMMIT_TOLERANCE_S:
        raise RuntimeError(
            "Mean first commit does not reproduce the previous stable2_hold1 result. MT inference will not be started."
        )

    if (
        abs(policy_summary["mean_first_text_to_commit_s"] - EXPECTED_MEAN_FIRST_TEXT_TO_COMMIT_S)
        > FIRST_TEXT_DELAY_TOLERANCE_S
    ):
        raise RuntimeError(
            "First-text-to-commit delay does not reproduce the previous baseline. MT inference will not be started."
        )

    macro_difference = abs(policy_summary["macro_incremental_coverage_percent"] - EXPECTED_INCREMENTAL_COVERAGE_PERCENT)

    micro_difference = abs(policy_summary["micro_incremental_coverage_percent"] - EXPECTED_INCREMENTAL_COVERAGE_PERCENT)

    # We previously retained only the aggregate
    # coverage result, not whether it was calculated
    # as macro or micro. Accept either if it
    # reproduces the recorded 86.1%.
    if macro_difference > COVERAGE_TOLERANCE_PP and micro_difference > COVERAGE_TOLERANCE_PP:
        raise RuntimeError(
            "Incremental coverage does not reproduce "
            "the previous stable2_hold1 baseline. "
            "MT inference will not be started."
        )

    print()
    print("stable2_hold1 validation : PASS")


# =============================================================================
# Source commit output
# =============================================================================


def write_source_commits(
    commits_by_utterance: dict[
        int,
        list[dict],
    ],
) -> None:
    with SOURCE_COMMITS_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for utterance_index in sorted(commits_by_utterance):
            for event in commits_by_utterance[utterance_index]:
                file.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


# =============================================================================
# NLLB
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
        raise RuntimeError(f"NLLB is not on cuda:0: {model_device}")

    if model_dtype != torch.float16:
        raise RuntimeError(f"NLLB is not FP16: {model_dtype}")

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


def translate(
    model,
    tokenizer,
    source_text: str,
    source_language: str,
    target_language: str,
) -> dict:
    tokenizer.src_lang = NLLB_LANGUAGE_CODES[source_language]

    total_start = time.perf_counter()

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

    total_end = decoding_end

    if not hypothesis.strip():
        raise RuntimeError("Empty MT hypothesis.")

    return {
        "hypothesis": hypothesis,
        "source_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": max(
            int(generated.shape[-1]) - 1,
            0,
        ),
        "preprocessing_ms": (preprocessing_end - preprocessing_start) * 1000.0,
        "generation_ms": (generation_end - generation_start) * 1000.0,
        "decoding_ms": (decoding_end - decoding_start) * 1000.0,
        "total_ms": (total_end - total_start) * 1000.0,
    }


# =============================================================================
# Warm-up
# =============================================================================


def warmup_all_directions(
    model,
    tokenizer,
    utterance_records: list[dict],
) -> None:
    print()
    print("=" * 80)
    print("WARM-UP")
    print("=" * 80)

    first_by_language = {}

    for utterance in sorted(
        utterance_records,
        key=lambda row: int(row["utterance_index"]),
    ):
        language = utterance["language"]

        if language not in first_by_language:
            first_by_language[language] = utterance

    runs = 0

    with torch.inference_mode():
        for source_language in LANGUAGES:
            source_text = first_by_language[source_language]["final_partial"]

            for target_language in LANGUAGES:
                if target_language == source_language:
                    continue

                translate(
                    model=model,
                    tokenizer=tokenizer,
                    source_text=source_text,
                    source_language=(source_language),
                    target_language=(target_language),
                )

                runs += 1

    torch.cuda.synchronize()

    print(f"Warm-up translations : {runs}")

    print(f"All {TOTAL_DIRECTIONS} directions    : PASS")


# =============================================================================
# Target stability analysis
# =============================================================================


def enrich_target_drafts(
    drafts: list[dict],
) -> dict:
    target_token_sequences = [draft["target_text"].split() for draft in drafts]

    final_tokens = target_token_sequences[-1]

    total_retracted_tokens = 0
    revision_events = 0

    transition_count = max(
        len(drafts) - 1,
        0,
    )

    # Consecutive revision behaviour.
    for index, draft in enumerate(drafts):
        current_tokens = target_token_sequences[index]

        if index == 0:
            previous_lcp = 0
            retracted_tokens = 0
            added_tokens = len(current_tokens)

        else:
            previous_tokens = target_token_sequences[index - 1]

            previous_lcp = len(
                longest_common_prefix_tokens(
                    previous_tokens,
                    current_tokens,
                )
            )

            retracted_tokens = len(previous_tokens) - previous_lcp

            added_tokens = len(current_tokens) - previous_lcp

            if retracted_tokens > 0:
                revision_events += 1

            total_retracted_tokens += retracted_tokens

        final_lcp = len(
            longest_common_prefix_tokens(
                current_tokens,
                final_tokens,
            )
        )

        prefix_of_final = final_lcp == len(current_tokens)

        future_sequences = target_token_sequences[index:]

        oracle_stable_prefix = len(longest_common_prefix_many(future_sequences))

        draft["previous_target_lcp_tokens"] = previous_lcp

        draft["retracted_target_tokens"] = retracted_tokens

        draft["added_target_tokens"] = added_tokens

        draft["target_final_lcp_tokens"] = final_lcp

        draft["target_is_prefix_of_final"] = prefix_of_final

        draft["oracle_stable_prefix_tokens"] = oracle_stable_prefix

        if final_tokens:
            draft["oracle_stable_prefix_coverage_percent"] = oracle_stable_prefix / len(final_tokens) * 100.0
        else:
            draft["oracle_stable_prefix_coverage_percent"] = 0.0

    safe_draft_count = sum(1 for draft in drafts if draft["target_is_prefix_of_final"])

    if len(drafts) >= 2:
        pre_final_stable_tokens = drafts[-2]["oracle_stable_prefix_tokens"]
    else:
        pre_final_stable_tokens = 0

    if final_tokens:
        pre_final_stable_coverage = pre_final_stable_tokens / len(final_tokens) * 100.0
    else:
        pre_final_stable_coverage = 0.0

    return {
        "target_final_token_count": (len(final_tokens)),
        "target_transition_count": (transition_count),
        "target_revision_events": (revision_events),
        "total_retracted_target_tokens": (total_retracted_tokens),
        "safe_target_drafts": (safe_draft_count),
        "target_draft_count": (len(drafts)),
        "safe_target_draft_percent": (safe_draft_count / len(drafts) * 100.0 if drafts else 0.0),
        "pre_final_stable_target_tokens": (pre_final_stable_tokens),
        "pre_final_stable_coverage_percent": (pre_final_stable_coverage),
    }


# =============================================================================
# Queue simulation
# =============================================================================


def simulate_incremental_runtime(
    drafts: list[dict],
) -> dict:
    previous_finish_ms = 0.0

    completed_before_next = 0
    comparable_updates = 0

    max_queue_delay_ms = 0.0

    for index, draft in enumerate(drafts):
        commit_available_ms = float(draft["audio_available_s"]) * 1000.0

        translation_start_ms = max(
            commit_available_ms,
            previous_finish_ms,
        )

        queue_delay_ms = translation_start_ms - commit_available_ms

        translation_finish_ms = translation_start_ms + float(draft["total_ms"])

        draft["simulated_translation_start_s"] = translation_start_ms / 1000.0

        draft["simulated_translation_finish_s"] = translation_finish_ms / 1000.0

        draft["simulated_queue_delay_ms"] = queue_delay_ms

        max_queue_delay_ms = max(
            max_queue_delay_ms,
            queue_delay_ms,
        )

        if index + 1 < len(drafts):
            next_commit_ms = float(drafts[index + 1]["audio_available_s"]) * 1000.0

            completed = translation_finish_ms <= next_commit_ms

            draft["completed_before_next_commit"] = completed

            comparable_updates += 1

            if completed:
                completed_before_next += 1

        else:
            draft["completed_before_next_commit"] = None

        previous_finish_ms = translation_finish_ms

    if comparable_updates > 0:
        before_next_percent = completed_before_next / comparable_updates * 100.0
    else:
        before_next_percent = 100.0

    return {
        "completed_before_next_commit": (completed_before_next),
        "comparable_updates": (comparable_updates),
        "completed_before_next_commit_percent": (before_next_percent),
        "max_queue_delay_ms": (max_queue_delay_ms),
        "first_target_completion_s": (drafts[0]["simulated_translation_finish_s"]),
        "final_target_completion_s": (drafts[-1]["simulated_translation_finish_s"]),
    }


# =============================================================================
# Incremental MT experiment
# =============================================================================


def run_incremental_experiment(
    model,
    tokenizer,
    commits_by_utterance: dict[
        int,
        list[dict],
    ],
    utterance_records: list[dict],
) -> tuple[
    list[dict],
    list[dict],
]:
    utterances_by_index = {int(row["utterance_index"]): row for row in utterance_records}

    utterance_results = []
    all_drafts = []

    direction_number = 0

    with torch.inference_mode():
        for source_language in LANGUAGES:
            source_utterance_indices = [
                index for index, utterance in utterances_by_index.items() if utterance["language"] == source_language
            ]

            source_utterance_indices.sort()

            if len(source_utterance_indices) != UTTERANCES_PER_LANGUAGE:
                raise RuntimeError(
                    f"Expected {UTTERANCES_PER_LANGUAGE} utterances for "
                    f"{source_language}, found {len(source_utterance_indices)}."
                )

            for target_language in LANGUAGES:
                if target_language == source_language:
                    continue

                direction_number += 1

                print()
                print(
                    f"[{direction_number:02d}/{TOTAL_DIRECTIONS}] {source_language.upper()} -> {target_language.upper()}"
                )

                for local_index, utterance_index in enumerate(
                    source_utterance_indices,
                    start=1,
                ):
                    utterance = utterances_by_index[utterance_index]

                    source_events = commits_by_utterance[utterance_index]

                    drafts = []

                    for event in source_events:
                        result = translate(
                            model=model,
                            tokenizer=tokenizer,
                            source_text=(event["committed_text"]),
                            source_language=(source_language),
                            target_language=(target_language),
                        )

                        draft = {
                            "utterance_index": (utterance_index),
                            "dataset_index": (event["dataset_index"]),
                            "sample_id": (event["sample_id"]),
                            "source_language": (source_language),
                            "target_language": (target_language),
                            "commit_index": (event["commit_index"]),
                            "audio_available_s": (event["audio_available_s"]),
                            "is_final_flush": (event["is_final_flush"]),
                            "source_committed_token_count": (event["committed_token_count"]),
                            "source_committed_text": (event["committed_text"]),
                            "target_text": (result["hypothesis"]),
                            "target_generated_tokens": (result["generated_tokens"]),
                            "preprocessing_ms": (result["preprocessing_ms"]),
                            "generation_ms": (result["generation_ms"]),
                            "decoding_ms": (result["decoding_ms"]),
                            "total_ms": (result["total_ms"]),
                        }

                        drafts.append(draft)

                    stability = enrich_target_drafts(drafts)

                    runtime = simulate_incremental_runtime(drafts)

                    latencies = [draft["total_ms"] for draft in drafts]

                    cumulative_compute_ms = sum(latencies)

                    duration_s = float(utterance["duration_s"])

                    final_completion_after_audio_end_ms = (runtime["final_target_completion_s"] - duration_s) * 1000.0

                    utterance_result = {
                        "utterance_index": (utterance_index),
                        "dataset_index": int(utterance["dataset_index"]),
                        "sample_id": (utterance["sample_id"]),
                        "source_language": (source_language),
                        "target_language": (target_language),
                        "duration_s": (duration_s),
                        "source_commit_count": (len(source_events)),
                        "mean_latency_ms": (statistics.mean(latencies)),
                        "median_latency_ms": (statistics.median(latencies)),
                        "p95_latency_ms": (
                            percentile(
                                latencies,
                                95,
                            )
                        ),
                        "cumulative_mt_compute_ms": (cumulative_compute_ms),
                        "cumulative_compute_to_audio_ratio": (cumulative_compute_ms / (duration_s * 1000.0)),
                        "completed_before_next_commit_percent": (runtime["completed_before_next_commit_percent"]),
                        "max_queue_delay_ms": (runtime["max_queue_delay_ms"]),
                        "first_source_commit_s": (float(source_events[0]["audio_available_s"])),
                        "first_target_completion_s": (runtime["first_target_completion_s"]),
                        "final_target_completion_s": (runtime["final_target_completion_s"]),
                        "final_completion_after_audio_end_ms": (final_completion_after_audio_end_ms),
                        **stability,
                    }

                    utterance_results.append(utterance_result)

                    all_drafts.extend(drafts)

                    print(
                        f"  [{local_index}/{UTTERANCES_PER_LANGUAGE}] "
                        f"utt={utterance_index:02d} | "
                        f"commits={len(source_events):2d} | "
                        f"median={utterance_result['median_latency_ms']:.1f} ms | "
                        f"revisions={stability['target_revision_events']} | "
                        f"pre-final stable="
                        f"{stability['pre_final_stable_coverage_percent']:.1f}%"
                    )

    return (
        utterance_results,
        all_drafts,
    )


# =============================================================================
# Direction summaries
# =============================================================================


def summarize_directions(
    utterance_results: list[dict],
    all_drafts: list[dict],
) -> list[dict]:
    results = []

    for source_language in LANGUAGES:
        for target_language in LANGUAGES:
            if source_language == target_language:
                continue

            sequence_rows = [
                row
                for row in utterance_results
                if (row["source_language"] == source_language and row["target_language"] == target_language)
            ]

            draft_rows = [
                row
                for row in all_drafts
                if (row["source_language"] == source_language and row["target_language"] == target_language)
            ]

            latencies = [row["total_ms"] for row in draft_rows]

            total_transitions = sum(row["target_transition_count"] for row in sequence_rows)

            revision_events = sum(row["target_revision_events"] for row in sequence_rows)

            safe_drafts = sum(row["safe_target_drafts"] for row in sequence_rows)

            total_drafts = sum(row["target_draft_count"] for row in sequence_rows)

            comparable_updates = sum(1 for row in draft_rows if row["completed_before_next_commit"] is not None)

            completed_updates = sum(1 for row in draft_rows if row["completed_before_next_commit"] is True)

            total_audio_ms = sum(row["duration_s"] * 1000.0 for row in sequence_rows)

            total_compute_ms = sum(row["cumulative_mt_compute_ms"] for row in sequence_rows)

            results.append(
                {
                    "source_language": (source_language),
                    "target_language": (target_language),
                    "utterances": len(sequence_rows),
                    "mt_invocations": len(draft_rows),
                    "mean_invocations_per_utterance": (len(draft_rows) / len(sequence_rows)),
                    "mean_latency_ms": (statistics.mean(latencies)),
                    "median_latency_ms": (statistics.median(latencies)),
                    "p95_latency_ms": (
                        percentile(
                            latencies,
                            95,
                        )
                    ),
                    "total_mt_compute_ms": (total_compute_ms),
                    "compute_to_audio_ratio": (total_compute_ms / total_audio_ms),
                    "completed_before_next_commit_percent": (
                        completed_updates / comparable_updates * 100.0 if comparable_updates else 100.0
                    ),
                    "mean_max_queue_delay_ms": (statistics.mean(row["max_queue_delay_ms"] for row in sequence_rows)),
                    "max_queue_delay_ms": max(row["max_queue_delay_ms"] for row in sequence_rows),
                    "target_revision_events": (revision_events),
                    "target_transitions": (total_transitions),
                    "target_revision_event_percent": (
                        revision_events / total_transitions * 100.0 if total_transitions else 0.0
                    ),
                    "total_retracted_target_tokens": sum(row["total_retracted_target_tokens"] for row in sequence_rows),
                    "safe_target_draft_percent": (safe_drafts / total_drafts * 100.0 if total_drafts else 0.0),
                    "mean_pre_final_stable_coverage_percent": (
                        statistics.mean(row["pre_final_stable_coverage_percent"] for row in sequence_rows)
                    ),
                    "mean_first_target_completion_s": (
                        statistics.mean(row["first_target_completion_s"] for row in sequence_rows)
                    ),
                    "mean_final_completion_after_audio_end_ms": (
                        statistics.mean(row["final_completion_after_audio_end_ms"] for row in sequence_rows)
                    ),
                }
            )

    return results


# =============================================================================
# CSV / JSON output
# =============================================================================


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


def write_target_drafts(
    drafts: list[dict],
) -> None:
    with TARGET_DRAFTS_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for draft in drafts:
            file.write(
                json.dumps(
                    draft,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


# =============================================================================
# Global summary
# =============================================================================


def build_global_summary(
    metadata: dict,
    policy_summary: dict,
    utterance_results: list[dict],
    all_drafts: list[dict],
    direction_results: list[dict],
    parameter_count: int,
    parameter_memory_mb: float,
) -> dict:
    latencies = [row["total_ms"] for row in all_drafts]

    total_transitions = sum(row["target_transition_count"] for row in utterance_results)

    revision_events = sum(row["target_revision_events"] for row in utterance_results)

    total_drafts = len(all_drafts)

    safe_drafts = sum(1 for row in all_drafts if row["target_is_prefix_of_final"])

    comparable_updates = [row for row in all_drafts if row["completed_before_next_commit"] is not None]

    completed_updates = sum(1 for row in comparable_updates if row["completed_before_next_commit"])

    total_compute_ms = sum(row["total_ms"] for row in all_drafts)

    # Each source utterance is evaluated against every other target language.
    effective_audio_ms = float(metadata["dataset"]["total_duration_s"]) * 1000.0 * TARGETS_PER_SOURCE

    return {
        "experiment": ("NLLB FP16 full committed-prefix retranslation on real Parakeet stable2_hold1 source commits"),
        "source": {
            "asr_model": (metadata["model"]["id"]),
            "asr_revision": (metadata["model"]["revision"]),
            "utterances": (EXPECTED_UTTERANCES),
            "partial_hypotheses": (EXPECTED_PARTIAL_HYPOTHESES),
            "streaming_config": (metadata["effective_streaming_config"]),
            "commit_policy": (policy_summary),
        },
        "mt": {
            "model": MODEL_NAME,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "precision": "float16",
            "device": "cuda:0",
            "parameters": (parameter_count),
            "parameter_memory_mb": (parameter_memory_mb),
            "strategy": ("full committed-prefix retranslation"),
            "batch_size": 1,
            "num_beams": 1,
            "do_sample": False,
            "max_length": (MAX_GENERATION_LENGTH),
        },
        "scope": {
            "languages": (LANGUAGES),
            "directions": (TOTAL_DIRECTIONS),
            "source_utterances_per_direction": (UTTERANCES_PER_LANGUAGE),
            "utterance_target_sequences": (len(utterance_results)),
            "mt_invocations": (total_drafts),
        },
        "performance": {
            "mean_latency_ms": (statistics.mean(latencies)),
            "median_latency_ms": (statistics.median(latencies)),
            "p95_latency_ms": (
                percentile(
                    latencies,
                    95,
                )
            ),
            "total_mt_compute_ms": (total_compute_ms),
            "compute_to_effective_audio_ratio": (total_compute_ms / effective_audio_ms),
            "completed_before_next_commit_percent": (
                completed_updates / len(comparable_updates) * 100.0 if comparable_updates else 100.0
            ),
            "mean_max_queue_delay_ms": (statistics.mean(row["max_queue_delay_ms"] for row in utterance_results)),
            "max_queue_delay_ms": max(row["max_queue_delay_ms"] for row in utterance_results),
            "mean_first_target_completion_s": (
                statistics.mean(row["first_target_completion_s"] for row in utterance_results)
            ),
            "mean_final_completion_after_audio_end_ms": (
                statistics.mean(row["final_completion_after_audio_end_ms"] for row in utterance_results)
            ),
        },
        "target_stability": {
            "target_transitions": (total_transitions),
            "target_revision_events": (revision_events),
            "target_revision_event_percent": (
                revision_events / total_transitions * 100.0 if total_transitions else 0.0
            ),
            "total_retracted_surface_tokens": sum(row["total_retracted_target_tokens"] for row in utterance_results),
            "safe_target_draft_percent": (safe_drafts / total_drafts * 100.0 if total_drafts else 0.0),
            "mean_pre_final_stable_coverage_percent": (
                statistics.mean(row["pre_final_stable_coverage_percent"] for row in utterance_results)
            ),
        },
        "direction_results": (direction_results),
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "transformers": (transformers.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
        },
        "notes": {
            "target_token_definition": (
                "Whitespace-separated surface tokens. "
                "These stability tokens are intended "
                "to represent TTS-facing text, not "
                "NLLB SentencePiece tokens."
            ),
            "oracle_stability": (
                "Target stable-prefix measurements "
                "look at future drafts and therefore "
                "are offline analysis metrics, not "
                "an online target commit policy."
            ),
            "performance": (
                "Laptop RTX 3060 development measurement. This is not the final Jetson Orin performance result."
            ),
        },
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 80)
    print("MT INCREMENTAL TRANSLATION ANALYSIS")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")

    print(f"PyTorch      : {torch.__version__}")

    print(f"Transformers : {transformers.__version__}")

    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    print()
    print("Loading fixed Parakeet traces...")

    (
        partial_records,
        utterance_records,
        metadata,
    ) = load_trace_data()

    print(f"Partial hypotheses : {len(partial_records)}")

    print(f"Utterances         : {len(utterance_records)}")

    print("Trace validation   : PASS")

    # ------------------------------------------------------------------
    # Reconstruct stable2_hold1 before any MT inference.
    # ------------------------------------------------------------------

    (
        commits_by_utterance,
        source_summaries,
        policy_summary,
    ) = reconstruct_source_commits(
        partial_records=partial_records,
        utterance_records=utterance_records,
    )

    validate_commit_policy(policy_summary)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_source_commits(commits_by_utterance)

    # ------------------------------------------------------------------
    # Load selected MT baseline.
    # ------------------------------------------------------------------

    (
        tokenizer,
        model,
        parameter_count,
        parameter_memory_mb,
    ) = load_nllb()

    warmup_all_directions(
        model=model,
        tokenizer=tokenizer,
        utterance_records=(utterance_records),
    )

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Incremental full-prefix translation.
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FULL COMMITTED-PREFIX RETRANSLATION")
    print("=" * 80)

    (
        utterance_results,
        all_drafts,
    ) = run_incremental_experiment(
        model=model,
        tokenizer=tokenizer,
        commits_by_utterance=(commits_by_utterance),
        utterance_records=(utterance_records),
    )

    torch.cuda.synchronize()

    peak_cuda_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2

    peak_cuda_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    direction_results = summarize_directions(
        utterance_results=(utterance_results),
        all_drafts=all_drafts,
    )

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------

    write_target_drafts(all_drafts)

    write_csv(
        UTTERANCE_RESULTS_PATH,
        utterance_results,
    )

    write_csv(
        DIRECTION_RESULTS_PATH,
        direction_results,
    )

    summary = build_global_summary(
        metadata=metadata,
        policy_summary=(policy_summary),
        utterance_results=(utterance_results),
        all_drafts=all_drafts,
        direction_results=(direction_results),
        parameter_count=(parameter_count),
        parameter_memory_mb=(parameter_memory_mb),
    )

    summary["performance"]["peak_cuda_allocated_mb"] = peak_cuda_allocated_mb

    summary["performance"]["peak_cuda_reserved_mb"] = peak_cuda_reserved_mb

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

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------

    performance = summary["performance"]

    stability = summary["target_stability"]

    print()
    print("=" * 80)
    print("GLOBAL INCREMENTAL MT SUMMARY")
    print("=" * 80)

    print(f"Utterance-target sequences : {summary['scope']['utterance_target_sequences']}")

    print(f"MT invocations              : {summary['scope']['mt_invocations']}")

    print()
    print("Performance")
    print("-" * 80)

    print(f"Mean latency                : {performance['mean_latency_ms']:.2f} ms")

    print(f"Median latency              : {performance['median_latency_ms']:.2f} ms")

    print(f"P95 latency                 : {performance['p95_latency_ms']:.2f} ms")

    print(f"Compute / effective audio   : {performance['compute_to_effective_audio_ratio']:.3f}x")

    print(f"Completed before next commit: {performance['completed_before_next_commit_percent']:.1f}%")

    print(f"Mean max queue delay        : {performance['mean_max_queue_delay_ms']:.2f} ms")

    print(f"Max queue delay             : {performance['max_queue_delay_ms']:.2f} ms")

    print(f"Mean first target complete  : {performance['mean_first_target_completion_s']:.2f} s")

    print(f"Final MT after audio end    : {performance['mean_final_completion_after_audio_end_ms']:.2f} ms")

    print(f"Peak CUDA allocated         : {performance['peak_cuda_allocated_mb']:.2f} MB")

    print()
    print("Target stability")
    print("-" * 80)

    print(f"Target transitions          : {stability['target_transitions']}")

    print(
        f"Revision events             : "
        f"{stability['target_revision_events']} "
        f"({stability['target_revision_event_percent']:.1f}%)"
    )

    print(f"Retracted surface tokens    : {stability['total_retracted_surface_tokens']}")

    print(f"Drafts already final-prefix : {stability['safe_target_draft_percent']:.1f}%")

    print(f"Pre-final stable coverage   : {stability['mean_pre_final_stable_coverage_percent']:.1f}%")

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Source commits : {SOURCE_COMMITS_PATH}")

    print(f"Target drafts  : {TARGET_DRAFTS_PATH}")

    print(f"Per sequence   : {UTTERANCE_RESULTS_PATH}")

    print(f"Per direction  : {DIRECTION_RESULTS_PATH}")

    print(f"Summary        : {SUMMARY_PATH}")

    print()
    print("=" * 80)
    print("MT INCREMENTAL TRANSLATION ANALYSIS PASSED")
    print("=" * 80)

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
