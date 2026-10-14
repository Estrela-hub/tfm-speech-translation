from __future__ import annotations

import csv
import json
import platform
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

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

EXPECTED_TARGET_DRAFTS = 2028
EXPECTED_SEQUENCES = 100
EXPECTED_DIRECTIONS = 20
EXPECTED_UTTERANCES_PER_DIRECTION = 5

EXPECTED_MODEL_ID = "facebook/nllb-200-distilled-600M"

EXPECTED_MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

LANGUAGES = ["en", "es", "fr", "de", "pt"]


# =============================================================================
# Segment definition
# =============================================================================

# Clause/sentence-level punctuation considered as a possible
# target commitment boundary.
#
# Consecutive punctuation characters such as "..." or "?!" are treated
# as a single boundary.
SEGMENT_BOUNDARY_PATTERN = re.compile(r"[,.;:!?]+")

SEGMENT_BOUNDARY_DESCRIPTION = "comma, period, semicolon, colon, exclamation mark, question mark"


# =============================================================================
# Policies
# =============================================================================

POLICIES = [
    {
        "name": "final_only",
        "stable_drafts": None,
    },
    {
        "name": "segment_immediate",
        "stable_drafts": 1,
    },
    {
        "name": "segment_stable2",
        "stable_drafts": 2,
    },
    {
        "name": "segment_stable3",
        "stable_drafts": 3,
    },
]


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "mt_selection" / "incremental_translation"

TARGET_DRAFTS_PATH = INPUT_DIR / "target_drafts.jsonl"

MT10_SUMMARY_PATH = INPUT_DIR / "incremental_summary.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "segment_commit_policies"

TARGET_COMMITS_PATH = RESULTS_DIR / "segment_commits.jsonl"

SEQUENCE_RESULTS_PATH = RESULTS_DIR / "segment_policy_by_sequence.csv"

POLICY_RESULTS_PATH = RESULTS_DIR / "segment_policy_summary.csv"

SUMMARY_PATH = RESULTS_DIR / "segment_commit_policy_summary.json"


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

            if not isinstance(record, dict):
                raise RuntimeError(f"Expected JSON object in {path.name}, line {line_number}.")

            records.append(record)

    return records


def count_surface_tokens(
    text: str,
) -> int:
    return len(text.split())


def is_text_prefix(
    prefix: str,
    text: str,
) -> bool:
    return text.startswith(prefix)


# =============================================================================
# Text stability helpers
# =============================================================================


def longest_common_prefix_text(
    first: str,
    second: str,
) -> str:
    limit = min(
        len(first),
        len(second),
    )

    index = 0

    while index < limit and first[index] == second[index]:
        index += 1

    return first[:index]


def longest_common_prefix_many(
    texts: list[str],
) -> str:
    if not texts:
        return ""

    prefix = texts[0]

    for text in texts[1:]:
        prefix = longest_common_prefix_text(
            prefix,
            text,
        )

        if not prefix:
            break

    return prefix


# =============================================================================
# Segment helpers
# =============================================================================


def truncate_to_last_segment_boundary(
    text: str,
) -> str:
    """
    Return the exact prefix of text ending at the last configured
    punctuation boundary.

    No lowercasing, token normalization, whitespace normalization,
    or punctuation rewriting is performed.
    """

    matches = list(SEGMENT_BOUNDARY_PATTERN.finditer(text))

    if not matches:
        return ""

    last_match = matches[-1]

    return text[: last_match.end()]


def count_segment_boundaries(
    text: str,
) -> int:
    """
    Number of punctuation-delimited segment boundaries.

    Runs such as "..." and "?!" count as one boundary.
    """

    return len(list(SEGMENT_BOUNDARY_PATTERN.finditer(text)))


# =============================================================================
# Input validation
# =============================================================================


def load_and_validate_mt10() -> tuple[
    list[dict],
    dict,
]:
    if not MT10_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"mt_10 summary not found:\n{MT10_SUMMARY_PATH}")

    with MT10_SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        mt10_summary = json.load(file)

    mt = mt10_summary.get(
        "mt",
        {},
    )

    if mt.get("model_id") != EXPECTED_MODEL_ID:
        raise RuntimeError(
            f"Unexpected MT model in mt_10.\nExpected: {EXPECTED_MODEL_ID}\nActual  : {mt.get('model_id')}"
        )

    if mt.get("revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Unexpected NLLB revision in mt_10.")

    if mt.get("precision") != "float16":
        raise RuntimeError("mt_10 was not generated with FP16.")

    scope = mt10_summary.get(
        "scope",
        {},
    )

    if scope.get("utterance_target_sequences") != EXPECTED_SEQUENCES:
        raise RuntimeError("Unexpected mt_10 sequence count.")

    if scope.get("mt_invocations") != EXPECTED_TARGET_DRAFTS:
        raise RuntimeError("Unexpected mt_10 target draft count.")

    drafts = load_jsonl(TARGET_DRAFTS_PATH)

    if len(drafts) != EXPECTED_TARGET_DRAFTS:
        raise RuntimeError(
            f"Unexpected number of target drafts.\nExpected: {EXPECTED_TARGET_DRAFTS}\nActual  : {len(drafts)}"
        )

    required_fields = {
        "utterance_index",
        "dataset_index",
        "sample_id",
        "source_language",
        "target_language",
        "commit_index",
        "audio_available_s",
        "is_final_flush",
        "target_text",
        "simulated_translation_finish_s",
        "oracle_stable_prefix_coverage_percent",
    }

    for index, draft in enumerate(
        drafts,
        start=1,
    ):
        missing = required_fields - set(draft.keys())

        if missing:
            raise RuntimeError(f"Draft {index} is missing fields: {sorted(missing)}")

        if not str(draft["target_text"]).strip():
            raise RuntimeError(f"Empty target_text in draft {index}.")

    return (
        drafts,
        mt10_summary,
    )


# =============================================================================
# Sequence reconstruction
# =============================================================================


def group_and_validate_sequences(
    drafts: list[dict],
) -> dict[
    tuple[int, str, str],
    list[dict],
]:
    grouped = defaultdict(list)

    for draft in drafts:
        key = (
            int(draft["utterance_index"]),
            draft["source_language"],
            draft["target_language"],
        )

        grouped[key].append(draft)

    if len(grouped) != EXPECTED_SEQUENCES:
        raise RuntimeError(
            "Unexpected number of "
            "utterance-target sequences.\n"
            f"Expected: "
            f"{EXPECTED_SEQUENCES}\n"
            f"Actual  : "
            f"{len(grouped)}"
        )

    direction_counts = defaultdict(int)

    for key, sequence in grouped.items():
        (
            utterance_index,
            source_language,
            target_language,
        ) = key

        if source_language == target_language:
            raise RuntimeError(f"Invalid same-language sequence: {key}")

        if source_language not in LANGUAGES or target_language not in LANGUAGES:
            raise RuntimeError(f"Unexpected language in sequence {key}.")

        sequence.sort(key=lambda row: int(row["commit_index"]))

        expected_indices = list(
            range(
                1,
                len(sequence) + 1,
            )
        )

        actual_indices = [int(row["commit_index"]) for row in sequence]

        if actual_indices != expected_indices:
            raise RuntimeError(f"Non-contiguous commit indices for {key}.")

        final_flush_count = sum(1 for row in sequence if row["is_final_flush"])

        if final_flush_count != 1:
            raise RuntimeError(f"Expected exactly one final flush for {key}.")

        if not sequence[-1]["is_final_flush"]:
            raise RuntimeError(f"Final flush is not last for {key}.")

        finish_times = [float(row["simulated_translation_finish_s"]) for row in sequence]

        if finish_times != sorted(finish_times):
            raise RuntimeError(f"Non-monotonic target availability times for {key}.")

        direction_counts[
            (
                source_language,
                target_language,
            )
        ] += 1

    if len(direction_counts) != EXPECTED_DIRECTIONS:
        raise RuntimeError("Unexpected number of translation directions.")

    for direction, count in direction_counts.items():
        if count != EXPECTED_UTTERANCES_PER_DIRECTION:
            raise RuntimeError(
                f"Direction {direction} has {count} sequences instead of {EXPECTED_UTTERANCES_PER_DIRECTION}."
            )

    return dict(grouped)


# =============================================================================
# Oracle segment coverage
# =============================================================================


def calculate_oracle_segment_coverage(
    drafts: list[dict],
) -> float:
    """
    Offline upper bound for punctuation-based target commitment.

    Immediately before the final target draft, determine which exact
    character prefix is shared with the final target and then truncate
    that prefix to the last punctuation segment boundary.

    This uses future information and is NOT an online policy.
    """

    if len(drafts) < 2:
        return 0.0

    pre_final_text = str(drafts[-2]["target_text"])

    final_text = str(drafts[-1]["target_text"])

    common_prefix = longest_common_prefix_text(
        pre_final_text,
        final_text,
    )

    safe_segment_prefix = truncate_to_last_segment_boundary(common_prefix)

    final_token_count = count_surface_tokens(final_text)

    if final_token_count == 0:
        return 0.0

    return count_surface_tokens(safe_segment_prefix) / final_token_count * 100.0


# =============================================================================
# Commit helper
# =============================================================================


def append_commit_event(
    events: list[dict],
    policy_name: str,
    sequence_key: tuple[
        int,
        str,
        str,
    ],
    draft: dict,
    committed_text: str,
    candidate_text: str,
    is_final_flush: bool,
) -> str:
    if not candidate_text.startswith(committed_text):
        raise RuntimeError("Attempted non-append-only segment commit.")

    if len(candidate_text) <= len(committed_text):
        return committed_text

    added_text_raw = candidate_text[len(committed_text) :]

    added_text = added_text_raw.strip()

    added_token_count = count_surface_tokens(added_text_raw)

    added_segment_count = count_segment_boundaries(added_text_raw)

    (
        utterance_index,
        source_language,
        target_language,
    ) = sequence_key

    events.append(
        {
            "policy": (policy_name),
            "utterance_index": (utterance_index),
            "dataset_index": int(draft["dataset_index"]),
            "sample_id": (draft["sample_id"]),
            "source_language": (source_language),
            "target_language": (target_language),
            "source_commit_index": int(draft["commit_index"]),
            "source_audio_available_s": float(draft["audio_available_s"]),
            "target_available_s": float(draft["simulated_translation_finish_s"]),
            "is_final_flush": (is_final_flush),
            "added_token_count": (added_token_count),
            "added_segment_count": (added_segment_count),
            "committed_token_count": (count_surface_tokens(candidate_text)),
            "added_text": (added_text),
            "committed_text": (candidate_text),
        }
    )

    return candidate_text


# =============================================================================
# Single policy / sequence evaluation
# =============================================================================


def evaluate_policy_on_sequence(
    policy: dict,
    sequence_key: tuple[
        int,
        str,
        str,
    ],
    drafts: list[dict],
) -> tuple[
    dict,
    list[dict],
]:
    policy_name = policy["name"]

    stable_drafts = policy["stable_drafts"]

    final_draft = drafts[-1]

    final_text = str(final_draft["target_text"])

    final_token_count = count_surface_tokens(final_text)

    if final_token_count == 0:
        raise RuntimeError(f"Empty final target for {sequence_key}.")

    committed_text = ""

    observed_texts: list[str] = []

    commit_events = []

    conflict_events = 0

    pre_final_drafts = drafts[:-1]

    # ------------------------------------------------------------------
    # Incremental punctuation-segment commitment
    # ------------------------------------------------------------------

    if policy_name != "final_only":
        for draft in pre_final_drafts:
            current_text = str(draft["target_text"])

            # Anything already delivered to TTS must remain an exact
            # character prefix of every future target draft.
            if committed_text and not is_text_prefix(
                committed_text,
                current_text,
            ):
                conflict_events += 1

            observed_texts.append(current_text)

            if len(observed_texts) < stable_drafts:
                continue

            window = observed_texts[-stable_drafts:]

            stable_text_prefix = longest_common_prefix_many(window)

            candidate_text = truncate_to_last_segment_boundary(stable_text_prefix)

            if not candidate_text:
                continue

            # Never retract text already committed.
            if not is_text_prefix(
                committed_text,
                candidate_text,
            ):
                continue

            if len(candidate_text) <= len(committed_text):
                continue

            committed_text = append_commit_event(
                events=(commit_events),
                policy_name=(policy_name),
                sequence_key=(sequence_key),
                draft=draft,
                committed_text=(committed_text),
                candidate_text=(candidate_text),
                is_final_flush=False,
            )

    # ------------------------------------------------------------------
    # State before final target draft
    # ------------------------------------------------------------------

    pre_final_committed_text = committed_text

    pre_final_is_final_prefix = is_text_prefix(
        pre_final_committed_text,
        final_text,
    )

    pre_final_committed_tokens = count_surface_tokens(pre_final_committed_text)

    if pre_final_is_final_prefix:
        pre_final_safe_tokens = pre_final_committed_tokens
    else:
        common_prefix = longest_common_prefix_text(
            pre_final_committed_text,
            final_text,
        )

        pre_final_safe_tokens = count_surface_tokens(common_prefix)

    pre_final_committed_coverage = pre_final_committed_tokens / final_token_count * 100.0

    pre_final_safe_coverage = pre_final_safe_tokens / final_token_count * 100.0

    oracle_segment_coverage = calculate_oracle_segment_coverage(drafts)

    # Keep the generic token-prefix oracle from mt_10 for comparison.
    generic_oracle_coverage = (
        float(pre_final_drafts[-1]["oracle_stable_prefix_coverage_percent"]) if pre_final_drafts else 0.0
    )

    # ------------------------------------------------------------------
    # Final target draft
    # ------------------------------------------------------------------

    if committed_text and not is_text_prefix(
        committed_text,
        final_text,
    ):
        conflict_events += 1

    if is_text_prefix(
        committed_text,
        final_text,
    ):
        if len(final_text) > len(committed_text):
            committed_text = append_commit_event(
                events=(commit_events),
                policy_name=(policy_name),
                sequence_key=(sequence_key),
                draft=(final_draft),
                committed_text=(committed_text),
                candidate_text=(final_text),
                is_final_flush=True,
            )

    final_committed_matches_target = committed_text == final_text

    final_prefix_safe = pre_final_is_final_prefix and final_committed_matches_target

    strict_safe = final_prefix_safe and conflict_events == 0

    strict_safe_pre_final_coverage = pre_final_committed_coverage if strict_safe else 0.0

    # ------------------------------------------------------------------
    # Commit event metrics
    # ------------------------------------------------------------------

    incremental_events = [event for event in commit_events if not event["is_final_flush"]]

    final_flush_events = [event for event in commit_events if event["is_final_flush"]]

    incremental_segment_count = sum(int(event["added_segment_count"]) for event in incremental_events)

    incremental_token_count = sum(int(event["added_token_count"]) for event in incremental_events)

    final_flush_token_count = sum(int(event["added_token_count"]) for event in final_flush_events)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    first_target_draft_s = float(drafts[0]["simulated_translation_finish_s"])

    final_source_commit_s = float(final_draft["audio_available_s"])

    if commit_events:
        first_commit_s = float(commit_events[0]["target_available_s"])

        first_commit_delay_ms = (first_commit_s - first_target_draft_s) * 1000.0

    else:
        first_commit_s = None
        first_commit_delay_ms = None

    if incremental_events:
        first_incremental_commit_s = float(incremental_events[0]["target_available_s"])

        first_incremental_delay_ms = (first_incremental_commit_s - first_target_draft_s) * 1000.0

    else:
        first_incremental_commit_s = None

        first_incremental_delay_ms = None

    final_text_committed_s = None

    for event in commit_events:
        if event["committed_text"] == final_text:
            final_text_committed_s = float(event["target_available_s"])
            break

    if final_text_committed_s is not None:
        final_commit_after_final_source_commit_ms = (final_text_committed_s - final_source_commit_s) * 1000.0

    else:
        final_commit_after_final_source_commit_ms = None

    incremental_commit_times = [float(event["target_available_s"]) for event in incremental_events]

    incremental_commit_intervals_ms = [
        (incremental_commit_times[index] - incremental_commit_times[index - 1]) * 1000.0
        for index in range(
            1,
            len(incremental_commit_times),
        )
    ]

    (
        utterance_index,
        source_language,
        target_language,
    ) = sequence_key

    result = {
        "policy": (policy_name),
        "stable_drafts": (stable_drafts),
        "utterance_index": (utterance_index),
        "dataset_index": int(final_draft["dataset_index"]),
        "sample_id": (final_draft["sample_id"]),
        "source_language": (source_language),
        "target_language": (target_language),
        "target_draft_count": (len(drafts)),
        "final_target_token_count": (final_token_count),
        "strict_safe": (strict_safe),
        "final_prefix_safe": (final_prefix_safe),
        "conflict_events": (conflict_events),
        "incremental_commit_count": (len(incremental_events)),
        "incremental_segment_count": (incremental_segment_count),
        "incremental_token_count": (incremental_token_count),
        "final_flush_token_count": (final_flush_token_count),
        "pre_final_committed_token_count": (pre_final_committed_tokens),
        "pre_final_is_final_prefix": (pre_final_is_final_prefix),
        "pre_final_committed_coverage_percent": (pre_final_committed_coverage),
        "pre_final_safe_coverage_percent": (pre_final_safe_coverage),
        "strict_safe_pre_final_coverage_percent": (strict_safe_pre_final_coverage),
        "oracle_segment_pre_final_coverage_percent": (oracle_segment_coverage),
        "generic_oracle_pre_final_coverage_percent": (generic_oracle_coverage),
        "first_target_draft_s": (first_target_draft_s),
        "first_commit_s": (first_commit_s),
        "first_incremental_commit_s": (first_incremental_commit_s),
        "first_commit_delay_from_first_draft_ms": (first_commit_delay_ms),
        "first_incremental_commit_delay_from_first_draft_ms": (first_incremental_delay_ms),
        "mean_incremental_commit_interval_ms": (
            statistics.mean(incremental_commit_intervals_ms) if incremental_commit_intervals_ms else 0.0
        ),
        "final_text_committed_s": (final_text_committed_s),
        "final_commit_after_final_source_commit_ms": (final_commit_after_final_source_commit_ms),
        "final_committed_matches_target": (final_committed_matches_target),
    }

    return (
        result,
        commit_events,
    )


# =============================================================================
# Evaluate policies
# =============================================================================


def evaluate_policies(
    sequences: dict[
        tuple[int, str, str],
        list[dict],
    ],
) -> tuple[
    list[dict],
    list[dict],
]:
    sequence_results = []

    all_commit_events = []

    for policy_index, policy in enumerate(
        POLICIES,
        start=1,
    ):
        print()

        print("=" * 80)

        print(f"[{policy_index:02d}/{len(POLICIES):02d}] POLICY: {policy['name']}")

        print("=" * 80)

        policy_results = []

        for sequence_key in sorted(
            sequences,
            key=lambda key: (
                key[1],
                key[2],
                key[0],
            ),
        ):
            (
                result,
                commit_events,
            ) = evaluate_policy_on_sequence(
                policy=(policy),
                sequence_key=(sequence_key),
                drafts=(sequences[sequence_key]),
            )

            policy_results.append(result)

            sequence_results.append(result)

            all_commit_events.extend(commit_events)

        strict_safe_count = sum(1 for row in policy_results if row["strict_safe"])

        conflict_sequences = sum(1 for row in policy_results if row["conflict_events"] > 0)

        sequences_with_incremental = sum(1 for row in policy_results if row["incremental_commit_count"] > 0)

        strict_coverage = statistics.mean(row["strict_safe_pre_final_coverage_percent"] for row in policy_results)

        print(f"Strict safe sequences       : {strict_safe_count}/{EXPECTED_SEQUENCES}")

        print(f"Sequences with conflicts    : {conflict_sequences}/{EXPECTED_SEQUENCES}")

        print(f"Sequences with pre-final segment commit: {sequences_with_incremental}/{EXPECTED_SEQUENCES}")

        print(f"Mean strict-safe coverage   : {strict_coverage:.1f}%")

    expected_result_count = len(POLICIES) * EXPECTED_SEQUENCES

    if len(sequence_results) != expected_result_count:
        raise RuntimeError(
            f"Unexpected result count.\nExpected: {expected_result_count}\nActual  : {len(sequence_results)}"
        )

    return (
        sequence_results,
        all_commit_events,
    )


# =============================================================================
# Policy summaries
# =============================================================================


def summarize_policies(
    sequence_results: list[dict],
) -> list[dict]:
    summaries = []

    for policy in POLICIES:
        policy_name = policy["name"]

        rows = [row for row in sequence_results if row["policy"] == policy_name]

        if len(rows) != EXPECTED_SEQUENCES:
            raise RuntimeError(f"Policy {policy_name} has {len(rows)} rows.")

        strict_safe_sequences = sum(1 for row in rows if row["strict_safe"])

        final_prefix_safe_sequences = sum(1 for row in rows if row["final_prefix_safe"])

        conflict_sequences = sum(1 for row in rows if row["conflict_events"] > 0)

        conflict_events = sum(int(row["conflict_events"]) for row in rows)

        sequences_with_incremental_commit = sum(1 for row in rows if row["incremental_commit_count"] > 0)

        final_only_fallback_sequences = EXPECTED_SEQUENCES - sequences_with_incremental_commit

        total_incremental_commits = sum(int(row["incremental_commit_count"]) for row in rows)

        total_incremental_segments = sum(int(row["incremental_segment_count"]) for row in rows)

        total_incremental_tokens = sum(int(row["incremental_token_count"]) for row in rows)

        total_final_flush_tokens = sum(int(row["final_flush_token_count"]) for row in rows)

        raw_coverages = [float(row["pre_final_committed_coverage_percent"]) for row in rows]

        safe_coverages = [float(row["pre_final_safe_coverage_percent"]) for row in rows]

        strict_coverages = [float(row["strict_safe_pre_final_coverage_percent"]) for row in rows]

        oracle_segment_coverages = [float(row["oracle_segment_pre_final_coverage_percent"]) for row in rows]

        generic_oracle_coverages = [float(row["generic_oracle_pre_final_coverage_percent"]) for row in rows]

        first_commit_times = [float(row["first_commit_s"]) for row in rows if row["first_commit_s"] is not None]

        first_incremental_commit_times = [
            float(row["first_incremental_commit_s"]) for row in rows if row["first_incremental_commit_s"] is not None
        ]

        first_incremental_delays = [
            float(row["first_incremental_commit_delay_from_first_draft_ms"])
            for row in rows
            if row["first_incremental_commit_delay_from_first_draft_ms"] is not None
        ]

        final_commit_delays = [
            float(row["final_commit_after_final_source_commit_ms"])
            for row in rows
            if row["final_commit_after_final_source_commit_ms"] is not None
        ]

        commit_intervals = [
            float(row["mean_incremental_commit_interval_ms"]) for row in rows if row["incremental_commit_count"] > 1
        ]

        summary = {
            "policy": (policy_name),
            "stable_drafts": (policy["stable_drafts"]),
            "strict_safe_sequences": (strict_safe_sequences),
            "strict_safe_percent": (strict_safe_sequences / EXPECTED_SEQUENCES * 100.0),
            "final_prefix_safe_sequences": (final_prefix_safe_sequences),
            "final_prefix_safe_percent": (final_prefix_safe_sequences / EXPECTED_SEQUENCES * 100.0),
            "conflict_sequences": (conflict_sequences),
            "conflict_events": (conflict_events),
            "sequences_with_incremental_commit": (sequences_with_incremental_commit),
            "sequences_with_incremental_commit_percent": (
                sequences_with_incremental_commit / EXPECTED_SEQUENCES * 100.0
            ),
            "final_only_fallback_sequences": (final_only_fallback_sequences),
            "final_only_fallback_percent": (final_only_fallback_sequences / EXPECTED_SEQUENCES * 100.0),
            "total_incremental_commits": (total_incremental_commits),
            "mean_incremental_commits_per_sequence": (total_incremental_commits / EXPECTED_SEQUENCES),
            "total_incremental_segments": (total_incremental_segments),
            "mean_incremental_segments_per_sequence": (total_incremental_segments / EXPECTED_SEQUENCES),
            "mean_tokens_per_incremental_segment": (
                total_incremental_tokens / total_incremental_segments if total_incremental_segments else 0.0
            ),
            "mean_incremental_tokens_per_sequence": (total_incremental_tokens / EXPECTED_SEQUENCES),
            "mean_final_flush_tokens_per_sequence": (total_final_flush_tokens / EXPECTED_SEQUENCES),
            "mean_raw_pre_final_coverage_percent": (statistics.mean(raw_coverages)),
            "mean_safe_pre_final_coverage_percent": (statistics.mean(safe_coverages)),
            "mean_strict_safe_pre_final_coverage_percent": (statistics.mean(strict_coverages)),
            "median_strict_safe_pre_final_coverage_percent": (statistics.median(strict_coverages)),
            "mean_oracle_segment_pre_final_coverage_percent": (statistics.mean(oracle_segment_coverages)),
            "mean_generic_oracle_pre_final_coverage_percent": (statistics.mean(generic_oracle_coverages)),
            "strict_coverage_gap_to_segment_oracle_pp": (
                statistics.mean(strict_coverages) - statistics.mean(oracle_segment_coverages)
            ),
            "mean_first_commit_s": (statistics.mean(first_commit_times) if first_commit_times else None),
            "median_first_commit_s": (statistics.median(first_commit_times) if first_commit_times else None),
            "mean_first_incremental_commit_s": (
                statistics.mean(first_incremental_commit_times) if first_incremental_commit_times else None
            ),
            "median_first_incremental_commit_s": (
                statistics.median(first_incremental_commit_times) if first_incremental_commit_times else None
            ),
            "mean_first_incremental_commit_delay_from_first_draft_ms": (
                statistics.mean(first_incremental_delays) if first_incremental_delays else None
            ),
            "mean_incremental_commit_interval_ms": (statistics.mean(commit_intervals) if commit_intervals else 0.0),
            "mean_final_commit_after_final_source_commit_ms": (
                statistics.mean(final_commit_delays) if final_commit_delays else None
            ),
        }

        summaries.append(summary)

    return summaries


# =============================================================================
# Pareto frontier
# =============================================================================


def dominates(
    candidate: dict,
    other: dict,
) -> bool:
    candidate_safety = float(candidate["strict_safe_percent"])

    other_safety = float(other["strict_safe_percent"])

    candidate_coverage = float(candidate["mean_strict_safe_pre_final_coverage_percent"])

    other_coverage = float(other["mean_strict_safe_pre_final_coverage_percent"])

    candidate_first_commit = candidate["mean_first_commit_s"]

    other_first_commit = other["mean_first_commit_s"]

    if candidate_first_commit is None or other_first_commit is None:
        return False

    candidate_latency = float(candidate_first_commit)

    other_latency = float(other_first_commit)

    no_worse = (
        candidate_safety >= other_safety and candidate_coverage >= other_coverage and candidate_latency <= other_latency
    )

    strictly_better = (
        candidate_safety > other_safety or candidate_coverage > other_coverage or candidate_latency < other_latency
    )

    return no_worse and strictly_better


def calculate_pareto_frontier(
    summaries: list[dict],
) -> list[dict]:
    frontier = []

    for candidate in summaries:
        dominated = False

        for other in summaries:
            if other["policy"] == candidate["policy"]:
                continue

            if dominates(
                other,
                candidate,
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(candidate.copy())

    frontier.sort(
        key=lambda row: (
            -float(row["strict_safe_percent"]),
            -float(row["mean_strict_safe_pre_final_coverage_percent"]),
            float(row["mean_first_commit_s"]),
        )
    )

    return frontier


# =============================================================================
# Output helpers
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


def write_commit_events(
    events: list[dict],
) -> None:
    with TARGET_COMMITS_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for event in events:
            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


# =============================================================================
# Console reporting
# =============================================================================


def print_policy_summary(
    summaries: list[dict],
) -> None:
    print()
    print("=" * 132)

    print("SEGMENT COMMIT POLICY SUMMARY")

    print("=" * 132)

    print(
        f"{'Policy':<20}"
        f"{'Strict':>10}"
        f"{'Conflicts':>11}"
        f"{'Strict cov.':>13}"
        f"{'Fallback':>11}"
        f"{'1st inc.':>12}"
        f"{'Seg/seq':>10}"
        f"{'Tok/seg':>10}"
        f"{'Final flush':>13}"
    )

    print("-" * 132)

    for row in summaries:
        first_incremental = (
            f"{row['mean_first_incremental_commit_s']:.2f}s"
            if row["mean_first_incremental_commit_s"] is not None
            else "-"
        )

        print(
            f"{row['policy']:<20}"
            f"{row['strict_safe_percent']:>9.1f}%"
            f"{row['conflict_sequences']:>11}"
            f"{row['mean_strict_safe_pre_final_coverage_percent']:>12.1f}%"
            f"{row['final_only_fallback_percent']:>10.1f}%"
            f"{first_incremental:>12}"
            f"{row['mean_incremental_segments_per_sequence']:>10.2f}"
            f"{row['mean_tokens_per_incremental_segment']:>10.2f}"
            f"{row['mean_final_flush_tokens_per_sequence']:>13.2f}"
        )

    print()

    print(f"Generic oracle pre-final coverage : {summaries[0]['mean_generic_oracle_pre_final_coverage_percent']:.1f}%")

    print(f"Segment oracle pre-final coverage : {summaries[0]['mean_oracle_segment_pre_final_coverage_percent']:.1f}%")


def print_pareto_frontier(
    frontier: list[dict],
) -> None:
    print()
    print("=" * 100)

    print("PARETO FRONTIER")

    print("=" * 100)

    print("Objectives: maximize strict safety, maximize strict-safe pre-final coverage, minimize first commit time.")

    print()

    print(f"{'Policy':<20}{'Strict safe':>14}{'Strict cov.':>14}{'1st commit':>14}{'Seg/seq':>12}{'Tok/seg':>12}")

    print("-" * 100)

    for row in frontier:
        print(
            f"{row['policy']:<20}"
            f"{row['strict_safe_percent']:>13.1f}%"
            f"{row['mean_strict_safe_pre_final_coverage_percent']:>13.1f}%"
            f"{row['mean_first_commit_s']:>13.2f}s"
            f"{row['mean_incremental_segments_per_sequence']:>12.2f}"
            f"{row['mean_tokens_per_incremental_segment']:>12.2f}"
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 80)

    print("MT SEGMENT COMMIT POLICY ANALYSIS")

    print("=" * 80)

    print(f"Python : {platform.python_version()}")

    print()

    print("Loading mt_10 target drafts...")

    (
        drafts,
        mt10_summary,
    ) = load_and_validate_mt10()

    print(f"Target drafts : {len(drafts)}")

    print(f"Model         : {EXPECTED_MODEL_ID}")

    print("Precision     : FP16")

    print("mt_10 input   : PASS")

    sequences = group_and_validate_sequences(drafts)

    print()

    print(f"Sequences     : {len(sequences)}")

    print(f"Directions    : {EXPECTED_DIRECTIONS}")

    print(f"Boundaries    : {SEGMENT_BOUNDARY_DESCRIPTION}")

    print("Sequence validation: PASS")

    # ------------------------------------------------------------------
    # Validate oracle values.
    # ------------------------------------------------------------------

    generic_oracle_mean = statistics.mean(
        float(sequence[-2]["oracle_stable_prefix_coverage_percent"]) if len(sequence) >= 2 else 0.0
        for sequence in sequences.values()
    )

    expected_generic_oracle = float(mt10_summary["target_stability"]["mean_pre_final_stable_coverage_percent"])

    if abs(generic_oracle_mean - expected_generic_oracle) > 0.05:
        raise RuntimeError(
            "Generic oracle coverage "
            "does not reproduce mt_10.\n"
            f"Calculated: "
            f"{generic_oracle_mean:.4f}%\n"
            f"mt_10      : "
            f"{expected_generic_oracle:.4f}%"
        )

    segment_oracle_mean = statistics.mean(
        calculate_oracle_segment_coverage(sequence) for sequence in sequences.values()
    )

    print()

    print(f"Generic oracle pre-final coverage : {generic_oracle_mean:.1f}%")

    print(f"Segment oracle pre-final coverage : {segment_oracle_mean:.1f}%")

    print("Oracle validation: PASS")

    # ------------------------------------------------------------------
    # Evaluate policies.
    # ------------------------------------------------------------------

    print()

    print("Evaluating segment policies...")

    (
        sequence_results,
        commit_events,
    ) = evaluate_policies(sequences)

    policy_summaries = summarize_policies(sequence_results)

    # ------------------------------------------------------------------
    # final_only sanity check.
    # ------------------------------------------------------------------

    final_only = next(row for row in policy_summaries if row["policy"] == "final_only")

    if final_only["strict_safe_sequences"] != EXPECTED_SEQUENCES:
        raise RuntimeError("final_only is not 100% strict-safe.")

    if final_only["conflict_sequences"] != 0:
        raise RuntimeError("final_only produced target conflicts.")

    if abs(final_only["mean_raw_pre_final_coverage_percent"]) > 1e-9:
        raise RuntimeError("final_only unexpectedly committed target text before final.")

    print()

    print("final_only sanity check: PASS")

    # ------------------------------------------------------------------
    # Pareto frontier.
    # ------------------------------------------------------------------

    pareto_frontier = calculate_pareto_frontier(policy_summaries)

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_commit_events(commit_events)

    write_csv(
        SEQUENCE_RESULTS_PATH,
        sequence_results,
    )

    write_csv(
        POLICY_RESULTS_PATH,
        policy_summaries,
    )

    summary = {
        "experiment": (
            "Offline evaluation of "
            "punctuation-segment target "
            "commit policies over NLLB "
            "FP16 incremental translation "
            "drafts."
        ),
        "input": {
            "source_experiment": ("mt_10_analyze_incremental_translation"),
            "target_drafts": (str(TARGET_DRAFTS_PATH)),
            "target_draft_count": (len(drafts)),
            "sequences": (len(sequences)),
            "directions": (EXPECTED_DIRECTIONS),
            "model_id": (EXPECTED_MODEL_ID),
            "model_revision": (EXPECTED_MODEL_REVISION),
            "precision": ("float16"),
        },
        "segment_definition": {
            "boundaries": (SEGMENT_BOUNDARY_DESCRIPTION),
            "regex": (SEGMENT_BOUNDARY_PATTERN.pattern),
            "candidate": ("Exact target-text prefix ending at the final available punctuation boundary."),
            "normalization": ("None. Stability is evaluated on exact decoded target text."),
        },
        "policies": {
            "final_only": ("No pre-final target commitment."),
            "segment_immediate": (
                "Commit the longest complete punctuation segment available in the current target draft."
            ),
            "segment_stable2": (
                "Compute the exact longest common "
                "character prefix of the last two "
                "target drafts and commit only "
                "through the last punctuation "
                "boundary in that prefix."
            ),
            "segment_stable3": (
                "Compute the exact longest common "
                "character prefix of the last three "
                "target drafts and commit only "
                "through the last punctuation "
                "boundary in that prefix."
            ),
        },
        "methodology": {
            "append_only": ("Committed target text is never retracted."),
            "strict_safety": (
                "No later target draft may "
                "contradict already committed "
                "text, and the final target "
                "must remain append-only "
                "completable."
            ),
            "strict_safe_coverage": ("Unsafe sequences contribute zero pre-final target coverage."),
            "generic_oracle_pre_final_coverage_percent": (generic_oracle_mean),
            "segment_oracle_pre_final_coverage_percent": (segment_oracle_mean),
            "segment_oracle": (
                "Offline upper bound obtained "
                "from the exact prefix shared "
                "between the final pre-final "
                "draft and final target, then "
                "truncated to the last punctuation "
                "boundary."
            ),
            "timing": (
                "Uses simulated_translation_finish_s "
                "from mt_10. Therefore these timings "
                "assume the same translate-every-"
                "source-commit scheduling used "
                "in mt_10."
            ),
            "inference": ("No MT inference is executed by mt_13."),
        },
        "results": (policy_summaries),
        "pareto_frontier": (pareto_frontier),
        "environment": {
            "python": (platform.python_version()),
        },
        "limitations": [
            (
                "Punctuation boundaries are a "
                "heuristic proxy for linguistic "
                "segments and are not a syntactic "
                "or semantic clause parser."
            ),
            ("Commas and periods may occasionally represent constructs other than natural TTS segment boundaries."),
            (
                "Only five source utterances are "
                "available per translation direction, "
                "so per-direction conclusions should "
                "not be generalized."
            ),
        ],
    }

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

    print_policy_summary(policy_summaries)

    print_pareto_frontier(pareto_frontier)

    print()

    print("=" * 80)

    print("OUTPUT")

    print("=" * 80)

    print(f"Segment commits : {TARGET_COMMITS_PATH}")

    print(f"Per sequence    : {SEQUENCE_RESULTS_PATH}")

    print(f"Policy summary  : {POLICY_RESULTS_PATH}")

    print(f"Summary         : {SUMMARY_PATH}")

    print()

    print("=" * 80)

    print("MT SEGMENT COMMIT POLICY ANALYSIS PASSED")

    print("=" * 80)


if __name__ == "__main__":
    main()
