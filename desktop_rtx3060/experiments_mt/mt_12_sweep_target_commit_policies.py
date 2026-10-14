from __future__ import annotations

import csv
import json
import platform
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
# Policy sweep
# =============================================================================

STABLE_DRAFT_VALUES = list(range(2, 9))
HOLD_TOKEN_VALUES = list(range(0, 5))


def build_policies() -> list[dict]:
    policies = [
        {
            "name": "final_only",
            "stable_drafts": None,
            "hold_tokens": None,
        }
    ]

    for stable_drafts in STABLE_DRAFT_VALUES:
        for hold_tokens in HOLD_TOKEN_VALUES:
            policies.append(
                {
                    "name": (f"stable{stable_drafts}_hold{hold_tokens}"),
                    "stable_drafts": stable_drafts,
                    "hold_tokens": hold_tokens,
                }
            )

    return policies


POLICIES = build_policies()

EXPECTED_POLICY_COUNT = 1 + len(STABLE_DRAFT_VALUES) * len(HOLD_TOKEN_VALUES)


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "mt_selection" / "incremental_translation"

TARGET_DRAFTS_PATH = INPUT_DIR / "target_drafts.jsonl"

MT10_SUMMARY_PATH = INPUT_DIR / "incremental_summary.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "target_commit_policy_sweep"

TARGET_COMMITS_PATH = RESULTS_DIR / "target_commits.jsonl"

SEQUENCE_RESULTS_PATH = RESULTS_DIR / "sweep_by_sequence.csv"

SWEEP_RESULTS_PATH = RESULTS_DIR / "sweep_summary.csv"

PARETO_RESULTS_PATH = RESULTS_DIR / "pareto_frontier.csv"

SUMMARY_PATH = RESULTS_DIR / "target_commit_policy_sweep_summary.json"


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
    limit = min(
        len(first),
        len(second),
    )

    index = 0

    while index < limit and first[index] == second[index]:
        index += 1

    return first[:index]


def longest_common_prefix_many(
    sequences: list[list[str]],
) -> list[str]:
    if not sequences:
        return []

    prefix = sequences[0].copy()

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


def apply_hold(
    tokens: list[str],
    hold_tokens: int,
) -> list[str]:
    if hold_tokens == 0:
        return tokens.copy()

    if len(tokens) <= hold_tokens:
        return []

    return tokens[:-hold_tokens]


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
        raise RuntimeError("mt_10 was not generated in FP16.")

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
            raise RuntimeError(f"Draft {index} is missing: {sorted(missing)}")

    return (
        drafts,
        mt10_summary,
    )


# =============================================================================
# Sequence reconstruction
# =============================================================================


def group_and_validate_sequences(
    drafts: list[dict],
    mt10_summary: dict,
) -> tuple[
    dict[
        tuple[int, str, str],
        list[dict],
    ],
    float,
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
        raise RuntimeError(f"Unexpected number of sequences.\nExpected: {EXPECTED_SEQUENCES}\nActual  : {len(grouped)}")

    direction_counts = defaultdict(int)

    oracle_coverages = []

    for key, sequence in grouped.items():
        (
            utterance_index,
            source_language,
            target_language,
        ) = key

        if source_language == target_language:
            raise RuntimeError(f"Invalid same-language sequence: {key}")

        if source_language not in LANGUAGES or target_language not in LANGUAGES:
            raise RuntimeError(f"Unexpected language in {key}.")

        sequence.sort(key=lambda row: int(row["commit_index"]))

        expected_indices = list(
            range(
                1,
                len(sequence) + 1,
            )
        )

        actual_indices = [int(row["commit_index"]) for row in sequence]

        if actual_indices != expected_indices:
            raise RuntimeError(f"Non-contiguous target draft indices for {key}.")

        final_flush_count = sum(1 for row in sequence if row["is_final_flush"])

        if final_flush_count != 1:
            raise RuntimeError(f"Expected one final flush for {key}.")

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

        if len(sequence) >= 2:
            oracle_coverages.append(float(sequence[-2]["oracle_stable_prefix_coverage_percent"]))
        else:
            oracle_coverages.append(0.0)

    if len(direction_counts) != EXPECTED_DIRECTIONS:
        raise RuntimeError("Unexpected number of directions.")

    for direction, count in direction_counts.items():
        if count != EXPECTED_UTTERANCES_PER_DIRECTION:
            raise RuntimeError(f"Direction {direction} has {count} sequences.")

    calculated_oracle_mean = statistics.mean(oracle_coverages)

    expected_oracle_mean = float(mt10_summary["target_stability"]["mean_pre_final_stable_coverage_percent"])

    if abs(calculated_oracle_mean - expected_oracle_mean) > 0.05:
        raise RuntimeError(
            "Oracle coverage does not match mt_10.\n"
            f"Calculated: "
            f"{calculated_oracle_mean:.4f}%\n"
            f"mt_10      : "
            f"{expected_oracle_mean:.4f}%"
        )

    return (
        dict(grouped),
        calculated_oracle_mean,
    )


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
    committed_tokens: list[str],
    candidate_tokens: list[str],
    is_final_flush: bool,
) -> list[str]:
    old_length = len(committed_tokens)

    new_committed_tokens = candidate_tokens.copy()

    added_tokens = new_committed_tokens[old_length:]

    if not added_tokens:
        return committed_tokens

    (
        utterance_index,
        source_language,
        target_language,
    ) = sequence_key

    events.append(
        {
            "policy": policy_name,
            "utterance_index": (utterance_index),
            "dataset_index": int(draft["dataset_index"]),
            "sample_id": (draft["sample_id"]),
            "source_language": (source_language),
            "target_language": (target_language),
            "source_commit_index": int(draft["commit_index"]),
            "source_audio_available_s": float(draft["audio_available_s"]),
            "target_available_s": float(draft["simulated_translation_finish_s"]),
            "is_final_flush": (is_final_flush),
            "added_token_count": len(added_tokens),
            "committed_token_count": len(new_committed_tokens),
            "added_text": " ".join(added_tokens),
            "committed_text": " ".join(new_committed_tokens),
        }
    )

    return new_committed_tokens


# =============================================================================
# Single policy evaluation
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

    hold_tokens = policy["hold_tokens"]

    final_draft = drafts[-1]

    final_tokens = final_draft["target_text"].split()

    if not final_tokens:
        raise RuntimeError(f"Empty final target for {sequence_key}.")

    committed_tokens: list[str] = []
    observed_target_tokens: list[list[str]] = []

    commit_events = []

    conflict_events = 0

    pre_final_drafts = drafts[:-1]

    # ------------------------------------------------------------------
    # Incremental target commitment
    # ------------------------------------------------------------------

    if policy_name != "final_only":
        for draft in pre_final_drafts:
            current_tokens = draft["target_text"].split()

            # If the model later produces a target
            # that no longer starts with text already
            # committed to TTS, this is a conflict.
            if committed_tokens and not is_token_prefix(
                committed_tokens,
                current_tokens,
            ):
                conflict_events += 1

            observed_target_tokens.append(current_tokens)

            if len(observed_target_tokens) < stable_drafts:
                continue

            window = observed_target_tokens[-stable_drafts:]

            stable_prefix = longest_common_prefix_many(window)

            candidate_tokens = apply_hold(
                stable_prefix,
                hold_tokens,
            )

            # Append-only target policy:
            # previously committed text is never
            # retracted.
            if not is_token_prefix(
                committed_tokens,
                candidate_tokens,
            ):
                continue

            if len(candidate_tokens) <= len(committed_tokens):
                continue

            committed_tokens = append_commit_event(
                events=commit_events,
                policy_name=(policy_name),
                sequence_key=(sequence_key),
                draft=draft,
                committed_tokens=(committed_tokens),
                candidate_tokens=(candidate_tokens),
                is_final_flush=False,
            )

    # ------------------------------------------------------------------
    # State before final source flush
    # ------------------------------------------------------------------

    pre_final_committed_tokens = committed_tokens.copy()

    pre_final_committed_count = len(pre_final_committed_tokens)

    final_token_count = len(final_tokens)

    pre_final_lcp_tokens = len(
        longest_common_prefix_tokens(
            pre_final_committed_tokens,
            final_tokens,
        )
    )

    pre_final_is_final_prefix = is_token_prefix(
        pre_final_committed_tokens,
        final_tokens,
    )

    pre_final_committed_coverage_percent = pre_final_committed_count / final_token_count * 100.0

    pre_final_safe_coverage_percent = pre_final_lcp_tokens / final_token_count * 100.0

    oracle_pre_final_coverage_percent = (
        float(pre_final_drafts[-1]["oracle_stable_prefix_coverage_percent"]) if pre_final_drafts else 0.0
    )

    # ------------------------------------------------------------------
    # Final target
    # ------------------------------------------------------------------

    if committed_tokens and not is_token_prefix(
        committed_tokens,
        final_tokens,
    ):
        conflict_events += 1

    if is_token_prefix(
        committed_tokens,
        final_tokens,
    ):
        if len(final_tokens) > len(committed_tokens):
            committed_tokens = append_commit_event(
                events=commit_events,
                policy_name=(policy_name),
                sequence_key=(sequence_key),
                draft=final_draft,
                committed_tokens=(committed_tokens),
                candidate_tokens=(final_tokens),
                is_final_flush=True,
            )

    final_committed_matches_target = committed_tokens == final_tokens

    final_prefix_safe = pre_final_is_final_prefix and final_committed_matches_target

    strict_safe = final_prefix_safe and conflict_events == 0

    # This is the amount of useful incremental
    # coverage if strict safety is mandatory.
    #
    # Unsafe sequences contribute zero because
    # their pre-final commitments could not safely
    # have been sent to irreversible TTS output.
    strict_safe_pre_final_coverage_percent = pre_final_committed_coverage_percent if strict_safe else 0.0

    # ------------------------------------------------------------------
    # Commit metrics
    # ------------------------------------------------------------------

    incremental_events = [event for event in commit_events if not event["is_final_flush"]]

    final_flush_events = [event for event in commit_events if event["is_final_flush"]]

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
    else:
        first_incremental_commit_s = None

    final_text_committed_s = None

    for event in commit_events:
        if event["committed_text"] == " ".join(final_tokens):
            final_text_committed_s = float(event["target_available_s"])
            break

    if final_text_committed_s is not None:
        final_commit_after_final_source_commit_ms = (final_text_committed_s - final_source_commit_s) * 1000.0
    else:
        final_commit_after_final_source_commit_ms = None

    commit_times = [float(event["target_available_s"]) for event in commit_events]

    commit_intervals_ms = [
        (commit_times[index] - commit_times[index - 1]) * 1000.0
        for index in range(
            1,
            len(commit_times),
        )
    ]

    added_token_counts = [int(event["added_token_count"]) for event in commit_events]

    incremental_added_tokens = sum(int(event["added_token_count"]) for event in incremental_events)

    final_flush_added_tokens = sum(int(event["added_token_count"]) for event in final_flush_events)

    (
        utterance_index,
        source_language,
        target_language,
    ) = sequence_key

    result = {
        "policy": policy_name,
        "stable_drafts": (stable_drafts),
        "hold_tokens": (hold_tokens),
        "utterance_index": (utterance_index),
        "dataset_index": int(final_draft["dataset_index"]),
        "sample_id": (final_draft["sample_id"]),
        "source_language": (source_language),
        "target_language": (target_language),
        "target_draft_count": len(drafts),
        "final_target_token_count": (final_token_count),
        "strict_safe": (strict_safe),
        "final_prefix_safe": (final_prefix_safe),
        "conflict_events": (conflict_events),
        "incremental_commit_count": len(incremental_events),
        "total_commit_count": len(commit_events),
        "incremental_added_tokens": (incremental_added_tokens),
        "final_flush_added_tokens": (final_flush_added_tokens),
        "mean_tokens_per_commit": (statistics.mean(added_token_counts) if added_token_counts else 0.0),
        "pre_final_committed_token_count": (pre_final_committed_count),
        "pre_final_lcp_tokens": (pre_final_lcp_tokens),
        "pre_final_is_final_prefix": (pre_final_is_final_prefix),
        "pre_final_committed_coverage_percent": (pre_final_committed_coverage_percent),
        "pre_final_safe_coverage_percent": (pre_final_safe_coverage_percent),
        "strict_safe_pre_final_coverage_percent": (strict_safe_pre_final_coverage_percent),
        "oracle_pre_final_coverage_percent": (oracle_pre_final_coverage_percent),
        "first_target_draft_s": (first_target_draft_s),
        "first_commit_s": (first_commit_s),
        "first_incremental_commit_s": (first_incremental_commit_s),
        "first_commit_delay_from_first_draft_ms": (first_commit_delay_ms),
        "mean_commit_interval_ms": (statistics.mean(commit_intervals_ms) if commit_intervals_ms else 0.0),
        "final_text_committed_s": (final_text_committed_s),
        "final_commit_after_final_source_commit_ms": (final_commit_after_final_source_commit_ms),
        "final_committed_matches_target": (final_committed_matches_target),
    }

    return (
        result,
        commit_events,
    )


# =============================================================================
# Evaluate complete sweep
# =============================================================================


def evaluate_sweep(
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
        print(f"[{policy_index:02d}/{EXPECTED_POLICY_COUNT:02d}] {policy['name']}")

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
                policy=policy,
                sequence_key=sequence_key,
                drafts=sequences[sequence_key],
            )

            sequence_results.append(result)

            all_commit_events.extend(commit_events)

    expected_result_count = EXPECTED_POLICY_COUNT * EXPECTED_SEQUENCES

    if len(sequence_results) != expected_result_count:
        raise RuntimeError(
            f"Unexpected sweep result count.\nExpected: {expected_result_count}\nActual  : {len(sequence_results)}"
        )

    return (
        sequence_results,
        all_commit_events,
    )


# =============================================================================
# Policy summaries
# =============================================================================


def summarize_sweep(
    sequence_results: list[dict],
) -> list[dict]:
    summaries = []

    for policy in POLICIES:
        policy_name = policy["name"]

        rows = [row for row in sequence_results if row["policy"] == policy_name]

        if len(rows) != EXPECTED_SEQUENCES:
            raise RuntimeError(f"Policy {policy_name} has {len(rows)} sequence rows.")

        strict_safe_sequences = sum(1 for row in rows if row["strict_safe"])

        final_prefix_safe_sequences = sum(1 for row in rows if row["final_prefix_safe"])

        conflict_sequences = sum(1 for row in rows if row["conflict_events"] > 0)

        conflict_events = sum(int(row["conflict_events"]) for row in rows)

        sequences_with_incremental_commit = sum(1 for row in rows if row["incremental_commit_count"] > 0)

        total_incremental_commits = sum(int(row["incremental_commit_count"]) for row in rows)

        total_commits = sum(int(row["total_commit_count"]) for row in rows)

        total_incremental_tokens = sum(int(row["incremental_added_tokens"]) for row in rows)

        total_final_flush_tokens = sum(int(row["final_flush_added_tokens"]) for row in rows)

        total_final_tokens = sum(int(row["final_target_token_count"]) for row in rows)

        pre_final_committed_coverages = [float(row["pre_final_committed_coverage_percent"]) for row in rows]

        pre_final_safe_coverages = [float(row["pre_final_safe_coverage_percent"]) for row in rows]

        strict_safe_coverages = [float(row["strict_safe_pre_final_coverage_percent"]) for row in rows]

        oracle_coverages = [float(row["oracle_pre_final_coverage_percent"]) for row in rows]

        first_commit_times = [float(row["first_commit_s"]) for row in rows if row["first_commit_s"] is not None]

        first_incremental_commit_times = [
            float(row["first_incremental_commit_s"]) for row in rows if row["first_incremental_commit_s"] is not None
        ]

        first_commit_delays = [
            float(row["first_commit_delay_from_first_draft_ms"])
            for row in rows
            if row["first_commit_delay_from_first_draft_ms"] is not None
        ]

        final_commit_delays = [
            float(row["final_commit_after_final_source_commit_ms"])
            for row in rows
            if row["final_commit_after_final_source_commit_ms"] is not None
        ]

        commit_intervals = [float(row["mean_commit_interval_ms"]) for row in rows if row["total_commit_count"] > 1]

        total_added_tokens = total_incremental_tokens + total_final_flush_tokens

        summary = {
            "policy": (policy_name),
            "stable_drafts": (policy["stable_drafts"]),
            "hold_tokens": (policy["hold_tokens"]),
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
            "total_incremental_commits": (total_incremental_commits),
            "mean_incremental_commits_per_sequence": (total_incremental_commits / EXPECTED_SEQUENCES),
            "total_commits": (total_commits),
            "mean_total_commits_per_sequence": (total_commits / EXPECTED_SEQUENCES),
            "mean_tokens_per_commit": (total_added_tokens / total_commits if total_commits else 0.0),
            "mean_incremental_tokens_per_sequence": (total_incremental_tokens / EXPECTED_SEQUENCES),
            "mean_final_flush_tokens_per_sequence": (total_final_flush_tokens / EXPECTED_SEQUENCES),
            "mean_pre_final_committed_coverage_percent": (statistics.mean(pre_final_committed_coverages)),
            "mean_pre_final_safe_coverage_percent": (statistics.mean(pre_final_safe_coverages)),
            "mean_strict_safe_pre_final_coverage_percent": (statistics.mean(strict_safe_coverages)),
            "median_strict_safe_pre_final_coverage_percent": (statistics.median(strict_safe_coverages)),
            "mean_oracle_pre_final_coverage_percent": (statistics.mean(oracle_coverages)),
            "strict_safe_coverage_gap_to_oracle_pp": (
                statistics.mean(strict_safe_coverages) - statistics.mean(oracle_coverages)
            ),
            "micro_incremental_coverage_percent": (
                total_incremental_tokens / total_final_tokens * 100.0 if total_final_tokens else 0.0
            ),
            "mean_first_commit_s": (statistics.mean(first_commit_times) if first_commit_times else None),
            "median_first_commit_s": (statistics.median(first_commit_times) if first_commit_times else None),
            "mean_first_incremental_commit_s": (
                statistics.mean(first_incremental_commit_times) if first_incremental_commit_times else None
            ),
            "mean_first_commit_delay_from_first_draft_ms": (
                statistics.mean(first_commit_delays) if first_commit_delays else None
            ),
            "mean_commit_interval_ms": (statistics.mean(commit_intervals) if commit_intervals else 0.0),
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

    candidate_latency = float(candidate["mean_first_commit_s"])

    other_latency = float(other["mean_first_commit_s"])

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
        is_dominated = False

        for other in summaries:
            if other["policy"] == candidate["policy"]:
                continue

            if dominates(
                other,
                candidate,
            ):
                is_dominated = True
                break

        if not is_dominated:
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
# Console reports
# =============================================================================


def print_top_safety_configs(
    summaries: list[dict],
) -> None:
    stable_policies = [row for row in summaries if row["policy"] != "final_only"]

    stable_policies.sort(
        key=lambda row: (
            -float(row["strict_safe_percent"]),
            -float(row["mean_strict_safe_pre_final_coverage_percent"]),
            float(row["mean_first_commit_s"]),
        )
    )

    print()
    print("=" * 118)
    print("TOP 10 POLICIES BY STRICT SAFETY")
    print("=" * 118)

    print(
        f"{'Policy':<18}"
        f"{'Safe':>9}"
        f"{'Conflicts':>11}"
        f"{'Raw cov.':>11}"
        f"{'Safe cov.':>11}"
        f"{'Strict cov.':>13}"
        f"{'1st commit':>13}"
        f"{'Inc/seq':>10}"
        f"{'Tok/commit':>12}"
    )

    print("-" * 118)

    for row in stable_policies[:10]:
        print(
            f"{row['policy']:<18}"
            f"{row['strict_safe_percent']:>8.1f}%"
            f"{row['conflict_sequences']:>11}"
            f"{row['mean_pre_final_committed_coverage_percent']:>10.1f}%"
            f"{row['mean_pre_final_safe_coverage_percent']:>10.1f}%"
            f"{row['mean_strict_safe_pre_final_coverage_percent']:>12.1f}%"
            f"{row['mean_first_commit_s']:>12.2f}s"
            f"{row['mean_incremental_commits_per_sequence']:>10.2f}"
            f"{row['mean_tokens_per_commit']:>12.2f}"
        )


def print_pareto_frontier(
    frontier: list[dict],
) -> None:
    print()
    print("=" * 100)
    print("PARETO FRONTIER")
    print("=" * 100)

    print("Objectives: maximize strict safety, maximize strict-safe pre-final coverage, minimize first commit time.")

    print()

    print(f"{'Policy':<18}{'Strict safe':>14}{'Strict cov.':>14}{'1st commit':>14}{'Inc/seq':>12}{'Tok/commit':>14}")

    print("-" * 100)

    for row in frontier:
        print(
            f"{row['policy']:<18}"
            f"{row['strict_safe_percent']:>13.1f}%"
            f"{row['mean_strict_safe_pre_final_coverage_percent']:>13.1f}%"
            f"{row['mean_first_commit_s']:>13.2f}s"
            f"{row['mean_incremental_commits_per_sequence']:>12.2f}"
            f"{row['mean_tokens_per_commit']:>14.2f}"
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 80)
    print("MT TARGET COMMIT POLICY SWEEP")
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

    (
        sequences,
        oracle_mean,
    ) = group_and_validate_sequences(
        drafts=drafts,
        mt10_summary=mt10_summary,
    )

    print()
    print(f"Sequences     : {len(sequences)}")

    print(f"Directions    : {EXPECTED_DIRECTIONS}")

    print(f"Oracle pre-final stable coverage: {oracle_mean:.1f}%")

    print("Sequence validation: PASS")

    if len(POLICIES) != EXPECTED_POLICY_COUNT:
        raise RuntimeError("Unexpected policy count.")

    print()
    print(f"Policies      : {len(POLICIES)}")

    print(f"Stable drafts : {STABLE_DRAFT_VALUES}")

    print(f"Hold tokens   : {HOLD_TOKEN_VALUES}")

    print()
    print("Evaluating sweep...")

    (
        sequence_results,
        commit_events,
    ) = evaluate_sweep(sequences)

    sweep_summaries = summarize_sweep(sequence_results)

    # ------------------------------------------------------------------
    # Sanity check: final_only
    # ------------------------------------------------------------------

    final_only = next(row for row in sweep_summaries if row["policy"] == "final_only")

    if final_only["strict_safe_sequences"] != EXPECTED_SEQUENCES:
        raise RuntimeError("final_only is not 100% strict-safe.")

    if final_only["conflict_sequences"] != 0:
        raise RuntimeError("final_only produced conflicts.")

    if abs(final_only["mean_pre_final_committed_coverage_percent"]) > 1e-9:
        raise RuntimeError("final_only unexpectedly committed target text before final.")

    print()
    print("final_only sanity check: PASS")

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    pareto_frontier = calculate_pareto_frontier(sweep_summaries)

    # ------------------------------------------------------------------
    # Outputs
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
        SWEEP_RESULTS_PATH,
        sweep_summaries,
    )

    write_csv(
        PARETO_RESULTS_PATH,
        pareto_frontier,
    )

    best_non_final_by_safety = max(
        (row for row in sweep_summaries if row["policy"] != "final_only"),
        key=lambda row: (
            float(row["strict_safe_percent"]),
            float(row["mean_strict_safe_pre_final_coverage_percent"]),
            -float(row["mean_first_commit_s"]),
        ),
    )

    best_non_final_by_strict_coverage = max(
        (row for row in sweep_summaries if row["policy"] != "final_only"),
        key=lambda row: (
            float(row["mean_strict_safe_pre_final_coverage_percent"]),
            float(row["strict_safe_percent"]),
            -float(row["mean_first_commit_s"]),
        ),
    )

    summary = {
        "experiment": ("Offline sweep of persistence-based append-only target commit policies."),
        "input": {
            "source_experiment": ("mt_10_analyze_incremental_translation"),
            "target_drafts": (str(TARGET_DRAFTS_PATH)),
            "target_draft_count": (len(drafts)),
            "sequences": (len(sequences)),
            "directions": (EXPECTED_DIRECTIONS),
            "model_id": (EXPECTED_MODEL_ID),
            "model_revision": (EXPECTED_MODEL_REVISION),
            "precision": "float16",
            "oracle_mean_pre_final_stable_coverage_percent": (oracle_mean),
        },
        "sweep": {
            "stable_draft_values": (STABLE_DRAFT_VALUES),
            "hold_token_values": (HOLD_TOKEN_VALUES),
            "stable_policy_count": (len(STABLE_DRAFT_VALUES) * len(HOLD_TOKEN_VALUES)),
            "total_policy_count": (len(POLICIES)),
        },
        "methodology": {
            "target_tokens": ("Whitespace-separated decoded surface tokens."),
            "candidate_prefix": ("Longest common prefix of the last N target drafts."),
            "hold": ("The final M tokens of the stable prefix remain uncommitted."),
            "append_only": ("Committed target text is never retracted."),
            "strict_safety": (
                "No future target draft may "
                "contradict already committed text, "
                "and the final target must be "
                "completable append-only."
            ),
            "strict_safe_pre_final_coverage": (
                "Pre-final target coverage is counted only for strict-safe sequences; unsafe sequences contribute zero."
            ),
            "pareto_frontier": {
                "maximize": [
                    "strict_safe_percent",
                    "mean_strict_safe_pre_final_coverage_percent",
                ],
                "minimize": [
                    "mean_first_commit_s",
                ],
            },
            "inference": ("No MT inference is executed."),
        },
        "final_only": (final_only),
        "best_non_final_by_safety": (best_non_final_by_safety),
        "best_non_final_by_strict_safe_coverage": (best_non_final_by_strict_coverage),
        "pareto_frontier": (pareto_frontier),
        "all_policies": (sweep_summaries),
        "environment": {
            "python": (platform.python_version()),
        },
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
    # Console summary
    # ------------------------------------------------------------------

    print_top_safety_configs(sweep_summaries)

    print_pareto_frontier(pareto_frontier)

    print()
    print("=" * 80)
    print("BEST NON-FINAL POLICIES")
    print("=" * 80)

    print()
    print("Highest strict safety:")

    print(f"  Policy          : {best_non_final_by_safety['policy']}")

    print(f"  Strict safe     : {best_non_final_by_safety['strict_safe_percent']:.1f}%")

    print(f"  Strict coverage : {best_non_final_by_safety['mean_strict_safe_pre_final_coverage_percent']:.1f}%")

    print(f"  First commit    : {best_non_final_by_safety['mean_first_commit_s']:.2f} s")

    print()
    print("Highest strict-safe coverage:")

    print(f"  Policy          : {best_non_final_by_strict_coverage['policy']}")

    print(f"  Strict safe     : {best_non_final_by_strict_coverage['strict_safe_percent']:.1f}%")

    print(
        f"  Strict coverage : {best_non_final_by_strict_coverage['mean_strict_safe_pre_final_coverage_percent']:.1f}%"
    )

    print(f"  First commit    : {best_non_final_by_strict_coverage['mean_first_commit_s']:.2f} s")

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Target commits : {TARGET_COMMITS_PATH}")

    print(f"Per sequence   : {SEQUENCE_RESULTS_PATH}")

    print(f"Sweep summary  : {SWEEP_RESULTS_PATH}")

    print(f"Pareto frontier: {PARETO_RESULTS_PATH}")

    print(f"Summary        : {SUMMARY_PATH}")

    print()
    print("=" * 80)
    print("MT TARGET COMMIT POLICY SWEEP PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
