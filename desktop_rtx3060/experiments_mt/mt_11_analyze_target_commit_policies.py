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
# Policies
# =============================================================================

POLICIES = [
    {
        "name": "final_only",
        "stable_drafts": None,
        "hold_tokens": None,
    },
    {
        "name": "stable2_hold1",
        "stable_drafts": 2,
        "hold_tokens": 1,
    },
    {
        "name": "stable2_hold2",
        "stable_drafts": 2,
        "hold_tokens": 2,
    },
    {
        "name": "stable3_hold1",
        "stable_drafts": 3,
        "hold_tokens": 1,
    },
    {
        "name": "stable3_hold2",
        "stable_drafts": 3,
        "hold_tokens": 2,
    },
]


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "mt_selection" / "incremental_translation"

TARGET_DRAFTS_PATH = INPUT_DIR / "target_drafts.jsonl"

MT10_SUMMARY_PATH = INPUT_DIR / "incremental_summary.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "target_commit_policies"

TARGET_COMMITS_PATH = RESULTS_DIR / "target_commits.jsonl"

SEQUENCE_RESULTS_PATH = RESULTS_DIR / "policy_by_sequence.csv"

POLICY_RESULTS_PATH = RESULTS_DIR / "policy_summary.csv"

SUMMARY_PATH = RESULTS_DIR / "target_commit_policy_summary.json"


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
            f"Unexpected MT model in mt_10 summary.\nExpected: {EXPECTED_MODEL_ID}\nActual  : {mt.get('model_id')}"
        )

    if mt.get("revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError(
            "Unexpected MT revision in mt_10 summary.\n"
            f"Expected: {EXPECTED_MODEL_REVISION}\n"
            f"Actual  : {mt.get('revision')}"
        )

    if mt.get("precision") != "float16":
        raise RuntimeError("mt_10 was not generated with FP16.")

    scope = mt10_summary.get(
        "scope",
        {},
    )

    if scope.get("utterance_target_sequences") != EXPECTED_SEQUENCES:
        raise RuntimeError("Unexpected mt_10 sequence count.")

    if scope.get("mt_invocations") != EXPECTED_TARGET_DRAFTS:
        raise RuntimeError("Unexpected mt_10 MT invocation count.")

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
        "source_committed_text",
        "target_text",
        "total_ms",
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
            f"Expected: {EXPECTED_SEQUENCES}\n"
            f"Actual  : {len(grouped)}"
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
            raise RuntimeError(f"Unexpected language in {key}.")

        sequence.sort(key=lambda row: int(row["commit_index"]))

        expected_commit_indices = list(
            range(
                1,
                len(sequence) + 1,
            )
        )

        actual_commit_indices = [int(row["commit_index"]) for row in sequence]

        if actual_commit_indices != expected_commit_indices:
            raise RuntimeError(f"Non-contiguous commit indices for sequence {key}.")

        final_flush_count = sum(1 for row in sequence if row["is_final_flush"])

        if final_flush_count != 1:
            raise RuntimeError(f"Expected exactly one final flush for sequence {key}, found {final_flush_count}.")

        if not sequence[-1]["is_final_flush"]:
            raise RuntimeError(f"Final source flush is not the last draft for sequence {key}.")

        finish_times = [float(row["simulated_translation_finish_s"]) for row in sequence]

        if finish_times != sorted(finish_times):
            raise RuntimeError(f"Non-monotonic simulated MT completion times for {key}.")

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
# Commit event helper
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
            "target_draft_available_s": float(draft["simulated_translation_finish_s"]),
            "is_final_flush": (is_final_flush),
            "added_token_count": len(added_tokens),
            "committed_token_count": len(new_committed_tokens),
            "added_text": " ".join(added_tokens),
            "committed_text": " ".join(new_committed_tokens),
        }
    )

    return new_committed_tokens


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

    hold_tokens = policy["hold_tokens"]

    final_draft = drafts[-1]

    final_tokens = final_draft["target_text"].split()

    if not final_tokens:
        raise RuntimeError(f"Empty final target for {sequence_key}.")

    committed_tokens: list[str] = []

    commit_events = []

    observed_target_tokens = []

    conflict_events = 0

    # ------------------------------------------------------------------
    # Online processing before the final source flush.
    # ------------------------------------------------------------------

    pre_final_drafts = drafts[:-1]

    if policy_name != "final_only":
        for draft in pre_final_drafts:
            current_tokens = draft["target_text"].split()

            # Once text has been committed, every
            # subsequent target draft should keep
            # that text as a prefix.
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

            if len(stable_prefix) <= hold_tokens:
                candidate_tokens = []

            else:
                candidate_tokens = stable_prefix[:-hold_tokens]

            # Never retract committed target text.
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
    # State immediately before the final target draft.
    # ------------------------------------------------------------------

    pre_final_committed_tokens = committed_tokens.copy()

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

    pre_final_safe_coverage_percent = pre_final_lcp_tokens / len(final_tokens) * 100.0

    oracle_pre_final_coverage_percent = (
        float(pre_final_drafts[-1]["oracle_stable_prefix_coverage_percent"]) if pre_final_drafts else 0.0
    )

    # ------------------------------------------------------------------
    # Final target draft.
    # ------------------------------------------------------------------

    if committed_tokens and not is_token_prefix(
        committed_tokens,
        final_tokens,
    ):
        conflict_events += 1

    # A final flush can only safely append the
    # remaining target if previous commitments
    # are still a prefix of the final translation.
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

    # Strict safety is the condition that matters
    # most for future TTS delivery:
    # no future target draft ever contradicted
    # already committed text.
    strict_safe = final_prefix_safe and conflict_events == 0

    # ------------------------------------------------------------------
    # Commit timing metrics
    # ------------------------------------------------------------------

    incremental_events = [event for event in commit_events if not event["is_final_flush"]]

    final_flush_events = [event for event in commit_events if event["is_final_flush"]]

    first_target_draft_s = float(drafts[0]["simulated_translation_finish_s"])

    final_source_commit_s = float(final_draft["audio_available_s"])

    if commit_events:
        first_commit_s = float(commit_events[0]["target_draft_available_s"])

        first_commit_delay_ms = (first_commit_s - first_target_draft_s) * 1000.0

    else:
        first_commit_s = None
        first_commit_delay_ms = None

    if incremental_events:
        first_incremental_commit_s = float(incremental_events[0]["target_draft_available_s"])
    else:
        first_incremental_commit_s = None

    # Find when the complete final target became
    # append-only committed.
    final_text_committed_s = None

    for event in commit_events:
        if event["committed_text"] == " ".join(final_tokens):
            final_text_committed_s = float(event["target_draft_available_s"])
            break

    if final_text_committed_s is not None:
        final_commit_after_final_source_commit_ms = (final_text_committed_s - final_source_commit_s) * 1000.0
    else:
        final_commit_after_final_source_commit_ms = None

    commit_times = [float(event["target_draft_available_s"]) for event in commit_events]

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
        "utterance_index": (utterance_index),
        "dataset_index": int(final_draft["dataset_index"]),
        "sample_id": (final_draft["sample_id"]),
        "source_language": (source_language),
        "target_language": (target_language),
        "target_draft_count": len(drafts),
        "final_target_token_count": len(final_tokens),
        "final_prefix_safe": (final_prefix_safe),
        "strict_safe": (strict_safe),
        "conflict_events": (conflict_events),
        "incremental_commit_count": len(incremental_events),
        "total_commit_count": len(commit_events),
        "incremental_added_tokens": (incremental_added_tokens),
        "final_flush_added_tokens": (final_flush_added_tokens),
        "mean_tokens_per_commit": (statistics.mean(added_token_counts) if added_token_counts else 0.0),
        "pre_final_committed_token_count": len(pre_final_committed_tokens),
        "pre_final_safe_lcp_tokens": (pre_final_lcp_tokens),
        "pre_final_is_final_prefix": (pre_final_is_final_prefix),
        "pre_final_safe_coverage_percent": (pre_final_safe_coverage_percent),
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
# Evaluate all policies
# =============================================================================


def evaluate_all_policies(
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

    for policy in POLICIES:
        print()
        print("=" * 80)
        print(f"POLICY: {policy['name']}")
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
            result, commit_events = evaluate_policy_on_sequence(
                policy=policy,
                sequence_key=(sequence_key),
                drafts=sequences[sequence_key],
            )

            policy_results.append(result)

            sequence_results.append(result)

            all_commit_events.extend(commit_events)

        strict_safe = sum(1 for row in policy_results if row["strict_safe"])

        final_safe = sum(1 for row in policy_results if row["final_prefix_safe"])

        conflict_sequences = sum(1 for row in policy_results if row["conflict_events"] > 0)

        coverage = statistics.mean(row["pre_final_safe_coverage_percent"] for row in policy_results)

        incremental_commit_sequences = sum(1 for row in policy_results if row["incremental_commit_count"] > 0)

        print(f"Strict safe sequences       : {strict_safe}/{EXPECTED_SEQUENCES}")

        print(f"Final-prefix safe sequences : {final_safe}/{EXPECTED_SEQUENCES}")

        print(f"Sequences with conflicts    : {conflict_sequences}/{EXPECTED_SEQUENCES}")

        print(f"Sequences with pre-final MT commit: {incremental_commit_sequences}/{EXPECTED_SEQUENCES}")

        print(f"Mean pre-final coverage     : {coverage:.1f}%")

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
            raise RuntimeError(f"Policy {policy_name} has {len(rows)} sequence results.")

        strict_safe_sequences = sum(1 for row in rows if row["strict_safe"])

        final_prefix_safe_sequences = sum(1 for row in rows if row["final_prefix_safe"])

        conflict_sequences = sum(1 for row in rows if row["conflict_events"] > 0)

        conflict_events = sum(int(row["conflict_events"]) for row in rows)

        sequences_with_incremental_commit = sum(1 for row in rows if row["incremental_commit_count"] > 0)

        total_incremental_commits = sum(int(row["incremental_commit_count"]) for row in rows)

        total_commits = sum(int(row["total_commit_count"]) for row in rows)

        total_incremental_tokens = sum(int(row["incremental_added_tokens"]) for row in rows)

        total_final_flush_tokens = sum(int(row["final_flush_added_tokens"]) for row in rows)

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

        pre_final_coverages = [float(row["pre_final_safe_coverage_percent"]) for row in rows]

        oracle_coverages = [float(row["oracle_pre_final_coverage_percent"]) for row in rows]

        total_final_target_tokens = sum(int(row["final_target_token_count"]) for row in rows)

        total_added_tokens = total_incremental_tokens + total_final_flush_tokens

        summary = {
            "policy": (policy_name),
            "stable_drafts": (policy["stable_drafts"]),
            "hold_tokens": (policy["hold_tokens"]),
            "sequences": (len(rows)),
            "strict_safe_sequences": (strict_safe_sequences),
            "strict_safe_percent": (strict_safe_sequences / len(rows) * 100.0),
            "final_prefix_safe_sequences": (final_prefix_safe_sequences),
            "final_prefix_safe_percent": (final_prefix_safe_sequences / len(rows) * 100.0),
            "conflict_sequences": (conflict_sequences),
            "conflict_events": (conflict_events),
            "sequences_with_incremental_commit": (sequences_with_incremental_commit),
            "sequences_with_incremental_commit_percent": (sequences_with_incremental_commit / len(rows) * 100.0),
            "total_incremental_commits": (total_incremental_commits),
            "mean_incremental_commits_per_sequence": (total_incremental_commits / len(rows)),
            "total_commits": (total_commits),
            "mean_total_commits_per_sequence": (total_commits / len(rows)),
            "mean_tokens_per_commit": (total_added_tokens / total_commits if total_commits else 0.0),
            "mean_incremental_tokens_per_sequence": (total_incremental_tokens / len(rows)),
            "mean_final_flush_tokens_per_sequence": (total_final_flush_tokens / len(rows)),
            "mean_pre_final_safe_coverage_percent": (statistics.mean(pre_final_coverages)),
            "median_pre_final_safe_coverage_percent": (statistics.median(pre_final_coverages)),
            "p95_pre_final_safe_coverage_percent": (
                percentile(
                    pre_final_coverages,
                    95,
                )
            ),
            "mean_oracle_pre_final_coverage_percent": (statistics.mean(oracle_coverages)),
            "coverage_gap_to_oracle_pp": (statistics.mean(pre_final_coverages) - statistics.mean(oracle_coverages)),
            "micro_pre_final_incremental_coverage_percent": (
                total_incremental_tokens / total_final_target_tokens * 100.0 if total_final_target_tokens else 0.0
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
# Console summary
# =============================================================================


def print_policy_summary(
    summaries: list[dict],
) -> None:
    print()
    print("=" * 120)
    print("TARGET COMMIT POLICY SUMMARY")
    print("=" * 120)

    header = (
        f"{'Policy':<18}"
        f"{'Strict safe':>14}"
        f"{'Conflicts':>12}"
        f"{'Pre-final':>12}"
        f"{'1st commit':>14}"
        f"{'Inc commits':>14}"
        f"{'Tok/commit':>13}"
        f"{'Final flush':>13}"
    )

    print(header)
    print("-" * 120)

    for row in summaries:
        first_commit = f"{row['mean_first_commit_s']:.2f}s" if row["mean_first_commit_s"] is not None else "-"

        print(
            f"{row['policy']:<18}"
            f"{row['strict_safe_percent']:>13.1f}%"
            f"{row['conflict_sequences']:>12}"
            f"{row['mean_pre_final_safe_coverage_percent']:>11.1f}%"
            f"{first_commit:>14}"
            f"{row['mean_incremental_commits_per_sequence']:>14.2f}"
            f"{row['mean_tokens_per_commit']:>13.2f}"
            f"{row['mean_final_flush_tokens_per_sequence']:>13.2f}"
        )

    print()
    print(f"Oracle mean pre-final stable coverage: {summaries[0]['mean_oracle_pre_final_coverage_percent']:.1f}%")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 80)
    print("MT TARGET COMMIT POLICY ANALYSIS")
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

    print("Sequence validation: PASS")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Evaluating target commit policies...")

    (
        sequence_results,
        commit_events,
    ) = evaluate_all_policies(sequences)

    policy_summaries = summarize_policies(sequence_results)

    write_csv(
        SEQUENCE_RESULTS_PATH,
        sequence_results,
    )

    write_csv(
        POLICY_RESULTS_PATH,
        policy_summaries,
    )

    write_commit_events(commit_events)

    summary = {
        "experiment": (
            "Offline evaluation of append-only target commit policies over NLLB FP16 full-prefix translation drafts."
        ),
        "input": {
            "source_experiment": ("mt_10_analyze_incremental_translation"),
            "target_drafts": (str(TARGET_DRAFTS_PATH)),
            "target_draft_count": (len(drafts)),
            "sequences": (len(sequences)),
            "directions": (EXPECTED_DIRECTIONS),
            "model_id": (EXPECTED_MODEL_ID),
            "model_revision": (EXPECTED_MODEL_REVISION),
            "precision": "float16",
        },
        "policies": (policy_summaries),
        "methodology": {
            "target_token_definition": ("Whitespace-separated surface tokens from decoded NLLB text."),
            "draft_availability_time": (
                "simulated_translation_finish_s "
                "from mt_10; therefore a target "
                "draft can only be committed after "
                "its MT inference has completed."
            ),
            "stable_policy": (
                "Longest common prefix of the "
                "last N target drafts, followed "
                "by holding the last M stable "
                "surface tokens."
            ),
            "append_only": ("Previously committed target tokens are never retracted."),
            "final_flush": (
                "At the final source commit, the "
                "remaining final target is appended "
                "only if previous commitments still "
                "form a prefix of the final target."
            ),
            "strict_safety": (
                "A sequence is strict-safe only "
                "when no future target draft "
                "contradicts already committed text "
                "and the final target can be "
                "completed append-only."
            ),
            "final_prefix_safety": (
                "Checks only whether pre-final committed text remains a prefix of the final target."
            ),
            "oracle_pre_final_coverage": (
                "Copied from mt_10 and uses future drafts. It is an analysis upper bound, not an online policy."
            ),
        },
        "environment": {
            "python": (platform.python_version()),
        },
        "notes": {
            "inference": ("No MT inference is executed by this experiment."),
            "tts": (
                "These policies are evaluated as candidate append-only text delivery mechanisms for a future TTS stage."
            ),
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

    print_policy_summary(policy_summaries)

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Target commits : {TARGET_COMMITS_PATH}")

    print(f"Per sequence   : {SEQUENCE_RESULTS_PATH}")

    print(f"Policy summary : {POLICY_RESULTS_PATH}")

    print(f"Summary        : {SUMMARY_PATH}")

    print()
    print("=" * 80)
    print("MT TARGET COMMIT POLICY ANALYSIS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
