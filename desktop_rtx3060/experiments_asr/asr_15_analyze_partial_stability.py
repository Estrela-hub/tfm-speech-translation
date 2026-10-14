from __future__ import annotations

import csv
import json
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

# ============================================================================
# Policies
# ============================================================================

POLICIES = {
    "stable2_hold1": {
        "stability_window": 2,
        "holdback_tokens": 1,
    },
    "stable3_hold1": {
        "stability_window": 3,
        "holdback_tokens": 1,
    },
}


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "asr_selection" / "parakeet_partial_traces"

PARTIAL_TRACES_PATH = INPUT_DIR / "partial_traces.jsonl"

UTTERANCES_PATH = INPUT_DIR / "utterances.jsonl"

METADATA_PATH = INPUT_DIR / "metadata.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "asr_selection" / "parakeet_partial_stability"

SUMMARY_JSON_PATH = RESULTS_DIR / "summary.json"

SUMMARY_CSV_PATH = RESULTS_DIR / "summary.csv"

UTTERANCE_ANALYSIS_PATH = RESULTS_DIR / "utterance_analysis.jsonl"

COMMIT_EVENTS_PATH = RESULTS_DIR / "commit_events.jsonl"


# ============================================================================
# JSON helpers
# ============================================================================


def read_jsonl(path: Path) -> list[dict]:
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
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from error

    return records


def read_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def write_json(
    path: Path,
    data: dict,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================================
# Text helpers
# ============================================================================


def normalize_spacing(text: str) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    text = normalize_spacing(text)

    if not text:
        return []

    return text.split(" ")


def tokens_to_text(
    tokens: list[str],
) -> str:
    return " ".join(tokens)


def starts_with(
    tokens: list[str],
    prefix: list[str],
) -> bool:
    if len(prefix) > len(tokens):
        return False

    return tokens[: len(prefix)] == prefix


def common_prefix_length(
    left: list[str],
    right: list[str],
) -> int:
    limit = min(
        len(left),
        len(right),
    )

    index = 0

    while index < limit and left[index] == right[index]:
        index += 1

    return index


def longest_common_prefix(
    sequences: list[list[str]],
) -> list[str]:
    if not sequences:
        return []

    prefix = list(sequences[0])

    for sequence in sequences[1:]:
        prefix_length = common_prefix_length(
            prefix,
            sequence,
        )

        prefix = prefix[:prefix_length]

        if not prefix:
            break

    return prefix


# ============================================================================
# Statistics helpers
# ============================================================================


def safe_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return statistics.mean(values)


def safe_median(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return statistics.median(values)


def round_or_none(
    value: float | None,
    digits: int = 4,
) -> float | None:
    if value is None:
        return None

    return round(
        value,
        digits,
    )


# ============================================================================
# Input validation
# ============================================================================


def validate_inputs() -> None:
    required_paths = [
        PARTIAL_TRACES_PATH,
        UTTERANCES_PATH,
        METADATA_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required partial-trace input was not found:\n{path}")


def group_traces(
    traces: list[dict],
) -> dict[int, list[dict]]:
    grouped = defaultdict(list)

    for trace in traces:
        utterance_index = int(trace["utterance_index"])

        grouped[utterance_index].append(trace)

    for utterance_index in grouped:
        grouped[utterance_index].sort(key=lambda record: int(record["chunk_index"]))

    return dict(grouped)


def validate_dataset(
    traces: list[dict],
    utterances: list[dict],
    metadata: dict,
) -> dict[int, list[dict]]:
    expected_utterances = metadata["collection"]["utterances"]

    if len(utterances) != expected_utterances:
        raise RuntimeError(
            "Utterance count differs from "
            "the collector metadata.\n\n"
            f"Metadata  : "
            f"{expected_utterances}\n"
            f"Actual    : "
            f"{len(utterances)}"
        )

    expected_partials = metadata["collection"]["partial_records"]

    if len(traces) != expected_partials:
        raise RuntimeError(
            "Partial count differs from "
            "the collector metadata.\n\n"
            f"Metadata  : "
            f"{expected_partials}\n"
            f"Actual    : "
            f"{len(traces)}"
        )

    final_matches = metadata["collection"]["final_partial_matches_prediction"]

    if final_matches != expected_utterances:
        raise RuntimeError(
            "The collector dataset is not clean: "
            "not every final partial matches the "
            "NeMo final prediction.\n\n"
            f"Matches: "
            f"{final_matches}/"
            f"{expected_utterances}"
        )

    grouped = group_traces(traces)

    for utterance in utterances:
        utterance_index = int(utterance["utterance_index"])

        if utterance_index not in grouped:
            raise RuntimeError(f"Missing partial traces for utterance {utterance_index}.")

        sample_traces = grouped[utterance_index]

        expected_chunk = 1

        for trace in sample_traces:
            actual_chunk = int(trace["chunk_index"])

            if actual_chunk != expected_chunk:
                raise RuntimeError(
                    "Non-contiguous chunk sequence "
                    f"for utterance "
                    f"{utterance_index}.\n\n"
                    f"Expected chunk "
                    f"{expected_chunk}, "
                    f"found "
                    f"{actual_chunk}."
                )

            expected_chunk += 1

        final_partial = normalize_spacing(sample_traces[-1]["partial_text"])

        final_prediction = normalize_spacing(utterance["final_prediction"])

        if final_partial != final_prediction:
            raise RuntimeError(f"Final partial does not match the final prediction for utterance {utterance_index}.")

    return grouped


# ============================================================================
# Token appearance analysis
# ============================================================================


def get_first_correct_appearance_times(
    traces: list[dict],
    final_tokens: list[str],
) -> list[float | None]:
    """
    For each token position in the final hypothesis,
    return the first audio time at which the partial
    hypothesis contained the correct final prefix up
    to that token.

    This is used to estimate how long the commit policy
    waits after a token has first appeared correctly.
    """

    appearance_times: list[float | None] = [None for _ in final_tokens]

    for trace in traces:
        partial_tokens = tokenize(trace["partial_text"])

        prefix_length = common_prefix_length(
            partial_tokens,
            final_tokens,
        )

        audio_available_s = float(trace["audio_available_s"])

        for token_index in range(prefix_length):
            if appearance_times[token_index] is None:
                appearance_times[token_index] = audio_available_s

    return appearance_times


# ============================================================================
# Policy evaluation
# ============================================================================


def evaluate_policy_on_utterance(
    policy_name: str,
    policy: dict,
    utterance: dict,
    traces: list[dict],
) -> tuple[
    dict,
    list[dict],
]:
    stability_window = int(policy["stability_window"])

    holdback_tokens = int(policy["holdback_tokens"])

    final_prediction = normalize_spacing(utterance["final_prediction"])

    final_tokens = tokenize(final_prediction)

    first_appearance_times = get_first_correct_appearance_times(
        traces=traces,
        final_tokens=final_tokens,
    )

    committed_tokens: list[str] = []

    # Commit time for each committed token.
    committed_token_times: list[float] = []

    history: list[list[str]] = []

    commit_events: list[dict] = []

    transient_conflict_partials = 0
    first_transient_conflict_s = None

    stable_commit_events = 0

    first_stable_commit_s = None

    first_non_empty_s = utterance.get("first_non_empty_audio_s")

    # ------------------------------------------------------------------
    # Online simulation
    # ------------------------------------------------------------------

    for trace in traces:
        chunk_index = int(trace["chunk_index"])

        audio_available_s = float(trace["audio_available_s"])

        partial_text = normalize_spacing(trace["partial_text"])

        partial_tokens = tokenize(partial_text)

        # If previously committed text is no longer
        # compatible with the current partial, this
        # represents a temporary retraction/conflict.
        if committed_tokens and not starts_with(
            partial_tokens,
            committed_tokens,
        ):
            transient_conflict_partials += 1

            if first_transient_conflict_s is None:
                first_transient_conflict_s = audio_available_s

        history.append(partial_tokens)

        if len(history) < stability_window:
            continue

        recent_hypotheses = history[-stability_window:]

        stable_prefix = longest_common_prefix(recent_hypotheses)

        # Hold back the configured number of trailing tokens.
        # With the current policies, the final token remains uncommitted
        # because the collector strips trailing whitespace and cannot
        # distinguish a complete word from an unfinished fragment.
        candidate_length = max(
            0,
            len(stable_prefix) - holdback_tokens,
        )

        candidate_tokens = stable_prefix[:candidate_length]

        if len(candidate_tokens) <= len(committed_tokens):
            continue

        # Once text is committed it cannot be retracted.
        # Therefore a new candidate may only extend the
        # previously committed prefix.
        if not starts_with(
            candidate_tokens,
            committed_tokens,
        ):
            continue

        new_tokens = candidate_tokens[len(committed_tokens) :]

        if not new_tokens:
            continue

        previous_committed_count = len(committed_tokens)

        committed_tokens.extend(new_tokens)

        committed_token_times.extend([audio_available_s for _ in new_tokens])

        stable_commit_events += 1

        if first_stable_commit_s is None:
            first_stable_commit_s = audio_available_s

        commit_events.append(
            {
                "policy": policy_name,
                "event_type": ("stable_commit"),
                "utterance_index": int(utterance["utterance_index"]),
                "language": utterance["language"],
                "sample_id": utterance["sample_id"],
                "chunk_index": (chunk_index),
                "audio_available_s": (audio_available_s),
                "new_token_start_index": (previous_committed_count),
                "new_token_count": len(new_tokens),
                "new_text": tokens_to_text(new_tokens),
                "committed_token_count": (len(committed_tokens)),
                "committed_text": (tokens_to_text(committed_tokens)),
                "partial_text": (partial_text),
            }
        )

    # ------------------------------------------------------------------
    # State before end-of-utterance flush
    # ------------------------------------------------------------------

    stable_committed_tokens = list(committed_tokens)

    stable_committed_token_count = len(stable_committed_tokens)

    final_token_count = len(final_tokens)

    if final_token_count:
        stable_coverage = stable_committed_token_count / final_token_count
    else:
        stable_coverage = 1.0

    # This is the main irreversible-error check.
    #
    # If committed_tokens are not a prefix of the final
    # prediction, the policy committed something that the
    # ASR later rejected.
    final_prefix_safe = starts_with(
        final_tokens,
        committed_tokens,
    )

    final_flush_tokens = []

    last_audio_s = float(traces[-1]["audio_available_s"])

    # At end-of-utterance the final NeMo hypothesis is known,
    # so the remaining suffix can safely be flushed if all
    # previous commitments were valid.
    if final_prefix_safe:
        final_flush_tokens = final_tokens[len(committed_tokens) :]

        if final_flush_tokens:
            previous_committed_count = len(committed_tokens)

            committed_tokens.extend(final_flush_tokens)

            committed_token_times.extend([last_audio_s for _ in final_flush_tokens])

            commit_events.append(
                {
                    "policy": policy_name,
                    "event_type": ("final_flush"),
                    "utterance_index": int(utterance["utterance_index"]),
                    "language": utterance["language"],
                    "sample_id": utterance["sample_id"],
                    "chunk_index": int(traces[-1]["chunk_index"]),
                    "audio_available_s": (last_audio_s),
                    "new_token_start_index": (previous_committed_count),
                    "new_token_count": len(final_flush_tokens),
                    "new_text": (tokens_to_text(final_flush_tokens)),
                    "committed_token_count": (len(committed_tokens)),
                    "committed_text": (tokens_to_text(committed_tokens)),
                    "partial_text": (normalize_spacing(traces[-1]["partial_text"])),
                }
            )

    # ------------------------------------------------------------------
    # Delay metrics
    # ------------------------------------------------------------------

    stable_token_delays = []

    all_token_delays = []

    for token_index, commit_time in enumerate(committed_token_times):
        if token_index >= len(first_appearance_times):
            break

        appearance_time = first_appearance_times[token_index]

        if appearance_time is None:
            continue

        delay = commit_time - appearance_time

        all_token_delays.append(delay)

        if token_index < stable_committed_token_count:
            stable_token_delays.append(delay)

    if first_stable_commit_s is not None and first_non_empty_s is not None:
        first_commit_delay_s = first_stable_commit_s - float(first_non_empty_s)
    else:
        first_commit_delay_s = None

    result = {
        "policy": policy_name,
        "stability_window": (stability_window),
        "holdback_tokens": (holdback_tokens),
        "utterance_index": int(utterance["utterance_index"]),
        "language": utterance["language"],
        "sample_id": utterance["sample_id"],
        "duration_s": float(utterance["duration_s"]),
        "partial_count": len(traces),
        "final_prediction": (final_prediction),
        "final_token_count": (final_token_count),
        "first_non_empty_audio_s": (first_non_empty_s),
        "first_stable_commit_audio_s": (first_stable_commit_s),
        "first_commit_delay_s": (first_commit_delay_s),
        "stable_commit_events": (stable_commit_events),
        "stable_committed_token_count": (stable_committed_token_count),
        "stable_coverage": (stable_coverage),
        "stable_coverage_percent": (stable_coverage * 100.0),
        "stable_committed_text": (tokens_to_text(stable_committed_tokens)),
        "final_flush_token_count": len(final_flush_tokens),
        "final_prefix_safe": (final_prefix_safe),
        "transient_conflict_partials": (transient_conflict_partials),
        "has_transient_conflict": (transient_conflict_partials > 0),
        "first_transient_conflict_s": (first_transient_conflict_s),
        "mean_stable_token_delay_s": (safe_mean(stable_token_delays)),
        "median_stable_token_delay_s": (safe_median(stable_token_delays)),
        "mean_all_token_delay_s": (safe_mean(all_token_delays)),
        "median_all_token_delay_s": (safe_median(all_token_delays)),
    }

    return (
        result,
        commit_events,
    )


# ============================================================================
# Summary
# ============================================================================


def build_policy_summary(
    policy_name: str,
    policy: dict,
    utterance_results: list[dict],
) -> dict:
    first_commit_times = [
        float(result["first_stable_commit_audio_s"])
        for result in utterance_results
        if (result["first_stable_commit_audio_s"] is not None)
    ]

    first_commit_delays = [
        float(result["first_commit_delay_s"])
        for result in utterance_results
        if (result["first_commit_delay_s"] is not None)
    ]

    stable_token_delays = [
        float(result["mean_stable_token_delay_s"])
        for result in utterance_results
        if (result["mean_stable_token_delay_s"] is not None)
    ]

    coverage_values = [float(result["stable_coverage_percent"]) for result in utterance_results]

    commit_event_counts = [int(result["stable_commit_events"]) for result in utterance_results]

    final_flush_counts = [int(result["final_flush_token_count"]) for result in utterance_results]

    final_safe_count = sum(1 for result in utterance_results if result["final_prefix_safe"])

    transient_conflict_utterances = sum(1 for result in utterance_results if result["has_transient_conflict"])

    transient_conflict_partials = sum(int(result["transient_conflict_partials"]) for result in utterance_results)

    language_summary = {}

    languages = sorted({result["language"] for result in utterance_results})

    for language in languages:
        language_results = [result for result in utterance_results if (result["language"] == language)]

        language_delays = [
            float(result["first_commit_delay_s"])
            for result in language_results
            if (result["first_commit_delay_s"] is not None)
        ]

        language_summary[language] = {
            "utterances": len(language_results),
            "final_safe_utterances": sum(1 for result in language_results if result["final_prefix_safe"]),
            "transient_conflict_utterances": sum(1 for result in language_results if result["has_transient_conflict"]),
            "mean_first_commit_delay_s": (round_or_none(safe_mean(language_delays))),
            "mean_stable_coverage_percent": (
                round_or_none(safe_mean([float(result["stable_coverage_percent"]) for result in language_results]))
            ),
        }

    return {
        "policy": policy_name,
        "stability_window": int(policy["stability_window"]),
        "holdback_tokens": int(policy["holdback_tokens"]),
        "utterances": len(utterance_results),
        "utterances_with_stable_commit": (len(first_commit_times)),
        "final_safe_utterances": (final_safe_count),
        "final_commit_errors": (len(utterance_results) - final_safe_count),
        "transient_conflict_utterances": (transient_conflict_utterances),
        "transient_conflict_partials": (transient_conflict_partials),
        "mean_first_commit_audio_s": (round_or_none(safe_mean(first_commit_times))),
        "median_first_commit_audio_s": (round_or_none(safe_median(first_commit_times))),
        "mean_first_commit_delay_s": (round_or_none(safe_mean(first_commit_delays))),
        "median_first_commit_delay_s": (round_or_none(safe_median(first_commit_delays))),
        "mean_stable_token_delay_s": (round_or_none(safe_mean(stable_token_delays))),
        "median_stable_token_delay_s": (round_or_none(safe_median(stable_token_delays))),
        "mean_stable_coverage_percent": (round_or_none(safe_mean(coverage_values))),
        "median_stable_coverage_percent": (round_or_none(safe_median(coverage_values))),
        "mean_stable_commit_events": (round_or_none(safe_mean([float(value) for value in commit_event_counts]))),
        "mean_final_flush_tokens": (round_or_none(safe_mean([float(value) for value in final_flush_counts]))),
        "languages": (language_summary),
    }


# ============================================================================
# Output
# ============================================================================


def write_summary_csv(
    summaries: list[dict],
) -> None:
    fieldnames = [
        "policy",
        "stability_window",
        "holdback_tokens",
        "utterances",
        "utterances_with_stable_commit",
        "final_safe_utterances",
        "final_commit_errors",
        "transient_conflict_utterances",
        "transient_conflict_partials",
        "mean_first_commit_audio_s",
        "median_first_commit_audio_s",
        "mean_first_commit_delay_s",
        "median_first_commit_delay_s",
        "mean_stable_token_delay_s",
        "median_stable_token_delay_s",
        "mean_stable_coverage_percent",
        "median_stable_coverage_percent",
        "mean_stable_commit_events",
        "mean_final_flush_tokens",
    ]

    with SUMMARY_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fieldnames})


def print_summary(
    summaries: list[dict],
) -> None:
    print()
    print("=" * 100)
    print("PARAKEET PARTIAL STABILITY ANALYSIS")
    print("=" * 100)

    print(
        f"{'Policy':<20}{'Safe':>9}{'Conflict':>12}{'1st commit':>13}{'+delay':>11}{'Token delay':>13}{'Coverage':>11}"
    )

    print("-" * 100)

    for summary in summaries:
        first_commit = summary["mean_first_commit_audio_s"]

        first_delay = summary["mean_first_commit_delay_s"]

        token_delay = summary["mean_stable_token_delay_s"]

        coverage = summary["mean_stable_coverage_percent"]

        first_commit_text = "-" if first_commit is None else f"{first_commit:.2f}s"

        first_delay_text = "-" if first_delay is None else f"{first_delay:.2f}s"

        token_delay_text = "-" if token_delay is None else f"{token_delay:.2f}s"

        coverage_text = "-" if coverage is None else f"{coverage:.1f}%"

        safe_text = f"{summary['final_safe_utterances']}/{summary['utterances']}"

        conflict_text = f"{summary['transient_conflict_utterances']}/{summary['utterances']}"

        print(
            f"{summary['policy']:<20}"
            f"{safe_text:>9}"
            f"{conflict_text:>12}"
            f"{first_commit_text:>13}"
            f"{first_delay_text:>11}"
            f"{token_delay_text:>13}"
            f"{coverage_text:>11}"
        )

    print("=" * 100)

    print()
    print("Safe       = committed tokens remain a prefix of the final ASR hypothesis.")

    print("Conflict   = at least one later partial temporarily disagreed with already committed text.")

    print("1st commit = mean audio time of the first stable commit.")

    print("+delay     = mean delay from first non-empty ASR text to first stable commit.")

    print("Token delay= mean waiting time between a token first appearing correctly and being committed.")

    print("Coverage   = percentage of final tokens committed before the final end-of-utterance flush.")

    print("=" * 100)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print("=" * 80)
    print("PARAKEET-TDT PARTIAL STABILITY ANALYZER")
    print("=" * 80)

    validate_inputs()

    traces = read_jsonl(PARTIAL_TRACES_PATH)

    utterances = read_jsonl(UTTERANCES_PATH)

    metadata = read_json(METADATA_PATH)

    grouped_traces = validate_dataset(
        traces=traces,
        utterances=utterances,
        metadata=metadata,
    )

    print(f"Utterances      : {len(utterances)}")

    print(f"Partial records : {len(traces)}")

    effective = metadata["effective_streaming_config"]

    print(
        "Streaming config: "
        f"left="
        f"{effective['left_context_secs']:.2f}s, "
        f"chunk="
        f"{effective['chunk_secs']:.2f}s, "
        f"right="
        f"{effective['right_context_secs']:.2f}s"
    )

    print(f"Latency         : {effective['theoretical_latency_secs']:.2f}s")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_utterance_results = []

    all_commit_events = []

    policy_summaries = []

    for policy_name, policy in POLICIES.items():
        policy_results = []

        for utterance in utterances:
            utterance_index = int(utterance["utterance_index"])

            sample_traces = grouped_traces[utterance_index]

            result, events = evaluate_policy_on_utterance(
                policy_name=(policy_name),
                policy=policy,
                utterance=utterance,
                traces=sample_traces,
            )

            policy_results.append(result)

            all_utterance_results.append(result)

            all_commit_events.extend(events)

        summary = build_policy_summary(
            policy_name=(policy_name),
            policy=policy,
            utterance_results=(policy_results),
        )

        policy_summaries.append(summary)

    write_jsonl(
        UTTERANCE_ANALYSIS_PATH,
        all_utterance_results,
    )

    write_jsonl(
        COMMIT_EVENTS_PATH,
        all_commit_events,
    )

    output_summary = {
        "experiment": ("parakeet_partial_stability_analysis"),
        "input": {
            "partial_traces": str(PARTIAL_TRACES_PATH),
            "utterances": str(UTTERANCES_PATH),
            "metadata": str(METADATA_PATH),
            "utterance_count": len(utterances),
            "partial_count": len(traces),
        },
        "streaming_config": (effective),
        "policy_definition": {
            "stable_prefix_unit": ("whitespace-delimited tokens"),
            "holdback_reason": (
                "The collector strips trailing "
                "whitespace, so the final token of "
                "the stable prefix is retained to "
                "avoid committing incomplete words."
            ),
            "retractions_allowed": False,
            "end_of_utterance_flush": True,
        },
        "policies": (policy_summaries),
    }

    write_json(
        SUMMARY_JSON_PATH,
        output_summary,
    )

    write_summary_csv(policy_summaries)

    print_summary(policy_summaries)

    print()
    print(f"Summary JSON       : {SUMMARY_JSON_PATH}")

    print(f"Summary CSV        : {SUMMARY_CSV_PATH}")

    print(f"Utterance analysis : {UTTERANCE_ANALYSIS_PATH}")

    print(f"Commit events      : {COMMIT_EVENTS_PATH}")

    print("=" * 80)


if __name__ == "__main__":
    main()
