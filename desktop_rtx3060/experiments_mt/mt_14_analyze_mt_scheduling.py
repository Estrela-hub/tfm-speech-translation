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

SCHEDULERS = [
    "every_commit_fifo",
    "latest_only",
    "final_only",
]

EPSILON_MS = 1e-9


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "mt_selection" / "incremental_translation"

TARGET_DRAFTS_PATH = INPUT_DIR / "target_drafts.jsonl"

MT10_SUMMARY_PATH = INPUT_DIR / "incremental_summary.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "mt_scheduling"

SCHEDULE_EVENTS_PATH = RESULTS_DIR / "schedule_events.jsonl"

SEQUENCE_RESULTS_PATH = RESULTS_DIR / "scheduling_by_sequence.csv"

SCHEDULING_SUMMARY_PATH = RESULTS_DIR / "scheduling_summary.csv"

SUMMARY_PATH = RESULTS_DIR / "mt_scheduling_summary.json"


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
        raise RuntimeError("mt_10 was not generated using FP16.")

    scope = mt10_summary.get(
        "scope",
        {},
    )

    if scope.get("utterance_target_sequences") != EXPECTED_SEQUENCES:
        raise RuntimeError("Unexpected sequence count in mt_10.")

    if scope.get("mt_invocations") != EXPECTED_TARGET_DRAFTS:
        raise RuntimeError("Unexpected MT invocation count in mt_10.")

    drafts = load_jsonl(TARGET_DRAFTS_PATH)

    if len(drafts) != EXPECTED_TARGET_DRAFTS:
        raise RuntimeError(
            f"Unexpected target draft count.\nExpected: {EXPECTED_TARGET_DRAFTS}\nActual  : {len(drafts)}"
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
    }

    for index, draft in enumerate(
        drafts,
        start=1,
    ):
        missing = required_fields - set(draft.keys())

        if missing:
            raise RuntimeError(f"Target draft {index} is missing fields: {sorted(missing)}")

        if float(draft["total_ms"]) <= 0.0:
            raise RuntimeError(f"Invalid MT latency in draft {index}.")

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
            raise RuntimeError(f"Expected one final flush for {key}.")

        if not sequence[-1]["is_final_flush"]:
            raise RuntimeError(f"Final flush is not last for {key}.")

        arrival_times = [float(row["audio_available_s"]) for row in sequence]

        if arrival_times != sorted(arrival_times):
            raise RuntimeError(f"Non-monotonic source commit times for {key}.")

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
            raise RuntimeError(f"Direction {direction} has {count} sequences.")

    return dict(grouped)


# =============================================================================
# Event helpers
# =============================================================================


def build_event(
    scheduler: str,
    sequence_key: tuple[
        int,
        str,
        str,
    ],
    draft: dict,
    status: str,
    start_ms: float | None,
    finish_ms: float | None,
    queue_delay_ms: float | None,
    stale_at_finish: bool | None,
) -> dict:
    (
        utterance_index,
        source_language,
        target_language,
    ) = sequence_key

    arrival_ms = float(draft["audio_available_s"]) * 1000.0

    return {
        "scheduler": (scheduler),
        "utterance_index": (utterance_index),
        "dataset_index": int(draft["dataset_index"]),
        "sample_id": (draft["sample_id"]),
        "source_language": (source_language),
        "target_language": (target_language),
        "commit_index": int(draft["commit_index"]),
        "is_final_flush": bool(draft["is_final_flush"]),
        "arrival_s": (arrival_ms / 1000.0),
        "service_ms": float(draft["total_ms"]),
        "status": (status),
        "start_s": (start_ms / 1000.0 if start_ms is not None else None),
        "finish_s": (finish_ms / 1000.0 if finish_ms is not None else None),
        "queue_delay_ms": (queue_delay_ms),
        "stale_at_finish": (stale_at_finish),
        "source_committed_text": (draft["source_committed_text"]),
        "target_text": (draft["target_text"]),
    }


# =============================================================================
# FIFO scheduler
# =============================================================================


def simulate_every_commit_fifo(
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
    events = []

    server_free_ms = 0.0

    completed_before_next = 0
    comparable_jobs = 0

    queue_delays = []

    service_times = []

    stale_completed_calls = 0

    first_finish_ms = None
    final_finish_ms = None

    for index, draft in enumerate(drafts):
        arrival_ms = float(draft["audio_available_s"]) * 1000.0

        service_ms = float(draft["total_ms"])

        start_ms = max(
            arrival_ms,
            server_free_ms,
        )

        queue_delay_ms = start_ms - arrival_ms

        finish_ms = start_ms + service_ms

        if first_finish_ms is None:
            first_finish_ms = finish_ms

        if index + 1 < len(drafts):
            next_arrival_ms = float(drafts[index + 1]["audio_available_s"]) * 1000.0

            stale_at_finish = finish_ms > next_arrival_ms

            comparable_jobs += 1

            if not stale_at_finish:
                completed_before_next += 1

            else:
                stale_completed_calls += 1

        else:
            stale_at_finish = False
            final_finish_ms = finish_ms

        events.append(
            build_event(
                scheduler=("every_commit_fifo"),
                sequence_key=(sequence_key),
                draft=draft,
                status="processed",
                start_ms=start_ms,
                finish_ms=finish_ms,
                queue_delay_ms=(queue_delay_ms),
                stale_at_finish=(stale_at_finish),
            )
        )

        queue_delays.append(queue_delay_ms)

        service_times.append(service_ms)

        server_free_ms = finish_ms

    final_source_commit_ms = float(drafts[-1]["audio_available_s"]) * 1000.0

    if first_finish_ms is None or final_finish_ms is None:
        raise RuntimeError("FIFO scheduler failed.")

    summary = {
        "processed_calls": len(drafts),
        "skipped_calls": 0,
        "stale_completed_calls": (stale_completed_calls),
        "total_compute_ms": sum(service_times),
        "first_target_completion_s": (first_finish_ms / 1000.0),
        "final_target_completion_s": (final_finish_ms / 1000.0),
        "final_completion_after_audio_end_ms": (final_finish_ms - final_source_commit_ms),
        "mean_queue_delay_ms": (statistics.mean(queue_delays)),
        "max_queue_delay_ms": max(queue_delays),
        "completed_before_next_commit": (completed_before_next),
        "comparable_jobs": (comparable_jobs),
        "completed_before_next_commit_percent": (
            completed_before_next / comparable_jobs * 100.0 if comparable_jobs else 100.0
        ),
        "final_translation_preserved": (True),
    }

    return (
        summary,
        events,
    )


# =============================================================================
# Latest-only scheduler
# =============================================================================


def simulate_latest_only(
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
    """
    Single-server latest-only scheduling.

    Behaviour:

    - If MT is idle, process the newest available source commit.
    - While MT is busy, new source commits may arrive.
    - When MT becomes free, discard all obsolete pending commits and
      process only the newest one.
    - A translation already running is not cancelled.
    - The final source commit is always processed.
    """

    count = len(drafts)

    events_by_index: dict[
        int,
        dict,
    ] = {}

    arrivals_ms = [float(draft["audio_available_s"]) * 1000.0 for draft in drafts]

    services_ms = [float(draft["total_ms"]) for draft in drafts]

    pending: list[int] = []

    next_index = 0

    current_time_ms = 0.0

    processed_indices = []

    queue_delays = []

    service_times = []

    stale_completed_calls = 0

    first_finish_ms = None
    final_finish_ms = None

    while next_index < count or pending:
        # --------------------------------------------------------------
        # If there is no pending work, advance to the next source commit.
        # --------------------------------------------------------------

        if not pending:
            if next_index >= count:
                break

            current_time_ms = max(
                current_time_ms,
                arrivals_ms[next_index],
            )

            while next_index < count and arrivals_ms[next_index] <= (current_time_ms + EPSILON_MS):
                pending.append(next_index)

                next_index += 1

        # --------------------------------------------------------------
        # Only the newest currently pending source prefix matters.
        # Older pending prefixes have already become obsolete.
        # --------------------------------------------------------------

        selected_index = pending[-1]

        obsolete_indices = pending[:-1]

        for obsolete_index in obsolete_indices:
            events_by_index[obsolete_index] = build_event(
                scheduler="latest_only",
                sequence_key=(sequence_key),
                draft=drafts[obsolete_index],
                status="skipped_obsolete",
                start_ms=None,
                finish_ms=None,
                queue_delay_ms=None,
                stale_at_finish=None,
            )

        pending = []

        selected_arrival_ms = arrivals_ms[selected_index]

        service_ms = services_ms[selected_index]

        start_ms = max(
            current_time_ms,
            selected_arrival_ms,
        )

        queue_delay_ms = start_ms - selected_arrival_ms

        finish_ms = start_ms + service_ms

        # --------------------------------------------------------------
        # Collect source commits arriving while this MT call is running.
        # --------------------------------------------------------------

        arrivals_during_inference = []

        while next_index < count and arrivals_ms[next_index] <= (finish_ms + EPSILON_MS):
            arrivals_during_inference.append(next_index)

            pending.append(next_index)

            next_index += 1

        stale_at_finish = len(arrivals_during_inference) > 0

        if stale_at_finish:
            stale_completed_calls += 1

        events_by_index[selected_index] = build_event(
            scheduler="latest_only",
            sequence_key=(sequence_key),
            draft=drafts[selected_index],
            status="processed",
            start_ms=start_ms,
            finish_ms=finish_ms,
            queue_delay_ms=(queue_delay_ms),
            stale_at_finish=(stale_at_finish),
        )

        processed_indices.append(selected_index)

        queue_delays.append(queue_delay_ms)

        service_times.append(service_ms)

        if first_finish_ms is None:
            first_finish_ms = finish_ms

        if selected_index == count - 1:
            final_finish_ms = finish_ms

        current_time_ms = finish_ms

    # ------------------------------------------------------------------
    # Any draft without an event is invalid.
    # ------------------------------------------------------------------

    if len(events_by_index) != count:
        missing = [index for index in range(count) if index not in events_by_index]

        raise RuntimeError(f"latest_only did not classify all source commits: {missing}")

    if count - 1 not in processed_indices:
        raise RuntimeError("latest_only failed to process the final source commit.")

    if first_finish_ms is None or final_finish_ms is None:
        raise RuntimeError("latest_only did not produce valid completion times.")

    events = [events_by_index[index] for index in range(count)]

    final_source_commit_ms = arrivals_ms[-1]

    processed_calls = len(processed_indices)

    skipped_calls = count - processed_calls

    summary = {
        "processed_calls": (processed_calls),
        "skipped_calls": (skipped_calls),
        "stale_completed_calls": (stale_completed_calls),
        "total_compute_ms": sum(service_times),
        "first_target_completion_s": (first_finish_ms / 1000.0),
        "final_target_completion_s": (final_finish_ms / 1000.0),
        "final_completion_after_final_source_commit_ms": (final_finish_ms - final_source_commit_ms),
        "mean_queue_delay_ms": (statistics.mean(queue_delays) if queue_delays else 0.0),
        "max_queue_delay_ms": (max(queue_delays) if queue_delays else 0.0),
        "completed_before_next_commit": (None),
        "comparable_jobs": (None),
        "completed_before_next_commit_percent": (None),
        "final_translation_preserved": (True),
    }

    return (
        summary,
        events,
    )


# =============================================================================
# Final-only scheduler
# =============================================================================


def simulate_final_only(
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
    events = []

    final_index = len(drafts) - 1

    for index, draft in enumerate(drafts):
        if index != final_index:
            events.append(
                build_event(
                    scheduler=("final_only"),
                    sequence_key=(sequence_key),
                    draft=draft,
                    status=("skipped_non_final"),
                    start_ms=None,
                    finish_ms=None,
                    queue_delay_ms=None,
                    stale_at_finish=None,
                )
            )

            continue

        arrival_ms = float(draft["audio_available_s"]) * 1000.0

        service_ms = float(draft["total_ms"])

        start_ms = arrival_ms

        finish_ms = start_ms + service_ms

        events.append(
            build_event(
                scheduler=("final_only"),
                sequence_key=(sequence_key),
                draft=draft,
                status="processed",
                start_ms=start_ms,
                finish_ms=finish_ms,
                queue_delay_ms=0.0,
                stale_at_finish=False,
            )
        )

    final_draft = drafts[-1]

    final_arrival_ms = float(final_draft["audio_available_s"]) * 1000.0

    final_service_ms = float(final_draft["total_ms"])

    final_finish_ms = final_arrival_ms + final_service_ms

    summary = {
        "processed_calls": 1,
        "skipped_calls": (len(drafts) - 1),
        "stale_completed_calls": 0,
        "total_compute_ms": (final_service_ms),
        "first_target_completion_s": (final_finish_ms / 1000.0),
        "final_target_completion_s": (final_finish_ms / 1000.0),
        "final_completion_after_final_source_commit_ms": (final_service_ms),
        "mean_queue_delay_ms": 0.0,
        "max_queue_delay_ms": 0.0,
        "completed_before_next_commit": (None),
        "comparable_jobs": (None),
        "completed_before_next_commit_percent": (None),
        "final_translation_preserved": (True),
    }

    return (
        summary,
        events,
    )


# =============================================================================
# Sequence result construction
# =============================================================================


def build_sequence_result(
    scheduler: str,
    sequence_key: tuple[
        int,
        str,
        str,
    ],
    drafts: list[dict],
    schedule_summary: dict,
) -> dict:
    (
        utterance_index,
        source_language,
        target_language,
    ) = sequence_key

    final_draft = drafts[-1]

    original_call_count = len(drafts)

    processed_calls = int(schedule_summary["processed_calls"])

    skipped_calls = int(schedule_summary["skipped_calls"])

    call_reduction_percent = (original_call_count - processed_calls) / original_call_count * 100.0

    audio_duration_s = float(final_draft["audio_available_s"])

    compute_to_audio_ratio = float(schedule_summary["total_compute_ms"]) / (audio_duration_s * 1000.0)

    return {
        "scheduler": (scheduler),
        "utterance_index": (utterance_index),
        "dataset_index": int(final_draft["dataset_index"]),
        "sample_id": (final_draft["sample_id"]),
        "source_language": (source_language),
        "target_language": (target_language),
        "audio_duration_s": (audio_duration_s),
        "original_commit_count": (original_call_count),
        "processed_calls": (processed_calls),
        "skipped_calls": (skipped_calls),
        "call_reduction_percent": (call_reduction_percent),
        "stale_completed_calls": int(schedule_summary["stale_completed_calls"]),
        "total_compute_ms": float(schedule_summary["total_compute_ms"]),
        "compute_to_audio_ratio": (compute_to_audio_ratio),
        "first_target_completion_s": float(schedule_summary["first_target_completion_s"]),
        "final_target_completion_s": float(schedule_summary["final_target_completion_s"]),
        "final_completion_after_final_source_commit_ms": float(
            schedule_summary["final_completion_after_final_source_commit_ms"]
        ),
        "mean_queue_delay_ms": float(schedule_summary["mean_queue_delay_ms"]),
        "max_queue_delay_ms": float(schedule_summary["max_queue_delay_ms"]),
        "completed_before_next_commit_percent": (schedule_summary["completed_before_next_commit_percent"]),
        "final_translation_preserved": bool(schedule_summary["final_translation_preserved"]),
    }


# =============================================================================
# Evaluate schedulers
# =============================================================================


def evaluate_schedulers(
    sequences: dict[
        tuple[int, str, str],
        list[dict],
    ],
) -> tuple[
    list[dict],
    list[dict],
]:
    sequence_results = []

    all_events = []

    for scheduler in SCHEDULERS:
        print()

        print("=" * 80)

        print(f"SCHEDULER: {scheduler}")

        print("=" * 80)

        scheduler_results = []

        for sequence_key in sorted(
            sequences,
            key=lambda key: (
                key[1],
                key[2],
                key[0],
            ),
        ):
            drafts = sequences[sequence_key]

            if scheduler == "every_commit_fifo":
                (
                    schedule_summary,
                    events,
                ) = simulate_every_commit_fifo(
                    sequence_key=(sequence_key),
                    drafts=drafts,
                )

            elif scheduler == "latest_only":
                (
                    schedule_summary,
                    events,
                ) = simulate_latest_only(
                    sequence_key=(sequence_key),
                    drafts=drafts,
                )

            elif scheduler == "final_only":
                (
                    schedule_summary,
                    events,
                ) = simulate_final_only(
                    sequence_key=(sequence_key),
                    drafts=drafts,
                )

            else:
                raise RuntimeError(f"Unknown scheduler: {scheduler}")

            result = build_sequence_result(
                scheduler=(scheduler),
                sequence_key=(sequence_key),
                drafts=drafts,
                schedule_summary=(schedule_summary),
            )

            scheduler_results.append(result)

            sequence_results.append(result)

            all_events.extend(events)

        total_calls = sum(row["processed_calls"] for row in scheduler_results)

        total_skipped = sum(row["skipped_calls"] for row in scheduler_results)

        mean_final_delay = statistics.mean(
            row["final_completion_after_final_source_commit_ms"] for row in scheduler_results
        )

        mean_first_completion = statistics.mean(row["first_target_completion_s"] for row in scheduler_results)

        print(f"Processed MT calls          : {total_calls}")

        print(f"Skipped MT calls            : {total_skipped}")

        print(f"Mean first target complete  : {mean_first_completion:.2f} s")

        print(f"Mean final MT after final source commit: {mean_final_delay:.2f} ms")

    return (
        sequence_results,
        all_events,
    )


# =============================================================================
# Global summaries
# =============================================================================


def summarize_schedulers(
    sequence_results: list[dict],
) -> list[dict]:
    baseline_rows = [row for row in sequence_results if row["scheduler"] == "every_commit_fifo"]

    baseline_total_calls = sum(int(row["processed_calls"]) for row in baseline_rows)

    summaries = []

    for scheduler in SCHEDULERS:
        rows = [row for row in sequence_results if row["scheduler"] == scheduler]

        if len(rows) != EXPECTED_SEQUENCES:
            raise RuntimeError(f"Scheduler {scheduler} has {len(rows)} rows.")

        total_calls = sum(int(row["processed_calls"]) for row in rows)

        total_skipped = sum(int(row["skipped_calls"]) for row in rows)

        stale_completed_calls = sum(int(row["stale_completed_calls"]) for row in rows)

        total_compute_ms = sum(float(row["total_compute_ms"]) for row in rows)

        total_audio_ms = sum(float(row["audio_duration_s"]) * 1000.0 for row in rows)

        first_target_times = [float(row["first_target_completion_s"]) for row in rows]

        final_delays = [float(row["final_completion_after_final_source_commit_ms"]) for row in rows]

        mean_queue_delays = [float(row["mean_queue_delay_ms"]) for row in rows]

        max_queue_delays = [float(row["max_queue_delay_ms"]) for row in rows]

        preserved_count = sum(1 for row in rows if row["final_translation_preserved"])

        if scheduler == "every_commit_fifo":
            comparable_jobs = sum(int(row["original_commit_count"]) - 1 for row in rows)

            completed_jobs = 0

            for row in rows:
                percent = row["completed_before_next_commit_percent"]

                sequence_comparable = int(row["original_commit_count"]) - 1

                if percent is not None:
                    completed_jobs += float(percent) / 100.0 * sequence_comparable

            completed_before_next_percent = completed_jobs / comparable_jobs * 100.0 if comparable_jobs else 100.0

        else:
            completed_before_next_percent = None

        summaries.append(
            {
                "scheduler": (scheduler),
                "sequences": (len(rows)),
                "processed_calls": (total_calls),
                "skipped_calls": (total_skipped),
                "call_reduction_vs_fifo_percent": ((baseline_total_calls - total_calls) / baseline_total_calls * 100.0),
                "stale_completed_calls": (stale_completed_calls),
                "stale_completed_call_percent": (stale_completed_calls / total_calls * 100.0 if total_calls else 0.0),
                "total_compute_ms": (total_compute_ms),
                "compute_to_effective_audio_ratio": (total_compute_ms / total_audio_ms),
                "compute_reduction_vs_fifo_percent": (0.0),
                "mean_first_target_completion_s": (statistics.mean(first_target_times)),
                "median_first_target_completion_s": (statistics.median(first_target_times)),
                "mean_final_completion_after_final_source_commit_ms": (statistics.mean(final_delays)),
                "median_final_completion_after_final_source_commit_ms": (statistics.median(final_delays)),
                "p95_final_completion_after_final_source_commit_ms": (
                    percentile(
                        final_delays,
                        95,
                    )
                ),
                "mean_queue_delay_ms": (statistics.mean(mean_queue_delays)),
                "mean_max_queue_delay_ms": (statistics.mean(max_queue_delays)),
                "max_queue_delay_ms": (max(max_queue_delays)),
                "completed_before_next_commit_percent": (completed_before_next_percent),
                "final_translation_preserved_sequences": (preserved_count),
                "final_translation_preserved_percent": (preserved_count / EXPECTED_SEQUENCES * 100.0),
            }
        )

    fifo_compute = next(row["total_compute_ms"] for row in summaries if row["scheduler"] == "every_commit_fifo")

    for row in summaries:
        row["compute_reduction_vs_fifo_percent"] = (fifo_compute - row["total_compute_ms"]) / fifo_compute * 100.0

    return summaries


# =============================================================================
# Validate mt_10 baseline reproduction
# =============================================================================


def validate_fifo_against_mt10(
    summaries: list[dict],
    mt10_summary: dict,
) -> None:
    fifo = next(row for row in summaries if row["scheduler"] == "every_commit_fifo")

    mt10_performance = mt10_summary["performance"]

    comparisons = [
        (
            "processed_calls",
            float(fifo["processed_calls"]),
            float(mt10_summary["scope"]["mt_invocations"]),
            0.0,
        ),
        (
            "compute_to_effective_audio_ratio",
            float(fifo["compute_to_effective_audio_ratio"]),
            float(mt10_performance["compute_to_effective_audio_ratio"]),
            0.001,
        ),
        (
            "completed_before_next_commit_percent",
            float(fifo["completed_before_next_commit_percent"]),
            float(mt10_performance["completed_before_next_commit_percent"]),
            0.05,
        ),
        (
            "mean_max_queue_delay_ms",
            float(fifo["mean_max_queue_delay_ms"]),
            float(mt10_performance["mean_max_queue_delay_ms"]),
            0.05,
        ),
        (
            "max_queue_delay_ms",
            float(fifo["max_queue_delay_ms"]),
            float(mt10_performance["max_queue_delay_ms"]),
            0.05,
        ),
        (
            "mean_first_target_completion_s",
            float(fifo["mean_first_target_completion_s"]),
            float(mt10_performance["mean_first_target_completion_s"]),
            0.01,
        ),
    ]

    for (
        name,
        calculated,
        expected,
        tolerance,
    ) in comparisons:
        difference = abs(calculated - expected)

        if difference > tolerance:
            raise RuntimeError(
                "FIFO scheduling does not "
                "reproduce mt_10.\n"
                f"Metric     : {name}\n"
                f"Calculated : "
                f"{calculated}\n"
                f"mt_10      : "
                f"{expected}\n"
                f"Difference : "
                f"{difference}"
            )

    print()
    print("every_commit_fifo reproduces mt_10: PASS")


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


def write_events(
    events: list[dict],
) -> None:
    with SCHEDULE_EVENTS_PATH.open(
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
# Console report
# =============================================================================


def print_summary(
    summaries: list[dict],
) -> None:
    print()
    print("=" * 138)

    print("MT SCHEDULING SUMMARY")

    print("=" * 138)

    print(
        f"{'Scheduler':<20}"
        f"{'Calls':>8}"
        f"{'Call red.':>11}"
        f"{'Compute red.':>14}"
        f"{'Compute/audio':>15}"
        f"{'First target':>14}"
        f"{'Final delay':>14}"
        f"{'P95 final':>12}"
        f"{'Max queue':>12}"
        f"{'Final preserved':>17}"
    )

    print("-" * 138)

    for row in summaries:
        print(
            f"{row['scheduler']:<20}"
            f"{row['processed_calls']:>8}"
            f"{row['call_reduction_vs_fifo_percent']:>10.1f}%"
            f"{row['compute_reduction_vs_fifo_percent']:>13.1f}%"
            f"{row['compute_to_effective_audio_ratio']:>15.3f}x"
            f"{row['mean_first_target_completion_s']:>13.2f}s"
            f"{row['mean_final_completion_after_final_source_commit_ms']:>13.1f}ms"
            f"{row['p95_final_completion_after_final_source_commit_ms']:>11.1f}ms"
            f"{row['max_queue_delay_ms']:>11.1f}ms"
            f"{row['final_translation_preserved_percent']:>16.1f}%"
        )

    print()

    latest = next(row for row in summaries if row["scheduler"] == "latest_only")

    final_only = next(row for row in summaries if row["scheduler"] == "final_only")

    print(
        "latest_only stale completed calls : "
        f"{latest['stale_completed_calls']} "
        f"("
        f"{latest['stale_completed_call_percent']:.1f}% "
        f"of processed calls)"
    )

    print(f"final_only mean MT service delay   : {final_only['mean_final_completion_after_audio_end_ms']:.1f} ms")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 80)

    print("MT SCHEDULING ANALYSIS")

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

    print()

    print("Schedulers:")

    for scheduler in SCHEDULERS:
        print(f"- {scheduler}")

    print()

    print("Evaluating scheduling policies...")

    (
        sequence_results,
        events,
    ) = evaluate_schedulers(sequences)

    summaries = summarize_schedulers(sequence_results)

    # ------------------------------------------------------------------
    # Reproduce mt_10 first.
    # ------------------------------------------------------------------

    validate_fifo_against_mt10(
        summaries=(summaries),
        mt10_summary=(mt10_summary),
    )

    # ------------------------------------------------------------------
    # Safety / quality-preservation gate.
    # ------------------------------------------------------------------

    for row in summaries:
        if row["final_translation_preserved_sequences"] != EXPECTED_SEQUENCES:
            raise RuntimeError(f"Scheduler {row['scheduler']} did not preserve all final translations.")

    print("Final translation preservation: PASS")

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_events(events)

    write_csv(
        SEQUENCE_RESULTS_PATH,
        sequence_results,
    )

    write_csv(
        SCHEDULING_SUMMARY_PATH,
        summaries,
    )

    summary = {
        "experiment": (
            "Offline single-server scheduling analysis over measured NLLB FP16 full-prefix translation latencies."
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
        "schedulers": {
            "every_commit_fifo": (
                "Translate every committed ASR prefix in arrival order. This reproduces the mt_10 naive baseline."
            ),
            "latest_only": (
                "If MT is busy, retain only "
                "the newest pending ASR prefix. "
                "Translations already running "
                "are allowed to complete."
            ),
            "final_only": ("Ignore all intermediate ASR commits and translate only the final complete source prefix."),
        },
        "quality_preservation": {
            "principle": (
                "Scheduling does not alter "
                "the NLLB model, decoding "
                "configuration, or source text "
                "of a selected translation."
            ),
            "final_translation": (
                "All schedulers process the "
                "same final full source prefix "
                "used by mt_10. Therefore the "
                "final NLLB translation is "
                "preserved exactly."
            ),
            "note": (
                "No claim is made that "
                "intermediate target behaviour "
                "is identical, because latest_only "
                "intentionally skips obsolete "
                "intermediate translations."
            ),
        },
        "methodology": {
            "arrival_time": ("audio_available_s from the stable2_hold1 ASR source commit."),
            "service_time": ("Measured total_ms of the corresponding NLLB FP16 full-prefix translation in mt_10."),
            "server": ("One non-preemptive MT worker per utterance-target sequence."),
            "latest_only_preemption": (
                "No running MT inference is cancelled. Only pending obsolete prefixes are discarded."
            ),
            "inference": ("No new NLLB inference is executed by mt_14."),
        },
        "results": (summaries),
        "environment": {
            "python": (platform.python_version()),
        },
        "limitations": [
            ("Service times are reused from mt_10 rather than remeasured in a live asynchronous scheduler."),
            (
                "Desktop RTX 3060 measurements "
                "remain development results and "
                "must not be treated as final "
                "Jetson Orin performance."
            ),
            ("The analysis models only the MT worker and does not yet include simultaneous ASR/TTS GPU contention."),
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
    # Console.
    # ------------------------------------------------------------------

    print_summary(summaries)

    print()

    print("=" * 80)

    print("OUTPUT")

    print("=" * 80)

    print(f"Schedule events : {SCHEDULE_EVENTS_PATH}")

    print(f"Per sequence    : {SEQUENCE_RESULTS_PATH}")

    print(f"Summary CSV     : {SCHEDULING_SUMMARY_PATH}")

    print(f"Summary JSON    : {SUMMARY_PATH}")

    print()

    print("=" * 80)

    print("MT SCHEDULING ANALYSIS PASSED")

    print("=" * 80)


if __name__ == "__main__":
    main()
