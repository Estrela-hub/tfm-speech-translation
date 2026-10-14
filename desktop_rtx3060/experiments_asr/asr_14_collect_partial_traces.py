from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

# ============================================================================
# Experiment configuration
# ============================================================================

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
MODEL_FILENAME = "parakeet-tdt-0.6b-v3.nemo"

NEMO_BASE_COMMIT = "fd6a877539710e2b98f28c43272ff81312f83417"
INSTRUMENTATION_BRANCH = "tfm-streaming-instrumentation"

COMPUTE_DTYPE = "float32"

LEFT_CONTEXT_SECS = 10.0
CHUNK_SECS = 0.16
RIGHT_CONTEXT_SECS = 1.84

EXPECTED_LANGUAGES = ("en", "es", "fr", "de", "pt")
EXPECTED_SAMPLES_PER_LANGUAGE = 5
EXPECTED_SAMPLE_COUNT = len(EXPECTED_LANGUAGES) * EXPECTED_SAMPLES_PER_LANGUAGE


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEMO_ROOT = PROJECT_ROOT / "external" / "NeMo-Speech"

NEMO_SCRIPT_RELATIVE = (
    Path("examples") / "asr" / "asr_chunked_inference" / "rnnt" / ("speech_to_text_streaming_infer_rnnt.py")
)

NEMO_SCRIPT = NEMO_ROOT / NEMO_SCRIPT_RELATIVE

EVALUATION_MANIFEST_PATH = (
    PROJECT_ROOT / "results" / "asr_selection" / "parakeet_streaming_configs" / "evaluation_manifest.jsonl"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "asr_selection" / "parakeet_partial_traces"

NEMO_OUTPUT_PATH = RESULTS_DIR / "final_predictions.jsonl"

NEMO_LOG_PATH = RESULTS_DIR / "nemo.log"

PARTIAL_TRACES_PATH = RESULTS_DIR / "partial_traces.jsonl"

UTTERANCES_PATH = RESULTS_DIR / "utterances.jsonl"

METADATA_PATH = RESULTS_DIR / "metadata.json"


# ============================================================================
# Log parsing
# ============================================================================

PARTIAL_PATTERN = re.compile(
    r"\[PARTIAL\s+(\d+)\]\s+"
    r"audio_available=([0-9]+(?:\.[0-9]+)?)s\s*"
    r"\|\s?(.*)$"
)

CORRECTED_CONTEXT_PATTERN = re.compile(
    r"Corrected contexts \(sec\): "
    r"Left ([0-9.]+), "
    r"Chunk ([0-9.]+), "
    r"Right ([0-9.]+)"
)

LATENCY_PATTERN = re.compile(r"Theoretical latency:\s*([0-9.]+)\s*seconds")

RTFX_PATTERN = re.compile(r"RTFx:\s*([0-9.]+)")


# ============================================================================
# Generic helpers
# ============================================================================


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def run_git(
    *arguments: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=NEMO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(f"Git command failed:\ngit {' '.join(arguments)}\n\n{result.stderr}")

    return result.stdout.strip()


# ============================================================================
# Validation
# ============================================================================


def validate_paths() -> None:
    if not NEMO_ROOT.exists():
        raise FileNotFoundError(f"NeMo repository was not found:\n{NEMO_ROOT}")

    if not NEMO_SCRIPT.exists():
        raise FileNotFoundError(f"NeMo streaming script was not found:\n{NEMO_SCRIPT}")

    if not EVALUATION_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Streaming evaluation manifest was not found:\n"
            f"{EVALUATION_MANIFEST_PATH}\n\n"
            "Run the streaming configuration evaluator first."
        )


def validate_instrumented_nemo() -> dict:
    branch = run_git(
        "branch",
        "--show-current",
    )

    if branch != INSTRUMENTATION_BRANCH:
        raise RuntimeError(
            "\nThe NeMo repository is not on the "
            "instrumentation branch.\n\n"
            f"Current branch : {branch or '<detached HEAD>'}\n"
            f"Required branch: {INSTRUMENTATION_BRANCH}\n\n"
            "Switch with:\n\n"
            "    cd "
            f"{NEMO_ROOT}\n"
            f"    git switch {INSTRUMENTATION_BRANCH}\n"
        )

    status = run_git(
        "status",
        "--porcelain",
    )

    if status:
        raise RuntimeError(
            "\nThe NeMo repository contains uncommitted "
            "changes.\n\n"
            "Commit or discard them before collecting "
            "partial traces.\n\n"
            f"{status}"
        )

    instrumentation_commit = run_git(
        "rev-parse",
        "HEAD",
    )

    ancestor_check = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            NEMO_BASE_COMMIT,
            instrumentation_commit,
        ],
        cwd=NEMO_ROOT,
    )

    if ancestor_check.returncode != 0:
        raise RuntimeError(
            "\nThe instrumentation branch is not based on "
            "the expected NeMo v3.0.0 commit.\n\n"
            f"Expected base: {NEMO_BASE_COMMIT}\n"
            f"Current HEAD : {instrumentation_commit}"
        )

    script_text = NEMO_SCRIPT.read_text(encoding="utf-8")

    if "[PARTIAL" not in script_text or "audio_available=" not in script_text:
        raise RuntimeError(f"\nThe expected partial-transcript instrumentation was not found in:\n{NEMO_SCRIPT}")

    diff_check = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            NEMO_BASE_COMMIT,
            "--",
            NEMO_SCRIPT_RELATIVE.as_posix(),
        ],
        cwd=NEMO_ROOT,
    )

    if diff_check.returncode == 0:
        raise RuntimeError(
            "\nThe streaming script is identical to the "
            "official NeMo base commit.\n"
            "The partial instrumentation is therefore "
            "not active."
        )

    if diff_check.returncode not in (
        0,
        1,
    ):
        raise RuntimeError("Could not compare the instrumented streaming script against the NeMo base commit.")

    return {
        "branch": branch,
        "commit": instrumentation_commit,
    }


def get_duration(record: dict) -> float:
    if "duration" in record:
        return float(record["duration"])

    if "duration_s" in record:
        return float(record["duration_s"])

    raise RuntimeError("Manifest record does not contain 'duration' or 'duration_s'.")


def validate_manifest(
    records: list[dict],
) -> None:
    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(
            "Unexpected number of samples in the "
            "streaming evaluation manifest.\n\n"
            f"Expected: {EXPECTED_SAMPLE_COUNT}\n"
            f"Actual  : {len(records)}"
        )

    required_fields = {
        "audio_filepath",
        "language",
        "sample_id",
    }

    language_counts = Counter()

    sample_keys = set()

    for index, record in enumerate(records):
        missing = required_fields - record.keys()

        if missing:
            raise RuntimeError(f"Manifest record {index} is missing fields: {sorted(missing)}")

        language = str(record["language"])

        if language not in EXPECTED_LANGUAGES:
            raise RuntimeError(f"Unexpected language '{language}' in manifest record {index}.")

        language_counts[language] += 1

        sample_key = (
            language,
            str(record["sample_id"]),
        )

        if sample_key in sample_keys:
            raise RuntimeError(f"Duplicate sample in manifest: {sample_key}")

        sample_keys.add(sample_key)

        duration = get_duration(record)

        if duration <= 0:
            raise RuntimeError(f"Invalid duration for sample {sample_key}: {duration}")

        audio_path = Path(record["audio_filepath"])

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file referenced by the manifest does not exist:\n{audio_path}")

    for language in EXPECTED_LANGUAGES:
        actual = language_counts[language]

        if actual != EXPECTED_SAMPLES_PER_LANGUAGE:
            raise RuntimeError(f"Expected {EXPECTED_SAMPLES_PER_LANGUAGE} samples for {language}, found {actual}.")


# ============================================================================
# Model
# ============================================================================


def resolve_model_path() -> Path:
    downloaded_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILENAME,
        revision=MODEL_REVISION,
    )

    model_path = Path(downloaded_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Downloaded Parakeet model was not found:\n{model_path}")

    return model_path


# ============================================================================
# NeMo execution
# ============================================================================


def build_nemo_command(
    model_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(NEMO_SCRIPT),
        f"model_path={model_path.as_posix()}",
        "pretrained_name=null",
        (f"dataset_manifest={EVALUATION_MANIFEST_PATH.as_posix()}"),
        (f"output_filename={NEMO_OUTPUT_PATH.as_posix()}"),
        f"left_context_secs={LEFT_CONTEXT_SECS}",
        f"chunk_secs={CHUNK_SECS}",
        f"right_context_secs={RIGHT_CONTEXT_SECS}",
        "batch_size=1",
        "num_workers=0",
        "cuda=0",
        f"compute_dtype={COMPUTE_DTYPE}",
        "simulated=false",
        "decoding.strategy=greedy_batch",
        "decoding.greedy.loop_labels=true",
        "att_context_size_as_chunk=true",
        "sort_by_duration=false",
        "overwrite_transcripts=true",
        "append_pred=false",
        "calculate_wer=false",
        "calculate_rtfx=true",
        "timestamps=false",
        "confidence=false",
    ]


def collect_partial_traces(
    command: list[str],
    manifest_records: list[dict],
) -> tuple[
    list[dict],
    dict,
]:
    traces: list[dict] = []

    current_utterance_index = -1
    previous_chunk_index: int | None = None

    effective_contexts = None
    theoretical_latency = None
    instrumented_rtfx = None

    print()
    print("=" * 80)
    print("STARTING INSTRUMENTED NEMO STREAMING INFERENCE")
    print("=" * 80)
    print()

    with NEMO_LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        child_env = os.environ.copy()

        # Force the NeMo subprocess to emit stdout/stderr as UTF-8.
        # Without this, accented multilingual partial transcripts can
        # be corrupted when captured through a pipe on Windows.
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"

        process = subprocess.Popen(
            command,
            cwd=NEMO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )

        if process.stdout is None:
            raise RuntimeError("Could not capture NeMo stdout.")

        for raw_line in process.stdout:
            print(
                raw_line,
                end="",
            )

            log_file.write(raw_line)

            line = raw_line.rstrip("\r\n")

            context_match = CORRECTED_CONTEXT_PATTERN.search(line)

            if context_match:
                effective_contexts = {
                    "left_context_secs": float(context_match.group(1)),
                    "chunk_secs": float(context_match.group(2)),
                    "right_context_secs": float(context_match.group(3)),
                }

            latency_match = LATENCY_PATTERN.search(line)

            if latency_match:
                theoretical_latency = float(latency_match.group(1))

            rtfx_match = RTFX_PATTERN.search(line)

            if rtfx_match:
                instrumented_rtfx = float(rtfx_match.group(1))

            partial_match = PARTIAL_PATTERN.search(line)

            if not partial_match:
                continue

            chunk_index = int(partial_match.group(1))

            audio_available_s = float(partial_match.group(2))

            partial_text = partial_match.group(3).strip()

            if chunk_index == 1:
                current_utterance_index += 1
                previous_chunk_index = 0

            if current_utterance_index < 0:
                raise RuntimeError("A partial transcript was found before PARTIAL 01.")

            if current_utterance_index >= len(manifest_records):
                raise RuntimeError("More utterances were found in the NeMo log than exist in the manifest.")

            expected_chunk_index = previous_chunk_index + 1 if previous_chunk_index is not None else 1

            if chunk_index != expected_chunk_index:
                raise RuntimeError(
                    "Unexpected partial sequence for "
                    f"utterance {current_utterance_index}.\n\n"
                    f"Expected PARTIAL "
                    f"{expected_chunk_index:02d}, "
                    f"found PARTIAL {chunk_index:02d}."
                )

            manifest_record = manifest_records[current_utterance_index]

            traces.append(
                {
                    "utterance_index": (current_utterance_index),
                    "language": str(manifest_record["language"]),
                    "sample_id": (manifest_record["sample_id"]),
                    "dataset_index": (manifest_record.get("dataset_index")),
                    "duration_s": get_duration(manifest_record),
                    "chunk_index": chunk_index,
                    "audio_available_s": (audio_available_s),
                    "partial_text": (partial_text),
                }
            )

            previous_chunk_index = chunk_index

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"\nNeMo streaming inference failed.\n\nExit code: {return_code}\nLog      : {NEMO_LOG_PATH}"
        )

    utterances_found = current_utterance_index + 1

    if utterances_found != len(manifest_records):
        raise RuntimeError(
            "The number of utterances reconstructed "
            "from PARTIAL 01 resets does not match "
            "the evaluation manifest.\n\n"
            f"Manifest   : {len(manifest_records)}\n"
            f"Log groups : {utterances_found}"
        )

    if not traces:
        raise RuntimeError("NeMo completed successfully but no [PARTIAL ...] records were found.")

    if effective_contexts is None:
        raise RuntimeError("Could not parse corrected streaming contexts from the NeMo log.")

    if theoretical_latency is None:
        raise RuntimeError("Could not parse theoretical latency from the NeMo log.")

    execution_metadata = {
        "effective_streaming_config": {
            **effective_contexts,
            "theoretical_latency_secs": (theoretical_latency),
        },
        "instrumented_rtfx": (instrumented_rtfx),
    }

    return (
        traces,
        execution_metadata,
    )


# ============================================================================
# Post-run validation
# ============================================================================


def build_utterance_records(
    manifest_records: list[dict],
    traces: list[dict],
) -> list[dict]:
    if not NEMO_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"NeMo completed but the final prediction file was not found:\n{NEMO_OUTPUT_PATH}")

    predictions = read_jsonl(NEMO_OUTPUT_PATH)

    if len(predictions) != len(manifest_records):
        raise RuntimeError(
            "Prediction count does not match manifest.\n\n"
            f"Manifest    : {len(manifest_records)}\n"
            f"Predictions : {len(predictions)}"
        )

    traces_by_utterance: dict[
        int,
        list[dict],
    ] = defaultdict(list)

    for trace in traces:
        traces_by_utterance[int(trace["utterance_index"])].append(trace)

    utterance_records: list[dict] = []

    for index, (
        manifest_record,
        prediction_record,
    ) in enumerate(
        zip(
            manifest_records,
            predictions,
        )
    ):
        sample_traces = traces_by_utterance[index]

        if not sample_traces:
            raise RuntimeError(f"No partial traces were found for utterance index {index}.")

        first_non_empty = next(
            (trace for trace in sample_traces if trace["partial_text"]),
            None,
        )

        final_partial = sample_traces[-1]["partial_text"]

        final_prediction = str(
            prediction_record.get(
                "pred_text",
                "",
            )
        ).strip()

        utterance_records.append(
            {
                "utterance_index": index,
                "language": str(manifest_record["language"]),
                "sample_id": (manifest_record["sample_id"]),
                "dataset_index": (manifest_record.get("dataset_index")),
                "duration_s": get_duration(manifest_record),
                "reference": str(
                    manifest_record.get(
                        "text",
                        "",
                    )
                ),
                "partial_count": len(sample_traces),
                "first_partial_audio_s": (sample_traces[0]["audio_available_s"]),
                "first_non_empty_audio_s": (None if first_non_empty is None else first_non_empty["audio_available_s"]),
                "first_non_empty_text": ("" if first_non_empty is None else first_non_empty["partial_text"]),
                "last_partial_audio_s": (sample_traces[-1]["audio_available_s"]),
                "final_partial": (final_partial),
                "final_prediction": (final_prediction),
                "final_partial_matches_prediction": (final_partial == final_prediction),
            }
        )

    return utterance_records


# ============================================================================
# Metadata
# ============================================================================


def build_metadata(
    manifest_records: list[dict],
    traces: list[dict],
    utterance_records: list[dict],
    nemo_info: dict,
    model_path: Path,
    execution_metadata: dict,
) -> dict:
    language_counts = Counter(str(record["language"]) for record in manifest_records)

    total_duration_s = sum(get_duration(record) for record in manifest_records)

    final_matches = sum(1 for record in utterance_records if record["final_partial_matches_prediction"])

    return {
        "experiment": ("parakeet_partial_trace_collection"),
        "created_at": (datetime.now().astimezone().isoformat(timespec="seconds")),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "file": MODEL_FILENAME,
            "resolved_path": str(model_path),
            "compute_dtype": (COMPUTE_DTYPE),
        },
        "nemo": {
            "repository": str(NEMO_ROOT),
            "base_commit": (NEMO_BASE_COMMIT),
            "instrumentation_branch": (nemo_info["branch"]),
            "instrumentation_commit": (nemo_info["commit"]),
            "script": str(NEMO_SCRIPT),
        },
        "requested_streaming_config": {
            "left_context_secs": (LEFT_CONTEXT_SECS),
            "chunk_secs": (CHUNK_SECS),
            "right_context_secs": (RIGHT_CONTEXT_SECS),
            "requested_latency_secs": (CHUNK_SECS + RIGHT_CONTEXT_SECS),
        },
        **execution_metadata,
        "dataset": {
            "manifest": str(EVALUATION_MANIFEST_PATH),
            "manifest_sha256": sha256_file(EVALUATION_MANIFEST_PATH),
            "samples": len(manifest_records),
            "samples_per_language": dict(language_counts),
            "total_duration_s": (total_duration_s),
            "total_duration_minutes": (total_duration_s / 60.0),
        },
        "collection": {
            "partial_records": len(traces),
            "utterances": len(utterance_records),
            "final_partial_matches_prediction": (final_matches),
        },
        "performance_note": (
            "instrumented_rtfx includes per-chunk "
            "hypothesis conversion and logging and "
            "must not be used as the streaming "
            "performance benchmark."
        ),
        "outputs": {
            "partial_traces": str(PARTIAL_TRACES_PATH),
            "utterances": str(UTTERANCES_PATH),
            "final_predictions": str(NEMO_OUTPUT_PATH),
            "nemo_log": str(NEMO_LOG_PATH),
            "metadata": str(METADATA_PATH),
        },
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print("=" * 80)
    print("PARAKEET-TDT PARTIAL TRACE COLLECTION")
    print("=" * 80)
    print(f"Model       : {MODEL_ID}")
    print(f"Revision    : {MODEL_REVISION}")
    print(f"Precision   : {COMPUTE_DTYPE}")
    print(f"Left        : {LEFT_CONTEXT_SECS:.2f} s")
    print(f"Chunk       : {CHUNK_SECS:.2f} s")
    print(f"Right       : {RIGHT_CONTEXT_SECS:.2f} s")
    print(f"Latency     : {CHUNK_SECS + RIGHT_CONTEXT_SECS:.2f} s")
    print("=" * 80)

    validate_paths()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_records = read_jsonl(EVALUATION_MANIFEST_PATH)

    validate_manifest(manifest_records)

    print()
    print("Manifest validation : OK")
    print(f"Samples             : {len(manifest_records)}")

    nemo_info = validate_instrumented_nemo()

    print()
    print("NeMo instrumentation: OK")
    print(f"Branch              : {nemo_info['branch']}")
    print(f"Commit              : {nemo_info['commit']}")

    model_path = resolve_model_path()

    print()
    print(f"Model file          : {model_path}")

    command = build_nemo_command(model_path)

    traces, execution_metadata = collect_partial_traces(
        command=command,
        manifest_records=(manifest_records),
    )

    utterance_records = build_utterance_records(
        manifest_records=(manifest_records),
        traces=traces,
    )

    write_jsonl(
        PARTIAL_TRACES_PATH,
        traces,
    )

    write_jsonl(
        UTTERANCES_PATH,
        utterance_records,
    )

    metadata = build_metadata(
        manifest_records=(manifest_records),
        traces=traces,
        utterance_records=(utterance_records),
        nemo_info=nemo_info,
        model_path=model_path,
        execution_metadata=(execution_metadata),
    )

    write_json(
        METADATA_PATH,
        metadata,
    )

    effective = metadata["effective_streaming_config"]

    final_matches = metadata["collection"]["final_partial_matches_prediction"]

    print()
    print("=" * 80)
    print("PARTIAL TRACE COLLECTION COMPLETED")
    print("=" * 80)
    print(f"Utterances          : {len(utterance_records)}")
    print(f"Partial records     : {len(traces)}")
    print(f"Final partial match : {final_matches}/{len(utterance_records)}")
    print()
    print(f"Effective left      : {effective['left_context_secs']:.2f} s")
    print(f"Effective chunk     : {effective['chunk_secs']:.2f} s")
    print(f"Effective right     : {effective['right_context_secs']:.2f} s")
    print(f"Theoretical latency : {effective['theoretical_latency_secs']:.2f} s")

    instrumented_rtfx = metadata.get("instrumented_rtfx")

    if instrumented_rtfx is not None:
        print(f"Instrumented RTFx   : {instrumented_rtfx:.2f}")
        print("                      (diagnostic only; not benchmark)")

    print("-" * 80)
    print(f"Traces              : {PARTIAL_TRACES_PATH}")
    print(f"Utterances          : {UTTERANCES_PATH}")
    print(f"Final predictions   : {NEMO_OUTPUT_PATH}")
    print(f"NeMo log            : {NEMO_LOG_PATH}")
    print(f"Metadata            : {METADATA_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
