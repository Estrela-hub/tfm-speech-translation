import argparse
import csv
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from statistics import mean

import soundfile as sf
from huggingface_hub import hf_hub_download
from jiwer import wer

from experiments_asr.asr_01_fleurs_subset import load_audio_from_entry, load_manifest, load_metadata, validate_manifest

# ============================================================================
# Experiment configuration
# ============================================================================

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
MODEL_FILENAME = "parakeet-tdt-0.6b-v3.nemo"

NEMO_COMMIT = "fd6a877539710e2b98f28c43272ff81312f83417"

COMPUTE_DTYPE = "float32"

LANGUAGES = (
    "en",
    "es",
    "fr",
    "de",
    "pt",
)


# Avoid the absolute shortest/longest outliers.
DURATION_QUANTILES = (
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
)

SAMPLES_PER_LANGUAGE = len(DURATION_QUANTILES)

CONFIGS = {
    "baseline_4s": {
        "left_context_secs": 10.0,
        "chunk_secs": 2.0,
        "right_context_secs": 2.0,
    },
    "lowlat_2s": {
        "left_context_secs": 10.0,
        "chunk_secs": 0.16,
        "right_context_secs": 1.84,
    },
    "lowlat_1_76s": {
        "left_context_secs": 10.0,
        "chunk_secs": 0.16,
        "right_context_secs": 1.60,
    },
    "lowlat_1_52s": {
        "left_context_secs": 10.0,
        "chunk_secs": 0.16,
        "right_context_secs": 1.36,
    },
    "lowlat_1_04s": {
        "left_context_secs": 10.0,
        "chunk_secs": 0.16,
        "right_context_secs": 0.88,
    },
    "lowlat_0_56s": {
        "left_context_secs": 10.0,
        "chunk_secs": 0.16,
        "right_context_secs": 0.40,
    },
    "left_6s": {
        "left_context_secs": 6.0,
        "chunk_secs": 0.16,
        "right_context_secs": 1.84,
    },
    "left_4s": {
        "left_context_secs": 4.0,
        "chunk_secs": 0.16,
        "right_context_secs": 1.84,
    },
    "left_2s": {
        "left_context_secs": 2.0,
        "chunk_secs": 0.16,
        "right_context_secs": 1.84,
    },
}


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEMO_REPO = PROJECT_ROOT / "external" / "NeMo-Speech"

NEMO_SCRIPT_RELATIVE = (
    Path("examples") / "asr" / "asr_chunked_inference" / "rnnt" / "speech_to_text_streaming_infer_rnnt.py"
)

NEMO_SCRIPT = NEMO_REPO / NEMO_SCRIPT_RELATIVE

RESULTS_DIR = PROJECT_ROOT / "results" / "asr_selection" / "parakeet_streaming_configs"

AUDIO_DIR = RESULTS_DIR / "audio"

SELECTION_PATH = RESULTS_DIR / "selection.jsonl"

NEMO_MANIFEST_PATH = RESULTS_DIR / "evaluation_manifest.jsonl"

COMBINED_SUMMARY_JSON = RESULTS_DIR / "summary.json"
COMBINED_SUMMARY_CSV = RESULTS_DIR / "summary.csv"

OFFLINE_PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "asr_selection" / "parakeet_tdt_0_6b_v3" / "fleurs_fp32_predictions.jsonl"
)

COMPARISON_JSON_PATH = RESULTS_DIR / "offline_streaming_comparison.json"

COMPARISON_CSV_PATH = RESULTS_DIR / "offline_streaming_comparison.csv"

# ============================================================================
# Utilities
# ============================================================================


def write_json(
    path: Path,
    data,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                + "\n"
            )


def read_jsonl(
    path: Path,
) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_text(
    text: str,
) -> str:
    """
    basic_v1 normalization used in the previous ASR evaluation:

    - Unicode NFKC
    - lowercase
    - punctuation -> whitespace
    - collapse whitespace
    - numbers are deliberately NOT normalized
    """

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    characters = []

    for character in text:
        if unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)

    text = "".join(characters)

    return " ".join(text.split())


def get_sample_id(
    entry: dict,
) -> str:
    if "sample_id" not in entry:
        raise KeyError("Manifest entry does not contain 'sample_id'.")

    return str(entry["sample_id"])


def get_dataset_index(
    entry: dict,
):
    if "dataset_index" in entry:
        return entry["dataset_index"]

    if "index" in entry:
        return entry["index"]

    return None


def get_manifest_duration(
    entry: dict,
) -> float | None:
    """
    Try common duration fields without decoding the audio.
    """

    for key in (
        "duration_s",
        "duration",
        "audio_duration",
        "duration_seconds",
    ):
        if key in entry and entry[key] is not None:
            return float(entry[key])

    return None


# ============================================================================
# Fixed evaluation subset
# ============================================================================


def create_selection(
    entries: list[dict],
) -> list[dict]:
    selections = []

    for language in LANGUAGES:
        language_entries = [entry for entry in entries if entry["language"] == language]

        if len(language_entries) < SAMPLES_PER_LANGUAGE:
            raise RuntimeError(f"Not enough entries for {language}: {len(language_entries)}")

        durations_available = all(get_manifest_duration(entry) is not None for entry in language_entries)

        if durations_available:
            ordered_entries = sorted(  # type: ignore
                language_entries,
                key=lambda entry: get_manifest_duration(entry),  # type: ignore
            )

            selection_basis = "duration_quantiles"

        else:
            # Explicit fallback.
            # We do NOT claim duration stratification in this case.
            ordered_entries = language_entries
            selection_basis = "manifest_order_quantiles"

            print(
                f"WARNING: Duration not stored in manifest for {language}. "
                "Using evenly distributed manifest positions instead."
            )

        number_of_entries = len(ordered_entries)

        for quantile in DURATION_QUANTILES:
            index = round(quantile * (number_of_entries - 1))

            entry = ordered_entries[index]

            selections.append(
                {
                    "language": language,
                    "sample_id": get_sample_id(entry),
                    "dataset_index": get_dataset_index(entry),
                    "requested_quantile": quantile,
                    "selection_basis": selection_basis,
                }
            )

    return selections


def resolve_selection_entries(
    selections: list[dict],
    entries: list[dict],
) -> list[dict]:
    lookup = {
        (
            entry["language"],
            get_sample_id(entry),
        ): entry
        for entry in entries
    }

    resolved = []

    for selection in selections:
        key = (
            selection["language"],
            str(selection["sample_id"]),
        )

        if key not in lookup:
            raise RuntimeError(f"Previously selected FLEURS sample could not be recovered: {key}")

        resolved.append(
            {
                "selection": selection,
                "entry": lookup[key],
            }
        )

    return resolved


def load_or_create_selection(
    entries: list[dict],
) -> list[dict]:
    if SELECTION_PATH.exists():
        print(f"Reusing fixed selection: {SELECTION_PATH}")

        selections = read_jsonl(SELECTION_PATH)

    else:
        print("Creating fixed multilingual streaming subset...")

        selections = create_selection(entries)

    return resolve_selection_entries(
        selections,
        entries,
    )


def prepare_evaluation_subset(
    selected_entries: list[dict],
    metadata: dict,
) -> list[dict]:
    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    nemo_records = []
    selection_records = []

    print()
    print("=" * 80)
    print("STREAMING EVALUATION SUBSET")
    print("=" * 80)

    for item in selected_entries:
        selection = item["selection"]
        entry = item["entry"]

        samples, sample_rate, reference = load_audio_from_entry(
            entry,
            metadata,
        )

        duration_s = len(samples) / sample_rate

        language = entry["language"]
        sample_id = get_sample_id(entry)

        audio_filename = f"{language}_{sample_id}.wav"

        audio_path = AUDIO_DIR / audio_filename

        # Keep the same WAV representation used by the streaming smoke test.
        sf.write(
            audio_path,
            samples,
            sample_rate,
            subtype="PCM_16",
        )

        nemo_record = {
            "audio_filepath": audio_path.resolve().as_posix(),
            "duration": duration_s,
            "text": reference,
            "language": language,
            "sample_id": sample_id,
            "dataset_index": get_dataset_index(entry),
        }

        nemo_records.append(nemo_record)

        selection_record = {
            "language": language,
            "sample_id": sample_id,
            "dataset_index": get_dataset_index(entry),
            "requested_quantile": selection["requested_quantile"],
            "selection_basis": selection["selection_basis"],
            "duration_s": duration_s,
            "reference": reference,
        }

        selection_records.append(selection_record)

        print(
            f"{language.upper():>2} | "
            f"sample={sample_id:<5} | "
            f"duration={duration_s:6.2f}s | "
            f"q={selection['requested_quantile']:.2f}"
        )

    write_jsonl(
        SELECTION_PATH,
        selection_records,
    )

    write_jsonl(
        NEMO_MANIFEST_PATH,
        nemo_records,
    )

    print("-" * 80)

    total_duration = sum(record["duration"] for record in nemo_records)

    print(f"Samples        : {len(nemo_records)}")

    print(f"Total duration : {total_duration:.2f} s ({total_duration / 60.0:.2f} min)")

    print(f"Manifest       : {NEMO_MANIFEST_PATH}")

    print("=" * 80)

    return nemo_records


# ============================================================================
# NeMo validation
# ============================================================================


def validate_nemo_script() -> None:
    if not NEMO_SCRIPT.exists():
        raise FileNotFoundError(f"NeMo streaming script was not found:\n{NEMO_SCRIPT}")

    if not (NEMO_REPO / ".git").exists():
        raise RuntimeError("NeMo-Speech directory is not a Git repository.")

    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            NEMO_COMMIT,
            "--",
            NEMO_SCRIPT_RELATIVE.as_posix(),
        ],
        cwd=NEMO_REPO,
        check=False,
    )

    if result.returncode == 1:
        raise RuntimeError(
            "\nThe NeMo streaming script differs from the "
            "official v3.0.0 version.\n\n"
            "The evaluation must NOT use the partial-transcript "
            "instrumentation because it changes execution cost.\n\n"
            "Save the instrumentation branch and switch to:\n\n"
            f"    git switch --detach {NEMO_COMMIT}\n"
        )

    if result.returncode != 0:
        raise RuntimeError(f"Could not validate the NeMo script against commit {NEMO_COMMIT}.")

    print()
    print("NeMo script validation : official v3.0.0 code")

    print(f"NeMo commit            : {NEMO_COMMIT}")


def resolve_model_path() -> Path:
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILENAME,
        revision=MODEL_REVISION,
    )

    path = Path(model_path)

    print(f"Parakeet revision      : {MODEL_REVISION}")

    print(f"Model file             : {path}")

    return path


# ============================================================================
# NeMo execution
# ============================================================================


CORRECTED_CONTEXT_PATTERN = re.compile(
    r"Corrected contexts \(sec\): "
    r"Left ([0-9.]+), "
    r"Chunk ([0-9.]+), "
    r"Right ([0-9.]+)"
)

LATENCY_PATTERN = re.compile(
    r"Theoretical latency: "
    r"([0-9.]+) seconds"
)

RTFX_PATTERN = re.compile(r"RTFx: ([0-9.]+)")


def build_nemo_command(
    configuration: dict,
    model_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(NEMO_SCRIPT),
        "pretrained_name=null",
        f"model_path={model_path.resolve().as_posix()}",
        "audio_dir=null",
        f"dataset_manifest={NEMO_MANIFEST_PATH.resolve().as_posix()}",
        f"output_filename={output_path.resolve().as_posix()}",
        f"chunk_secs={configuration['chunk_secs']}",
        f"left_context_secs={configuration['left_context_secs']}",
        f"right_context_secs={configuration['right_context_secs']}",
        # Single-stream evaluation: process one utterance at a time.
        "batch_size=1",
        "num_workers=0",
        "cuda=0",
        f"compute_dtype={COMPUTE_DTYPE}",
        # Real stateful streaming.
        "simulated=false",
        "decoding.strategy=greedy_batch",
        "decoding.greedy.loop_labels=true",
        "att_context_size_as_chunk=true",
        # Preserve manifest order so predictions align exactly
        # with the fixed 25-sample subset.
        "sort_by_duration=false",
        "overwrite_transcripts=true",
        # We calculate WER ourselves using basic_v1.
        "calculate_wer=false",
        # Let NeMo calculate streaming throughput.
        "calculate_rtfx=true",
        "clean_groundtruth_text=false",
        "timestamps=false",
        "confidence=false",
    ]


def run_nemo_configuration(
    config_name: str,
    configuration: dict,
    model_path: Path,
) -> tuple[Path, dict]:
    config_dir = RESULTS_DIR / config_name

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = config_dir / "predictions.jsonl"

    log_path = config_dir / "nemo.log"

    if output_path.exists():
        output_path.unlink()

    command = build_nemo_command(
        configuration,
        model_path,
        output_path,
    )

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    runtime_info = {
        "effective_left_context_secs": None,
        "effective_chunk_secs": None,
        "effective_right_context_secs": None,
        "theoretical_latency_secs": None,
        "rtfx": None,
    }

    print()
    print("=" * 80)
    print(f"RUNNING CONFIGURATION: {config_name}")
    print("=" * 80)

    print(f"Requested left  : {configuration['left_context_secs']:.2f} s")

    print(f"Requested chunk : {configuration['chunk_secs']:.2f} s")

    print(f"Requested right : {configuration['right_context_secs']:.2f} s")

    print(f"Requested latency: {configuration['chunk_secs'] + configuration['right_context_secs']:.2f} s")

    print("=" * 80)
    print()

    process = subprocess.Popen(
        command,
        cwd=NEMO_SCRIPT.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Could not capture NeMo output.")

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        for line in process.stdout:
            print(
                line,
                end="",
            )

            log_file.write(line)

            context_match = CORRECTED_CONTEXT_PATTERN.search(line)

            if context_match:
                runtime_info["effective_left_context_secs"] = float(context_match.group(1))  # type: ignore

                runtime_info["effective_chunk_secs"] = float(context_match.group(2))  # type: ignore

                runtime_info["effective_right_context_secs"] = float(context_match.group(3))  # type: ignore

            latency_match = LATENCY_PATTERN.search(line)

            if latency_match:
                runtime_info["theoretical_latency_secs"] = float(latency_match.group(1))  # type: ignore

            rtfx_match = RTFX_PATTERN.search(line)

            if rtfx_match:
                runtime_info["rtfx"] = float(rtfx_match.group(1))  # type: ignore

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"NeMo failed for {config_name}.\nExit code: {return_code}\nLog: {log_path}")

    if not output_path.exists():
        raise RuntimeError(f"NeMo completed but predictions were not created:\n{output_path}")

    missing_metrics = [key for key, value in runtime_info.items() if value is None]

    if missing_metrics:
        raise RuntimeError(
            f"Could not extract the following NeMo metrics from the log: {missing_metrics}\nLog: {log_path}"
        )

    return (
        output_path,
        runtime_info,
    )


# ============================================================================
# WER evaluation
# ============================================================================


def evaluate_predictions(
    prediction_path: Path,
    evaluation_records: list[dict],
    evaluated_output_path: Path,
) -> dict:
    prediction_records = read_jsonl(prediction_path)

    if len(prediction_records) != len(evaluation_records):
        raise RuntimeError(
            "Prediction count does not match "
            "evaluation manifest.\n"
            f"Expected: {len(evaluation_records)}\n"
            f"Found   : {len(prediction_records)}"
        )

    evaluated_records = []

    for source_record, prediction_record in zip(
        evaluation_records,
        prediction_records,
    ):
        if "pred_text" not in prediction_record:
            raise RuntimeError("NeMo prediction does not contain 'pred_text'.")

        reference = str(source_record["text"]).strip()

        prediction = str(prediction_record["pred_text"]).strip()

        evaluated_records.append(
            {
                "language": source_record["language"],
                "sample_id": source_record["sample_id"],
                "dataset_index": source_record["dataset_index"],
                "duration_s": source_record["duration"],
                "reference": reference,
                "prediction": prediction,
                "reference_normalized": normalize_text(reference),
                "prediction_normalized": normalize_text(prediction),
            }
        )

    write_jsonl(
        evaluated_output_path,
        evaluated_records,
    )

    per_language = {}

    for language in LANGUAGES:
        language_records = [record for record in evaluated_records if record["language"] == language]

        references = [record["reference_normalized"] for record in language_records]

        predictions = [record["prediction_normalized"] for record in language_records]

        language_wer = wer(
            references,
            predictions,
        )

        empty_predictions = sum(1 for prediction in predictions if not prediction)

        per_language[language] = {
            "wer": language_wer,
            "wer_percent": language_wer * 100.0,
            "samples": len(language_records),
            "empty_predictions": empty_predictions,
        }

    all_references = [record["reference_normalized"] for record in evaluated_records]

    all_predictions = [record["prediction_normalized"] for record in evaluated_records]

    overall_wer = wer(
        all_references,
        all_predictions,
    )

    macro_wer = mean(per_language[language]["wer"] for language in LANGUAGES)

    empty_predictions = sum(1 for prediction in all_predictions if not prediction)

    return {
        "normalization": "basic_v1",
        "samples": len(evaluated_records),
        "macro_wer": macro_wer,
        "macro_wer_percent": macro_wer * 100.0,
        "overall_wer": overall_wer,
        "overall_wer_percent": overall_wer * 100.0,
        "empty_predictions": empty_predictions,
        "per_language": per_language,
    }


def evaluate_offline_subset(
    evaluation_records: list[dict],
) -> dict:
    if not OFFLINE_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Offline Parakeet predictions were not found:\n{OFFLINE_PREDICTIONS_PATH}")

    offline_records = read_jsonl(OFFLINE_PREDICTIONS_PATH)

    lookup = {
        (
            str(record["language"]),
            str(record["sample_id"]),
        ): record
        for record in offline_records
    }

    selected_records = []

    for evaluation_record in evaluation_records:
        key = (
            str(evaluation_record["language"]),
            str(evaluation_record["sample_id"]),
        )

        if key not in lookup:
            raise RuntimeError(f"Offline prediction not found for language={key[0]}, sample_id={key[1]}")

        offline_record = lookup[key]

        # Validate that we are comparing exactly the same FLEURS sample.
        expected_dataset_index = evaluation_record.get("dataset_index")

        offline_dataset_index = offline_record.get("dataset_index")

        if (
            expected_dataset_index is not None
            and offline_dataset_index is not None
            and int(expected_dataset_index) != int(offline_dataset_index)
        ):
            raise RuntimeError(
                f"Dataset index mismatch for {key}: streaming={expected_dataset_index}, offline={offline_dataset_index}"
            )

        reference = str(offline_record["reference"]).strip()

        prediction = str(offline_record["prediction"]).strip()

        # Important:
        # Ignore the legacy normalized_* and WER fields.
        # Recompute everything with the SAME normalizer
        # currently used for the streaming experiment.
        selected_records.append(
            {
                "language": key[0],
                "sample_id": key[1],
                "reference": reference,
                "prediction": prediction,
                "reference_normalized": normalize_text(reference),
                "prediction_normalized": normalize_text(prediction),
            }
        )

    per_language = {}

    for language in LANGUAGES:
        language_records = [record for record in selected_records if record["language"] == language]

        references = [record["reference_normalized"] for record in language_records]

        predictions = [record["prediction_normalized"] for record in language_records]

        language_wer = wer(
            references,
            predictions,
        )

        empty_predictions = sum(1 for prediction in predictions if not prediction)

        per_language[language] = {
            "wer": language_wer,
            "wer_percent": language_wer * 100.0,
            "samples": len(language_records),
            "empty_predictions": empty_predictions,
        }

    all_references = [record["reference_normalized"] for record in selected_records]

    all_predictions = [record["prediction_normalized"] for record in selected_records]

    overall_wer = wer(
        all_references,
        all_predictions,
    )

    macro_wer = mean(per_language[language]["wer"] for language in LANGUAGES)

    return {
        "mode": "offline_full_utterance",
        "samples": len(selected_records),
        "macro_wer": macro_wer,
        "macro_wer_percent": macro_wer * 100.0,
        "overall_wer": overall_wer,
        "overall_wer_percent": overall_wer * 100.0,
        "empty_predictions": sum(1 for prediction in all_predictions if not prediction),
        "per_language": per_language,
    }


# ============================================================================
# Results
# ============================================================================


def build_config_summary(
    config_name: str,
    configuration: dict,
    runtime_info: dict,
    quality_info: dict,
    evaluation_records: list[dict],
) -> dict:
    return {
        "config_name": config_name,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "format": "nemo",
        },
        "nemo": {
            "version": "3.0.0",
            "commit": NEMO_COMMIT,
        },
        "precision": COMPUTE_DTYPE,
        "requested_streaming_config": {
            **configuration,
            "latency_secs": (configuration["chunk_secs"] + configuration["right_context_secs"]),
        },
        "effective_streaming_config": {
            "left_context_secs": runtime_info["effective_left_context_secs"],
            "chunk_secs": runtime_info["effective_chunk_secs"],
            "right_context_secs": runtime_info["effective_right_context_secs"],
            "theoretical_latency_secs": runtime_info["theoretical_latency_secs"],
        },
        "performance": {
            "rtfx": runtime_info["rtfx"],
        },
        "quality": quality_info,
        "evaluation_subset": {
            "samples": len(evaluation_records),
            "samples_per_language": (SAMPLES_PER_LANGUAGE),
            "languages": list(LANGUAGES),
            "total_audio_seconds": sum(record["duration"] for record in evaluation_records),
        },
    }


def print_config_summary(
    summary: dict,
) -> None:
    print()
    print("=" * 80)
    print(f"RESULT: {summary['config_name']}")
    print("=" * 80)

    effective = summary["effective_streaming_config"]

    quality = summary["quality"]

    print(f"Effective left    : {effective['left_context_secs']:.2f} s")

    print(f"Effective chunk   : {effective['chunk_secs']:.2f} s")

    print(f"Effective right   : {effective['right_context_secs']:.2f} s")

    print(f"Theoretical latency: {effective['theoretical_latency_secs']:.2f} s")

    print(f"RTFx              : {summary['performance']['rtfx']:.2f}")

    print()

    for language in LANGUAGES:
        language_result = quality["per_language"][language]

        print(
            f"{language.upper():>2} WER            : "
            f"{language_result['wer_percent']:.2f}% "
            f"(empty={language_result['empty_predictions']})"
        )

    print("-" * 80)

    print(f"Macro WER         : {quality['macro_wer_percent']:.2f}%")

    print(f"Overall WER       : {quality['overall_wer_percent']:.2f}%")

    print(f"Empty predictions : {quality['empty_predictions']}")

    print("=" * 80)


def collect_existing_summaries() -> list[dict]:
    summaries = []

    for config_name in CONFIGS:
        summary_path = RESULTS_DIR / config_name / "summary.json"

        if summary_path.exists():
            with summary_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                summaries.append(json.load(file))

    return summaries


def write_combined_summary(
    summaries: list[dict],
) -> None:
    combined = {
        "experiment": ("parakeet_streaming_config_selection"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "nemo_version": "3.0.0",
        "nemo_commit": NEMO_COMMIT,
        "precision": COMPUTE_DTYPE,
        "samples_per_language": SAMPLES_PER_LANGUAGE,
        "languages": list(LANGUAGES),
        "normalization": "basic_v1",
        "configurations": summaries,
    }

    write_json(
        COMBINED_SUMMARY_JSON,
        combined,
    )

    fieldnames = [
        "config_name",
        "requested_chunk_s",
        "requested_right_s",
        "requested_latency_s",
        "effective_chunk_s",
        "effective_right_s",
        "effective_latency_s",
        "rtfx",
        "wer_en_percent",
        "wer_es_percent",
        "wer_fr_percent",
        "wer_de_percent",
        "wer_pt_percent",
        "macro_wer_percent",
        "overall_wer_percent",
        "empty_predictions",
    ]

    with COMBINED_SUMMARY_CSV.open(
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
            requested = summary["requested_streaming_config"]

            effective = summary["effective_streaming_config"]

            quality = summary["quality"]

            writer.writerow(
                {
                    "config_name": summary["config_name"],
                    "requested_chunk_s": requested["chunk_secs"],
                    "requested_right_s": requested["right_context_secs"],
                    "requested_latency_s": requested["latency_secs"],
                    "effective_chunk_s": effective["chunk_secs"],
                    "effective_right_s": effective["right_context_secs"],
                    "effective_latency_s": effective["theoretical_latency_secs"],
                    "rtfx": summary["performance"]["rtfx"],
                    "wer_en_percent": quality["per_language"]["en"]["wer_percent"],
                    "wer_es_percent": quality["per_language"]["es"]["wer_percent"],
                    "wer_fr_percent": quality["per_language"]["fr"]["wer_percent"],
                    "wer_de_percent": quality["per_language"]["de"]["wer_percent"],
                    "wer_pt_percent": quality["per_language"]["pt"]["wer_percent"],
                    "macro_wer_percent": quality["macro_wer_percent"],
                    "overall_wer_percent": quality["overall_wer_percent"],
                    "empty_predictions": quality["empty_predictions"],
                }
            )


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Evaluate Parakeet-TDT streaming configurations on a fixed multilingual FLEURS subset.")
    )

    parser.add_argument(
        "--config",
        choices=[
            "all",
            "compare",
            *CONFIGS.keys(),
        ],
        default="baseline_4s",
        help=(
            "Streaming configuration to evaluate, "
            "'all' to run every configuration, or "
            "'compare' to analyse existing results "
            "without running inference."
        ),
    )

    return parser.parse_args()


def write_offline_streaming_comparison(
    offline_quality: dict,
    streaming_summaries: list[dict],
) -> None:
    offline_macro_wer = offline_quality["macro_wer_percent"]

    offline_overall_wer = offline_quality["overall_wer_percent"]

    rows = [
        {
            "mode": "offline_full_utterance",
            "latency_s": None,
            "rtfx": None,
            "macro_wer_percent": offline_macro_wer,
            "overall_wer_percent": offline_overall_wer,
            "delta_macro_wer_pp": 0.0,
            "delta_overall_wer_pp": 0.0,
            "empty_predictions": offline_quality["empty_predictions"],
        }
    ]

    for summary in streaming_summaries:
        quality = summary["quality"]

        effective = summary["effective_streaming_config"]

        macro_wer = quality["macro_wer_percent"]

        overall_wer = quality["overall_wer_percent"]

        rows.append(
            {
                "mode": summary["config_name"],
                "latency_s": effective["theoretical_latency_secs"],
                "rtfx": summary["performance"]["rtfx"],
                "macro_wer_percent": macro_wer,
                "overall_wer_percent": overall_wer,
                "delta_macro_wer_pp": (macro_wer - offline_macro_wer),
                "delta_overall_wer_pp": (overall_wer - offline_overall_wer),
                "empty_predictions": quality["empty_predictions"],
            }
        )

    comparison = {
        "experiment": ("parakeet_offline_vs_streaming"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "samples": offline_quality["samples"],
        "samples_per_language": (SAMPLES_PER_LANGUAGE),
        "languages": list(LANGUAGES),
        "normalization": {
            "name": "comparison_v1",
            "description": (
                "NFKC, lowercase, punctuation to whitespace, "
                "whitespace collapse, no numerical normalization. "
                "Recomputed from raw reference and prediction text "
                "for both offline and streaming outputs."
            ),
        },
        "offline": offline_quality,
        "rows": rows,
    }

    write_json(
        COMPARISON_JSON_PATH,
        comparison,
    )

    fieldnames = [
        "mode",
        "latency_s",
        "rtfx",
        "macro_wer_percent",
        "overall_wer_percent",
        "delta_macro_wer_pp",
        "delta_overall_wer_pp",
        "empty_predictions",
    ]

    with COMPARISON_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print()
    print("=" * 80)
    print("OFFLINE VS STREAMING COMPARISON")
    print("=" * 80)

    print(f"{'Mode':<24}{'Latency':>10}{'Macro WER':>12}{'Delta':>10}{'RTFx':>10}{'Empty':>8}")

    print("-" * 80)

    for row in rows:
        latency = "-" if row["latency_s"] is None else f"{row['latency_s']:.2f}s"

        rtfx = "-" if row["rtfx"] is None else f"{row['rtfx']:.2f}"

        print(
            f"{row['mode']:<24}"
            f"{latency:>10}"
            f"{row['macro_wer_percent']:>11.2f}%"
            f"{row['delta_macro_wer_pp']:>+9.2f}"
            f"{rtfx:>10}"
            f"{row['empty_predictions']:>8}"
        )

    print("=" * 80)

    print(f"Comparison JSON : {COMPARISON_JSON_PATH}")

    print(f"Comparison CSV  : {COMPARISON_CSV_PATH}")

    print("=" * 80)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("PARAKEET-TDT STREAMING CONFIGURATION EVALUATION")
    print("=" * 80)

    print(f"Model       : {MODEL_ID}")

    print(f"Revision    : {MODEL_REVISION}")

    print(f"Precision   : {COMPUTE_DTYPE}")

    print(f"Languages   : {', '.join(LANGUAGES)}")

    print(f"Samples/lang: {SAMPLES_PER_LANGUAGE}")

    print(f"Configuration: {args.config}")

    metadata = load_metadata()
    entries = load_manifest()

    validate_manifest(
        entries,
        metadata,
    )

    selected_entries = load_or_create_selection(entries)

    evaluation_records = prepare_evaluation_subset(
        selected_entries,
        metadata,
    )

    offline_quality = evaluate_offline_subset(evaluation_records)

    if args.config != "compare":
        validate_nemo_script()

        model_path = resolve_model_path()

    else:
        model_path = None

    if args.config == "all":
        configurations_to_run = list(CONFIGS.items())
    elif args.config == "compare":
        configurations_to_run = []

    else:
        configurations_to_run = [
            (
                args.config,
                CONFIGS[args.config],
            )
        ]

    for (
        config_name,
        configuration,
    ) in configurations_to_run:
        if model_path is None:
            raise RuntimeError("Model path was not initialized.")
        prediction_path, runtime_info = run_nemo_configuration(
            config_name=config_name,
            configuration=configuration,
            model_path=model_path,
        )

        config_dir = RESULTS_DIR / config_name

        evaluated_output_path = config_dir / "evaluated_predictions.jsonl"

        quality_info = evaluate_predictions(
            prediction_path=prediction_path,
            evaluation_records=evaluation_records,
            evaluated_output_path=(evaluated_output_path),
        )

        summary = build_config_summary(
            config_name=config_name,
            configuration=configuration,
            runtime_info=runtime_info,
            quality_info=quality_info,
            evaluation_records=evaluation_records,
        )

        summary_path = config_dir / "summary.json"

        write_json(
            summary_path,
            summary,
        )

        print_config_summary(summary)

    summaries = collect_existing_summaries()

    write_combined_summary(summaries)

    write_offline_streaming_comparison(
        offline_quality=offline_quality,
        streaming_summaries=summaries,
    )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)

    print(f"Combined JSON : {COMBINED_SUMMARY_JSON}")

    print(f"Combined CSV  : {COMBINED_SUMMARY_CSV}")

    print("=" * 80)


if __name__ == "__main__":
    main()
