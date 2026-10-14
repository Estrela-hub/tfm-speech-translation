import json
import platform
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path

import datasets
import jiwer
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from experiments_asr.asr_01_fleurs_subset import load_audio_from_entry, load_manifest, load_metadata, validate_manifest

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# Revision already validated by the smoke test.
MODEL_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"

LANGUAGE_LOCALES = {
    "en": "en-US",
    "es": "es-US",
    "fr": "fr-FR",
    "de": "de-DE",
    "pt": "pt-BR",
}

NORMALIZATION_VERSION = "basic_v1"


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "asr_selection" / "nemotron_3_5_asr_streaming_0_6b"

PREDICTIONS_PATH = RESULTS_DIR / "fleurs_fp32_predictions.jsonl"

SUMMARY_PATH = RESULTS_DIR / "fleurs_fp32_summary.json"


def normalize_for_wer(text: str) -> str:
    """
    Basic language-independent normalization.

    - Unicode canonicalization
    - lowercase
    - punctuation -> whitespace
    - whitespace normalization

    Numbers are deliberately NOT normalized.
    Accents and other language-specific characters are preserved.
    """

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    text = "".join(" " if unicodedata.category(character).startswith("P") else character for character in text)

    return " ".join(text.split())


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 70)
    print("NEMOTRON 3.5 - FLEURS QUALITY EVALUATION")
    print("=" * 70)

    print(f"GPU            : {torch.cuda.get_device_name(0)}")

    print(f"Model          : {MODEL_ID}")
    print(f"Revision       : {MODEL_REVISION}")
    print("Precision      : FP32")
    print("Inference mode : full utterance")
    print()

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    print("Loading model...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    model = model.to("cuda")
    model.eval()

    print("Model loaded.")
    print()

    return processor, model


def transcribe(
    samples,
    sample_rate: int,
    locale: str,
    processor,
    model,
) -> str:
    inputs = processor(
        samples,
        sampling_rate=sample_rate,
        language=locale,
        return_tensors="pt",
    )

    inputs = inputs.to(
        model.device,
        dtype=model.dtype,
    )

    # The model currently emits a warning about the default max_length.
    # We preserve the default generation behaviour used in the smoke test and suppress only that warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Using the model-agnostic default `max_length`.*",
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                return_dict_in_generate=True,
            )

    transcriptions = processor.decode(
        output.sequences,
        skip_special_tokens=True,
    )

    return transcriptions[0].strip()


def load_existing_predictions() -> dict:
    if not PREDICTIONS_PATH.exists():
        return {}

    predictions = {}

    with PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record["model_revision"] != MODEL_REVISION:
                raise RuntimeError("Existing predictions were generated with a different model revision.")

            if record["normalization_version"] != NORMALIZATION_VERSION:
                raise RuntimeError("Existing predictions use a different normalization version.")

            key = (
                record["language"],
                record["dataset_index"],
            )

            if key in predictions:
                raise RuntimeError(f"Duplicate prediction found: {key}")

            predictions[key] = record

    return predictions


def run_inference(
    entries: list[dict],
    metadata: dict,
    processor,
    model,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = load_existing_predictions()

    remaining_entries = [
        entry
        for entry in entries
        if (
            entry["language"],
            entry["dataset_index"],
        )
        not in existing
    ]

    print(f"Completed      : {len(existing)}")

    print(f"Remaining      : {len(remaining_entries)}")

    print()

    if not remaining_entries:
        return

    with PREDICTIONS_PATH.open(
        "a",
        encoding="utf-8",
    ) as output_file:
        for entry in tqdm(
            remaining_entries,
            desc="Transcribing",
            unit="utterance",
        ):
            language = entry["language"]

            samples, sample_rate, reference = load_audio_from_entry(
                entry,
                metadata,
            )

            prediction = transcribe(
                samples=samples,
                sample_rate=sample_rate,
                locale=LANGUAGE_LOCALES[language],
                processor=processor,
                model=model,
            )

            normalized_reference = normalize_for_wer(reference)

            normalized_prediction = normalize_for_wer(prediction)

            sample_wer = jiwer.wer(
                normalized_reference,
                normalized_prediction,
            )

            record = {
                "language": language,
                "locale": LANGUAGE_LOCALES[language],
                "dataset_config": entry["dataset_config"],
                "dataset_index": entry["dataset_index"],
                "sample_id": entry["sample_id"],
                "duration_s": len(samples) / sample_rate,
                "reference": reference,
                "prediction": prediction,
                "normalized_reference": normalized_reference,
                "normalized_prediction": normalized_prediction,
                "wer": sample_wer,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "precision": "fp32",
                "inference_mode": "full_utterance",
                "normalization_version": NORMALIZATION_VERSION,
                "manifest_sha256": metadata["manifest_sha256"],
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # Preserve progress if execution is interrupted.
            output_file.flush()


def load_all_predictions(
    metadata: dict,
) -> list[dict]:
    predictions = load_existing_predictions()

    records = list(predictions.values())

    expected_samples = metadata["total_samples"]

    if len(records) != expected_samples:
        raise RuntimeError(f"Evaluation is incomplete.\nExpected: {expected_samples}\nFound   : {len(records)}")

    for record in records:
        if record["manifest_sha256"] != metadata["manifest_sha256"]:
            raise RuntimeError("Prediction manifest SHA256 mismatch.")

    return records


def calculate_corpus_wer(
    records: list[dict],
) -> dict:
    references = [record["normalized_reference"] for record in records]

    predictions = [record["normalized_prediction"] for record in records]

    result = jiwer.process_words(
        references,
        predictions,
    )

    return {
        "wer": result.wer,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "hits": result.hits,
    }


def build_summary(
    records: list[dict],
    metadata: dict,
) -> dict:
    by_language = defaultdict(list)

    for record in records:
        by_language[record["language"]].append(record)

    language_results = {}

    for language in LANGUAGE_LOCALES:
        language_records = by_language[language]

        metrics = calculate_corpus_wer(language_records)

        total_duration_s = sum(record["duration_s"] for record in language_records)

        language_results[language] = {
            "locale": LANGUAGE_LOCALES[language],
            "samples": len(language_records),
            "duration_minutes": (total_duration_s / 60.0),
            **metrics,
        }

    macro_wer = sum(result["wer"] for result in language_results.values()) / len(language_results)

    overall_metrics = calculate_corpus_wer(records)

    return {
        "experiment": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "precision": "fp32",
            "inference_mode": "full_utterance",
            "normalization_version": (NORMALIZATION_VERSION),
        },
        "dataset": {
            "name": metadata["dataset"],
            "revision": metadata["dataset_revision"],
            "split": metadata["split"],
            "manifest_sha256": metadata["manifest_sha256"],
            "samples": len(records),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "jiwer": getattr(
                jiwer,
                "__version__",
                "unknown",
            ),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "languages": language_results,
        "macro_wer": macro_wer,
        "overall_corpus_wer": overall_metrics["wer"],
        "overall": overall_metrics,
    }


def print_summary(summary: dict) -> None:
    print()
    print("=" * 70)
    print("WER RESULTS")
    print("=" * 70)

    for language in LANGUAGE_LOCALES:
        result = summary["languages"][language]

        print(
            f"{language.upper():2} | "
            f"WER {result['wer'] * 100:6.2f}% | "
            f"{result['samples']:3} samples | "
            f"{result['duration_minutes']:6.2f} min"
        )

    print("-" * 70)

    print(f"Macro-average WER : {summary['macro_wer'] * 100:.2f}%")

    print(f"Overall corpus WER: {summary['overall_corpus_wer'] * 100:.2f}%")

    print("=" * 70)


def main() -> None:
    metadata = load_metadata()
    entries = load_manifest()

    validate_manifest(
        entries,
        metadata,
    )

    processor, model = load_model()

    run_inference(
        entries=entries,
        metadata=metadata,
        processor=processor,
        model=model,
    )

    records = load_all_predictions(
        metadata,
    )

    summary = build_summary(
        records,
        metadata,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(summary)

    print()
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(f"Summary     : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
