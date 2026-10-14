from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import sacrebleu
import torch
import transformers
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore


# ---------------------------------------------------------------------------
# Fixed experimental configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda"
DTYPE = torch.float16

EXPECTED_SAMPLE_COUNT = 200

EXPECTED_MANIFEST_SHA256 = "b2e3f73365232c700a1246b7bf4726eba2401f743a6a644066c5f35d4c470399"

EXPECTED_DATASET_REVISION = "71abf77d8b7beb5cfef59898d6b24d92ab7654fc"

MAX_GENERATION_LENGTH = 512

MAX_RECORDED_GENERATED_TOKENS = MAX_GENERATION_LENGTH - 1
NEAR_MAX_THRESHOLD = 500


# ---------------------------------------------------------------------------
# NLLB
# ---------------------------------------------------------------------------

MODEL_NAME = "NLLB"

MODEL_ID = "facebook/nllb-200-distilled-600M"

MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGUAGES = ["EN", "ES", "FR", "DE", "PT"]

EXPECTED_TRANSLATION_COUNT = EXPECTED_SAMPLE_COUNT * len(LANGUAGES) * (len(LANGUAGES) - 1)

FLORES_LANGUAGE_CODES = {
    "EN": "eng_Latn",
    "ES": "spa_Latn",
    "FR": "fra_Latn",
    "DE": "deu_Latn",
    "PT": "por_Latn",
}

NLLB_LANGUAGE_CODES = {
    "EN": "eng_Latn",
    "ES": "spa_Latn",
    "FR": "fra_Latn",
    "DE": "deu_Latn",
    "PT": "por_Latn",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "mt_selection" / "flores_dev_200.jsonl"

DATASET_METADATA_PATH = PROJECT_ROOT / "data" / "mt_selection" / "flores_dev_200_metadata.json"

FP32_RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "quality"

FP32_PREDICTIONS_PATH = FP32_RESULTS_DIR / "nllb_predictions.jsonl"

FP32_SUMMARY_PATH = FP32_RESULTS_DIR / "quality_summary.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "quality_fp16"

FP16_PREDICTIONS_PATH = RESULTS_DIR / "nllb_fp16_predictions.jsonl"

SCORES_CSV_PATH = RESULTS_DIR / "quality_scores.csv"

CHANGED_PREDICTIONS_CSV_PATH = RESULTS_DIR / "changed_predictions.csv"

SUMMARY_JSON_PATH = RESULTS_DIR / "quality_summary.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def load_and_validate_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found:\n{MANIFEST_PATH}")

    if not DATASET_METADATA_PATH.exists():
        raise FileNotFoundError(f"Dataset metadata not found:\n{DATASET_METADATA_PATH}")

    actual_sha256 = calculate_sha256(MANIFEST_PATH)

    if actual_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest SHA256 mismatch.\nExpected: {EXPECTED_MANIFEST_SHA256}\nActual  : {actual_sha256}"
        )

    with DATASET_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    actual_revision = metadata.get("dataset_revision")

    if actual_revision != EXPECTED_DATASET_REVISION:
        raise RuntimeError(
            f"Dataset revision mismatch.\nExpected: {EXPECTED_DATASET_REVISION}\nActual  : {actual_revision}"
        )

    records = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if "id" not in record:
                raise RuntimeError("Manifest record without ID.")

            for language_code in FLORES_LANGUAGE_CODES.values():
                if language_code not in record:
                    raise RuntimeError(f"Missing language {language_code} for ID {record['id']}.")

            records.append(record)

    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLE_COUNT} manifest records, found {len(records)}.")

    return records


# ---------------------------------------------------------------------------
# FP32 baseline
# ---------------------------------------------------------------------------


def load_fp32_baseline() -> tuple[
    dict[tuple[int, str, str], dict],
    dict,
]:
    if not FP32_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"FP32 NLLB predictions not found:\n{FP32_PREDICTIONS_PATH}")

    if not FP32_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"FP32 quality summary not found:\n{FP32_SUMMARY_PATH}")

    predictions = {}

    with FP32_PREDICTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record["model"] != MODEL_NAME:
                continue

            key = (
                int(record["id"]),
                record["source_language"],
                record["target_language"],
            )

            if key in predictions:
                raise RuntimeError(f"Duplicate FP32 prediction: {key}")

            predictions[key] = record

    if len(predictions) != EXPECTED_TRANSLATION_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_TRANSLATION_COUNT} NLLB FP32 predictions, found {len(predictions)}.")

    with FP32_SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    dataset = summary.get("dataset", {})

    if dataset.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("FP32 quality baseline used a different manifest SHA256.")

    nllb_result = None

    for model_result in summary.get(
        "models",
        [],
    ):
        if model_result.get("model") == MODEL_NAME:
            nllb_result = model_result
            break

    if nllb_result is None:
        raise RuntimeError("NLLB result not found in FP32 summary.")

    if nllb_result.get("revision") != MODEL_REVISION:
        raise RuntimeError("FP32 NLLB revision does not match the FP16 experiment revision.")

    return predictions, nllb_result


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translate(
    model,
    tokenizer,
    source_text: str,
    source_language: str,
    target_language: str,
) -> tuple[str, int]:
    tokenizer.src_lang = NLLB_LANGUAGE_CODES[source_language]

    inputs = tokenizer(
        source_text,
        return_tensors="pt",
    ).to(DEVICE)

    target_token_id = tokenizer.convert_tokens_to_ids(NLLB_LANGUAGE_CODES[target_language])

    if target_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Unknown NLLB target language token: {NLLB_LANGUAGE_CODES[target_language]}")

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_token_id,
        num_beams=1,
        do_sample=False,
        max_length=MAX_GENERATION_LENGTH,
    )

    generated_token_count = max(
        int(generated_tokens.shape[-1]) - 1,
        0,
    )

    hypothesis = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

    return (
        hypothesis,
        generated_token_count,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_fp16(
    records: list[dict],
    fp32_predictions: dict,
    bleu_metric: BLEU,
    chrf_metric: CHRF,
) -> dict:
    print()
    print("=" * 80)
    print("MODEL: NLLB FP16")
    print("=" * 80)

    print(f"Model ID : {MODEL_ID}")
    print(f"Revision : {MODEL_REVISION}")

    print("Loading tokenizer from local cache...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )

    print("Loading model directly in FP16 from local cache...")

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

    print(f"Model device: {model_device}")
    print(f"Model dtype : {model_dtype}")

    if model_device != "cuda:0":
        raise RuntimeError("NLLB is not running on cuda:0.")

    if model_dtype != torch.float16:
        raise RuntimeError(f"NLLB is not FP16: {model_dtype}")

    direction_scores = []

    total_translations = 0
    exact_same = 0
    changed = 0
    near_max_length = 0
    reached_max_length = 0

    changed_records = []

    score_fieldnames = [
        "model",
        "precision",
        "source_language",
        "target_language",
        "samples",
        "spbleu",
        "chrf_pp",
        "fp32_exact_same",
        "fp32_changed",
    ]

    with (
        FP16_PREDICTIONS_PATH.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as predictions_file,
        SCORES_CSV_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as score_file,
    ):
        score_writer = csv.DictWriter(
            score_file,
            fieldnames=score_fieldnames,
        )

        score_writer.writeheader()

        direction_number = 0

        total_directions = len(LANGUAGES) * (len(LANGUAGES) - 1)

        with torch.inference_mode():
            for source_language in LANGUAGES:
                for target_language in LANGUAGES:
                    if source_language == target_language:
                        continue

                    direction_number += 1

                    print()
                    print(f"[{direction_number:02d}/{total_directions}] {source_language} -> {target_language}")

                    source_code = FLORES_LANGUAGE_CODES[source_language]

                    target_code = FLORES_LANGUAGE_CODES[target_language]

                    hypotheses = []
                    references = []

                    direction_exact_same = 0
                    direction_changed = 0

                    for sample_index, record in enumerate(
                        records,
                        start=1,
                    ):
                        sample_id = int(record["id"])

                        source_text = record[source_code]

                        reference_text = record[target_code]

                        (
                            hypothesis,
                            generated_token_count,
                        ) = translate(
                            model=model,
                            tokenizer=tokenizer,
                            source_text=source_text,
                            source_language=(source_language),
                            target_language=(target_language),
                        )

                        if not hypothesis.strip():
                            raise RuntimeError(
                                f"Empty FP16 translation for ID {sample_id} {source_language}->{target_language}."
                            )

                        key = (
                            sample_id,
                            source_language,
                            target_language,
                        )

                        if key not in fp32_predictions:
                            raise RuntimeError(f"No matching FP32 prediction for {key}.")

                        fp32_hypothesis = fp32_predictions[key]["hypothesis"]

                        is_exact_same = hypothesis == fp32_hypothesis

                        if is_exact_same:
                            exact_same += 1
                            direction_exact_same += 1
                        else:
                            changed += 1
                            direction_changed += 1

                            changed_records.append(
                                {
                                    "id": sample_id,
                                    "source_language": (source_language),
                                    "target_language": (target_language),
                                    "source": source_text,
                                    "reference": (reference_text),
                                    "fp32_hypothesis": (fp32_hypothesis),
                                    "fp16_hypothesis": (hypothesis),
                                    "fp16_generated_tokens": (generated_token_count),
                                }
                            )

                        if generated_token_count >= NEAR_MAX_THRESHOLD:
                            near_max_length += 1

                        if generated_token_count >= MAX_RECORDED_GENERATED_TOKENS:
                            reached_max_length += 1

                        hypotheses.append(hypothesis)

                        references.append(reference_text)

                        prediction_record = {
                            "model": MODEL_NAME,
                            "precision": "float16",
                            "model_id": MODEL_ID,
                            "revision": MODEL_REVISION,
                            "id": sample_id,
                            "source_language": (source_language),
                            "target_language": (target_language),
                            "source": source_text,
                            "reference": (reference_text),
                            "hypothesis": hypothesis,
                            "generated_tokens": (generated_token_count),
                            "exact_same_as_fp32": (is_exact_same),
                        }

                        predictions_file.write(
                            json.dumps(
                                prediction_record,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )

                        total_translations += 1

                        if sample_index % 50 == 0 or sample_index == len(records):
                            print(f"  Translated: {sample_index}/{len(records)}")

                    predictions_file.flush()

                    bleu_score = bleu_metric.corpus_score(
                        hypotheses,
                        [references],
                    ).score

                    chrf_score = chrf_metric.corpus_score(
                        hypotheses,
                        [references],
                    ).score

                    direction_result = {
                        "model": MODEL_NAME,
                        "precision": "float16",
                        "source_language": (source_language),
                        "target_language": (target_language),
                        "samples": len(records),
                        "spbleu": bleu_score,
                        "chrf_pp": chrf_score,
                        "fp32_exact_same": (direction_exact_same),
                        "fp32_changed": (direction_changed),
                    }

                    direction_scores.append(direction_result)

                    score_writer.writerow(direction_result)

                    score_file.flush()

                    print(f"  spBLEU      : {bleu_score:.2f}")

                    print(f"  chrF++      : {chrf_score:.2f}")

                    print(f"  Same as FP32: {direction_exact_same}/{len(records)}")

    if total_translations != EXPECTED_TRANSLATION_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_TRANSLATION_COUNT} translations, completed {total_translations}.")

    macro_spbleu = sum(row["spbleu"] for row in direction_scores) / len(direction_scores)

    macro_chrf_pp = sum(row["chrf_pp"] for row in direction_scores) / len(direction_scores)

    worst_spbleu = min(
        direction_scores,
        key=lambda row: row["spbleu"],
    )

    worst_chrf_pp = min(
        direction_scores,
        key=lambda row: row["chrf_pp"],
    )

    return {
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": model_device,
        "dtype": str(model_dtype),
        "directions": direction_scores,
        "macro_spbleu": macro_spbleu,
        "macro_chrf_pp": macro_chrf_pp,
        "worst_spbleu": worst_spbleu,
        "worst_chrf_pp": worst_chrf_pp,
        "total_translations": (total_translations),
        "exact_same_as_fp32": exact_same,
        "changed_from_fp32": changed,
        "exact_same_percent": (exact_same / total_translations * 100.0),
        "near_max_length": (near_max_length),
        "reached_max_length": (reached_max_length),
        "changed_records": (changed_records),
    }


# ---------------------------------------------------------------------------
# Changed predictions output
# ---------------------------------------------------------------------------


def write_changed_predictions(
    changed_records: list[dict],
) -> None:
    fieldnames = [
        "id",
        "source_language",
        "target_language",
        "fp16_generated_tokens",
        "source",
        "reference",
        "fp32_hypothesis",
        "fp16_hypothesis",
    ]

    with CHANGED_PREDICTIONS_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in changed_records:
            writer.writerow(record)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("MT NLLB FP16 QUALITY EVALUATION")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")

    print(f"PyTorch      : {torch.__version__}")

    print(f"Transformers : {transformers.__version__}")

    print(f"SacreBLEU    : {sacrebleu.__version__}")

    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    print()
    print("Validating fixed FLORES manifest...")

    records = load_and_validate_manifest()

    print(f"Samples      : {len(records)}")
    print(f"SHA256       : {EXPECTED_MANIFEST_SHA256}")
    print("Manifest     : PASS")

    print()
    print("Loading validated FP32 baseline...")

    (
        fp32_predictions,
        fp32_result,
    ) = load_fp32_baseline()

    print(f"FP32 predictions : {len(fp32_predictions)}")

    print(f"FP32 Macro spBLEU: {fp32_result['macro_spbleu']:.2f}")

    print(f"FP32 Macro chrF++: {fp32_result['macro_chrf_pp']:.2f}")

    print("FP32 baseline     : PASS")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same metrics as mt_03_evaluate_quality.py.
    bleu_metric = BLEU(
        tokenize="flores200",
    )

    chrf_metric = CHRF(
        word_order=2,
    )

    fp16_result = evaluate_fp16(
        records=records,
        fp32_predictions=fp32_predictions,
        bleu_metric=bleu_metric,
        chrf_metric=chrf_metric,
    )

    write_changed_predictions(fp16_result["changed_records"])

    fp32_spbleu = float(fp32_result["macro_spbleu"])

    fp32_chrf_pp = float(fp32_result["macro_chrf_pp"])

    fp16_spbleu = float(fp16_result["macro_spbleu"])

    fp16_chrf_pp = float(fp16_result["macro_chrf_pp"])

    spbleu_delta = fp16_spbleu - fp32_spbleu

    chrf_delta = fp16_chrf_pp - fp32_chrf_pp

    bleu_signature = str(bleu_metric.get_signature())

    chrf_signature = str(chrf_metric.get_signature())

    # Do not duplicate all changed translations
    # inside the summary JSON.
    compact_fp16_result = {key: value for key, value in fp16_result.items() if key != "changed_records"}

    summary = {
        "experiment": ("NLLB FP32 vs FP16 quality comparison"),
        "dataset": {
            "dataset_id": "facebook/flores",
            "revision": (EXPECTED_DATASET_REVISION),
            "split": "dev",
            "samples": (EXPECTED_SAMPLE_COUNT),
            "manifest_sha256": (EXPECTED_MANIFEST_SHA256),
            "translation_directions": len(LANGUAGES) * (len(LANGUAGES) - 1),
        },
        "generation": {
            "batch_size": 1,
            "num_beams": 1,
            "do_sample": False,
            "max_length": (MAX_GENERATION_LENGTH),
        },
        "metrics": {
            "spbleu_signature": (bleu_signature),
            "chrf_pp_signature": (chrf_signature),
        },
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "transformers": (transformers.__version__),
            "sacrebleu": (sacrebleu.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
        },
        "fp32": {
            "dtype": "float32",
            "macro_spbleu": (fp32_spbleu),
            "macro_chrf_pp": (fp32_chrf_pp),
        },
        "fp16": compact_fp16_result,
        "delta_fp16_minus_fp32": {
            "macro_spbleu": (spbleu_delta),
            "macro_chrf_pp": (chrf_delta),
        },
    }

    SUMMARY_JSON_PATH.write_text(
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

    print()
    print("=" * 80)
    print("FP32 vs FP16 QUALITY COMPARISON")
    print("=" * 80)

    print()
    print("FP32")
    print(f"  Macro spBLEU : {fp32_spbleu:.2f}")
    print(f"  Macro chrF++ : {fp32_chrf_pp:.2f}")

    print()
    print("FP16")
    print(f"  Macro spBLEU : {fp16_spbleu:.2f}")
    print(f"  Macro chrF++ : {fp16_chrf_pp:.2f}")

    print()
    print("FP16 - FP32")
    print(f"  Δ spBLEU     : {spbleu_delta:+.4f}")
    print(f"  Δ chrF++     : {chrf_delta:+.4f}")

    print()
    print(
        f"Exact same hypotheses : "
        f"{fp16_result['exact_same_as_fp32']}/"
        f"{EXPECTED_TRANSLATION_COUNT} "
        f"({fp16_result['exact_same_percent']:.2f}%)"
    )

    print(f"Changed hypotheses    : {fp16_result['changed_from_fp32']}/{EXPECTED_TRANSLATION_COUNT}")

    print(f"Near max length       : {fp16_result['near_max_length']}")

    print(f"Reached max length    : {fp16_result['reached_max_length']}")

    print()
    print(
        f"Worst FP16 spBLEU     : "
        f"{fp16_result['worst_spbleu']['source_language']} -> "
        f"{fp16_result['worst_spbleu']['target_language']} "
        f"({fp16_result['worst_spbleu']['spbleu']:.2f})"
    )

    print(
        f"Worst FP16 chrF++     : "
        f"{fp16_result['worst_chrf_pp']['source_language']} -> "
        f"{fp16_result['worst_chrf_pp']['target_language']} "
        f"({fp16_result['worst_chrf_pp']['chrf_pp']:.2f})"
    )

    print()
    print(f"BLEU signature : {bleu_signature}")

    print(f"chrF signature : {chrf_signature}")

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Predictions : {FP16_PREDICTIONS_PATH}")

    print(f"Scores      : {SCORES_CSV_PATH}")

    print(f"Changes     : {CHANGED_PREDICTIONS_CSV_PATH}")

    print(f"Summary     : {SUMMARY_JSON_PATH}")

    print()
    print("=" * 80)
    print("NLLB FP16 QUALITY EVALUATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
