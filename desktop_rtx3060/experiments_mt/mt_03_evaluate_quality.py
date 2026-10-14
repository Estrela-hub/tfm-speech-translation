from __future__ import annotations

import csv
import gc
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

EXPECTED_SAMPLE_COUNT = 200

EXPECTED_MANIFEST_SHA256 = "b2e3f73365232c700a1246b7bf4726eba2401f743a6a644066c5f35d4c470399"

EXPECTED_DATASET_REVISION = "71abf77d8b7beb5cfef59898d6b24d92ab7654fc"

MAX_GENERATION_LENGTH = 512


# ---------------------------------------------------------------------------
# Model revisions fixed from mt_01_smoke_test_models.py
# ---------------------------------------------------------------------------

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

M2M100_MODEL_ID = "facebook/m2m100_418M"
M2M100_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGUAGES = ["EN", "ES", "FR", "DE", "PT"]

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

M2M100_LANGUAGE_CODES = {
    "EN": "en",
    "ES": "es",
    "FR": "fr",
    "DE": "de",
    "PT": "pt",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "mt_selection" / "flores_dev_200.jsonl"

DATASET_METADATA_PATH = PROJECT_ROOT / "data" / "mt_selection" / "flores_dev_200_metadata.json"

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "quality"

SCORES_CSV_PATH = RESULTS_DIR / "quality_scores.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "quality_summary.json"


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_and_validate_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found:\n{MANIFEST_PATH}")

    if not DATASET_METADATA_PATH.exists():
        raise FileNotFoundError(f"Dataset metadata not found:\n{DATASET_METADATA_PATH}")

    actual_sha256 = calculate_sha256(MANIFEST_PATH)

    if actual_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Actual  : {actual_sha256}\n\n"
            "The MT quality evaluation must use exactly the dataset "
            "validated during mt_02_prepare_flores_subset.py."
        )

    with DATASET_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    actual_revision = metadata.get("dataset_revision")

    if actual_revision != EXPECTED_DATASET_REVISION:
        raise RuntimeError(
            f"FLORES revision mismatch.\nExpected: {EXPECTED_DATASET_REVISION}\nActual  : {actual_revision}"
        )

    records = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if "id" not in record:
                raise RuntimeError(f"Missing ID on manifest line {line_number}.")

            for language_code in FLORES_LANGUAGE_CODES.values():
                if language_code not in record:
                    raise RuntimeError(f"Missing language {language_code} on manifest line {line_number}.")

                sentence = record[language_code]

                if not isinstance(sentence, str) or not sentence.strip():
                    raise RuntimeError(f"Empty sentence for {language_code} on manifest line {line_number}.")

            records.append(record)

    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLE_COUNT} manifest records, found {len(records)}.")

    ids = [record["id"] for record in records]

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate sentence IDs found in the manifest.")

    return records


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translate_nllb(
    model,
    tokenizer,
    source_text: str,
    source_language: str,
    target_language: str,
) -> str:
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

    return tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]


def translate_m2m100(
    model,
    tokenizer,
    source_text: str,
    source_language: str,
    target_language: str,
) -> str:
    tokenizer.src_lang = M2M100_LANGUAGE_CODES[source_language]

    inputs = tokenizer(
        source_text,
        return_tensors="pt",
    ).to(DEVICE)

    target_token_id = tokenizer.get_lang_id(M2M100_LANGUAGE_CODES[target_language])

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_token_id,
        num_beams=1,
        do_sample=False,
        max_length=MAX_GENERATION_LENGTH,
    )

    return tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model_name: str,
    model_id: str,
    revision: str,
    records: list[dict],
    bleu_metric: BLEU,
    chrf_metric: CHRF,
    csv_writer,
) -> dict:
    print()
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    print(f"Model ID : {model_id}")
    print(f"Revision : {revision}")
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
    )

    print("Loading model...")

    # The pinned revisions use the original PyTorch checkpoint.
    # Explicitly disabling safetensors avoids automatic background conversion
    # or download activity during the experiment.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        revision=revision,
        use_safetensors=False,
    )

    model.to(DEVICE)
    model.eval()

    model_device = str(next(model.parameters()).device)
    model_dtype = str(next(model.parameters()).dtype)

    print(f"Model device: {model_device}")
    print(f"Model dtype : {model_dtype}")

    if model_device != "cuda:0":
        raise RuntimeError(f"{model_name} is not running on cuda:0.")

    if model_dtype != "torch.float32":
        raise RuntimeError(f"{model_name} is not running in FP32. Detected dtype: {model_dtype}")

    predictions_path = RESULTS_DIR / f"{model_name.lower()}_predictions.jsonl"

    direction_scores = []

    total_directions = len(LANGUAGES) * (len(LANGUAGES) - 1)
    direction_number = 0

    with predictions_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as predictions_file:
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

                    for sample_index, record in enumerate(
                        records,
                        start=1,
                    ):
                        source_text = record[source_code]
                        reference_text = record[target_code]

                        if model_name == "NLLB":
                            hypothesis = translate_nllb(
                                model=model,
                                tokenizer=tokenizer,
                                source_text=source_text,
                                source_language=source_language,
                                target_language=target_language,
                            )

                        elif model_name == "M2M100":
                            hypothesis = translate_m2m100(
                                model=model,
                                tokenizer=tokenizer,
                                source_text=source_text,
                                source_language=source_language,
                                target_language=target_language,
                            )

                        else:
                            raise RuntimeError(f"Unknown model name: {model_name}")

                        hypotheses.append(hypothesis)
                        references.append(reference_text)

                        prediction_record = {
                            "model": model_name,
                            "model_id": model_id,
                            "revision": revision,
                            "id": record["id"],
                            "source_language": source_language,
                            "target_language": target_language,
                            "source": source_text,
                            "reference": reference_text,
                            "hypothesis": hypothesis,
                        }

                        predictions_file.write(
                            json.dumps(
                                prediction_record,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )

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

                    score_record = {
                        "model": model_name,
                        "model_id": model_id,
                        "revision": revision,
                        "source_language": source_language,
                        "target_language": target_language,
                        "samples": len(records),
                        "spbleu": bleu_score,
                        "chrf_pp": chrf_score,
                    }

                    direction_scores.append(score_record)

                    csv_writer.writerow(score_record)

                    print(f"  spBLEU : {bleu_score:.2f}")
                    print(f"  chrF++ : {chrf_score:.2f}")

    macro_spbleu = sum(row["spbleu"] for row in direction_scores) / len(direction_scores)

    macro_chrf_pp = sum(row["chrf_pp"] for row in direction_scores) / len(direction_scores)

    worst_bleu = min(
        direction_scores,
        key=lambda row: row["spbleu"],
    )

    worst_chrf = min(
        direction_scores,
        key=lambda row: row["chrf_pp"],
    )

    print()
    print("-" * 80)
    print(f"{model_name} SUMMARY")
    print("-" * 80)
    print(f"Macro spBLEU : {macro_spbleu:.2f}")
    print(f"Macro chrF++ : {macro_chrf_pp:.2f}")
    print(
        "Worst spBLEU : "
        f"{worst_bleu['source_language']} -> "
        f"{worst_bleu['target_language']} "
        f"({worst_bleu['spbleu']:.2f})"
    )
    print(
        "Worst chrF++ : "
        f"{worst_chrf['source_language']} -> "
        f"{worst_chrf['target_language']} "
        f"({worst_chrf['chrf_pp']:.2f})"
    )

    result = {
        "model": model_name,
        "model_id": model_id,
        "revision": revision,
        "device": model_device,
        "dtype": model_dtype,
        "directions": direction_scores,
        "macro_spbleu": macro_spbleu,
        "macro_chrf_pp": macro_chrf_pp,
        "worst_spbleu": worst_bleu,
        "worst_chrf_pp": worst_chrf,
        "predictions_file": predictions_path.name,
    }

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("MT QUALITY EVALUATION")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"Transformers : {transformers.__version__}")
    print(f"SacreBLEU    : {sacrebleu.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. The MT quality experiment requires CUDA.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print()

    print("Validating fixed FLORES manifest...")

    records = load_and_validate_manifest()

    print(f"Manifest       : {MANIFEST_PATH}")
    print(f"Samples        : {len(records)}")
    print(f"SHA256         : {EXPECTED_MANIFEST_SHA256}")
    print(f"Dataset rev.   : {EXPECTED_DATASET_REVISION}")
    print("Manifest status: PASS")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # spBLEU:
    # FLORES-200 SentencePiece tokenizer.
    bleu_metric = BLEU(
        tokenize="flores200",
    )

    # chrF++:
    # character n-grams + word n-grams up to order 2.
    chrf_metric = CHRF(
        word_order=2,
    )

    results = []

    with SCORES_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = [
            "model",
            "model_id",
            "revision",
            "source_language",
            "target_language",
            "samples",
            "spbleu",
            "chrf_pp",
        ]

        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        csv_writer.writeheader()

        nllb_result = evaluate_model(
            model_name="NLLB",
            model_id=NLLB_MODEL_ID,
            revision=NLLB_REVISION,
            records=records,
            bleu_metric=bleu_metric,
            chrf_metric=chrf_metric,
            csv_writer=csv_writer,
        )

        csv_file.flush()
        results.append(nllb_result)

        m2m100_result = evaluate_model(
            model_name="M2M100",
            model_id=M2M100_MODEL_ID,
            revision=M2M100_REVISION,
            records=records,
            bleu_metric=bleu_metric,
            chrf_metric=chrf_metric,
            csv_writer=csv_writer,
        )

        csv_file.flush()
        results.append(m2m100_result)

    bleu_signature = str(bleu_metric.get_signature())

    chrf_signature = str(chrf_metric.get_signature())

    summary = {
        "experiment": "MT model selection - quality",
        "dataset": {
            "dataset_id": "facebook/flores",
            "revision": EXPECTED_DATASET_REVISION,
            "split": "dev",
            "samples": EXPECTED_SAMPLE_COUNT,
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "languages": FLORES_LANGUAGE_CODES,
            "translation_directions": len(LANGUAGES) * (len(LANGUAGES) - 1),
        },
        "generation": {
            "device": DEVICE,
            "dtype": "float32",
            "batch_size": 1,
            "num_beams": 1,
            "do_sample": False,
            "max_length": MAX_GENERATION_LENGTH,
        },
        "metrics": {
            "spbleu_signature": bleu_signature,
            "chrf_pp_signature": chrf_signature,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "sacrebleu": sacrebleu.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "models": results,
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
    print("FINAL COMPARISON")
    print("=" * 80)

    for result in results:
        print(
            f"{result['model']:<8} "
            f"Macro spBLEU = {result['macro_spbleu']:.2f} | "
            f"Macro chrF++ = {result['macro_chrf_pp']:.2f}"
        )

    print()
    print(f"BLEU signature : {bleu_signature}")
    print(f"chrF signature : {chrf_signature}")

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Scores      : {SCORES_CSV_PATH}")
    print(f"Summary     : {SUMMARY_JSON_PATH}")
    print(f"NLLB preds  : {RESULTS_DIR / 'nllb_predictions.jsonl'}")
    print(f"M2M100 preds: {RESULTS_DIR / 'm2m100_predictions.jsonl'}")

    print()
    print("=" * 80)
    print("MT QUALITY EVALUATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
