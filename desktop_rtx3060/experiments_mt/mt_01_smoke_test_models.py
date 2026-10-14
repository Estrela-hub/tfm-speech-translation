from __future__ import annotations

import gc
import sys

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Force UTF-8 output on Windows so accents are not corrupted in the console/logs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore


DEVICE = "cuda"
MAX_NEW_TOKENS = 64

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

M2M100_MODEL_ID = "facebook/m2m100_418M"
M2M100_MODEL_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"


LANGUAGES = {
    "EN": "English",
    "ES": "Spanish",
    "FR": "French",
    "DE": "German",
    "PT": "Portuguese",
}


# Same meaning in all five languages.
# These sentences are only used for functional validation.
# They are NOT part of the later quality benchmark.
TEST_SENTENCES = {
    "EN": "The meeting starts tomorrow at nine in the morning.",
    "ES": "La reunión empieza mañana a las nueve de la mañana.",
    "FR": "La réunion commence demain à neuf heures du matin.",
    "DE": "Das Treffen beginnt morgen früh um neun Uhr.",
    "PT": "A reunião começa amanhã às nove da manhã.",
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


def print_environment() -> None:
    print("=" * 80)
    print("MT MODEL SMOKE TEST")
    print("=" * 80)

    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Transformers  : {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This smoke test intentionally requires CUDA "
            "so that a silent CPU fallback cannot be mistaken for a valid GPU run."
        )

    print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
    print()


def run_nllb() -> None:
    print("=" * 80)
    print("MODEL: NLLB-200 DISTILLED 600M")
    print("=" * 80)

    revision = NLLB_MODEL_REVISION

    print(f"Model ID : {NLLB_MODEL_ID}")
    print(f"Revision : {revision}")
    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        NLLB_MODEL_ID,
        revision=revision,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_ID,
        revision=revision,
    )

    model.to(DEVICE)
    model.eval()

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype : {next(model.parameters()).dtype}")
    print()

    translations = 0

    with torch.inference_mode():
        for src_lang in LANGUAGES:
            source_text = TEST_SENTENCES[src_lang]

            tokenizer.src_lang = NLLB_LANGUAGE_CODES[src_lang]

            print(f"SOURCE [{src_lang}] {source_text}")

            for tgt_lang in LANGUAGES:
                if tgt_lang == src_lang:
                    continue

                inputs = tokenizer(
                    source_text,
                    return_tensors="pt",
                ).to(DEVICE)

                target_token_id = tokenizer.convert_tokens_to_ids(NLLB_LANGUAGE_CODES[tgt_lang])

                if target_token_id == tokenizer.unk_token_id:
                    raise RuntimeError(f"Unknown NLLB target language token: {NLLB_LANGUAGE_CODES[tgt_lang]}")

                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=target_token_id,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                translation = tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )[0]

                print(f"  {src_lang} -> {tgt_lang}: {translation}")

                translations += 1

            print()

    expected_translations = len(LANGUAGES) * (len(LANGUAGES) - 1)

    print(f"Completed translations: {translations}/{expected_translations}")

    if translations != expected_translations:
        raise RuntimeError(f"Expected {expected_translations} translations, got {translations}.")

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    print("NLLB smoke test completed successfully.")
    print()


def run_m2m100() -> None:
    print("=" * 80)
    print("MODEL: M2M100 418M")
    print("=" * 80)

    revision = M2M100_MODEL_REVISION

    print(f"Model ID : {M2M100_MODEL_ID}")
    print(f"Revision : {revision}")
    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        M2M100_MODEL_ID,
        revision=revision,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        M2M100_MODEL_ID,
        revision=revision,
    )

    model.to(DEVICE)
    model.eval()

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype : {next(model.parameters()).dtype}")
    print()

    translations = 0

    with torch.inference_mode():
        for src_lang in LANGUAGES:
            source_text = TEST_SENTENCES[src_lang]

            tokenizer.src_lang = M2M100_LANGUAGE_CODES[src_lang]

            print(f"SOURCE [{src_lang}] {source_text}")

            for tgt_lang in LANGUAGES:
                if tgt_lang == src_lang:
                    continue

                inputs = tokenizer(
                    source_text,
                    return_tensors="pt",
                ).to(DEVICE)

                target_token_id = tokenizer.get_lang_id(M2M100_LANGUAGE_CODES[tgt_lang])

                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=target_token_id,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                translation = tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )[0]

                print(f"  {src_lang} -> {tgt_lang}: {translation}")

                translations += 1

            print()

    expected_translations = len(LANGUAGES) * (len(LANGUAGES) - 1)

    print(f"Completed translations: {translations}/{expected_translations}")

    if translations != expected_translations:
        raise RuntimeError(f"Expected {expected_translations} translations, got {translations}.")

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    print("M2M100 smoke test completed successfully.")
    print()


def main() -> None:
    print_environment()

    run_nllb()
    run_m2m100()

    print("=" * 80)
    print("SMOKE TEST PASSED")
    print("Both models completed all 20 translation directions on CUDA.")
    print("=" * 80)


if __name__ == "__main__":
    main()
