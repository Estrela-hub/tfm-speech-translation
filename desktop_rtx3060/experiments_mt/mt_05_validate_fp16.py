from __future__ import annotations

import gc
import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda"
DTYPE = torch.float16
MAX_LENGTH = 64


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

M2M100_MODEL_ID = "facebook/m2m100_418M"
M2M100_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGUAGES = ["EN", "ES", "FR", "DE", "PT"]

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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "mt_selection" / "fp16"

RESULT_PATH = RESULTS_DIR / "fp16_validation.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bytes_to_mb(value: int | float) -> float:
    return value / (1024**2)


def parameter_memory_mb(model) -> float:
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    return bytes_to_mb(total_bytes)


def set_source_language(
    model_name: str,
    tokenizer,
    source_language: str,
) -> None:
    if model_name == "NLLB":
        tokenizer.src_lang = NLLB_LANGUAGE_CODES[source_language]

    elif model_name == "M2M100":
        tokenizer.src_lang = M2M100_LANGUAGE_CODES[source_language]

    else:
        raise RuntimeError(f"Unknown model: {model_name}")


def get_target_token_id(
    model_name: str,
    tokenizer,
    target_language: str,
) -> int:
    if model_name == "NLLB":
        token_id = tokenizer.convert_tokens_to_ids(NLLB_LANGUAGE_CODES[target_language])

        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"Unknown NLLB target language token: {NLLB_LANGUAGE_CODES[target_language]}")

        return token_id

    if model_name == "M2M100":
        return tokenizer.get_lang_id(M2M100_LANGUAGE_CODES[target_language])

    raise RuntimeError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Numerical validation
# ---------------------------------------------------------------------------


def validate_finite_logits(
    model_name: str,
    model,
    tokenizer,
    source_language: str,
    source_text: str,
) -> None:
    """
    Run one encoder/decoder forward pass and verify that the produced logits
    are finite.

    This is not a quality test. Its only purpose is to detect obvious FP16
    numerical instability such as NaN or Inf values.
    """

    set_source_language(
        model_name,
        tokenizer,
        source_language,
    )

    inputs = tokenizer(
        source_text,
        return_tensors="pt",
    ).to(DEVICE)

    decoder_start_token_id = model.config.decoder_start_token_id

    if decoder_start_token_id is None:
        raise RuntimeError(f"{model_name} has no decoder_start_token_id.")

    decoder_input_ids = torch.tensor(
        [[decoder_start_token_id]],
        dtype=torch.long,
        device=DEVICE,
    )

    outputs = model(
        **inputs,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits

    if not torch.isfinite(logits).all().item():
        nan_count = torch.isnan(logits).sum().item()
        inf_count = torch.isinf(logits).sum().item()

        raise RuntimeError(
            f"Non-finite FP16 logits detected in {model_name} "
            f"for source language {source_language}.\n"
            f"NaN values: {nan_count}\n"
            f"Inf values: {inf_count}"
        )


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translate(
    model_name: str,
    model,
    tokenizer,
    source_text: str,
    source_language: str,
    target_language: str,
) -> str:
    set_source_language(
        model_name,
        tokenizer,
        source_language,
    )

    inputs = tokenizer(
        source_text,
        return_tensors="pt",
    ).to(DEVICE)

    target_token_id = get_target_token_id(
        model_name,
        tokenizer,
        target_language,
    )

    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_token_id,
        num_beams=1,
        do_sample=False,
        max_length=MAX_LENGTH,
    )

    translation = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]

    return translation


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def validate_model(
    model_name: str,
    model_id: str,
    revision: str,
) -> dict:
    print()
    print("=" * 80)
    print(f"MODEL: {model_name}")
    print("=" * 80)

    print(f"Model ID : {model_id}")
    print(f"Revision : {revision}")
    print("Loading tokenizer from local cache...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
    )

    print("Loading model directly in FP16...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=DTYPE,
        use_safetensors=False,
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()

    torch.cuda.synchronize()

    device = str(next(model.parameters()).device)
    dtype = next(model.parameters()).dtype

    parameters = sum(parameter.numel() for parameter in model.parameters())

    parameter_mb = parameter_memory_mb(model)

    floating_parameter_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}

    print(f"Device           : {device}")
    print(f"Dtype            : {dtype}")
    print(f"Parameters       : {parameters:,}")
    print(f"Parameter memory : {parameter_mb:.2f} MB")

    print("Floating parameter dtypes: " + ", ".join(sorted(str(value) for value in floating_parameter_dtypes)))

    if device != "cuda:0":
        raise RuntimeError(f"{model_name} is not running on cuda:0.")

    if dtype != torch.float16:
        raise RuntimeError(f"{model_name} was not loaded in FP16. Detected dtype: {dtype}")

    if floating_parameter_dtypes != {torch.float16}:
        raise RuntimeError(
            f"{model_name} contains unexpected floating-point parameter dtypes: {floating_parameter_dtypes}"
        )

    # ------------------------------------------------------------------
    # Numerical validation
    # ------------------------------------------------------------------

    print()
    print("Checking FP16 numerical stability...")

    finite_checks = 0

    with torch.inference_mode():
        for source_language in LANGUAGES:
            validate_finite_logits(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                source_language=source_language,
                source_text=TEST_SENTENCES[source_language],
            )

            finite_checks += 1

            print(f"  {source_language}: finite logits PASS")

    print(f"Finite-logit checks: {finite_checks}/{len(LANGUAGES)}")

    # ------------------------------------------------------------------
    # Functional translation validation
    # ------------------------------------------------------------------

    expected_translations = len(LANGUAGES) * (len(LANGUAGES) - 1)

    print()
    print(f"Checking all {expected_translations} translation directions...")

    completed_translations = 0
    empty_translations = 0

    with torch.inference_mode():
        for source_language in LANGUAGES:
            source_text = TEST_SENTENCES[source_language]

            print()
            print(f"SOURCE [{source_language}] {source_text}")

            for target_language in LANGUAGES:
                if source_language == target_language:
                    continue

                translation = translate(
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    source_text=source_text,
                    source_language=source_language,
                    target_language=target_language,
                )

                if not translation.strip():
                    empty_translations += 1

                    raise RuntimeError(
                        f"Empty FP16 translation produced by {model_name}: {source_language}->{target_language}"
                    )

                print(f"  {source_language} -> {target_language}: {translation}")

                completed_translations += 1

    if completed_translations != expected_translations:
        raise RuntimeError(f"{model_name} completed {completed_translations}/{expected_translations} directions.")

    torch.cuda.synchronize()

    cuda_allocated_mb = bytes_to_mb(torch.cuda.memory_allocated())

    print()
    print("-" * 80)
    print(f"{model_name} FP16 VALIDATION SUMMARY")
    print("-" * 80)

    print(f"Finite-logit checks : {finite_checks}/{len(LANGUAGES)}")

    print(f"Translations        : {completed_translations}/{expected_translations}")

    print(f"Empty translations  : {empty_translations}")

    print(f"Parameter memory    : {parameter_mb:.2f} MB")

    print(f"CUDA allocated      : {cuda_allocated_mb:.2f} MB (diagnostic only)")

    result = {
        "model": model_name,
        "model_id": model_id,
        "revision": revision,
        "device": device,
        "dtype": str(dtype),
        "parameters": parameters,
        "parameter_memory_mb": parameter_mb,
        "floating_parameter_dtypes": sorted(str(value) for value in floating_parameter_dtypes),
        "finite_logit_checks": finite_checks,
        "finite_logit_checks_expected": len(LANGUAGES),
        "completed_translation_directions": completed_translations,
        "expected_translation_directions": expected_translations,
        "empty_translations": empty_translations,
        "cuda_allocated_mb_diagnostic": cuda_allocated_mb,
        "status": "PASS",
    }

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("MT FP16 FUNCTIONAL VALIDATION")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"Transformers : {transformers.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. FP16 validation requires CUDA.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    capability = torch.cuda.get_device_capability(0)

    print(f"Compute cap. : {capability[0]}.{capability[1]}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    nllb_result = validate_model(
        model_name="NLLB",
        model_id=NLLB_MODEL_ID,
        revision=NLLB_REVISION,
    )

    results.append(nllb_result)

    m2m100_result = validate_model(
        model_name="M2M100",
        model_id=M2M100_MODEL_ID,
        revision=M2M100_REVISION,
    )

    results.append(m2m100_result)

    summary = {
        "experiment": "MT FP16 functional validation",
        "precision": "float16",
        "generation": {
            "batch_size": 1,
            "num_beams": 1,
            "do_sample": False,
            "max_length": MAX_LENGTH,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": (f"{capability[0]}.{capability[1]}"),
        },
        "models": results,
    }

    RESULT_PATH.write_text(
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
    print("FINAL FP16 VALIDATION")
    print("=" * 80)

    for result in results:
        print(
            f"{result['model']:<8} "
            f"dtype={result['dtype']} | "
            f"finite="
            f"{result['finite_logit_checks']}/"
            f"{result['finite_logit_checks_expected']} | "
            f"directions="
            f"{result['completed_translation_directions']}/"
            f"{result['expected_translation_directions']} | "
            f"parameter_memory="
            f"{result['parameter_memory_mb']:.2f} MB"
        )

    print()
    print(f"Result: {RESULT_PATH}")

    print()
    print("=" * 80)
    print("MT FP16 FUNCTIONAL VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
