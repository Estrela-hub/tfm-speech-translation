import torch
from transformers import AutoModel, AutoProcessor

from experiments_asr.asr_01_fleurs_subset import load_audio_from_entry, load_manifest, load_metadata, validate_manifest

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"


# FLEURS configuration -> Nemotron locale.
#
# es_419 represents Latin-American Spanish. Nemotron does not expose
# an es-419 prompt, so es-US is used as the closest supported locale.
LANGUAGE_LOCALES = {
    "en": "en-US",
    "es": "es-US",
    "fr": "fr-FR",
    "de": "de-DE",
    "pt": "pt-BR",
}


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This experiment is intended to run on the RTX 3060.")

    device = torch.device("cuda")

    print("=" * 70)
    print("NEMOTRON 3.5 ASR SMOKE TEST")
    print("=" * 70)

    print(f"CUDA device : {torch.cuda.get_device_name(0)}")

    revision = MODEL_REVISION

    print(f"Model       : {MODEL_ID}")
    print(f"Revision    : {revision}")
    print()

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=revision,
    )

    print("Loading model...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
    )

    model = model.to(device)
    model.eval()

    print()
    print("Model loaded.")
    print(f"Device      : {next(model.parameters()).device}")
    print(f"Dtype       : {next(model.parameters()).dtype}")
    print(f"Sample rate : {processor.feature_extractor.sampling_rate} Hz")

    return processor, model, revision


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


def main() -> None:
    metadata = load_metadata()
    entries = load_manifest()

    validate_manifest(
        entries,
        metadata,
    )

    processor, model, revision = load_model()

    print()
    print("=" * 70)
    print("MULTILINGUAL TRANSCRIPTION TEST")
    print("=" * 70)

    for language, locale in LANGUAGE_LOCALES.items():
        entry = next(item for item in entries if item["language"] == language)

        samples, sample_rate, reference = load_audio_from_entry(
            entry,
            metadata,
        )

        duration = len(samples) / sample_rate

        print()
        print("-" * 70)
        print(f"{language.upper()} ({entry['dataset_config']} -> {locale})")
        print("-" * 70)

        print(f"Sample ID : {entry['sample_id']}")
        print(f"Duration  : {duration:.3f} s")
        print()
        print(f"Reference : {reference}")

        prediction = transcribe(
            samples=samples,
            sample_rate=sample_rate,
            locale=locale,
            processor=processor,
            model=model,
        )

        print(f"Prediction: {prediction}")

    print()
    print("=" * 70)
    print("SMOKE TEST COMPLETED")
    print(f"Model revision: {revision}")
    print("=" * 70)


if __name__ == "__main__":
    main()
