import json
from pathlib import Path

import soundfile as sf

from experiments_asr.asr_01_fleurs_subset import MANIFEST_PATH, METADATA_PATH, load_audio_from_entry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTION_PATH = PROJECT_ROOT / "results" / "asr_selection" / "parakeet_streaming_configs" / "selection.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "data" / "multilingual_eval" / "warmup"
OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "multilingual_eval" / "warmup_manifest.jsonl"

LANGUAGES = ["en", "es", "fr", "de", "pt"]

MIN_DURATION_S = 7.0
MAX_DURATION_S = 10.0
TARGET_DURATION_S = 8.0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    manifest = read_jsonl(MANIFEST_PATH)
    evaluation_selection = read_jsonl(SELECTION_PATH)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    expected_sample_rate = int(metadata["sample_rate"])

    used_samples = {(entry["language"], str(entry["sample_id"])) for entry in evaluation_selection}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_records = []

    print("=" * 80)
    print("MULTILINGUAL WARM-UP PREPARATION")
    print("=" * 80)

    for language in LANGUAGES:
        candidates = [
            entry
            for entry in manifest
            if entry["language"] == language
            and (
                language,
                str(entry["sample_id"]),
            )
            not in used_samples
            and MIN_DURATION_S <= float(entry["duration_s"]) <= MAX_DURATION_S
        ]

        if not candidates:
            raise RuntimeError(f"No valid warm-up candidate for language: {language}")

        candidates.sort(
            key=lambda entry: (
                abs(float(entry["duration_s"]) - TARGET_DURATION_S),
                int(entry["sample_id"]),
            )
        )

        selected = candidates[0]

        samples, sample_rate, reference = load_audio_from_entry(
            selected,
            metadata,
        )

        if samples.ndim != 1:
            raise RuntimeError(f"{language}: expected mono audio, got shape {samples.shape}")

        output_wav = OUTPUT_DIR / f"warmup_{language}.wav"

        sf.write(
            output_wav,
            samples,
            sample_rate,
            subtype="PCM_16",
        )

        info = sf.info(output_wav)

        if info.samplerate != expected_sample_rate:
            raise RuntimeError(f"{language}: unexpected sample rate {info.samplerate}")

        if info.channels != 1:
            raise RuntimeError(f"{language}: expected mono audio, got {info.channels} channels")

        if info.subtype != "PCM_16":
            raise RuntimeError(f"{language}: expected PCM_16, got {info.subtype}")

        record = {
            "language": language,
            "dataset_config": selected["dataset_config"],
            "dataset_index": int(selected["dataset_index"]),
            "sample_id": int(selected["sample_id"]),
            "duration_s": float(selected["duration_s"]),
            "reference": reference,
            "raw_transcription": selected["raw_transcription"],
            "audio_filename": output_wav.name,
        }

        output_records.append(record)

        print()
        print(f"Language      : {language.upper()}")
        print(f"Sample ID     : {selected['sample_id']}")
        print(f"Dataset index : {selected['dataset_index']}")
        print(f"Duration      : {selected['duration_s']:.2f} s")
        print(f"Reference     : {reference}")
        print(f"Output        : {output_wav}")
        print(f"WAV contract  : {info.samplerate} Hz / mono / {info.subtype}")

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in output_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 80)
    print(f"Warm-up WAVs : {len(output_records)}")
    print(f"Manifest     : {OUTPUT_MANIFEST}")
    print("Preparation  : PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
