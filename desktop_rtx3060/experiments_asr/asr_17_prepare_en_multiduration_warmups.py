import json
from pathlib import Path

import soundfile as sf

from experiments_asr.asr_01_fleurs_subset import MANIFEST_PATH, METADATA_PATH, load_audio_from_entry

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_SELECTION_PATH = (
    PROJECT_ROOT / "results" / "asr_selection" / "parakeet_streaming_configs" / "selection.jsonl"
)

CURRENT_WARMUP_MANIFEST_PATH = PROJECT_ROOT / "data" / "multilingual_eval" / "warmup_manifest.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "data" / "multilingual_eval" / "warmup_multiduration" / "en"

OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "multilingual_eval" / "warmup_multiduration" / "en_manifest.jsonl"

TARGET_DURATIONS_S = [6.0, 10.0, 14.0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    manifest = read_jsonl(MANIFEST_PATH)
    evaluation_selection = read_jsonl(EVALUATION_SELECTION_PATH)
    current_warmups = read_jsonl(CURRENT_WARMUP_MANIFEST_PATH)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    expected_sample_rate = int(metadata["sample_rate"])

    excluded_ids = {int(row["sample_id"]) for row in evaluation_selection if row["language"] == "en"}

    excluded_ids.update(int(row["sample_id"]) for row in current_warmups if row["language"] == "en")

    candidates = [row for row in manifest if row["language"] == "en" and int(row["sample_id"]) not in excluded_ids]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_ids = set()
    output_records = []

    print("=" * 80)
    print("EN MULTI-DURATION WARM-UP PREPARATION")
    print("=" * 80)

    for target_duration in TARGET_DURATIONS_S:
        available = [row for row in candidates if int(row["sample_id"]) not in selected_ids]

        available.sort(
            key=lambda row: (
                abs(float(row["duration_s"]) - target_duration),
                int(row["sample_id"]),
            )
        )

        if not available:
            raise RuntimeError(f"No candidate for target duration {target_duration:.1f} s")

        selected = available[0]
        selected_ids.add(int(selected["sample_id"]))

        samples, sample_rate, reference = load_audio_from_entry(
            selected,
            metadata,
        )

        label = f"{int(target_duration):02d}s"

        output_wav = OUTPUT_DIR / f"warmup_en_{label}.wav"

        sf.write(
            output_wav,
            samples,
            sample_rate,
            subtype="PCM_16",
        )

        info = sf.info(output_wav)

        if info.samplerate != expected_sample_rate:
            raise RuntimeError(f"Unexpected sample rate: {info.samplerate}")

        if info.channels != 1:
            raise RuntimeError(f"Expected mono, got {info.channels} channels")

        if info.subtype != "PCM_16":
            raise RuntimeError(f"Expected PCM_16, got {info.subtype}")

        record = {
            "target_duration_s": target_duration,
            "actual_duration_s": float(selected["duration_s"]),
            "dataset_config": selected["dataset_config"],
            "dataset_index": int(selected["dataset_index"]),
            "sample_id": int(selected["sample_id"]),
            "reference": reference,
            "raw_transcription": selected["raw_transcription"],
            "audio_filename": output_wav.name,
        }

        output_records.append(record)

        print()
        print(f"Target duration : {target_duration:.2f} s")
        print(f"Actual duration : {selected['duration_s']:.2f} s")
        print(f"Sample ID       : {selected['sample_id']}")
        print(f"Dataset index   : {selected['dataset_index']}")
        print(f"Reference       : {reference}")
        print(f"Output          : {output_wav}")
        print(f"WAV contract    : {info.samplerate} Hz / mono / {info.subtype}")

    with OUTPUT_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for record in output_records:
            handle.write(
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
