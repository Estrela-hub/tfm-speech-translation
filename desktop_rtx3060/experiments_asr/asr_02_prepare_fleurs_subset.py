import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "google/fleurs"
DATASET_REVISION = "70bb2e8"
DATASET_SPLIT = "test"

FLEURS_SAMPLE_RATE = 16_000

LANGUAGES = {
    "en": "en_us",
    "es": "es_419",
    "fr": "fr_fr",
    "de": "de_de",
    "pt": "pt_br",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "asr_selection"
MANIFEST_PATH = OUTPUT_DIR / "fleurs_manifest.jsonl"
METADATA_PATH = OUTPUT_DIR / "fleurs_manifest_metadata.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible multilingual FLEURS subset for ASR evaluation."
    )

    parser.add_argument(
        "--samples-per-language",
        type=int,
        default=200,
        help="Number of test utterances selected for each language.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sample selection.",
    )

    return parser.parse_args()


def load_language_dataset(config_name: str):
    return load_dataset(
        DATASET_NAME,
        config_name,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )


def select_samples(
    dataset,
    samples_per_language: int,
    seed: int,
):
    if samples_per_language <= 0:
        raise ValueError("samples_per_language must be greater than zero.")

    if len(dataset) < samples_per_language:
        raise ValueError(f"Requested {samples_per_language} samples, but the dataset only contains {len(dataset)}.")

    dataset = dataset.add_column(
        "_dataset_index",
        list(range(len(dataset))),
    )

    dataset = dataset.shuffle(seed=seed)

    return dataset.select(range(samples_per_language))


def build_manifest_entries(
    language: str,
    config_name: str,
    dataset,
) -> list[dict]:
    entries = []

    for sample in dataset:
        transcription = sample["transcription"].strip()

        if not transcription:
            raise ValueError(f"Empty transcription found for {config_name}, sample id {sample['id']}.")

        num_samples = int(sample["num_samples"])

        entries.append(
            {
                "language": language,
                "dataset_config": config_name,
                "dataset_index": int(sample["_dataset_index"]),
                "sample_id": int(sample["id"]),
                "num_samples": num_samples,
                "duration_s": num_samples / FLEURS_SAMPLE_RATE,
                "transcription": transcription,
                "raw_transcription": sample["raw_transcription"],
            }
        )

    return entries


def write_manifest(entries: list[dict]) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for entry in entries:
            file.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def write_metadata(
    samples_per_language: int,
    seed: int,
    entries: list[dict],
) -> None:
    language_summary = {}

    for language in LANGUAGES:
        language_entries = [entry for entry in entries if entry["language"] == language]

        language_summary[language] = {
            "dataset_config": LANGUAGES[language],
            "samples": len(language_entries),
            "total_duration_s": sum(entry["duration_s"] for entry in language_entries),
        }

    metadata = {
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "sample_rate": FLEURS_SAMPLE_RATE,
        "seed": seed,
        "samples_per_language": samples_per_language,
        "total_samples": len(entries),
        "languages": language_summary,
        "manifest_sha256": calculate_sha256(MANIFEST_PATH),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(entries: list[dict]) -> None:
    print()
    print("=" * 70)
    print("FLEURS ASR SELECTION DATASET")
    print("=" * 70)

    for language, config_name in LANGUAGES.items():
        language_entries = [entry for entry in entries if entry["language"] == language]

        duration = sum(entry["duration_s"] for entry in language_entries)

        print(f"{language.upper():2} ({config_name:6}) | {len(language_entries):3} samples | {duration / 60:.2f} min")

    print("-" * 70)
    print(f"Total samples : {len(entries)}")
    print(f"Manifest      : {MANIFEST_PATH}")
    print(f"Metadata      : {METADATA_PATH}")
    print(f"SHA256        : {calculate_sha256(MANIFEST_PATH)}")
    print("=" * 70)


def main() -> None:
    args = parse_arguments()

    all_entries = []

    for language, config_name in LANGUAGES.items():
        print()
        print(f"Loading {language.upper()} ({config_name})...")

        dataset = load_language_dataset(config_name)

        print(f"Available test samples: {len(dataset)}")

        selected_dataset = select_samples(
            dataset=dataset,
            samples_per_language=args.samples_per_language,
            seed=args.seed,
        )

        entries = build_manifest_entries(
            language=language,
            config_name=config_name,
            dataset=selected_dataset,
        )

        all_entries.extend(entries)

    write_manifest(all_entries)

    write_metadata(
        samples_per_language=args.samples_per_language,
        seed=args.seed,
        entries=all_entries,
    )

    print_summary(all_entries)


if __name__ == "__main__":
    main()
