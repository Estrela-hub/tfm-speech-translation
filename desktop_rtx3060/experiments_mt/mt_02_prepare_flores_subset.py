from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import datasets
from datasets import load_dataset

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

DATASET_ID = "facebook/flores"
DATASET_REVISION = "71abf77d8b7beb5cfef59898d6b24d92ab7654fc"
SPLIT = "dev"

SAMPLE_COUNT = 200
SEED = 42

LANGUAGES = {
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

OUTPUT_DIR = PROJECT_ROOT / "data" / "mt_selection"

MANIFEST_PATH = OUTPUT_DIR / "flores_dev_200.jsonl"
METADATA_PATH = OUTPUT_DIR / "flores_dev_200_metadata.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_language_dataset(
    language_name: str,
    language_code: str,
    revision: str,
):
    print(f"Loading {language_name} ({language_code})...")

    try:
        dataset = load_dataset(
            DATASET_ID,
            language_code,
            split=SPLIT,
            revision=revision,
        )
    except Exception as exc:
        raise RuntimeError(
            "\nCould not load the official FLORES dataset.\n\n"
            "The dataset is gated on Hugging Face. Make sure that:\n"
            "  1. You have accepted the FLORES dataset access conditions.\n"
            "  2. You are authenticated locally with Hugging Face.\n\n"
            "After obtaining access, run:\n"
            "  hf auth login\n"
        ) from exc

    print(f"  Rows: {len(dataset)}")

    return dataset


def build_id_map(dataset, language_code: str) -> dict[int, str]:
    id_map: dict[int, str] = {}

    for row in dataset:
        row_id = int(row["id"])
        sentence = row["sentence"]

        if row_id in id_map:
            raise RuntimeError(f"Duplicate ID {row_id} found in {language_code}.")

        if not isinstance(sentence, str) or not sentence.strip():
            raise RuntimeError(f"Empty sentence found for ID {row_id} in {language_code}.")

        id_map[row_id] = sentence

    return id_map


def validate_alignment(
    language_maps: dict[str, dict[int, str]],
) -> list[int]:
    language_codes = list(language_maps.keys())

    reference_code = language_codes[0]
    reference_ids = set(language_maps[reference_code].keys())

    for language_code in language_codes[1:]:
        current_ids = set(language_maps[language_code].keys())

        if current_ids != reference_ids:
            missing = sorted(reference_ids - current_ids)
            extra = sorted(current_ids - reference_ids)

            raise RuntimeError(
                f"ID alignment failed for {language_code}.\nMissing IDs: {missing[:10]}\nExtra IDs  : {extra[:10]}"
            )

    return sorted(reference_ids)


def select_ids(all_ids: list[int]) -> list[int]:
    if SAMPLE_COUNT > len(all_ids):
        raise RuntimeError(f"Requested {SAMPLE_COUNT} samples, but only {len(all_ids)} are available.")

    rng = random.Random(SEED)

    selected = rng.sample(all_ids, SAMPLE_COUNT)

    # Preserve the original dataset order in the generated manifest.
    return sorted(selected)


def build_manifest_records(
    selected_ids: list[int],
    language_maps: dict[str, dict[int, str]],
) -> list[dict]:
    records = []

    for row_id in selected_ids:
        record = {
            "id": row_id,
        }

        for language_code in LANGUAGES.values():
            record[language_code] = language_maps[language_code][row_id]  # type: ignore

        records.append(record)

    return records


def write_manifest(records: list[dict]) -> str:
    lines = []

    for record in records:
        line = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        lines.append(line)

    # Explicit UTF-8 and LF line endings make the resulting hash independent
    # of the Windows default text line ending.
    content = "\n".join(lines) + "\n"
    encoded = content.encode("utf-8")

    MANIFEST_PATH.write_bytes(encoded)

    return hashlib.sha256(encoded).hexdigest()


def write_metadata(
    revision: str,
    total_samples: int,
    manifest_sha256: str,
) -> None:
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_revision": revision,
        "split": SPLIT,
        "source_split_size": total_samples,
        "selection_size": SAMPLE_COUNT,
        "selection_seed": SEED,
        "languages": LANGUAGES,
        "manifest": MANIFEST_PATH.name,
        "manifest_sha256": manifest_sha256,
        "purpose": "MT model selection",
    }

    content = json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    METADATA_PATH.write_text(
        content + "\n",
        encoding="utf-8",
        newline="\n",
    )


def print_example_records(records: list[dict]) -> None:
    print()
    print("Example selected samples")
    print("-" * 80)

    for record in records[:3]:
        print(f"ID {record['id']}")
        print(f"  EN: {record['eng_Latn']}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("FLORES MT SELECTION DATASET PREPARATION")
    print("=" * 80)

    print(f"Datasets version : {datasets.__version__}")
    print(f"Dataset          : {DATASET_ID}")
    print(f"Split            : {SPLIT}")
    print(f"Selection size   : {SAMPLE_COUNT}")
    print(f"Seed             : {SEED}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    revision = DATASET_REVISION

    print(f"Resolved revision: {revision}")
    print()

    language_maps: dict[str, dict[int, str]] = {}

    for language_name, language_code in LANGUAGES.items():
        dataset = load_language_dataset(
            language_name,
            language_code,
            revision,
        )

        id_map = build_id_map(
            dataset,
            language_code,
        )

        language_maps[language_code] = id_map

    print()
    print("Validating cross-language alignment...")
    all_ids = validate_alignment(language_maps)

    print(f"Aligned IDs: {len(all_ids)}")

    if len(all_ids) != 997:
        print(f"WARNING: expected 997 rows for FLORES dev, but found {len(all_ids)}.")

    selected_ids = select_ids(all_ids)

    print(f"Selected IDs: {len(selected_ids)}")
    print(f"First selected ID: {selected_ids[0]}")
    print(f"Last selected ID : {selected_ids[-1]}")

    records = build_manifest_records(
        selected_ids,
        language_maps,
    )

    manifest_sha256 = write_manifest(records)

    write_metadata(
        revision=revision,
        total_samples=len(all_ids),
        manifest_sha256=manifest_sha256,
    )

    print_example_records(records)

    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Manifest : {MANIFEST_PATH}")
    print(f"Metadata : {METADATA_PATH}")
    print(f"SHA256   : {manifest_sha256}")
    print()

    print("=" * 80)
    print("DATASET PREPARATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
