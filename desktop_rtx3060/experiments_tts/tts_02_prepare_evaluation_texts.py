from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# =============================================================================
# Console
# =============================================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore


# =============================================================================
# Fixed experiment configuration
# =============================================================================

EXPECTED_CASES = 100
EXPECTED_UTTERANCES = 25
EXPECTED_DIRECTIONS = 20
EXPECTED_CASES_PER_DIRECTION = 5
EXPECTED_CASES_PER_TARGET_LANGUAGE = 20

LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "pt",
]


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = PROJECT_ROOT / "results" / "mt_selection" / "final_only_live" / "final_only_predictions.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "data" / "tts_selection"

MANIFEST_PATH = OUTPUT_DIR / "tts_evaluation_manifest.jsonl"

METADATA_PATH = OUTPUT_DIR / "tts_evaluation_metadata.json"


# =============================================================================
# File helpers
# =============================================================================


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_jsonl(
    path: Path,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.rstrip("\n")

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path.name}, line {line_number}.") from exc

            if not isinstance(
                record,
                dict,
            ):
                raise RuntimeError(f"Expected JSON object in {path.name}, line {line_number}.")

            records.append(record)

    return records


def write_manifest(
    rows: list[dict],
) -> None:
    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )


# =============================================================================
# Input validation
# =============================================================================


def validate_source_rows(
    rows: list[dict],
) -> None:
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"Unexpected mt_15 prediction count.\nExpected: {EXPECTED_CASES}\nActual  : {len(rows)}")

    required_fields = {
        "case_index",
        "utterance_index",
        "dataset_index",
        "sample_id",
        "source_language",
        "target_language",
        "source_text",
        "prediction",
        "exact_match_mt10",
        "reached_max_length",
    }

    case_indices = []

    sequence_keys = set()

    utterance_indices = set()

    direction_counts = Counter()

    target_counts = Counter()

    source_counts = Counter()

    for row_number, row in enumerate(
        rows,
        start=1,
    ):
        missing = required_fields - set(row.keys())

        if missing:
            raise RuntimeError(f"Row {row_number} is missing fields: {sorted(missing)}")

        case_index = int(row["case_index"])

        utterance_index = int(row["utterance_index"])

        source_language = str(row["source_language"])

        target_language = str(row["target_language"])

        source_text = str(row["source_text"])

        prediction = str(row["prediction"])

        # --------------------------------------------------------------
        # Case identity
        # --------------------------------------------------------------

        case_indices.append(case_index)

        utterance_indices.add(utterance_index)

        sequence_key = (
            utterance_index,
            source_language,
            target_language,
        )

        if sequence_key in sequence_keys:
            raise RuntimeError(f"Duplicate utterance-direction sequence: {sequence_key}")

        sequence_keys.add(sequence_key)

        # --------------------------------------------------------------
        # Language validation
        # --------------------------------------------------------------

        if source_language not in LANGUAGES:
            raise RuntimeError(f"Unexpected source language in case {case_index}: {source_language}")

        if target_language not in LANGUAGES:
            raise RuntimeError(f"Unexpected target language in case {case_index}: {target_language}")

        if source_language == target_language:
            raise RuntimeError(f"Same source and target language in case {case_index}.")

        direction_counts[
            (
                source_language,
                target_language,
            )
        ] += 1

        source_counts[source_language] += 1

        target_counts[target_language] += 1

        # --------------------------------------------------------------
        # MT validation inherited from mt_15
        # --------------------------------------------------------------

        if not bool(row["exact_match_mt10"]):
            raise RuntimeError(f"Case {case_index} did not match the validated mt_10 final translation.")

        if bool(row["reached_max_length"]):
            raise RuntimeError(f"Case {case_index} reached the MT generation length guard.")

        # --------------------------------------------------------------
        # Text validation
        # --------------------------------------------------------------

        if not source_text.strip():
            raise RuntimeError(f"Empty source text in case {case_index}.")

        if not prediction.strip():
            raise RuntimeError(f"Empty target text in case {case_index}.")

        if "\x00" in prediction:
            raise RuntimeError(f"NUL character found in target text for case {case_index}.")

    # ------------------------------------------------------------------
    # case_index must be a stable 1..100 key.
    # ------------------------------------------------------------------

    expected_indices = list(
        range(
            1,
            EXPECTED_CASES + 1,
        )
    )

    if sorted(case_indices) != expected_indices:
        raise RuntimeError("case_index is not exactly 1..100.")

    # ------------------------------------------------------------------
    # 25 ASR source utterances.
    # ------------------------------------------------------------------

    if len(utterance_indices) != EXPECTED_UTTERANCES:
        raise RuntimeError(
            "Unexpected number of source "
            "utterances.\n"
            f"Expected: {EXPECTED_UTTERANCES}\n"
            f"Actual  : {len(utterance_indices)}"
        )

    # ------------------------------------------------------------------
    # 20 directed language pairs × 5 cases.
    # ------------------------------------------------------------------

    if len(direction_counts) != EXPECTED_DIRECTIONS:
        raise RuntimeError(
            "Unexpected number of translation "
            "directions.\n"
            f"Expected: {EXPECTED_DIRECTIONS}\n"
            f"Actual  : {len(direction_counts)}"
        )

    for direction, count in sorted(direction_counts.items()):
        if count != EXPECTED_CASES_PER_DIRECTION:
            raise RuntimeError(f"Direction {direction} has {count} cases; expected {EXPECTED_CASES_PER_DIRECTION}.")

    # ------------------------------------------------------------------
    # Each language appears 20 times as TTS target.
    # ------------------------------------------------------------------

    for language in LANGUAGES:
        if target_counts[language] != EXPECTED_CASES_PER_TARGET_LANGUAGE:
            raise RuntimeError(
                f"Target language {language} "
                f"has "
                f"{target_counts[language]} "
                f"cases; expected "
                f"{EXPECTED_CASES_PER_TARGET_LANGUAGE}."
            )

        if source_counts[language] != EXPECTED_CASES_PER_TARGET_LANGUAGE:
            raise RuntimeError(
                f"Source language {language} "
                f"has "
                f"{source_counts[language]} "
                f"cases; expected "
                f"{EXPECTED_CASES_PER_TARGET_LANGUAGE}."
            )


# =============================================================================
# Manifest construction
# =============================================================================


def build_manifest(
    source_rows: list[dict],
) -> list[dict]:
    source_rows = sorted(
        source_rows,
        key=lambda row: int(row["case_index"]),
    )

    manifest = []

    for source_row in source_rows:
        case_index = int(source_row["case_index"])

        text = str(source_row["prediction"])

        manifest_row = {
            # ----------------------------------------------------------
            # Stable TTS identifier.
            #
            # sample_id is deliberately NOT used as a unique key.
            # ----------------------------------------------------------
            "tts_case_id": (f"tts_{case_index:03d}"),
            "case_index": (case_index),
            # ----------------------------------------------------------
            # Provenance
            # ----------------------------------------------------------
            "utterance_index": int(source_row["utterance_index"]),
            "dataset_index": int(source_row["dataset_index"]),
            "sample_id": (source_row["sample_id"]),
            "source_language": (source_row["source_language"]),
            "target_language": (source_row["target_language"]),
            # ----------------------------------------------------------
            # Actual TTS input
            #
            # IMPORTANT:
            # No normalization, stripping or deduplication is applied.
            # ----------------------------------------------------------
            "text": (text),
            # ----------------------------------------------------------
            # Text statistics
            # ----------------------------------------------------------
            "character_count": (len(text)),
            "utf8_byte_count": (len(text.encode("utf-8"))),
            "word_count": (len(text.split())),
            # ----------------------------------------------------------
            # Useful source context for later inspection.
            # Not used as TTS input.
            # ----------------------------------------------------------
            "source_text": (source_row["source_text"]),
        }

        manifest.append(manifest_row)

    return manifest


# =============================================================================
# Manifest validation
# =============================================================================


def validate_manifest(
    manifest: list[dict],
) -> None:
    if len(manifest) != EXPECTED_CASES:
        raise RuntimeError("Unexpected manifest size.")

    case_ids = [row["tts_case_id"] for row in manifest]

    if len(set(case_ids)) != EXPECTED_CASES:
        raise RuntimeError("tts_case_id is not unique.")

    for row in manifest:
        text = row["text"]

        if not text.strip():
            raise RuntimeError(f"Empty TTS text: {row['tts_case_id']}")

        if row["character_count"] != len(text):
            raise RuntimeError(f"Character count mismatch for {row['tts_case_id']}.")

        if row["utf8_byte_count"] != len(text.encode("utf-8")):
            raise RuntimeError(f"UTF-8 byte count mismatch for {row['tts_case_id']}.")

        if row["word_count"] != len(text.split()):
            raise RuntimeError(f"Word count mismatch for {row['tts_case_id']}.")


# =============================================================================
# Statistics
# =============================================================================


def summarize_texts(
    rows: list[dict],
) -> dict:
    char_counts = [int(row["character_count"]) for row in rows]

    word_counts = [int(row["word_count"]) for row in rows]

    texts = [row["text"] for row in rows]

    frequencies = Counter(texts)

    duplicate_occurrences = sum(count - 1 for count in frequencies.values() if count > 1)

    duplicate_groups = sum(1 for count in frequencies.values() if count > 1)

    return {
        "cases": (len(rows)),
        "unique_texts": (len(frequencies)),
        "duplicate_occurrences": (duplicate_occurrences),
        "duplicate_groups": (duplicate_groups),
        "characters": {
            "mean": (statistics.mean(char_counts)),
            "median": (statistics.median(char_counts)),
            "min": (min(char_counts)),
            "max": (max(char_counts)),
        },
        "words": {
            "mean": (statistics.mean(word_counts)),
            "median": (statistics.median(word_counts)),
            "min": (min(word_counts)),
            "max": (max(word_counts)),
        },
    }


def build_metadata(
    manifest: list[dict],
    source_sha256: str,
    manifest_sha256: str,
) -> dict:
    per_target_language = {}

    for language in LANGUAGES:
        language_rows = [row for row in manifest if row["target_language"] == language]

        per_target_language[language] = summarize_texts(language_rows)

    direction_counts = Counter(
        (
            row["source_language"],
            row["target_language"],
        )
        for row in manifest
    )

    directions = {}

    for (
        source_language,
        target_language,
    ), count in sorted(direction_counts.items()):
        key = f"{source_language}->{target_language}"

        directions[key] = {"cases": (count)}

    return {
        "experiment": ("Fixed TTS evaluation text set derived from validated mt_15 final-only NLLB translations."),
        "source": {
            "experiment": ("mt_15_benchmark_final_only"),
            "path": (str(SOURCE_PATH)),
            "sha256": (source_sha256),
            "validated_mt_outputs": (EXPECTED_CASES),
        },
        "manifest": {
            "path": (str(MANIFEST_PATH)),
            "sha256": (manifest_sha256),
            "cases": (len(manifest)),
        },
        "scope": {
            "languages": (LANGUAGES),
            "source_utterances": (EXPECTED_UTTERANCES),
            "translation_directions": (EXPECTED_DIRECTIONS),
            "cases_per_direction": (EXPECTED_CASES_PER_DIRECTION),
            "cases_per_target_language": (EXPECTED_CASES_PER_TARGET_LANGUAGE),
        },
        "methodology": {
            "tts_input_field": ("text"),
            "text_source": ("Exact final NLLB prediction produced by mt_15."),
            "normalization": ("None."),
            "deduplication": (
                "None. Duplicate translations are preserved because each row represents a real pipeline case."
            ),
            "ordering": ("Ascending mt_15 case_index."),
            "unique_key": ("tts_case_id"),
        },
        "statistics": {
            "overall": (summarize_texts(manifest)),
            "per_target_language": (per_target_language),
            "directions": (directions),
        },
    }


# =============================================================================
# Console output
# =============================================================================


def print_summary(
    metadata: dict,
) -> None:
    overall = metadata["statistics"]["overall"]

    print()
    print("=" * 96)
    print("TTS EVALUATION TEXT SUMMARY")
    print("=" * 96)

    print(f"Cases                    : {overall['cases']}")

    print(f"Unique texts             : {overall['unique_texts']}")

    print(f"Duplicate occurrences    : {overall['duplicate_occurrences']}")

    print(f"Duplicate groups         : {overall['duplicate_groups']}")

    print()

    print(f"Mean characters          : {overall['characters']['mean']:.2f}")

    print(f"Median characters        : {overall['characters']['median']:.2f}")

    print(f"Character range          : {overall['characters']['min']} - {overall['characters']['max']}")

    print()

    print(f"Mean words               : {overall['words']['mean']:.2f}")

    print(f"Median words             : {overall['words']['median']:.2f}")

    print(f"Word range               : {overall['words']['min']} - {overall['words']['max']}")

    print()
    print("Per target language")

    print("-" * 96)

    print(f"{'Language':<12}{'Cases':>8}{'Unique':>10}{'Duplicates':>12}{'Mean chars':>14}{'Mean words':>14}")

    print("-" * 96)

    for language in LANGUAGES:
        stats = metadata["statistics"]["per_target_language"][language]

        print(
            f"{language.upper():<12}"
            f"{stats['cases']:>8}"
            f"{stats['unique_texts']:>10}"
            f"{stats['duplicate_occurrences']:>12}"
            f"{stats['characters']['mean']:>14.2f}"
            f"{stats['words']['mean']:>14.2f}"
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 80)
    print("TTS EVALUATION TEXT PREPARATION")
    print("=" * 80)

    print(f"Source   : {SOURCE_PATH}")

    print(f"Manifest : {MANIFEST_PATH}")

    print()

    print("Loading validated mt_15 outputs...")

    source_rows = load_jsonl(SOURCE_PATH)

    print(f"Rows loaded : {len(source_rows)}")

    validate_source_rows(source_rows)

    print("Source validation : PASS")

    source_sha256 = sha256_file(SOURCE_PATH)

    print(f"Source SHA256 : {source_sha256}")

    print()

    print("Building TTS evaluation manifest...")

    manifest = build_manifest(source_rows)

    validate_manifest(manifest)

    print(f"Manifest cases : {len(manifest)}")

    print("Manifest validation : PASS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_manifest(manifest)

    manifest_sha256 = sha256_file(MANIFEST_PATH)

    metadata = build_metadata(
        manifest=manifest,
        source_sha256=(source_sha256),
        manifest_sha256=(manifest_sha256),
    )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print_summary(metadata)

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Manifest : {MANIFEST_PATH}")

    print(f"Metadata : {METADATA_PATH}")

    print(f"SHA256   : {manifest_sha256}")

    print()

    print("=" * 80)
    print("TTS EVALUATION TEXT PREPARATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
