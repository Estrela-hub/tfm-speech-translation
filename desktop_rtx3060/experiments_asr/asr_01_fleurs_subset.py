import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "asr_selection"

MANIFEST_PATH = DATA_DIR / "fleurs_manifest.jsonl"
METADATA_PATH = DATA_DIR / "fleurs_manifest_metadata.json"


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest file not found: {MANIFEST_PATH}")

    entries = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line:
                entries.append(json.loads(line))

    return entries


def validate_manifest(
    entries: list[dict],
    metadata: dict,
) -> None:
    expected_hash = metadata["manifest_sha256"]
    actual_hash = calculate_sha256(MANIFEST_PATH)

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Manifest SHA256 does not match metadata.\nExpected: {expected_hash}\nActual  : {actual_hash}"
        )

    expected_samples = metadata["total_samples"]

    if len(entries) != expected_samples:
        raise RuntimeError(
            f"Unexpected number of manifest entries.\nExpected: {expected_samples}\nActual  : {len(entries)}"
        )


@lru_cache(maxsize=None)
def load_fleurs_dataset(
    dataset_name: str,
    revision: str,
    split: str,
    config_name: str,
):
    return load_dataset(
        dataset_name,
        config_name,
        split=split,
        revision=revision,
    )


def decode_audio(audio) -> tuple[np.ndarray, int]:
    """
    Decode a Hugging Face Audio value into a mono float32 NumPy array.
    """

    # Handle audio exposed through TorchCodec.
    if hasattr(audio, "get_all_samples"):
        decoded = audio.get_all_samples()

        samples = decoded.data

        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().numpy()

        samples = np.asarray(
            samples,
            dtype=np.float32,
        )

        # TorchCodec normally returns [channels, samples].
        if samples.ndim == 2:
            samples = samples.mean(axis=0)

        if samples.ndim != 1:
            raise ValueError(f"Unsupported decoded audio shape: {samples.shape}")

        return (
            np.ascontiguousarray(samples),
            int(decoded.sample_rate),
        )

    # Handle the dictionary format used by older Datasets versions.
    if isinstance(audio, dict):
        samples = np.asarray(
            audio["array"],
            dtype=np.float32,
        )

        return (
            np.ascontiguousarray(samples),
            int(audio["sampling_rate"]),
        )

    raise TypeError(f"Unsupported audio object: {type(audio)}")


def load_audio_from_entry(
    entry: dict,
    metadata: dict,
) -> tuple[np.ndarray, int, str]:
    dataset = load_fleurs_dataset(
        dataset_name=metadata["dataset"],
        revision=metadata["dataset_revision"],
        split=metadata["split"],
        config_name=entry["dataset_config"],
    )

    dataset_index = entry["dataset_index"]

    sample = dataset[dataset_index]

    if int(sample["id"]) != int(entry["sample_id"]):
        raise RuntimeError(f"Sample ID mismatch for {entry['dataset_config']} index {dataset_index}.")

    reference = sample["transcription"].strip()

    if reference != entry["transcription"]:
        raise RuntimeError(f"Transcription mismatch for {entry['dataset_config']} index {dataset_index}.")

    samples, sample_rate = decode_audio(sample["audio"])

    if sample_rate != metadata["sample_rate"]:
        raise RuntimeError(f"Unexpected sample rate: {sample_rate} Hz")

    return samples, sample_rate, reference
