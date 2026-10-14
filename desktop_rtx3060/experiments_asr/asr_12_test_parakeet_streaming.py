import json
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from experiments_asr.asr_01_fleurs_subset import load_audio_from_entry, load_manifest, load_metadata, validate_manifest

# The original experiment used Parakeet revision
# 541d1f99c6b0c3cd0b11a95167540bb8edefd82b.
# This smoke test delegates model loading to the NeMo streaming script
# through pretrained_name, so the revision is not explicitly pinned here.
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

# First configuration recommended by NVIDIA for this streaming script.
CHUNK_SECS = 0.16
LEFT_CONTEXT_SECS = 10.0
RIGHT_CONTEXT_SECS = 1.84

# Keep the first smoke test simple.
TEST_LANGUAGE = "en"

# Use FP32 first so that precision is not another experimental variable.
COMPUTE_DTYPE = "float32"


PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEMO_SCRIPT = (
    PROJECT_ROOT
    / "external"
    / "NeMo-Speech"
    / "examples"
    / "asr"
    / "asr_chunked_inference"
    / "rnnt"
    / "speech_to_text_streaming_infer_rnnt.py"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "asr_selection" / "parakeet_streaming_smoke"

AUDIO_PATH = RESULTS_DIR / "sample.wav"
MANIFEST_PATH = RESULTS_DIR / "manifest.jsonl"
OUTPUT_PATH = RESULTS_DIR / "streaming_output.jsonl"


def find_test_entry(
    entries: list[dict],
    language: str,
) -> dict:
    for entry in entries:
        if entry["language"] == language:
            return entry

    raise RuntimeError(f"No FLEURS entry found for language: {language}")


def write_audio(
    samples,
    sample_rate: int,
) -> None:
    sf.write(
        AUDIO_PATH,
        samples,
        sample_rate,
        subtype="PCM_16",
    )


def write_manifest(
    reference: str,
    duration_s: float,
) -> None:
    record = {
        "audio_filepath": str(AUDIO_PATH.resolve()),
        "duration": duration_s,
        "text": reference,
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def remove_previous_output() -> None:
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()


def build_command() -> list[str]:
    return [
        sys.executable,
        str(NEMO_SCRIPT),
        f"pretrained_name={MODEL_ID}",
        "model_path=null",
        "audio_dir=null",
        f"dataset_manifest={MANIFEST_PATH.as_posix()}",
        f"output_filename={OUTPUT_PATH.as_posix()}",
        f"chunk_secs={CHUNK_SECS}",
        f"left_context_secs={LEFT_CONTEXT_SECS}",
        f"right_context_secs={RIGHT_CONTEXT_SECS}",
        "batch_size=1",
        "num_workers=0",
        # Explicitly use CUDA device 0.
        "cuda=0",
        # Controlled baseline precision.
        f"compute_dtype={COMPUTE_DTYPE}",
        # This must remain False:
        # False = actual streaming decoding with state between chunks.
        # True = encoder chunks are accumulated and decoded afterwards.
        "simulated=false",
        # Streaming script currently expects greedy_batch +
        # label-looping for non-simulated streaming.
        "decoding.strategy=greedy_batch",
        "decoding.greedy.loop_labels=true",
        # Keep the attention context aligned with the selected chunk.
        "att_context_size_as_chunk=true",
        "sort_by_duration=false",
        "overwrite_transcripts=true",
        # Quality measurement is not part of this smoke test.
        "calculate_wer=false",
        "calculate_rtfx=true",
        "clean_groundtruth_text=false",
        "timestamps=false",
        "confidence=false",
    ]


def run_streaming_script() -> None:
    command = build_command()

    environment = os.environ.copy()

    # Helps keep terminal/file handling predictable on Windows.
    environment["PYTHONUTF8"] = "1"

    print("Starting official NeMo streaming inference...")
    print()

    result = subprocess.run(
        command,
        cwd=NEMO_SCRIPT.parent,
        env=environment,
        check=False,
    )

    print()

    if result.returncode != 0:
        raise RuntimeError(f"NeMo streaming inference failed.\nExit code: {result.returncode}")


def read_prediction() -> str:
    if not OUTPUT_PATH.exists():
        raise RuntimeError(f"NeMo finished but the expected output file was not created:\n{OUTPUT_PATH}")

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = [json.loads(line) for line in file if line.strip()]

    if len(records) != 1:
        raise RuntimeError(f"Expected exactly one output record, found {len(records)}.")

    record = records[0]

    if "pred_text" not in record:
        print()
        print("Output record produced by NeMo:")
        print(
            json.dumps(
                record,
                ensure_ascii=False,
                indent=2,
            )
        )

        raise RuntimeError("The output record does not contain 'pred_text'.")

    return str(record["pred_text"]).strip()


def main() -> None:
    if not NEMO_SCRIPT.exists():
        raise FileNotFoundError(f"NeMo streaming script was not found:\n{NEMO_SCRIPT}")

    metadata = load_metadata()
    entries = load_manifest()

    validate_manifest(
        entries,
        metadata,
    )

    entry = find_test_entry(
        entries,
        TEST_LANGUAGE,
    )

    samples, sample_rate, reference = load_audio_from_entry(
        entry,
        metadata,
    )

    duration_s = len(samples) / sample_rate

    requested_latency_s = CHUNK_SECS + RIGHT_CONTEXT_SECS

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_previous_output()

    write_audio(
        samples=samples,
        sample_rate=sample_rate,
    )

    write_manifest(
        reference=reference,
        duration_s=duration_s,
    )

    print("=" * 72)
    print("PARAKEET-TDT 0.6B V3 STREAMING SMOKE TEST")
    print("=" * 72)

    print(f"Model               : {MODEL_ID}")
    print(f"Language            : {TEST_LANGUAGE.upper()}")
    print(f"Sample ID           : {entry['sample_id']}")
    print(f"Duration            : {duration_s:.3f} s")
    print(f"Sample rate         : {sample_rate} Hz")
    print(f"Compute dtype       : {COMPUTE_DTYPE}")

    print()
    print("Streaming configuration")
    print("-" * 72)

    print(f"Left context        : {LEFT_CONTEXT_SECS:.2f} s")

    print(f"Chunk               : {CHUNK_SECS:.2f} s")

    print(f"Right context       : {RIGHT_CONTEXT_SECS:.2f} s")

    print(f"Requested latency   : {requested_latency_s:.2f} s")

    print()

    print(f"Reference           : {reference}")

    print()
    print("=" * 72)

    run_streaming_script()

    prediction = read_prediction()

    print()
    print("=" * 72)
    print("STREAMING RESULT")
    print("=" * 72)

    print(f"Reference  : {reference}")
    print(f"Prediction : {prediction}")

    print()
    print(f"Output     : {OUTPUT_PATH}")

    print("=" * 72)
    print("SMOKE TEST COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()
