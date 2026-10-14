from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import sys
import time
import unicodedata
import wave
from collections import Counter
from pathlib import Path
from typing import cast

import nemo
import nemo.collections.asr as nemo_asr
import numpy as np
import torch

# Import torch before ORT so CUDA/cuDNN DLLs from PyTorch are already loaded.
import onnxruntime as ort
import scipy
import transformers
from huggingface_hub import hf_hub_download
from piper import PiperVoice
from scipy.signal import resample_poly
from transformers import AutoModelForTextToWaveform, AutoTokenizer, set_seed

ort.preload_dlls()

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

DEVICE = "cuda:0"

EXPECTED_CASES = 100
EXPECTED_CASES_PER_LANGUAGE = 20

EXPECTED_MANIFEST_SHA256 = "821e4b4c2ef51ca68e98831b43b436d3fba70ea90f8b2541009dd7b67a0f54b4"

LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "pt",
]

MMS_SEED = 42

PIPER_RUN_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

PIPER_RUNS = len(PIPER_RUN_SEEDS)

ASR_SAMPLE_RATE = 16000
ASR_BATCH_SIZE = 4

LISTENING_CASES_PER_LANGUAGE = 5
LISTENING_SEED = 42


# =============================================================================
# ASR evaluator
# =============================================================================

ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

ASR_MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"

ASR_NEMO_FILENAME = "parakeet-tdt-0.6b-v3.nemo"


# =============================================================================
# MMS-TTS
# =============================================================================

MMS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "es": "facebook/mms-tts-spa",
    "fr": "facebook/mms-tts-fra",
    "de": "facebook/mms-tts-deu",
    "pt": "facebook/mms-tts-por",
}


# =============================================================================
# Piper
# =============================================================================

PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "es": "es_ES-sharvard-medium",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "pt": "pt_PT-tugão-medium",
}


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "tts_selection" / "tts_evaluation_manifest.jsonl"

PIPER_VOICE_DIR = PROJECT_ROOT / "data" / "tts_selection" / "piper_voices"

RESULTS_DIR = PROJECT_ROOT / "results" / "tts_selection" / "quality_multirun"

NATIVE_AUDIO_DIR = RESULTS_DIR / "audio_native"

ASR_AUDIO_DIR = RESULTS_DIR / "audio_asr_16k"

RAW_RESULTS_PATH = RESULTS_DIR / "tts_quality_results.jsonl"

RUN_SUMMARY_PATH = RESULTS_DIR / "tts_quality_by_run.csv"

LANGUAGE_SUMMARY_PATH = RESULTS_DIR / "tts_quality_by_language.csv"

GLOBAL_SUMMARY_PATH = RESULTS_DIR / "tts_quality_summary.csv"

CASE_COMPARISON_PATH = RESULTS_DIR / "tts_quality_case_comparison.csv"

SUMMARY_JSON_PATH = RESULTS_DIR / "tts_quality_summary.json"


# =============================================================================
# Listening test
# =============================================================================

LISTENING_DIR = RESULTS_DIR / "listening_test"

LISTENING_AUDIO_DIR = LISTENING_DIR / "audio"

LISTENING_SHEET_PATH = LISTENING_DIR / "listening_sheet.csv"

LISTENING_KEY_PATH = LISTENING_DIR / "listening_key.json"


# =============================================================================
# Generic helpers
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

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path.name}, line {line_number}.") from exc

            if not isinstance(
                row,
                dict,
            ):
                raise RuntimeError(f"Expected JSON object in {path.name}, line {line_number}.")

            rows.append(row)

    return rows


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    with path.open(
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
                )
                + "\n"
            )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()

        writer.writerows(rows)


def clear_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def seed_piper_runtime(
    seed: int,
) -> None:
    """
    Fix the intended random seed for a Piper synthesis run.

    ONNX Runtime does not guarantee complete deterministic reproducibility.
    The seed is therefore part of the experimental protocol, not a claim
    that identical PCM will be produced across executions.
    """

    random.seed(seed)

    np.random.seed(seed)

    ort.set_seed(seed)


def sample_std(
    values: list[float],
) -> float | None:
    if len(values) <= 1:
        return None

    return statistics.stdev(values)


# =============================================================================
# Environment
# =============================================================================


def validate_environment() -> None:
    print("=" * 80)
    print("TTS MULTI-RUN QUALITY EVALUATION")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")

    print(f"PyTorch      : {torch.__version__}")

    print(f"Transformers : {transformers.__version__}")

    print(f"NeMo         : {nemo.__version__}")

    print(f"SciPy        : {scipy.__version__}")

    print(f"ONNX Runtime : {ort.__version__}")

    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    providers = ort.get_available_providers()

    print(f"ORT providers: {providers}")

    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is not available.")

    print()
    print(f"MMS seed       : {MMS_SEED}")

    print(f"Piper runs     : {PIPER_RUNS}")

    print(f"Piper seeds    : {PIPER_RUN_SEEDS}")

    print(f"Listening seed : {LISTENING_SEED}")

    print()
    print("Environment validation: PASS")


# =============================================================================
# Manifest
# =============================================================================


def load_and_validate_manifest() -> list[dict]:
    print()
    print("=" * 80)
    print("FIXED TTS EVALUATION MANIFEST")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found:\n{MANIFEST_PATH}")

    manifest_sha256 = sha256_file(MANIFEST_PATH)

    print(f"Path   : {MANIFEST_PATH}")

    print(f"SHA256 : {manifest_sha256}")

    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Unexpected manifest SHA256.\nExpected: {EXPECTED_MANIFEST_SHA256}\nActual  : {manifest_sha256}"
        )

    rows = load_jsonl(MANIFEST_PATH)

    if len(rows) != EXPECTED_CASES:
        raise RuntimeError("Unexpected manifest size.")

    required_fields = {
        "tts_case_id",
        "case_index",
        "source_language",
        "target_language",
        "text",
        "character_count",
        "word_count",
    }

    case_ids = set()
    target_counts = Counter()

    for row in rows:
        missing = required_fields - set(row.keys())

        if missing:
            raise RuntimeError(f"Missing fields: {sorted(missing)}")

        case_id = str(row["tts_case_id"])

        if case_id in case_ids:
            raise RuntimeError(f"Duplicate case ID: {case_id}")

        case_ids.add(case_id)

        language = str(row["target_language"])

        if language not in LANGUAGES:
            raise RuntimeError(f"Unexpected target language: {language}")

        target_counts[language] += 1

        if not str(row["text"]).strip():
            raise RuntimeError(f"Empty TTS text: {case_id}")

    for language in LANGUAGES:
        if target_counts[language] != EXPECTED_CASES_PER_LANGUAGE:
            raise RuntimeError(
                f"{language} has {target_counts[language]} cases; expected {EXPECTED_CASES_PER_LANGUAGE}."
            )

    rows.sort(key=lambda row: int(row["case_index"]))

    print(f"Cases : {len(rows)}")

    for language in LANGUAGES:
        print(f"{language.upper():>2}    : {target_counts[language]}")

    print()
    print("Manifest validation: PASS")

    return rows


# =============================================================================
# WAV helpers
# =============================================================================


def float_to_pcm16(
    waveform: np.ndarray,
) -> np.ndarray:
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    if waveform.size == 0:
        raise RuntimeError("Empty waveform.")

    if not np.isfinite(waveform).all():
        raise RuntimeError("Waveform contains NaN or Inf.")

    waveform = np.clip(
        waveform,
        -1.0,
        1.0,
    )

    return (waveform * 32767.0).astype(np.int16)


def write_pcm16_mono_wav(
    path: Path,
    waveform: np.ndarray,
    sample_rate: int,
) -> None:
    pcm = float_to_pcm16(waveform)

    with wave.open(
        str(path),
        "wb",
    ) as wav_file:
        wav_file.setnchannels(1)

        wav_file.setsampwidth(2)

        wav_file.setframerate(sample_rate)

        wav_file.writeframes(pcm.tobytes())


def write_raw_pcm_wav(
    path: Path,
    pcm_bytes: bytes,
    sample_rate: int,
    sample_width: int,
    channels: int,
) -> None:
    with wave.open(
        str(path),
        "wb",
    ) as wav_file:
        wav_file.setnchannels(channels)

        wav_file.setsampwidth(sample_width)

        wav_file.setframerate(sample_rate)

        wav_file.writeframes(pcm_bytes)


def read_pcm16_wav(
    path: Path,
) -> tuple[
    np.ndarray,
    int,
]:
    with wave.open(
        str(path),
        "rb",
    ) as wav_file:
        channels = wav_file.getnchannels()

        sample_width = wav_file.getsampwidth()

        sample_rate = wav_file.getframerate()

        frame_count = wav_file.getnframes()

        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(f"{path.name}: expected 16-bit PCM.")

    if frame_count <= 0:
        raise RuntimeError(f"{path.name}: empty WAV.")

    pcm = np.frombuffer(
        raw,
        dtype=np.int16,
    )

    if channels > 1:
        pcm = pcm.reshape(
            -1,
            channels,
        )

        waveform = pcm.astype(np.float32).mean(axis=1) / 32768.0

    else:
        waveform = pcm.astype(np.float32) / 32768.0

    return (
        waveform,
        sample_rate,
    )


def prepare_asr_audio(
    native_path: Path,
    output_path: Path,
) -> dict:
    waveform, source_rate = read_pcm16_wav(native_path)

    source_samples = int(waveform.size)

    if source_rate != ASR_SAMPLE_RATE:
        divisor = math.gcd(
            source_rate,
            ASR_SAMPLE_RATE,
        )

        up = ASR_SAMPLE_RATE // divisor

        down = source_rate // divisor

        waveform = resample_poly(
            waveform,
            up,
            down,
        ).astype(np.float32)

    write_pcm16_mono_wav(
        path=output_path,
        waveform=waveform,
        sample_rate=(ASR_SAMPLE_RATE),
    )

    return {
        "source_sample_rate": (source_rate),
        "source_samples": (source_samples),
        "asr_sample_rate": (ASR_SAMPLE_RATE),
        "asr_samples": int(waveform.size),
        "duration_s": (waveform.size / ASR_SAMPLE_RATE),
    }


# =============================================================================
# MMS generation
# =============================================================================


def generate_mms_audio(
    manifest: list[dict],
) -> tuple[
    list[dict],
    list[dict],
]:
    print()
    print("=" * 80)
    print("GENERATING MMS-TTS AUDIO")
    print("=" * 80)

    records = []
    metadata = []

    for language in LANGUAGES:
        model_id = MMS_MODELS[language]

        cases = [row for row in manifest if row["target_language"] == language]

        native_dir = NATIVE_AUDIO_DIR / "mms" / language

        asr_dir = ASR_AUDIO_DIR / "mms" / language

        native_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        asr_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("-" * 80)
        print(f"MMS-TTS | {language.upper()}")
        print("-" * 80)

        clear_cuda()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
        )

        model = AutoModelForTextToWaveform.from_pretrained(
            model_id,
            local_files_only=True,
        )

        model.to(DEVICE)

        model.eval()

        actual_device = str(next(model.parameters()).device)

        actual_dtype = next(model.parameters()).dtype

        if actual_device != DEVICE:
            raise RuntimeError(f"{model_id}: model is on {actual_device}, expected {DEVICE}.")

        if actual_dtype != torch.float32:
            raise RuntimeError(f"{model_id}: model dtype is {actual_dtype}, expected torch.float32.")

        revision = getattr(
            model.config,
            "_commit_hash",
            None,
        )

        sample_rate = int(model.config.sampling_rate)

        print(f"Model       : {model_id}")

        print(f"Revision    : {revision}")

        print(f"Seed        : {MMS_SEED}")

        print(f"Sample rate : {sample_rate} Hz")

        for index, case in enumerate(
            cases,
            start=1,
        ):
            set_seed(MMS_SEED)

            text = str(case["text"])

            inputs = tokenizer(
                text=text,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.inference_mode():
                output = model(**inputs)

            torch.cuda.synchronize()

            waveform = output.waveform.detach().float().cpu().numpy().reshape(-1)

            native_path = native_dir / f"{case['tts_case_id']}.wav"

            asr_path = asr_dir / f"{case['tts_case_id']}.wav"

            write_pcm16_mono_wav(
                path=native_path,
                waveform=waveform,
                sample_rate=(sample_rate),
            )

            audio_info = prepare_asr_audio(
                native_path=(native_path),
                output_path=(asr_path),
            )

            records.append(
                {
                    "family": ("MMS-TTS"),
                    "run_index": (1),
                    "synthesis_seed": (MMS_SEED),
                    "language": (language),
                    "tts_case_id": (case["tts_case_id"]),
                    "case_index": int(case["case_index"]),
                    "source_language": (case["source_language"]),
                    "target_language": (case["target_language"]),
                    "reference_text": (text),
                    "native_audio_path": (str(native_path)),
                    "native_audio_sha256": (sha256_file(native_path)),
                    "asr_audio_path": (str(asr_path)),
                    "asr_audio_sha256": (sha256_file(asr_path)),
                    "native_sample_rate": (audio_info["source_sample_rate"]),
                    "asr_sample_rate": (ASR_SAMPLE_RATE),
                    "audio_duration_s": (audio_info["duration_s"]),
                }
            )

            print(f"[{index:02d}/{EXPECTED_CASES_PER_LANGUAGE}] ")

            del output
            del inputs

        metadata.append(
            {
                "family": ("MMS-TTS"),
                "run_index": (1),
                "synthesis_seed": (MMS_SEED),
                "language": (language),
                "model_or_voice": (model_id),
                "revision": (revision),
                "precision": ("float32"),
                "runtime": ("PyTorch CUDA"),
            }
        )

        del model
        del tokenizer

        clear_cuda()

    return (
        records,
        metadata,
    )


# =============================================================================
# Piper generation
# =============================================================================


def get_piper_paths(
    language: str,
) -> tuple[
    Path,
    Path,
]:
    voice_name = PIPER_VOICES[language]

    return (
        PIPER_VOICE_DIR / f"{voice_name}.onnx",
        PIPER_VOICE_DIR / f"{voice_name}.onnx.json",
    )


def load_piper_voice(
    language: str,
    seed: int,
) -> PiperVoice:
    voice_name = PIPER_VOICES[language]

    (
        model_path,
        config_path,
    ) = get_piper_paths(language)

    if not model_path.exists():
        raise FileNotFoundError(f"Piper model not found:\n{model_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Piper config not found:\n{config_path}")

    seed_piper_runtime(seed)

    voice = PiperVoice.load(
        model_path,
        config_path=(config_path),
        use_cuda=True,
    )

    providers = voice.session.get_providers()

    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"{voice_name}: CUDA is not the preferred provider.\nProviders: {providers}")

    return voice


def synthesize_piper_pcm(
    voice: PiperVoice,
    text: str,
) -> tuple[
    bytes,
    int,
    int,
    int,
]:
    audio = bytearray()

    sample_rate = None
    sample_width = None
    channels = None

    for chunk in voice.synthesize(text):
        if sample_rate is None:
            sample_rate = int(chunk.sample_rate)

            sample_width = int(chunk.sample_width)

            channels = int(chunk.sample_channels)

        else:
            if int(chunk.sample_rate) != sample_rate:
                raise RuntimeError("Piper sample rate changed between chunks.")

            if int(chunk.sample_width) != sample_width:
                raise RuntimeError("Piper sample width changed between chunks.")

            if int(chunk.sample_channels) != channels:
                raise RuntimeError("Piper channel count changed between chunks.")

        audio.extend(chunk.audio_int16_bytes)

    if not audio:
        raise RuntimeError("Piper produced no audio.")

    if sample_rate is None or sample_width is None or channels is None:
        raise RuntimeError("Piper audio format is missing.")

    return (
        bytes(audio),
        sample_rate,
        sample_width,
        channels,
    )


def generate_piper_audio(
    manifest: list[dict],
) -> tuple[
    list[dict],
    list[dict],
]:
    print()
    print("=" * 80)
    print("GENERATING PIPER MULTI-RUN AUDIO")
    print("=" * 80)

    records = []
    metadata = []

    for run_index, seed in enumerate(
        PIPER_RUN_SEEDS,
        start=1,
    ):
        print()
        print("=" * 80)

        print(f"PIPER RUN {run_index}/{PIPER_RUNS}")

        print(f"Seed: {seed}")

        print("=" * 80)

        for language in LANGUAGES:
            voice_name = PIPER_VOICES[language]

            model_path, config_path = get_piper_paths(language)

            model_sha256 = sha256_file(model_path)

            config_sha256 = sha256_file(config_path)

            cases = [row for row in manifest if row["target_language"] == language]

            native_dir = NATIVE_AUDIO_DIR / "piper" / f"run_{run_index:02d}" / language

            asr_dir = ASR_AUDIO_DIR / "piper" / f"run_{run_index:02d}" / language

            native_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            asr_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            print()
            print("-" * 80)

            print(f"PIPER RUN {run_index} | {language.upper()}")

            print("-" * 80)

            clear_cuda()

            voice = load_piper_voice(
                language=(language),
                seed=(seed),
            )

            providers = voice.session.get_providers()

            print(f"Voice     : {voice_name}")

            print(f"Seed      : {seed}")

            print(f"Providers : {providers}")

            for index, case in enumerate(
                cases,
                start=1,
            ):
                text = str(case["text"])

                (
                    pcm_bytes,
                    sample_rate,
                    sample_width,
                    channels,
                ) = synthesize_piper_pcm(
                    voice=voice,
                    text=text,
                )

                native_path = native_dir / f"{case['tts_case_id']}.wav"

                asr_path = asr_dir / f"{case['tts_case_id']}.wav"

                write_raw_pcm_wav(
                    path=native_path,
                    pcm_bytes=(pcm_bytes),
                    sample_rate=(sample_rate),
                    sample_width=(sample_width),
                    channels=(channels),
                )

                audio_info = prepare_asr_audio(
                    native_path=(native_path),
                    output_path=(asr_path),
                )

                records.append(
                    {
                        "family": ("Piper"),
                        "run_index": (run_index),
                        "synthesis_seed": (seed),
                        "language": (language),
                        "tts_case_id": (case["tts_case_id"]),
                        "case_index": int(case["case_index"]),
                        "source_language": (case["source_language"]),
                        "target_language": (case["target_language"]),
                        "reference_text": (text),
                        "native_audio_path": (str(native_path)),
                        "native_audio_sha256": (sha256_file(native_path)),
                        "asr_audio_path": (str(asr_path)),
                        "asr_audio_sha256": (sha256_file(asr_path)),
                        "native_sample_rate": (audio_info["source_sample_rate"]),
                        "asr_sample_rate": (ASR_SAMPLE_RATE),
                        "audio_duration_s": (audio_info["duration_s"]),
                    }
                )

                print(f"[{index:02d}/{EXPECTED_CASES_PER_LANGUAGE}] ")

            metadata.append(
                {
                    "family": ("Piper"),
                    "run_index": (run_index),
                    "synthesis_seed": (seed),
                    "language": (language),
                    "model_or_voice": (voice_name),
                    "revision": (None),
                    "model_sha256": (model_sha256),
                    "config_sha256": (config_sha256),
                    "precision": ("ONNX model default"),
                    "runtime": ("ONNX Runtime CUDA"),
                    "providers": (providers),
                }
            )

            del voice

            clear_cuda()

    return (
        records,
        metadata,
    )


# =============================================================================
# ASR normalization
# =============================================================================


def normalize_asr_text(
    text: str,
) -> str:
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    characters = []

    for character in text:
        category = unicodedata.category(character)

        if category.startswith("P"):
            characters.append(" ")

        else:
            characters.append(character)

    text = "".join(characters)

    return " ".join(text.split())


# =============================================================================
# Error metrics
# =============================================================================


def edit_distance(
    reference: list,
    hypothesis: list,
) -> int:
    previous = list(range(len(hypothesis) + 1))

    for reference_index, reference_item in enumerate(
        reference,
        start=1,
    ):
        current = [reference_index]

        for hypothesis_index, hypothesis_item in enumerate(
            hypothesis,
            start=1,
        ):
            substitution_cost = 0 if reference_item == hypothesis_item else 1

            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )

        previous = current

    return previous[-1]


def calculate_error_metrics(
    reference: str,
    hypothesis: str,
) -> dict:
    normalized_reference = normalize_asr_text(reference)

    normalized_hypothesis = normalize_asr_text(hypothesis)

    reference_words = normalized_reference.split()

    hypothesis_words = normalized_hypothesis.split()

    if not reference_words:
        raise RuntimeError("Normalized reference has zero words.")

    word_edits = edit_distance(
        reference_words,
        hypothesis_words,
    )

    reference_characters = list(normalized_reference)

    hypothesis_characters = list(normalized_hypothesis)

    if not reference_characters:
        raise RuntimeError("Normalized reference has zero characters.")

    character_edits = edit_distance(
        reference_characters,
        hypothesis_characters,
    )

    wer = word_edits / len(reference_words)

    cer = character_edits / len(reference_characters)

    return {
        "normalized_reference": (normalized_reference),
        "normalized_hypothesis": (normalized_hypothesis),
        "word_reference_count": (len(reference_words)),
        "word_edits": (word_edits),
        "wer": (wer),
        "wer_percent": (wer * 100.0),
        "character_reference_count": (len(reference_characters)),
        "character_edits": (character_edits),
        "cer": (cer),
        "cer_percent": (cer * 100.0),
        "exact_normalized_match": (normalized_reference == normalized_hypothesis),
    }


# =============================================================================
# ASR evaluator
# =============================================================================


def load_asr_evaluator():
    print()
    print("=" * 80)
    print("LOADING PARAKEET ASR EVALUATOR")
    print("=" * 80)

    clear_cuda()

    print(f"Model ID : {ASR_MODEL_ID}")

    print(f"Revision : {ASR_MODEL_REVISION}")

    nemo_path = hf_hub_download(
        repo_id=(ASR_MODEL_ID),
        filename=(ASR_NEMO_FILENAME),
        revision=(ASR_MODEL_REVISION),
    )

    print(f"Checkpoint: {nemo_path}")

    start = time.perf_counter()

    model = cast(
        nemo_asr.models.ASRModel,
        nemo_asr.models.ASRModel.restore_from(
            restore_path=(nemo_path),
            map_location=(torch.device(DEVICE)),
        ),
    )

    model.eval()

    torch.cuda.synchronize()

    end = time.perf_counter()

    print(f"Device    : {next(model.parameters()).device}")

    print(f"Load time : {(end - start) * 1000.0:.2f} ms")

    print("ASR evaluator load: PASS")

    return model


def extract_transcription_text(
    output,
) -> str:
    if isinstance(
        output,
        str,
    ):
        return output

    if hasattr(
        output,
        "text",
    ):
        return str(output.text)

    raise RuntimeError(f"Unexpected ASR output type: {type(output)}")


def evaluate_with_asr(
    model,
    generation_records: list[dict],
) -> list[dict]:
    print()
    print("=" * 80)
    print("ASR INTELLIGIBILITY EVALUATION")
    print("=" * 80)

    expected_total = EXPECTED_CASES + EXPECTED_CASES * PIPER_RUNS

    if len(generation_records) != expected_total:
        raise RuntimeError(
            f"Unexpected generated audio count.\nExpected: {expected_total}\nActual  : {len(generation_records)}"
        )

    ordered_records = sorted(
        generation_records,
        key=lambda row: (
            row["family"],
            int(row["run_index"]),
            row["language"],
            int(row["case_index"]),
        ),
    )

    results = []

    for batch_start in range(
        0,
        len(ordered_records),
        ASR_BATCH_SIZE,
    ):
        batch_records = ordered_records[batch_start : batch_start + ASR_BATCH_SIZE]

        audio_paths = [row["asr_audio_path"] for row in batch_records]

        with torch.inference_mode():
            outputs = model.transcribe(
                audio=audio_paths,
                batch_size=(ASR_BATCH_SIZE),
                return_hypotheses=False,
                num_workers=0,
                verbose=False,
            )

        if len(outputs) != len(batch_records):
            raise RuntimeError("ASR output count mismatch.")

        for record, output in zip(
            batch_records,
            outputs,
        ):
            hypothesis = extract_transcription_text(output)

            metrics = calculate_error_metrics(
                reference=(record["reference_text"]),
                hypothesis=(hypothesis),
            )

            result = {
                **record,
                "asr_model_id": (ASR_MODEL_ID),
                "asr_model_revision": (ASR_MODEL_REVISION),
                "asr_hypothesis": (hypothesis),
                **metrics,
            }

            results.append(result)

            print(
                f"[{len(results):03d}/{expected_total}] "
                f"{record['family']:<7} "
                f"R{record['run_index']} "
                f"{record['language'].upper()} "
                f"{record['tts_case_id']} "
                f"| WER "
                f"{metrics['wer_percent']:.2f}% "
                f"| CER "
                f"{metrics['cer_percent']:.2f}%"
            )

    return results


# =============================================================================
# Summary metrics
# =============================================================================


def summarize_group(
    rows: list[dict],
) -> dict:
    if not rows:
        raise RuntimeError("Cannot summarize empty group.")

    wers = [float(row["wer_percent"]) for row in rows]

    cers = [float(row["cer_percent"]) for row in rows]

    total_word_edits = sum(int(row["word_edits"]) for row in rows)

    total_reference_words = sum(int(row["word_reference_count"]) for row in rows)

    total_character_edits = sum(int(row["character_edits"]) for row in rows)

    total_reference_characters = sum(int(row["character_reference_count"]) for row in rows)

    exact_matches = sum(1 for row in rows if row["exact_normalized_match"])

    return {
        "cases": (len(rows)),
        "macro_wer_percent": (statistics.mean(wers)),
        "median_wer_percent": (statistics.median(wers)),
        "micro_wer_percent": (total_word_edits / total_reference_words * 100.0),
        "macro_cer_percent": (statistics.mean(cers)),
        "median_cer_percent": (statistics.median(cers)),
        "micro_cer_percent": (total_character_edits / total_reference_characters * 100.0),
        "exact_normalized_matches": (exact_matches),
        "exact_normalized_match_percent": (exact_matches / len(rows) * 100.0),
    }


def build_run_summaries(
    results: list[dict],
) -> list[dict]:
    summaries = []

    experiment_runs = [
        (
            "MMS-TTS",
            1,
            MMS_SEED,
        )
    ]

    experiment_runs.extend(
        (
            "Piper",
            run_index,
            seed,
        )
        for run_index, seed in enumerate(
            PIPER_RUN_SEEDS,
            start=1,
        )
    )

    for (
        family,
        run_index,
        seed,
    ) in experiment_runs:
        run_rows = [row for row in results if (row["family"] == family and int(row["run_index"]) == run_index)]

        if len(run_rows) != EXPECTED_CASES:
            raise RuntimeError(f"{family} run {run_index}: expected {EXPECTED_CASES} rows, got {len(run_rows)}.")

        for language in LANGUAGES:
            language_rows = [row for row in run_rows if row["language"] == language]

            if len(language_rows) != EXPECTED_CASES_PER_LANGUAGE:
                raise RuntimeError(
                    f"{family} run {run_index} {language}: expected {EXPECTED_CASES_PER_LANGUAGE} rows, got {len(language_rows)}."
                )

            summaries.append(
                {
                    "family": (family),
                    "run_index": (run_index),
                    "synthesis_seed": (seed),
                    "scope": ("language"),
                    "language": (language),
                    **summarize_group(language_rows),
                }
            )

        summaries.append(
            {
                "family": (family),
                "run_index": (run_index),
                "synthesis_seed": (seed),
                "scope": ("global"),
                "language": ("ALL"),
                **summarize_group(run_rows),
            }
        )

    return summaries


def aggregate_metric(
    run_rows: list[dict],
    metric: str,
) -> dict:
    values = [float(row[metric]) for row in run_rows]

    return {
        f"mean_{metric}": (statistics.mean(values)),
        f"std_{metric}": (sample_std(values)),
        f"min_{metric}": (min(values)),
        f"max_{metric}": (max(values)),
    }


def build_aggregate_row(
    family: str,
    run_rows: list[dict],
    language: str,
) -> dict:
    if not run_rows:
        raise RuntimeError("Cannot aggregate empty run set.")

    output = {
        "family": (family),
        "language": (language),
        "runs": (len(run_rows)),
        "cases_per_run": int(run_rows[0]["cases"]),
    }

    metrics = [
        "macro_wer_percent",
        "micro_wer_percent",
        "macro_cer_percent",
        "micro_cer_percent",
        "exact_normalized_match_percent",
    ]

    for metric in metrics:
        output.update(
            aggregate_metric(
                run_rows,
                metric,
            )
        )

    return output


def build_language_aggregates(
    run_summaries: list[dict],
) -> list[dict]:
    rows = []

    for family in [
        "MMS-TTS",
        "Piper",
    ]:
        for language in LANGUAGES:
            matching = [
                row
                for row in run_summaries
                if (row["family"] == family and row["scope"] == "language" and row["language"] == language)
            ]

            expected_runs = 1 if family == "MMS-TTS" else PIPER_RUNS

            if len(matching) != expected_runs:
                raise RuntimeError(f"Unexpected aggregate run count for {family}/{language}.")

            rows.append(
                build_aggregate_row(
                    family=(family),
                    run_rows=(matching),
                    language=(language),
                )
            )

    return rows


def build_global_aggregates(
    run_summaries: list[dict],
) -> list[dict]:
    rows = []

    for family in [
        "MMS-TTS",
        "Piper",
    ]:
        matching = [row for row in run_summaries if (row["family"] == family and row["scope"] == "global")]

        expected_runs = 1 if family == "MMS-TTS" else PIPER_RUNS

        if len(matching) != expected_runs:
            raise RuntimeError(f"Unexpected global run count for {family}.")

        rows.append(
            build_aggregate_row(
                family=(family),
                run_rows=(matching),
                language=("ALL"),
            )
        )

    return rows


# =============================================================================
# Paired case comparison
# =============================================================================


def build_case_comparison(
    results: list[dict],
) -> tuple[
    list[dict],
    dict,
]:
    comparison = []

    wins = {
        "MMS-TTS": 0,
        "Piper": 0,
        "Tie": 0,
    }

    by_language = {
        language: {
            "MMS-TTS": 0,
            "Piper": 0,
            "Tie": 0,
        }
        for language in LANGUAGES
    }

    case_ids = sorted({row["tts_case_id"] for row in results})

    if len(case_ids) != EXPECTED_CASES:
        raise RuntimeError("Unexpected unique case count.")

    for case_id in case_ids:
        mms_rows = [row for row in results if (row["tts_case_id"] == case_id and row["family"] == "MMS-TTS")]

        piper_rows = [row for row in results if (row["tts_case_id"] == case_id and row["family"] == "Piper")]

        if len(mms_rows) != 1:
            raise RuntimeError(f"{case_id}: expected one MMS result.")

        if len(piper_rows) != PIPER_RUNS:
            raise RuntimeError(f"{case_id}: expected {PIPER_RUNS} Piper results.")

        mms = mms_rows[0]

        piper_wers = [float(row["wer_percent"]) for row in piper_rows]

        piper_cers = [float(row["cer_percent"]) for row in piper_rows]

        piper_mean_wer = statistics.mean(piper_wers)

        mms_wer = float(mms["wer_percent"])

        tolerance = 1e-12

        if mms_wer + tolerance < piper_mean_wer:
            winner = "MMS-TTS"

        elif piper_mean_wer + tolerance < mms_wer:
            winner = "Piper"

        else:
            winner = "Tie"

        language = mms["language"]

        wins[winner] += 1

        by_language[language][winner] += 1

        comparison.append(
            {
                "tts_case_id": (case_id),
                "case_index": (mms["case_index"]),
                "language": (language),
                "mms_wer_percent": (mms_wer),
                "mms_cer_percent": (mms["cer_percent"]),
                "piper_mean_wer_percent": (piper_mean_wer),
                "piper_std_wer_percent": (sample_std(piper_wers)),
                "piper_min_wer_percent": (min(piper_wers)),
                "piper_max_wer_percent": (max(piper_wers)),
                "piper_mean_cer_percent": (statistics.mean(piper_cers)),
                "piper_std_cer_percent": (sample_std(piper_cers)),
                "wer_difference_piper_minus_mms": (piper_mean_wer - mms_wer),
                "winner_lower_wer": (winner),
            }
        )

    return (
        comparison,
        {
            "global": (wins),
            "by_language": (by_language),
        },
    )


# =============================================================================
# Listening test
# =============================================================================


def select_listening_cases(
    manifest: list[dict],
) -> list[dict]:
    selected = []

    for language in LANGUAGES:
        rows = [row for row in manifest if row["target_language"] == language]

        rows.sort(
            key=lambda row: (
                int(row["word_count"]),
                int(row["case_index"]),
            )
        )

        positions = [
            0,
            4,
            9,
            14,
            19,
        ]

        for position in positions:
            selected.append(rows[position])

    expected = len(LANGUAGES) * LISTENING_CASES_PER_LANGUAGE

    if len(selected) != expected:
        raise RuntimeError("Unexpected listening-test selection size.")

    return selected


def prepare_listening_test(
    manifest: list[dict],
) -> None:
    print()
    print("=" * 80)
    print("PREPARING BLIND A/B LISTENING TEST")
    print("=" * 80)

    selected = select_listening_cases(manifest)

    LISTENING_AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(LISTENING_SEED)

    sheet_rows = []
    key_rows = []

    for pair_index, case in enumerate(
        selected,
        start=1,
    ):
        language = case["target_language"]

        case_id = case["tts_case_id"]

        # Round-robin Piper run allocation.
        #
        # Because each language contributes exactly five listening pairs,
        # every target language contains exactly one sample from each of
        # the five Piper synthesis runs.
        piper_run = (pair_index - 1) % PIPER_RUNS + 1

        piper_seed = PIPER_RUN_SEEDS[piper_run - 1]

        mms_path = NATIVE_AUDIO_DIR / "mms" / language / f"{case_id}.wav"

        piper_path = NATIVE_AUDIO_DIR / "piper" / f"run_{piper_run:02d}" / language / f"{case_id}.wav"

        if not mms_path.exists():
            raise FileNotFoundError(str(mms_path))

        if not piper_path.exists():
            raise FileNotFoundError(str(piper_path))

        if rng.random() < 0.5:
            model_a = "MMS-TTS"

            model_b = "Piper"

            source_a = mms_path

            source_b = piper_path

        else:
            model_a = "Piper"

            model_b = "MMS-TTS"

            source_a = piper_path

            source_b = mms_path

        pair_id = f"pair_{pair_index:03d}"

        destination_a = LISTENING_AUDIO_DIR / f"{pair_id}_A.wav"

        destination_b = LISTENING_AUDIO_DIR / f"{pair_id}_B.wav"

        shutil.copy2(
            source_a,
            destination_a,
        )

        shutil.copy2(
            source_b,
            destination_b,
        )

        sheet_rows.append(
            {
                "pair_id": (pair_id),
                "target_language": (language),
                "tts_case_id": (case_id),
                "text": (case["text"]),
                "A_intelligibility_1_5": "",
                "A_naturalness_1_5": "",
                "A_pronunciation_1_5": "",
                "B_intelligibility_1_5": "",
                "B_naturalness_1_5": "",
                "B_pronunciation_1_5": "",
                "preference_A_B_tie": "",
                "notes": "",
            }
        )

        key_rows.append(
            {
                "pair_id": (pair_id),
                "target_language": (language),
                "tts_case_id": (case_id),
                "A": (model_a),
                "B": (model_b),
                "piper_run": (piper_run),
                "piper_seed": (piper_seed),
                "A_sha256": (sha256_file(destination_a)),
                "B_sha256": (sha256_file(destination_b)),
            }
        )

        print(f"{pair_id} | {language.upper()} | {case_id} | Piper run {piper_run}")

    write_csv(
        LISTENING_SHEET_PATH,
        sheet_rows,
    )

    LISTENING_KEY_PATH.write_text(
        json.dumps(
            {
                "assignment_seed": (LISTENING_SEED),
                "piper_run_policy": (
                    "Round-robin across runs 1-5. Each target language uses one example from every Piper run."
                ),
                "pairs": (key_rows),
                "instructions": ("Do not inspect this file before completing the blind listening evaluation."),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(f"Listening pairs : {len(sheet_rows)}")

    print(f"Audio files     : {len(sheet_rows) * 2}")

    print("Listening package: PASS")


# =============================================================================
# Console reporting
# =============================================================================


def format_mean_std(
    mean_value: float,
    std_value: float | None,
) -> str:
    if std_value is None:
        return f"{mean_value:.2f}"

    return f"{mean_value:.2f} ± {std_value:.2f}"


def print_piper_run_summary(
    run_summaries: list[dict],
) -> None:
    print()
    print("=" * 100)
    print("PIPER GLOBAL RESULTS BY SYNTHESIS RUN")
    print("=" * 100)

    print(f"{'Run':<8}{'Seed':<10}{'Macro WER':>14}{'Micro WER':>14}{'Macro CER':>14}{'Exact':>12}")

    print("-" * 100)

    rows = [row for row in run_summaries if (row["family"] == "Piper" and row["scope"] == "global")]

    rows.sort(key=lambda row: int(row["run_index"]))

    for row in rows:
        print(
            f"{row['run_index']:<8}"
            f"{row['synthesis_seed']:<10}"
            f"{row['macro_wer_percent']:>13.2f}%"
            f"{row['micro_wer_percent']:>13.2f}%"
            f"{row['macro_cer_percent']:>13.2f}%"
            f"{row['exact_normalized_match_percent']:>11.1f}%"
        )


def print_language_summary(
    rows: list[dict],
) -> None:
    print()
    print("=" * 120)
    print("TTS INTELLIGIBILITY BY TARGET LANGUAGE")
    print("=" * 120)

    print(
        f"{'Family':<12}{'Lang':<8}{'Runs':>6}{'Macro WER mean ± SD':>24}{'Micro WER':>16}{'Macro CER mean ± SD':>24}"
    )

    print("-" * 120)

    for row in rows:
        macro_wer = format_mean_std(
            row["mean_macro_wer_percent"],
            row["std_macro_wer_percent"],
        )

        macro_cer = format_mean_std(
            row["mean_macro_cer_percent"],
            row["std_macro_cer_percent"],
        )

        print(
            f"{row['family']:<12}"
            f"{row['language'].upper():<8}"
            f"{row['runs']:>6}"
            f"{macro_wer:>24}"
            f"{row['mean_micro_wer_percent']:>15.2f}%"
            f"{macro_cer:>24}"
        )


def print_global_summary(
    rows: list[dict],
) -> None:
    print()
    print("=" * 120)
    print("GLOBAL TTS INTELLIGIBILITY SUMMARY")
    print("=" * 120)

    print(
        f"{'Family':<14}"
        f"{'Runs':>8}"
        f"{'Macro WER mean ± SD':>26}"
        f"{'Micro WER mean':>18}"
        f"{'Macro CER mean ± SD':>26}"
        f"{'Exact mean':>14}"
    )

    print("-" * 120)

    for row in rows:
        macro_wer = format_mean_std(
            row["mean_macro_wer_percent"],
            row["std_macro_wer_percent"],
        )

        macro_cer = format_mean_std(
            row["mean_macro_cer_percent"],
            row["std_macro_cer_percent"],
        )

        print(
            f"{row['family']:<14}"
            f"{row['runs']:>8}"
            f"{macro_wer:>26}"
            f"{row['mean_micro_wer_percent']:>17.2f}%"
            f"{macro_cer:>26}"
            f"{row['mean_exact_normalized_match_percent']:>13.1f}%"
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    validate_environment()

    manifest = load_and_validate_manifest()

    # Start the official multi-run experiment from a clean output folder.
    # The previous exploratory `quality` directory is deliberately preserved.
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Generate MMS
    # ------------------------------------------------------------------

    (
        mms_records,
        mms_metadata,
    ) = generate_mms_audio(manifest)

    if len(mms_records) != EXPECTED_CASES:
        raise RuntimeError(f"Expected {EXPECTED_CASES} MMS samples.")

    # ------------------------------------------------------------------
    # Generate Piper multi-run
    # ------------------------------------------------------------------

    (
        piper_records,
        piper_metadata,
    ) = generate_piper_audio(manifest)

    expected_piper = EXPECTED_CASES * PIPER_RUNS

    if len(piper_records) != expected_piper:
        raise RuntimeError(f"Expected {expected_piper} Piper samples.")

    generation_records = mms_records + piper_records

    expected_total = EXPECTED_CASES + expected_piper

    print()
    print("=" * 80)
    print("AUDIO GENERATION VALIDATION")
    print("=" * 80)

    print(f"MMS samples   : {len(mms_records)}/{EXPECTED_CASES}")

    print(f"Piper samples : {len(piper_records)}/{expected_piper}")

    print(f"ASR inputs    : {len(generation_records)}/{expected_total}")

    print("All ASR inputs are 16 kHz mono PCM16: PASS")

    # ------------------------------------------------------------------
    # Load ASR evaluator
    # ------------------------------------------------------------------

    clear_cuda()

    asr_model = load_asr_evaluator()

    # ------------------------------------------------------------------
    # 600 ASR transcriptions
    # ------------------------------------------------------------------

    quality_results = evaluate_with_asr(
        model=(asr_model),
        generation_records=(generation_records),
    )

    if len(quality_results) != expected_total:
        raise RuntimeError(f"Expected {expected_total} quality results.")

    del asr_model

    clear_cuda()

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    run_summaries = build_run_summaries(quality_results)

    language_aggregates = build_language_aggregates(run_summaries)

    global_aggregates = build_global_aggregates(run_summaries)

    (
        case_comparison,
        paired_wins,
    ) = build_case_comparison(quality_results)

    # ------------------------------------------------------------------
    # Listening package
    # ------------------------------------------------------------------

    prepare_listening_test(manifest)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    write_jsonl(
        RAW_RESULTS_PATH,
        quality_results,
    )

    write_csv(
        RUN_SUMMARY_PATH,
        run_summaries,
    )

    write_csv(
        LANGUAGE_SUMMARY_PATH,
        language_aggregates,
    )

    write_csv(
        GLOBAL_SUMMARY_PATH,
        global_aggregates,
    )

    write_csv(
        CASE_COMPARISON_PATH,
        case_comparison,
    )

    summary = {
        "experiment": ("Multi-run TTS intelligibility evaluation using Parakeet ASR."),
        "status": ("official_multi_run_quality_evaluation"),
        "manifest": {
            "path": (str(MANIFEST_PATH)),
            "sha256": (EXPECTED_MANIFEST_SHA256),
            "cases": (EXPECTED_CASES),
        },
        "experimental_design": {
            "mms_runs": (1),
            "mms_seed": (MMS_SEED),
            "piper_runs": (PIPER_RUNS),
            "piper_run_seeds": (PIPER_RUN_SEEDS),
            "mms_samples": (EXPECTED_CASES),
            "piper_samples": (expected_piper),
            "total_asr_evaluations": (expected_total),
            "piper_stochasticity": (
                "A previous reproducibility gate "
                "showed that identical ORT seeds "
                "did not produce byte-identical "
                "Piper PCM on CUDA. Piper quality "
                "is therefore evaluated across "
                "five separate synthesis runs."
            ),
            "piper_seed_interpretation": (
                "Seeds define the intended run "
                "configuration but are not treated "
                "as a guarantee of byte-level "
                "reproducibility."
            ),
            "run_variability": (
                "Reported standard deviation is the sample standard deviation across the five Piper run-level metrics."
            ),
        },
        "tts": {
            "mms": {
                "models": (MMS_MODELS),
                "precision": ("float32"),
                "runtime": ("PyTorch CUDA"),
            },
            "piper": {
                "voices": (PIPER_VOICES),
                "runtime": ("ONNX Runtime CUDA"),
                "synthesis_parameters": (
                    "Voice defaults. noise_scale, length_scale and noise_w_scale are not modified."
                ),
            },
        },
        "asr_evaluator": {
            "model_id": (ASR_MODEL_ID),
            "revision": (ASR_MODEL_REVISION),
            "sample_rate": (ASR_SAMPLE_RATE),
            "batch_size": (ASR_BATCH_SIZE),
        },
        "normalization": {
            "unicode": ("NFKC"),
            "case": ("lowercase"),
            "punctuation": ("Unicode punctuation replaced with spaces"),
            "whitespace": ("collapsed"),
            "accents": ("preserved"),
            "number_normalization": (False),
        },
        "metrics": {
            "automatic": [
                "macro WER",
                "micro WER",
                "macro CER",
                "micro CER",
                "exact normalized match",
            ],
            "interpretation": (
                "ASR round-trip error is used "
                "as an objective proxy for TTS "
                "intelligibility. It is not a "
                "measure of naturalness."
            ),
        },
        "model_metadata": (mms_metadata + piper_metadata),
        "by_run": (run_summaries),
        "by_language": (language_aggregates),
        "global": (global_aggregates),
        "paired_case_comparison": (paired_wins),
        "listening_test": {
            "pairs": (len(LANGUAGES) * LISTENING_CASES_PER_LANGUAGE),
            "cases_per_language": (LISTENING_CASES_PER_LANGUAGE),
            "piper_run_assignment": (
                "Round-robin. Each target language includes exactly one listening sample from each Piper run."
            ),
            "ab_assignment_seed": (LISTENING_SEED),
            "criteria": [
                "intelligibility",
                "naturalness",
                "pronunciation",
                "overall preference",
            ],
            "sheet": (str(LISTENING_SHEET_PATH)),
            "hidden_key": (str(LISTENING_KEY_PATH)),
        },
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "transformers": (transformers.__version__),
            "nemo": (nemo.__version__),
            "scipy": (scipy.__version__),
            "onnxruntime": (ort.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
        },
        "target_hardware": {
            "device": ("NVIDIA Jetson Orin"),
            "memory_gb": (8),
            "note": (
                "Quality measurements are "
                "hardware-independent in intent; "
                "performance and memory must be "
                "remeasured on the target device."
            ),
        },
    }

    SUMMARY_JSON_PATH.write_text(
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

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    print_piper_run_summary(run_summaries)

    print_language_summary(language_aggregates)

    print_global_summary(global_aggregates)

    print()
    print("=" * 80)
    print("PAIRED CASE COMPARISON")
    print("=" * 80)

    print(f"MMS lower WER   : {paired_wins['global']['MMS-TTS']}")

    print(f"Piper lower WER : {paired_wins['global']['Piper']}")

    print(f"Ties            : {paired_wins['global']['Tie']}")

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Native audio      : {NATIVE_AUDIO_DIR}")

    print(f"ASR 16 kHz audio  : {ASR_AUDIO_DIR}")

    print(f"Raw results       : {RAW_RESULTS_PATH}")

    print(f"By run            : {RUN_SUMMARY_PATH}")

    print(f"By language       : {LANGUAGE_SUMMARY_PATH}")

    print(f"Global summary    : {GLOBAL_SUMMARY_PATH}")

    print(f"Case comparison   : {CASE_COMPARISON_PATH}")

    print(f"Summary JSON      : {SUMMARY_JSON_PATH}")

    print()

    print(f"Listening audio   : {LISTENING_AUDIO_DIR}")

    print(f"Listening sheet   : {LISTENING_SHEET_PATH}")

    print(f"Listening key     : {LISTENING_KEY_PATH}")

    print()
    print("=" * 80)
    print("TTS MULTI-RUN QUALITY EVALUATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
