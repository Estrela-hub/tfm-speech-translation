from __future__ import annotations

import gc
import json
import platform
import sys
import wave
from pathlib import Path

import numpy as np

# Import torch before ONNX Runtime so that the CUDA/cuDNN DLLs bundled
# with PyTorch are available to ORT on Windows.
import torch
import onnxruntime as ort
import transformers
from piper import PiperVoice
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

LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "pt",
]

TEST_SENTENCES = {
    "en": ("The meeting starts tomorrow morning at nine o'clock."),
    "es": ("La reunión comienza mañana por la mañana a las nueve."),
    "fr": ("La réunion commence demain matin à neuf heures."),
    "de": ("Die Besprechung beginnt morgen früh um neun Uhr."),
    "pt": ("A reunião começa amanhã de manhã às nove horas."),
}


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

PIPER_VOICE_DIR = PROJECT_ROOT / "data" / "tts_selection" / "piper_voices"

RESULTS_DIR = PROJECT_ROOT / "results" / "tts_selection" / "smoke_test"

AUDIO_DIR = RESULTS_DIR / "audio"

SUMMARY_PATH = RESULTS_DIR / "smoke_test_summary.json"


# =============================================================================
# Generic helpers
# =============================================================================


def validate_environment() -> None:
    print("=" * 80)
    print("TTS MODEL SMOKE TEST")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")

    print(f"PyTorch      : {torch.__version__}")

    print(f"Transformers : {transformers.__version__}")

    print(f"ONNX Runtime : {ort.__version__}")

    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    providers = ort.get_available_providers()

    print(f"ORT providers: {providers}")

    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is not available in ONNX Runtime.")

    print()
    print("Environment validation: PASS")


def validate_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    label: str,
) -> dict:
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    if waveform.size == 0:
        raise RuntimeError(f"{label}: empty waveform.")

    if not np.isfinite(waveform).all():
        raise RuntimeError(f"{label}: waveform contains NaN or Inf values.")

    if sample_rate <= 0:
        raise RuntimeError(f"{label}: invalid sample rate {sample_rate}.")

    peak = float(np.max(np.abs(waveform)))

    rms = float(np.sqrt(np.mean(waveform**2)))

    duration_s = waveform.size / sample_rate

    if duration_s <= 0.0:
        raise RuntimeError(f"{label}: invalid duration.")

    if peak <= 0.0:
        raise RuntimeError(f"{label}: waveform is silent.")

    return {
        "samples": int(waveform.size),
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "peak_abs": (peak),
        "rms": (rms),
    }


def write_float_wav(
    output_path: Path,
    waveform: np.ndarray,
    sample_rate: int,
) -> None:
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    # Avoid clipping while preserving relative amplitude.
    peak = float(np.max(np.abs(waveform)))

    if peak > 1.0:
        waveform = waveform / peak

    pcm = np.clip(
        waveform,
        -1.0,
        1.0,
    )

    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(
        str(output_path),
        "wb",
    ) as wav_file:
        wav_file.setnchannels(1)

        wav_file.setsampwidth(2)

        wav_file.setframerate(sample_rate)

        wav_file.writeframes(pcm.tobytes())


def validate_written_wav(
    path: Path,
) -> dict:
    if not path.exists():
        raise RuntimeError(f"WAV was not created: {path}")

    if path.stat().st_size <= 44:
        raise RuntimeError(f"WAV appears empty: {path}")

    with wave.open(
        str(path),
        "rb",
    ) as wav_file:
        channels = wav_file.getnchannels()

        sample_width = wav_file.getsampwidth()

        sample_rate = wav_file.getframerate()

        frames = wav_file.getnframes()

    if channels != 1:
        raise RuntimeError(f"{path.name}: expected mono audio, got {channels} channels.")

    if frames <= 0:
        raise RuntimeError(f"{path.name}: no audio frames.")

    if sample_rate <= 0:
        raise RuntimeError(f"{path.name}: invalid sample rate.")

    return {
        "channels": (channels),
        "sample_width_bytes": (sample_width),
        "sample_rate": (sample_rate),
        "frames": (frames),
        "duration_s": (frames / sample_rate),
        "file_size_bytes": (path.stat().st_size),
    }


def clear_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        torch.cuda.synchronize()


# =============================================================================
# MMS-TTS smoke test
# =============================================================================


def run_mms_smoke_test(
    language: str,
) -> dict:
    model_id = MMS_MODELS[language]

    text = TEST_SENTENCES[language]

    output_path = AUDIO_DIR / f"mms_{language}.wav"

    print()
    print("-" * 80)

    print(f"MMS-TTS | {language.upper()}")

    print("-" * 80)

    print(f"Model : {model_id}")

    print(f"Text  : {text}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForTextToWaveform.from_pretrained(model_id)

    model.to(DEVICE)

    model.eval()

    actual_device = str(next(model.parameters()).device)

    actual_dtype = str(next(model.parameters()).dtype)

    if actual_device != DEVICE:
        raise RuntimeError(f"{model_id}: model is on {actual_device}, expected {DEVICE}.")

    if actual_dtype != "torch.float32":
        raise RuntimeError(f"{model_id}: model dtype is {actual_dtype}, expected torch.float32.")

    set_seed(42)

    inputs = tokenizer(
        text=text,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output = model(**inputs)

    torch.cuda.synchronize()

    waveform = output.waveform.detach().float().cpu().numpy().reshape(-1)

    sample_rate = int(model.config.sampling_rate)

    validation = validate_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        label=(f"MMS {language}"),
    )

    write_float_wav(
        output_path=(output_path),
        waveform=(waveform),
        sample_rate=(sample_rate),
    )

    file_validation = validate_written_wav(output_path)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    parameter_memory_mb = (
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1024**2
    )

    revision = getattr(
        model.config,
        "_commit_hash",
        None,
    )

    print(f"Device      : {actual_device}")

    print(f"Dtype       : {actual_dtype}")

    print(f"Parameters  : {parameter_count:,}")

    print(f"Param memory: {parameter_memory_mb:.2f} MB")

    print(f"Revision    : {revision}")

    print(f"Sample rate : {sample_rate} Hz")

    print(f"Duration    : {validation['duration_s']:.2f} s")

    print(f"Peak        : {validation['peak_abs']:.4f}")

    print(f"RMS         : {validation['rms']:.4f}")

    print(f"Output      : {output_path}")

    print("Result      : PASS")

    result = {
        "family": ("MMS-TTS"),
        "language": (language),
        "model_id": (model_id),
        "revision": (revision),
        "device": (actual_device),
        "dtype": (actual_dtype),
        "parameters": (parameter_count),
        "parameter_memory_mb": (parameter_memory_mb),
        "text": (text),
        "output_path": (str(output_path)),
        "waveform": (validation),
        "wav_file": (file_validation),
        "status": ("PASS"),
    }

    del output
    del inputs
    del model
    del tokenizer

    clear_cuda()

    return result


# =============================================================================
# Piper smoke test
# =============================================================================


def get_piper_paths(
    language: str,
) -> tuple[
    Path,
    Path,
]:
    voice_name = PIPER_VOICES[language]

    model_path = PIPER_VOICE_DIR / f"{voice_name}.onnx"

    config_path = PIPER_VOICE_DIR / f"{voice_name}.onnx.json"

    return (
        model_path,
        config_path,
    )


def validate_piper_models_exist() -> None:
    missing = []

    for language in LANGUAGES:
        (
            model_path,
            config_path,
        ) = get_piper_paths(language)

        if not model_path.exists():
            missing.append(str(model_path))

        if not config_path.exists():
            missing.append(str(config_path))

    if missing:
        print()
        print("Missing Piper files:")

        for path in missing:
            print(f"- {path}")

        raise RuntimeError("Download all Piper voices before running the smoke test.")


def run_piper_smoke_test(
    language: str,
) -> dict:
    voice_name = PIPER_VOICES[language]

    text = TEST_SENTENCES[language]

    (
        model_path,
        config_path,
    ) = get_piper_paths(language)

    output_path = AUDIO_DIR / f"piper_{language}.wav"

    print()
    print("-" * 80)

    print(f"PIPER | {language.upper()}")

    print("-" * 80)

    print(f"Voice : {voice_name}")

    print(f"Text  : {text}")

    voice = PiperVoice.load(
        model_path,
        config_path=config_path,
        use_cuda=True,
    )

    providers = voice.session.get_providers()

    print(f"Providers   : {providers}")

    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"{voice_name}: Piper did not create a CUDA session. Providers: {providers}")

    with wave.open(
        str(output_path),
        "wb",
    ) as wav_file:
        voice.synthesize_wav(
            text,
            wav_file,
        )

    file_validation = validate_written_wav(output_path)

    model_size_mb = model_path.stat().st_size / 1024**2

    config_size_kb = config_path.stat().st_size / 1024

    print(f"Model size  : {model_size_mb:.2f} MB")

    print(f"Config size : {config_size_kb:.2f} KB")

    print(f"Sample rate : {file_validation['sample_rate']} Hz")

    print(f"Duration    : {file_validation['duration_s']:.2f} s")

    print(f"Output      : {output_path}")

    print("Result      : PASS")

    result = {
        "family": ("Piper"),
        "language": (language),
        "voice": (voice_name),
        "provider": (providers[0]),
        "providers": (providers),
        "text": (text),
        "model_path": (str(model_path)),
        "config_path": (str(config_path)),
        "model_size_mb": (model_size_mb),
        "config_size_kb": (config_size_kb),
        "output_path": (str(output_path)),
        "wav_file": (file_validation),
        "status": ("PASS"),
    }

    del voice

    gc.collect()

    return result


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    validate_environment()

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_piper_models_exist()

    results = []

    # ------------------------------------------------------------------
    # MMS-TTS
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MMS-TTS SMOKE TEST")
    print("=" * 80)

    for language in LANGUAGES:
        result = run_mms_smoke_test(language)

        results.append(result)

    # ------------------------------------------------------------------
    # Piper
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PIPER SMOKE TEST")
    print("=" * 80)

    for language in LANGUAGES:
        result = run_piper_smoke_test(language)

        results.append(result)

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------

    mms_results = [row for row in results if row["family"] == "MMS-TTS"]

    piper_results = [row for row in results if row["family"] == "Piper"]

    mms_pass = sum(1 for row in mms_results if row["status"] == "PASS")

    piper_pass = sum(1 for row in piper_results if row["status"] == "PASS")

    summary = {
        "experiment": ("Initial functional smoke test for TTS candidate selection."),
        "scope": {
            "languages": (LANGUAGES),
            "families": [
                "MMS-TTS",
                "Piper",
            ],
            "tests": (len(results)),
        },
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "transformers": (transformers.__version__),
            "onnxruntime": (ort.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
            "ort_providers": (ort.get_available_providers()),
        },
        "methodology": {
            "purpose": ("Functional validation only. Latency and RTF are deliberately not benchmarked in tts_01."),
            "mms_runtime": ("PyTorch CUDA, FP32."),
            "piper_runtime": ("ONNX Runtime with mandatory CUDAExecutionProvider."),
            "validation": [
                "model/voice loads",
                "correct GPU runtime is used",
                "waveform/file is non-empty",
                "sample rate is valid",
                "audio duration is positive",
                "MMS waveform values are finite",
                "output WAV is readable",
            ],
        },
        "results": (results),
        "summary": {
            "mms_pass": (mms_pass),
            "mms_total": (len(mms_results)),
            "piper_pass": (piper_pass),
            "piper_total": (len(piper_results)),
            "overall_pass": ((mms_pass == len(LANGUAGES)) and (piper_pass == len(LANGUAGES))),
        },
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
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
    print("TTS SMOKE TEST SUMMARY")
    print("=" * 80)

    print(f"MMS-TTS : {mms_pass}/{len(LANGUAGES)} PASS")

    print(f"Piper   : {piper_pass}/{len(LANGUAGES)} PASS")

    print()

    print(f"Audio   : {AUDIO_DIR}")

    print(f"Summary : {SUMMARY_PATH}")

    if not summary["summary"]["overall_pass"]:
        raise RuntimeError("TTS smoke test did not pass for all models/languages.")

    print()
    print("=" * 80)
    print("TTS MODEL SMOKE TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
