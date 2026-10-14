from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import wave
from pathlib import Path
from typing import cast

import nemo
import nemo.collections.asr as nemo_asr

# Torch is intentionally imported before ONNX Runtime so that the
# CUDA/cuDNN libraries already available through PyTorch can be reused.
import torch
import onnxruntime as ort
import transformers
from huggingface_hub import hf_hub_download
from piper import PiperVoice
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ort.preload_dlls()


# =============================================================================
# Console
# =============================================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore


# =============================================================================
# Project
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "cuda:0"


# =============================================================================
# Supported languages
# =============================================================================

SUPPORTED_LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "pt",
]

NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
}


# =============================================================================
# ASR
# =============================================================================

ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

ASR_MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"

ASR_NEMO_FILENAME = "parakeet-tdt-0.6b-v3.nemo"


# =============================================================================
# MT
# =============================================================================

MT_MODEL_ID = "facebook/nllb-200-distilled-600M"

MT_MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

MT_DTYPE = torch.float16

# Operational generation cap used by the integration pipeline.
# MT selection experiments used max_length=512. In the validated
# final-only set, generated sequences remained well below 256 tokens,
# so this cap would not constrain the evaluated translations.
MT_MAX_NEW_TOKENS = 256


# =============================================================================
# TTS
# =============================================================================

PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "es": "es_ES-sharvard-medium",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "pt": "pt_PT-tugão-medium",
}

PIPER_VOICE_DIR = PROJECT_ROOT / "data" / "tts_selection" / "piper_voices"


# =============================================================================
# CLI
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Offline file-to-file Speech-to-Speech integration baseline."))

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Input audio file.",
    )

    parser.add_argument(
        "--source",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="Configured source language.",
    )

    parser.add_argument(
        "--target",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="Configured target language.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output synthesized WAV file.",
    )

    return parser.parse_args()


# =============================================================================
# Validation
# =============================================================================


def validate_environment() -> None:
    print("=" * 80)
    print("S2S INTEGRATION BASELINE")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")

    print(f"PyTorch      : {torch.__version__}")

    print(f"Transformers : {transformers.__version__}")

    print(f"NeMo         : {nemo.__version__}")

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
    print("Environment validation: PASS")


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    audio_path = args.audio.resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio not found:\n{audio_path}")

    if not audio_path.is_file():
        raise RuntimeError(f"Input audio is not a file:\n{audio_path}")

    if args.source == args.target:
        raise ValueError("Source and target languages must be different.")

    print()
    print("=" * 80)
    print("PIPELINE CONFIGURATION")
    print("=" * 80)

    print(f"Input audio : {audio_path}")

    print(f"Source      : {args.source.upper()}")

    print(f"Target      : {args.target.upper()}")

    print(f"Output      : {args.output.resolve()}")


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


# =============================================================================
# GPU inspection
# =============================================================================


def print_gpu_memory(
    label: str,
) -> dict:
    torch.cuda.synchronize()

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    free_mb = free_bytes / 1024**2

    total_mb = total_bytes / 1024**2

    used_mb = total_mb - free_mb

    print(f"{label:<24}: {used_mb:.2f} MB used / {free_mb:.2f} MB free")

    return {
        "used_mb": (used_mb),
        "free_mb": (free_mb),
        "total_mb": (total_mb),
    }


# =============================================================================
# ASR loading
# =============================================================================


def load_asr():
    print()
    print("=" * 80)
    print("LOADING ASR")
    print("=" * 80)

    print(f"Model ID : {ASR_MODEL_ID}")

    print(f"Revision : {ASR_MODEL_REVISION}")

    try:
        nemo_path = hf_hub_download(
            repo_id=(ASR_MODEL_ID),
            filename=(ASR_NEMO_FILENAME),
            revision=(ASR_MODEL_REVISION),
            local_files_only=True,
        )

    except Exception as exc:
        raise RuntimeError(
            "The exact Parakeet checkpoint "
            "is not available in the local "
            "Hugging Face cache.\n"
            "Runtime is intentionally configured "
            "for offline model access."
        ) from exc

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

    load_ms = (time.perf_counter() - start) * 1000.0

    actual_device = str(next(model.parameters()).device)

    if actual_device != DEVICE:
        raise RuntimeError(f"ASR model is on {actual_device}, expected {DEVICE}.")

    print(f"Checkpoint: {nemo_path}")

    print(f"Device    : {actual_device}")

    print(f"Load time : {load_ms:.2f} ms")

    print("ASR load: PASS")

    return (
        model,
        load_ms,
    )


# =============================================================================
# MT loading
# =============================================================================


def load_mt(
    source_language: str,
):
    print()
    print("=" * 80)
    print("LOADING MT")
    print("=" * 80)

    source_code = NLLB_LANGUAGE_CODES[source_language]

    print(f"Model ID : {MT_MODEL_ID}")

    print(f"Revision : {MT_MODEL_REVISION}")

    print(f"Precision: {MT_DTYPE}")

    print(f"Source   : {source_code}")

    start = time.perf_counter()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MT_MODEL_ID,
            revision=(MT_MODEL_REVISION),
            src_lang=(source_code),
            local_files_only=True,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            MT_MODEL_ID,
            revision=(MT_MODEL_REVISION),
            dtype=(MT_DTYPE),
            local_files_only=True,
        )

    except Exception as exc:
        raise RuntimeError(
            "The exact NLLB model/tokenizer "
            "is not available locally.\n"
            "Runtime is intentionally configured "
            "for offline model access."
        ) from exc

    model.to(DEVICE)

    model.eval()

    torch.cuda.synchronize()

    load_ms = (time.perf_counter() - start) * 1000.0

    actual_device = str(next(model.parameters()).device)

    actual_dtype = next(model.parameters()).dtype

    if actual_device != DEVICE:
        raise RuntimeError(f"MT model is on {actual_device}, expected {DEVICE}.")

    if actual_dtype != MT_DTYPE:
        raise RuntimeError(f"MT model dtype is {actual_dtype}, expected {MT_DTYPE}.")

    print(f"Device    : {actual_device}")

    print(f"Dtype     : {actual_dtype}")

    print(f"Load time : {load_ms:.2f} ms")

    print("MT load: PASS")

    return (
        tokenizer,
        model,
        load_ms,
    )


# =============================================================================
# TTS loading
# =============================================================================


def load_tts(
    target_language: str,
):
    print()
    print("=" * 80)
    print("LOADING TTS")
    print("=" * 80)

    voice_name = PIPER_VOICES[target_language]

    model_path = PIPER_VOICE_DIR / f"{voice_name}.onnx"

    config_path = PIPER_VOICE_DIR / f"{voice_name}.onnx.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Piper model not found:\n{model_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Piper config not found:\n{config_path}")

    model_sha256 = sha256_file(model_path)

    config_sha256 = sha256_file(config_path)

    print(f"Voice     : {voice_name}")

    start = time.perf_counter()

    voice = PiperVoice.load(
        model_path,
        config_path=(config_path),
        use_cuda=True,
    )

    load_ms = (time.perf_counter() - start) * 1000.0

    providers = voice.session.get_providers()

    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"Piper did not select CUDA as its preferred Execution Provider.\nProviders: {providers}")

    print(f"Providers : {providers}")

    print(f"Load time : {load_ms:.2f} ms")

    print("TTS load: PASS")

    return (
        voice,
        voice_name,
        model_sha256,
        config_sha256,
        load_ms,
    )


# =============================================================================
# ASR inference
# =============================================================================


def extract_asr_text(
    output,
) -> str:
    if isinstance(
        output,
        str,
    ):
        return output.strip()

    if hasattr(
        output,
        "text",
    ):
        return str(output.text).strip()

    raise RuntimeError(f"Unexpected ASR output type: {type(output)}")


def run_asr(
    model,
    audio_path: Path,
) -> tuple[
    str,
    float,
]:
    print()
    print("=" * 80)
    print("ASR")
    print("=" * 80)

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.transcribe(
            audio=[str(audio_path.resolve())],
            batch_size=1,
            return_hypotheses=False,
            num_workers=0,
            verbose=False,
        )

    torch.cuda.synchronize()

    end = time.perf_counter()

    if len(outputs) != 1:
        raise RuntimeError("ASR returned an unexpected number of outputs.")

    text = extract_asr_text(outputs[0])

    if not text:
        raise RuntimeError("ASR returned empty text.")

    latency_ms = (end - start) * 1000.0

    print(f"Text    : {text}")

    print(f"Time    : {latency_ms:.2f} ms")

    print("ASR inference: PASS")

    return (
        text,
        latency_ms,
    )


# =============================================================================
# MT inference
# =============================================================================


def run_mt(
    tokenizer,
    model,
    source_text: str,
    target_language: str,
) -> tuple[
    str,
    float,
]:
    print()
    print("=" * 80)
    print("MT")
    print("=" * 80)

    target_code = NLLB_LANGUAGE_CODES[target_language]

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)

    if forced_bos_token_id is None or forced_bos_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Could not resolve NLLB target language token: {target_code}")

    torch.cuda.synchronize()

    start = time.perf_counter()

    inputs = tokenizer(
        source_text,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=(forced_bos_token_id),
            num_beams=1,
            do_sample=False,
            max_new_tokens=(MT_MAX_NEW_TOKENS),
        )

    torch.cuda.synchronize()

    translated_text = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    end = time.perf_counter()

    if not translated_text:
        raise RuntimeError("MT returned empty text.")

    latency_ms = (end - start) * 1000.0

    print(f"Target  : {target_code}")

    print(f"Text    : {translated_text}")

    print(f"Time    : {latency_ms:.2f} ms")

    print("MT inference: PASS")

    return (
        translated_text,
        latency_ms,
    )


# =============================================================================
# TTS inference
# =============================================================================


def validate_output_wav(
    output_path: Path,
) -> dict:
    if not output_path.exists():
        raise RuntimeError("TTS output WAV was not created.")

    with wave.open(
        str(output_path),
        "rb",
    ) as wav_file:
        channels = wav_file.getnchannels()

        sample_width = wav_file.getsampwidth()

        sample_rate = wav_file.getframerate()

        frame_count = wav_file.getnframes()

    if channels != 1:
        raise RuntimeError(f"Expected mono TTS output, got {channels} channels.")

    if sample_width != 2:
        raise RuntimeError(f"Expected PCM16 TTS output, got {sample_width}-byte samples.")

    if sample_rate <= 0:
        raise RuntimeError("Invalid TTS sample rate.")

    if frame_count <= 0:
        raise RuntimeError("TTS generated an empty WAV.")

    duration_s = frame_count / sample_rate

    return {
        "channels": (channels),
        "sample_width_bytes": (sample_width),
        "sample_rate": (sample_rate),
        "frames": (frame_count),
        "duration_s": (duration_s),
    }


def run_tts(
    voice: PiperVoice,
    text: str,
    output_path: Path,
) -> tuple[
    dict,
    float,
]:
    print()
    print("=" * 80)
    print("TTS")
    print("=" * 80)

    output_path = output_path.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.perf_counter()

    with wave.open(
        str(output_path),
        "wb",
    ) as wav_file:
        voice.synthesize_wav(
            text,
            wav_file,
        )

    end = time.perf_counter()

    audio_info = validate_output_wav(output_path)

    latency_ms = (end - start) * 1000.0

    print(f"Output  : {output_path}")

    print(f"Rate    : {audio_info['sample_rate']} Hz")

    print(f"Duration: {audio_info['duration_s']:.2f} s")

    print(f"Time    : {latency_ms:.2f} ms")

    print("TTS inference: PASS")

    return (
        audio_info,
        latency_ms,
    )


# =============================================================================
# Report
# =============================================================================


def write_report(
    args: argparse.Namespace,
    voice_name: str,
    tts_model_sha256: str,
    tts_config_sha256: str,
    asr_text: str,
    translated_text: str,
    audio_info: dict,
    load_times: dict,
    stage_times: dict,
    gpu_memory: dict,
) -> Path:
    output_path = args.output.resolve()

    report_path = output_path.with_suffix(".json")

    report = {
        "experiment": ("S2S file-to-file integration baseline"),
        "important_note": ("These timings are functional integration timings, not the final performance benchmark."),
        "configuration": {
            "input_audio": (str(args.audio.resolve())),
            "source_language": (args.source),
            "target_language": (args.target),
            "output_audio": (str(output_path)),
        },
        "models": {
            "asr": {
                "model_id": (ASR_MODEL_ID),
                "revision": (ASR_MODEL_REVISION),
            },
            "mt": {
                "model_id": (MT_MODEL_ID),
                "revision": (MT_MODEL_REVISION),
                "precision": (str(MT_DTYPE)),
            },
            "tts": {
                "family": ("Piper"),
                "voice": (voice_name),
                "model_sha256": (tts_model_sha256),
                "config_sha256": (tts_config_sha256),
                "runtime": ("ONNX Runtime CUDA"),
            },
        },
        "outputs": {
            "asr_text": (asr_text),
            "translated_text": (translated_text),
            "tts_audio": (audio_info),
        },
        "load_times_ms": (load_times),
        "stage_times_ms": (stage_times),
        "processing_total_ms": (sum(stage_times.values())),
        "gpu_memory_snapshots": (gpu_memory),
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "transformers": (transformers.__version__),
            "nemo": (nemo.__version__),
            "onnxruntime": (ort.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return report_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_arguments()

    validate_environment()

    validate_arguments(args)

    print()
    print("=" * 80)
    print("MODEL COEXISTENCE TEST")
    print("=" * 80)

    gpu_memory = {}

    gpu_memory["before_models"] = print_gpu_memory("Before models")

    # ------------------------------------------------------------------
    # Load ASR
    # ------------------------------------------------------------------

    (
        asr_model,
        asr_load_ms,
    ) = load_asr()

    gpu_memory["after_asr"] = print_gpu_memory("After ASR")

    # ------------------------------------------------------------------
    # Load MT
    # ------------------------------------------------------------------

    (
        mt_tokenizer,
        mt_model,
        mt_load_ms,
    ) = load_mt(source_language=(args.source))

    gpu_memory["after_mt"] = print_gpu_memory("After MT")

    # ------------------------------------------------------------------
    # Load TTS
    # ------------------------------------------------------------------

    (
        tts_voice,
        voice_name,
        tts_model_sha256,
        tts_config_sha256,
        tts_load_ms,
    ) = load_tts(target_language=(args.target))

    gpu_memory["after_tts"] = print_gpu_memory("After TTS")

    print()
    print("All three model families are loaded simultaneously.")

    print("Model coexistence: PASS")

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    (
        asr_text,
        asr_ms,
    ) = run_asr(
        model=(asr_model),
        audio_path=(args.audio),
    )

    # ------------------------------------------------------------------
    # MT
    # ------------------------------------------------------------------

    (
        translated_text,
        mt_ms,
    ) = run_mt(
        tokenizer=(mt_tokenizer),
        model=(mt_model),
        source_text=(asr_text),
        target_language=(args.target),
    )

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    (
        output_audio_info,
        tts_ms,
    ) = run_tts(
        voice=(tts_voice),
        text=(translated_text),
        output_path=(args.output),
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    load_times = {
        "asr": (asr_load_ms),
        "mt": (mt_load_ms),
        "tts": (tts_load_ms),
    }

    stage_times = {
        "asr": (asr_ms),
        "mt": (mt_ms),
        "tts": (tts_ms),
    }

    report_path = write_report(
        args=(args),
        voice_name=(voice_name),
        tts_model_sha256=(tts_model_sha256),
        tts_config_sha256=(tts_config_sha256),
        asr_text=(asr_text),
        translated_text=(translated_text),
        audio_info=(output_audio_info),
        load_times=(load_times),
        stage_times=(stage_times),
        gpu_memory=(gpu_memory),
    )

    processing_total_ms = sum(stage_times.values())

    print()
    print("=" * 80)
    print("S2S RESULT")
    print("=" * 80)

    print(f"ASR : {asr_text}")

    print(f"MT  : {translated_text}")

    print(f"WAV : {args.output.resolve()}")

    print()
    print(f"ASR time : {asr_ms:.2f} ms")

    print(f"MT time  : {mt_ms:.2f} ms")

    print(f"TTS time : {tts_ms:.2f} ms")

    print("-" * 40)

    print(f"Processing total : {processing_total_ms:.2f} ms")

    print()
    print("NOTE: these are functional integration timings.")

    print("They are NOT yet the official end-to-end latency benchmark.")

    print()
    print(f"Report: {report_path}")

    print()
    print("=" * 80)
    print("S2S INTEGRATION BASELINE PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
