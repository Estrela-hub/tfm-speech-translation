from __future__ import annotations

import argparse
import copy
import io
import json
import platform
import statistics
import sys
import time
import wave
from pathlib import Path

import ctranslate2
import nemo
import nemo.collections.asr as nemo_asr
import numpy as np
import onnxruntime as ort
import torch
import transformers
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
from omegaconf import open_dict
from piper import PiperVoice
from transformers import AutoTokenizer

# =============================================================================
# Console
# =============================================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# =============================================================================
# Project
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = torch.device("cuda:0")


# =============================================================================
# Languages
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

ASR_MODEL_PATH = PROJECT_ROOT / "models" / "parakeet" / "parakeet-tdt-0.6b-v3.nemo"

ASR_DTYPE = torch.float16


# =============================================================================
# MT
# =============================================================================

MT_MODEL_ID = "facebook/nllb-200-distilled-600M"

MT_MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"

MT_COMPUTE_TYPE = "float16"

MT_MAX_DECODING_LENGTH = 256

MT_CTRANSLATE2_MODEL_PATH = PROJECT_ROOT / "models" / "ctranslate2" / "nllb-200-distilled-600M-fp16"

MT_TOKENIZER_PATH = PROJECT_ROOT / "models" / "nllb_tokenizer"


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

PIPER_VOICE_DIR = PROJECT_ROOT / "models" / "piper"


# =============================================================================
# CLI
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persistent offline Speech-to-Speech pipeline using "
            "full-utterance Parakeet, NLLB with CTranslate2 FP16, "
            "and Piper."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--audio",
        type=Path,
        help="Single input WAV: 16 kHz / mono / PCM16.",
    )

    input_group.add_argument(
        "--batch",
        type=Path,
        help=("JSONL batch manifest. Models are loaded only once."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Output WAV. Required with --audio. Batch outputs are specified in the JSONL."),
    )

    parser.add_argument(
        "--source",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="Source language used by NLLB.",
    )

    parser.add_argument(
        "--target",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="Target language used by NLLB and Piper.",
    )

    parser.add_argument(
        "--warmup-audio",
        type=Path,
        default=None,
        help=("Optional warm-up WAV. Defaults to the first input job."),
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help=("Disable warm-up. Use only for cold-start analysis."),
    )

    return parser.parse_args()


# =============================================================================
# Jobs
# =============================================================================


def resolve_relative_path(
    path_value: str | Path,
    base_dir: Path,
) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path.resolve()

    return (base_dir / path).resolve()


def load_jobs(
    args: argparse.Namespace,
) -> list[dict]:
    if args.audio is not None:
        if args.output is None:
            raise ValueError("--output is required when --audio is used.")

        return [
            {
                "job_id": "single",
                "audio": args.audio.resolve(),
                "output": args.output.resolve(),
            }
        ]

    if args.output is not None:
        raise ValueError("--output is only valid with --audio. Batch outputs belong in the JSONL manifest.")

    batch_path = args.batch.resolve()

    if not batch_path.exists():
        raise FileNotFoundError(f"Batch manifest not found:\n{batch_path}")

    jobs = []

    with batch_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in batch manifest line {line_number}.") from exc

            if "audio" not in row:
                raise ValueError(f"Batch line {line_number} is missing 'audio'.")

            if "output" not in row:
                raise ValueError(f"Batch line {line_number} is missing 'output'.")

            job_id = str(
                row.get(
                    "id",
                    f"job_{line_number:04d}",
                )
            )

            jobs.append(
                {
                    "job_id": job_id,
                    "audio": resolve_relative_path(
                        row["audio"],
                        batch_path.parent,
                    ),
                    "output": resolve_relative_path(
                        row["output"],
                        batch_path.parent,
                    ),
                }
            )

    if not jobs:
        raise RuntimeError("Batch manifest contains no jobs.")

    output_paths = [str(job["output"]) for job in jobs]

    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Batch manifest contains duplicate output paths.")

    return jobs


# =============================================================================
# Validation
# =============================================================================


def validate_environment() -> None:
    print("=" * 80)
    print("S2S PERSISTENT FULL-UTTERANCE PIPELINE")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"Transformers : {transformers.__version__}")
    print(f"CTranslate2  : {ctranslate2.__version__}")
    print(f"NeMo         : {nemo.__version__}")
    print(f"ONNX Runtime : {ort.__version__}")

    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    ct2_cuda_devices = ctranslate2.get_cuda_device_count()

    print(f"CT2 CUDA GPUs: {ct2_cuda_devices}")

    if ct2_cuda_devices < 1:
        raise RuntimeError("CTranslate2 does not detect a CUDA device.")

    ct2_compute_types = ctranslate2.get_supported_compute_types("cuda")

    print(f"CT2 compute  : {sorted(ct2_compute_types)}")

    if MT_COMPUTE_TYPE not in ct2_compute_types:
        raise RuntimeError(f"CTranslate2 does not support {MT_COMPUTE_TYPE} on this device.")

    providers = ort.get_available_providers()

    print(f"ORT providers: {providers}")

    # Piper is intentionally executed on CPU.
    if "CPUExecutionProvider" not in providers:
        raise RuntimeError("CPUExecutionProvider is not available for Piper.")

    print()
    print("Environment validation: PASS")


def validate_configuration(
    args: argparse.Namespace,
    jobs: list[dict],
) -> None:
    if args.source == args.target:
        raise ValueError("Source and target languages must be different.")

    for job in jobs:
        audio_path = job["audio"]
        output_path = job["output"]

        if not audio_path.exists():
            raise FileNotFoundError(f"Input WAV not found:\n{audio_path}")

        if not audio_path.is_file():
            raise RuntimeError(f"Input path is not a file:\n{audio_path}")

        if audio_path == output_path:
            raise ValueError(f"Input and output WAV cannot use the same path:\n{audio_path}")

    if args.warmup_audio is not None:
        warmup_path = args.warmup_audio.resolve()

        if not warmup_path.exists():
            raise FileNotFoundError(f"Warm-up WAV not found:\n{warmup_path}")

        if not warmup_path.is_file():
            raise RuntimeError(f"Warm-up path is not a file:\n{warmup_path}")

    mode = "single" if args.audio is not None else "batch"

    print()
    print("=" * 80)
    print("PIPELINE CONFIGURATION")
    print("=" * 80)

    print(f"Mode        : {mode}")
    print(f"Jobs        : {len(jobs)}")
    print(f"Source      : {args.source.upper()}")
    print(f"Target      : {args.target.upper()}")

    print("Warm-up     : " + ("disabled" if args.no_warmup else "enabled"))

    print("ASR mode    : full utterance")
    print()
    print("Input contract: WAV / 16 kHz / mono / PCM16")
    print("Parakeet detects the spoken language automatically; --source configures NLLB.")


# =============================================================================
# Memory
# =============================================================================


def system_available_memory_mb() -> float:
    meminfo = Path("/proc/meminfo")

    if not meminfo.exists():
        return float("nan")

    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0

    return float("nan")


def print_memory(
    label: str,
) -> dict:
    torch.cuda.synchronize()

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    free_mb = free_bytes / 1024**2
    total_mb = total_bytes / 1024**2

    allocated_mb = torch.cuda.memory_allocated() / 1024**2

    reserved_mb = torch.cuda.memory_reserved() / 1024**2

    system_available_mb = system_available_memory_mb()

    print()
    print(f"{label}:")
    print(f"  Device free      : {free_mb:.2f} MiB")
    print(f"  Torch allocated  : {allocated_mb:.2f} MiB")
    print(f"  Torch reserved   : {reserved_mb:.2f} MiB")
    print(f"  System available : {system_available_mb:.2f} MiB")

    return {
        "device_free_mb": free_mb,
        "device_total_mb": total_mb,
        "torch_allocated_mb": allocated_mb,
        "torch_reserved_mb": reserved_mb,
        "system_available_mb": system_available_mb,
    }


# =============================================================================
# WAV input
# =============================================================================


def load_input_wav(
    audio_path: Path,
    expected_sample_rate: int,
) -> tuple[np.ndarray, dict, float]:
    start = time.perf_counter()

    with wave.open(
        str(audio_path),
        "rb",
    ) as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        compression = wav_file.getcomptype()

        raw_audio = wav_file.readframes(frame_count)

    load_ms = (time.perf_counter() - start) * 1000.0

    if channels != 1:
        raise ValueError(f"Input must be mono, got {channels} channels:\n{audio_path}")

    if sample_width != 2:
        raise ValueError(f"Input must use PCM16 samples, got {sample_width * 8}-bit:\n{audio_path}")

    if sample_rate != expected_sample_rate:
        raise ValueError(f"Input sample rate must be {expected_sample_rate} Hz, got {sample_rate} Hz:\n{audio_path}")

    if compression != "NONE":
        raise ValueError(f"Input WAV must be uncompressed PCM, got {compression}:\n{audio_path}")

    if frame_count <= 0:
        raise ValueError(f"Input WAV is empty:\n{audio_path}")

    pcm16 = np.frombuffer(
        raw_audio,
        dtype="<i2",
    )

    if pcm16.size != frame_count:
        raise RuntimeError("Unexpected PCM sample count.")

    waveform = pcm16.astype(np.float32) / 32768.0

    if not np.isfinite(waveform).all():
        raise RuntimeError("Input WAV contains NaN/Inf.")

    duration_s = frame_count / sample_rate

    metadata = {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "frames": frame_count,
        "duration_s": duration_s,
    }

    return (
        waveform,
        metadata,
        load_ms,
    )


# =============================================================================
# ASR
# =============================================================================


def load_asr():
    print()
    print("=" * 80)
    print("LOADING ASR")
    print("=" * 80)

    if not ASR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Parakeet checkpoint not found:\n{ASR_MODEL_PATH}")

    print(f"Model ID   : {ASR_MODEL_ID}")
    print(f"Revision   : {ASR_MODEL_REVISION}")
    print(f"Checkpoint : {ASR_MODEL_PATH}")
    print(f"Precision  : {ASR_DTYPE}")

    start = time.perf_counter()

    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(ASR_MODEL_PATH),
        map_location=DEVICE,
    )

    model.freeze()
    model.eval()

    model.to(
        device=DEVICE,
        dtype=ASR_DTYPE,
    )

    # Jetson's PyTorch build has torch.distributed
    # disabled.
    model.encoder.sync_max_audio_length = False

    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    # Stable TDT decoding configuration validated
    # on the final Jetson runtime.
    decoding_config = copy.deepcopy(model.cfg.decoding)

    with open_dict(decoding_config):
        decoding_config.strategy = "greedy_batch"

        decoding_config.tdt_include_token_duration = False

        decoding_config.greedy.loop_labels = True

        decoding_config.greedy.preserve_alignments = False

        decoding_config.greedy.use_cuda_graph_decoder = False

        decoding_config.fused_batch_size = -1

    model.change_decoding_strategy(decoding_config)

    try:
        decoding_computer = model.decoding.decoding.decoding_computer

    except AttributeError as exc:
        raise RuntimeError("Could not access RNNT/TDT decoding computer.") from exc

    sample_rate = int(model.cfg.preprocessor.sample_rate)

    torch.cuda.synchronize()

    load_ms = (time.perf_counter() - start) * 1000.0

    actual_device = next(model.parameters()).device

    actual_dtype = next(model.parameters()).dtype

    if actual_device.type != "cuda":
        raise RuntimeError(f"ASR is on {actual_device}, expected CUDA.")

    if actual_dtype != ASR_DTYPE:
        raise RuntimeError(f"ASR dtype is {actual_dtype}, expected {ASR_DTYPE}.")

    # Release temporary allocator cache left by
    # restore_from().
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"Device      : {actual_device}")
    print(f"Dtype       : {actual_dtype}")
    print(f"Sample rate : {sample_rate} Hz")
    print(f"Sync max    : {model.encoder.sync_max_audio_length}")
    print("CUDA graph  : disabled")
    print("ASR mode    : full utterance")
    print()
    print(f"Load time   : {load_ms:.2f} ms")
    print("ASR load: PASS")

    asr_config = {
        "sample_rate": sample_rate,
        "mode": "full_utterance",
        "compute_dtype": str(actual_dtype),
        "cuda_graph_decoder": False,
    }

    return (
        model,
        decoding_computer,
        asr_config,
        load_ms,
    )


def run_asr_waveform(
    model,
    decoding_computer,
    asr_config: dict,
    waveform: np.ndarray,
) -> tuple[str, dict]:
    sample_rate = int(asr_config["sample_rate"])

    audio_duration_s = waveform.size / sample_rate

    audio_batch = torch.from_numpy(waveform).float().unsqueeze(0).to(DEVICE)

    audio_lengths = torch.tensor(
        [waveform.size],
        dtype=torch.long,
        device=DEVICE,
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        (
            encoder_output,
            encoder_output_length,
        ) = model(
            input_signal=audio_batch,
            input_signal_length=audio_lengths,
        )

        # NeMo model.forward() returns [B, D, T].
        # The TDT decoder expects [B, T, D].
        encoder_output = encoder_output.transpose(
            1,
            2,
        )

        batched_hyps, _ = decoding_computer(
            x=encoder_output,
            out_len=encoder_output_length,
            prev_batched_state=None,
            multi_biasing_ids=None,
        )

    torch.cuda.synchronize()

    compute_ms = (time.perf_counter() - start) * 1000.0

    hypotheses = batched_hyps_to_hypotheses(
        batched_hyps,
        batch_size=1,
    )

    if len(hypotheses) != 1:
        raise RuntimeError("ASR returned an unexpected number of hypotheses.")

    final_text = model.tokenizer.ids_to_text(hypotheses[0].y_sequence.tolist()).strip()

    if not final_text:
        raise RuntimeError("ASR returned empty text.")

    rtf = (compute_ms / 1000.0) / audio_duration_s

    metrics = {
        "audio_duration_s": audio_duration_s,
        "compute_ms": compute_ms,
        "rtf": rtf,
    }

    return (
        final_text,
        metrics,
    )


# =============================================================================
# MT
# =============================================================================


def load_mt(
    source_language: str,
):
    print()
    print("=" * 80)
    print("LOADING MT")
    print("=" * 80)

    source_code = NLLB_LANGUAGE_CODES[source_language]

    if not MT_CTRANSLATE2_MODEL_PATH.exists():
        raise FileNotFoundError(f"CTranslate2 NLLB model not found:\n{MT_CTRANSLATE2_MODEL_PATH}")

    if not MT_TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"NLLB tokenizer not found:\n{MT_TOKENIZER_PATH}")

    if not ctranslate2.contains_model(str(MT_CTRANSLATE2_MODEL_PATH)):
        raise RuntimeError("The configured MT directory is not a valid CTranslate2 model.")

    print(f"Model ID     : {MT_MODEL_ID}")
    print(f"Revision     : {MT_MODEL_REVISION}")
    print("Runtime      : CTranslate2")
    print(f"Compute type : {MT_COMPUTE_TYPE}")
    print(f"Source       : {source_code}")
    print(f"Model path   : {MT_CTRANSLATE2_MODEL_PATH}")
    print(f"Tokenizer    : {MT_TOKENIZER_PATH}")

    start = time.perf_counter()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MT_TOKENIZER_PATH),
            src_lang=source_code,
            local_files_only=True,
        )

    except Exception as exc:
        raise RuntimeError("Could not load the local NLLB tokenizer.") from exc

    translator = ctranslate2.Translator(
        str(MT_CTRANSLATE2_MODEL_PATH),
        device="cuda",
        device_index=0,
        compute_type=MT_COMPUTE_TYPE,
        inter_threads=1,
    )

    load_ms = (time.perf_counter() - start) * 1000.0

    if translator.device != "cuda":
        raise RuntimeError(f"CTranslate2 selected an unexpected device: {translator.device}")

    if translator.compute_type != MT_COMPUTE_TYPE:
        raise RuntimeError(f"CTranslate2 selected compute type {translator.compute_type}, expected {MT_COMPUTE_TYPE}.")

    print(f"Device       : {translator.device}")
    print(f"Compute type : {translator.compute_type}")
    print(f"Load time    : {load_ms:.2f} ms")

    print("MT load: PASS")

    return (
        tokenizer,
        translator,
        load_ms,
    )


def run_mt(
    tokenizer,
    translator,
    source_text: str,
    target_language: str,
) -> tuple[str, float]:
    target_code = NLLB_LANGUAGE_CODES[target_language]

    start = time.perf_counter()

    # NLLB source tokenization.
    source_ids = tokenizer.encode(source_text)

    source_tokens = tokenizer.convert_ids_to_tokens(source_ids)

    if not source_tokens:
        raise RuntimeError("NLLB source tokenization produced no tokens.")

    # CTranslate2 greedy final-only translation.
    # The target prefix forces the NLLB
    # target-language token.
    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[target_code]],
        beam_size=1,
        max_decoding_length=(MT_MAX_DECODING_LENGTH),
    )

    if not results:
        raise RuntimeError("CTranslate2 returned no translation result.")

    if not results[0].hypotheses:
        raise RuntimeError("CTranslate2 returned no translation hypothesis.")

    result_tokens = results[0].hypotheses[0]

    # CTranslate2 returns the forced language
    # token as part of the hypothesis.
    if result_tokens and result_tokens[0] == target_code:
        target_tokens = result_tokens[1:]

    else:
        target_tokens = result_tokens

    if not target_tokens:
        raise RuntimeError("CTranslate2 returned an empty target sequence.")

    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    target_text = tokenizer.decode(
        target_ids,
        skip_special_tokens=True,
    ).strip()

    latency_ms = (time.perf_counter() - start) * 1000.0

    if not target_text:
        raise RuntimeError("NLLB CTranslate2 returned empty text.")

    return (
        target_text,
        latency_ms,
    )


# =============================================================================
# TTS
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

    print(f"Voice     : {voice_name}")
    print("Runtime   : ONNX Runtime CPU")

    start = time.perf_counter()

    voice = PiperVoice.load(
        model_path,
        config_path=config_path,
        use_cuda=False,
    )

    load_ms = (time.perf_counter() - start) * 1000.0

    providers = voice.session.get_providers()

    if not providers or providers[0] != "CPUExecutionProvider":
        raise RuntimeError(f"Piper did not select CPUExecutionProvider.\nProviders: {providers}")

    print(f"Providers : {providers}")
    print(f"Load time : {load_ms:.2f} ms")
    print("TTS load: PASS")

    return (
        voice,
        voice_name,
        load_ms,
    )


def inspect_wav_bytes(
    wav_bytes: bytes,
) -> dict:
    with wave.open(
        io.BytesIO(wav_bytes),
        "rb",
    ) as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()

    if channels <= 0 or sample_width <= 0 or sample_rate <= 0 or frames <= 0:
        raise RuntimeError("Piper generated invalid audio.")

    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_s": (frames / sample_rate),
    }


def synthesize_tts_to_memory(
    voice: PiperVoice,
    text: str,
) -> tuple[bytes, dict, float]:
    memory_file = io.BytesIO()

    start = time.perf_counter()

    with wave.open(
        memory_file,
        "wb",
    ) as wav_file:
        voice.synthesize_wav(
            text,
            wav_file,
        )

    synthesis_ms = (time.perf_counter() - start) * 1000.0

    wav_bytes = memory_file.getvalue()

    audio_info = inspect_wav_bytes(wav_bytes)

    return (
        wav_bytes,
        audio_info,
        synthesis_ms,
    )


def write_output_wav(
    output_path: Path,
    wav_bytes: bytes,
) -> float:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.perf_counter()

    output_path.write_bytes(wav_bytes)

    return (time.perf_counter() - start) * 1000.0


# =============================================================================
# Warm-up
# =============================================================================


def warm_up_runtime(
    warmup_audio_path: Path,
    asr_model,
    decoding_computer,
    asr_config: dict,
    mt_tokenizer,
    mt_translator,
    tts_voice: PiperVoice,
    target_language: str,
) -> dict:
    print()
    print("=" * 80)
    print("RUNTIME WARM-UP")
    print("=" * 80)

    print(f"Warm-up audio: {warmup_audio_path}")

    total_start = time.perf_counter()

    (
        waveform,
        audio_info,
        input_load_ms,
    ) = load_input_wav(
        warmup_audio_path,
        expected_sample_rate=(asr_config["sample_rate"]),
    )

    (
        asr_text,
        asr_metrics,
    ) = run_asr_waveform(
        model=asr_model,
        decoding_computer=(decoding_computer),
        asr_config=asr_config,
        waveform=waveform,
    )

    (
        target_text,
        mt_ms,
    ) = run_mt(
        tokenizer=mt_tokenizer,
        translator=mt_translator,
        source_text=asr_text,
        target_language=(target_language),
    )

    (
        _,
        tts_info,
        tts_ms,
    ) = synthesize_tts_to_memory(
        voice=tts_voice,
        text=target_text,
    )

    total_ms = (time.perf_counter() - total_start) * 1000.0

    print(f"Input load       : {input_load_ms:.2f} ms")
    print(f"ASR              : {asr_metrics['compute_ms']:.2f} ms")
    print(f"ASR RTF          : {asr_metrics['rtf']:.4f}")
    print(f"MT               : {mt_ms:.2f} ms")
    print(f"TTS              : {tts_ms:.2f} ms")
    print(f"Warm-up total    : {total_ms:.2f} ms")

    print()
    print(f"Warm-up ASR text : {asr_text}")
    print(f"Warm-up MT text  : {target_text}")
    print(f"Warm-up TTS audio: {tts_info['duration_s']:.2f} s (discarded)")

    print()
    print("Runtime warm-up: PASS")

    return {
        "input_audio": str(warmup_audio_path),
        "input_duration_s": (audio_info["duration_s"]),
        "input_load_ms": (input_load_ms),
        "asr_ms": (asr_metrics["compute_ms"]),
        "asr_rtf": (asr_metrics["rtf"]),
        "mt_ms": mt_ms,
        "tts_ms": tts_ms,
        "total_ms": total_ms,
        "asr_text": asr_text,
        "translated_text": (target_text),
    }


# =============================================================================
# Job execution
# =============================================================================


def process_job(
    job_index: int,
    job_count: int,
    job: dict,
    target_language: str,
    asr_model,
    decoding_computer,
    asr_config: dict,
    mt_tokenizer,
    mt_translator,
    tts_voice: PiperVoice,
    voice_name: str,
) -> dict:
    print()
    print("=" * 80)
    print(f"JOB {job_index}/{job_count}")
    print("=" * 80)

    print(f"ID     : {job['job_id']}")
    print(f"Input  : {job['audio']}")
    print(f"Output : {job['output']}")

    job_start = time.perf_counter()

    (
        waveform,
        input_info,
        input_load_ms,
    ) = load_input_wav(
        job["audio"],
        expected_sample_rate=(asr_config["sample_rate"]),
    )

    (
        asr_text,
        asr_metrics,
    ) = run_asr_waveform(
        model=asr_model,
        decoding_computer=(decoding_computer),
        asr_config=asr_config,
        waveform=waveform,
    )

    del waveform

    (
        translated_text,
        mt_ms,
    ) = run_mt(
        tokenizer=mt_tokenizer,
        translator=mt_translator,
        source_text=asr_text,
        target_language=(target_language),
    )

    (
        wav_bytes,
        tts_info,
        tts_ms,
    ) = synthesize_tts_to_memory(
        voice=tts_voice,
        text=translated_text,
    )

    output_write_ms = write_output_wav(
        job["output"],
        wav_bytes,
    )

    wall_ms = (time.perf_counter() - job_start) * 1000.0

    compute_total_ms = asr_metrics["compute_ms"] + mt_ms + tts_ms

    print()
    print("ASR")
    print(f"Text               : {asr_text}")
    print("Mode               : full utterance")
    print(f"Compute            : {asr_metrics['compute_ms']:.2f} ms")
    print(f"RTF                : {asr_metrics['rtf']:.4f}")

    print()
    print("MT")
    print(f"Text               : {translated_text}")
    print(f"Time               : {mt_ms:.2f} ms")

    print()
    print("TTS")
    print(f"Voice              : {voice_name}")
    print("Runtime            : ONNX Runtime CPU")
    print(f"Synthesis          : {tts_ms:.2f} ms")
    print(f"Audio duration     : {tts_info['duration_s']:.2f} s")
    print(f"Output write       : {output_write_ms:.2f} ms")

    print()
    print("STEADY-STATE")
    print(f"Input WAV load     : {input_load_ms:.2f} ms")
    print(f"Compute/runtime    : {compute_total_ms:.2f} ms")
    print(f"File-to-file wall  : {wall_ms:.2f} ms")

    print()
    print(f"Job {job_index}: PASS")

    return {
        "job_id": (job["job_id"]),
        "input_audio": str(job["audio"]),
        "output_audio": str(job["output"]),
        "input": input_info,
        "input_load_ms": (input_load_ms),
        "asr_text": asr_text,
        "asr_ms": (asr_metrics["compute_ms"]),
        "asr_rtf": (asr_metrics["rtf"]),
        "translated_text": (translated_text),
        "mt_ms": mt_ms,
        "tts_ms": tts_ms,
        "tts_info": tts_info,
        "output_write_ms": (output_write_ms),
        "compute_total_ms": (compute_total_ms),
        "file_to_file_wall_ms": (wall_ms),
    }


# =============================================================================
# Reports
# =============================================================================


def write_job_report(
    result: dict,
    source_language: str,
    target_language: str,
    voice_name: str,
    asr_config: dict,
    load_times: dict,
    warmup_info: dict | None,
    memory_after_job: dict,
) -> Path:
    output_path = Path(result["output_audio"])

    report_path = output_path.with_suffix(".json")

    report = {
        "experiment": ("Jetson persistent full-utterance Speech-to-Speech pipeline"),
        "job_id": result["job_id"],
        "architecture": (
            "16 kHz PCM16 WAV -> "
            "full-utterance Parakeet -> "
            "final ASR hypothesis -> "
            "NLLB CTranslate2 FP16 final-only -> "
            "Piper ONNX Runtime CPU -> "
            "output WAV"
        ),
        "configuration": {
            "input_audio": (result["input_audio"]),
            "output_audio": (result["output_audio"]),
            "source_language": (source_language),
            "target_language": (target_language),
        },
        "startup": {
            "model_load_ms": (load_times),
            "warmup_enabled": (warmup_info is not None),
            "warmup": warmup_info,
        },
        "asr": {
            "model_id": ASR_MODEL_ID,
            "revision": (ASR_MODEL_REVISION),
            "checkpoint": str(ASR_MODEL_PATH),
            "precision": str(ASR_DTYPE),
            "method": ("NeMo full-utterance RNNT/TDT inference"),
            "mode": (asr_config["mode"]),
            "cuda_graph_decoder": (False),
            "final_text": (result["asr_text"]),
            "compute_ms": (result["asr_ms"]),
            "rtf": (result["asr_rtf"]),
        },
        "mt": {
            "model_id": MT_MODEL_ID,
            "revision": (MT_MODEL_REVISION),
            "runtime": ("CTranslate2"),
            "compute_type": (MT_COMPUTE_TYPE),
            "model_path": str(MT_CTRANSLATE2_MODEL_PATH),
            "tokenizer_path": str(MT_TOKENIZER_PATH),
            "policy": ("final_only"),
            "beam_size": 1,
            "max_decoding_length": (MT_MAX_DECODING_LENGTH),
            "calls": 1,
            "translated_text": (result["translated_text"]),
            "latency_ms": (result["mt_ms"]),
        },
        "tts": {
            "family": "Piper",
            "voice": voice_name,
            "runtime": ("ONNX Runtime CPU"),
            "execution_provider": ("CPUExecutionProvider"),
            "synthesis_ms": (result["tts_ms"]),
            "output_write_ms": (result["output_write_ms"]),
            "audio": (result["tts_info"]),
        },
        "input": {
            **result["input"],
            "load_ms": (result["input_load_ms"]),
        },
        "steady_state_metrics": {
            "input_load_ms": (result["input_load_ms"]),
            "asr_compute_ms": (result["asr_ms"]),
            "asr_rtf": (result["asr_rtf"]),
            "mt_ms": (result["mt_ms"]),
            "tts_synthesis_ms": (result["tts_ms"]),
            "output_write_ms": (result["output_write_ms"]),
            "compute_and_runtime_ms": (result["compute_total_ms"]),
            "file_to_file_wall_ms": (result["file_to_file_wall_ms"]),
        },
        "memory_after_job": (memory_after_job),
        "memory_note": ("Jetson uses unified memory. CUDA and system memory values must not be added together."),
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "cuda": (torch.version.cuda),
            "transformers": (transformers.__version__),
            "ctranslate2": (ctranslate2.__version__),
            "nemo": nemo.__version__,
            "onnxruntime": (ort.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
        },
        "target_hardware": ("NVIDIA Jetson Orin Nano 8 GB"),
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
# Persistent summary
# =============================================================================


def build_batch_summary(
    results: list[dict],
    memory_after_warmup: dict | None,
    memory_per_job: list[dict],
) -> dict:
    def metric_summary(
        key: str,
    ) -> dict:
        values = [result[key] for result in results]

        return {
            "mean": (statistics.mean(values)),
            "median": (statistics.median(values)),
            "min": min(values),
            "max": max(values),
        }

    unique_input_audio = {result["input_audio"] for result in results}

    repeated_same_input = len(unique_input_audio) == 1

    unique_asr = sorted({result["asr_text"] for result in results})

    unique_mt = sorted({result["translated_text"] for result in results})

    if repeated_same_input:
        asr_exactly_stable = len(unique_asr) == 1

        mt_exactly_stable = len(unique_mt) == 1

    else:
        asr_exactly_stable = None
        mt_exactly_stable = None

    memory_drift_mb = None

    if memory_after_warmup is not None and memory_per_job:
        memory_drift_mb = memory_per_job[-1]["system_available_mb"] - memory_after_warmup["system_available_mb"]

    return {
        "jobs": len(results),
        "input_load_ms": (metric_summary("input_load_ms")),
        "asr_ms": (metric_summary("asr_ms")),
        "asr_rtf": (metric_summary("asr_rtf")),
        "mt_ms": (metric_summary("mt_ms")),
        "tts_ms": (metric_summary("tts_ms")),
        "output_write_ms": (metric_summary("output_write_ms")),
        "compute_total_ms": (metric_summary("compute_total_ms")),
        "file_to_file_wall_ms": (metric_summary("file_to_file_wall_ms")),
        "output_consistency": {
            "comparable": (repeated_same_input),
            "reason": (
                "same input audio repeated" if repeated_same_input else ("batch contains different input audio paths")
            ),
            "asr_exactly_stable": (asr_exactly_stable),
            "asr_unique_outputs": (unique_asr if repeated_same_input else None),
            "mt_exactly_stable": (mt_exactly_stable),
            "mt_unique_outputs": (unique_mt if repeated_same_input else None),
        },
        "memory": {
            "after_warmup": (memory_after_warmup),
            "after_each_job": (memory_per_job),
            "system_available_change_mb": (memory_drift_mb),
        },
    }


def print_batch_summary(
    summary: dict,
) -> None:
    if summary["jobs"] <= 1:
        return

    print()
    print("=" * 80)
    print("PERSISTENT RUNTIME SUMMARY")
    print("=" * 80)

    print(f"Jobs                   : {summary['jobs']}")

    print(f"Mean input load        : {summary['input_load_ms']['mean']:.2f} ms")

    print()

    print(f"Mean ASR               : {summary['asr_ms']['mean']:.2f} ms")

    print(f"Median ASR             : {summary['asr_ms']['median']:.2f} ms")

    print(f"Mean ASR RTF           : {summary['asr_rtf']['mean']:.4f}")

    print()

    print(f"Mean MT                : {summary['mt_ms']['mean']:.2f} ms")

    print(f"Median MT              : {summary['mt_ms']['median']:.2f} ms")

    print()

    print(f"Mean TTS               : {summary['tts_ms']['mean']:.2f} ms")

    print(f"Median TTS             : {summary['tts_ms']['median']:.2f} ms")

    print()

    print(f"Mean output write      : {summary['output_write_ms']['mean']:.2f} ms")

    print(f"Mean compute/runtime   : {summary['compute_total_ms']['mean']:.2f} ms")

    print(f"Mean file-to-file wall : {summary['file_to_file_wall_ms']['mean']:.2f} ms")

    print(f"Median file-to-file    : {summary['file_to_file_wall_ms']['median']:.2f} ms")

    print(f"Min file-to-file       : {summary['file_to_file_wall_ms']['min']:.2f} ms")

    print(f"Max file-to-file       : {summary['file_to_file_wall_ms']['max']:.2f} ms")

    consistency = summary["output_consistency"]

    print()

    if consistency["comparable"]:
        print("ASR exact consistency  : " + ("PASS" if consistency["asr_exactly_stable"] else "WARNING"))

        print("MT exact consistency   : " + ("PASS" if consistency["mt_exactly_stable"] else "WARNING"))

    else:
        print("ASR exact consistency  : N/A (different inputs)")

        print("MT exact consistency   : N/A (different inputs)")

    memory_change = summary["memory"]["system_available_change_mb"]

    if memory_change is not None:
        print()

        print(f"System memory change   : {memory_change:+.2f} MiB")

        print("(after last job relative to after warm-up)")


def write_batch_summary(
    jobs: list[dict],
    source_language: str,
    target_language: str,
    load_times: dict,
    warmup_info: dict | None,
    summary: dict,
) -> Path | None:
    if len(jobs) <= 1:
        return None

    summary_path = jobs[0]["output"].parent / "s2s_batch_summary.json"

    report = {
        "experiment": ("Jetson persistent full-utterance S2S repeated-execution test"),
        "configuration": {
            "source_language": (source_language),
            "target_language": (target_language),
            "jobs": len(jobs),
            "asr_mode": ("full_utterance"),
            "mt_runtime": ("CTranslate2"),
            "mt_compute_type": (MT_COMPUTE_TYPE),
            "tts_runtime": ("ONNX Runtime CPU"),
        },
        "startup": {
            "model_load_ms": (load_times),
            "warmup": (warmup_info),
        },
        "summary": summary,
    }

    summary_path.write_text(
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

    return summary_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_arguments()

    validate_environment()

    jobs = load_jobs(args)

    validate_configuration(
        args,
        jobs,
    )

    startup_start = time.perf_counter()

    memory = {}

    memory["before_models"] = print_memory("Before models")

    # -------------------------------------------------------------------------
    # Load ASR once
    # -------------------------------------------------------------------------

    (
        asr_model,
        decoding_computer,
        asr_config,
        asr_load_ms,
    ) = load_asr()

    memory["after_asr"] = print_memory("After ASR")

    # -------------------------------------------------------------------------
    # Load MT once
    # -------------------------------------------------------------------------

    (
        mt_tokenizer,
        mt_translator,
        mt_load_ms,
    ) = load_mt(source_language=(args.source))

    memory["after_mt"] = print_memory("After MT")

    # -------------------------------------------------------------------------
    # Load TTS once
    # -------------------------------------------------------------------------

    (
        tts_voice,
        voice_name,
        tts_load_ms,
    ) = load_tts(target_language=(args.target))

    memory["after_tts"] = print_memory("After TTS")

    print()
    print("All three model families are loaded simultaneously.")
    print("Model coexistence: PASS")

    load_times = {
        "asr": asr_load_ms,
        "mt": mt_load_ms,
        "tts": tts_load_ms,
        "total": (asr_load_ms + mt_load_ms + tts_load_ms),
    }

    # -------------------------------------------------------------------------
    # Warm-up once
    # -------------------------------------------------------------------------

    warmup_info = None

    if not args.no_warmup:
        if args.warmup_audio is not None:
            warmup_audio = args.warmup_audio.resolve()

        else:
            warmup_audio = jobs[0]["audio"]

        warmup_info = warm_up_runtime(
            warmup_audio_path=(warmup_audio),
            asr_model=(asr_model),
            decoding_computer=(decoding_computer),
            asr_config=(asr_config),
            mt_tokenizer=(mt_tokenizer),
            mt_translator=(mt_translator),
            tts_voice=(tts_voice),
            target_language=(args.target),
        )

        memory["after_warmup"] = print_memory("After warm-up")

    ready_ms = (time.perf_counter() - startup_start) * 1000.0

    print()
    print("=" * 80)
    print("SYSTEM READY")
    print("=" * 80)

    print(f"Model load total : {load_times['total']:.2f} ms")

    if warmup_info is not None:
        print(f"Warm-up total    : {warmup_info['total_ms']:.2f} ms")

    print(f"READY after      : {ready_ms:.2f} ms")

    print()
    print("Models remain resident for all subsequent jobs.")

    # -------------------------------------------------------------------------
    # Persistent processing
    # -------------------------------------------------------------------------

    results = []
    memory_per_job = []

    for index, job in enumerate(
        jobs,
        start=1,
    ):
        result = process_job(
            job_index=index,
            job_count=len(jobs),
            job=job,
            target_language=(args.target),
            asr_model=(asr_model),
            decoding_computer=(decoding_computer),
            asr_config=(asr_config),
            mt_tokenizer=(mt_tokenizer),
            mt_translator=(mt_translator),
            tts_voice=(tts_voice),
            voice_name=(voice_name),
        )

        memory_after_job = print_memory(f"After job {index}")

        memory_per_job.append(memory_after_job)

        report_path = write_job_report(
            result=result,
            source_language=(args.source),
            target_language=(args.target),
            voice_name=(voice_name),
            asr_config=(asr_config),
            load_times=(load_times),
            warmup_info=(warmup_info),
            memory_after_job=(memory_after_job),
        )

        print(f"Report            : {report_path}")

        results.append(result)

    # -------------------------------------------------------------------------
    # Persistent summary
    # -------------------------------------------------------------------------

    summary = build_batch_summary(
        results=results,
        memory_after_warmup=(memory.get("after_warmup")),
        memory_per_job=(memory_per_job),
    )

    print_batch_summary(summary)

    summary_path = write_batch_summary(
        jobs=jobs,
        source_language=(args.source),
        target_language=(args.target),
        load_times=(load_times),
        warmup_info=(warmup_info),
        summary=summary,
    )

    if summary_path is not None:
        print()
        print(f"Batch report           : {summary_path}")

    print()
    print("=" * 80)
    print("S2S PERSISTENT FULL-UTTERANCE RUNTIME: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
