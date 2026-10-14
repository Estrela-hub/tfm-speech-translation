from __future__ import annotations

import argparse
import copy
import io
import json
import os
import platform
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any

import torch
import nemo
import nemo.collections.asr as nemo_asr
import numpy as np
import onnxruntime as ort
import transformers
from huggingface_hub import hf_hub_download
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_dtype
from omegaconf import open_dict
from piper import PiperVoice
from torch.nn.modules.rnn import RNNBase
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =============================================================================
# Runtime readiness timing
#
# Starts after module imports. READY includes model loading,
# warm-up and runtime initialization, but excludes import time.
# =============================================================================

PROGRAM_START = time.perf_counter()


# =============================================================================
# CUDA allocator
#
# NeMo's official streaming script uses expandable_segments on supported
# Linux environments. Windows currently does not support it in our setup.
# =============================================================================

if sys.platform.startswith("linux"):
    allocator_config = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "",
    )

    if "expandable_segments" not in allocator_config:
        if allocator_config:
            allocator_config += ",expandable_segments:True"
        else:
            allocator_config = "expandable_segments:True"

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_config

# On Windows, PyTorch has already loaded compatible CUDA/cuDNN DLLs.
if sys.platform == "win32" and hasattr(ort, "preload_dlls"):
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

ASR_NEMO_FILENAME = "parakeet-tdt-0.6b-v3.nemo"


# =============================================================================
# Selected streaming configuration
# =============================================================================

ASR_LEFT_CONTEXT_S = 10.00
ASR_CHUNK_S = 0.16
ASR_RIGHT_CONTEXT_S = 1.84

EXPECTED_THEORETICAL_LATENCY_S = ASR_CHUNK_S + ASR_RIGHT_CONTEXT_S


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
    parser = argparse.ArgumentParser(
        description=("Persistent S2S runtime using stateful chunked Parakeet ASR, NLLB FP16 and Piper.")
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--audio",
        type=Path,
        help=("Single input WAV. Must be 16 kHz mono PCM16."),
    )

    input_group.add_argument(
        "--batch",
        type=Path,
        help=("JSONL batch manifest. Models are loaded once for all jobs."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=("Output WAV for single-input mode."),
    )

    parser.add_argument(
        "--source",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="Configured NLLB source language.",
    )

    parser.add_argument(
        "--target",
        choices=SUPPORTED_LANGUAGES,
        required=True,
        help="Target MT/TTS language.",
    )

    parser.add_argument(
        "--warmup-audio",
        type=Path,
        default=None,
        help=("Optional WAV used for warm-up. Defaults to the first input job."),
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help=("Disable warm-up. Useful only when explicitly studying cold inference."),
    )

    return parser.parse_args()


# =============================================================================
# Paths / jobs
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
        raise ValueError("--output is only valid with --audio. Batch outputs are specified inside the JSONL.")

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
    print("S2S PERSISTENT STREAMING RUNTIME")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")

    print(f"PyTorch      : {torch.__version__}")

    print(f"Transformers : {transformers.__version__}")

    print(f"NeMo         : {nemo.__version__}")

    print(f"ONNX Runtime : {ort.__version__}")

    print(f"CUDA         : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print(f"GPU          : {torch.cuda.get_device_name(0)}")

    providers = ort.get_available_providers()

    print(f"ORT providers: {providers}")

    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is not available.")

    print()
    print("Environment validation: PASS")


def validate_configuration(
    args: argparse.Namespace,
    jobs: list[dict],
) -> None:
    if args.source == args.target:
        raise ValueError("Source and target languages must be different.")

    for job in jobs:
        if not job["audio"].exists():
            raise FileNotFoundError(f"Input WAV not found:\n{job['audio']}")

        if not job["audio"].is_file():
            raise RuntimeError(f"Input path is not a file:\n{job['audio']}")

    if args.warmup_audio is not None:
        warmup_path = args.warmup_audio.resolve()

        if not warmup_path.exists():
            raise FileNotFoundError(f"Warm-up WAV not found:\n{warmup_path}")

    print()
    print("=" * 80)
    print("PIPELINE CONFIGURATION")
    print("=" * 80)

    print(f"Mode        : {'single' if len(jobs) == 1 and args.audio else 'batch'}")

    print(f"Jobs        : {len(jobs)}")

    print(f"Source      : {args.source.upper()}")

    print(f"Target      : {args.target.upper()}")

    print(f"Warm-up     : {'disabled' if args.no_warmup else 'enabled'}")

    print()
    print("Input contract:")

    print("WAV / 16 kHz / mono / PCM16")

    print()
    print("Parakeet does not use --source; --source configures NLLB.")


# =============================================================================
# GPU memory
# =============================================================================


def print_gpu_memory(
    label: str,
) -> dict:
    torch.cuda.synchronize()

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    free_mb = free_bytes / 1024**2

    total_mb = total_bytes / 1024**2

    device_used_mb = total_mb - free_mb

    torch_allocated_mb = torch.cuda.memory_allocated() / 1024**2

    torch_reserved_mb = torch.cuda.memory_reserved() / 1024**2

    print(
        f"{label:<18}: "
        f"device used {device_used_mb:.2f} MB | "
        f"torch allocated "
        f"{torch_allocated_mb:.2f} MB | "
        f"torch reserved "
        f"{torch_reserved_mb:.2f} MB"
    )

    return {
        "device_used_mb": device_used_mb,
        "device_free_mb": free_mb,
        "device_total_mb": total_mb,
        "torch_allocated_mb": (torch_allocated_mb),
        "torch_reserved_mb": (torch_reserved_mb),
    }


def release_unused_torch_cache() -> float:
    """
    Release unused PyTorch cached CUDA blocks.

    Model parameters remain allocated. This is useful in the current
    mixed PyTorch + ONNX Runtime process, particularly on the 6 GB
    development GPU where both frameworks compete for CUDA memory.
    """

    torch.cuda.synchronize()

    start = time.perf_counter()

    torch.cuda.empty_cache()

    torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0


# =============================================================================
# Direct WAV input
# =============================================================================


def load_input_wav(
    audio_path: Path,
    expected_sample_rate: int,
) -> tuple[
    np.ndarray,
    dict,
    float,
]:
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
# ASR helpers
# =============================================================================


def make_divisible_by(
    number: int,
    factor: int,
) -> int:
    return (number // factor) * factor


def flatten_recurrent_weights(
    model,
) -> int:
    flattened = 0

    for module in model.modules():
        if isinstance(
            module,
            RNNBase,
        ):
            module.flatten_parameters()
            flattened += 1

    return flattened


# =============================================================================
# ASR loading
# =============================================================================


def load_streaming_asr():
    print()
    print("=" * 80)
    print("LOADING STREAMING ASR")
    print("=" * 80)

    print(f"Model ID : {ASR_MODEL_ID}")

    print(f"Revision : {ASR_MODEL_REVISION}")

    try:
        checkpoint = hf_hub_download(
            repo_id=ASR_MODEL_ID,
            filename=(ASR_NEMO_FILENAME),
            revision=(ASR_MODEL_REVISION),
            local_files_only=True,
        )

    except Exception as exc:
        raise RuntimeError("Exact Parakeet checkpoint is not available locally.") from exc

    compute_dtype = get_inference_dtype(
        compute_dtype=None,
        device=DEVICE,
    )

    start = time.perf_counter()

    model: Any = nemo_asr.models.ASRModel.restore_from(
        restore_path=checkpoint,
        map_location=DEVICE,
    )

    model.freeze()
    model.eval()
    model.to(DEVICE)
    model.to(compute_dtype)

    model.preprocessor.featurizer.dither = 0.0

    model.preprocessor.featurizer.pad_to = 0

    # ------------------------------------------------------------------
    # Stateful TDT streaming decoder
    #
    # IMPORTANT:
    #
    # CUDA graph decoding is deliberately disabled.
    #
    # NeMo has known issues in persistent TDT workloads where the CUDA
    # graph decoder can leave process-wide CUDA state corrupted after
    # inference. Our runtime must support repeated ASR calls in the same
    # process, therefore correctness and stability take precedence over
    # the potential CUDA graph speed-up.
    # ------------------------------------------------------------------

    decoding_config = copy.deepcopy(model.cfg.decoding)

    with open_dict(decoding_config):
        decoding_config.strategy = "greedy_batch"

        decoding_config.tdt_include_token_duration = False

        decoding_config.greedy.loop_labels = True

        decoding_config.greedy.preserve_alignments = False

        decoding_config.greedy.use_cuda_graph_decoder = False

        decoding_config.fused_batch_size = -1

        decoding_config.beam.return_best_hypothesis = True

    model.change_decoding_strategy(decoding_config)

    try:
        decoding_computer = model.decoding.decoding.decoding_computer

    except AttributeError as exc:
        raise RuntimeError("Could not access streaming RNNT/TDT decoding computer.") from exc

    # ------------------------------------------------------------------
    # RNN compaction attempt
    #
    # Keep it for now, but do not assume it eliminates the cuDNN warning.
    # The current Windows environment still reports non-contiguous RNN
    # weights during inference.
    # ------------------------------------------------------------------

    flattened_rnns = flatten_recurrent_weights(model)

    # ------------------------------------------------------------------
    # Streaming contexts
    # ------------------------------------------------------------------

    sample_rate = int(model.cfg.preprocessor.sample_rate)

    feature_stride_s = float(model.cfg.preprocessor.window_stride)

    features_per_second = 1.0 / feature_stride_s

    subsampling_factor = int(model.encoder.subsampling_factor)

    feature_frame_to_audio_samples = make_divisible_by(
        int(sample_rate * feature_stride_s),
        factor=(subsampling_factor),
    )

    encoder_frame_to_audio_samples = feature_frame_to_audio_samples * subsampling_factor

    encoder_context = ContextSize(
        left=int(ASR_LEFT_CONTEXT_S * features_per_second / subsampling_factor),
        chunk=int(ASR_CHUNK_S * features_per_second / subsampling_factor),
        right=int(ASR_RIGHT_CONTEXT_S * features_per_second / subsampling_factor),
    )

    audio_context = ContextSize(
        left=(encoder_context.left * subsampling_factor * feature_frame_to_audio_samples),
        chunk=(encoder_context.chunk * subsampling_factor * feature_frame_to_audio_samples),
        right=(encoder_context.right * subsampling_factor * feature_frame_to_audio_samples),
    )

    attention_style = getattr(
        model.cfg.encoder,
        "att_context_style",
        None,
    )

    if attention_style == "chunked_limited_with_rc":
        model.encoder.set_default_att_context_size(
            att_context_size=[
                encoder_context.left,
                encoder_context.chunk,
                encoder_context.right,
            ]
        )

    corrected_left_s = audio_context.left / sample_rate

    corrected_chunk_s = audio_context.chunk / sample_rate

    corrected_right_s = audio_context.right / sample_rate

    theoretical_latency_s = corrected_chunk_s + corrected_right_s

    if abs(theoretical_latency_s - EXPECTED_THEORETICAL_LATENCY_S) > 1e-6:
        raise RuntimeError("Streaming latency differs from the selected 2.00 s configuration.")

    torch.cuda.synchronize()

    load_ms = (time.perf_counter() - start) * 1000.0

    actual_device = str(next(model.parameters()).device)

    actual_dtype = next(model.parameters()).dtype

    print(f"Checkpoint : {checkpoint}")

    print(f"Device     : {actual_device}")

    print(f"Dtype      : {actual_dtype}")

    print(f"Sample rate: {sample_rate} Hz")

    print(f"Attention  : {attention_style}")

    print(f"RNNs flat  : {flattened_rnns}")

    print("CUDA graph decoder: DISABLED")

    print()
    print("Corrected streaming context:")

    print(f"Left       : {corrected_left_s:.2f} s")

    print(f"Chunk      : {corrected_chunk_s:.2f} s")

    print(f"Right      : {corrected_right_s:.2f} s")

    print(f"Latency    : {theoretical_latency_s:.2f} s")

    print(f"Encoder ctx: left={encoder_context.left}, chunk={encoder_context.chunk}, right={encoder_context.right}")

    print(f"Audio ctx  : left={audio_context.left}, chunk={audio_context.chunk}, right={audio_context.right}")

    print()
    print(f"Load time  : {load_ms:.2f} ms")

    print("Streaming ASR load: PASS")

    streaming_config = {
        "sample_rate": (sample_rate),
        "encoder_frame_to_audio_samples": (encoder_frame_to_audio_samples),
        "encoder_context": (encoder_context),
        "audio_context": (audio_context),
        "left_context_s": (corrected_left_s),
        "chunk_s": (corrected_chunk_s),
        "right_context_s": (corrected_right_s),
        "theoretical_latency_s": (theoretical_latency_s),
        "compute_dtype": (str(actual_dtype)),
        "attention_style": (attention_style),
        "flattened_rnns": (flattened_rnns),
        "cuda_graph_decoder": (False),
    }

    return (
        model,
        decoding_computer,
        streaming_config,
        load_ms,
    )


# =============================================================================
# Streaming ASR inference
# =============================================================================


def run_streaming_asr_waveform(
    model,
    decoding_computer,
    streaming_config: dict,
    waveform: np.ndarray,
) -> tuple[
    str,
    dict,
]:
    sample_rate = int(streaming_config["sample_rate"])

    audio_duration_s = waveform.size / sample_rate

    audio_batch = torch.from_numpy(waveform).float().unsqueeze(0).to(DEVICE)

    audio_lengths = torch.tensor(
        [waveform.size],
        dtype=torch.long,
        device=DEVICE,
    )

    audio_context = streaming_config["audio_context"]

    encoder_frame_to_audio_samples = int(streaming_config["encoder_frame_to_audio_samples"])

    current_batched_hyps = None
    decoder_state = None

    left_sample = 0

    right_sample = min(
        (audio_context.chunk + audio_context.right),
        audio_batch.shape[1],
    )

    buffer = StreamingBatchedAudioBuffer(
        batch_size=1,
        context_samples=(audio_context),
        dtype=(audio_batch.dtype),
        device=DEVICE,
    )

    remaining_audio_lengths = audio_lengths.clone()

    chunk_count = 0

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        while left_sample < audio_batch.shape[1]:
            chunk_count += 1

            chunk_length = (
                min(
                    right_sample,
                    audio_batch.shape[1],
                )
                - left_sample
            )

            is_last_chunk_batch = chunk_length >= remaining_audio_lengths

            is_last_chunk = right_sample >= audio_batch.shape[1]

            chunk_lengths_batch = torch.where(
                is_last_chunk_batch,
                remaining_audio_lengths,
                torch.full_like(
                    remaining_audio_lengths,
                    fill_value=(chunk_length),
                ),
            )

            buffer.add_audio_batch_(
                audio_batch[:, left_sample:right_sample],
                audio_lengths=(chunk_lengths_batch),
                is_last_chunk=(is_last_chunk),
                is_last_chunk_batch=(is_last_chunk_batch),
            )

            (
                encoder_output,
                encoder_output_length,
            ) = model(
                input_signal=(buffer.samples),
                input_signal_length=(buffer.context_size_batch.total()),
            )

            encoder_output = encoder_output.transpose(
                1,
                2,
            )

            encoder_context = buffer.context_size.subsample(factor=(encoder_frame_to_audio_samples))

            encoder_context_batch = buffer.context_size_batch.subsample(factor=(encoder_frame_to_audio_samples))

            encoder_output = encoder_output[:, encoder_context.left :]

            encoder_output_length_to_decode = torch.where(
                is_last_chunk_batch,
                (encoder_output_length - encoder_context_batch.left),
                encoder_context_batch.chunk,
            )

            (
                chunk_batched_hyps,
                decoder_state,
            ) = decoding_computer(
                x=encoder_output,
                out_len=(encoder_output_length_to_decode),
                prev_batched_state=(decoder_state),
                multi_biasing_ids=None,
            )

            if current_batched_hyps is None:
                current_batched_hyps = chunk_batched_hyps

            else:
                current_batched_hyps.merge_(chunk_batched_hyps)

            remaining_audio_lengths -= chunk_lengths_batch

            left_sample = right_sample

            right_sample = min(
                (right_sample + audio_context.chunk),
                audio_batch.shape[1],
            )

    torch.cuda.synchronize()

    compute_ms = (time.perf_counter() - start) * 1000.0

    if current_batched_hyps is None:
        raise RuntimeError("Streaming ASR produced no hypothesis.")

    final_hypothesis = batched_hyps_to_hypotheses(
        current_batched_hyps,
        batch_size=1,
    )[0]

    y_sequence = final_hypothesis.y_sequence

    if isinstance(y_sequence, torch.Tensor):
        token_ids = y_sequence.tolist()
    else:
        token_ids = list(y_sequence)

    final_text = model.tokenizer.ids_to_text(token_ids).strip()

    if not final_text:
        raise RuntimeError("Streaming ASR returned empty text.")

    rtf = (compute_ms / 1000.0) / audio_duration_s

    metrics = {
        "audio_duration_s": (audio_duration_s),
        "chunk_count": (chunk_count),
        "compute_ms": (compute_ms),
        "rtf": rtf,
        "theoretical_latency_s": (streaming_config["theoretical_latency_s"]),
    }

    return (
        final_text,
        metrics,
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
            dtype=MT_DTYPE,
            local_files_only=True,
        )

    except Exception as exc:
        raise RuntimeError("Exact NLLB checkpoint is not available locally.") from exc

    model.to(DEVICE)
    model.eval()

    torch.cuda.synchronize()

    load_ms = (time.perf_counter() - start) * 1000.0

    actual_device = str(next(model.parameters()).device)

    actual_dtype = next(model.parameters()).dtype

    if actual_device != str(DEVICE):
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
    target_code = NLLB_LANGUAGE_CODES[target_language]

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)

    if forced_bos_token_id is None or forced_bos_token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Could not resolve NLLB target token: {target_code}")

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

    target_text = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    latency_ms = (time.perf_counter() - start) * 1000.0

    if not target_text:
        raise RuntimeError("NLLB returned empty text.")

    return (
        target_text,
        latency_ms,
    )


# =============================================================================
# Piper loading
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

    start = time.perf_counter()

    voice = PiperVoice.load(
        model_path,
        config_path=config_path,
        use_cuda=True,
    )

    load_ms = (time.perf_counter() - start) * 1000.0

    providers = voice.session.get_providers()

    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"Piper did not select CUDA as preferred Execution Provider:\n{providers}")

    print(f"Providers : {providers}")

    print(f"Load time : {load_ms:.2f} ms")

    print("TTS load: PASS")

    return (
        voice,
        voice_name,
        load_ms,
    )


# =============================================================================
# Piper synthesis
# =============================================================================


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

    if channels != 1:
        raise RuntimeError(f"Expected mono Piper output, got {channels} channels.")

    if sample_width != 2:
        raise RuntimeError(f"Expected PCM16 Piper output, got {sample_width}-byte samples.")

    if sample_rate <= 0:
        raise RuntimeError("Piper generated an invalid sample rate.")

    if frames <= 0:
        raise RuntimeError("Piper generated empty audio.")

    return {
        "channels": channels,
        "sample_width_bytes": (sample_width),
        "sample_rate": (sample_rate),
        "frames": frames,
        "duration_s": (frames / sample_rate),
    }


def synthesize_tts_to_memory(
    voice: PiperVoice,
    text: str,
) -> tuple[
    bytes,
    dict,
    float,
]:
    memory_file = io.BytesIO()

    # Ensure previous PyTorch CUDA work has
    # completed before ORT starts using the
    # shared CUDA stream.
    torch.cuda.synchronize()

    start = time.perf_counter()

    with wave.open(
        memory_file,
        "wb",
    ) as wav_file:
        voice.synthesize_wav(
            text,
            wav_file,
        )

    # Critical for the persistent mixed-runtime
    # path:
    #
    # 1. include all CUDA work in TTS timing;
    # 2. surface a CUDA error here, if Piper/ORT
    #    caused one;
    # 3. guarantee that the stream is clean
    #    before control returns to PyTorch ASR.
    torch.cuda.synchronize()

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
    streaming_config: dict,
    mt_tokenizer,
    mt_model,
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
        audio_load_ms,
    ) = load_input_wav(
        warmup_audio_path,
        expected_sample_rate=(streaming_config["sample_rate"]),
    )

    (
        asr_text,
        asr_metrics,
    ) = run_streaming_asr_waveform(
        model=asr_model,
        decoding_computer=(decoding_computer),
        streaming_config=(streaming_config),
        waveform=waveform,
    )

    cache_release_ms = release_unused_torch_cache()

    (
        target_text,
        mt_ms,
    ) = run_mt(
        tokenizer=mt_tokenizer,
        model=mt_model,
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

    print(f"Input load       : {audio_load_ms:.2f} ms")

    print(f"ASR              : {asr_metrics['compute_ms']:.2f} ms")

    print(f"ASR RTF          : {asr_metrics['rtf']:.4f}")

    print(f"Cache release    : {cache_release_ms:.2f} ms")

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
        "input_audio": (str(warmup_audio_path)),
        "input_duration_s": (audio_info["duration_s"]),
        "input_load_ms": (audio_load_ms),
        "asr_ms": (asr_metrics["compute_ms"]),
        "asr_rtf": (asr_metrics["rtf"]),
        "cache_release_ms": (cache_release_ms),
        "mt_ms": mt_ms,
        "tts_ms": tts_ms,
        "total_ms": total_ms,
    }


# =============================================================================
# Per-job report
# =============================================================================


def write_job_report(
    job: dict,
    args: argparse.Namespace,
    voice_name: str,
    streaming_config: dict,
    input_info: dict,
    asr_text: str,
    asr_metrics: dict,
    cache_release_ms: float,
    translated_text: str,
    mt_ms: float,
    tts_info: dict,
    tts_ms: float,
    input_load_ms: float,
    output_write_ms: float,
    steady_wall_ms: float,
    startup_info: dict,
    memory: dict,
) -> Path:
    output_path = job["output"]

    report_path = output_path.with_suffix(".json")

    compute_total_ms = asr_metrics["compute_ms"] + cache_release_ms + mt_ms + tts_ms

    file_io_ms = input_load_ms + output_write_ms

    report = {
        "experiment": ("Persistent integrated S2S streaming runtime"),
        "job_id": (job["job_id"]),
        "architecture": (
            "16 kHz PCM16 WAV -> stateful "
            "chunked Parakeet -> final ASR "
            "hypothesis -> NLLB FP16 final-only "
            "-> Piper -> output WAV"
        ),
        "configuration": {
            "input_audio": (str(job["audio"])),
            "output_audio": (str(output_path)),
            "source_language": (args.source),
            "target_language": (args.target),
        },
        "startup": (startup_info),
        "asr": {
            "model_id": (ASR_MODEL_ID),
            "revision": (ASR_MODEL_REVISION),
            "streaming_method": ("NeMo stateful chunked RNNT/TDT streaming"),
            "cache_aware": False,
            "left_context_s": (streaming_config["left_context_s"]),
            "chunk_s": (streaming_config["chunk_s"]),
            "right_context_s": (streaming_config["right_context_s"]),
            "theoretical_latency_s": (streaming_config["theoretical_latency_s"]),
            "compute_dtype": (streaming_config["compute_dtype"]),
            "flattened_rnns": (streaming_config["flattened_rnns"]),
            "final_text": (asr_text),
            "metrics": (asr_metrics),
        },
        "mt": {
            "model_id": (MT_MODEL_ID),
            "revision": (MT_MODEL_REVISION),
            "precision": (str(MT_DTYPE)),
            "policy": ("final_only"),
            "calls": 1,
            "generation": {
                "num_beams": 1,
                "do_sample": False,
                "max_new_tokens": (MT_MAX_NEW_TOKENS),
            },
            "translated_text": (translated_text),
            "latency_ms": (mt_ms),
        },
        "tts": {
            "family": "Piper",
            "voice": voice_name,
            "runtime": ("ONNX Runtime CUDA"),
            "synthesis_ms": (tts_ms),
            "output_write_ms": (output_write_ms),
            "audio": (tts_info),
        },
        "input": {
            **input_info,
            "load_ms": (input_load_ms),
        },
        "steady_state_metrics": {
            "input_load_ms": (input_load_ms),
            "asr_compute_ms": (asr_metrics["compute_ms"]),
            "torch_cache_release_ms": (cache_release_ms),
            "mt_ms": mt_ms,
            "tts_synthesis_ms": (tts_ms),
            "output_write_ms": (output_write_ms),
            "compute_and_runtime_ms": (compute_total_ms),
            "file_io_ms": (file_io_ms),
            "file_to_file_wall_ms": (steady_wall_ms),
        },
        "memory_snapshots": (memory),
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "transformers": (transformers.__version__),
            "nemo": (nemo.__version__),
            "onnxruntime": (ort.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
        },
        "target_hardware": ("NVIDIA Jetson Orin 8 GB"),
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
# Process one job using already-loaded models
# =============================================================================


def process_job(
    job_index: int,
    job_count: int,
    job: dict,
    args: argparse.Namespace,
    asr_model,
    decoding_computer,
    streaming_config: dict,
    mt_tokenizer,
    mt_model,
    tts_voice: PiperVoice,
    voice_name: str,
    startup_info: dict,
    memory: dict,
) -> dict:
    print()
    print("=" * 80)
    print(f"JOB {job_index}/{job_count}")
    print("=" * 80)

    print(f"ID     : {job['job_id']}")

    print(f"Input  : {job['audio']}")

    print(f"Output : {job['output']}")

    job_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    (
        waveform,
        input_info,
        input_load_ms,
    ) = load_input_wav(
        job["audio"],
        expected_sample_rate=(streaming_config["sample_rate"]),
    )

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    (
        asr_text,
        asr_metrics,
    ) = run_streaming_asr_waveform(
        model=asr_model,
        decoding_computer=(decoding_computer),
        streaming_config=(streaming_config),
        waveform=waveform,
    )

    del waveform

    # Mixed PyTorch/ORT memory handoff.
    cache_release_ms = release_unused_torch_cache()

    # ------------------------------------------------------------------
    # MT
    # ------------------------------------------------------------------

    (
        translated_text,
        mt_ms,
    ) = run_mt(
        tokenizer=mt_tokenizer,
        model=mt_model,
        source_text=asr_text,
        target_language=(args.target),
    )

    # ------------------------------------------------------------------
    # TTS - synthesis to memory, disk I/O separate
    # ------------------------------------------------------------------

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

    steady_wall_ms = (time.perf_counter() - job_start) * 1000.0

    compute_total_ms = asr_metrics["compute_ms"] + cache_release_ms + mt_ms + tts_ms

    print()
    print("ASR")

    print(f"Text              : {asr_text}")

    print(f"Chunks            : {asr_metrics['chunk_count']}")

    print(f"Compute           : {asr_metrics['compute_ms']:.2f} ms")

    print(f"RTF               : {asr_metrics['rtf']:.4f}")

    print(f"Theoretical latency: {asr_metrics['theoretical_latency_s']:.2f} s")

    print()
    print("MT")

    print(f"Text              : {translated_text}")

    print(f"Time              : {mt_ms:.2f} ms")

    print()
    print("TTS")

    print(f"Voice             : {voice_name}")

    print(f"Synthesis         : {tts_ms:.2f} ms")

    print(f"Audio duration    : {tts_info['duration_s']:.2f} s")

    print(f"Output write      : {output_write_ms:.2f} ms")

    print()
    print("STEADY-STATE")

    print(f"Input WAV load    : {input_load_ms:.2f} ms")

    print(f"Torch cache release: {cache_release_ms:.2f} ms")

    print(f"Compute/runtime   : {compute_total_ms:.2f} ms")

    print(f"File-to-file wall : {steady_wall_ms:.2f} ms")

    report_path = write_job_report(
        job=job,
        args=args,
        voice_name=voice_name,
        streaming_config=(streaming_config),
        input_info=(input_info),
        asr_text=(asr_text),
        asr_metrics=(asr_metrics),
        cache_release_ms=(cache_release_ms),
        translated_text=(translated_text),
        mt_ms=mt_ms,
        tts_info=tts_info,
        tts_ms=tts_ms,
        input_load_ms=(input_load_ms),
        output_write_ms=(output_write_ms),
        steady_wall_ms=(steady_wall_ms),
        startup_info=(startup_info),
        memory=memory,
    )

    print(f"Report            : {report_path}")

    print()
    print("Job: PASS")

    return {
        "job_id": (job["job_id"]),
        "audio_duration_s": (input_info["duration_s"]),
        "input_load_ms": (input_load_ms),
        "asr_ms": (asr_metrics["compute_ms"]),
        "asr_rtf": (asr_metrics["rtf"]),
        "cache_release_ms": (cache_release_ms),
        "mt_ms": (mt_ms),
        "tts_ms": (tts_ms),
        "output_write_ms": (output_write_ms),
        "compute_total_ms": (compute_total_ms),
        "file_to_file_wall_ms": (steady_wall_ms),
    }


# =============================================================================
# Batch summary
# =============================================================================


def print_batch_summary(
    results: list[dict],
) -> None:
    if len(results) <= 1:
        return

    print()
    print("=" * 80)
    print("PERSISTENT RUNTIME SUMMARY")
    print("=" * 80)

    print(f"Jobs                   : {len(results)}")

    print(f"Mean input load        : {statistics.mean(x['input_load_ms'] for x in results):.2f} ms")

    print(f"Mean ASR               : {statistics.mean(x['asr_ms'] for x in results):.2f} ms")

    print(f"Mean ASR RTF           : {statistics.mean(x['asr_rtf'] for x in results):.4f}")

    print(f"Mean cache release     : {statistics.mean(x['cache_release_ms'] for x in results):.2f} ms")

    print(f"Mean MT                : {statistics.mean(x['mt_ms'] for x in results):.2f} ms")

    print(f"Mean TTS               : {statistics.mean(x['tts_ms'] for x in results):.2f} ms")

    print(f"Mean output write      : {statistics.mean(x['output_write_ms'] for x in results):.2f} ms")

    print(f"Mean compute/runtime   : {statistics.mean(x['compute_total_ms'] for x in results):.2f} ms")

    print(f"Mean file-to-file wall : {statistics.mean(x['file_to_file_wall_ms'] for x in results):.2f} ms")


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

    memory = {}

    print()
    print("=" * 80)
    print("MODEL COEXISTENCE")
    print("=" * 80)

    memory["before_models"] = print_gpu_memory("Before models")

    # ------------------------------------------------------------------
    # Load ASR once
    # ------------------------------------------------------------------

    (
        asr_model,
        decoding_computer,
        streaming_config,
        asr_load_ms,
    ) = load_streaming_asr()

    memory["after_asr"] = print_gpu_memory("After ASR")

    # ------------------------------------------------------------------
    # Load MT once
    # ------------------------------------------------------------------

    (
        mt_tokenizer,
        mt_model,
        mt_load_ms,
    ) = load_mt(source_language=(args.source))

    memory["after_mt"] = print_gpu_memory("After MT")

    # ------------------------------------------------------------------
    # Load TTS once
    # ------------------------------------------------------------------

    (
        tts_voice,
        voice_name,
        tts_load_ms,
    ) = load_tts(target_language=(args.target))

    memory["after_tts"] = print_gpu_memory("After TTS")

    print()
    print("All three model families are loaded simultaneously.")

    print("Model coexistence: PASS")

    model_load_total_ms = asr_load_ms + mt_load_ms + tts_load_ms

    # ------------------------------------------------------------------
    # Warm-up once
    # ------------------------------------------------------------------

    warmup_info = None

    if not args.no_warmup:
        if args.warmup_audio is not None:
            warmup_audio = args.warmup_audio.resolve()
        else:
            warmup_audio = jobs[0]["audio"]

        warmup_info = warm_up_runtime(
            warmup_audio_path=(warmup_audio),
            asr_model=asr_model,
            decoding_computer=(decoding_computer),
            streaming_config=(streaming_config),
            mt_tokenizer=(mt_tokenizer),
            mt_model=(mt_model),
            tts_voice=(tts_voice),
            target_language=(args.target),
        )

        memory["after_warmup"] = print_gpu_memory("After warm-up")

    ready_ms = (time.perf_counter() - PROGRAM_START) * 1000.0

    startup_info = {
        "model_load_ms": {
            "asr": asr_load_ms,
            "mt": mt_load_ms,
            "tts": tts_load_ms,
            "total": (model_load_total_ms),
        },
        "warmup_enabled": (not args.no_warmup),
        "warmup": (warmup_info),
        "ready_ms_since_module_start": (ready_ms),
        "note": (
            "READY timing starts after module imports and includes model loading "
            "and warm-up; it excludes module import time and Python interpreter startup."
        ),
    }

    print()
    print("=" * 80)
    print("SYSTEM READY")
    print("=" * 80)

    print(f"Model load total : {model_load_total_ms:.2f} ms")

    if warmup_info is not None:
        print(f"Warm-up total    : {warmup_info['total_ms']:.2f} ms")

    print(f"READY after      : {ready_ms:.2f} ms")

    print()
    print("Models remain resident for all subsequent jobs.")

    # ------------------------------------------------------------------
    # Steady-state processing
    # ------------------------------------------------------------------

    results = []

    for index, job in enumerate(
        jobs,
        start=1,
    ):
        result = process_job(
            job_index=index,
            job_count=len(jobs),
            job=job,
            args=args,
            asr_model=asr_model,
            decoding_computer=(decoding_computer),
            streaming_config=(streaming_config),
            mt_tokenizer=(mt_tokenizer),
            mt_model=(mt_model),
            tts_voice=(tts_voice),
            voice_name=(voice_name),
            startup_info=(startup_info),
            memory=memory,
        )

        results.append(result)

    print_batch_summary(results)

    print()
    print("=" * 80)
    print("S2S PERSISTENT STREAMING RUNTIME PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
