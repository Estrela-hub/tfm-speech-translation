from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Import torch before ORT so CUDA/cuDNN DLLs from PyTorch are already loaded.
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

WARMUP_RUNS = 3

MMS_SEED = 42


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

RESULTS_DIR = PROJECT_ROOT / "results" / "tts_selection" / "performance"

RAW_RESULTS_PATH = RESULTS_DIR / "tts_performance_results.jsonl"

LANGUAGE_SUMMARY_PATH = RESULTS_DIR / "tts_performance_by_language.csv"

GLOBAL_SUMMARY_PATH = RESULTS_DIR / "tts_performance_summary.csv"

SUMMARY_JSON_PATH = RESULTS_DIR / "tts_performance_summary.json"


# =============================================================================
# Generic helpers
# =============================================================================


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100.0

    lower = int(math.floor(position))

    upper = int(math.ceil(position))

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
                raise RuntimeError(f"Expected object in {path.name}, line {line_number}.")

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


def cuda_free_mb() -> float:
    free_bytes, _ = torch.cuda.mem_get_info()

    return free_bytes / 1024**2


def clear_cuda() -> None:
    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.synchronize()


# =============================================================================
# Environment and manifest validation
# =============================================================================


def validate_environment() -> None:
    print("=" * 80)
    print("TTS PERFORMANCE BENCHMARK")
    print("=" * 80)

    print(f"Python       : {platform.python_version()}")

    print(f"PyTorch      : {torch.__version__}")

    print(f"Transformers : {transformers.__version__}")

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


def load_and_validate_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"TTS manifest not found:\n{MANIFEST_PATH}")

    manifest_sha256 = sha256_file(MANIFEST_PATH)

    print()
    print("=" * 80)
    print("FIXED EVALUATION MANIFEST")
    print("=" * 80)

    print(f"Path   : {MANIFEST_PATH}")

    print(f"SHA256 : {manifest_sha256}")

    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Unexpected TTS manifest SHA256.\nExpected: {EXPECTED_MANIFEST_SHA256}\nActual  : {manifest_sha256}"
        )

    rows = load_jsonl(MANIFEST_PATH)

    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"Unexpected manifest size.\nExpected: {EXPECTED_CASES}\nActual  : {len(rows)}")

    required_fields = {
        "tts_case_id",
        "case_index",
        "utterance_index",
        "source_language",
        "target_language",
        "text",
        "character_count",
        "word_count",
    }

    case_ids = set()

    target_counts = Counter()

    for row_number, row in enumerate(
        rows,
        start=1,
    ):
        missing = required_fields - set(row.keys())

        if missing:
            raise RuntimeError(f"Manifest row {row_number} is missing fields: {sorted(missing)}")

        case_id = str(row["tts_case_id"])

        if case_id in case_ids:
            raise RuntimeError(f"Duplicate TTS case ID: {case_id}")

        case_ids.add(case_id)

        language = str(row["target_language"])

        if language not in LANGUAGES:
            raise RuntimeError(f"Unexpected target language: {language}")

        text = str(row["text"])

        if not text.strip():
            raise RuntimeError(f"Empty text in {case_id}.")

        target_counts[language] += 1

    for language in LANGUAGES:
        if target_counts[language] != EXPECTED_CASES_PER_LANGUAGE:
            raise RuntimeError(
                f"{language} has {target_counts[language]} cases; expected {EXPECTED_CASES_PER_LANGUAGE}."
            )

    rows.sort(key=lambda row: int(row["case_index"]))

    print(f"Cases  : {len(rows)}")

    for language in LANGUAGES:
        print(f"{language.upper():>2}     : {target_counts[language]}")

    print()
    print("Manifest validation: PASS")

    return rows


# =============================================================================
# Output validation
# =============================================================================


def validate_float_waveform(
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
        raise RuntimeError(f"{label}: waveform contains NaN or Inf.")

    if sample_rate <= 0:
        raise RuntimeError(f"{label}: invalid sample rate.")

    duration_s = waveform.size / sample_rate

    if duration_s <= 0.0:
        raise RuntimeError(f"{label}: invalid audio duration.")

    peak = float(np.max(np.abs(waveform)))

    if peak <= 0.0:
        raise RuntimeError(f"{label}: silent waveform.")

    return {
        "samples": int(waveform.size),
        "duration_s": float(duration_s),
        "peak_abs": (peak),
    }


# =============================================================================
# MMS synthesis
# =============================================================================


def synthesize_mms(
    model,
    tokenizer,
    text: str,
) -> dict:
    # VITS contains stochastic components.
    # Seed is set outside the timed region.
    set_seed(MMS_SEED)

    torch.cuda.synchronize()

    start = time.perf_counter()

    inputs = tokenizer(
        text=text,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output = model(**inputs)

    torch.cuda.synchronize()

    waveform = output.waveform.detach().float().cpu().numpy().reshape(-1)

    end = time.perf_counter()

    sample_rate = int(model.config.sampling_rate)

    waveform_info = validate_float_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        label="MMS-TTS",
    )

    synthesis_ms = (end - start) * 1000.0

    audio_duration_s = waveform_info["duration_s"]

    rtf = synthesis_ms / 1000.0 / audio_duration_s

    return {
        "synthesis_ms": (synthesis_ms),
        "sample_rate": (sample_rate),
        "audio_samples": (waveform_info["samples"]),
        "audio_duration_s": (audio_duration_s),
        "rtf": (rtf),
        "peak_abs": (waveform_info["peak_abs"]),
    }


def benchmark_mms(
    manifest: list[dict],
) -> tuple[
    list[dict],
    list[dict],
]:
    print()
    print("=" * 80)
    print("MMS-TTS FP32 BENCHMARK")
    print("=" * 80)

    raw_results = []

    model_metadata = []

    for language in LANGUAGES:
        model_id = MMS_MODELS[language]

        cases = [row for row in manifest if row["target_language"] == language]

        print()
        print("-" * 80)
        print(f"MMS-TTS | {language.upper()}")
        print("-" * 80)

        clear_cuda()

        free_before_load_mb = cuda_free_mb()

        load_start = time.perf_counter()

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

        torch.cuda.synchronize()

        load_end = time.perf_counter()

        load_ms = (load_end - load_start) * 1000.0

        actual_device = str(next(model.parameters()).device)

        actual_dtype = next(model.parameters()).dtype

        if actual_device != DEVICE:
            raise RuntimeError(f"{model_id} loaded on {actual_device}.")

        if actual_dtype != torch.float32:
            raise RuntimeError(f"{model_id} is not FP32.")

        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        parameter_memory_mb = (
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1024**2
        )

        revision = getattr(
            model.config,
            "_commit_hash",
            None,
        )

        free_after_load_mb = cuda_free_mb()

        print(f"Model         : {model_id}")

        print(f"Revision      : {revision}")

        print(f"Device        : {actual_device}")

        print(f"Dtype         : {actual_dtype}")

        print(f"Parameters    : {parameter_count:,}")

        print(f"Param memory  : {parameter_memory_mb:.2f} MB")

        print(f"Load time     : {load_ms:.2f} ms")

        print(f"Warm-up runs  : {WARMUP_RUNS}")

        warmup_text = cases[0]["text"]

        for _ in range(WARMUP_RUNS):
            synthesize_mms(
                model=model,
                tokenizer=tokenizer,
                text=warmup_text,
            )

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

        minimum_free_mb = cuda_free_mb()

        for index, case in enumerate(
            cases,
            start=1,
        ):
            result = synthesize_mms(
                model=model,
                tokenizer=tokenizer,
                text=case["text"],
            )

            current_free_mb = cuda_free_mb()

            minimum_free_mb = min(
                minimum_free_mb,
                current_free_mb,
            )

            raw_row = {
                "family": ("MMS-TTS"),
                "language": (language),
                "model_or_voice": (model_id),
                "tts_case_id": (case["tts_case_id"]),
                "case_index": int(case["case_index"]),
                "source_language": (case["source_language"]),
                "target_language": (case["target_language"]),
                "character_count": int(case["character_count"]),
                "word_count": int(case["word_count"]),
                "synthesis_ms": (result["synthesis_ms"]),
                "audio_duration_s": (result["audio_duration_s"]),
                "sample_rate": (result["sample_rate"]),
                "rtf": (result["rtf"]),
                "realtime": (result["rtf"] < 1.0),
            }

            raw_results.append(raw_row)

            print(
                f"[{index:02d}/{EXPECTED_CASES_PER_LANGUAGE}] "
                f"{case['tts_case_id']} "
                f"| "
                f"{result['synthesis_ms']:.1f} ms "
                f"| audio "
                f"{result['audio_duration_s']:.2f} s "
                f"| RTF "
                f"{result['rtf']:.4f}"
            )

        torch.cuda.synchronize()

        peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2

        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

        approximate_device_delta_mb = max(
            0.0,
            free_before_load_mb - minimum_free_mb,
        )

        model_metadata.append(
            {
                "family": ("MMS-TTS"),
                "language": (language),
                "model_or_voice": (model_id),
                "revision": (revision),
                "runtime": ("PyTorch CUDA"),
                "precision": ("float32"),
                "load_ms": (load_ms),
                "parameters": (parameter_count),
                "parameter_memory_mb": (parameter_memory_mb),
                "model_file_size_mb": (None),
                "free_before_load_mb": (free_before_load_mb),
                "free_after_load_mb": (free_after_load_mb),
                "minimum_free_observed_mb": (minimum_free_mb),
                "approx_device_memory_delta_mb": (approximate_device_delta_mb),
                "torch_peak_allocated_mb": (peak_allocated_mb),
                "torch_peak_reserved_mb": (peak_reserved_mb),
                "providers": (None),
            }
        )

        del model
        del tokenizer

        clear_cuda()

    return (
        raw_results,
        model_metadata,
    )


# =============================================================================
# Piper synthesis
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


def synthesize_piper(
    voice: PiperVoice,
    text: str,
) -> dict:
    start = time.perf_counter()

    chunk_count = 0
    total_audio_bytes = 0

    sample_rate = None
    sample_width = None
    sample_channels = None

    for audio_chunk in voice.synthesize(text):
        chunk_count += 1

        chunk_bytes = audio_chunk.audio_int16_bytes

        if not chunk_bytes:
            raise RuntimeError("Piper produced an empty audio chunk.")

        if sample_rate is None:
            sample_rate = int(audio_chunk.sample_rate)

            sample_width = int(audio_chunk.sample_width)

            sample_channels = int(audio_chunk.sample_channels)

        else:
            if int(audio_chunk.sample_rate) != sample_rate:
                raise RuntimeError("Piper sample rate changed between chunks.")

            if int(audio_chunk.sample_width) != sample_width:
                raise RuntimeError("Piper sample width changed between chunks.")

            if int(audio_chunk.sample_channels) != sample_channels:
                raise RuntimeError("Piper channel count changed between chunks.")

        total_audio_bytes += len(chunk_bytes)

    end = time.perf_counter()

    if chunk_count <= 0:
        raise RuntimeError("Piper produced no audio chunks.")

    if sample_rate is None or sample_width is None or sample_channels is None:
        raise RuntimeError("Piper audio format is missing.")

    if sample_rate <= 0 or sample_width <= 0 or sample_channels <= 0:
        raise RuntimeError("Piper returned an invalid audio format.")

    bytes_per_frame = sample_width * sample_channels

    if total_audio_bytes % bytes_per_frame != 0:
        raise RuntimeError("Piper audio byte count is not aligned to complete frames.")

    audio_frames = total_audio_bytes // bytes_per_frame

    audio_duration_s = audio_frames / sample_rate

    if audio_duration_s <= 0.0:
        raise RuntimeError("Piper produced invalid audio duration.")

    synthesis_ms = (end - start) * 1000.0

    rtf = synthesis_ms / 1000.0 / audio_duration_s

    return {
        "synthesis_ms": (synthesis_ms),
        "sample_rate": (sample_rate),
        "sample_width": (sample_width),
        "sample_channels": (sample_channels),
        "audio_frames": (audio_frames),
        "audio_duration_s": (audio_duration_s),
        "chunk_count": (chunk_count),
        "rtf": (rtf),
    }


def benchmark_piper(
    manifest: list[dict],
) -> tuple[
    list[dict],
    list[dict],
]:
    print()
    print("=" * 80)
    print("PIPER CUDA BENCHMARK")
    print("=" * 80)

    raw_results = []

    model_metadata = []

    for language in LANGUAGES:
        voice_name = PIPER_VOICES[language]

        (
            model_path,
            config_path,
        ) = get_piper_paths(language)

        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found:\n{model_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Piper config not found:\n{config_path}")

        cases = [row for row in manifest if row["target_language"] == language]

        print()
        print("-" * 80)
        print(f"PIPER | {language.upper()}")
        print("-" * 80)

        clear_cuda()

        free_before_load_mb = cuda_free_mb()

        load_start = time.perf_counter()

        voice = PiperVoice.load(
            model_path,
            config_path=(config_path),
            use_cuda=True,
        )

        load_end = time.perf_counter()

        load_ms = (load_end - load_start) * 1000.0

        providers = voice.session.get_providers()

        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{voice_name} did not use CUDA as preferred provider.\nProviders: {providers}")

        free_after_load_mb = cuda_free_mb()

        model_size_mb = model_path.stat().st_size / 1024**2

        model_sha256 = sha256_file(model_path)

        config_sha256 = sha256_file(config_path)

        print(f"Voice         : {voice_name}")

        print(f"Providers     : {providers}")

        print(f"Model size    : {model_size_mb:.2f} MB")

        print(f"Load time     : {load_ms:.2f} ms")

        print(f"Warm-up runs  : {WARMUP_RUNS}")

        warmup_text = cases[0]["text"]

        for _ in range(WARMUP_RUNS):
            synthesize_piper(
                voice=voice,
                text=warmup_text,
            )

        minimum_free_mb = cuda_free_mb()

        for index, case in enumerate(
            cases,
            start=1,
        ):
            result = synthesize_piper(
                voice=voice,
                text=case["text"],
            )

            current_free_mb = cuda_free_mb()

            minimum_free_mb = min(
                minimum_free_mb,
                current_free_mb,
            )

            raw_row = {
                "family": ("Piper"),
                "language": (language),
                "model_or_voice": (voice_name),
                "tts_case_id": (case["tts_case_id"]),
                "case_index": int(case["case_index"]),
                "source_language": (case["source_language"]),
                "target_language": (case["target_language"]),
                "character_count": int(case["character_count"]),
                "word_count": int(case["word_count"]),
                "synthesis_ms": (result["synthesis_ms"]),
                "audio_duration_s": (result["audio_duration_s"]),
                "sample_rate": (result["sample_rate"]),
                "rtf": (result["rtf"]),
                "realtime": (result["rtf"] < 1.0),
            }

            raw_results.append(raw_row)

            print(
                f"[{index:02d}/20] "
                f"{case['tts_case_id']} "
                f"| "
                f"{result['synthesis_ms']:.1f} ms "
                f"| audio "
                f"{result['audio_duration_s']:.2f} s "
                f"| RTF "
                f"{result['rtf']:.4f}"
            )

        approximate_device_delta_mb = max(
            0.0,
            free_before_load_mb - minimum_free_mb,
        )

        model_metadata.append(
            {
                "family": ("Piper"),
                "language": (language),
                "model_or_voice": (voice_name),
                "revision": (None),
                "model_sha256": (model_sha256),
                "config_sha256": (config_sha256),
                "runtime": ("ONNX Runtime CUDA"),
                "precision": ("ONNX model default"),
                "load_ms": (load_ms),
                "parameters": (None),
                "parameter_memory_mb": (None),
                "model_file_size_mb": (model_size_mb),
                "free_before_load_mb": (free_before_load_mb),
                "free_after_load_mb": (free_after_load_mb),
                "minimum_free_observed_mb": (minimum_free_mb),
                "approx_device_memory_delta_mb": (approximate_device_delta_mb),
                "torch_peak_allocated_mb": (None),
                "torch_peak_reserved_mb": (None),
                "providers": (providers),
            }
        )

        del voice

        gc.collect()

        # ORT owns its own CUDA allocator, so torch.empty_cache()
        # cannot directly release ORT allocations. Deleting the session
        # should release them when its destructor runs.
        clear_cuda()

    return (
        raw_results,
        model_metadata,
    )


# =============================================================================
# Summary construction
# =============================================================================


def build_language_summaries(
    raw_results: list[dict],
    model_metadata: list[dict],
) -> list[dict]:
    summaries = []

    metadata_lookup = {
        (
            row["family"],
            row["language"],
        ): row
        for row in model_metadata
    }

    families = [
        "MMS-TTS",
        "Piper",
    ]

    for family in families:
        for language in LANGUAGES:
            rows = [row for row in raw_results if (row["family"] == family and row["language"] == language)]

            if len(rows) != EXPECTED_CASES_PER_LANGUAGE:
                raise RuntimeError(f"{family}/{language} has {len(rows)} results.")

            latencies = [float(row["synthesis_ms"]) for row in rows]

            rtfs = [float(row["rtf"]) for row in rows]

            durations = [float(row["audio_duration_s"]) for row in rows]

            realtime_count = sum(1 for row in rows if row["realtime"])

            metadata = metadata_lookup[
                (
                    family,
                    language,
                )
            ]

            summaries.append(
                {
                    "family": (family),
                    "language": (language),
                    "cases": (len(rows)),
                    "mean_latency_ms": (statistics.mean(latencies)),
                    "median_latency_ms": (statistics.median(latencies)),
                    "p95_latency_ms": (
                        percentile(
                            latencies,
                            95,
                        )
                    ),
                    "min_latency_ms": (min(latencies)),
                    "max_latency_ms": (max(latencies)),
                    "mean_rtf": (statistics.mean(rtfs)),
                    "median_rtf": (statistics.median(rtfs)),
                    "p95_rtf": (
                        percentile(
                            rtfs,
                            95,
                        )
                    ),
                    "mean_audio_duration_s": (statistics.mean(durations)),
                    "realtime_cases": (realtime_count),
                    "realtime_percent": (realtime_count / len(rows) * 100.0),
                    "load_ms": (metadata["load_ms"]),
                    "parameter_memory_mb": (metadata["parameter_memory_mb"]),
                    "model_file_size_mb": (metadata["model_file_size_mb"]),
                    "approx_device_memory_delta_mb": (metadata["approx_device_memory_delta_mb"]),
                    "torch_peak_allocated_mb": (metadata["torch_peak_allocated_mb"]),
                    "torch_peak_reserved_mb": (metadata["torch_peak_reserved_mb"]),
                }
            )

    return summaries


def build_global_summaries(
    raw_results: list[dict],
    language_summaries: list[dict],
) -> list[dict]:
    summaries = []

    for family in [
        "MMS-TTS",
        "Piper",
    ]:
        rows = [row for row in raw_results if row["family"] == family]

        if len(rows) != EXPECTED_CASES:
            raise RuntimeError(f"{family} has {len(rows)} results; expected {EXPECTED_CASES}.")

        language_rows = [row for row in language_summaries if row["family"] == family]

        latencies = [float(row["synthesis_ms"]) for row in rows]

        rtfs = [float(row["rtf"]) for row in rows]

        durations = [float(row["audio_duration_s"]) for row in rows]

        realtime_count = sum(1 for row in rows if row["realtime"])

        summaries.append(
            {
                "family": (family),
                "cases": (len(rows)),
                "mean_latency_ms": (statistics.mean(latencies)),
                "median_latency_ms": (statistics.median(latencies)),
                "p95_latency_ms": (
                    percentile(
                        latencies,
                        95,
                    )
                ),
                "max_latency_ms": (max(latencies)),
                "mean_rtf": (statistics.mean(rtfs)),
                "median_rtf": (statistics.median(rtfs)),
                "p95_rtf": (
                    percentile(
                        rtfs,
                        95,
                    )
                ),
                "mean_audio_duration_s": (statistics.mean(durations)),
                "realtime_cases": (realtime_count),
                "realtime_percent": (realtime_count / len(rows) * 100.0),
                "mean_load_ms": (statistics.mean(row["load_ms"] for row in language_rows)),
                "max_approx_device_memory_delta_mb": (
                    max(row["approx_device_memory_delta_mb"] for row in language_rows)
                ),
                "max_torch_peak_allocated_mb": (
                    max((row["torch_peak_allocated_mb"] or 0.0) for row in language_rows)
                    if family == "MMS-TTS"
                    else None
                ),
            }
        )

    return summaries


# =============================================================================
# Console report
# =============================================================================


def print_language_summary(
    rows: list[dict],
) -> None:
    print()
    print("=" * 112)
    print("TTS PERFORMANCE BY TARGET LANGUAGE")
    print("=" * 112)

    print(
        f"{'Family':<12}"
        f"{'Lang':<8}"
        f"{'Mean ms':>12}"
        f"{'Median':>12}"
        f"{'P95 ms':>12}"
        f"{'Mean RTF':>12}"
        f"{'P95 RTF':>12}"
        f"{'RT cases':>12}"
    )

    print("-" * 112)

    for row in rows:
        print(
            f"{row['family']:<12}"
            f"{row['language'].upper():<8}"
            f"{row['mean_latency_ms']:>12.2f}"
            f"{row['median_latency_ms']:>12.2f}"
            f"{row['p95_latency_ms']:>12.2f}"
            f"{row['mean_rtf']:>12.4f}"
            f"{row['p95_rtf']:>12.4f}"
            f"{row['realtime_percent']:>11.1f}%"
        )


def print_global_summary(
    rows: list[dict],
) -> None:
    print()
    print("=" * 112)
    print("GLOBAL TTS PERFORMANCE SUMMARY")
    print("=" * 112)

    print(
        f"{'Family':<14}"
        f"{'Cases':>8}"
        f"{'Mean ms':>12}"
        f"{'Median':>12}"
        f"{'P95 ms':>12}"
        f"{'Mean RTF':>12}"
        f"{'P95 RTF':>12}"
        f"{'Realtime':>12}"
    )

    print("-" * 112)

    for row in rows:
        print(
            f"{row['family']:<14}"
            f"{row['cases']:>8}"
            f"{row['mean_latency_ms']:>12.2f}"
            f"{row['median_latency_ms']:>12.2f}"
            f"{row['p95_latency_ms']:>12.2f}"
            f"{row['mean_rtf']:>12.4f}"
            f"{row['p95_rtf']:>12.4f}"
            f"{row['realtime_percent']:>11.1f}%"
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    validate_environment()

    manifest = load_and_validate_manifest()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # MMS-TTS
    # ------------------------------------------------------------------

    (
        mms_results,
        mms_metadata,
    ) = benchmark_mms(manifest)

    # ------------------------------------------------------------------
    # Piper
    # ------------------------------------------------------------------

    (
        piper_results,
        piper_metadata,
    ) = benchmark_piper(manifest)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    raw_results = mms_results + piper_results

    model_metadata = mms_metadata + piper_metadata

    if len(mms_results) != EXPECTED_CASES:
        raise RuntimeError(f"MMS benchmark did not produce {EXPECTED_CASES} results.")

    if len(piper_results) != EXPECTED_CASES:
        raise RuntimeError(f"Piper benchmark did not produce {EXPECTED_CASES} results.")

    for row in raw_results:
        if row["synthesis_ms"] <= 0.0:
            raise RuntimeError("Invalid synthesis latency.")

        if row["audio_duration_s"] <= 0.0:
            raise RuntimeError("Invalid synthesized audio duration.")

        if row["rtf"] <= 0.0:
            raise RuntimeError("Invalid RTF.")

    language_summaries = build_language_summaries(
        raw_results=(raw_results),
        model_metadata=(model_metadata),
    )

    global_summaries = build_global_summaries(
        raw_results=(raw_results),
        language_summaries=(language_summaries),
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    write_jsonl(
        RAW_RESULTS_PATH,
        raw_results,
    )

    write_csv(
        LANGUAGE_SUMMARY_PATH,
        language_summaries,
    )

    write_csv(
        GLOBAL_SUMMARY_PATH,
        global_summaries,
    )

    summary = {
        "experiment": ("TTS performance comparison on 100 fixed final MT outputs."),
        "manifest": {
            "path": (str(MANIFEST_PATH)),
            "sha256": (EXPECTED_MANIFEST_SHA256),
            "cases": (EXPECTED_CASES),
            "cases_per_target_language": (EXPECTED_CASES_PER_LANGUAGE),
        },
        "methodology": {
            "warmup_runs_per_language": (WARMUP_RUNS),
            "model_loading_in_latency": (False),
            "disk_io_in_latency": (False),
            "mms_timing": (
                "Tokenizer preprocessing, "
                "device transfer, VITS inference, "
                "CUDA synchronization, and final "
                "waveform transfer to host memory."
            ),
            "piper_timing": (
                "Text phonemization, ONNX Runtime inference and complete consumption of all synthesized audio chunks."
            ),
            "rtf": ("wall-clock synthesis time divided by generated audio duration."),
            "realtime_definition": ("RTF < 1.0."),
            "mms_seed": (MMS_SEED),
            "mms_precision": ("FP32"),
            "piper_runtime": ("ONNX Runtime CUDAExecutionProvider"),
        },
        "memory_methodology": {
            "mms": ("PyTorch CUDA peak allocated/reserved memory plus device free-memory delta."),
            "piper": (
                "ONNX Runtime does not use the "
                "PyTorch CUDA allocator. Therefore "
                "Piper memory is reported using "
                "observed CUDA device free-memory "
                "delta before model load versus the "
                "minimum observed value."
            ),
            "warning": (
                "The Piper device-memory delta is an "
                "approximation and must not be treated "
                "as directly equivalent to PyTorch "
                "max_memory_allocated(). Final memory "
                "measurements must be repeated on the "
                "Jetson Orin 8 GB."
            ),
        },
        "model_metadata": (model_metadata),
        "by_language": (language_summaries),
        "global": (global_summaries),
        "environment": {
            "python": (platform.python_version()),
            "pytorch": (torch.__version__),
            "transformers": (transformers.__version__),
            "onnxruntime": (ort.__version__),
            "gpu": (torch.cuda.get_device_name(0)),
            "ort_providers": (ort.get_available_providers()),
        },
        "target_hardware": {
            "device": ("NVIDIA Jetson Orin"),
            "memory_gb": (8),
            "note": ("Laptop results are development measurements only."),
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

    print_language_summary(language_summaries)

    print_global_summary(global_summaries)

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(f"MMS-TTS cases : {len(mms_results)}/{EXPECTED_CASES}")

    print(f"Piper cases   : {len(piper_results)}/{EXPECTED_CASES}")

    print("Outputs valid : PASS")

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(f"Raw results : {RAW_RESULTS_PATH}")

    print(f"By language : {LANGUAGE_SUMMARY_PATH}")

    print(f"Summary CSV : {GLOBAL_SUMMARY_PATH}")

    print(f"Summary JSON: {SUMMARY_JSON_PATH}")

    print()
    print("=" * 80)
    print("TTS PERFORMANCE BENCHMARK PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
