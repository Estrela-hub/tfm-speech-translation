# Jetson Deployment

This directory contains the final deployment implementation and benchmark artefacts used on the NVIDIA Jetson Orin Nano Developer Kit with 8 GB of unified memory.

## Final Architecture

The deployed Speech-to-Speech pipeline consists of:

1. **ASR — NVIDIA Parakeet-TDT 0.6B v3**
   - NeMo / PyTorch
   - CUDA
   - FP16
   - Full-utterance inference
   - Fixed-shape ASR buckets of 8, 12, 16 and 24 seconds

2. **MT — NLLB-200 Distilled 600M**
   - CTranslate2 4.8.2
   - CUDA
   - FP16
   - Greedy final-only translation

3. **TTS — Piper**
   - ONNX Runtime
   - CPUExecutionProvider

All three model families remain resident simultaneously during persistent Speech-to-Speech execution.

## Hardware and Host Environment

The final target platform was:

- NVIDIA Jetson Orin Nano Developer Kit
- 8 GB unified memory
- L4T R36.5.2
- Linux kernel 5.15.199-tegra
- Power mode: 25W

The final experiments were executed inside Docker rather than directly in the Jetson host Python environment.

## Docker Environment

The ASR environment is based on:

`nvcr.io/nvidia/pytorch:25.06-py3-igpu`

The base image is pinned to:

`sha256:05bb8855c5ab94a556653fe4f7206e922c8ec93fdb0c88a02c2e013e20cff1d2`

The Docker environment is built in two stages.

On the **Jetson**, from the project root, the ASR image can be built with:

```bash
sudo docker build \
    -f docker/Dockerfile.asr \
    -t tfm-asr:baseline \
    .
```

The final S2S image can then be built with:

```bash
sudo docker build \
    -f docker/Dockerfile.tts \
    -t tfm-tts:ct2 \
    .
```

The final Docker image used for the benchmark experiments had the image ID:

`sha256:bfbe391915560d2d3fbb651351f3916532a0d98e99921b3f91f5fef694098e47`

The runtime package snapshots are stored in:

- `environment/final_image_packages.txt`
- `environment/runtime_snapshot.json`

## Model Files

Model weights are intentionally excluded from Git.

The expected local model layout is:

```text
models/
├── parakeet/
│   └── parakeet-tdt-0.6b-v3.nemo
├── ctranslate2/
│   └── nllb-200-distilled-600M-fp16/
├── nllb_tokenizer/
└── piper/
    ├── de_DE-thorsten-medium.onnx
    ├── en_US-lessac-medium.onnx
    ├── es_ES-sharvard-medium.onnx
    ├── fr_FR-siwis-medium.onnx
    └── pt_PT-tugão-medium.onnx
```

The SHA-256 hashes of the model artefacts used in the final environment are stored in:

`environment/model_sha256.txt`

## Source Code

The final persistent Speech-to-Speech implementation is:

`src/s2s_pipeline.py`

The standalone benchmark scripts are:

- `src/asr_benchmark.py`
- `src/mt_benchmark.py`
- `src/tts_benchmark.py`
- `src/s2s_benchmark.py`

The hashes of the source files used for the final benchmark experiments are stored in:

`results/final_benchmarks/source_sha256.txt`

The current `src/s2s_pipeline.py` corresponds exactly to the final post-bucketing implementation used to produce the final benchmark results.

## Data

`data/multilingual_eval/` contains the multilingual evaluation audio, manifests and warm-up inputs used in the Jetson experiments.

`data/mt_selection/` contains the fixed FLORES evaluation manifest used for MT validation.

`data/scalability_eval/` contains the manifest used for the S2S scalability evaluation.

## Results

Structured final benchmark results are stored under:

`results/final_benchmarks/`

Generated benchmark WAV outputs and raw execution logs are intentionally excluded from Git.

The CTranslate2 MT quality validation is stored under:

`results/mt_ctranslate2_quality/`

This evaluation contains 4000 translations across 20 translation directions and was used to verify that the CTranslate2 FP16 deployment preserved the translation quality observed with the previously evaluated Transformers FP16 implementation.

## Benchmark Interpretation

Model loading and stabilization runs are excluded from the reported steady-state benchmark statistics.

`file_to_file_wall_ms` measures steady-state latency from loading the input WAV until the output WAV write operation completes.

The ASR `compute_ms` metric excludes input file loading and the initial CPU-side tensor preparation and transfer.

The MT latency includes tokenization, CTranslate2 translation and detokenization.

The TTS latency measures in-memory Piper synthesis. Output file writing is measured separately by the end-to-end pipeline.

The Jetson uses unified memory. CUDA memory measurements and system-memory measurements must therefore not be added together.

The persistent pipeline warm-up exercises all supported fixed ASR bucket shapes. Consequently, an audio file used for this all-bucket warm-up must fit within the smallest 8-second bucket.
