# Jetson Deployment

This directory contains the final implementation and benchmark results produced on the NVIDIA Jetson Orin Nano Developer Kit with 8 GB of unified memory.

## Final Architecture

The final Speech-to-Speech pipeline consists of:

1. **ASR — NVIDIA Parakeet-TDT 0.6B v3**
   - NeMo / PyTorch
   - CUDA
   - FP16
   - full-utterance inference
   - fixed-shape ASR buckets of 8, 12, 16 and 24 seconds

2. **MT — NLLB-200 Distilled 600M**
   - CTranslate2 4.8.2
   - CUDA
   - FP16
   - greedy final-only translation

3. **TTS — Piper**
   - ONNX Runtime
   - CPUExecutionProvider

All three components remain loaded during the execution of the Speech-to-Speech pipeline.

## Hardware and Host Environment

The final target platform was:

- NVIDIA Jetson Orin Nano Developer Kit
- 8 GB unified memory
- L4T R36.5.2
- Linux kernel 5.15.199-tegra
- power mode: 25W

The final experiments were executed inside Docker instead of directly using the Jetson host Python environment.

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

The Docker image used for the final benchmark experiments had the following image ID:

`sha256:bfbe391915560d2d3fbb651351f3916532a0d98e99921b3f91f5fef694098e47`

The runtime package information is stored in:

- `environment/final_image_packages.txt`
- `environment/runtime_snapshot.json`

## Model Files

Model weights are not included in Git.

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

The SHA-256 hashes of the model files used in the final environment are stored in:

`environment/model_sha256.txt`

## Source Code

The final Speech-to-Speech implementation is:

`src/s2s_pipeline.py`

The standalone benchmark scripts are:

- `src/asr_benchmark.py`
- `src/mt_benchmark.py`
- `src/tts_benchmark.py`
- `src/s2s_benchmark.py`

The hashes of the source files used for the final benchmark experiments are stored in:

`results/final_benchmarks/source_sha256.txt`

The current `src/s2s_pipeline.py` is the same post-bucketing implementation used to produce the final benchmark results.

## Data

`data/multilingual_eval/` contains the multilingual evaluation audio, manifests and warm-up inputs used in the Jetson experiments.

`data/mt_selection/` contains the fixed FLORES evaluation manifest used for MT validation.

`data/scalability_eval/` contains the manifest used for the S2S scalability evaluation.

## Results

The final benchmark results are stored in:

`results/final_benchmarks/`

Generated benchmark WAV outputs and raw execution logs are not included in Git.

The CTranslate2 MT quality validation is stored in:

`results/mt_ctranslate2_quality/`

This evaluation contains 4000 translations across 20 translation directions. It was used to verify that the CTranslate2 FP16 implementation maintained the translation quality measured with the previously evaluated Transformers FP16 implementation.

## Benchmark Interpretation

Model loading and stabilization runs are not included in the reported steady-state benchmark statistics.

`file_to_file_wall_ms` measures the steady-state latency from loading the input WAV until the output WAV write operation is completed.

The ASR `compute_ms` metric does not include input file loading, initial CPU-side tensor preparation or the initial transfer to the GPU.

The MT latency includes tokenization, CTranslate2 translation and detokenization.

The TTS latency measures Piper synthesis in memory. The output file write is included separately in the end-to-end pipeline measurement.

The Jetson uses unified memory, so CUDA memory measurements and system-memory measurements should not be added together.

The persistent pipeline warm-up runs all supported fixed ASR bucket shapes. The audio file used for this warm-up must fit within the smallest 8-second bucket.