# Lightweight Real-Time Speech-to-Speech Translation on Edge Devices

Repository accompanying the Master's thesis **Lightweight Real-Time Speech-to-Speech Translation on Edge Devices**.

This project evaluates a fully offline multilingual Speech-to-Speech (S2S) translation system designed to run on edge hardware. The system uses a cascaded architecture:

**ASR → MT → TTS**

The final system is deployed on an **NVIDIA Jetson Orin Nano Developer Kit with 8 GB of unified memory**.

The repository is divided between the experiments performed on an NVIDIA RTX 3060 Laptop GPU during model selection and development, and the final implementation and evaluation on the Jetson.

## Repository Structure

### `desktop_rtx3060/`

Experiments and initial integration work performed on an NVIDIA RTX 3060 Laptop GPU.

This directory includes:

- ASR model selection and processing-strategy experiments;
- MT model selection and translation-policy experiments;
- TTS model selection and evaluation;
- initial end-to-end Speech-to-Speech integration prototypes;
- evaluation datasets, manifests and experimental results.

These experiments were used to select the models and processing strategies that were later used on the Jetson.

### `jetson/`

Final implementation and evaluation on the NVIDIA Jetson Orin Nano.

This directory includes:

- the final S2S implementation;
- standalone ASR, MT, TTS and end-to-end benchmark scripts;
- Dockerfiles used to build the runtime environment;
- runtime and model information;
- multilingual and scalability evaluation data;
- results from the final Jetson experiments.

More information about the Jetson implementation is available in `jetson/README.md`.

## Final System

The final pipeline uses the following models and runtimes.

### ASR

**NVIDIA Parakeet-TDT 0.6B v3**

- NeMo / PyTorch
- CUDA
- FP16
- full-utterance inference
- fixed-shape input bucketing at 8, 12, 16 and 24 seconds

### MT

**NLLB-200 Distilled 600M**

- CTranslate2
- CUDA
- FP16
- greedy decoding
- final-only translation

### TTS

**Piper**

- ONNX Runtime
- CPUExecutionProvider
- language-specific voices for English, Spanish, French, German and Portuguese

All three components remain loaded during the execution of the S2S pipeline.

## Supported Languages

The final system is configured for:

- English (`en`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Portuguese (`pt`)

The ASR model automatically identifies the spoken language. The source and target languages used by the translation stage are configured for each translation direction.

## Evaluation

The final evaluation on the Jetson uses audio files as input and generates translated audio files as output.

The evaluation includes:

- standalone ASR latency and RTF;
- standalone MT latency;
- standalone TTS latency and RTF;
- end-to-end file-to-file S2S latency;
- multilingual execution;
- scalability across different input durations;
- memory usage;
- validation of CTranslate2 MT quality against the Transformers FP16 implementation evaluated during model selection.

Model loading and runtime stabilization are not included in the reported steady-state execution measurements.

The final benchmark results are available in:

`jetson/results/final_benchmarks/`

The CTranslate2 translation quality validation is available in:

`jetson/results/mt_ctranslate2_quality/`

## Reproducibility

The repository includes information about the software environment, model versions and source code used for the final experiments.

For the Jetson implementation, this includes:

- the Docker base image used;
- Docker build configuration;
- installed package versions;
- runtime version information;
- model SHA-256 hashes;
- source-code SHA-256 hashes for the final benchmarks.

The relevant files are available in:

`jetson/environment/`

and:

`jetson/results/final_benchmarks/`

## Repository Notes

Large model weights and external repositories are not included in version control.

Generated audio outputs from the final benchmarks and raw execution logs are also excluded when the corresponding structured results are available.

The audio files used as input for the evaluation are included in the repository.

The repository contains the source code, evaluation data, experimental configuration and results needed to document the development and evaluation of the system, without including the large third-party model files.