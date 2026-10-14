# Lightweight Real-Time Speech-to-Speech Translation on Edge Devices

Repository accompanying the Master's thesis **Lightweight Real-Time Speech-to-Speech Translation on Edge Devices**.

The project investigates a fully offline multilingual Speech-to-Speech (S2S) translation system designed for execution on resource-constrained edge hardware. The system follows a cascaded architecture:

**ASR → MT → TTS**

The final deployment platform is the **NVIDIA Jetson Orin Nano Developer Kit with 8 GB of unified memory**.

The repository separates the model-selection and development experiments performed on an NVIDIA RTX 3060 Laptop GPU from the final deployment and evaluation performed on the Jetson.

## Repository Structure

### `desktop_rtx3060/`

Experiments and initial integration work executed on an NVIDIA RTX 3060 Laptop GPU.

This directory includes:

- ASR model selection and processing-strategy experiments;
- MT model selection and translation-policy experiments;
- TTS model selection and evaluation;
- initial end-to-end Speech-to-Speech integration prototypes;
- evaluation datasets, manifests and structured experimental results.

The desktop experiments were primarily used to select the models and processing strategies later deployed on the Jetson.

### `jetson/`

Final deployment implementation and evaluation for the NVIDIA Jetson Orin Nano.

This directory includes:

- the final persistent S2S implementation;
- standalone ASR, MT, TTS and end-to-end benchmark scripts;
- Dockerfiles used to construct the final runtime environment;
- runtime and model provenance information;
- multilingual and scalability evaluation data;
- structured results from the final Jetson experiments.

Additional deployment information is available in `jetson/README.md`.

## Final System

The final pipeline uses the following models and runtimes:

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

The three model families remain resident simultaneously during persistent S2S execution.

## Supported Languages

The final system is configured for:

- English (`en`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Portuguese (`pt`)

The ASR model performs automatic spoken-language recognition, while the source and target languages used by the translation stage are explicitly configured.

## Evaluation

The final Jetson evaluation uses audio files as input and generates translated audio files as output.

The evaluation includes:

- standalone ASR latency and RTF;
- standalone MT latency;
- standalone TTS latency and RTF;
- end-to-end file-to-file S2S latency;
- multilingual execution;
- scalability across different input durations;
- memory behaviour;
- validation of CTranslate2 MT quality against the previously evaluated Transformers FP16 implementation.

Model loading and runtime stabilization are kept separate from the reported steady-state execution measurements.

The structured final results are available under:

`jetson/results/final_benchmarks/`

The CTranslate2 translation-quality validation is available under:

`jetson/results/mt_ctranslate2_quality/`

## Reproducibility

The repository retains the configuration and provenance information required to identify the software and model artefacts used during the experiments.

For the Jetson deployment, this includes:

- pinned Docker base image;
- Docker build configuration;
- package snapshots;
- runtime version information;
- model SHA-256 hashes;
- source-code SHA-256 hashes associated with the final benchmarks.

Relevant files are available under:

`jetson/environment/`

and:

`jetson/results/final_benchmarks/`

## Repository Notes

Large model weights and external repositories are intentionally excluded from version control.

Generated benchmark audio outputs and raw execution logs are also excluded when the corresponding structured results are retained.

Evaluation audio required to reproduce the reported experiments is included where appropriate.

The repository therefore focuses on source code, experimental configuration, evaluation inputs, structured results and provenance information rather than distributing large third-party model artefacts.
