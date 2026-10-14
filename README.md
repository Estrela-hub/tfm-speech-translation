# Lightweight Real-Time Speech-to-Speech Translation on Edge Devices

Repository accompanying the Master's thesis on fully offline multilingual speech-to-speech translation on edge devices.

The system follows a cascaded architecture:

ASR → MT → TTS

with NVIDIA Jetson Orin Nano 8 GB as the final deployment platform.

## Repository structure

### `desktop_rtx3060/`

Experiments executed on an NVIDIA RTX 3060 Laptop GPU, including:

- ASR model selection and streaming experiments;
- MT model selection and translation-policy experiments;
- TTS model selection and evaluation;
- initial end-to-end speech-to-speech integration prototypes.

### `jetson/`

Final deployment implementation and evaluation for the NVIDIA Jetson Orin Nano.

## Final model selection

- ASR: NVIDIA Parakeet-TDT 0.6B v3
- MT: NLLB-200 Distilled 600M
- TTS: Piper

## Notes

Large model weights, external repositories and reproducible generated audio
artefacts are intentionally excluded from version control.