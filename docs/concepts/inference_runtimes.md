# Inference Runtimes

cells2table currently supports OpenCV, ONNX Runtime, and Transformers inference runtimes.

## ONNX Runtime

**Extra**: `onnxruntime`

Needs OpenCV for image preprocessing.

Although untested, enabled ONNX Runtime builds should allow running on GPUs.

## OpenCV

**Extra**: `opencv` / `opencv-headless`

ONNX inference using OpenCV's DNN module.

## Transformers (Hugging Face)

**Extra**: `transformers`

Safetensors inference using the popular Hugging Face library.
