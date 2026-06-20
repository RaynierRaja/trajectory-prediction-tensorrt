# Autonomous Trajectory Prediction Pipeline

This repository contains a low-latency, end-to-end autonomous trajectory prediction pipeline designed for streaming telemetry data. It features a sequence-based Transformer architecture trained in PyTorch and compiled for high-performance edge deployment.

## Key Features
* **Sequence-Based Transformer:** Captures complex temporal dependencies in continuous streaming telemetry for highly accurate spatial predictions.
* **Optimized Deployment Workflow:** Model export via ONNX with subsequent compilation into NVIDIA TensorRT execution wrappers.
* **High-Performance Execution:** Leverages TensorRT FP16 precision, drastically reducing inference latency while preserving predictive fidelity on edge hardware.

## Tech Stack
* **Framework:** PyTorch
* **Optimization:** ONNX, NVIDIA TensorRT (FP16)
* **Languages:** Python, C++ (Inference Wrapper)

## Pipeline Overview
1. **Training & Sequence Modeling:** PyTorch Transformer handles historical trajectory sequences.
2. **ONNX Export:** Freezes weights and exports the dynamic graph structure.
3. **TensorRT Compilation:** Optimizes CUDA kernels and quantizes the model to FP16.

## Usage

### Inference Pipeline
To run the TensorRT optimized inference engine on a sample telemetry stream:
```bash
python inference/predict_stream.py --engine models/trajectory_transformer_fp16.engine --input data/sample_telemetry.csv
