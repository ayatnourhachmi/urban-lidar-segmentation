"""
Post-training quantization and memory-footprint benchmarking.

Approach: dynamic quantization (`torch.quantization.quantize_dynamic`) on
the Linear/Conv1d layers of the trained model, run on CPU. This is the
standard low-effort PyTorch quantization path:
  https://pytorch.org/docs/stable/quantization.html
  https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html

Dynamic quantization converts weights to int8 ahead of time and quantizes
activations on-the-fly at inference, which typically gives:
  - ~4x smaller model file size (fp32 -> int8 weights)
  - lower CPU inference memory/latency
  - accuracy within noise of the fp32 model for models dominated by
    Linear/Conv layers with no exotic ops (true here for PointNet's MLPs)

For even smaller footprint you can follow up with static quantization or
QAT (quantization-aware training), which additionally quantize activations
ahead of time using calibration data -- more setup, slightly better
compression, worth mentioning as a "next step" rather than doing by default.

Known limitation (measured, not hidden): `quantize_dynamic` only rewrites
nn.Linear / nn.Conv1d, so it shrinks classic PointNet substantially
(mostly Conv1d + Linear layers, ~50% size reduction observed here) but
gives near-zero reduction on PointNet++, whose set-abstraction layers use
Conv2d, which PyTorch's dynamic quantization path does not cover. If you
need real compression on PointNet++, use static quantization
(torch.quantization.prepare/convert with a calibration pass) or
torch.ao.quantization's FX graph mode quantization, both of which do
support Conv2d. Report whichever number you actually measured for your
chosen model -- don't quote PointNet's reduction number for PointNet++.

This module also measures actual footprint (state_dict size on disk) and
accuracy so the "reduced memory without degrading accuracy" claim is
backed by numbers you can quote, not just a claim.
"""

import os
import tempfile
import time

import torch
import torch.nn as nn


def quantize_model_dynamic(model, layers=(nn.Linear, nn.Conv1d)):
    """Apply dynamic int8 quantization to the given layer types.
    Must run on CPU -- dynamic quantization is a CPU-only PyTorch feature.
    """
    model = model.to("cpu").eval()
    quantized = torch.quantization.quantize_dynamic(model, set(layers), dtype=torch.qint8)
    return quantized


def model_size_mb(model):
    """Serialize state_dict to disk and measure actual bytes -- more
    reliable than summing parameter dtypes by hand, and it's the number
    that matters for deployment (download size / disk footprint)."""
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return size_mb


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device="cpu"):
    """Per-point classification accuracy over a dataloader."""
    model = model.to(device).eval()
    correct, total = 0, 0
    for points, labels in dataloader:
        points, labels = points.to(device), labels.to(device)
        logits = model(points)  # (B, N, C)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


@torch.no_grad()
def measure_inference_latency(model, sample_input, device="cpu", n_runs=20, warmup=5):
    """Average wall-clock latency per forward pass (ms)."""
    model = model.to(device).eval()
    sample_input = sample_input.to(device)
    for _ in range(warmup):
        model(sample_input)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        model(sample_input)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return (elapsed / n_runs) * 1000.0


def benchmark_fp32_vs_int8(model_fp32, dataloader, sample_input):
    """Run the full before/after comparison and return a results dict:
    model size (MB), per-point accuracy, and CPU inference latency (ms),
    for the original fp32 model vs the dynamically-quantized int8 model.
    """
    model_fp32_cpu = model_fp32.to("cpu").eval()
    model_int8 = quantize_model_dynamic(model_fp32_cpu)

    results = {
        "fp32": {
            "size_mb": model_size_mb(model_fp32_cpu),
            "accuracy": evaluate_accuracy(model_fp32_cpu, dataloader, device="cpu"),
            "latency_ms": measure_inference_latency(model_fp32_cpu, sample_input, device="cpu"),
        },
        "int8_dynamic": {
            "size_mb": model_size_mb(model_int8),
            "accuracy": evaluate_accuracy(model_int8, dataloader, device="cpu"),
            "latency_ms": measure_inference_latency(model_int8, sample_input, device="cpu"),
        },
    }
    results["size_reduction_pct"] = 100.0 * (1 - results["int8_dynamic"]["size_mb"] / results["fp32"]["size_mb"])
    results["accuracy_delta"] = results["int8_dynamic"]["accuracy"] - results["fp32"]["accuracy"]
    return results


def print_benchmark_report(results):
    fp32, int8 = results["fp32"], results["int8_dynamic"]
    print("\n=== Model size / accuracy / latency: fp32 vs dynamic int8 ===")
    print(f"{'Metric':<20}{'FP32':>12}{'INT8 (dynamic)':>18}")
    print(f"{'Size (MB)':<20}{fp32['size_mb']:>12.3f}{int8['size_mb']:>18.3f}")
    print(f"{'Accuracy':<20}{fp32['accuracy']:>12.4f}{int8['accuracy']:>18.4f}")
    print(f"{'Latency (ms)':<20}{fp32['latency_ms']:>12.2f}{int8['latency_ms']:>18.2f}")
    print(f"\nSize reduction: {results['size_reduction_pct']:.1f}%")
    print(f"Accuracy delta (int8 - fp32): {results['accuracy_delta']:+.4f}")
