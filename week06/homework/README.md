# Week 6 Home Assignment: Optimizing Transformer Training

This assignment focuses on applying kernel fusion, efficient operators, and distributed training techniques to optimize transformer training pipeline.

## Overview

You are given a **baseline transformer training script** that works correctly but uses non-optimized implementations. Your task is to apply various optimizations covered in the lecture and seminar, verify that model quality is preserved, and analyze the performance improvements.

### Baseline Model Inefficiencies (`baseline_model.py`)

1. Unfused RMSNorm (separate rms computation, division, scaling)
2. Unfused RoPE (separate sin/cos computation and rotation)
3. Unfused SwiGLU (separate silu and multiply kernels)
4. Vanilla attention + separate Q/K/V projections (instead of Flash Attention + fused QKV)
5. Separate lm_head + F.cross_entropy (instead of Fused Linear Cross Entropy)

### Baseline Training Inefficiencies (`baseline_train.py`)

1. DDP instead of FSDP (no parameter sharding)
2. Non-fused AdamW optimizer
3. No data prefetching (synchronous data loading)

## Part 1: Optimizations (4 points)

Apply all optimizations below. Each optimization is worth **0.5 points** (8 total = 4 points).
Points are awarded only if quality tests pass.

### Model Optimizations (2.5 points)

#### 1.1 Fused RMSNorm (0.5 points)

Replace unfused RMSNorm with `LigerRMSNorm`.

#### 1.2 Fused RoPE (0.5 points)

Replace unfused RoPE with `LigerRopeFunction`.

#### 1.3 Fused SwiGLU (0.5 points)

Replace unfused SwiGLU with `LigerSwiGLUMLP`.

#### 1.4 Flash Attention + Fused QKV (0.5 points)

Replace vanilla attention and separate Q/K/V projections.

#### 1.5 Fused Linear Cross Entropy (0.5 points)

Replace separate lm_head + F.cross_entropy with fused version.

### Training Optimizations (3 x 0.5 = 1.5 points)

#### 1.6 FSDP (0.5 points)

Replace DDP with Fully Sharded Data Parallel.

#### 1.7 Fused AdamW (0.5 points)

Enable fused optimizer.

#### 1.8 Data Prefetching (0.5 points)

Implement async data loading with CUDA streams to overlap data transfer with compute.

---

## Part 2: Calculators and Comparisons (4 points)

Implement calculators in `calculators.py` and analyze performance.

### 2.1 Theoretical Calculator (2 points)

Compute based on formulas:
- **FLOPs:** `...` for training
- **Memory:** params + gradients + optimizer states + activations
- **Time:** Roofline model (memory-bound vs compute-bound)

### 2.2 Practical Calculator (2 points)

Measure using PyTorch:
- `FlopCounterMode` for FLOPs
- `torch.cuda.max_memory_allocated()` for peak memory
- CUDA events for timing

### 2.3 Comparison Analysis (1 point)

---

## Report and Analysis (2 points)

Create a report (`report.md` or `report.pdf`) with:

1. **Speedup table** - Time comparison for each optimization
2. **Memory table** - Peak memory comparison for each optimization
3. **Theoretical vs Practical** - Compare theoretical vs practical results and explain discrepancies
