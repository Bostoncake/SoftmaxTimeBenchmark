# Attention Implementations Guide

This document explains the different attention implementations in transformers and how they affect benchmarking.

## Overview

Modern transformers support multiple attention implementations, each with different performance characteristics and profiling implications:

1. **Eager Attention** - Standard implementation with explicit operations
2. **SDPA** - PyTorch's fused Scaled Dot-Product Attention
3. **Flash Attention 2** - Highly optimized fused kernel

## Attention Implementations

### 1. Eager Attention (`--attn-implementation eager`)

**Description:**
- Standard attention implementation with explicit matrix operations
- Softmax is called as a separate `F.softmax()` operation
- All operations are discrete and measurable

**Advantages:**
- ✅ **Exact softmax profiling** - we can precisely measure softmax time
- ✅ Easy to understand and debug
- ✅ Works on all hardware

**Disadvantages:**
- ❌ Slower than fused implementations
- ❌ Higher memory usage

**When to use:**
- **For accurate benchmarking** - this is the default and recommended option
- When you need exact softmax timing measurements
- For educational purposes or debugging

**Example:**
```bash
python run_benchmark.py --attn-implementation eager
```

### 2. SDPA - Scaled Dot-Product Attention (`--attn-implementation sdpa`)

**Description:**
- Uses PyTorch's `torch.nn.functional.scaled_dot_product_attention`
- Fuses QK^T, softmax, and attention matrix multiplication into one operation
- Available in PyTorch 2.0+

**Advantages:**
- ✅ Faster than eager attention
- ✅ Lower memory usage
- ✅ Still reasonably profil able

**Disadvantages:**
- ❌ **Softmax time is estimated** (~10% of SDPA time)
- ❌ Less precise measurements

**When to use:**
- For production inference (faster)
- When approximate softmax timing is acceptable

**Example:**
```bash
python run_benchmark.py --attn-implementation sdpa
```

**Profiling Note:**
Our profiler estimates softmax time as approximately 10% of total SDPA time, based on the typical breakdown:
- QK^T computation: ~40%
- Softmax: ~10%
- Attention matrix multiplication: ~50%

### 3. Flash Attention 2 (`--attn-implementation flash_attention_2`)

**Description:**
- Highly optimized kernel from the Flash Attention paper
- Fuses all attention operations with memory-efficient implementation
- Requires specific hardware (modern NVIDIA GPUs)

**Advantages:**
- ✅ Fastest implementation
- ✅ Most memory efficient
- ✅ Best for production

**Disadvantages:**
- ❌ **Very difficult to profile softmax separately**
- ❌ Requires flash-attn package installation
- ❌ Only works on compatible GPUs (Ampere or newer)

**When to use:**
- For production inference when speed is critical
- Not recommended for benchmarking softmax specifically

**Example:**
```bash
pip install flash-attn
python run_benchmark.py --attn-implementation flash_attention_2
```

**Profiling Note:**
Flash Attention is a single fused kernel. Our profiler cannot accurately measure softmax time separately. Use `eager` for accurate measurements.

## Profiling Mechanism

### How We Profile Each Implementation

#### Eager Attention
```python
# Transformers code (simplified)
def attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))  # QK^T
    attn_weights = F.softmax(scores, dim=-1)        # ← We patch this!
    output = torch.matmul(attn_weights, V)
    return output
```

We monkey-patch `F.softmax()` to measure execution time precisely.

#### SDPA
```python
# Transformers code (simplified)
def attention(Q, K, V):
    output = F.scaled_dot_product_attention(Q, K, V)  # ← Fused operation
    return output
```

We monkey-patch `F.scaled_dot_product_attention()` and estimate softmax as ~10% of total time.

#### Flash Attention 2
```python
# Transformers code (simplified)
from flash_attn import flash_attn_func

def attention(Q, K, V):
    output = flash_attn_func(Q, K, V)  # ← Fully fused kernel
    return output
```

Cannot separate softmax time from the fused kernel. Measurements are approximate.

## Benchmark Results Interpretation

### Result Fields

```json
{
  "softmax_time": 2.15,
  "softmax_count": 12288,
  "sdpa_count": 0,
  "attention_implementation": "eager (softmax time exact)",
  ...
}
```

**Fields:**
- `softmax_time` - Time spent in softmax operations (exact or estimated)
- `softmax_count` - Number of softmax calls detected
- `sdpa_count` - Number of SDPA calls detected
- `sdpa_time` - Total time in SDPA operations
- `attention_implementation` - Which implementation was detected

**Interpretation:**
- `eager (softmax time exact)` - Measurements are precise ✅
- `sdpa (softmax time estimated)` - Softmax time is ~10% of SDPA time ⚠️
- `unknown (possibly flash_attention_2)` - Cannot measure softmax separately ❌

## Recommendations

### For Accurate Benchmarking

**✅ DO:**
- Use `--attn-implementation eager` (default)
- Run multiple times (`--num-runs 3`)
- Use consistent hardware and settings

**❌ DON'T:**
- Mix different attention implementations in comparisons
- Use flash_attention_2 for softmax-specific benchmarks
- Compare eager vs SDPA results directly

### For Production Inference

**✅ DO:**
- Use `flash_attention_2` for maximum speed
- Use `sdpa` as a fallback
- Profile once with `eager` to understand bottlenecks

## Example Comparisons

### Same Model, Different Implementations

```bash
# Exact softmax measurement
python run_benchmark.py --models llama2-7b --attn-implementation eager

# Faster inference, estimated softmax
python run_benchmark.py --models llama2-7b --attn-implementation sdpa

# Fastest inference, no softmax breakdown
python run_benchmark.py --models llama2-7b --attn-implementation flash_attention_2
```

Expected results (approximate):
- Eager: 100% relative time, exact softmax measurement
- SDPA: ~70% relative time, estimated softmax (~10% of SDPA)
- Flash2: ~40% relative time, cannot measure softmax separately

### Comparing Models (Always Use Same Implementation)

```bash
# Compare all models with eager attention for accurate softmax profiling
python run_benchmark.py \
    --models all \
    --attn-implementation eager \
    --num-runs 3
```

## Technical Details

### Why Can't We Profile Flash Attention?

Flash Attention is implemented as a CUDA kernel that:
1. Loads Q, K, V from HBM to SRAM in blocks
2. Computes attention in SRAM (QK^T, softmax, matmul all fused)
3. Writes output back to HBM

The softmax is computed inside the kernel without separate function calls, making it impossible to intercept and measure separately.

### SDPA Estimation Accuracy

The 10% estimation for softmax in SDPA is based on:
- Theoretical FLOPs analysis
- Empirical measurements with eager attention
- May vary by sequence length and hardware

For sequence length L and dimension D:
- QK^T: O(L² × D) operations
- Softmax: O(L²) operations (much smaller!)
- Attention @ V: O(L² × D) operations

The softmax is typically 5-15% of total SDPA time, depending on L and D ratio.

## Troubleshooting

### "No softmax or SDPA operations detected!"

This warning means the profiler couldn't detect softmax operations. Possible causes:

1. **Flash Attention is being used**
   - Solution: Use `--attn-implementation eager`

2. **Model uses custom attention**
   - Some models have custom implementations
   - Check model config and force eager attention

3. **Generation didn't happen**
   - Check for errors in token generation
   - Verify model loaded correctly

### "Softmax time is 0"

Possible causes:

1. **Using Flash Attention**
   - Switch to `--attn-implementation eager`

2. **Profiler not enabled**
   - This shouldn't happen; file a bug report

3. **Model doesn't use attention**
   - Very unlikely for LLMs; check model architecture

### "Results vary between runs"

This is normal due to:
- GPU thermal throttling
- System load variations
- CUDA kernel variations

Solutions:
- Use `--num-runs 5` or more
- Ensure consistent GPU state
- Check GPU temperature and clocks

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [PyTorch SDPA Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Transformers Attention Implementations](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)

## Summary

| Implementation | Softmax Accuracy | Speed | Memory | Recommended For |
|----------------|------------------|-------|--------|-----------------|
| Eager | ✅ Exact | Slow | High | **Benchmarking** |
| SDPA | ⚠️ Estimated | Medium | Medium | Production (fallback) |
| Flash Attention 2 | ❌ Unknown | Fast | Low | Production (best) |

**For this benchmark suite: Always use `eager` (default) for accurate softmax measurements.**
