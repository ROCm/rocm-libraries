# GEMM Data Distribution Benchmark Guide

This guide explains how to benchmark the 8192x8192x8192 GEMM kernel with different data distributions to determine if performance is data-dependent.

## Available Data Distributions

| Init Value | Distribution | Description | Use Case |
|------------|-------------|-------------|----------|
| `0` | **Uniform Random** | Random values from [-1.0, 1.0] | Realistic general workload |
| `1` | **Monotonic Sequence** | Sequential values (0, 1, 2, ...) | Cache-friendly pattern |
| `2` | **Constant (1.0)** | All values = 1.0 | Best-case scenario |
| `3` | **Zeros** | All values = 0.0 | Minimal computation |
| `4` | **Constant (π)** | All values = 3.14159... | Different constant pattern |
| `5` | **Checkerboard** | Alternating +1/-1 pattern | Worst-case cache pattern |

## Quick Start

### 1. Build the Benchmark

First, make sure you have a compiled GEMM benchmark binary. If using the build system:

```bash
# Navigate to ComposableKernel
cd /workspace/rocm-libraries/projects/composablekernel

# Build your specific GEMM kernel benchmark
# (Replace with your actual build commands)
```

### 2. Run Single Distribution Test

Test with a specific data distribution:

```bash
./gemm_benchmark \
    --m 8192 \
    --n 8192 \
    --k 8192 \
    --init 0 \
    --warmup 50 \
    --repeat 100 \
    --verify 0 \
    --json_output true
```

Change `--init` value (0-5) to test different distributions.

### 3. Run Full Distribution Comparison

Use the provided script to test all distributions:

```bash
# Make script executable
chmod +x benchmark_data_distributions.sh

# Run benchmark (adjust paths as needed)
./benchmark_data_distributions.sh \
    ./path/to/gemm_benchmark \
    8192 \
    50 \
    100 \
    ./results
```

**Arguments:**
1. Path to GEMM benchmark binary
2. Matrix size (default: 8192)
3. Warmup iterations (default: 50)
4. Repeat iterations (default: 100)
5. Output directory (default: ./benchmark_results)

### 4. Analyze Results

Use the Python analysis script:

```bash
# Make script executable
chmod +x analyze_distribution_benchmark.py

# Analyze CSV results
python3 analyze_distribution_benchmark.py \
    ./results/distribution_benchmark_8192_*.csv \
    --output report.md

# Or analyze JSON results
python3 analyze_distribution_benchmark.py \
    ./results/distribution_benchmark_8192_*.json \
    --output report.md
```

## Understanding Results

### Performance Metrics

- **Latency (ms)**: Time per GEMM operation (lower is better)
- **TFlops**: Floating-point operations per second (higher is better)
- **Bandwidth (GB/s)**: Memory bandwidth utilization (higher is better)

### What Variation Means

| Variation | Interpretation |
|-----------|----------------|
| **< 2%** | Performance is **NOT data-dependent** |
| **2-5%** | Minimal data dependency (within measurement noise) |
| **5-10%** | Moderate data dependency |
| **> 10%** | Strong data dependency |

### Expected Patterns

Different distributions may reveal:

1. **Cache Effects**
   - Monotonic sequence: Better prefetching
   - Checkerboard: Poor cache locality

2. **Denormal Numbers**
   - Zeros: No denormals
   - Random: May generate denormals (slower)

3. **Numerical Stability**
   - Large constants (π): Test accumulation errors
   - Alternating signs: Test cancellation

4. **Memory Patterns**
   - Sequential: Coalesced memory access
   - Random: Scattered access patterns

## Example Output

```
Distribution                    | Latency (ms) | TFlops | Bandwidth (GB/s)
Uniform_Random_[-1,1]          |        45.23 | 242.18 |         3245.67
Monotonic_Sequence             |        44.89 | 243.91 |         3268.45
Constant_1.0                   |        45.01 | 243.25 |         3257.12
Zeros                          |        43.12 | 253.87 |         3402.34
Constant_Pi                    |        45.34 | 241.52 |         3236.89
Checkerboard_Pattern           |        46.78 | 234.12 |         3138.45
```

## Advanced Usage

### Custom Matrix Sizes

Test multiple sizes to see if data dependency varies with problem size:

```bash
for size in 2048 4096 8192 16384; do
    ./benchmark_data_distributions.sh \
        ./gemm_benchmark \
        $size \
        50 \
        100 \
        ./results/size_${size}
done
```

### Focus on Specific Distributions

Compare just the extremes:

```bash
# Best case (constant)
./gemm_benchmark --m 8192 --n 8192 --k 8192 --init 2 --repeat 500

# Worst case (checkerboard)
./gemm_benchmark --m 8192 --n 8192 --k 8192 --init 5 --repeat 500

# Realistic (random)
./gemm_benchmark --m 8192 --n 8192 --k 8192 --init 0 --repeat 500
```

## Troubleshooting

### Script Not Finding Binary

```bash
# Explicitly specify the full path
./benchmark_data_distributions.sh \
    /full/path/to/gemm_benchmark \
    8192
```

### Parsing Errors

If the analysis script fails to parse results, check that your benchmark outputs JSON in this format:

```json
{
  "latency(ms)": 45.23,
  "tflops(TFlops)": 242.18,
  "bandwidth(GB/s)": 3245.67
}
```

### Missing Performance Metrics

Ensure `--json_output true` is set in your benchmark command.

## Files

- `benchmark_data_distributions.sh` - Orchestration script
- `analyze_distribution_benchmark.py` - Analysis script
- `gemm_profiler.hpp` - Contains distribution implementations
- `gemm_benchmark_single.cpp` - Benchmark entry point

## Contributing New Distributions

To add a new distribution:

1. Edit [gemm_profiler.hpp](tile_engine/ops/gemm/gemm_universal/gemm_profiler.hpp)
2. Add new `else if(setting_.init_method_ == X)` case
3. Update documentation in [gemm_benchmark_single.cpp](tile_engine/ops/gemm/gemm_universal/gemm_benchmark_single.cpp)
4. Add to `DISTRIBUTIONS` array in `benchmark_data_distributions.sh`
