# Grouped Convolution Benchmark Architecture

## Overview

This benchmark system collects ML training data for grouped convolution kernel selection heuristics. It mirrors FMHA's proven architecture to avoid Python ctypes library loading limitations.

## Architecture Design

### Key Principle: Subprocess Isolation

**Problem**: Python's `ctypes.CDLL()` cannot unload shared libraries. After loading 3-4 GPU kernel .so files in a single process, HIP/ROCm initialization fails with "GPU not available" errors.

**Solution**: Each kernel (or small batch of kernels) runs in a separate subprocess with a fresh Python interpreter and GPU context.

### Two-Phase Design

#### Phase 1: Compile (Parallel, Main Process)
- `setup_multiple_grouped_conv_dispatchers()` builds all kernels in parallel
- **CRITICAL**: Returns only `List[Path]` (file paths), does NOT load .so files
- Main process never initializes GPU context during compilation
- Parallelized build uses ProcessPoolExecutor for fast compilation

#### Phase 2: Benchmark (Subprocess Batching, Serial GPU)
- Each subprocess receives JSON payload with kernel paths + problem parameters
- Subprocess loads library ONLY inside its process using `GpuGroupedConvRunner`
- Subprocess outputs JSON results to stdout (flushed per-kernel)
- Parent process parses results and writes to CSV
- Batch size (default 20) balances subprocess overhead vs fault isolation

## Files

### Core Benchmark Scripts

#### `grouped_conv_full_benchmark.py`
Main benchmark orchestrator. Architecture:
```python
# Phase 1: Compile (returns Path objects only)
lib_paths = setup_multiple_grouped_conv_dispatchers(configs, ...)
# lib_paths: List[Optional[Path]] - NO .so loading in main process!

# Phase 2: Benchmark via subprocess batching
for batch_start in range(0, len(kernels), batch_size):
    # Build JSON payload with .so paths
    payload = {"items": [{"so_path": str(lib_path), "problem": {...}}, ...]}

    # Run worker subprocess
    proc = subprocess.Popen([sys.executable, "run_one_grouped_conv_kernel.py"], ...)
    stdout, _ = proc.communicate(input=payload.encode())

    # Parse JSON results (one line per kernel)
    for line in stdout.split('\n'):
        result = json.loads(line)
        # Write to CSV
```

**Usage**:
```bash
# Full benchmark: 20 kernels x 200 problems = 4,000 measurements
python3 grouped_conv_full_benchmark.py configs/forward_bf16.json \
    --arch gfx950 \
    --problems forward_training \
    --csv training_data.csv \
    --workers 8 \
    --batch-size 20

# Quick test: 2 kernels x 20 problems = 40 measurements
python3 grouped_conv_full_benchmark.py configs/forward_bf16.json \
    --arch gfx950 \
    --problems forward_training_small \
    --csv test.csv \
    --max-kernels 2 \
    --batch-size 2
```

#### `run_one_grouped_conv_kernel.py`
Worker script that runs in subprocess. Architecture mirrors FMHA's `run_one_kernel.py`:

```python
# Reads JSON from stdin
payload = json.loads(sys.stdin.buffer.read())

# For each kernel in batch
for item in payload["items"]:
    # CRITICAL: Load library ONLY inside subprocess
    runner = GpuGroupedConvRunner(lib_path=item["so_path"])
    result = runner.run(input_data, weight_data, problem)

    # Output JSON result to stdout (flushed immediately)
    print(json.dumps({"idx": i, "ok": True, "ms": ..., "tflops": ...}), flush=True)
```

**Key Design Points**:
- Each subprocess has fresh Python interpreter and GPU context
- Per-line JSON output with flush ensures parent can recover partial results if GPU fault kills subprocess
- No return value - all communication via stdout JSON

### Test Scripts

#### `test_worker_script.py`
Tests single kernel in subprocess isolation:
```bash
python3 test_worker_script.py
# ✓ KERNEL SUCCESS! Time: 0.014 ms, TFLOPS: 4.19
```

#### `test_batch_benchmark.py`
Tests full pipeline with 2 kernels x 20 problems = 40 measurements:
```bash
python3 test_batch_benchmark.py
# ✓ Batch benchmark test PASSED
# Results (40 measurements)
```

## Comparison: Old vs New Architecture

### Old (`grouped_conv_benchmark_subprocess.py`) - FLAWED
```python
# ARCHITECTURAL FLAW: Loads .so files in main process
lib_paths = setup_multiple_grouped_conv_dispatchers(configs, ...)
# lib_paths contains loaded library objects with GPU context

# Subprocess spawns with parent's GPU context already initialized
for prob in problems:
    for cfg, lib_path in built_kernels:
        subprocess.run(worker_code)  # NO benefit - GPU already loaded in parent!
```

**Why it failed**:
- Main process loaded .so files before subprocess spawned
- GPU context initialized in parent process
- Subprocess provided no isolation benefit
- Still hit 3-4 library loading limit

### New (`grouped_conv_full_benchmark.py`) - CORRECT
```python
# CORRECT: Only file paths, no .so loading in main process
lib_paths = setup_multiple_grouped_conv_dispatchers(configs, ...)
# lib_paths: List[Optional[Path]] - pure file paths, no GPU context!

# Subprocess loads library inside its own process
payload = {"items": [{"so_path": str(lib_path), ...}, ...]}
subprocess.Popen([sys.executable, "run_one_grouped_conv_kernel.py"], ...)
# Worker script does: GpuGroupedConvRunner(lib_path=so_path)
# Library loaded ONLY inside subprocess, not in parent
```

**Why it works**:
- Main process never initializes GPU context
- Each subprocess has fresh GPU driver state
- No library loading limit (each subprocess loads fresh)
- Clean isolation between kernel invocations

## Key Insights from FMHA Design

1. **Build returns paths only**: `setup_multiple_fmha_dispatchers` returns `FmhaSetupResult` with `library_path: str` attribute, NOT loaded library objects

2. **Serializable kernel configs**: Convert kernel configs to dictionaries with `so_path` string for JSON serialization

3. **Batch mode for efficiency**: Worker script supports both single and batch mode:
   - Single: `{"so_path": "...", "problem": {...}}`
   - Batch: `{"items": [{"so_path": "...", "problem": {...}}, ...]}`

4. **Per-line JSON output**: Flush after each result allows parent to recover partial results if GPU fault kills subprocess

5. **Environment variables for paths**: Pass Python paths via environment variables to avoid hardcoding

## Performance Characteristics

### Test Results (2 kernels x 20 problems = 40 measurements)
- Build time: 21s (parallel compilation)
- Benchmark time: 9s (subprocess batching with batch_size=2)
- Total time: 30s
- Success rate: 100% (40/40 measurements)
- No GPU faults or crashes

### Scaling to Full Dataset (20 kernels x 200 problems = 4,000 measurements)
- Estimated build time: ~2-3 minutes (parallel with 8 workers)
- Estimated benchmark time: ~15-20 minutes (batch_size=20, ~200 subprocess invocations)
- Total estimated time: ~20-25 minutes
- Expected success rate: >99% (isolated subprocess prevents cascading failures)

## CSV Output Format

```csv
kernel,problem_idx,N,C,K,G,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w,latency_ms,tflops,non_zero
grouped_conv_forward_bf16_2d_16x64x64_compv3,0,1,64,64,1,56,56,3,3,1,1,1,1,0.0084,27.53,200704
grouped_conv_forward_bf16_2d_16x64x64_compv4,0,1,64,64,1,56,56,3,3,1,1,1,1,0.0072,32.29,200704
...
```

**Fields**:
- `kernel`: Kernel configuration name
- `problem_idx`: Problem index (0-199 for full dataset)
- `N, C, K, G, Hi, Wi, Y, X, stride_h, stride_w, pad_h, pad_w`: Problem dimensions
- `latency_ms`: Kernel execution time in milliseconds
- `tflops`: Compute throughput in TFLOPS
- `non_zero`: Number of non-zero outputs (sanity check)

## Running Full Data Collection

```bash
# Full training data collection (20 kernels x 200 problems = 4,000 measurements)
cd /workspace/rocm-libraries/projects/composablekernel/tile_engine/ops/grouped_conv

python3 grouped_conv_full_benchmark.py \
    configs/forward_bf16.json \
    --arch gfx950 \
    --problems forward_training \
    --csv training_data_forward_bf16_20.csv \
    --workers 8 \
    --batch-size 20 \
    --kernel-timeout 30

# Expected output:
# - Build time: ~3 minutes
# - Benchmark time: ~20 minutes
# - Total: ~25 minutes
# - 4,000 measurements in CSV
```

## Troubleshooting

### GPU Not Available After Crashes
```bash
# Reset GPU state
python3 -c "import ctypes; hip = ctypes.CDLL('libamdhip64.so'); hip.hipDeviceReset()"

# Kill all Python processes
pkill -9 python
```

### Subprocess Timeouts
- Increase `--kernel-timeout` (default 30s per kernel)
- Reduce `--batch-size` for faster fault isolation (at cost of more subprocess overhead)

### Build Failures
- Check `dispatcher/build/examples/` for .so files
- Verify GPU architecture: `rocminfo | grep gfx`
- Check compiler: `hipcc --version`

## References

- FMHA benchmark: `tile_engine/ops/fmha/fmha_full_benchmark.py`
- FMHA worker: `tile_engine/ops/fmha/run_one_kernel.py`
- Grouped conv utils: `dispatcher/python/grouped_conv_utils.py`
- Instance builder: `tile_engine/ops/grouped_conv/grouped_conv_instance_builder.py`
