# ComposableKernel Development Guidelines

## Python Script Execution

When running Python scripts that produce long-running output (benchmarks, training, etc.), ALWAYS use the `-u` flag to disable output buffering. This allows real-time monitoring without killing and restarting processes.

### Correct Usage:
```bash
# Background execution with unbuffered output
nohup python3 -u script.py args... > output.log 2>&1 &

# Foreground execution with unbuffered output
python3 -u script.py args...
```

### Why This Matters:
- Without `-u`, Python buffers stdout/stderr internally
- Buffered output may not appear in logs until the buffer fills or process exits
- This makes monitoring progress difficult and can lead to unnecessary process kills
- The `-u` flag forces unbuffered output, showing results immediately

### Examples:

**Benchmarking:**
```bash
python3 -u grouped_conv_full_benchmark.py configs/forward_bf16_3d.json \
    --arch gfx950 --problems validation_holdout \
    --csv results.csv --workers 8 > benchmark.log 2>&1 &
```

**Training:**
```bash
python3 -u train.py --data_dir data/ --output_dir models/ > training.log 2>&1 &
```

## GPU Testing

- We have **1 GPU instance** - only test one problem at a time
- Never run multiple benchmark processes simultaneously (causes GPU contention/crashes)
- Always check for running processes before starting new benchmarks:
  ```bash
  ps aux | grep -E 'grouped_conv|benchmark' | grep -v grep
  ```

## Working Directory

Always run grouped convolution scripts from the correct directory:
```bash
cd /workspace/rocm-libraries/projects/composablekernel/tile_engine/ops/grouped_conv
```

## Kernel Compilation

- Kernels are compiled once and cached in `/tmp/dispatcher/` directories
- Use `--workers 8` for parallel compilation (first run only)
- Subsequent runs reuse compiled `.so` files (no recompilation needed)
