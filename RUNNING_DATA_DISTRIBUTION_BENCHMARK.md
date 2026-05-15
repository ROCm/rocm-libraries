# Running the Data Distribution Benchmark for 8192x8192x8192 GEMM

## Current Status

✅ **Code Changes Complete**: Added π and checkerboard distributions (init methods 4 and 5)
✅ **Benchmark Scripts Ready**: Automation scripts created
⚠️ **Build Required**: Need to build for gfx950 architecture

## What We've Done

### 1. Added New Data Distributions

Modified files:
- `tile_engine/ops/gemm/gemm_universal/gemm_profiler.hpp` - Added init methods 4 and 5
- `tile_engine/ops/gemm/gemm_universal/gemm_benchmark_single.cpp` - Updated documentation

New distributions available:
- **init=4**: Constant π (3.14159...)
- **init=5**: Checkerboard pattern (+1, -1, +1, -1...)

### 2. Created Automation Scripts

- `benchmark_data_distributions.sh` - Runs all 6 distributions automatically
- `analyze_distribution_benchmark.py` - Statistical analysis and reporting
- `DATA_DISTRIBUTION_BENCHMARK.md` - Complete user guide

## Current Build Issue

The existing binaries in `build/bin/` were compiled for **gfx1030**, but you're running on **gfx950**.
Running these binaries causes: `Memory access fault by GPU node-2`

The build system has cache issues when trying to reconfigure for gfx950.

## How to Fix and Run

### Step 1: Clean Build for gfx950

```bash
cd /workspace/rocm-libraries/projects/composablekernel

# Remove problematic cache
rm -rf build/_deps/gtest-*
# OR create fresh build directory
rm -rf build && mkdir build

# Configure for gfx950
cd build
cmake \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -D CMAKE_BUILD_TYPE=Release \
  -D GPU_TARGETS="gfx950" \
  -G Ninja \
  ..
```

### Step 2: Build a GEMM Benchmark

Pick a kernel configuration (example: 128x128x64 tile):

```bash
cd build
ninja benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_128x128x64_1x4x1_16x16x16 -j64
```

This will compile with the new data distribution code (π and checkerboard).

### Step 3: Run Single Distribution Test

Test the benchmark binary with one distribution:

```bash
cd /workspace/rocm-libraries/projects/composablekernel/build

./bin/benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_128x128x64_1x4x1_16x16x16 \
  -m=8192 \
  -n=8192 \
  -k=8192 \
  -init=0 \
  -warmup=50 \
  -repeat=100 \
  -verify=0 \
  -json_output=true
```

Change `-init=` value to test different distributions:
- `0` = Uniform random
- `1` = Monotonic sequence
- `2` = Constant 1.0
- `3` = Zeros
- `4` = Constant π (NEW!)
- `5` = Checkerboard (NEW!)

### Step 4: Run Full Benchmark Suite

Use the automation script to test all distributions:

```bash
cd /workspace/rocm-libraries

./benchmark_data_distributions.sh \
  ./projects/composablekernel/build/bin/benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_128x128x64_1x4x1_16x16x16 \
  8192 \
  50 \
  100 \
  ./benchmark_results
```

This will:
- Run all 6 data distributions
- Save results to JSON and CSV
- Generate performance comparison

### Step 5: Analyze Results

```bash
python3 analyze_distribution_benchmark.py \
  benchmark_results/distribution_benchmark_8192_*.csv \
  --output analysis_report.md
```

The analysis will tell you:
- Performance variation across distributions
- Whether the kernel is data-dependent
- Best and worst performing distributions

## Expected Output

After running all distributions, you'll see output like:

```
Distribution                    | Latency (ms) | TFlops | Bandwidth (GB/s)
Uniform_Random_[-1,1]          |        45.23 | 242.18 |         3245.67
Monotonic_Sequence             |        44.89 | 243.91 |         3268.45
Constant_1.0                   |        45.01 | 243.25 |         3257.12
Zeros                          |        43.12 | 253.87 |         3402.34
Constant_Pi                    |        45.34 | 241.52 |         3236.89
Checkerboard_Pattern           |        46.78 | 234.12 |         3138.45
```

If performance variation is < 5%, the kernel is **NOT data-dependent**.
If variation is > 10%, the kernel **IS data-dependent**.

## Troubleshooting

### "Memory access fault"
- Binary was built for wrong GPU architecture
- Rebuild for gfx950

### "CMake error: generator mismatch"
- Clean cmake cache: `rm -rf CMakeCache.txt CMakeFiles _deps`
- Reconfigure with `-G Ninja`

### "Skipping Tile Engine GEMM Universal"
- GPU_TARGETS not set correctly
- Must include gfx950: `-D GPU_TARGETS="gfx950"`

### Binary not found
- Check correct path: `ls build/bin/benchmark_gemm_universal*`
- May need to build the specific kernel configuration first

## Files Created

```
/workspace/rocm-libraries/
├── benchmark_data_distributions.sh     # Main benchmark script
├── analyze_distribution_benchmark.py    # Analysis tool
└── DATA_DISTRIBUTION_BENCHMARK.md       # User guide

projects/composablekernel/
├── tile_engine/ops/gemm/gemm_universal/
│   ├── gemm_profiler.hpp               # Modified: added init 4,5
│   └── gemm_benchmark_single.cpp       # Modified: updated help text
```

## Next Steps

Once you fix the build:

1. Build one benchmark kernel for gfx950
2. Run the automated benchmark script
3. Analyze results to see if performance depends on data distribution
4. Share findings!

The code is ready - just needs a clean build for gfx950.
