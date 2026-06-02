# FP64 Emulation Accuracy Test

Standalone benchmark that evaluates the accuracy and runtime of hipBLASLt's
FP64 GEMM emulation (Ozaki Scheme II, OS II-accu) accessed entirely through the
**public hipBLASLt API** — no internal headers are included.

The benchmark compares:

1. A **double-double (DD) reference GEMM** computed on GPU (~106-bit mantissa),
   which serves as the ground truth.
2. The **native hipBLASLt DGEMM** (`HIPBLAS_COMPUTE_64F`, emulation explicitly
   disabled via `hipblasLtSetEmulationEnabled(handle, false)`).
3. The **emulated DGEMM** with a fixed number of INT8 moduli `s = min_s..max_s`
   (configured via `hipblasLtSetFixedPointEmulationMaxMantissaBitCount`).
4. The **adaptive emulated DGEMM** (library-default precision, configured via
   `hipblasLtSetFixedPointEmulationMantissaControl(DYNAMIC)`).

The test closely follows the GEMMul8 benchmark methodology
(<https://github.com/RIKEN-RCCS/GEMMul8>):

- Matrix elements are drawn from the GEMMul8 distribution:  
  `A[i] = (U(0,1) - 0.5) * exp(N(0,1) * phi)` where phi controls the dynamic
  range.  Larger phi → more cancellation → harder for GEMM.
- Error metric: componentwise relative error vs the DD reference,
  `|err[i]| = |(D[i] - D_exact[i]) / D_exact[i]|`, reported as max and median.

## Prerequisites

- ROCm (HIP + hipBLASLt)
- hipBLASLt built **and installed** into `projects/hipblaslt/hipblaslt-install/`
  (run `cmake --install` from the hipBLASLt build tree so that the public headers
  and the shared library are up-to-date)

## Build

```bash
# 1. Build and install hipBLASLt (from the hipblaslt build directory)
cd projects/hipblaslt/build/release
cmake --build . --target hipblaslt -j$(nproc)
cmake --install .

# 2. Build this test
cd projects/fp64_emul_test
rm -rf build   # needed if re-configuring after header changes
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j$(nproc)
```

If your hipBLASLt build tree is not at the default location:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DHIPBLASLT_BUILD_DIR=/path/to/hipblaslt/build/release
```

## Run

```
./build/fp64_emul_accuracy [options]

Options:
  -n N           Square matrix size M=N=K (default: 512)
  --num-runs R   Number of timed iterations per configuration (default: 30)
                 Warmup runs = same count as num-runs.
  --min-s S      Minimum num_moduli to test (default: 2)
  --max-s S      Maximum num_moduli to test (default: 20)
  --phi-list P   Comma-separated phi values (default: 0.5,1,2,4)
  --no-adaptive  Skip the adaptive-s run
  -h, --help     Print this help
```

### Example

```bash
# Quick test (N=512, all phi, all moduli counts)
./build/fp64_emul_accuracy

# Larger problem
./build/fp64_emul_accuracy -n 1024 --num-runs 10

# Only sweep s=7..14, no adaptive run
./build/fp64_emul_accuracy --min-s 7 --max-s 14 --no-adaptive
```

## Output

Results are printed as CSV to stdout (suitable for piping to a file):

```
phi,N,algo,crt_bits,err_max,err_med,ms_per_run
0.5,512,DGEMM,53.0,5.12e-16,2.30e-16,0.421
0.5,512,OS2-accu-adaptive,125.4,4.44e-16,2.22e-16,38.2
0.5,512,OS2-accu-s2,16.0,3.47e-04,1.20e-05,12.3
0.5,512,OS2-accu-s3,24.0,8.90e-07,3.10e-08,18.1
...
0.5,512,OS2-accu-s20,155.4,4.44e-16,2.22e-16,58.1
1.0,512,DGEMM,...
...
```

Columns:
- `phi` — matrix difficulty parameter (see GEMMul8); larger phi = wider dynamic range
- `N` — matrix dimension (square: M=N=K)
- `algo` — `DGEMM` (native FP64), `OS2-accu-adaptive` (library-default precision),
  or `OS2-accu-s<s>` (emulation with exactly `s` moduli)
- `crt_bits` — CRT mantissa bit capacity of the chosen moduli set (53 for native FP64)
- `err_max` — maximum componentwise relative error vs DD reference
- `err_med` — median componentwise relative error vs DD reference
- `ms_per_run` — wall time per GEMM call (milliseconds)

## API usage

The benchmark demonstrates the correct public-API workflow for FP64 emulation:

```cpp
// Native DGEMM — emulation explicitly disabled
hipblasLtSetEmulationEnabled(native_handle, false);

// Emulated DGEMM — enable + configure
hipblasLtSetEmulationEnabled(emul_handle, true);
hipblasLtSetEmulationStrategy(emul_handle, HIPBLASLT_EMULATION_STRATEGY_EAGER);
hipblasLtSetEmulationSpecialValuesSupport(emul_handle, 0u);  // skip Inf/NaN check

// Fixed precision sweep (selects exactly s moduli)
hipblasLtSetFixedPointEmulationMantissaControl(emul_handle,
    HIPBLAS_EMULATION_MANTISSA_CONTROL_FIXED);
hipblasLtSetFixedPointEmulationMaxMantissaBitCount(emul_handle, maxBits);

// Adaptive (library chooses num_moduli automatically)
hipblasLtSetFixedPointEmulationMantissaControl(emul_handle,
    HIPBLAS_EMULATION_MANTISSA_CONTROL_DYNAMIC);

// All GEMM calls go through the standard hipblasLtMatmul path
hipblasLtMatmul(handle, desc, &alpha, A, layoutA, B, layoutB,
                &beta, D, layoutD, D, layoutD, algo, workspace, ws_bytes, stream);
```
