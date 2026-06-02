# FP64 Emulation Accuracy Test

Standalone benchmark that evaluates the accuracy and runtime of the
`fp64EmulatedGemm()` Ozaki Scheme II (OS II-accu) implementation in hipBLASLt
against:

1. A **double-double (DD) reference GEMM** computed on GPU (~106-bit mantissa),
   which serves as the ground truth.
2. The **native hipBLASLt DGEMM** (standard FP64, `HIPBLAS_COMPUTE_64F`).

The test closely follows the GEMMul8 benchmark methodology
(<https://github.com/RIKEN-RCCS/GEMMul8>):

- Matrix elements are drawn from the GEMMul8 distribution:  
  `A[i] = (U(0,1) - 0.5) * exp(N(0,1) * phi)` where phi controls the dynamic
  range.  Larger phi → more cancellation → harder for GEMM.
- The same phi values as GEMMul8 are swept: `{-1, 0, 0.5, 1, 2, 4}`.
- Error metric: componentwise relative error vs the DD reference,
  `|err[i]| = |(D[i] - D_exact[i]) / D_exact[i]|`, reported as max and median.

## Build

Prerequisites: ROCm (HIP + hipBLASLt) and the hipblaslt **source** tree
(needed for the internal `fp64_emulation.hpp` / `rocblaslt.h` headers).

```bash
cd projects/fp64_emul_test
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    [-DHIPBLASLT_SRC_DIR=/path/to/rocm-libraries/projects/hipblaslt]
cmake --build build -j$(nproc)
```

If this project lives at `projects/fp64_emul_test/` inside the rocm-libraries
workspace, `HIPBLASLT_SRC_DIR` defaults to `../hipblaslt` and the `-D` flag is
not needed.

## Run

```
./build/fp64_emul_accuracy [options]

Options:
  -n N          Square matrix size M=N=K (default: 512)
  --num-runs R  Number of timed iterations per configuration (default: 30)
                Warmup runs = same count as num-runs.
  --min-s S     Minimum num_moduli to test (default: 2)
  --max-s S     Maximum num_moduli to test (default: 14)
  --phi-list    Comma-separated phi values (default: -1,0,0.5,1,2,4)
  -h, --help    Print this help
```

### Example

```bash
# Quick test (N=512, all phi, all moduli counts)
./build/fp64_emul_accuracy

# Larger problem
./build/fp64_emul_accuracy -n 1024 --num-runs 10
```

## Output

Results are printed as CSV to stdout (suitable for piping to a file):

```
phi,N,algo,err_max,err_med,ms_per_run
-1.000,512,DGEMM,5.12e-16,2.30e-16,0.421
-1.000,512,OS2-accu-s2,3.47e-04,1.20e-05,12.3
-1.000,512,OS2-accu-s3,8.90e-07,3.10e-08,18.1
...
-1.000,512,OS2-accu-s14,4.44e-16,2.22e-16,45.6
 0.000,512,DGEMM,...
...
 4.000,512,OS2-accu-s14,...
```

Columns:
- `phi` — matrix difficulty parameter (see GEMMul8)
- `N` — matrix dimension (square: M=N=K)
- `algo` — `DGEMM` (native) or `OS2-accu-s<s>` (emulation with `s` moduli)
- `err_max` — maximum componentwise relative error vs DD reference
- `err_med` — median componentwise relative error vs DD reference
- `ms_per_run` — wall time per GEMM call (milliseconds)
