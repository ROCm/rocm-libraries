# MIOpenDriver Smoke Test — hipDNN Shim Builds

Date: 2026-05-27
Branch: `users/nhanna/miopen-hipdnn-shim-investigation-1`
GPU: AMD Instinct MI300X (gfx942), ROCm 7.13

## Builds Under Test

The two builds differ in a single CMake flag:

| Build | `MIOPEN_ENABLE_HIPDNN_WRAPPER` |
| --- | --- |
| `build-flagon` | `ON` |
| `build-flagoff` | `OFF` |

`MIOpenDriver` was built in both trees via `make -j32 MIOpenDriver`; both linked successfully.

## Linkage Verification

`ldd` confirms the flag produces the expected library split:

- **flagon** — `MIOpenDriver` links against both `libMIOpen.so.1` and `libMIOpen_private.so.1` (private interface split out for the hipDNN wrapper).
- **flagoff** — `MIOpenDriver` links against `libMIOpen.so.1` only (no private library produced).

No `hipdnn*` symbols are exported from `libMIOpen.so` in either build, which is expected — the shim lives in a separate target.

## Smoke Tests

The same four invocations were run against each build. Verification was enabled (`-V 1`) in all cases.

### 1. Forward + Backward Convolution (verify only)
```
MIOpenDriver conv -n 1 -c 3 -H 32 -W 32 -k 16 -y 3 -x 3 -p 1 -q 1 -V 1
```

| Build | Forward | Bwd Data | Bwd Weights |
| --- | --- | --- | --- |
| flagon  | OK (3.08e-08) | OK (4.46e-08) | OK (7.87e-08) |
| flagoff | OK (3.08e-08) | OK (4.46e-08) | OK (7.91e-08) |

### 2. Convolution with Timing (`-t 1`)
```
MIOpenDriver conv -n 1 -c 3 -H 32 -W 32 -k 16 -y 3 -x 3 -p 1 -q 1 -V 1 -t 1
```

Same solver selections on both builds:
- Forward: solution 84 / `ConvBinWinogradRxSf2x3g1`
- Bwd Data: solution 84 / `ConvBinWinogradRxSf2x3g1`
- Bwd Weights: solution 110 / `ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC`

| Stage | flagon (ms) | flagoff (ms) |
| --- | --- | --- |
| Fwd | 0.01046 | 0.01052 |
| BwdD | 0.01259 | 0.01204 |
| BwdW | 0.02244 | 0.02333 |

Timing variance is within noise; all stages verified OK on both builds.

### 3. GEMM
```
MIOpenDriver gemm -m 64 -n 64 -k 64 -V 1
```
Both builds: `Forward GEMM Verifies on CPU and GPU (err=0.000000)`.

### 4. Activation
```
MIOpenDriver activ -n 1 -c 3 -H 32 -W 32 -V 1
```
Both builds: forward and backward activation verify on CPU and GPU.

## Result

Both builds pass every smoke test with identical correctness results. The
`MIOPEN_ENABLE_HIPDNN_WRAPPER=ON` build adds the private library split (visible
in `ldd`) without altering MIOpen driver behavior or solver selection on the
tested workloads — i.e., enabling the hipDNN shim has not regressed the
underlying MIOpen public API.
