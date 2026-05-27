# MIOpen hipDNN Shim — Runtime Wrapper Logging

Date: 2026-05-27
Branch: `users/nhanna/miopen-hipdnn-shim-investigation-1`
Hardware: AMD Instinct MI300X (gfx942), ROCm 7.13, MIOpen 3.5.2
Builds:
- `build-flagoff` — `MIOPEN_ENABLE_HIPDNN_WRAPPER=OFF` (single `libMIOpen.so.1`)
- `build-flagon`  — `MIOPEN_ENABLE_HIPDNN_WRAPPER=ON`  (`libMIOpen.so.1` wrapper + `libMIOpen_private.so.1` impl)

## What was changed

`src/private/wrapper.cpp` was instrumented so each of the 263 forwarding
stubs announces itself on `stderr` before delegating to its `*_impl`
counterpart. One `#include <cstdio>` was added and one line was inserted
as the first statement of every stub:

```cpp
extern "C" miopenStatus_t miopenCreate(miopenHandle_t* handle)
{
    fprintf(stderr, "[MIOPEN_HIPDNN_WRAPPER] miopenCreate\n");
    return miopenCreate_impl(handle);
}
```

Only the `MIOpen` CMake target in `build-flagon` was rebuilt; `build-flagoff`
was not touched (it does not compile `wrapper.cpp`). The original wrapper
file was backed up to `/tmp/wrapper.cpp.bak` for revert.

## Test command

A single MIOpenDriver invocation was issued against each build:

```
MIOpenDriver convfp16 -n 1 -c 1 -H 8 -W 8 -k 1 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 \
             -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -i 1 -V 0
```

`stdout` and `stderr` were captured separately so the wrapper trace is
trivially isolated from normal driver output. Both runs exited `0` and
selected the same solver (`85/ConvDirectNaiveConvFwd`).

Logs: `perf-results/wrapper-proof/{flagon,flagoff}.instr.{stdout,stderr}`.

## Full output — flagon

`flagon.instr.stdout` (7 lines — same shape as flagoff):

```
MIOpenDriver convfp16 -n 1 -c 1 -H 8 -W 8 -k 1 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -i 1 -V 0
PRNG seed: 12345678
Timestamp: 2026-05-27 14:49:25 UTC; Host Name: 684148a436c9; Operating System: Linux 5.15.0-173-generic; ROCm: 7.13.61040; MIOpen Driver: 3.5.2; CPU Vendor: Intel; CPU Model: 2 x Intel(R) Xeon(R) Platinum 8480C; RAM Size: 2015 GB; GPU Model: 8 x AMD Instinct MI300X; AMDGPU Driver: 6.16.13
MIOpen Forward Conv. Algorithm: 1, Solution: 85/ConvDirectNaiveConvFwd
GPU Kernel Time Forward Conv. Elapsed: 0.006776 ms (average)
stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs
stats: fwd-conv3x3u1, 1, 1, 8, 8, 3, 3, 1, 1152, 146, 128, 0, 0, 0.006776
```

`flagon.instr.stderr` (45 lines — every line is wrapper proof):

```
[MIOPEN_HIPDNN_WRAPPER] miopenCreateWithStream
[MIOPEN_HIPDNN_WRAPPER] miopenGetStream
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateConvolutionDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenCreateConvolutionDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenEnableProfiling
[MIOPEN_HIPDNN_WRAPPER] miopenSetTensorDescriptorV2
[MIOPEN_HIPDNN_WRAPPER] miopenSetTensorDescriptorV2
[MIOPEN_HIPDNN_WRAPPER] miopenInitConvolutionNdDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenSetConvolutionGroupCount
[MIOPEN_HIPDNN_WRAPPER] miopenSetConvolutionAttribute
[MIOPEN_HIPDNN_WRAPPER] miopenGetConvolutionNdForwardOutputDim
[MIOPEN_HIPDNN_WRAPPER] miopenSetTensorDescriptorV2
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetTensorDescriptorSize
[MIOPEN_HIPDNN_WRAPPER] miopenConvolutionForwardGetWorkSpaceSize
[MIOPEN_HIPDNN_WRAPPER] miopenGetVersion
[MIOPEN_HIPDNN_WRAPPER] miopenFindConvolutionForwardAlgorithm
[MIOPEN_HIPDNN_WRAPPER] miopenConvolutionForwardGetSolution
[MIOPEN_HIPDNN_WRAPPER] miopenConvolutionForward
[MIOPEN_HIPDNN_WRAPPER] miopenGetKernelTime
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyConvolutionDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyTensorDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroyConvolutionDescriptor
[MIOPEN_HIPDNN_WRAPPER] miopenDestroy
```

## Full output — flagoff

`flagoff.instr.stdout` (7 lines — same shape, same result):

```
MIOpenDriver convfp16 -n 1 -c 1 -H 8 -W 8 -k 1 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 -i 1 -V 0
PRNG seed: 12345678
Timestamp: 2026-05-27 14:49:27 UTC; Host Name: 684148a436c9; Operating System: Linux 5.15.0-173-generic; ROCm: 7.13.61040; MIOpen Driver: 3.5.2; CPU Vendor: Intel; CPU Model: 2 x Intel(R) Xeon(R) Platinum 8480C; RAM Size: 2015 GB; GPU Model: 8 x AMD Instinct MI300X; AMDGPU Driver: 6.16.13
MIOpen Forward Conv. Algorithm: 1, Solution: 85/ConvDirectNaiveConvFwd
GPU Kernel Time Forward Conv. Elapsed: 0.006936 ms (average)
stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs
stats: fwd-conv3x3u1, 1, 1, 8, 8, 3, 3, 1, 1152, 146, 128, 0, 0, 0.006936
```

`flagoff.instr.stderr` (0 lines):

```
[empty]
```

## The lines that show the difference

The full contrast is the entire stderr stream — 45 wrapper hits on flagon
vs. 0 on flagoff. The structurally important hits inside the flagon trace
are the ones that map to the actual conv operation rather than per-tensor
bookkeeping:

| Line in `flagon.instr.stderr` | What it proves |
| --- | --- |
| `[MIOPEN_HIPDNN_WRAPPER] miopenCreateWithStream`               | Very first MIOpen API call traverses the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenCreateConvolutionDescriptor`    | Conv descriptor setup goes through the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenFindConvolutionForwardAlgorithm`| Solver search routed through the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenConvolutionForward`             | The actual GPU enqueue routed through the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenGetKernelTime`                  | Timing readback routed through the wrapper |
| `[MIOPEN_HIPDNN_WRAPPER] miopenDestroy`                        | Final teardown traverses the wrapper |

The flagoff side prints none of these because its `libMIOpen.so.1`
contains the real functions directly — there is no stub layer to print
from. Stdout is byte-shape-identical between the two runs; the entire
behavioral delta is on stderr, and it is exactly what the wrapper design
predicts.

## Reverting the instrumentation

```bash
cp /tmp/wrapper.cpp.bak src/private/wrapper.cpp
cmake --build build-flagon --target MIOpen -- -j$(nproc)
```
