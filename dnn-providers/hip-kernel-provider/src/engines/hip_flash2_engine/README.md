# HipFlash2Engine -- Flash-Attention 2 FP16 SDPA Engine

Flash-Attention 2 V7 implemented as a hipDNN `IEngine` plugin for FP16 SDPA on
gfx942 (MI300X/MI325X). gfx950 (MI355X) is not supported in this version.

## Performance

Measured on real hardware, FP16, seq=4096 causal D=128:

| GPU | Config | TFLOPS | vs unfused |
|-----|--------|:---:|:---:|
| MI300X | MHA seq=4096 causal D=128 | 71.27 | **+8.1x** |
| MI325X | MHA seq=4096 causal D=128 | **78.98** | **+8.1x** |
| MI325X | GQA4 seq=4096 causal D=128 | 78.16 | **+8.1x** |
| MI325X | MHA seq=2048 causal D=64 | 87.85 | **+7.1x** |

Correctness: 9/9 shapes PASS, MaxErr < 0.002 vs CPU FP32 reference (gfx942).

> **Stale.** Measured against the previous kernel binary, which has since been regenerated
> with a newer toolchain and is substantially faster on the D=128 path (D=64 unchanged).
> Re-measure before release.

## Build

Enable with the CMake option (off by default):

```bash
cmake -DENABLE_HIP_FLASH2_ENGINE=OFF ..   # default -- disabled
cmake -DENABLE_HIP_FLASH2_ENGINE=ON  ..   # enable for development
```

Requires rocWMMA (rocm-libraries component).

## Rebuilding the kernel binary

`kernels/hip_flash2_fwd_gfx942.co` is a pre-compiled artifact committed to the tree; the
CMake build copies and installs it but does not regenerate it. Rebuild it explicitly after
any change to `HipFlash2FwdPlan.hip`, or the shipped binary will silently diverge from the
source.

```bash
cd dnn-providers/hip-kernel-provider/src/engines/hip_flash2_engine
hipcc --genco -O3 -std=c++17 --offload-arch=gfx942 \
      -I"${ROCWMMA_INCLUDE}" \
      HipFlash2FwdPlan.hip -o kernels/hip_flash2_fwd_gfx942.co
```

`ROCWMMA_INCLUDE` must point at a rocWMMA 2.2.x `include` directory. rocWMMA is not part of
a base ROCm install, so `-I/opt/rocm/include` alone will not compile.

**The compiler version matters.** The same source produces materially different code across
ROCm releases -- roughly a 40% throughput spread on the D=128 path in our testing, from
register allocation alone. Re-benchmark after changing toolchain, and don't assume that newer,
or that fewer registers, is faster; neither held.

The committed binary was built with ROCm **10.0.0** (HIP 7.15.26302, AMD clang 23.0.0git
`8f497e099`) and rocWMMA **2.2.0**, yielding 340 VGPRs for `d128` and 212 for `d64`.

### Validating a rebuild

A regenerated binary must keep the host-side launch ABI unchanged -- `kernarg_segment_size`
112, `max_flat_workgroup_size` 64, LDS 32768 (`d128`) / 16384 (`d64`):

```bash
llvm-readelf --notes kernels/hip_flash2_fwd_gfx942.co | \
    grep -E 'kernarg_segment_size|max_flat_workgroup_size|group_segment_fixed_size'
```

Then run the integration tests, which validate the output against a CPU FP32 reference
(`CpuFpReferenceValidation`) at tolerance `1e-2`:

```bash
HIP_FLASH2_KERNEL_DIR=<directory containing the .co> \
    ./hip_kernel_provider_integration_tests --gtest_filter='*HipFlash2*'
```

Two things to know when running these:

- `HIP_FLASH2_KERNEL_DIR` is required for a build-tree run. The compiled-in default points at
  the install prefix, and unlike the ASM engine these tests do not set it through CTest
  `ENVIRONMENT`, so without it every case fails in `hipModuleLoad`.
- The suite is registered with CTest under the target name
  (`hip_kernel_provider_integration_tests`), not per test case -- `ctest -R HipFlash2` matches
  nothing.

The tighter `MaxErr < 0.002` bound quoted under *Performance* is not enforced by any test in
the tree; it has only been checked manually against a CPU FP32 reference.

## Known Issues

- **gfx950 (CDNA4/MI355X)**: The softmax reduction in the V7 kernel diverges to
  inf on gfx950 due to MFMA fragment lane-to-row mapping differences between
  CDNA3 and CDNA4. Under investigation. The engine's `isApplicable()` currently
  enables gfx950 for the fix to be validated; set to gfx942-only until resolved. **DONE: gfx950 disabled in isApplicable() as of this commit.**

## Engine Registration

Registered via `HIPDNN_REGISTER_ENGINE(HIP_FLASH2_ENGINE)` in EngineNames.hpp.
Engine ID is derived from the name via FNV-1a hash (same mechanism as other engines).
