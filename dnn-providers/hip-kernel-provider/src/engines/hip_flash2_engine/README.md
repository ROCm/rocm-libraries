# HipFlash2Engine — Flash-Attention 2 FP16 SDPA Engine

Flash-Attention 2 V7 implemented as a hipDNN `IEngine` plugin for FP16 SDPA on
gfx942 (MI300X/MI325X). gfx950 (MI355X) support is in progress — see Known Issues.

## Performance

Measured on real hardware, FP16, seq=4096 causal D=128:

| GPU | Config | TFLOPS | vs unfused |
|-----|--------|:---:|:---:|
| MI300X | MHA seq=4096 causal D=128 | 71.27 | **+8.1×** |
| MI325X | MHA seq=4096 causal D=128 | **78.98** | **+8.1×** |
| MI325X | GQA4 seq=4096 causal D=128 | 78.16 | **+8.1×** |
| MI325X | MHA seq=2048 causal D=64 | 87.85 | **+7.1×** |

Correctness: 9/9 shapes PASS, MaxErr < 0.002 vs CPU FP32 reference (gfx942).

## Build

Enable with the CMake option (off by default):

```bash
cmake -DENABLE_HIP_FLASH2_ENGINE=OFF ..   # default — disabled
cmake -DENABLE_HIP_FLASH2_ENGINE=ON  ..   # enable for development
```

Requires rocWMMA (rocm-libraries component).

## Known Issues

- **gfx950 (CDNA4/MI355X)**: The softmax reduction in the V7 kernel diverges to
  inf on gfx950 due to MFMA fragment lane-to-row mapping differences between
  CDNA3 and CDNA4. Under investigation. The engine's `isApplicable()` currently
  enables gfx950 for the fix to be validated; set to gfx942-only until resolved.

## Engine Registration

Registered via `HIPDNN_REGISTER_ENGINE(HIP_FLASH2_ENGINE)` in EngineNames.hpp.
Engine ID is derived from the name via FNV-1a hash (same mechanism as other engines).
