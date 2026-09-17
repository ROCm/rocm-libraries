# gfx1250 block-scaled GEMM examples

Each example selects one homogeneous input family and reuses the shared
spec-driven builder, compiler, launcher, and numerical verifier.

| Module | Matrix inputs | Block scales |
| --- | --- | --- |
| `mxfp8_gemm` | FP8 E4M3 on A and B | E8M0 |
| `mxfp6_gemm` | FP6 E2M3 or E3M2 on A and B | E8M0 |

Run from an environment with rocKE installed and a visible gfx1250 device:

```bash
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mxfp8_gemm
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mxfp6_gemm --dtype fp6
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mxfp6_gemm --dtype bf6
```

Defaults use SCALE with K=32 scale groups. Add `--matrix-path wmma_scale16`
for K=16 groups, `--compile-route hip` for HIP compilation, or `--case all`
for neutral, one-sided, combined, and isolated scale-group fixtures.
The `mixed` fixture name means combined A/B scale variation, not mixed dtypes.
See [FP6.md](FP6.md) for the packed FP6 input contract. FP6 does not require FP4.
