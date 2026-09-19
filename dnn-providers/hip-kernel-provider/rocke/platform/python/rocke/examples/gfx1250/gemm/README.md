# gfx1250 block-scaled GEMM examples

`mxfp8_gemm` runs homogeneous FP8 E4M3 and BF8 E5M2 separately by default.
Use `--dtype fp8` or `--dtype bf8` to run one encoding; `--dtype both` is
the explicit default. Canonical names are `fp8e4m3` and `bf8e5m2`.

```bash
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mxfp8_gemm
ROCKE_LLVM_FLAVOR=llvm23 python -m rocke.examples.gfx1250.gemm.mxfp8_gemm --dtype bf8 --matrix-path wmma_scale16
```

SCALE uses E8M0 blocks of 32 K elements; SCALE16 uses blocks of 16.
Use `--compile-route hip` for HIP compilation and `--case all` for neutral,
one-sided, combined, and isolated scale-group fixtures. The `mixed` fixture
means scale variation, not different A/B matrix formats.

The examples share argument parsing, the spec-driven builder, compiler,
launcher, and numerical verifier. Numerical tests use opt-in fixtures in
`tests/instances/conftest.py` and are independently runnable by family.

The target-independent aliases in `core/dtypes.py` normalize logical format
names. Recognition does not imply that a target supports an atom; use its
catalog to query supported operand formats and shapes. Each example defines
its accepted matrix and scale formats.

`e8m0` identifies a scale format carried in bytes and packed integer operands.
It is not a general scalar IR type or conversion API. Native scaled WMMA uses
one `wmma_scaled` catalog family. Its operation IDs encode the matrix source
and accumulator dtype, each source's scale format, and a shared K-group size; the backend
selects the LLVM intrinsic and packed operand types from that contract.
