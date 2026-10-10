# Native gfx950 block-scaled GEMM

The spec-driven Python and native C++ builders implement densely packed RCR
GEMM: A[M,K], B[N,K], A_scale[M,K/32], B_scale[K/32,N] and C[M,N]. M/N are
positive multiples of 32 and K is a positive multiple of 64. All tensor indices
fit signed i32. M/N/K scalar parameters mirror the specialization and launch
grid; callers must supply the spec's dimensions. Each wave64 owns one 32x32
output tile, with grid `(N/32, M/32, 1)` and block `(64, 1, 1)`.

A/B may independently use FP8 E4M3, BF8 E5M2, FP6 E2M3, BF6 E3M2 or FP4 E2M1.
Scales are raw E8M0 bytes, one K32 group per half-wave, with independent A/B
VGPR sources. The kernel uses FP32 accumulation and BF16 or FP16 output. Base
matrix/output pointers must meet their declared 16-byte alignment; scale pointers
need byte alignment. All five buffers are disjoint (`noalias`).

Use `build_block_scaled_gemm(spec, arch="gfx950")`, or the
`dispatch.gemm` APIs `BlockScaledGemmRequest`, `dispatch_block_scaled_gemm`,
`dispatch_block_scaled_gemm_all`, `registered_block_scaled_gemm_combos` and
`block_scaled_gemm_sweep_space`. Selection supports the builder's shape domain;
the example binding uses bounded K64/K128/K256 fixtures and rejects other K
values when creating those fixtures. It is a correctness harness, not an
arbitrary-input quantizer or tuned production policy.

The native ABI is `rocke_gfx950_block_scaled_gemm_spec_t` and
`rocke_build_gfx950_block_scaled_gemm_new`. The pybind engine exposes native
spec-to-IR, LLVM lowering and verification. Full modules match Python byte for
byte. LLVM 20/22 and other targets reject this atom family.

[Example commands, storage, ISA references and numerical limits](../../python/rocke/examples/gfx950/gemm/README.md)
explain the four focused entrypoints and their mandatory non-unity scale cases.
