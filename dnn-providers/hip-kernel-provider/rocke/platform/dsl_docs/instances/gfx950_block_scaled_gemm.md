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

Use `build_block_scaled_gemm(spec, arch="gfx950")` directly. The four focused
examples construct a spec and invoke the shared verifier to prepare inputs,
compile, launch and compare with an independent decoded reference. The bounded
fixtures accept K64/K128/K256. These are untuned correctness examples
demonstrating native scaled MFMA support. They make no performance claim.

The native ABI is `rocke_gfx950_block_scaled_gemm_spec_t` and
`rocke_build_gfx950_block_scaled_gemm_new`. The pybind engine exposes native
spec-to-IR, LLVM lowering and verification. Full modules match Python byte for
byte. LLVM 20, 22 and 23 use the same nine-argument overloaded intrinsic ABI.
Other targets reject this atom family.

[Example commands, storage, ISA references and numerical limits](../../python/rocke/examples/gfx950/gemm/README.md)
explain the four focused entrypoints and their mandatory non-unity scale cases.

## Relationship to Existing gfx950 MFMA Support

This builder consumes the 32x32x64 `mfma_scaled` target-catalog family. It
selects the complete shape/dtype/scale contract with `op_for_shape`, loads and
packs the matrices and scales, then emits `IRBuilder.mma(atom, a, b, c, sa, sb)`.
The Python/C++ builder modules are siblings of the gfx1250 block-scaled GEMM
modules, and both instruction shapes share the existing MFMA lowering files.

The existing `MfmaAtom.emit_scaled()` helper always calls the 16x16x128
scaled wrapper. Separately, `MfmaAtom.fp8_16x16x128().emit()` is used by the
gfx950 FP8 fused-MoE dispatcher, with constant in-instruction scales and
FP32 block-scale factors applied outside the MFMA. This 32x32x64 builder
passes independent E8M0 scales to the instruction and currently runs directly
from a spec, without a GEMM dispatcher registration. See the
[API and consumer comparison](../reference/mma_metadata.md#gfx950-scaled-mfma-paths).
