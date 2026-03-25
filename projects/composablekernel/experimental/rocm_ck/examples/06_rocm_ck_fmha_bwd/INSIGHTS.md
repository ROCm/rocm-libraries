# INSIGHTS.md — FMHA BWD kpack Example

Design decisions and lessons learned from mapping the CK Tile FMHA backward
kernels (OGradDotO, DqDkDv, ConvertDQ) to the kpack pattern.

## Two Args Structs Instead of a Union

The batch and group modes have different extension fields:
- **Batch**: 3 x `index_t` (batch_stride_do, batch_stride_o, batch_stride_d) = 12 bytes
- **Group**: 3 x pointer (seqstart_q_ptr, seqlen_q_ptr, cu_seqlen_q_ptr) = 24 bytes

Using a union would require runtime mode discrimination and waste space. Since mode
is a compile-time constant per variant (baked into the `.hip` file), separate structs
are cleaner: the host knows which struct to populate from `kernel.mode`, and the
device code uses the correct Kargs type via `std::conditional_t`.

## ABI Verification via static_assert

The host populates flat C structs (`FmhaBwdOGradDotOBatchArgs` /
`FmhaBwdOGradDotOGroupArgs`) and passes them by value through
`hipModuleLaunchKernel`. The device code uses `__builtin_bit_cast` to convert
to CK Tile's internal Kargs type (which uses C++ inheritance).

This works because:
1. Both are standard-layout types with the same fields in the same order
2. CK Tile's Kargs uses simple single inheritance (no vtable, no virtual)
3. We verify `sizeof(ApiArgs) == sizeof(Kargs)` and
   `alignof(ApiArgs) == alignof(Kargs)` at compile time in `dev.hpp`

The `api.hpp` file also has self-consistency asserts (`trivially_copyable`,
`standard_layout`) to catch accidental additions of non-trivial members.

## Group Mode Requires pad_seqlen_q

The CK Tile dispatcher (`fmha_instance_builder.py`) filters out group-mode
instances where `spad != "t"`. This is because group mode has variable-length
sequences within a batch, so partial tiles at sequence boundaries are inherent.
The `make_kernel` consteval validation enforces this same constraint at compile
time — attempting to create a group-mode variant with `pad_seqlen_q = false`
produces a clear error message.

## Group Mode Memory Layout

Group mode uses a different memory layout than batch mode:
- **Batch**: `[batch, nhead, seqlen_q, hdim_v]` with batch strides
- **Group**: `[total_seq, nhead, hdim_v]` where `total_seq = sum(seqlen_q_i)`

The kernel computes per-batch offsets from `seqstart_q_ptr` (cumulative sequence
lengths) rather than fixed batch strides. This means the host test must re-layout
data when testing group-mode variants against a batch-mode CPU reference.

## D Shares LSE Stride Layout

In `fmha_bwd_dot_do_o_create_kargs_and_grids()`, the D output stride arguments
use `args.nhead_stride_lsed` and `args.batch_stride_lsed` — D shares the
log-sum-exp (LSE) stride layout since both are 1D per (batch, head, seqlen_q).
Our API struct names them `nhead_stride_d` / `batch_stride_d` matching the
CK Tile Kargs field names.

## Naming Convention: BwdOGradDotO

We follow the CK Tile internal naming (`FmhaBwdOGradDotOKernel`,
`BlockFmhaBwdOGradDotO`, `TileFmhaBwdOGradDotOTraits`) rather than the legacy
dispatcher naming (`bwd_dot_do_o`). This aligns type names across our API and
the CK Tile template chain, reducing confusion when tracing the template
instantiation path.

## IGLP Pipeline Crash on clang 22.0.0git (ROCm Mainline)

**Severity: BLOCKER for pad_hdim_q=8 / pad_hdim_v=8 configs**

The `BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP` pipeline variant produces
`HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` at runtime on gfx942 when compiled
with clang 22.0.0git (ROCm mainline, commit `c849bc16`). The kernel compiles
without errors but crashes immediately on launch.

**Pipeline selection logic** (from `block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp`):
- `kUseTrLoad=false` AND `has_dpad1=false` → selects IGLP variant (crashes)
- `kUseTrLoad=false` AND `has_dpad1=true`  → selects non-IGLP variant (works)
- `has_dpad1 = (kPadHeadDimQ == 1 || kPadHeadDimV == 1)`

**Workaround**: Use `pad_hdim_q=1` and `pad_hdim_v=1` instead of `8` for all
DqDkDv variants. This forces `has_dpad1=true`, selecting
`BlockFmhaBwdDQDKDVPipelineKRKTRVR` (the non-IGLP variant) which works
correctly and passes numerical verification.

**Impact on performance**: The `pad=1` variant adds minimal bounds checking
that `pad=8` (vector-aligned padding) would optimize away. For the kpack demo
this is acceptable. For production use, this needs to be investigated with the
CK Tile team — the IGLP pipeline may have a compiler-specific bug with the
`amd-mainline` clang branch.

**How we found this**: The OGradDotO kernel (simple 1D reduction) worked
perfectly with generic Args. The DqDkDv kernel (5-GEMM template chain)
crashed on every launch. Bisecting by pipeline variant isolated the IGLP
pipeline as the crash source. The non-IGLP pipeline with identical Kargs
initialization passes all tests.

**TODO**: File a bug against the CK Tile IGLP pipeline with this compiler
version. Test with release ROCm compilers (6.x, 7.x) to determine if this
is a mainline regression.

## Migration from Flat Args to Generic rocm_ck::Args

The original OGradDotO example used per-mode flat structs
(`FmhaBwdOGradDotOBatchArgs`, `FmhaBwdOGradDotOGroupArgs`) with
`__builtin_bit_cast` to CK Tile's Kargs. The migration to generic
`rocm_ck::Args` (1408 bytes) required:

1. **Named slot constants** — `fmha_bwd_ograd_dot_o_slots::O`, `::DO`, `::D`
   etc. prevent off-by-one slot mapping errors.
2. **Aggregate Kargs initialization** — the device bridge constructs CK Tile's
   Kargs directly via aggregate init (matching the inheritance order), instead
   of `__builtin_bit_cast`. This works because CK Tile's own `MakeKargsImpl`
   uses the same aggregate init pattern.
3. **1D tensor stride convention** — tensors without a row stride (D, LSE)
   pack `strides[0]=nhead_stride, strides[1]=batch_stride` directly, NOT
   `strides[0]=1, strides[1]=nhead_stride`. A spurious `1` in `strides[0]`
   shifts all subsequent strides and causes wrong results.
4. **Dimension packing** — the DqDkDv dev bridge reads all problem dimensions
   from `Q.lengths[0..5]` (seqlen_q, seqlen_k, hdim_q, hdim_v, num_head_q,
   nhead_ratio_qk). The host must populate all 6 lengths, not just the
   tensor's own dimensions.

## HIP Device Consteval Limitations

The `.hip` files use `static constexpr <ExplicitType> kernel = make_kernel(...)`
as an NTTP for the device bridge template. Two HIP compiler limitations apply:

1. **`constexpr auto` fails on device** — `static constexpr auto kernel = ...`
   causes "const variable cannot be emitted on device side due to dynamic
   initialization." Use an explicit type (`FmhaBwdOGradDotOKernel`,
   `FmhaBwdDQDKDVKernel`, etc.) instead of `auto`.
2. **`__launch_bounds__` with struct members fails** —
   `__launch_bounds__(kernel.block_size, kernel.block_per_cu)` causes
   "'amdgpu_flat_work_group_size' attribute requires parameter 1 to be an
   integer constant." Use integer literals: `__launch_bounds__(256, 1)`.
3. **`#include <rocm_ck/args.hpp>` in API headers breaks device consteval** —
   `args.hpp` includes `<array>` which contains non-constexpr static
   initializers. API headers (included by `.hip` files) must NOT include
   `args.hpp`. Only the dev bridge (which has the CK Tile dependency already)
   should include it.

## Potential Padding Between Common and Group Extension

The common Kargs ends with `nhead_stride_d` (`index_t`, 4 bytes). The group
extension starts with `seqstart_q_ptr` (pointer, 8-byte aligned). The compiler
may insert 4 bytes of padding between them. This is handled automatically by
using the same inheritance structure (CK Tile side) vs flat struct (API side)
and verifying with `static_assert(sizeof)`. If the sizes don't match, the
`static_assert` in `dev.hpp` will catch it at compile time.
