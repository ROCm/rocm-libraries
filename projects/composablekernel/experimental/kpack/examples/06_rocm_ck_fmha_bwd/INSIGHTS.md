# INSIGHTS.md — FMHA BWD OGradDotO kpack Example

Design decisions and lessons learned from mapping the CK Tile FMHA backward
OGradDotO kernel to the kpack pattern.

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

## Potential Padding Between Common and Group Extension

The common Kargs ends with `nhead_stride_d` (`index_t`, 4 bytes). The group
extension starts with `seqstart_q_ptr` (pointer, 8-byte aligned). The compiler
may insert 4 bytes of padding between them. This is handled automatically by
using the same inheritance structure (CK Tile side) vs flat struct (API side)
and verifying with `static_assert(sizeof)`. If the sizes don't match, the
`static_assert` in `dev.hpp` will catch it at compile time.
