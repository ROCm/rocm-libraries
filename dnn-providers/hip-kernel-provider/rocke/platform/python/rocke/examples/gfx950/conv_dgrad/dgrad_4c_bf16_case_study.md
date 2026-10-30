<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# 4c dgrad on the batched 4x4x4 MFMA — gfx950 case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
This study records the mechanism, the levers that were swept, the keep/revert
decisions, what the AOT port changed, and the commands to replay them.

Knob reference (`dgrad_fused_weights`, `dgrad_weights_lds`, `stage_rows`,
`waves_q`, name flags, the cached image):
[`dsl_docs/instances/convolution.md`](../../../../../dsl_docs/instances/convolution.md),
section "4c dgrad". The fused weight transform:
[`dgrad_fused_weights_case_study.md`](dgrad_fused_weights_case_study.md).

## Kernel and where it lives

`DirectConv4cSpec` / `build_direct_conv_4c` and the dgrad entry points
(`make_dgrad_4c_spec`, `dgrad_4c_transposed_problem`,
`is_valid_dgrad_4c_problem`, `build_direct_4c_dgrad`, `direct_4c_dgrad_launch`;
`dgrad_4c_spec_for_problem` is a kernel-level helper)
in [`conv_direct_grouped.py`](../../../../../../library/kernels/common/conv_direct_grouped.py);
C++ mirror in [`conv_direct_grouped_build_4c_phases_and_loop.cpp`](../../../../../cpp/instances/common/conv_direct_grouped_build_4c_phases_and_loop.cpp)
and `conv_direct_grouped_spec_and_validate.cpp`. AOT cache variant
`direct_grouped_dgrad_4c` (`DGRAD_4C_*` in
[`direct_aot_jobs.py`](../../../../../../library/kernels/conv/direct_aot_jobs.py));
not dispatched yet (the grouped-convolution dispatcher is being reworked
separately), so it is reached through the cache (`--run-from-cache`). Scope: `cpg = kpg = 4`, fp16 /
bf16, stride 1, "same"-padded 1x1 or 3x3, `groups % 16 == 0`.

## Finding: at cpg = 4 the 16x16x16 atom does one sixteenth useful work

The generic direct-MFMA kernel maps one group onto a `16x16x16` MFMA; with four
channels per side only a 4x4 corner of the operand is real, the rest is masked
(`v_cndmask` on every fragment) and only 16 of 64 lanes store. Counters showed
the MFMA pipe as the dominant busy unit although almost all its work is
padding. The wave64 `4x4x4` MFMA computes sixteen independent 4x4x4 products,
so one wave covers sixteen groups with no wasted lanes; the 4c fprop kernel
used it already, but fp16-only, because the IR had no bf16 atom.

## Change 1: `mfma_f32_4x4x4_bf16`

`llvm.amdgcn.mfma.f32.4x4x4bf16.1k` (CDNA2+) selects
`v_mfma_f32_4x4x4_16b_bf16` on gfx950. Like the 16x16x16 bf16 atom it takes
`<4 x bfloat>` IR operands bitcast to `<4 x i16>` at the intrinsic. Both
engines: `IRBuilder.mfma_f32_4x4x4_bf16` / `rocke_b_mfma_f32_4x4x4_bf16`, the
MMA fragment tables (`target.py`, `arch/data.cpp`), the intrinsic declaration
tables (same position), the LLVM lowering (`_op_tile_mfma_f32_4x4x4_bf16`, the
`MFMA_SPECS` row in `mma.cpp`) and the Python HIP lowering. It goes through the
neutral `tile.mma` op with an `op_id`, so no new opcode. The lane map was pinned
on hardware first (`platform/tests/core/test_mfma_4x4x4_numerics.py`): lane `l`
(batch `l // 4`, `i = l % 4`) supplies row `i` of `A_batch` and column `i` of
`B_batch` and receives `D_batch[m][i]` in slot `m`; f16 and bf16 agree.

## Change 2: bf16 in the 4c kernel

`build_direct_conv_4c` takes its I/O type from `problem.dtype` (`_io_type`,
`_buf_load_vN`, `_buf_store_vN`, `_mfma(b, dtype, "4x4x4", ...)`); the kernel
name gains a `bf16` flag, and fp16 names and IR are unchanged (the fp16 4c
golden entries did not move). Both atoms exist on gfx942 and gfx950, so the
kernel stays arch-neutral. The C++ mirror carries `io_type` / `is_bf16` in
`rocke_dconv_4c_ctx_t`.

## Change 3: the 4c dgrad entry

```text
dX = conv(dY, W_T),   W_T[c, r', s', k] = W[k, KH-1-r', KW-1-s', c]  (per group),  PAD' = KH-1-PAD
```

so the 4c kernel runs as a forward kernel over the transposed problem. It keeps
`KH` accumulator slots and one weight fragment per tap in registers, hence
`KH in (1, 3)`; `is_valid_spec_4c` bounds `KH*KW` by `DCONV4C_MAX_TAPS`
(`ROCKE_DCONV4C_MAX_TAPS` sizes the C++ per-tap arrays), so a directly built 5x5
spec is rejected on both engines with the same reason instead of overflowing.

**Knob sweep (direct-load kernel).** `block_q = 4` was best or tied on every
cpg = 4 shape swept: larger `block_q` grows VGPR + AGPR use and MFMAs per wave
with no reuse gain (each q-tile reloads its own windows). `block_groups` 16 and
32 tie; 64 never won. Kept: `DGRAD_4C_DEFAULT_BLOCK_Q` /
`DGRAD_4C_DEFAULT_BLOCK_GROUPS`.

**Why it beats the generic kernel (ISA).** Equal MFMA instructions per wave,
but `v_mfma_f32_4x4x4_16b_bf16` is cheaper than `v_mfma_f32_16x16x16_bf16`, the
4c grid needs fewer than a quarter of the waves for the same output, the
direct-load 4c kernel has no LDS row staging or per-row barrier, both use fewer
registers with no spills, and a 4c workgroup reads 16 groups x 4 channels =
128 contiguous bytes per pixel instead of 32.

## Row-staged form (`stage_rows`)

**Diagnosis.** With fused LDS weights in place, counters showed the
texture-address unit busy for most of the kernel and the memory unit stalled:
in the 4x4x4 B layout consecutive lanes of every `buffer_load_dwordx2` sit on
pixels a channel row apart, three loads and one store per wave per row.
Register prefetch of the next row (waits left the critical path, no gain:
occupancy already hid the latency) and an XCD-contiguous tile order (noise)
were kept off.

**Change.** Once per input row the workgroup copies its row into a
double-buffered, padded LDS row, one row ahead through registers, and every
wave reads its B fragments with `ds_read_b64`: one 16-byte load per thread per
row instead of three strided 8-byte loads per lane. LDS-staged output stores,
an LDS-only row barrier, the XCD order and wider group tiles did not pay.
`_stage_rows_reject_reason` (mirrored, same text): needs the fused LDS weight
form, `block_q == 4 * waves_q`, stride 1 and "same" padding; `waves_q > 1`
needs `stage_rows`; bounded by `DCONV4C_MAX_THREADS`, `DCONV4C_MAX_ROW_PASSES`
and the target's LDS.

## What the AOT port changed

- **ABI.** The *forward* direct ABI over the transposed problem
  (`conv_direct_args_signature`, buffers `(dy, w, dx)`, no workspace);
  `plan_for` (`benchmarks/common/direct_kernel_sweep.py`) treats the variant
  as a single forward step (`DIRECT_DGRAD_FPROP_SINGLE` in
  `kernels/conv/direct_variants.py`).
- **Capabilities vs kernargs.** Filter, "same" padding, stride 1 and
  `cpg = kpg = 4` are baked; N, H, W, groups and the strides are kernargs.
  `groups % block_groups` is the one shape rule, re-checked by `plan_for`.
- **Runtime row stream.** The pre-port staged kernel unrolled the whole image and
  skipped padding rows at build time. Now the rows stream through a runtime
  `scf.for` from `PAD` to `p_Hi + KH - 1`, unrolled `direct_row_loop_unroll(KH)`
  (`lcm(KH, 2)`) rows so buffer parity and accumulator slots stay build-time
  constants; rows outside `[0, p_Hi)` load zeros through the sentinel and drop
  their stores; offsets come from the stride kernargs.
- **Cached image.** `DGRAD_4C_BLOCK_Q` × `DGRAD_4C_BLOCK_GROUPS` ×
  `DGRAD_4C_FORMS` (gathered, LDS-staged, row-staged weights; the validator
  drops the LDS forms on gfx942). Every cached 4c binary has fused weights
  (`make_direct_spec` in `kernels/conv/direct_variants.py`); the pre-pass
  form is not cached.

## Choosing between the 4c forms

The kernel is not dispatched yet; the cache carries the image of the 4c
choice (`DGRAD_4C_*`: the `(4, 16)` tile in all three weight forms plus a
small neighbourhood of wider column and group tiles), and which entry serves
a request is left to the dispatcher rework. What the measurements showed,
on same-session cohorts of 1x1 / 3x3 `cpg = kpg = 4` shapes checked on a
disjoint hold-out with warm and with flushed caches:

- the **row-staged** kernel pays on larger grids and taller images; on 3x3
  two q waves sharing the staged row and its halo (`block_q = 8`) was the
  fastest cached tile. Taking it on small grids and short images won warm
  but lost cold: the generic kernel's row tiles give more workgroups to hide
  DRAM latency, and implicit GEMM held up better on the smallest grids;
- the **direct-load** kernel does not tile H, so on small grids the fused
  generic kernel, which does, is the better choice.

## Correctness gates

- `platform/tests/core/test_mfma_4x4x4_numerics.py`: lane map and numerics of
  both atoms, LLVM and HIP lowering text.
- `library/tests/test_conv_dgrad_4c.py`: atom per dtype and arch, dgrad spec
  plumbing, tap bound, `stage_rows` validation and a runtime-loop emission
  check; GPU numerics of every form (pre-pass, gathered, LDS, row-staged) on
  the forward ABI, on odd H / W, H = 1, W = 1, N = 1, partial q tiles and
  staging passes, the knob grid and 1x1, plus one binary per form on several
  shapes. Bad unless `|out - ref| <= tol + tol * |ref|`, `tol = 1e-2`, output
  NaN-filled first.
- Public dual-engine entry. `rocke.core.backend.lower_conv_direct_grouped`
  flattens the spec (`conv_direct_grouped_spec_to_dict`) for the binding
  (`fill_direct_conv_problem`, `dg4_build_spec`), which must read every
  kernel-shaping field. The dict once dropped `dtype`, so the C++ backend built
  an fp16 kernel for a bf16 spec — invisible to byte identity, which drives the
  `_emit.c` parity pair. `library/tests/test_conv_direct_grouped_backend_parity.py`
  runs `backend="both"` over the 4c fprop and dgrad forms, with matching reject
  reasons.
- Cache: `test_dgrad_cpg4_*` in `library/tests/test_direct_kernel_cache.py`
  (cached scope, the shapes it rejects, LDS forms gfx950-only, identity
  lookup).
- Byte identity: `conv_direct_grouped` 148..150 (bf16 4c fprop),
  `conv_direct_dgrad_ext` 20..29 (4c dgrad; 28, 29 both reject),
  `target_intrinsics` (the atom). Golden: `conv_direct/gfx950/4c_bf16_n2h13`,
  `conv_direct/gfx942/4c_bf16_n1h8`, `conv_direct_dgrad/gfx950/4c_dgrad_fwl_bf16_n2h13`,
  `.../4c_dgrad_fwl_sr2_bf16_n2h9`, `conv_direct_dgrad/gfx942/4c_dgrad_fw_fp16_n1h8`.

## Replay

From `library/`, `PYTHONPATH=$(pwd)/../platform/python:$(pwd)`:

```bash
python3 -m benchmarks.common.benchmark_direct_conv --compile-all \
    --direction dgrad --dtype bf16 --cache-dir <direct cache> --jobs 0
# the cached 4c binaries (all three weight forms) on the shape
python3 -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --N 24 --Hi 12 --Wi 20 --C 192 --K 192 --groups 48 --Y 3 --X 3 \
    --pH 1 --pW 1 --dtype bf16 --verify --run-from-cache <direct cache> \
    --variants direct_grouped_dgrad_4c
# every cached 4c / fused binary on the shape, each verified
python3 -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --N 24 --Hi 12 --Wi 20 --C 192 --K 192 --groups 48 --Y 3 --X 3 \
    --pH 1 --pW 1 --dtype bf16 --verify --run-from-cache <direct cache> \
    --variants direct_grouped_dgrad_4c,direct_grouped_dgrad_fused
python3 -m pytest -q tests/test_conv_dgrad_4c.py \
    tests/test_conv_direct_grouped_backend_parity.py
cd ../platform && export ROCKE=$(pwd) PYTHONPATH=$(pwd)/python:$(pwd)/../library
python3 -m pytest -q tests/core/test_mfma_4x4x4_numerics.py
python3 tools/check_byte_identity.py \
    --only conv_direct_grouped,conv_direct_dgrad_ext,target_intrinsics
python3 tests/instances/rocke_ir_parity_harness.py \
    --check tests/golden/rocke_representative_ir_sha256.json --flavor llvm22
```

Give concurrent worktrees their own `TMPDIR` and `--build-root` for
`check_byte_identity.py`. Time short kernels from `rocprofv3 --kernel-trace`,
compared within one session.

## Remaining gap

Output stores are 8-byte MFMA-layout stores, halo columns are fetched once per
XCD, and the padded LDS row still has some bank conflicts; LDS-staged stores
did not pay on the shapes tried.
