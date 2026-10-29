<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Single-kernel direct dgrad: fusing the weight transform — gfx950 case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
This study records the mechanism, the levers that were tried (including the
ones that lost), the keep/revert decisions, what the AOT port changed, and the
commands to replay them.

Knob reference (`preload_weights`, `dgrad_fused_weights`, `dgrad_weights_lds`,
`waves_per_eu`, `row_guard`, `row_exit`, the row-stream knobs, name flags and
the cached images):
[`dsl_docs/instances/convolution.md`](../../../../../dsl_docs/instances/convolution.md),
sections "4c dgrad" and "Generic Kernel dgrad: fused weights". The 4c kernel:
[`dgrad_4c_bf16_case_study.md`](dgrad_4c_bf16_case_study.md).

## Kernel and where it lives

Generic `DirectConvSpec` / `build_direct_conv` (transposed spec from
`make_dgrad_fprop_spec`) and the 4c `DirectConv4cSpec` /
`build_direct_conv_4c` in
[`conv_direct_grouped.py`](../../../../../../library/kernels/common/conv_direct_grouped.py);
C++ mirrors `rocke_build_direct_conv`
([`conv_direct_grouped_build_generic.cpp`](../../../../../cpp/instances/common/conv_direct_grouped_build_generic.cpp))
and the 4c builder. AOT cache variants
`direct_grouped_dgrad_fused` and `direct_grouped_dgrad_4c` in
[`direct_aot_jobs.py`](../../../../../../library/kernels/conv/direct_aot_jobs.py);
cache specs from `make_direct_spec` in
[`direct_variants.py`](../../../../../../library/kernels/conv/direct_variants.py).
Not dispatched yet (the grouped-convolution dispatcher is being reworked
separately), so both are reached through the cache (`--run-from-cache`).

## Starting point

Stride-1 grouped dgrad on the direct-MFMA kernels is the identity

```text
dX = conv(dY, W_T),   W_T[c, r', s', k] = W[k, KH-1-r', KW-1-s', c]  (per group),  PAD' = KH-1-PAD
```

and both direct paths first wrote `W_T` to a workspace with a scalar transpose
kernel (`build_direct_transpose_weights_dgrad`). The transpose is short, but it
is a second dependent launch, and on small-channel grouped shapes the main
kernel is short too, so pre-pass plus launch gap is a large share of the
pipeline. A reference library does it in one kernel: raw weights to LDS,
transpose on the LDS read, flip by tap index.

## Levers

**1 — `preload_weights` (`waves_k = 1`): no effect, kept as a knob.** A
prologue preload of every fragment gave instruction-for-instruction identical
ISA on a `cpg = kpg = 16` shape: LLVM already hoists the per-row weight loads
out of the unrolled row loop (readonly, noalias buffer). It stays because it is
the register layout the fused path needs.

**2 — `dgrad_fused_weights`, gather form: correct, slower main kernel.** The
kernel takes the ORIGINAL weight as `B` and builds the fragment the `W_T` path
loads,

```text
W_T[g*kpg' + m*16 + q, r, s, ch + e]  ==  W[g*cpg' + ch + e, KH-1-r, KW-1-s, m*16 + q]
```

(`cpg' = kpg`, `kpg' = cpg` of the transposed problem). Element `e` sits
`KH*KW*kpg'` elements after `e-1`, so the gather issues scalar
`buffer_load_ushort` per element and packs them; partial K atoms point at the
out-of-range sentinel; the 4c kernel needs four gathers per tap. The pipeline
wins (the second launch is gone), but the main kernel was slower than the `W_T`
kernel at `cpg = kpg = 16`: the gathers issue four at a time, each batch
followed by `s_waitcnt vmcnt` and a `v_perm_b32` pack into a recycled register
quad, so the prologue serialises into dependent L2 round trips. On the
register-bound `fold_k32` kernel and the 4c kernel it costs little.
**Decision:** the fallback for targets without transpose LDS reads (gfx942).

**3 — `dgrad_weights_lds`: wide copy + `ds_read_b64_tr_b16` (KEEP, gfx950
default).** A workgroup's raw weights are one contiguous run, copied to LDS
with `buffer_load_dwordx4` (all passes issued before the stores); every
fragment is one `ds_read_b64_tr_b16` per 16 K-rows. The instruction hands lane
`16h + 4a + b` element `b` of the 8-byte row read by lane `16h + 4j + a`
(`j = 0..3`), so for the generic kernel lane `16h + 4j' + a'` reads row
`k0 + h*KS + j'` (`KS = K_ATOM/4`), columns `m*16 + 4a' .. +3` of tap
`(KH-1-r)*KW + (KW-1-s)` and receives exactly the `16x16x16` A operand
(`fold_k32`: a second read 4 rows down); for 4c, lane `16h + 4j + a` reads the
run `W[g(4h + a)*4 + j, KH-1-r, KW-1-s, 0:4]` and receives the 4x4x4 A
operand. The flip is free: it is only the compile-time tap index.
*First attempt REVERTED:* the staged slice lived alongside the row double
buffer; at `cpg = kpg = 32` with `fold_k32` fewer workgroups fit per CU than
the grid needed and the kernel lost to the gather form. *Fix:* issue the first
input row's loads, stage, barrier, do every transpose read, barrier, then
allocate the row buffers, whose live ranges now start after the slice dies, so
the LDS pool packer overlays them (`_collect_smem_liveness` keys on the
`smem_alloc` op, which is why the allocation itself moved;
`direct_conv_lds_bytes` mirrors the placement). Cost: two prologue barriers.
Result: the single kernel matches the `W_T` main kernel at `cpg = kpg` 16 and
32 and on the 4c shapes. ISA (`cpg = kpg = 16`): 9 `ds_read_b64_tr_b16`, a
handful of `buffer_load_dwordx4` + `ds_write_b128`, no `buffer_load_ushort`, no
spills. Bounded by `DGRAD_WEIGHTS_LDS_BUDGET`.

**4 — `waves_per_eu` (KEEP as knob).** On a low-occupancy `cpg = kpg = 16`
shape the preloaded loop regressed at `block_q = 32`: the scheduler serialised
the next-row input loads (`buffer_load_dwordx2` → `s_waitcnt vmcnt(0)` → reuse
of the register pair) to meet its register target. `amdgpu-waves-per-eu = 4`
keeps them in flight; 1 and 2 were worse, and on the register-bound `fold_k32`
kernel 4 is slightly worse than unset — so it is set only for small
16-wide-atom preload footprints.

**`block_q = 32` on H-tiled `fold_k32` shapes: not adopted.** Best for the
pre-pass pipeline, but with fused weights two Q sub-tiles double the
accumulators on top of the resident weights: the VGPR+AGPR footprint grows from
about 168 to about 268 per lane and occupancy drops from three waves per SIMD to
one (no spills either way).

## Eligibility (correctness finding)

The generic row stream flushes output row `y - (KH-1)` against `H` and maps
output columns 1:1 onto input columns, so it is correct only for "same" padding
(`2*PAD == KH-1 == KW-1`), and its epilogue writes `kpg` in whole 4-channel
slices. Outside that domain it produced wrong values for fprop and the pre-pass
dgrad pipeline. `is_valid_spec` rejects those shapes (`forward_padding_reason`,
the column-padding and `kpg % 4` rules); `is_valid_dgrad_fused_problem` checks
the same on the original problem.

## What the AOT port changed

- **ABI.** Both kernels are *forward* kernels over the transposed problem with
  the forward direct ABI (`conv_direct_args_signature`, buffers `(dy, w, dx)`, no
  workspace); `plan_for` (`benchmarks/common/direct_kernel_sweep.py`) treats
  both variants as one forward step (`DIRECT_DGRAD_FPROP_SINGLE` in
  `kernels/conv/direct_variants.py`); no pre-pass is planned.
- **Capabilities vs kernargs.** Filter, padding, stride 1, `cpg`, `kpg` are
  baked (every weight stride is a product of baked counts); N, H, W, groups and
  strides are kernargs; `groups % block_groups` is re-checked by `plan_for`.
- **`block_h > 0` is mandatory**; the pre-port `block_h = 0` ("whole image") does not
  exist. A whole-image tile maps to the smallest cached tile at least as tall
  (`dgrad_fused_block_h`), and every cached fused entry has `row_guard`
  (`make_direct_spec`): each row past `p_Hi` runs its MFMAs in a
  block-uniform one-trip loop. The 16- and 32-row tiles, which only a
  whole-image tile maps to, also carry `row_exit` (`DGRAD_FUSED_EXIT_BLOCK_H`).
- **Cached image** (`direct_grouped_dgrad_fused`, filtered by
  `_dgrad_fused_policy_reason`): `DGRAD_FUSED_FILTERS` × `DGRAD_FUSED_CHANNELS`
  on both sides, minus 5x5/7x7 pairs over `PRELOAD_WEIGHT_VGPR_BUDGET`
  (`_dgrad_fused_weights_fit`); the knobs are the policy's image —
  `DGRAD_FUSED_BLOCK_Q`, `DGRAD_FUSED_BLOCK_GROUPS` up to the per-channel rule,
  `DGRAD_FUSED_BLOCK_H` (`DGRAD_FUSED_BLOCK_H_LARGE` for 5x5/7x7), `fold_k32`
  and LDS staging where valid, `DGRAD_FUSED_WPE` up to
  `DGRAD_FUSED_WPE_MAX_VGPRS`, and on gfx950 the row-stream stack
  (`DGRAD_FUSED_STREAM`). A separate variant from the pre-pass
  `direct_grouped_dgrad_mfma`, so existing identities did not move.
- **C++.** The generic builder's C++ twin gained every new knob,
  so the fused generic kernel is byte-identity gated, not only golden-covered.
- `direct_dgrad_spec_for_problem` stays a kernel-level helper with its own
  tiling rule; the cache does not use it (its knob image follows a
  spatial table that won on aggregate before the port).

## The cached knob image

The kernel is not dispatched yet. The cache carries the image of the generic
direct dgrad policy per capability rather than a product; the cache-scope
rules are `_dgrad_fused_policy_reason` in `direct_aot_jobs.py` (the validator
still accepts what they skip, so a JIT sweep can try it):

- `block_groups` at or below the per-channel row
  (`_dgrad_fused_rule_block_groups`: the wider channel count times
  `block_groups` stays near a 32..64-channel slice, halved for 5x5/7x7,
  whose accumulator ring is KH deep);
- `block_h` in `DGRAD_FUSED_BLOCK_H`, 4 and 8 only for 5x5/7x7; 32-column
  strips only on 4- and 8-row tiles with wide channel groups;
- `fold_k32` where valid (16-wide atoms only where it is not), LDS-staged
  weights where valid (gathered weights only where they are not);
- `waves_per_eu = DGRAD_FUSED_WPE` exactly where the preloaded weights stay
  within `DGRAD_FUSED_WPE_MAX_VGPRS` registers on 16-wide atoms;
- the row-stream stack on square channel groups on 16-column strips, not
  where `waves_m = 2` is valid (see
  [`grouped_generic_row_stream_case_study.md`](grouped_generic_row_stream_case_study.md)).

Shapes outside the capability rows (strided, filters past 7x7, channels past
32, 5x5/7x7 pairs over the register budget) have no fused binary and are
implicit-GEMM territory; which of the two serves a covered request is left to
the dispatcher rework. No weight pre-pass pipeline is cached for the fused
variants.

## Correctness

Rule: `|out - ref| <= tol + tol * |ref|`, `tol = 1e-2`, zero bad elements,
output NaN-filled first and the check written `~(diff <= ...)` so an unwritten
(NaN) element counts as bad. Adversarial shapes: odd H / W, N = 1, W not a
multiple of `block_q`, partial K atoms and M tiles, `cpg != kpg`, `fold_k32`,
tiles taller than the image, 1x1 to 7x7.

- `library/tests/test_conv_dgrad_fused_weights.py`: name flags, invalid
  combinations, `test_row_guard_is_one_trip_loop_per_late_row`, the helper and
  `test_dgrad_spec_hook_rejects_unsupported_shapes`; GPU numerics of every mode
  and row-stream knob, one binary per form on several shapes.
- `library/tests/test_direct_mfma_dgrad_correctness.py`: validation, LDS
  footprint vs the lowered pool, and the cached binaries on adversarial shapes
  through the AOT ABI.
- `library/tests/test_direct_kernel_cache.py` (`test_dgrad_fused_*`, incl.
  `test_dgrad_fused_policy_image_is_enumerated`: every pick of a model of the
  policy on a spread of problems is a cached, valid identity);
  `library/tests/test_conv_direct_grouped_backend_parity.py` (`backend="both"`,
  every field forwarded, matching reject reasons).
- Byte identity: `conv_direct_dgrad_ext` 30..44 (generic; rejects listed in the
  emitter header) and 20..29 (4c). Golden: `conv_direct_dgrad/gfx950/mfma_dgrad_fwl_c16_bf16`,
  `mfma_dgrad_fwl_c32_k32_bf16`, `mfma_dgrad_fwl_c16_stream_bf16`,
  `mfma_dgrad_fwl_c32_k32_wm2_bf16`, `conv_direct_dgrad/gfx942/mfma_dgrad_fw_c12_fp16`.

## Remaining gap

Input rows still go global → registers → LDS with per-row waits and a barrier;
a direct-to-LDS double-buffered row load is the next lever. (The other pre-port
next levers, LDS-staged 16-byte output stores and `waves_m = 2`, landed as the
`stage_out` and `waves_m` row-stream knobs.)

## Replay

From `library/`, `PYTHONPATH=$(pwd)/../platform/python:$(pwd)`:

```bash
python3 -m benchmarks.common.benchmark_direct_conv --compile-all \
    --direction dgrad --dtype bf16 --cache-dir <direct cache> --jobs 0
# the cached fused binaries on the shape
python3 -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --N 2 --Hi 12 --Wi 12 --C 96 --K 96 --groups 8 --Y 3 --X 3 \
    --pH 1 --pW 1 --dtype bf16 --verify --run-from-cache <direct cache> \
    --variants direct_grouped_dgrad_fused
# every cached single-kernel form on the shape, each verified
python3 -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --N 2 --Hi 12 --Wi 12 --C 96 --K 96 --groups 8 --Y 3 --X 3 \
    --pH 1 --pW 1 --dtype bf16 --verify --run-from-cache <direct cache> \
    --variants direct_grouped_dgrad_4c,direct_grouped_dgrad_fused
python3 -m pytest -q tests/test_conv_dgrad_fused_weights.py \
    tests/test_direct_mfma_dgrad_correctness.py tests/test_direct_kernel_cache.py
cd ../platform && export ROCKE=$(pwd) PYTHONPATH=$(pwd)/python:$(pwd)/../library
python3 tools/check_byte_identity.py --only conv_direct_dgrad_ext,conv_direct_grouped
python3 tests/instances/rocke_ir_parity_harness.py \
    --check tests/golden/rocke_representative_ir_sha256.json --flavor llvm22
```

Host-event timing includes launch overhead, which favours a single kernel;
compare kernels with `rocprofv3 --kernel-trace`. ISA: `llvm-objdump -d
--mcpu=gfx950`, count `ds_read_b64_tr_b16`, `buffer_load_ushort`,
`buffer_load_dwordx4`; `llvm-readelf --notes` for spills and LDS size.
