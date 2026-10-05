<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Single-kernel direct dgrad: fusing the weight transform — gfx950 case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
This study records the mechanism, the levers that were tried (including the
ones that lost), the keep/revert decisions and the commands to replay them.
All measurements behind these decisions were taken with the default LLVM
flavor (`llvm20`); `llvm22` was not available on the measurement machine, so
none of the keep/revert decisions has been re-checked there.

## Starting point

Stride-1 grouped dgrad on the direct-MFMA kernels is the identity

```text
dX = conv(dY, W_T),   W_T[c, r', s', k] = W[k, KH-1-r', KW-1-s', c]  (per group),  PAD' = KH-1-PAD
```

and both direct paths materialised `W_T` first:

* generic (`make_dgrad_fprop_spec` + `build_direct_conv`): a scalar transpose
  kernel (`build_direct_transpose_weights_dgrad`) into a workspace, then the
  `16x16x16` / `16x16x32` row-streaming kernel on `(dY, W_T)`;
* 4c (`make_dgrad_4c_spec` + `build_direct_conv_4c`, cpg = kpg = 4): the same
  transpose kernel, then the batched `4x4x4` kernel.

The transpose kernel itself is short, but it is a second launch with a
dependency on the first, and on the 16- and 32-channel shapes that dominate
grouped dgrad the main kernel is short too, so the pre-pass plus the gap
between the launches is a large share of the pipeline. The reference
implementation does it in one kernel: raw `KYXC` weights to LDS, transpose on
the LDS read, flip by tap index.

## Lever 1 — `preload_weights` (waves_k = 1): no effect, kept as a knob

The plan assumed the `waves_k = 1` path re-reads weights on every output row
(the explicit preload existed only for `waves_k > 1`). A knob that loads every
fragment once in the prologue was added and compared at the ISA level: on the
16-channel shape the main kernel's ISA was instruction-for-instruction
identical with and without it. LLVM already hoists the per-row weight loads
out of the fully unrolled row loop (readonly, noalias buffer), so the knob is
a no-op there. It stays as a knob (it is also the register layout the fused
path needs), default off.

## Lever 2 — `dgrad_fused_weights`, gather form: correct, but slower kernel

The main kernel takes the ORIGINAL weight as `B` and builds the same fragment
the `W_T` path loads:

```text
W_T[g*kpg' + m*16 + q, r, s, ch + e]  ==  W[g*cpg' + ch + e, KH-1-r, KW-1-s, m*16 + q]
```

(`cpg' = kpg`, `kpg' = cpg` of the transposed problem). Element `e` of a lane's
fragment sits `KH*KW*kpg'` elements after element `e-1`, so the first form
gathers `LOAD_VEC` scalar `buffer_load_ushort` per fragment and packs them
(`vec_pack`); partial K atoms are masked by pointing the load at the
out-of-range sentinel. For the 4c kernel the lane needs
`W[g*4 + e, KH-1-r, KW-1-s, lane_q]`, four gathers per tap.

Result: correct everywhere, the pipeline wins because the second launch is
gone, but the main kernel is slower than the `W_T` kernel on the 16-channel
shape. ISA: the gathers are issued four at a time, each batch followed by
`s_waitcnt vmcnt` and a `v_perm_b32` pack into a recycled register quad — the
scheduler serialises the prologue into many dependent L2 round trips to keep
the kernel at its occupancy target. On the 32-channel `fold_k32` shape the
same pattern costs little (that kernel is already register-bound), and on the
4c kernel it costs a small prologue delay.

Decision: keep as the fallback for targets without transpose LDS reads
(gfx942), not as the gfx950 default.

## Lever 3 — `dgrad_weights_lds`: wide copy + `ds_read_b64_tr_b16`

Within one group the raw weights `W[g*cpg' .. +cpg', :, :, :]` are one
contiguous run, and so are the `block_groups` groups of a workgroup. The
kernel copies that slice into LDS with `buffer_load_dwordx4` (one 16-byte
chunk per thread per pass, all passes issued before the stores) and builds
every fragment with one `ds_read_b64_tr_b16` per 16 K-rows.

The lane map follows from the instruction's semantics (pinned by the wgrad
kernel's transpose reads and re-derived here): `ds_read_b64_tr_b16` hands lane
`16h + 4a + b` element `b` of the 8-byte row read by lane `16h + 4j + a`, for
`j = 0..3`. So

* generic kernel — lane `16h + 4j' + a'` reads row `k0 + h*KS + j'`
  (`KS = K_ATOM/4`), columns `m*16 + 4a' .. +3` of tap
  `(KH-1-r)*KW + (KW-1-s)`; the receiving lane `(c4 = h, q = 4a + b)` gets
  `W[k0 + c4*KS + j][m*16 + q]`, exactly the `16x16x16` A operand. For
  `fold_k32` a second read 4 rows down supplies elements 4..7.
* 4c kernel — lane `16h + 4j + a` reads the 4-channel run
  `W[g(4h + a)*4 + j, KH-1-r, KW-1-s, 0:4]`; the receiving lane
  `(batch = 4h + a, lane_q = b)` gets `W[g*4 + j][.][.][lane_q]`, the 4x4x4 A
  operand.

The flip is free: it is only the compile-time tap index.

First attempt: the staged slice was a separate LDS allocation live alongside
the row double buffer. On the 32-channel `fold_k32` shape that slice is large
enough that fewer workgroups fit per CU than the grid needs for a single
wave, and the kernel lost to the gather form. Fix: issue the first input row's
global loads, stage the weights, barrier, do every transpose read, barrier,
then allocate the row buffers and publish the first row. The row buffers'
live ranges now start after the staging slice dies, so the smem pool packer
overlays them (`_collect_smem_liveness` keys the range on the `smem_alloc`
op, which is why the allocation itself had to move). Cost: two extra
barriers in the prologue only.

Result: on gfx950 the single LDS-staged kernel matches the `W_T` main kernel
at the kernel level on the 16- and 32-channel shapes and on the 4c shapes, so
the whole pre-pass and its launch gap are removed. ISA (16-channel): 9
`ds_read_b64_tr_b16`, a handful of `buffer_load_dwordx4` + `ds_write_b128`
in the prologue, no `buffer_load_ushort`, no spills.

Decision: keep; default on gfx950 for both kernels.

## Lever 4 — `waves_per_eu` on the preloaded path

Comparing the preloaded-weight main loop with the original runtime-loop path
on a low-occupancy 16-channel shape showed a regression at `block_q = 32`
that is not about weights at all: the scheduler serialised the next-row input
loads (`buffer_load_dwordx2` → `s_waitcnt vmcnt(0)` → reuse of the same
register pair) to stay within the register target it picked. Setting
`"amdgpu-waves-per-eu"` (new `DirectConvSpec.waves_per_eu`) to 4 lets those
loads stay in flight together; 1 and 2 were worse (more registers, fewer
waves), and on the register-bound `fold_k32` kernel 4 is slightly worse than
leaving it unset.

Decision: keep as a knob; the dispatch hook sets 4 only for the small
(16-channel, non-`fold_k32`) preload footprint.

## Selection policy (`direct_dgrad_spec_for_problem`)

* cpg = kpg = 4 (stride 1, same padding, 1x1/3x3, groups % 16 == 0): 4c
  kernel, `dgrad_fused_weights`, `dgrad_weights_lds` where the target has
  transpose reads (gathers otherwise), `block_q = 4`, `block_groups = 16`.
* otherwise `DirectConvSpec`: `block_q = 16`; `block_groups = 2` while one
  group's weights are at most 8 KiB, else 1 (keeps the staged slice small for
  32-channel groups); `block_h = 0` unless that leaves fewer than
  `_DGRAD_MIN_WAVES` waves, then 16-row H tiles; `fold_k32` when
  `kpg % 32 == 0` (falls back without it, e.g. gfx942); LDS staging when legal
  and inside `DGRAD_WEIGHTS_LDS_BUDGET`, gathers otherwise; `waves_per_eu = 4`
  for the small preload footprint. `None` when the preloaded fragments exceed
  `PRELOAD_WEIGHT_VGPR_BUDGET` (large cpg x kpg): keep the pre-pass pipeline.
  `None` also outside the kernel's domain (see Eligibility below).

**Eligibility.** The generic row-streaming kernel flushes output row
`y - (KH-1)` against `H` and maps output columns 1:1 onto input columns, so
it is only correct for "same" padding (`2*PAD == KH-1 == KW-1`). Its
write-back covers `kpg` in whole 4-channel slices, so `kpg` (the original
`cpg` in the transposed dgrad problem) must be a multiple of 4. Outside that
domain it produced wrong values for both fprop and the pre-pass dgrad
pipeline (`PAD = 0` or `2` with 3x3, 1x3 with `PAD = 1`, original `cpg` of 2,
6 or 10). `DirectConvSpec.validate` / `is_valid_spec` now reject those shapes
(`_direct_conv_shape_reason`), so the hook, the pre-pass sweep and fprop all
skip them instead of building a wrong kernel, and the hook checks the same
rule up front on the original problem. The hook returns `None` for them and
the caller has to use a non-direct dgrad.

**`block_q = 32` on H-tiled `fold_k32` shapes: not adopted.** For the
pre-pass pipeline, `block_q = 32` is the best main kernel on the low-batch
32-channel cohort shape. With fused weights it loses clearly to
`block_q = 16`, at both `block_groups` 1 and 2. The reason is registers:
two Q sub-tiles double the accumulators on top of the preloaded weight
fragments. The unified VGPR+AGPR footprint grows from about 168 to about 268
registers per lane, and occupancy drops from three waves per SIMD to one (no
spills either way). The pre-pass kernel at `block_q = 32` is also at one wave
per SIMD, but it does not hold the weights resident. The hook keeps
`block_q = 16`.

The H-tiling and `block_groups` rules came from a low-batch 32-channel cohort
shape where whole-column streaming leaves the device under-filled and a
two-group workgroup doubles the staged slice; on the H-tiled shapes the fused
kernel's per-workgroup prologue (stage + two barriers) is amortised over
fewer rows, so the margin over the pre-pass pipeline is smaller there.

## Correctness

Every variant was checked with the conv tolerance rule
(`|out - ref| <= tol + tol * |ref|`, `tol = 1e-2`, zero bad elements, output
NaN-filled before the launch) on adversarial shapes: odd H / W, N = 1, W not a
multiple of `block_q`, partial K atoms (cpg 8 / 12), partial M tiles,
`cpg != kpg`, three K atoms, `fold_k32`, `block_h > 0`, `block_q = 32`, 1x1
filters, and the full 4c `block_q x block_groups` grid. The check is written
NaN-safe (`~(diff <= tol + tol*|ref|)`): a NaN, i.e. an unwritten output,
counts as bad (`diff > ...` would silently pass it). The hook's rejections
(non-"same" padding, `cpg` / `kpg` not a multiple of 4) are pinned by
`test_dispatch_rejects_unsupported_shapes`.

## Remaining gap and next levers

The fused kernels now cost what the old main kernels cost; the remaining
distance to the reference is in the row loop:

* input rows go global → registers → LDS with 8-byte lanes and per-row
  `s_waitcnt vmcnt(0)` + barrier; a direct-to-LDS (`buffer_load ... lds`)
  double-buffered row load is the next lever;
* 8-byte output stores from 16 of 64 lanes (generic) — LDS-staged 16-byte
  stores;
* the 4c kernel loads each input row `KW` times and masks every load with
  `v_cndmask`;
* `waves_m = 2` for 32-channel groups to cut registers per wave.

## Replay

From `library/` with the C++ engine on `PYTHONPATH` and `ROCKE_CPP_STRICT=1`:

```bash
# numerics (GPU): every mode, adversarial shapes, the dispatch defaults
python -m pytest tests/test_conv_dgrad_fused_weights.py tests/test_conv_dgrad_4c.py -v

# sweep: single-kernel forms only, or everything incl. the pre-pass pipelines
python -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --dgrad-family fused --verify --jobs 24 \
    --N 128 --Hi 14 --Wi 14 --C 512 --K 512 --Y 3 --X 3 --pH 1 --pW 1 \
    --groups 32 --dtype bf16
python -m benchmarks.common.benchmark_direct_conv --direction dgrad \
    --dgrad-family all --verify --jobs 24 \
    --N 128 --Hi 56 --Wi 56 --C 128 --K 128 --Y 3 --X 3 --pH 1 --pW 1 \
    --groups 32 --dtype bf16
```

The benchmark's event timing includes host launch overhead per launch, which
favours the single kernel more than kernel time does; use
`rocprofv3 --kernel-trace` durations (sum over the kernels of one pipeline)
for the kernel-level comparison. ISA: `llvm-objdump -d --mcpu=gfx950` on the
compiled code object; count `ds_read_b64_tr_b16`, `buffer_load_ushort`,
`buffer_load_dwordx4`, `s_waitcnt vmcnt(0)` per row, and check
`.vgpr_spill_count` / `.group_segment_fixed_size` in `llvm-readelf --notes`.

Byte identity (4c mirror, parity configs 45-48) and the representative-IR
golden:

```bash
cd platform && export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python:$ROCKE/../library
TMPDIR=<private dir> python tools/check_byte_identity.py \
    --only conv_direct_grouped,target_intrinsics --build-root <private dir>
for f in llvm20 llvm22 llvm23; do
  python tests/instances/rocke_ir_parity_harness.py \
      --check tests/golden/rocke_representative_ir_sha256.json --flavor $f
done
```

The parity pair covers the emitters, not the public
`rocke.core.backend.lower_conv_direct_grouped` path, which flattens the spec
into a dict for the C++ binding. The fused knobs and the problem `dtype` must
be forwarded there too; `library/tests/test_conv_direct_grouped_backend_parity.py`
checks that path with `backend="both"` (see the 4c case study, "Correctness
gates").

## In production dispatch

`direct_mfma_conv_dgrad` now selects the fused form by default: the dispatcher's
own knob table (block_groups / block_h / block_q) with `dgrad_fused_weights`,
LDS staging where the slice fits, and the `waves_per_eu = 4` rule from this
study for small preload footprints. The hook's own knob choice
(`direct_dgrad_spec_for_problem`) was compared against that and not adopted for
dispatch: the table's spatial tiling won on aggregate. Past
`PRELOAD_WEIGHT_VGPR_BUDGET` the candidate falls back to the pre-pass pipeline;
`plan_direct_mfma_dgrad` describes both forms, so the dispatcher, the benchmark
and the tests launch them the same way. Removing the pre-pass also changed
where the direct candidate beats the igemm one, so the dispatch eligibility was
refitted; see `grouped_direct_dgrad_dispatch_case_study.md`.
