<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Windowed depthwise dgrad — gfx950 case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
What is recorded here is mechanism, instruction-count evidence, the levers that
were tried (kept and reverted), and the replay path.

Kernel: `build_direct_depthwise_dgrad_windowed` /
`DirectDepthwiseDgradWindowedSpec` in `library/kernels/common/conv_direct_grouped.py`,
C++ mirror `platform/cpp/instances/common/conv_direct_grouped_build_depthwise_dgrad_win.cpp`,
dispatch candidate `direct_depthwise_dgrad_win` in
`library/dispatch/grouped_convolution.py` (gfx950, cpg = kpg = 1, stride 1).

## Starting point

Depthwise dgrad had no dispatch path on gfx950: the implicit-GEMM dgrad candidate
declines `cpg = 1`, and the only fast kernel was the benchmark-only
`build_direct_depthwise_dgrad_streaming`. Its ISA on a 7×7, W = 14 shape showed
three costs that have nothing to do with the arithmetic:

| Symptom (per lane) | Cause |
| --- | --- |
| hundreds of `buffer_load_ushort`, several times the unique dY values | every `(r, s, j)` tap issues its own guarded load |
| over a thousand `v_cndmask_b32` | each tap selects both the address *and* the loaded value, although the OOB sentinel already returns zero |
| tile tail waste | the swept `block_w` values (4/8/16/32) do not divide W = 14 or W = 12 |

It also took minutes to compile a single 7×7 config at `block_w = 8`, and never
finished at `block_w = 32`, which stalled `--direction dgrad` sweeps.

## Lever 1 — one window load per dY row (KEEP)

For dY row `ho`, block columns `[wi0, wi0 + block_w)` need dY columns
`wi0 + PAD - (KW - 1) + t` for `t < block_w + KW - 1`. Load that window once and
read tap `(s, j)` from `window[j + KW - 1 - s]`. Out-of-range columns load through
the sentinel offset and read as zero, so the FMA chain has no select. Loads per
row drop from `KW * block_w` to `block_w + KW - 1`, and the selects collapse
into one per window column. Compile time drops by orders of magnitude because
the unrolled body loses its per-tap address arithmetic.

## Lever 2 — lane/row split addressing with two sentinels (KEEP, second form)

Every dY/dX offset is `lane + row`:

- the **lane** part is the row-invariant `(channel, column)` byte offset,
  computed once per window column, or `DW_DGRAD_WIN_OOB_LANE = 2**30` when the
  channel or column is out of range;
- the **row** part is the block-uniform row offset, or
  `DW_DGRAD_WIN_OOB_UNIFORM = 2**30 - 1` for a padding row of a split-H block.

The IR `add` lowers to `add nsw`, so a single `2**31 - 1` sentinel plus an offset
would be signed overflow (poison). The split sentinels sum to `2**31 - 1`, which
keeps every add in range and every invalid access past the buffer; the validator
therefore caps dY and dX below `2**30` bytes (dispatch falls back above that).

**First form REVERTED.** Folding the *column* validity into the uniform part
made all `(row, column)` offsets scalar: LLVM hoisted every `s_cselect`, ran out
of SGPRs and spilled them to VGPR lanes (`v_writelane` / `v_readlane` storm,
more `s_waitcnt`). Moving column validity into the per-column lane VGPR (one
`v_cndmask` per window column, ever) removed the spills and nearly all selects.

## Lever 3 — software prefetch of the next row + `sched_barrier` (KEEP, hardwired)

With cheap addressing the machine scheduler issued each window load just before
its first use (`load, short FMA run, s_waitcnt`), exposing memory latency; the
result was bimodal across otherwise identical binaries. Issuing the next live
row's window before the current row's FMAs, followed by `sched_barrier(0)`,
spreads the loads through the previous row's FMA block and moves the waits to
row boundaries. It also lowered VGPR use. A two-row lookahead and the no-barrier
variant were measured and dropped. One split-H, `ch_per_lane = 2` configuration
regressed; it is never a selected configuration. Hardwired (no knob) because no
configuration that dispatch selects preferred the old schedule.

## Lever 4 — `dot2` tap pairing (KEEP as knob; dispatch: KW ≥ 5)

7×7 depthwise is bound on the VALU issue rate, not memory. `v_dot2c_f32_bf16` /
`v_dot2c_f32_f16` compute `a.x*b.x + a.y*b.y + c` in one issue, so pairing taps
`(s, s+1)` halves the multiply-add issue count and removes the 16-to-32-bit
widening of every window value (the window stays packed). Window columns are
packed once per row as `(x[t], x[t-1])` (`v_perm_b32`); weights are packed once
in the prologue, with a zero partner for an odd KW.

**Odd-KW tail pair (correctness fix, review round 1).** The first form fed the
last pair `(w[KW-1], 0)` the operand `(x[j], x[j-1])`, so the zero weight met
window column `j-1`, one column outside the receptive field (or a duplicate of
`x[0]`). Numerically harmless for finite data, but an Inf there gave
`0 * Inf = NaN` in a dX column whose true value is finite (or Inf). The tail now
reads `(x[j], 0)`, so the zero weight only ever meets a zero. Three forms were
tried, each verified and measured same-session against the first form:

| Form | ISA (7×7, W = 14 / 16) | Outcome |
| --- | --- | --- |
| `vec_pack(x[j], 0.0)` | one extra `v_perm_b32` per column per row; more VGPRs | REVERT — measurably slower on the W = 16 shape |
| zero-extend the 16-bit bits (`bitcast` → `zext` i32 → `vector.bitcast`) | no pack at all: `buffer_load_ushort` already zero-fills the high half; fewer `v_perm_b32` than the first form (full pairs only need `t >= 2`). But the raw columns stay live across the full-pair `fdot2`s, VGPR rises past 256 on W = 16 and extra `v_mov_b32` appear | REVERT |
| zero-extend + tail pair issued first in the tap loop | raw columns die before the full pairs; VGPRs back under 256, `v_mov_b32` back to the first form's count, fewer instructions than the first form | KEEP — at parity with the first form |

`tests/test_conv_dgrad_depthwise_windowed.py::TestNonFiniteGradients` injects
Inf into dY on the first, an inner and the last column and requires dX to be
finite exactly where the reference is; it fails on the first form (every odd-KW
dot2 case) and passes on the kept one. Parity configs 40 (KW = 1 with dot2: tail
only) and 41 (non-square 3×5 fp16 dot2 with an H split) cover the new paths.

This needed a new IR op, `arith.fdot2`, in both engines (Python `IRBuilder.fdot2`,
`lower_llvm._op_arith_fdot2`; C++ `ROCKE_OP_ARITH_FDOT2` appended to the opcode
enum so no existing opcode value moves, name/purity tables, `rocke_b_fdot2`, LLVM
handler, declaration-table entries at the same relative position as the Python
table). gfx942's backend cannot select `llvm.amdgcn.fdot2.f32.bf16`, so the knob
is gfx950-only in both validators. Neutral on memory-bound 3×3, where dispatch
leaves it off.

## Other levers

| Lever | Outcome |
| --- | --- |
| `block_w` dividing W (whole row up to 16 columns, else about 8) | KEEP — the non-dividing widths are always worse; full-width rows become viable once `dot2` lowers VGPR use |
| `ch_per_lane = 2` (dword channel pairs, `v_pk_fma_f32`) | KEEP for 3×3 with enough channels (memory-bound, wider accesses); loses on 7×7 (VGPR pressure, occupancy 1) and when C is small (idle lanes) |
| `ch_per_lane = 4` | never wins (VGPR) — kept legal, not selected |
| `block_h` split | KEEP for small grids (large H × small N·C); loses on the target shapes because a split block cannot prune padding rows at build time and re-reads the halo |
| `block_waves` | 1–4, chosen so `block_ch` does not exceed C by a whole wave |
| Unroll budget on large filters (31×31, 33×33) | KEEP balanced shrink — rows are first halved down to about one filter height; past that the larger of rows and `block_w` is halved (the first form halved rows only, down to 1, so each block re-read a whole filter-height halo for one output row). Faster on every large-filter shape tried, same unroll size, so no compile-time change |
| Matrix-core (Toeplitz) formulation | not attempted here — the remaining 7×7 gap vs. a matrix-core reference is structural (VALU issue rate); see the synthesis plan item for a Toeplitz MFMA kernel |

Open lever: with the odd-KW tail fixed, `dot2` on 3×3 (fp16, 1536 channels)
measured slightly ahead of the selected `ch_per_lane = 2` form. Dispatch still
gates `dot2` on KW ≥ 5; widening it needs the 3×3 cohort re-swept first.

Honest losses of the dispatch heuristic against the per-shape sweep optimum are
recorded with the measurements outside the repo; the largest is a 3×3 shape with
192 channels where `block_waves = 1` beats the selected `block_waves = 2`.

## ISA evidence (per lane, 7×7, W = 14, before → after)

- `v_cndmask_b32`: over a thousand → a few dozen.
- `buffer_load_ushort`: several hundred at `block_w = 8` → fewer at
  `block_w = 14` (window loads only).
- `s_waitcnt`: several hundred → about a hundred.
- multiply-add issue: `v_fmac_f32` per tap → `v_dot2c_f32_bf16` per tap pair.
- instructions per output pixel: less than half.
- no scratch, no VGPR/SGPR spills.

## Replay

```bash
cd rocke/library
export PYTHONPATH=<engine-build>/cpp/bindings:../platform/python:. ROCKE_CPP_STRICT=1

# correctness (adversarial shapes, Inf propagation, dispatch end-to-end),
# manifest rule bad == 0
python -m pytest -q tests/test_conv_dgrad_depthwise_windowed.py
python -m pytest -q tests/dispatch/test_grouped_conv_dgrad_depthwise_dispatch.py

# sweep (stream + windowed variants, each verified) for one depthwise layer
python -m benchmarks.common.benchmark_direct_conv --direction dgrad --verify \
    --N 128 --Hi 14 --Wi 14 --C 512 --K 512 --groups 512 --Y 7 --X 7 --pH 3 --pW 3 --dtype bf16

# byte identity (Python vs C++ engine), scoped to the family
cd ../platform && ROCKE=$(pwd) PYTHONPATH=$(pwd)/python \
    python tools/check_byte_identity.py --only conv_direct_grouped
```

ISA: compile the dispatched spec with `rocke.compile_kernel` and disassemble the
HSACO with `llvm-objdump -d --mcpu=gfx950`; count `v_cndmask_b32`,
`buffer_load_*`, `s_waitcnt`, `v_fmac_f32` / `v_dot2c_*` and check the
`.vgpr_spill_count` note (`llvm-readelf --notes`).

ABI note: the kernel takes only the six-argument prefix
`(A, B, D, A_bytes, B_bytes, D_bytes)` shared by every conv-grouped candidate;
the implicit-GEMM dgrad kernel appends two more. A launcher sizes the argument
list from the kernel, not from `CONV_GROUPED_ABI_VERSION`.

Test isolation note: the GPU test module imports torch (when installed) before
rocke's launcher so only torch's bundled HIP runtime is loaded; otherwise later
torch-based tests in the same pytest process report no GPUs.

Toolchain note: measured with the LLVM 20 comgr shipped with ROCm 7.1 (the
`llvm22` flavor cannot run on that toolchain); the byte-identity gate was run
for llvm20, llvm22 and llvm23.
