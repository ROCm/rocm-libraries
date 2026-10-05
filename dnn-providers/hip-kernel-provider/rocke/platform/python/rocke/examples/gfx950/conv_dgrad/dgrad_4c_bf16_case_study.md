<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# 4c dgrad on the batched 4x4x4 MFMA — gfx950 case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
This study records the mechanism, the levers that were swept, the keep/revert
decisions and the commands to replay them.

## What was investigated

Grouped backward-data with four channels per group (`cpg = kpg = 4`), bf16 and
fp16, stride 1, 3x3 "same" padding. Before this change the only direct dgrad
path for that shape was the generic `DirectConvSpec` pipeline
(`make_dgrad_fprop_spec`): a weight transpose pre-pass, then the generic
direct-MFMA kernel on `(dY, W_T)`, which uses the `16x16x16` atom with one wave
per group.

## Finding 1: at cpg = 4 the 16x16x16 atom does one sixteenth useful work

The generic kernel maps one group onto a `16x16x16` MFMA. With four input and
four output channels per group only a 4x4 corner of the M x K operand is real:
the other rows and K lanes are masked to zero (`v_cndmask` on every fragment)
and only 16 of 64 lanes store. Hardware counters on the generic kernel show the
MFMA pipe as the dominant busy unit even though almost all of that work is
padding.

The `DirectConv4cSpec` kernel already solved this for fprop: the wave64
`4x4x4` MFMA computes sixteen independent 4x4x4 products, so one wave covers
sixteen groups with no wasted lanes. It was fp16-only because the IR had no
bf16 twin of the atom.

## Change 1: `mfma_f32_4x4x4_bf16`

`llvm.amdgcn.mfma.f32.4x4x4bf16.1k` exists on CDNA2+ and selects
`v_mfma_f32_4x4x4_16b_bf16` on gfx950. It follows the same `_1k` convention as
`mfma_f32_16x16x16_bf16`: the IR operands are `<4 x bfloat>`, bitcast to
`<4 x i16>` at the intrinsic boundary.

Wired in both engines:

| Piece | Python | C++ |
| --- | --- | --- |
| builder method | `IRBuilder.mfma_f32_4x4x4_bf16` (`core/ir.py`) | `rocke_b_mfma_f32_4x4x4_bf16` (`include/rocke/ir.h`, `core/ir/ir_tile.cpp`) |
| accumulator fragment length | `_MMA_FRAGMENT_INFO` (`core/arch/target.py`) | `rocke_ati_mma_frag` (`core/arch/data.cpp`) |
| intrinsic declaration | `_INTRINSIC_DECLS` (`core/lower_llvm.py`) | `core/lower_llvm/data.cpp` (same table position) |
| LLVM lowering | `_op_tile_mfma_f32_4x4x4_bf16` | `MFMA_SPECS` row with `bitcast_to = "<4 x i16>"` (`core/lower_llvm/mma.cpp`) |
| HIP lowering | `_op_tile_mfma_f32_4x4x4_bf16` (`core/lower_hip.py`) | — (the C++ HIP path has no per-atom MFMA handlers) |

The op goes through the neutral `tile.mma` op with an `op_id` attribute, so no
new opcode enumerator was needed.

The lane map was pinned on hardware before relying on it:
`platform/tests/core/test_mfma_4x4x4_numerics.py` issues one MFMA per dtype and
checks lane `l` (batch `l // 4`, `i = l % 4`): operand `a` is row `i` of
`A_batch`, operand `b` is column `i` of `B_batch`, result slot `m` is
`D_batch[m][i]`. The f16 and bf16 atoms agree.

## Change 2: bf16 in the 4c kernel

`build_direct_conv_4c` now takes its I/O type from `problem.dtype`
(`_io_type`, `_buf_load_vN`, `_buf_store_vN`, `_mfma(b, dtype, "4x4x4", ...)`)
and the kernel name gains a `_bf16` suffix for bf16. The fp16 emission is
unchanged byte for byte (the representative-IR golden for the existing fp16 4c
cases did not move). The C++ mirror carries `io_type` / `is_bf16` in
`rocke_dconv_4c_ctx_t` and branches at the same three emission points.

## Change 3: the 4c dgrad entry

For stride 1, same padding and a 1x1 or 3x3 filter,

```text
dX = conv(dY, W_T),   W_T[c, r', s', k] = W[k, KH-1-r', KW-1-s', c]  (per group),  PAD' = KH-1-PAD
```

so the 4c kernel runs unchanged on `(dY, W_T)`. `make_dgrad_4c_spec` builds the
transposed spec, `build_direct_4c_dgrad` returns `(transpose_kernel,
main_kernel)` (the transpose is the existing
`build_direct_transpose_weights_dgrad`), `direct_4c_dgrad_launch` gives both
launch geometries and the workspace size, and `dgrad_4c_spec_for_problem` is
the dispatch hook (returns `None` when the 4c path does not apply). The 4c
kernel keeps `KH` accumulator slots and one weight fragment per tap in
registers, which is why only `KH in (1, 3)` qualifies.

## Step 0: knob sweep

Knobs: `block_q in {4, 8, 16, 32}` (multiple of 4; the C++ engine bounds
`block_q / 4` by its tile array) and `block_groups in {16, 32, 64}` (one wave
per 16 groups, `groups % block_groups == 0`).

- `block_q = 4` was best or tied on every cpg = 4 shape swept (large and small
  spatial extents, batch 1 and large batch). Larger `block_q` raises live
  accumulators and the unrolled-row code size; the ISA shows VGPR + AGPR use
  growing with `block_q` and the MFMA count per wave scaling with it, with no
  gain in reuse because each q-tile reloads its own input windows.
- `block_groups = 16` and `32` tie; `64` (four waves) was never better and lost
  on one shape. **Kept:** `block_q = 4`, `block_groups = 16`
  (`DGRAD_4C_DEFAULT_BLOCK_Q` / `DGRAD_4C_DEFAULT_BLOCK_GROUPS`); the hook falls
  back to `block_groups = 16` when `groups` is not a multiple of the default.

## Why it is faster (ISA)

Same problem, generic kernel (`block_q 16, block_groups 4`) versus 4c
(`block_q 4, block_groups 16`), per wave:

| | generic | 4c |
| --- | --- | --- |
| MFMA | `v_mfma_f32_16x16x16_bf16` | `v_mfma_f32_4x4x4_16b_bf16` |
| MFMA instructions per wave | equal counts | equal counts |
| waves in the grid | more than four times as many | baseline |
| LDS / barriers | `ds_write_b64`, `ds_read_b64`, `ds_read2_b64`, `s_barrier` per row | none |
| registers | higher VGPR + AGPR | lower VGPR + AGPR, no spills on either |

Both kernels issue the same number of MFMAs per wave, but the 4c grid needs
fewer than a quarter of the waves for the same output, and each 4x4x4 MFMA is a
cheaper instruction than a 16x16x16 one. The counters agree: the MFMA busy
fraction drops from the dominant unit to a minor one, and memory fetch drops
because a 4c workgroup reads 16 groups x 4 channels = 128 contiguous bytes per
pixel instead of 32.

## What still separates it from a single-kernel implementation

Read off the 4c ISA; these are the next levers, not done here:

1. **Weight pre-pass.** The transpose kernel plus its launch gap is still on
   the critical path. Next stage: read `W` with flipped, transposed addressing
   in the 4c prologue (the weights already live in registers for the whole
   kernel).
2. **Input re-loads.** Each output row issues `KW` separate
   `buffer_load_dwordx2` per q-tile — one per filter column — although the
   windows overlap by `KW - 1` columns. A lane-shift (DPP / `ds_bpermute`) or an
   LDS row would load each input once.
3. **Per-load OOB select.** Every input load is followed by a
   `v_cndmask_b32` pair to zero padding; a descriptor-clamped buffer load
   (out-of-range offset returns zero) would remove them.
4. **Fully unrolled H loop.** Code size grows linearly with `H`; a runtime row
   loop with a compile-time accumulator ring would shrink it.
5. **8-byte stores from the MFMA layout.** An LDS-staged 16-byte store would
   cut store instructions.

## Correctness gates

- `platform/tests/core/test_mfma_4x4x4_numerics.py` — on-device lane map +
  numerics for both atoms, LLVM and HIP lowering text.
- `library/tests/test_conv_dgrad_4c.py` — spec plumbing, emitted atom per dtype
  and arch, bf16 4c fprop, 4c dgrad fp16/bf16 on odd H/W, N = 1, W not a
  multiple of `block_q`, the full `block_q x block_groups` grid, and 1x1. Rule:
  an element is bad unless `|out - ref| <= tol + tol * |ref|`, `tol = 1e-2`
  (written that way so a NaN, i.e. an unwritten element of the NaN-filled
  output, counts as bad); zero bad elements required.
- Filter taps are bounded at `KH*KW <= DCONV4C_MAX_TAPS` (Python) /
  `ROCKE_DCONV4C_MAX_TAPS` (C++, which sizes the builder's per-tap `weights[]`
  and `s_consts[]` arrays), checked in `is_valid_spec_4c` on both engines with
  the same reason text, so a directly built 5x5 spec is rejected instead of
  overflowing the C++ arrays. The C++ `is_valid_spec_4c` also gained the
  `stride > 1` reject the Python one already had. LDS weight staging stays
  bounded at `DCONV4C_MAX_WL_PASSES` / `ROCKE_DCONV4C_MAX_WL_PASSES`; with
  the tap bound in place no legal spec reaches it.
- Public dual-engine entry: `rocke.core.backend.lower_conv_direct_grouped`
  flattens the spec for the C++ binding. The flattened dict must carry every
  kernel-shaping field — the problem `dtype` and the 4c
  `dgrad_fused_weights` / `dgrad_weights_lds` knobs — and the binding
  (`fill_direct_conv_problem`, `dg4_build_spec`) must read them. The first
  version of this change wired the knobs only into the binding, so the
  default cpp backend silently emitted the fp16, non-fused kernel for bf16
  and fused specs; the byte-identity gate did not see it because it drives
  the `_emit.c` parity pair, not this path.
  `library/tests/test_conv_direct_grouped_backend_parity.py` runs
  `backend="both"` over 4c fp16/bf16 fprop, 4c dgrad with and without the
  fused transform and LDS staging (gfx950 and gfx942), and 16c fp16/bf16, and
  checks the 5x5 reject on both engines. Removing the forwarded fields makes
  it fail with `BackendMismatch`.
- Byte-identity: `library/tests/parity/conv_direct_grouped_emit.{py,c}` configs
  42–44 (bf16 4c on gfx950 / gfx942, two-wave dgrad shape) and two bf16 4c
  cases in the representative-IR golden.

## Replay

```bash
cd rocke/library
# sweep (4c only), verify every config
python benchmarks/common/benchmark_direct_conv.py --direction dgrad \
    --N 8 --Hi 56 --Wi 56 --C 128 --K 128 --groups 32 --dtype bf16 \
    --dgrad-family 4c --verify --jobs 8
# generic pipeline on the same shape, for the same-session A/B
python benchmarks/common/benchmark_direct_conv.py --direction dgrad \
    --N 8 --Hi 56 --Wi 56 --C 128 --K 128 --groups 32 --dtype bf16 \
    --dgrad-family generic --verify --jobs 8
# tests
python -m pytest tests/test_conv_dgrad_4c.py tests/test_conv_direct_grouped_backend_parity.py
cd ../platform && python -m pytest tests/core/test_mfma_4x4x4_numerics.py
# private scratch + build root: run_diff.py compiles the C emitters under
# $TMPDIR, which concurrent worktrees otherwise share
TMPDIR=<private dir> python tools/check_byte_identity.py \
    --only conv_direct_grouped,target_intrinsics --build-root <private dir>
```

Short kernels should be timed from `rocprofv3 --kernel-trace` rather than
the host-side event timer; compare only within one session.

## Follow-up

The weight-transpose pre-pass listed above as remaining work is removed by the
fused weight transform (`dgrad_fused_weights` / `dgrad_weights_lds`), now the
default of `dgrad_4c_spec_for_problem`; see
`dgrad_fused_weights_case_study.md` in this folder.

## In production dispatch

The grouped-convolution dispatcher selects this kernel (fused weights,
LDS-staged on gfx950) for `cpg == kpg == 4` with a 1x1 or 3x3 filter and
`groups % 16 == 0`, as the `"4c"` variant of `direct_mfma_conv_dgrad`, but only
when the grid at the default `block_q` / `block_groups` has enough workgroups
to fill the device: one wave is one workgroup and the kernel streams whole
columns without H tiling, so on one or two images the generic kernel (which
tiles H) is faster. The floor and how it was measured are in
`grouped_direct_dgrad_dispatch_case_study.md`.
