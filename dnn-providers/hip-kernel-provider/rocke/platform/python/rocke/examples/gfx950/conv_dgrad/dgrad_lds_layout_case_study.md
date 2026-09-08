<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# dgrad LDS layout — gfx950 case study

Companion to [`../conv_wgrad/wgrad_lds_layout_case_study.md`](../conv_wgrad/wgrad_lds_layout_case_study.md).
Read that one first: it establishes why the M-outer LDS tile forces a transpose
*on store* and why the row stride makes those stores bank-degenerate. This study
covers only what is **different** about backward-data.

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
What is recorded here is mechanism, instruction counts and the replay path.

## What was investigated

Whether the K-outer LDS tile + transpose-read fetch that landed for wgrad
transfers to dgrad, and if so, to which operands.

The answer is **B only**, and the asymmetry is the whole finding. Applying the
wgrad change symmetrically would have been the obvious move and would have made
dgrad slower.

## Finding 1: only one of the two operands pays the scatter

| Operand | Tensor / layout | GEMM reduction axis | Loader | Transpose on store? |
| --- | --- | --- | --- | --- |
| A | `dY`, NHWK | `k_out`, **stride-1** | `vector_axis="col"` | No — already one wide `smem_store_vN` |
| B | `W`, KYXC | free axis `c` is stride-1 | `vector_axis="row"` | **Yes** — per-element `ds_write_b16` |

dgrad's A operand is the one wgrad does not have: `k_out` is innermost in NHWK
*and* is the GEMM reduction axis, so the global vector already runs along K and
the tile lands M-outer with a single wide store. Its fragment read is already
conflict-free via `lds_k_pad`, which is derived for a row step of 1 — the read
path — and works exactly as intended here.

Flipping A anyway would put the global vector along `m = (n, hi, wi)`, which is
stride `K` in NHWK. That destroys global coalescing to remove a write-side cost
that does not exist. The predicate is therefore asymmetric by construction:
`DgradConvSpec.default_lds_k_outer` takes `dtype_b` and `warp_tile_n` and never
their A-side counterparts.

## Finding 2: the saving is proportional to the B load width, so `cpg` gates it

K-outer's store-side win is "collapse `load_vec` narrow writes into one wide
one". When the per-group channel run `cpg = C / groups` is odd, `choose_vec`
cannot find a width > 1, the loader already falls back to `vector_axis="col"`,
and there is no scatter left to remove. What remains is only the read-side cost:
`n / 4` `ds_read_b64_tr_b16` where M-outer issued a single `smem_load_vN`, i.e.
**+1** instruction per fragment on the `n = 8` atoms and **zero** on the `n = 4`
atoms.

So the predicate additionally keys on `cpg` — the only shape-dependent term in
either backward deducer. A grouped problem with an odd channel run correctly
declines K-outer, which
`library/tests/dispatch/test_grouped_conv_wgrad_dispatch.py::TestGroupedDgradDispatch`
pins from the dispatch side.

## Finding 3: two pipelines had to be gated, for different reasons

Neither is a limitation of the lane mapping; both are loaders that write the
tile in a layout the compute phase no longer expects.

- **`async_dma`** — rejected under `lds_k_outer`. The tilde builder has no
  direct global→LDS path at all, so unlike wgrad (where `async_dma` *requires*
  K-outer, because the packed lane-contiguous tile the intrinsic writes **is**
  the K-outer layout) there is nothing to pair it with.
- **`pipeline="wavelet"`** — rejected under `lds_k_outer`. `build_wavelet_loaders`
  pins the B tile to `(block_n, block_k)` and takes the *unswapped* descriptor,
  so it would write M-outer into a K-outer allocation: wrong row stride for
  every element, and a store past `B_smem` whenever `tile_n > tile_k` — into the
  cshuffle C region under `_no_alias`, or off the end of the pool. Silent
  wrongness, not a crash. Gated rather than fixed because the loader is shared
  with forward conv and needs its own verification.

## The correctness argument

K-outer is a pure re-layout: the same values reach the same MFMA lanes by a
different route. So the bar is not "close enough" but **bitwise identical** to
the M-outer path, and that is what `TestConvDgradLdsKOuter` asserts with
`torch.equal`. A tolerance-based comparison would hide a lane-mapping error that
happens to be small.

The one exception is `split_k > 1`, which uses the atomic epilogue: its
accumulation order is not deterministic and the M-outer kernel does not
reproduce *itself* bitwise. That case compares against a torch reference within
tolerance instead. Do not "fix" the others to match it.

Byte-identity between the two engines is a separate and weaker claim: it proves
Python and C++ agree, never that the formula is right. Two engines agreeing on
the same wrong lane map is exactly the failure the gfx1250 study documents.

## Replay

From `platform/`, with `PYTHONPATH=$(pwd)/python`.

Correctness — the bitwise A/B, and the full suite:

```bash
python3 -m pytest tests/instances/test_conv_dgrad_correctness.py -q -rs -k KOuter
python3 -m pytest tests/instances/test_conv_dgrad_correctness.py -q
```

All six `TestConvDgradLdsKOuter` cases must **run** on gfx950 — wave64,
`family="mma"`, the 32x32x16 and 16x16x16 atoms, strided (tilde), and a
non-tile-aligned K. Nothing should skip for arch reasons. A class whose subTests
all skip still reports `passed`.

Byte identity, both flavors (the merge gate for the two-engine mirror):

```bash
export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python3 tools/check_byte_identity.py --only conv
ROCKE_LLVM_FLAVOR=llvm22 python3 tools/check_byte_identity.py --only conv
```

Dispatch-side policy (CPU-only, no GPU):

```bash
python3 -m pytest library/tests/dispatch/test_grouped_conv_wgrad_dispatch.py -q \
    -k Dgrad
```

Step 0 lever sweep. Shapes come from `--miopen-cmd` / `--miopen-file`; `-F 2`
selects the backward-data direction in the MIOpen driver grammar:

```bash
python3 python/rocke/benchmark/benchmark_implicit_gemm_conv.py \
    --direction dgrad --arch gfx950 --dtype bf16 \
    --miopen-cmd "./MIOpenDriver convbfp16 -n 8 -c 128 -H 32 -W 32 -k 128 \
        -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -in_layout=NHWC" \
    --jobs 32 --sample 0.05 --seed 0 --top 5 --warmup 3 --iters 10 \
    --csv dgrad_sweep.csv
```

As with wgrad there is **no `--lds-k-outer` flag**: the layout is deduced by
`DgradConvSpec.default_lds_k_outer`, which the sweep driver and dispatch both
call. Keep `--csv` output outside the git work tree.

## Config table

| Axis | Value | Note |
| --- | --- | --- |
| tile (m, n, k) | 64, 64, 64 | same as wgrad |
| warp (m, n) | 2, 2 | 256 threads at wave64 |
| atom edge | 32 (also validated at 16) | `n = 8` and `n = 4` fragments |
| pipeline | `mem` | `compv3`/`compv4` are rejected for dgrad |
| epilogue | `default` | dgrad dispatches internally on `needs_atomic` |
| split-K | 1 | no CK auto-formula; see below |
| `lds_k_outer` | deduced, B tile only | declines on odd `cpg` |

Split-K is pinned to 1 in dispatch rather than auto-resolved. The CK formula
wgrad uses keys on its lopsided `N*Ho*Wo` reduction; dgrad's reduction is
`Y*X*K`, which is not that shape, so borrowing the formula would be unjustified.

## Still open

- **Teach the wavelet loader K-outer** instead of gating it. Shared with forward
  conv, so it needs its own verification.
- **No gfx942 or gfx1250 dgrad dispatch candidate.** gfx942 lacks the 32x32x16
  atom; the gfx1250 wave32 path is reachable through the sweep driver but has no
  dispatch-level dual-engine test. See the gfx1250 study.
- **A dgrad parity config for the gfx1250 wave32 path**, to match wgrad's
  config 15. dgrad's wave32 path is currently exercised only through the shared
  compute phase.
