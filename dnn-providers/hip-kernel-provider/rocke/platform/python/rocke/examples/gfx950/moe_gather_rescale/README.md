# MoE activation gather/rescale prologue

The kernel that makes a serving stack's activations legal input for the fused MoE
mega-kernel. In one pass over the activation it gathers the token rows that route
to each expert block, **restates them under a single block-uniform activation
scale**, re-rounds to fp8, and writes the `A` matrix, the scale array, and the
scatter metadata that `examples/gfx950/fused_mega_moe/` consumes.

```
per-token-quantized Aq + AqScale     ->  gather rows of one expert block
routing (SortedIds, TopkWeights)     ->  one scale per (block, 128-group)
                                     ->  rescale each row by s_row / s_block
                                     ->  A, AScale, SortedTokenIds, SortedWeights
```

> For the math, the data layout, and the per-threadgroup steps, see
> [`ALGORITHM.md`](ALGORITHM.md). This file is what it is and how to run it.

## Why it exists

The mega-kernel's dequant fold applies **one per-lane scalar, indexed by the row
of the A fragment, to accumulator slots that belong to several different output
rows**. Only an activation scale that is uniform across the rows of a block
survives that fold.

A per-token scale — which is exactly what an inference framework produces, since
it quantizes before it routes — does not. And nothing rejects it: the fold reads a
legal address and multiplies by another row's scale, so the result is **silently
wrong rather than an error**. That is the failure mode this kernel removes. It
makes the conforming layout cheap enough to produce that a caller can satisfy the
requirement by construction instead of by remembering it.

The restatement is value-preserving by design: `A_q · s_token == A_q' · s_block`,
so the only thing that changes is where the fp8 rounding lands. The same job done
in the framework cannot stay in fp8 — it has to widen to f32 to apply the rescale,
turning one streaming pass into several passes over a 4x larger temporary. Here
the widening lives in registers between a vector load and a vector store.

## What it produces

| output | shape | notes |
|---|---|---|
| `A` | `[n_blocks·tile_m, hidden]` fp8 e4m3 | gathered, padded, restated under the block scale |
| `AScale` | `[n_blocks·tile_m, hidden/128]` f32 | the block scale, broadcast to **every** row including pads, because that is the form the fold reads |
| `SortedTokenIds` | `[n_blocks·tile_m]` i32 | token id per row, `-1` on a pad row |
| `SortedWeights` | `[n_blocks·tile_m]` f32 | the routing weight for that `(token, slot)`, `0` on a pad row |

Pad rows are written as exact zeros. That is a requirement, not tidiness: a pad
row's activations still enter the block-wide amax the mega-kernel reduces to set
the intermediate's scale, so anything non-zero there corrupts a **real** row's
output.

## The knobs

`MoeGatherRescaleSpec` (`instances/common/moe_gather_rescale_a.py`):

| field | default | what it constrains |
|---|--:|---|
| `tile_m` | 16 | rows per expert block. **Must equal the consuming MoE kernel's `tile_m`** — it is the row blocking the scale is uniform over, and a mismatch is silently wrong in exactly the same way a per-token scale is. Take it from the same spec you launch the GEMM with. |
| `max_n_hb` | 32 | upper bound on `hidden/128`, used only to size the LDS scratch statically. `hidden` itself stays a runtime argument. |
| `block_size` | 256 | threads per workgroup (4 waves). |
| `vec` | 8 | fp8 elements per thread per load/store. Must divide 128 so a vector cannot straddle two scale groups, which is what lets one ratio cover the whole vector. Checked in `__post_init__`. |

Caller obligations the kernel does not check: `hidden` must be a multiple of 128,
and `hidden/128` must not exceed `max_n_hb`.

## How to run it

`verify_gather_rescale.py` builds the kernel, launches it once, and compares all
four outputs against a numpy model, plus the round-trip invariant the rescale is
supposed to preserve. It needs a gfx950 GPU and numpy, and it must run on an
interpreter where **torch is not importable** — a resident torch changes which
LLVM Comgr resolves, and the compile then stops finishing rather than failing.
The script asserts this so the mistake is an error instead of a hang.

```bash
cd <repo>/dnn-providers/hip-kernel-provider/rocke/platform/python
NOTORCH=python3   # must NOT be able to import torch

PYTHONPATH=$(pwd) $NOTORCH -u \
    rocke/examples/gfx950/moe_gather_rescale/verify_gather_rescale.py

# the decode shape, a wider tile, and a single-token case
... verify_gather_rescale.py --tokens 64 --hidden 2048 --experts 128 --topk 8
... verify_gather_rescale.py --tile-m 32 --tokens 40
... verify_gather_rescale.py --tokens 1 --experts 8 --topk 2 --hidden 4096
```

Expected output:

```
  PASS  SortedTokenIds exact
  PASS  SortedWeights exact
  PASS  AScale exact (block scale on every row, pads included)
  PASS  A bytes (ignoring +-0 encoding)  0 real / 0 total differing of 4128768
  PASS  pad rows dequantize to zero
  dequant round-trip: max 2.674e-02  mean 1.105e-04  ...
VERDICT: PROLOGUE OK
```

`A` is compared **byte for byte**, not within a tolerance, which is the only
check strong enough to catch the rounding bug described below. The one exception
is the `±0` encoding: e4m3 is sign-magnitude, so a pad row whose source byte was
negative rescales to `-0.0` (`0x80`) rather than `0x00`. Both dequantize to zero,
which is all the consumer's amax cares about, so the verifier counts those
separately and requires zero *other* disagreements.

The round-trip residual is not an error budget — it is the precision a row
genuinely loses when it is restated under a scale larger than its own. It is
`0.000e+00` when a block holds one real row (the block scale *is* that row's
scale, so the ratio is exactly 1), and grows with the spread of per-token
magnitudes inside a block. That trade is the price of the layout the mega-kernel
requires; see [`ALGORITHM.md`](ALGORITHM.md) §6.

## The rounding trap worth knowing about

The per-row ratio `s_row / s_block` is computed with an **exact divide**, not a
fast hardware reciprocal. A 1-ULP error in that ratio flips the fp8 rounding of
elements sitting near a tie, which surfaces as a handful of mismatched bytes
across millions of elements — small enough to look like noise and pass any
tolerance-based check, which is exactly why the verifier compares bytes.

The exact divide costs nothing worth counting: the ratio is computed **once per
(row, 128-element group)** and then reused across all 128 elements of the group,
so a division that is an order of magnitude more expensive than a reciprocal is
still amortized to nothing. See [`ALGORITHM.md`](ALGORITHM.md) §5.

## File map

| path | purpose |
|---|---|
| `README.md` | this document |
| `ALGORITHM.md` | the math, the data layout, and the per-threadgroup steps |
| `verify_gather_rescale.py` | torch-free numpy correctness driver |
| `../../../instances/common/moe_gather_rescale_a.py` | the kernel builder (`build_moe_gather_rescale_a`) |
| `../fused_mega_moe/` | the consumer, and the fold that forces this layout |
