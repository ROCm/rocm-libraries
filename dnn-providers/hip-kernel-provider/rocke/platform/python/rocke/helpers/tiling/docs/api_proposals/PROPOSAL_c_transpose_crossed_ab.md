# PROPOSAL: `c_transpose` — declarative crossed A↔B path (lane-major flip for a coalesced store)

- **Status:** proposed
- **Raised while:** authoring the CRC interleaved GEMM (`kernels/tiling_gemm_crc_demo.py`), `ab_swap` knob —
  the crossed path measured faster than the direct path by making the M-contiguous C store wave-coalesced
  (the direct path emits ~`lanes_N/lanes_M` = 4× more store transactions).
- **Extends SOT:** `tiling_api_surface.md` §6/§9 (TileMma; `c_transpose` is listed **PLANNED / Part D**) +
  `tiling_interleaving_design.md` §7 "C-store coalescing — the lane-major axis".

## Friction (what's awkward today)

The MFMA fixes which C axis is lane-major (consecutive lanes) — for the 16×16 atom, **N is lane-major, M is
block-major** (`lane(M,N)=16·(M//16)+(N//4)`). So a **col-major (M-contiguous) C output** stores per-lane-wide
but **not** wave-coalesced (~`lanes_N/lanes_M` = 4× more store transactions, §7). The fix is to put M on the
lane-major axis by running the machine crossed (emit `Cᵀ`), but there is no knob for it — the author must
hand-plumb three coupled changes and get them all consistent:

```python
# today (awkward) — the "ab_swap" crossed path, hand-wired in build_crc_gemm:
mma = TileMma((warp_n, warp_m, tile_k), a="f16", b="f16", c="f32", target=arch)  # 1. swap M<->N in the shape
...
acc = mma(b_frag, a_frag, acc)          # 2. feed B into the A-slot, A into the B-slot (M<->N relabel)
...
store_fragment(b, c_ptr, make_window(c_td.permute([1,0]), (n0, m0)), acc, lane)  # 3. store through the (N,M) view
# + the golden/readback must account for the transpose; a stray transform_fragment to a hand-authored
#   M-inner crossed store desc came back cross_lane (changed lane ownership) and had to be dropped.
```

Three places must agree (shape swap, slot feed, permuted store desc); any one wrong is silently wrong-until-
bit-exact. It is exactly the kind of coupled boilerplate the tiling model should express in one place.

## Proposed addition

A `c_transpose: bool = False` knob on `TileMma` (lives in `mma/mma_operation.py`). When set, the object runs
the machine crossed internally — swaps the M/N atom-grid roles, feeds the operands into the transposed slots,
and exposes `c_desc` as `(N,M)` — so the author writes the **normal** `mma()` + `store_fragment` and gets a
`Cᵀ`-native accumulator whose store is coalesced for a col-major (M-contiguous) output. It is a **relabel-tier**
transform (register-identity), not cross-lane.

```python
# proposed API
mma = TileMma((M, N, K), a="f16", b="f16", c="f32", target=arch, c_transpose=True)
acc = mma(a_frag, b_frag, acc)                 # normal call; object handles the crossed feed
store_fragment(b, c_ptr, make_window(mma.c_desc, (m0, n0)), acc, lane)   # c_desc already (N,M)-oriented
```

## Example — before → after

```python
# before: swap the TileMma shape, feed B->A-slot/A->B-slot, store through c_td.permute([1,0]) — 3 coupled edits
# after:  TileMma(..., c_transpose=True); mma(a,b,acc); store_fragment(..., mma.c_desc, ...)  — 1 declarative knob
```

## Soundness / perf caveats

- **Must stay relabel-tier / bit-exact.** The crossed path is a free M↔N relabel (§8 symmetry-2) + a `(N,M)`
  descriptor view — register-identity, no cross-lane. Validate `classify_transform` stays `reorder`/`relabel`
  (a naive `transform_fragment` to a hand-authored crossed store desc goes **cross_lane** — do NOT lower it
  that way; the point is to store the native crossed C directly). Gate on `max_abs_diff==0.0`.
- **Only helps when the output major is the block-major axis** (e.g. col-major/M-contiguous C on the 16×16
  atom). For a row-major (N-contiguous) C the direct path already coalesces — `c_transpose=True` would be the
  pessimal choice. The picker/author must select it by output major (SOT §7).
- **Perf is measured, not assumed:** the C store is an epilogue often hidden in the MFMA shadow; the speedup
  is realized at device-filling sizes. Confirm per case (it can be a no-op at small/occupancy-bound sizes).
- **Arch/atom scope:** derived for the MmaDim-16 (16×16) atom; 32×32 accumulator crossing is out of scope
  (cross-lane, §7).

## Notes

- Relation to existing verbs: subsumes the manual `ab_swap` plumbing in `tiling_gemm_crc_demo.py`; complements
  the C-shuffle (§7) — `c_transpose` picks WHICH axis is lane-major, the C-shuffle orders within it.
- Open question: expose as `TileMma(c_transpose=)` (structural) vs a `mma_layout`/picker flag; former is
  simplest and matches the §9 PLANNED entry.
- Also surfaced (separate proposals, not here): `mma_layout=interleaved()` factory; `space="lds"` staging
  desc; `mac_prio` knob; the `as_u8_buffer` C-contiguity requirement for col-major host arrays.
