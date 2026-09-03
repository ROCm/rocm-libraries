# rocKE Tiling Primitives -- API Surface, Options & Composability

**Status:** reflects the BUILT M1 + wave-tile subtiling + clipping surface (95/95 tests, bit-exact on gfx90a).
**Companion to:** `tiling_design_proposal.md` (the why + architecture). This doc is the
**how-to-use** catalog: every API surface, its **default (MMA-driven) mode**, and its
**manual override**, with runnable examples that exercise the extents of the API.

> **Location:** `platform/python/rocke/helpers/tiling/` (package `rocke.helpers.tiling`).
> **Compliance:** the sibling `reference_docs/` holds internal artifacts -- scrub NPI /
> product / perf data before any Confluence publication. ASCII-only for clean conversion.

---

## 0. The one idea

**You state intent; the MMA object supplies the rest.** Every knob has a sensible default
that is *driven by the resolved MMA atom*, and every default is a value you can replace.
The spectrum runs from "say almost nothing" to "pin the exact backend intrinsic and
loop-nest order" -- with the *same* dataflow (`load -> mma -> store`) the whole way.

```
DEFAULT (MMA drives everything)                         FULL OVERRIDE (author pins everything)
|-----------------------------------------------------------------------------------------|
TileMma((16,16,16), a,b,c, target)          TileMma((64,64,32), a,b,c, target,
  -> atom = shape, single MMA, layouts,        tiling=Tiling(atom_shape="mfma_f32_16x16x16f16",
     wave_size, op_id, frag sizes all                             order="KNM"))
     resolved from traits                       -> pinned intrinsic by name, pinned 4x4x2
                                                   subtile grid + iteration order
```

---

## 1. The surface at a glance

| Surface | What it is | Constructed with `b`? | Default source |
|---|---|---|---|
| `make_tensor_desc(lengths, strides, dtype)` | ptr-free strided memory descriptor | no (IR-free) | strides = data layout; `lengths` = valid extent (the default clip) |
| `make_window(tensor, origin, bounds=None)` | positioned + bounded window | no | `bounds` = clip per axis (overrides the desc lengths); `None` = desc lengths |
| `make_tile_desc(shape=, thread_dist=, thread_tile=, ...)` | author a logical->register layout (quantity-major) | no | for MMA operands use `mma.a/b/c_desc` instead |
| `TileDesc(shape, layout)` | dtype-free logical->register layout value | no | `mma.a/b/c_desc`, self-composed, or `make_tile_desc(...)` |
| `make_fragment(tile_desc, dtype)` | realized per-lane registers | no (value filled by verbs) | dtype from the operand |
| `fill_fragment(b, frag, 0)` | element-wise materialize | yes (b-first) | -- |
| `load_fragment(b, ptr, window, desc, lane, *, pad=0, lds_swizzle=False)` | memory -> Fragment (zero-pads OOB) | yes | addressing from `desc.layout`; `pad` = clip fill (0); `lds_swizzle` = bank-swizzle POLICY (§5c) |
| `store_fragment(b, ptr, window, frag, lane, *, lds_swizzle=False)` | Fragment -> memory (drops OOB) | yes | cast frag dtype -> desc dtype; `lds_swizzle` = bank-swizzle POLICY (§5c) |
| `TileMma(shape, a,b,c, target, tiling=)` | intrinsic resolver + subtile driver | no | atom/layouts/op_id/wave_size from traits |
| `Tiling(atom_shape=, order=)` | the MMA object's knobs | no | atom=shape (single MMA), order="MNK" |
| `TilingGemmSpec(tile, atom, order, dtypes)` | instance knobs (spec->builder) | no | atom=None, order="MNK", f16->f32 |

Free `make_*` factories over thin value objects (`TensorDesc`/`TensorWindow` in `descriptors.py`;
`TileDesc`/`Fragment` in `fragments.py`); three verbs (`emit.py`); one driver (`mma/`). The
factories are the public surface -- you never call a method on the descriptor to make a window
(the descriptor is a pure memory layout; `ptr` binds at the verb, not on the descriptor).

---

## 2. The default dataflow (say the minimum)

The smallest complete kernel body. Everything not stated is resolved from the MMA atom.

```python
mma = TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")

acc = make_fragment(mma.c_desc, F32)          # desc + dtype -- both defaulted by the MMA object
fill_fragment(b, acc, 0)
for tile_k_base in range(0, K_LEN, 16):
    k = b.const_i32(tile_k_base)
    a_frag = load_fragment(b, a_ptr, make_window(a_td, (m, k)), mma.a_desc, lane)  # clips at the desc extent
    b_frag = load_fragment(b, b_ptr, make_window(b_td, (n, k)), mma.b_desc, lane)
    acc = mma(b, a_frag, b_frag, acc)
store_fragment(b, c_ptr, make_window(c_td, (m, n)), acc, lane)   # drops OOB
```
(The clip defaults to the `TensorDesc` `lengths`; for a full, tile-aligned matrix that is a
no-op, byte-identical to no-clip. Pass `bounds=` to override the clip per axis.)

What you did **not** have to say (all MMA-driven): the concrete intrinsic (`op_id`), the wave
size, the per-lane register counts, the lane->element mapping, the number of atoms, or the
accumulator width. See sec 8 for the full "you didn't have to say it" list.

---

## 3. `TensorDesc` -- ptr-free strided memory descriptor

Built with `make_tensor_desc(lengths, strides, dtype)`; `ptr` binds later at the verb. `lengths`
are the valid (compute) extent and double as the default clip bound; `strides` are the physical
layout (a stride-1 axis is the memory-contiguous one).

### Default: lengths = valid extent, strides = the data layout
```python
a_td = make_tensor_desc((M_LEN, K_LEN), strides=(K_LEN, 1), dtype=F16)   # A(M,K) row-major
```
- `dtype` is the element type of memory; `load_fragment` reads it, `store_fragment` casts to it.
- `strides` ARE the data layout -- one generic verb serves A/B/C; the operand difference is
  entirely the strides + the layout, never the verb.

### Override: express any data layout via strides (RCR shown)
```python
a_td = make_tensor_desc((M_LEN, K_LEN), (K_LEN, 1), F16)   # A row-major   (m,k) . (K,1)
b_td = make_tensor_desc((N_LEN, K_LEN), (1, N_LEN), F16)   # B col-ordered (n,k) . (1,N)
c_td = make_tensor_desc((M_LEN, N_LEN), (N_LEN, 1), F16)   # C row-major   (m,n) . (N,1)
```
A logical view onto a physically-transposed operand is a pure re-label: `a_td.permute([1, 0])`
(the 2-D swap) reads the same bytes with the axes reordered.

### Override: dtype is whatever the pointer is (no dtype baked into verbs)
```python
a_td = make_tensor_desc((M_LEN, K_LEN), (K_LEN, 1), BF16)  # bf16 in
c_td = make_tensor_desc((M_LEN, N_LEN), (N_LEN, 1), F32)   # f32 store (no narrowing cast)
```

### Reserved (post-M1)
- LDS-space descriptors for staging.
- per-load `coherency=` (param exists on the verbs; ignored in M1).

---

## 4. `TileDesc` -- dtype-free logical->register layout (comes from MANY places)

A `TileDesc` (shape + layout) is what the verbs and `Fragment` consume. Crucially, **a desc
is just a value -- it can come from several sources, all interchangeable at the verb.** The
MMA object is the common default, but it is not the only producer.

`TileDesc` is **dtype-free and memory-free** -- pure "where each element lives". The same
`TileDesc` is reusable across dtypes; the type is bound only when a `Fragment` is realized.

### Source 1 (default): straight from the MMA object
```python
a_desc = mma.a_desc     # wave (M, K) shape + resolved a_layout, ready for load_fragment
b_desc = mma.b_desc     # wave (N, K)
c_desc = mma.c_desc     # wave (M, N)  -- use for the accumulator Fragment
```
This is the strong default -- shape and layout both resolved from the atom; you author nothing.

### Source 2: author it yourself with `make_tile_desc` (the human-approachable surface)
For a NON-MMA tile (a plain data load, a prefetch/staging tile, a custom distribution) author
the layout directly as a **geometric table** -- one axes-ordered list per quantity -- and get a
ready `TileDesc` back. No raw encoding integers. See sec 4b for the full quantity glossary.
```python
a_desc = make_tile_desc(                    # a 256x32 DRAM load tile (4 M-waves x wave64)
    shape=[256, 32], thread_dist=[16, 4], wave_dist=[4, 1],
    thread_tile=[1, 8], block_repeat=[4, 1], wave_size=64,
)
```

### Source 3: self-composed (pair any shape with any layout value)
```python
a_desc = TileDesc((TILE_M, TILE_K), mma.a_layout)   # same result as mma.a_desc, built by hand
```
Use this when you already hold a `WarpDistributionEncoding` (e.g. `mma.a_layout`) and just want
to pair it with a shape.

### Source 4 (reserved): future primitives
Data-tile / LDS-staging profiles (post-M1) will emit their own `TileDesc` for non-MMA tiles
through the same interface -- so `load_fragment(b, ptr, window, desc, lane)` never changes; only
*where the desc came from* does.

> **The invariant:** verbs take a `TileDesc`; they neither know nor care whether it came from
> `mma.*_desc`, from `make_tile_desc(...)`, or from `TileDesc(shape, layout)`.

---

## 4b. `make_tile_desc` -- human-approachable tile authoring (the geometric table)

The anti-anti-pattern: instead of hand-writing raw `WarpDistributionEncoding` integer sequences,
author a distribution as a **struct-of-arrays** -- one axes-ordered list per geometric quantity,
so the COLUMNS are the logical matrix axes (column `i` = axis `i` in every list) and the ROWS are
the quantities. It reads like a table and returns a ready `TileDesc` in one call. Rank-agnostic
(N-D), no `M/N/K` labels.

```python
make_tile_desc(
    *, shape, thread_tile=None, thread_dist=None, thread_order=None, thread_broadcast=1,
    block_repeat=None, wave_dist=None, wave_order=None, wave_broadcast=1, wave_size,
) -> TileDesc
```

| Quantity | Per-axis meaning | Consumer |
|---|---|---|
| `shape` | the overall tile size per axis | -- |
| `thread_dist` | how the wave's LANES spread over each axis (product == `wave_size`) | lane (P) |
| `thread_tile` | contiguous elements each lane holds per axis (stride-1 axis = the vector) | inner register (Y) |
| `block_repeat` | the whole lane tile STAMPED as strided registers per axis | outer register (Y) |
| `wave_dist` | how the block's WAVES spread over each axis | wave (P) |
| `thread_order` | lane-carrying axes, fastest-moving axis RIGHT-MOST (default = axis order) | lane significance |
| `wave_order` | wave-carrying axes, fastest right-most (default = axis order) | wave significance |
| `thread_broadcast` | duplicate the tile across LANES: int (whole-tile) or `[size, count]` (positioned) | replication R (lane) |
| `wave_broadcast` | duplicate the tile across WAVES: int or `[size, count]` | replication R (wave) |
| `wave_size` | lanes per wave (64 CDNA / 32 RDNA) | -- |

**Invariants (fail-fast, in author vocabulary):** each axis factors as
`thread_dist * wave_dist * thread_tile * block_repeat == shape[axis]`; and
`product(thread_dist) * thread_broadcast == wave_size`. Omitted list quantities default to all-1s;
`thread_order`/`wave_order` default to axis order. REGISTER order is canonical (no knob):
`block_repeat` is the outer (major) register, `thread_tile` the inner stride-1 vector.

```python
# ck_tile MakeADramTileDistribution (M=256, K=32) -- the geometric table IS the input:
td = make_tile_desc(shape=[256, 32], thread_dist=[16, 4], wave_dist=[4, 1],
                    thread_tile=[1, 8], block_repeat=[4, 1], wave_size=64)

# a dense C accumulator (M capture on the first axis, N across lanes):
c  = make_tile_desc(shape=[16, 16], thread_dist=[4, 16], thread_tile=[4, 1], wave_size=64)

# lane duplication (RDNA3 half-wave): rows 0-15 duplicated onto lanes 16-31:
a  = make_tile_desc(shape=[16, 16], thread_dist=[16, 1], thread_tile=[1, 16],
                    wave_size=32, thread_broadcast=2)

# cross-axis lane significance: make the contiguous axis the major lane (atom wiring):
a  = make_tile_desc(shape=[16, 16], thread_dist=[16, 4], thread_tile=[1, 4],
                    thread_order=[1, 0], wave_size=64)
```

The returned `TileDesc` plugs straight into `load_fragment`/`store_fragment` like any other -- an
authored non-MMA distribution and an `mma.a_desc` are the same type at the verb.

### Mapping / reflection -- see what you authored (never decode integers)
```python
describe(td.layout)              # structured dict: lane/register counts, matrix sizes, provenance
render_forward_map(td.layout)    # ASCII lane x register -> (row, col)
render_inverse_map(td.layout)    # ASCII (row, col) -> (lane, register)
td.register_count                # per-lane register count implied by the layout
```

---

## 5. `Fragment` + the verbs

### Default: born from a TileDesc, materialized by fill/load
```python
accumulator = make_fragment(c_desc, F32)          # declare: (tile_desc, dtype); no registers yet
fill_fragment(b, accumulator, 0)                  # materialize with 0 (the accumulator identity)
a_frag = load_fragment(b, a_ptr, make_window(a_td, (m, k)), a_desc, lane)   # from memory
```
- dtype lives on the `Fragment` (D16); `store_fragment` casts fragment dtype -> desc dtype
  on the honest path only (identity, or f32->{f16,bf16}); anything else fails fast.

### Override: fragments can be ANY dtype
```python
acc_f32 = make_fragment(c_desc, F32)              # f32 accumulator (typical)
# a bf16 A operand fragment is just a different desc dtype + the same verb:
a_bf16 = make_tensor_desc((M_LEN, K_LEN), (K_LEN, 1), BF16)
a_frag = load_fragment(b, a_ptr, make_window(a_bf16, (m, k)), a_desc, lane)
```

### Reserved (post-M1)
- `fill_fragment(b, frag, value)` for `value != 0`; `coherency=` effect on load/store.

---

## 5b. Clipping / bounds -- partial tiles at M/N/K edges (Part C, BUILT)

rocWMMA limit #5: `load/store_matrix_sync` assume a full tile, so a matrix that is not a multiple
of the tile forces tensor padding or a separate edge kernel. Here clipping is **auto-enabled by a
window bound** -- no separate remainder path.

- The clip coordinate rides the **window** (per-tile), not the descriptor (shared). The upper clip
  defaults to the `TensorDesc` `lengths`; pass `bounds=` to override per axis.
- **Presence of a clip bound auto-enables it, default = ZERO-PAD:** an OOB load contributes 0, an
  OOB store is dropped. `load_fragment(..., pad=0)` is the default; `pad != 0` (constant fill) is
  reserved and raises today.
- **Per-axis, compile-time fast path:** an axis whose bound is a compile-time multiple of the tile
  can never overhang, so NO compare is emitted -- byte-identical to no-clip. A K-only-ragged GEMM
  checks K only.
- The predicate is on the **logical coordinate** (`origin + coord < bound`), per axis -- so it is
  correct even for a tile that overhangs mid-tensor (not just the contiguous tail).

```python
# ragged 255x255x255 over a 256^3 tile: the edge CTAs clip internally, zero-padded load / dropped store.
a_win = make_window(a_td, (m, k))                 # clip defaults to a_td.lengths (the compute extent)
a_win = make_window(a_td, (m, k), bounds=(M, K))  # or state the bound explicitly per axis
a_frag = load_fragment(b, a_ptr, a_win, mma.a_desc, lane)     # OOB -> 0
store_fragment(b, c_ptr, make_window(c_td, (m, n)), acc, lane)  # OOB writes dropped
```
Proven bit-exact on gfx90a: ragged 255^3 (OOB edge), 256x256x250 (K-only), and 250^3 into a
256-allocated tensor (clip INSIDE a valid space via leading dims -- masked cells hold real data yet
are excluded, and the C tail stays untouched). The reserved efficient lowering (C2) uses the
buffer-descriptor `num_records` path for wide vectorized tail loads.

---

## 5c. LDS bank-swizzle policy (Part C, BUILT)

`load_fragment`/`store_fragment` take `lds_swizzle`, a **customizable bank-swizzle POLICY** for LDS accesses
(no effect on a global `ptr`). It is `bool | Callable`:

- `False` (default) — no swizzle; the natural contiguous vector width is used.
- `True` — the built-in block-preserving swizzle (`_swizzle_lds_positions`).
- a **callable policy** `(builder, positions) -> positions` that remaps the LDS index. It declares its
  granularity via a `vw_elems` attribute (the block width, in elements, it keeps contiguous); the emit
  resolves the access width to `min(vw_elems, natural_run)` and **range-checks it to `[1, natural_run]`**
  (`_swizzle_vw`) — so a policy can only relocate whole blocks of its own granularity, never widen past what
  the layout is contiguous for. Store and read MUST use the same policy (a bijection → bit-exact).

Provided policies (`kernels/tiling_gemm_interleaved_demo.py`, built by `_bank_swizzle(width_elems)`):
`b32_swizzle` (b32-granular), `b64_swizzle` (b64-granular); alias `full_perm_swizzle = b32_swizzle`. These
de-alias the K-aliased interleaved coop store; narrower granularity → fewer bank conflicts but more
instructions.

```python
from ...kernels.tiling_gemm_interleaved_demo import b32_swizzle
store_fragment(b, lds_a, win, fa, tid, lds_swizzle=b32_swizzle)          # store with the policy...
fa = load_fragment(b, lds_a, win, a_desc, tid, lds_swizzle=b32_swizzle)  # ...SAME policy on read (bijection)
```

**Whether / which policy to apply is a decision, not a default** — a bank-conflict vs bandwidth vs
instruction-issue tradeoff. The model, the width ladder, and the binding-stage decision live in
`lds_banks.md` (measure; do not pick blind).

---

## 6. `TileMma` -- the resolver + subtile driver (where the defaults live)

`TileMma(shape, *, a, b, c, target, tiling=None)`. `shape` is the **wave tile**. Everything
below is resolved for you and is inspectable. The **intrinsic** SSOT is **`traits/data/mma_traits.json`**
(atom shapes + dtypes = the hardware truth; resolved automatically, never hand-edit it). Its **canonical**
A/B/C encodings are the **baseline/fallback** layout — a starting place, NOT a requirement: you are free to
hand-roll valid non-canonical (interleaved / custom) layouts (canonical is never required — `mma_is_machinery.md`).

### Default: single MMA, everything resolved from traits
```python
mma = TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")
mma.op_id       # 'mfma_f32_16x16x16f16'   (resolved intrinsic)
mma.atom_shape  # (16, 16, 16)
mma.subtiles    # (1, 1, 1)                (single atom)
mma.wave_size   # 64
mma.a_layout    # WarpDistributionEncoding (inspectable, never hand-authored)
mma.a_desc      # TileDesc((16,16), a_layout) -- shape + layout, ready for load_fragment
mma.c_desc      # TileDesc((16,16), c_layout) -- use for the accumulator Fragment
```

### Override: wave macro-tile -- the object owns the subtile grid
```python
mma = TileMma((32, 32, 32), a="f16", b="f16", c="f32", target="gfx90a",
              tiling=Tiling(atom_shape=(16, 16, 16)))
mma.subtiles    # (2, 2, 2)  -> the object walks a 2x2x2 atom grid inside one mma() call
# author STILL writes one load/mma/store per K-tile; no atom loop.
```

### Validation is automatic and fail-fast
```python
TileMma((32, 32, 16), a="f16", b="bf16", c="f32", target="gfx90a")   # ValueError: A/B dtypes
TileMma((13, 16, 16), a="f16", b="f16",  c="f32", target="gfx90a")   # ValueError: no intrinsic
TileMma((32, 32, 24), a="f16", b="f16",  c="f32", target="gfx90a",
        tiling=Tiling(atom_shape=(16,16,16)))   # ValueError: wave K not a multiple of atom K
```

---

## 7. `Tiling` -- the two knobs (atom selection + iteration order)

### 7a. `atom_shape` -- THREE ways to pick the atom

**Default (None): single MMA -- atom == wave shape.**
```python
TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")   # tiling=None
# -> atom_shape == (16,16,16), subtiles == (1,1,1)
```

**Override by SHAPE tuple: target-agnostic.** The atom is resolved for the bound target +
dtypes -- MFMA on CDNA, WMMA on RDNA -- with no author-side gfx branch.
```python
TileMma((32, 32, 16), a="f16", b="f16", c="f32", target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)))     # op_id -> mfma_f32_16x16x16f16
# same author code, different target (verified):
TileMma((32, 32, 16), a="f16", b="f16", c="f32", target="gfx12",
        tiling=Tiling(atom_shape=(16, 16, 16)))     # op_id -> wmma_f32_16x16x16_f16_w32_gfx12 (wave 32)
```

**Override by NAME string: the escape hatch (target-specific by construction).** Pin the
exact backend intrinsic; validated (exists / runs on target / dtypes match) fail-fast.
```python
TileMma((32, 32, 16), a="f16", b="f16", c="f32", target="gfx90a",
        tiling=Tiling(atom_shape="mfma_f32_16x16x16f16"))   # exact op resolved in the backend
```

**Why it matters (the concern this solves):** a bare 32x32 tile with `atom=None` would
resolve the *native* 32x32 atom (a single MMA), NOT a 2x2 grid of 16x16 atoms. The knob
lets you force the small atom either way:

```python
TileMma((32, 32, 8), a="f16", b="f16", c="f32", target="gfx90a")     # -> 32x32x8 native, subtiles (1,1,1)
TileMma((32, 32, 16), ..., tiling=Tiling(atom_shape=(16,16,16)))     # -> 16x16 atom, subtiles (2,2,1)
```

### 7b. `order` -- the M/N/K subtile loop-nest (a permutation of "MNK")

**Default "MNK": K innermost** (natural accumulation order).
```python
Tiling(atom_shape=(16, 16, 16))                       # order defaults to "MNK"
```

**Override: any of the 6 permutations.** Stride convention -- the RIGHT-MOST axis varies
fastest (innermost); the left-most is outermost. Since C accumulation is commutative, every
order is **bit-exact**; the knob is for schedule / locality.
```python
Tiling(atom_shape=(16, 16, 16), order="KNM")   # K outermost, M innermost/fastest
Tiling(atom_shape=(16, 16, 16), order="NMK")   # N outer, K inner
# invalid orders fail fast:
Tiling(atom_shape=(16, 16, 16), order="diagonal")   # ValueError: unknown subtile order
```

---

## 8. Sensible defaults are DRIVEN BY THE MMA

Everything the author does not state is resolved from the one `TileMma` call. This is the
"say the minimum" payoff:

| You did NOT specify | Resolved from | Exposed as |
|---|---|---|
| the concrete intrinsic | atom shape + dtypes + target | `mma.op_id` |
| the wave size | the resolved traits row | `mma.wave_size` |
| per-lane A/B/C register counts | the layouts | `fragment_length(mma.a_layout)` etc. |
| lane x register -> (row,col) map | `calculate_x` over the encoding | `describe(mma.c_layout)` |
| number of atoms in the tile | wave shape / atom shape | `mma.subtiles` |
| accumulator dtype/width | the `c=` dtype | `Fragment(c_desc, <c dtype>)` |
| the subtile iteration | `Tiling.order` | `mma(b, ...)` (internal) |
| load/store alignment | the view dtype's byte width | (internal) |

Override any single one without touching the others -- see the composability matrix.

---

## 9. Composability matrix (extents of the API)

Each knob is independent; combine freely. `BUILT` = works today on gfx90a; `RESERVED` =
designed-in seam, not yet implemented.

| Axis | Default | Override options | Status |
|---|---|---|---|
| operand dtype | -- | f16, bf16, f32, f8/bf8 (via desc + `a/b/c=`) | BUILT (f16->f32 proven; casts honest/fail-fast) |
| data layout | -- | any via `make_tensor_desc(..., strides)` + `.permute([...])` (RCR shown) | BUILT |
| tile distribution | `mma.a/b/c_desc` | author any via `make_tile_desc(...)` (quantity-major) | BUILT |
| atom selection | single MMA (`None`) | shape tuple (agnostic) / intrinsic name (pinned) | BUILT |
| wave tile size | == atom | any integer multiple of atom (M and/or N and/or K) | BUILT |
| subtile order | `"MNK"` | any of 6 `MNK` permutations | BUILT |
| target | -- | any gfx via `target=` / `arch=` (agnostic authorship) | BUILT (gfx90a proven; RDNA resolves) |
| clipping / bounds | desc `lengths` (auto) | `make_window(tensor, origin, bounds)` -> zero-pad load / drop store | BUILT (bit-exact incl within a valid space) |
| clip fill | 0 (zero-pad) | `pad=` on load (`constant(value)`) | RESERVED (`pad!=0` raises) |
| C transpose | off | `c_transpose=True` on `TileMma` (-> C^T, `c_desc` = (N,M)) | PLANNED (analyzed; Part D) |
| interleaved layout | default per-atom | `mma_layout=interleaved()` | PLANNED (Part D) |
| operand layout | `mma.a/b/c_layout` | `make_tile_desc(...)` / `custom_layout(...)` | BUILT (make_tile_desc) / RESERVED (custom_layout) |
| memory space | global | LDS view (`space="lds"`) | RESERVED |
| coherency | default | `CACHE_STREAM` / `NON_TEMPORAL` per load | RESERVED (param present) |
| kind | dense | `sparse(...)` / `scaled(...)` (+index/scale operands) | RESERVED |

---

## 10. Worked progression -- one kernel, escalating overrides

The same GEMM, adding exactly one override at a time. Note the body never changes -- only
the spec.

```python
# (0) DEFAULT: single 16x16x16 atom, f16->f32, MNK order. Say almost nothing.
spec = TilingGemmSpec(tile=(16, 16, 16))

# (1) BIGGER K per tile: 4 K-subtiles, MMA owns the inner K loop.
spec = TilingGemmSpec(tile=(16, 16, 64), atom=(16, 16, 16))

# (2) COOPERATIVE M/N: a 2x2 output grid, MMA walks it internally.
spec = TilingGemmSpec(tile=(32, 32, 16), atom=(16, 16, 16))

# (3) FULL GRID + iteration order: 2x2x2, K outermost.
spec = TilingGemmSpec(tile=(32, 32, 32), atom=(16, 16, 16), order="KNM")

# (4) PIN THE EXACT INTRINSIC by name (target-specific escape hatch).
spec = TilingGemmSpec(tile=(32, 32, 16), atom="mfma_f32_16x16x16f16")

# every one of these: build + run the SAME body, bit-exact on gfx90a.
kernel, mma = build_tiling_gemm(spec, 256, 256, 256, arch="gfx90a")
```

Each step is one field on the spec; the loop body (`load_fragment -> mma -> store_fragment`)
is byte-for-byte identical across all five. That is the composability claim, made concrete.

---

## 11. Spec -> builder (the instance layer)

Knobs travel as data; the builder validates then lowers -- rocke's normal flow.

```python
@dataclass(frozen=True)
class TilingGemmSpec:
    tile: tuple[int, int, int]                       # REQUIRED wave-tile size
    atom: tuple[int, int, int] | str | None = None   # shape tuple | intrinsic name | single-MMA
    order: str = "MNK"                               # subtile loop-nest (permutation of MNK)
    a_dtype: str = "f16"
    b_dtype: str = "f16"
    c_dtype: str = "f32"
    name: str = "tiling_gemm_demo"

ok, why = is_valid_spec(spec, arch="gfx90a")          # fail-fast pre-check (bool, reason)
kernel, mma = build_tiling_gemm(spec, M_LEN, N_LEN, K_LEN, arch="gfx90a")
```
- Spec is **target-agnostic**; the arch binds at `build_tiling_gemm(..., arch=)`.
- `is_valid_spec` returns `(ok, reason)`; `build_tiling_gemm` raises `ValueError` on an invalid
  spec, `NotImplementedError` on a reserved combination.

---

## 12. Reflection -- see what any default resolved to

Because every default is a resolved *value*, a human or agent can inspect it:

```python
describe(mma.a_layout)          # structured dataclass: sizes, lane/register map, provenance
render_forward_map(mma.c_layout)   # ASCII lane x register -> (row, col)
render_inverse_map(mma.c_layout)   # ASCII (row, col) -> (lane, register)
mma.op_id, mma.atom_shape, mma.subtiles, mma.wave_size   # what got chosen
```

This is the antidote to the raw-encoding anti-pattern: you never author integer sequences,
and you can always see the one the MMA object produced.

---

## 13. Quick reference -- default vs override per knob

| Knob | Default (MMA-driven) | Manual override |
|---|---|---|
| intrinsic | resolved from atom+dtypes+target | `Tiling(atom_shape="mfma_...")` |
| atom shape | == wave shape (single MMA) | `Tiling(atom_shape=(16,16,16))` |
| wave tile | 1 atom | `tile=(32,32,32)` etc. (multiple of atom) |
| subtile order | `"MNK"` (K innermost) | `order="KNM"` (any permutation) |
| operand desc | `mma.a/b/c_desc` | `make_tile_desc(...)` (author it) / `TileDesc(shape, layout)` (self-composed) |
| operand dtype | from `a/b/c=` + desc | any typed `make_tensor_desc` + `Fragment` dtype |
| data layout | -- | `make_tensor_desc(..., strides)` (+ `.permute([...])`) |
| clipping | desc `lengths` (auto zero-pad) | `make_window(tensor, origin, bounds)` |
| target | -- | `target=`/`arch=` (agnostic) |
| coherency | default | `coherency=` (reserved) |
| memory space | global | LDS desc (reserved) |
| kind | dense | `sparse`/`scaled` (reserved) |
