# rocKE Tiling Primitives -- Design Proposal

**Status:** M1 + wave-tile subtiling + CLIPPING (Part C) COMPLETE (95/95 tests, BIT-EXACT on gfx90a) and MIGRATED into the tracked tree; c_transpose + interleaved layouts planned.
**Date:** 2026-07-23 (migrated 2026-07-28) , **Primary target:** gfx90a (wave64 -- the host)
**Location:** package `platform/python/rocke/helpers/tiling/` (import `rocke.helpers.tiling`); docs in its `docs/`.
**Provenance / full detail:** `tiling_review_report.md` (sec 0-sec 18, adversarial + expert reviews), `tiling_design_plan.md` (original approved plan).

> **Compliance:** this document + the sibling `reference_docs/` reference internal artifacts and
> NPI (out of scope). **Scrub NPI/product/perf data before any Confluence publication.** Convert
> via the `ck-confluence` skill.

> **START HERE (new agent, zero context):** read sec 1-sec 2 (what/why), then **Appendix B**
> (how the encoding actually works -- worked gfx90a example; this is the core mechanic),
> then sec 3-sec 4 (architecture + author surface). **Appendix C** is the file map + how to run.
> **Appendix D** indexes the provenance report (`tiling_review_report.md`). The one thing
> to internalize: the raw `TileDistributionEncoding` (integer sequences) is powerful but
> is the **anti-pattern**; this project hides it behind named `layout` values.

---

## 1. Executive summary

We are building a **human-approachable tiling primitives layer** for rocKE that overcomes
four fundamental limits of rocWMMA, while reusing rocKE's existing static tile-distribution
substrate. It is CK-inspired (the WMMA/MFMA unification project) but rocke-native, and it is
designed from day one for the **whole gamut (dense / sparse / MX), all gfx targets,
debuggability, and agentic workflows**.

**The five rocWMMA limits we overcome:**
1. Fragments are always assumed MMA-ready (no free register layout).
2. Cooperative/interleaved layouts split only in K (not M/N). [M/N subtiling BUILT; interleaved variant planned -- sec 14]
3. No layout control beyond row/col-major (no custom swizzle; LDS forced to mirror global).
4. No memory-coherence flags (no cache-bypass -- needed for StreamK).
5. No clipping / bounds checking (partial tiles force padding or an edge kernel). [BUILT -- Part C: `make_window(tensor, origin, bounds)` + zero-pad load / drop store]

**The central idea:** *specialization is data, not code.* Layouts, kinds, transforms,
targets, and traits are **values / table rows**, not code paths -- so dense M1 is a strict
subset and sparse / MX / RDNA slot in additively.

---

## 2. Design principles (the rubric -- applied to everything)

1. **Human-approachable API.** A competent author comprehends it without decoding integer
   sequences. The raw `Ps2RHs/Ys2RHs` encoding is the explicit **anti-pattern to beat**.
2. **Strong overridable defaults; infinite customization & composability.**
3. **Authors think implementation**; architecture / auto-tuning is a later, separate (agent) concern.
4. **Target-agnostic authorship** (no per-gfx branches in author code).
5. **Dual Python + C++ authorship from ONE language-neutral source of truth.**

---

## 2.5 Provenance & rationale -- what we borrow, and what we reject

This design stands on three prior bodies of work. We take the good ideas from each and
fix their specific failures.

### ck_tile (Composable Kernel's tile framework)
- **Borrow:** the `tile_distribution_encoding` concept -- a coordinate-transform graph over
  unmerge (`Hs`) / lane-partition (`Ps`) / register-item (`Ys`) / replication (`Rs`) axes,
  which rocke already ports as `TileDistributionEncoding`; the unmerge/merge algebra; the
  `calculate_x` mixed-radix coordinate math; `shuffle_tile` and reduce-distribution
  transforms.
- **Reject/fix:** authoring by hand-writing raw `Ps2RHs/Ys2RHs` integer sequences is
  powerful but **incomprehensible to humans** -- the anti-pattern this project exists to
  beat. Its "tensor descriptor" naming also *lies* (it is a transform graph, not a data
  descriptor).
- **Rationale:** the substrate math (`calculate_x`) is excellent and MMA-agnostic; the
  problem is the *authoring* of the encoding. So we keep the math untouched and hide the
  encoding behind named `layout` values.

### rocWMMA
- **Borrow:** the **three-layer separation** (DataLayout / MatrixLayout / RegisterLayout);
  register-layout-as-transform-graph-state with NOP elision; compile-time MaxVW/MmaDim
  autoselect (a good *mechanism*); the fragment + `load`/`mma`/`store` dataflow vocabulary;
  coop-as-a-wrapper over a base layout.
- **Reject/fix:** fragments are always MMA-ready (no free register layout); the transform
  system is a **closed enum + hand-specialized pairs** that silently returns wrong
  registers on a missing path; no custom swizzle; no coherence flags; cooperative layouts
  split only in K; the `nvcuda::wmma` compatibility tax (an AoS shape the transform
  machinery then exists to undo); wave size baked as a compile-time macro; the
  DPP-per-issue gfx11 storage policy is baked in, not a choice.
- **Rationale:** rocWMMA's *decomposition* is right, but every layer is closed/baked. We
  make each layer **open and overridable** and drop the nvcuda tax (we no longer need CUDA
  API compatibility).

### CK WMMA/MFMA unification project (the most direct ancestor)
- **Borrow (wholesale):** the parameterized **encoding calculator** (`TileDistrEncCalc`) and
  **register mapper** (`TileDistrEncRegMap`) -- ported ~1:1 because CK's encoding type is
  *field-identical* to rocke's; the **traits-table-as-SSOT + codegen** approach; the
  **Compact Unmerge-Merge Descriptor** (any intrinsic's layout = MNKBRS + the K/M unmerge
  numbers); the **selector / `WaveWiseMma`** target-agnostic pattern; **whole-gamut coverage**
  (dense/sparse/MX; distinct R/B/S axes); the register-mapper print helpers (our
  visualization baseline); and the concrete **interleave transform formulas** (the CDNA/RDNA
  layout spreadsheets).
- **Reject/fix:** it still exposes layout *integers* to higher-level code (the team itself
  asked "should the layout integers be private?" -- our answer: yes); it is C++-only and does
  not address the human-API or agentic problem.
- **Rationale:** the unification did the hard, broad hardware-coverage + calculator work
  "across the whole gamut." We take that as our engine and add the piece they did not build:
  a **human-approachable, agentic, dual-language authoring layer** on top. This is why the
  project is "CK-inspired, rocke-native."

### House standards (borrowed as-is)
- **PYTHON_STYLE** (`style/PYTHON_STYLE.md`) and the project's error-handling convention define
  our engineering conventions (sec 11) -- we conform rather than invent.

## 2.6 The synthesis -- have we finally found a decent tiling API?

Each prior system nailed **one** hard sub-problem and failed the others. None is, alone, a
decent tiling API -- which is why "a good AMD tiling API" has stayed elusive.

| Source | The problem it *solves well* | What we take | Why it isn't enough alone |
|---|---|---|---|
| **ck_tile** | **Expressiveness** -- one encoding can represent *any* hardware layout (the mixed-radix unmerge/merge graph is fully general) | the encoding + `calculate_x` math | human-hostile: you author raw integer sequences; the abstraction leaks everywhere |
| **rocWMMA** | **Ergonomic decomposition** -- the 3-layer split + fragment `load/mma/store` reads like intent | the layering + the dataflow verbs | closed & baked: MMA-ready-only, closed transform set, no swizzle/coherence, nvcuda tax |
| **CK unification** | **Coverage + parameterization** -- one calculator + traits table spans dense/sparse/MX across every gfx, target-agnostically | the calculator, reg-mapper, traits-SSOT, descriptor, selector | C++-only; still exposes layout integers; no human/agent-facing surface |

**The realization that unlocks it:** the encoding *math* was never the problem -- the
*authoring* was. ck_tile proves the substrate can express anything; rocWMMA proves the
dataflow can read cleanly; the unification proves one parameterized engine can cover the
whole hardware gamut. So we **keep the expressive substrate untouched, borrow the clean
dataflow (made open, not baked), drive it with the unification's whole-gamut calculator, and
hide all of it behind named `layout` values** -- with reflection so a human or agent can see
what any value resolved to.

**A "decent tiling API" must be six things at once:** (1) expressive enough for any hardware
layout, (2) approachable enough for a human/agent to author without decoding integers,
(3) open/overridable for infinite customization, (4) target-agnostic, (5) whole-gamut
(dense/sparse/MX), (6) debuggable. ck_tile has (1); rocWMMA has (2) partially; the
unification has (1)(4)(5). **No prior system has all six.** This design targets all six by
construction -- by *separating the substrate we keep from the surface we build new*.

**Honest verdict:** plausibly, yes -- but it is earned, not assumed. It becomes real when the
M1 gates turn green: (a) an author snippet lowers with **no raw encoding and no gfx branch**,
(b) the resulting layout **field-equals the oracle** (`make_c_warp_dstr_encoding`), and
(c) the layout **algebra's closure** holds (every term is a valid encoding). If those hold,
we will have the first AMD tiling API that is expressive *and* human/agent-approachable *and*
open *and* whole-gamut. That is the bet this project is making.

---

## 3. Architecture (top-down)

```
 SPEC -> BUILDER (instance)       TilingGemmSpec(tile, atom, order, dtypes) -> build_*(spec, arch)
   |                              knobs/levers as data; is_valid_spec(spec, arch) fail-fast
   v
 AUTHOR SURFACE (public)          make_tensor_desc(lengths, strides, dtype) , make_tile_desc(...) ,
   |                              TileDesc(shape, layout) , make_fragment(desc, dtype)
   |                              verbs: fill_fragment / load_fragment / store_fragment  (b-first, ptr at the verb)
   |                              TileMma(wave_shape, a,b,c, target, tiling=Tiling(...)) ; mma(b, a,b,acc)
   v
 MMA OPERATION (target-aware)     TileMma: selector resolves op_id from atom (shape OR name) + target;
   |                              PROCESSES THE WHOLE WAVE TILE -- OWNS the M x N x K subtile grid and its
   |                              iteration + order (m_iter/n_iter/k_iter, `order` knob); validates operand dtypes
   v
 LAYOUT SYSTEM                    Layout = WarpDistributionEncoding value; never author-visible as raw
   |                              wave layout = atom encoding + m_iter/n_iter/k_iter (subtile-contiguous)
   v
 ENCODING CALCULATOR (internal)   port of CK TileDistrEncCalc -> WarpDistributionEncoding (bijection-checked)
   REGISTER MAPPER (internal)     port of CK TileDistrEncRegMap (num_lanes / num_vector_items / inverse)
   |
   v
 rocKE SUBSTRATE (reused)         distribution.py: TileDistributionEncoding, make_static_tile_distribution,
                                  calculate_x (mixed-radix, MMA-agnostic), shuffle_tile
   REFLECTION                     describe() (structured) + text visualizer (forward/inverse maps)
```

**Key seam:** CK's `tile_distribution_encoding<Rs,Hs,Ps2RHs*,Ys2RHs*>` is field-identical to
rocke's `TileDistributionEncoding`, so the unified MMA layer ports ~1:1 onto our substrate.

---

## 4. Author surface (public API)

> **See also:** `tiling_api_surface.md` -- the API-surface + options/features catalog, with
> default-mode vs manual-override examples for every knob. This section is the summary; that
> doc is the exhaustive reference.

The surface is **free factory functions** (ck_tile idiom) over thin IR-free value objects, three
b-first verbs, and the MMA driver. The value objects (`TensorDesc`/`TensorWindow` in
`descriptors.py`; `TileDesc`/`Fragment` in `fragments.py`) are dataclasses; you build them with
`make_*`, never with a method on another object (the `ptr` binds at the verb, not on the desc):
- **`make_tensor_desc(lengths, strides, dtype)`** -- a ptr-free strided memory descriptor; `lengths` are the valid extent (the default clip), `strides` the physical layout.
- **`make_window(tensor, origin, bounds=None)`** -- a positioned window; `bounds` (per-axis) overrides the clip (load zero-pads, store drops). `None` = the desc `lengths` (aligned = no clip).
- **`make_tile_desc(shape=, thread_dist=, thread_tile=, ...)`** -- author a DTYPE-FREE logical-matrix -> per-lane-register layout as a geometric table (human-approachable; see `tiling_api_surface.md` sec 4b). `TileDesc(shape, layout)` pairs a shape with an existing layout value.
- **`make_fragment(tile_desc, dtype)`** -- realized register data (dtype lives here).
- **verbs** `fill_fragment(b, frag, 0)` / `load_fragment(b, ptr, window, desc, lane, *, pad=0)` / `store_fragment(b, ptr, window, frag, lane)`.
- **`TileMma`** -- resolves the intrinsic, exposes wave `a/b/c_layout` + `a/b/c_desc`, and **processes the whole wave tile: it owns the M x N x K subtile grid, the iteration (m/n/k_iter) and the `order`** -- the author writes ONE load/mma/store, never an atom loop.

```python
mma = TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")  # IR-free resolve

acc = make_fragment(mma.c_desc, F32)                # born MMA-format, kept across the K loop
fill_fragment(b, acc, 0)
a_win = make_window(a_td, (m, k))                    # clip defaults to a_td.lengths (compute extent)
a_frag = load_fragment(b, a_ptr, a_win, mma.a_desc, lane)  # descs come FROM the mma; load zero-pads OOB
b_frag = load_fragment(b, b_ptr, make_window(b_td, (n, k)), mma.b_desc, lane)
acc = mma(b, a_frag, b_frag, acc)                   # one call; walks the whole subtile grid in `order`
store_fragment(b, c_ptr, make_window(c_td, (m, n)), acc, lane)   # drops OOB; f32 -> desc dtype
```

### 4.1 A `TileDesc` can come from several sources
`load_fragment` / `store_fragment` / `Fragment` all take a `TileDesc` (shape + layout). That
desc can be sourced three ways -- all interchangeable at the verb:
- **From the MMA object (default):** `mma.a_desc` / `b_desc` / `c_desc` -- wave shape + the
  resolved layout (M/N/K subtiles folded in). The strong default; a human never authors raw encodings.
- **Self-composed:** `TileDesc(shape, layout)` -- pair any shape with a layout value
  (e.g. `mma.a_layout`, or a reserved `custom_layout(...)`) when you want to build it yourself.
- **From future primitives (reserved):** data-tile / LDS-staging profiles will emit their own
  `TileDesc` for non-MMA tiles through the same interface.

### 4.2 The `TileMma` object + `Tiling` policy
- **Target-binding line.** `TileMma(wave_shape, a=, b=, c=, target=, tiling=)` is target-aware; authorship stays target-agnostic (target is data, not author branches). Analogue of CK `MmaDefaultSelector`/`WaveWiseMma`.
- **Shape = the wave tile.** `TileMma(atom_shape)` = single MMA; `TileMma(multiple-of-atom)` = wave macro-tile -- the object iterates the M x N x K subtile grid internally; **the author never hand-loops atoms**.
- **`Tiling(atom_shape=, order=)`** -- the object's knobs:
  - **`atom_shape`** -- how to pick the hardware atom: a shape `(M,N,K)` tuple (TARGET-AGNOSTIC; resolved for the bound target + dtypes), an explicit intrinsic name `str` (e.g. `"mfma_f32_16x16x16f16"` -- the target-specific escape hatch), or `None` (single MMA = wave shape). Validated: atom exists for (target, dtypes); wave shape is an integer multiple of the atom.
  - **`order`** -- the M/N/K subtile loop-nest order, a permutation of `"MNK"`. Stride convention: the RIGHT-MOST axis varies fastest (innermost); `"MNK"` runs K innermost. C accumulation is commutative, so every order is bit-exact -- the knob is for schedule/locality.
- **`kind`** = `dense()` (default) / `sparse(...)` / `scaled(...)` -- required extra operands + atom family (reserved in M1; see 4.6).
- **Mandatory validation** in `mma(...)`: operand fragment dtypes match the resolved A/B/C dtypes -- else fail-fast. (Mismatched dtypes silently produce wrong results.)

### 4.3 Fragment transforms
`transform_fragment(input_fragment, target_layout)` -- the input carries its own shape + current layout; only the target is passed.
- **A/B:** interleave-to-MMA on load. **Accumulator:** born in `c_layout`, kept across the K loop, transformed back only at store.
- The MMA interleave is a **compile-time, intra-lane** register permutation (`interleave_idx<1, dim_per_thread, dim_per_thread,k_per_thread>`) -> maps to the `shuffle_tile` (Y-relabel) class; no cross-lane movement.

### 4.4 Example 1 -- simple GEMM (single 16x16x16 tile, the M1 shape) -- BUILT

This is the real, running M1 demo (`kernels/tiling_gemm_demo.py`), following rocke's normal
**spec -> builder** flow. Integer inputs make it **bit-exact** on gfx90a.

```python
spec = TilingGemmSpec(tile=(16, 16, 16))                    # knobs as data (atom=None -> single MMA)
kernel, mma = build_tiling_gemm(spec, M_LEN, N_LEN, K_LEN, arch="gfx90a")

# inside build_tiling_gemm:
mma = TileMma(spec.tile, a="f16", b="f16", c="f32", target=arch, tiling=Tiling(atom_shape=spec.atom))
TILE_M, TILE_N, TILE_K = spec.tile

a_td = make_tensor_desc((M_LEN, K_LEN), (K_LEN, 1), F16)    # RCR: A(M,K) row-major
b_td = make_tensor_desc((N_LEN, K_LEN), (1, N_LEN), F16)    #      B stored so coords (n,k) walk (1, ld)
c_td = make_tensor_desc((M_LEN, N_LEN), (N_LEN, 1), F16)    #      C(M,N) row-major

accumulator = make_fragment(mma.c_desc, F32)               # descs come from the mma
fill_fragment(b, accumulator, 0)
for tile_k_base in range(0, K_LEN, TILE_K):
    k_base = b.const_i32(tile_k_base)
    a_fragment = load_fragment(b, a_ptr, make_window(a_td, (m_tile_base, k_base)), mma.a_desc, lane)
    b_fragment = load_fragment(b, b_ptr, make_window(b_td, (n_tile_base, k_base)), mma.b_desc, lane)
    accumulator = mma(b, a_fragment, b_fragment, accumulator)
store_fragment(b, c_ptr, make_window(c_td, (m_tile_base, n_tile_base)), accumulator, lane)
b.ret()
```

### 4.5 Example 2 -- wave macro-tile (subtile grid) -- BUILT

Same verbs and dataflow as Example 1; the **only** author change is the `Mma` shape + a
`Tiling` knob. A 32x32x32 wave tile over a 16x16x16 atom is a **2x2x2 subtile grid** the
object iterates internally -- the author still writes ONE `load`/`mma`/`store` per K-tile.
Verified **bit-exact** on gfx90a for 16x16x64 (K subtiles), 32x32x16 (M/N subtiles),
32x32x32 (full grid), and every `order` permutation.

```python
# The ONLY delta from Example 1: a bigger wave shape + the atom/order knobs.
spec = TilingGemmSpec(tile=(32, 32, 32), atom=(16, 16, 16), order="MNK")
# atom may also be an explicit intrinsic name (escape hatch): atom="mfma_f32_16x16x16f16"
kernel, mma = build_tiling_gemm(spec, M_LEN, N_LEN, K_LEN, arch="gfx90a")

# mma.subtiles == (2, 2, 2); mma.a_layout now carries m_iter=2, k_iter=2 (subtile-contiguous).
# The body is byte-for-byte the same loop as Example 1 -- load a tile, load b tile, mma, store.
# `mma(b, a_fragment, b_fragment, accumulator)` walks the 2x2x2 atom grid internally in
# `order`, accumulating each C subtile; the author writes no atom loop.
```

**Reserved additive overrides (post-M1):** ping-pong LDS staging (`LdsTile` + `async_load` /
`wait_async` / `barrier`), per-load coherency (`coherency=CACHE_STREAM`), and swizzled LDS
views (`swizzled_layout()`) layer on top without changing the core `load -> mma -> store`
dataflow -- see `tiling_api_surface.md` for the full option catalog.

**Where the IRBuilder fits:** `b = IRBuilder(...)` is created once and **every IR-emitting verb takes it first** (`fill_fragment(b, ...)`, `load_fragment(b, ...)`, `mma(b, ...)`, `store_fragment(b, ...)`); the kernel is returned as `b.kernel`. The **IR-free** objects (`TileMma`, `TensorDesc`, `TileDesc`, layout values) are constructed *without* `b` -- they only resolve/compute -- which is why the M1 oracle is provable offline. `b` is the single vessel everything lowers into.

**What each example demonstrates:** the *core dataflow is identical* -- `load_fragment(b, ..., desc)` -> `mma(b, ...)` -> `store_fragment(b, ...)`, accumulator stays MMA-format. Everything hard (the K subtile stacking, the M x N grid walk, and on RDNA3 the GFX11 replicate) is under the hood and target-driven -- swap the bound `ArchTarget` and it re-lowers with no author change.

### 4.6 Whole-gamut usage (design-in; reserved/`NotImplementedError` in M1)

The same surface extends to sparse and MX by **adding operands, not code paths** -- the
index/scale operands are ordinary tiles loaded through the same `load_fragment` verb (same
`TileMma` / `TileDesc` / `Fragment` vocabulary as the dense examples above):

```python
# 2:4 sparse -- A is compressed in K; an index (metadata) operand joins the call.
# Main operands take the mma's descs; the index operand's desc is self-composed from its layout.
mma = TileMma((16, 16, 32), a="f16", b="f16", c="f32", target="gfx90a",
              kind=sparse(compression_ratio=2))
a_frag       = load_fragment(b, a_ptr,     make_window(a_td, (m, k)),     mma.a_desc,                            lane)  # KP=K/2
a_index_frag = load_fragment(b, index_ptr, make_window(index_td, (m, k)), TileDesc((16, 16), mma.a_index_layout), lane)  # 2-bit meta
accumulator = mma(b, a_frag, b_frag, accumulator, a_index=a_index_frag)

# MX-scaled -- A/B carry 8-bit scale operands (K/32); independent A/B scales.
mma = TileMma((16, 16, 128), a="fp4", b="fp4", c="f32", target="gfx90a",
              kind=scaled(scale_divisor=32))
a_scale_frag = load_fragment(b, a_scale_ptr, make_window(a_scale_td, (m, k)), TileDesc((16, 4), mma.a_scale_layout), lane)
b_scale_frag = load_fragment(b, b_scale_ptr, make_window(b_scale_td, (n, k)), TileDesc((16, 4), mma.b_scale_layout), lane)
accumulator = mma(b, a_frag, b_frag, accumulator, a_scale=a_scale_frag, b_scale=b_scale_frag)
```

### 4.7 Advanced overrides (the "infinite customization" seam)
Every default is a value you can replace -- see `tiling_api_surface.md` for the full
default-vs-override catalog. Built today: `Tiling(atom_shape=(32, 32, 8))` (shape) or
`Tiling(atom_shape="mfma_f32_16x16x16f16")` (explicit intrinsic name), `order="KNM"` (any
`MNK` permutation), `TileDesc(shape, custom_layout)`. Reserved (post-M1):
`coherency=NON_TEMPORAL` on a load, `swizzled_layout(...)` on an LDS view, `kind=sparse(...)`
/ `scaled(...)`. Reflection (`describe`/`visualize`) lets a human or agent inspect exactly
what any of these resolved to.

### 4.8 The IRBuilder is the lowering vessel

The examples above omit it for readability, but the IR-emitting verbs thread rocke's
`IRBuilder` (`b`) -- the vessel everything lowers into -- following rocke's b-first
convention: `load_fragment(b, ptr, window, desc, lane)`, `mma(b, a_frag, b_frag, acc)`,
`store_fragment(b, ptr, window, frag, lane)`. Two layers (see review sec 19): the **IR-free** layer
(traits / layout / encoding calculator / register mapper / reflection) computes the
distribution with no IR -- which is why the M1 oracle is provable offline; the
**IR-emitting** verbs drive addressing/vectorization from the layout and bottom out at raw
b-first `IRBuilder` ops (`global_load` / `global_store` / `b.mma` / integer arithmetic) --
NOT rocke's `mfma_gemm_inner` `load_tile` / `store_tile` / `mfma_k_loop` (the isolated
new-API surface). Validation stays in the IR-free layer / at verb entry, never in
IR-emitting inner loops.

---

## 5. Layout system internals

- **Compact Unmerge-Merge Descriptor** (SOT formalism): beyond MNKBRS, any intrinsic's layout is fully specified by the K-unmerge and M-unmerge sizes. Universal order A/B/Index/Scale = `L{RK1BM} V{SK2K0}`, C/D = `L{M1N} V{BM2M0S}`.
- Maps onto our encoding: unmerge = `Hs`; merge-into-lane = `Ps2RHs`; merge-into-vector-item = `Ys2RHs`; **R** (replicate) = `Rs`; **B** (block) and **S** (sparse-index/scale OPSEL) are **distinct axes** -- never flatten B into R.
- **Must-specs (non-negotiable):** (1) all layout sources are terms in one algebra with closure = the set of valid `calculate_x`+bijection encodings, enforced at construction in axis vocabulary; (2) `default_layout()` resolves wave/atom/unmerge from the same traits row the selector chose (kills the silent half-fragment hazard); (3) `load()` returns polymorphic `ScalarTile` (element-addressable) vs `PackedFragment` (opaque until `mma()`), behind one `DistributedTensor` interface.

---

## 6. Transform engine (register-op algebra)

Intrinsic, all **intra-lane except `replicate`**:
`{ interleave_idx, unpack_lo_hi_16/32, flip_and_zip, replicate (swap_and_concatenate), extract_lo, pad }`.
- `interleave_idx<g,s,c>`: intra-lane register-array transpose (`i -> (i%(c/s)),s + i//(c/s)`); `<1,X,X>`/`<1,1,X>` are NOPs. **~free.**
- `replicate` (RDNA3 R): **cross-lane DPP** (`permlanex16` + concat) -- real VALU cost, doubles register width.
- `pad` (RDNA3 S): 16-bit output in a 32-bit VGPR via OPSEL.
- Cost classes (`FREE`/`DPP`/`LDS`) are **reported** by the transform (not encoded in verb names); the cost model lives in the traits row. Cross-lane / transpose relayout (non-RCR, staging) routes to `ds_read_tr` / LDS round-trip -- a separate concern from the MMA interleave.

---

## 7. Hardware coverage

| Family | Interleave | Notes |
|---|---|---|
| **CDNA (MFMA)** | `interleave_idx` + bit-ops | gfx90a/942/950; 16x16 acc = gather-4; **32x32 acc needs its own permutation term** (c_frag_len=16) |
| **RDNA4 (WMMA)** | `interleave_idx` only | CDNA-like; folds into the engine, no new ops |
| **RDNA3 (WMMA GFX11)** | `replicate`/`extract_lo` + `pad` | R (repeat) via `WMMA_INPUT_GFX11`; S via `WMMA_ACC_GFX11`; 16-bit `flip_and_zip` |

- **gfx11 storage policy -- DECISION:** default to **duplication (expand-at-load, hold `WMMA_INPUT_GFX11`)** -- reads are cached, only extra register space, avoids per-issue DPP -> better gfx11 perf, and matches the builtin ABI (`a_frag_len=16`). Compression (tight store + `extract` before MMA, rocWMMA's approach) is a deferred override.
- **Wave size** is a bind-time `ArchTarget` property (MFMA=64, WMMA=32), read from the chosen traits row.
- **RCR** (A row / B col / C row) is the aligned case: both operands K-contiguous -> interleave-only, no transpose. Data layout is **per-view** (composable), not a kernel-level template flag; non-RCR routes the delta through a transpose.
- **Coherency** (GLC/SLC/NT) rides the view/load-store, not the layout. **StreamK needs memory *scope*** (`syncscope("agent")`) + atomics, not just coherency hints -- reserved `scope=` seam (post-M1).

---

## 8. Whole-gamut coverage: dense / sparse / MX

Design for all three now; implement dense in M1 (subset, not refactor).
- **Sparse:** A compressed 4:2 in K -> `KP=K/2` (`compression_ratio`, not hard-coded to 2); an **Index operand** (2-bit, S/OPSEL) with A's dimensionality.
- **MX:** K-unmerge of 32, drop K0 (`scale_divisor=32`); **A/B Scale operands** (8-bit, S/OPSEL, independent A/B). Flagship `mfma_scale_f32_16x16x128_f8f6f4` needs two K-unmerges (exception flag).
- **Biggest lesson:** treat index/scale as **tiles**, not flags -- same `Tile`/`load`/`layout` path. No special-case code.
- **No-dead-end gate (before schema freeze):** a paper walkthrough expressing a sparse tile and an MX tile on the same surface (`mma_op(..., a_index=...)` / `mma_op(..., a_scale=..., b_scale=...)`).

### 8.1 Traits SOT & schema
- **SOT = the intrinsic support matrix** (the CK-unification codegen input). **Never hand-type values** -- `mma_traits.json` is regenerated from the support matrix, not hand-edited.
- Schema (per op, whole-gamut): `op_id`, `llvm_builtin`, `family`, `wave_size`, dtypes, `dims{M,N,K,B,R,S}`, `layout_params{ABK,AKN,AR,BKN,BR,CM,CMN}`, `flags`, `supported_targets[]`, `a/b/c_d_layout` descriptors, `_meta.column_glossary`. Sparse/MX add `compression_ratio`, `scale_divisor`, per-operand `sparse_vgpr{kind,s_size,placement}`, index/scale operand descriptors.

---

## 9. Debugging, reflection & visualization (first-class)

Antidote to the anti-pattern: understand a layout by **seeing** it. Baseline = CK's
`TileDistrEncRegMap` print helpers; generalize.
- **Reflection:** `describe(x)` -> structured data (frozen dataclass) + provenance (op_id, target, traits row) + cost class + subtile grid + operand kind.
- **Visualization:** forward (lanexreg->element), inverse (element->lane,reg[,replica]), named-axis encoding view, transform op-chain + permutation table, `explain(mma_op)`. Text-first (REPL/CI/error messages) + optional rich (matplotlib/HTML/SVG).
- **Errors render through the same visualizer** (axis vocabulary).

---

## 10. Agentic-workflow friendliness

- **Machine-readable reflection** (structured `describe()`), deterministic + inspectable defaults, fail-fast errors agents can parse.
- **Agent-facing teaching docs are deliverables** -- "author a tiled GEMM", "the layout-value model", "add a new intrinsic family", the migration map, `ERROR_HANDLING.md`.
- Stable, named vocabulary so agents compose reliably.

---

## 11. Engineering standards

- **PYTHON_STYLE** (policies branch `style/PYTHON_STYLE.md`): modern typing (`list[int]`, `X | None`, `collections.abc`), `from __future__ import annotations`, relative imports, black 88, `@dataclass(frozen=True)` value objects, `__all__` + `__init__` re-exports. `M/N/K` reserved for the MMA atom (`MMA_*`); tiling uses `TILE_*`/`WAVE_*`.
- **Error handling** (the project's error-handling convention): fail-fast at the API boundary (never in IR-emitting inner loops); message template `"{what_failed} -- {param}={bad}, expected {constraint}"`; builtin exceptions (`ValueError`/`TypeError`/`NotImplementedError`/`RuntimeError`, never `AssertionError`); dataclasses validate in `__post_init__`; positivity checks on high-risk dims; every raise gets a unit test; C++ parity via identical messages. (No custom exception hierarchy yet -- a "may want" future.)
- **Public vs internal boundary (designed-in, not retrofitted):** only `__all__`/`__init__`-exported names are public; the **raw encoding is never public**; calculator/mapper/selector/traits-internals/register-ops are internal.
- **Testing:** unit tests per component; `tests/` mirrors the source tree; `mma_layout`/oracle tests.
- **Migrated:** the package now lives in the tracked tree at `platform/python/rocke/helpers/tiling/`; tests at `platform/python/rocke/tests/helpers/tiling/`; every committed file carries the MIT header.
- **Composition over inheritance; simple classes; descriptive names** (no obfuscation).

---

## 12. Substrate & external references

- **Reused rocke substrate** (imported at marked seams on integration): `helpers/distribution.py` (`TileDistributionEncoding`, `make_static_tile_distribution`, `calculate_x`, `shuffle_tile`), `helpers/atoms.py` (`make_c_warp_dstr_encoding`, `c_warp_params` -- the **oracle**), `core/arch/target.py` (`ArchTarget`, `MmaOp`), `helpers/mfma_gemm_inner.py` (`mfma_k_loop`, `store_acc_to_global` -- the **integration seam**), `helpers/layouts.py` (`LdsLayout`), `core/ir.py` (coherency, `ds_swizzle`, `ds_bpermute`, `permlanex16`, `ds_read_tr`).
- **CK unification (ROCm/rocm-libraries `users/krithalith/ck/unification_all_fixes`):** `TileDistrEncCalc`, `TileDistrEncRegMap` (fetched, ported), `amdgcn_mma` (codegen'd specializations).
- **SOT / reference artifacts** (the sibling `reference_docs/`): internal reference material (the intrinsic support matrix / traits SOT, the interleave-layout transform formulas, and related notes) lives in the **gitignored** `reference_docs/` sibling and is NOT committed. Scrub before any external publication.

---

## 13. Plan & milestones

### Milestone 1 -- dense gfx90a, offline-proof-first
Port the dense no-block path and prove the human-API + target-agnostic thesis, then wire
one real kernel. Two live author surfaces only: `default_layout()` + `mma.a/b/c_layout`;
everything else reserved (`NotImplementedError`).
1. **Offline proof (no IR):** authored `mma.c_layout` field-equals `make_c_warp_dstr_encoding`; reproduces `MfmaAtom.lane_to_output` via untouched `calculate_x`; broken input errors in axis vocabulary; re-resolves gfx90a (MFMA) vs an RDNA (WMMA) target by swapping `ArchTarget`.
2. **End-to-end:** minimal dense-RCR kernel authored ENTIRELY through the new verbs (`load_fragment` / `mma` / `store_fragment`), bottoming out at raw `IRBuilder` ops -- **NOT** `mfma_gemm_inner` (the surface stays isolated). *(As built, the demo went further than this original plan: it uses no `mfma_gemm_inner` helpers at all.)*
**Untouched by M1:** `gemm_universal.py`, the three K-loop drivers + byte-identical gate, relayout engine, DPP storage policy, cshuffle, DTL, sched hints. (NB: the `atom_shape`/`order`/`tiling` subtile machinery, originally deferred, is now BUILT -- wave macro-tiles iterate the M/N/K grid; see sec 14.)
**First external action:** confirm the CK unification handoff (calculator / reg-mapper / traits, and their gfx11 storage choice).

### Beyond M1 (deferred, additive)
CDNA 32x32 acc term , RDNA4 (folds in) , RDNA3 GFX11 (replicate/pad + R/S axes) , sparse (index operand) , MX (scale operands) , cross-lane relayout (ds_bpermute/LDS) , StreamK scope/atomics , C++ calculator + build-time codegen , rich visualization + tracing , future NPI parts / D-matrix (out of scope).

---

## 14. Current status (living -- update as we go)

**Legend:** [DONE] done , [WIP] in progress , [TODO] not started

### Design
- [DONE] Full design decided and reviewed (two expert deep-dives; verdict: ready to prototype M1). Detail: `tiling_review_report.md` sec 0-sec 18.

### Package (`platform/python/rocke/helpers/tiling/`, tracked)
- [DONE] Module tree: root `encoding.py` / `register_mapper.py` / `descriptors.py` / `fragments.py` / `emit.py` + subpackages `traits/ layouts/ mma/ reflection/ kernels/`; documented public surface in `__init__.py` (23 names); tests mirror the package at `platform/python/rocke/tests/helpers/tiling/`
- [DONE] MIT headers on every committed file; import smoke failsafe; PYTHON_STYLE + Error-Handling-Proposal conformance

### M1 build + wave-tile subtiling + clipping -- COMPLETE (95/95 tests; BIT-EXACT GEMM on gfx90a)
- [DONE] **Committed `mma_traits.json`** -- 188 ops; gfx90a `mfma_f32_16x16x16f16` verified against the traits table.
- [DONE] **Typed `MmaTraits` loader + validation + tests** -- descriptive fields from SOT columns; `__post_init__` fail-fast; resilient load (block/incomplete rows reserved); `select(...)` resolves the atom by intent; `get(op_id)` resolves by name.
- [DONE] **Encoding calculator (dense no-block A/B/C) + subtiling** -- `WarpDistributionEncoding` (bijection-validated) from `MmaTraits`; generalized with `m_iter`/`n_iter`/`k_iter` so a wave layout stacks M/N/K subtiles **subtile-contiguous** (iter=1 is byte-identical to the atom encoding); rocke-native drop of the trivial C R.
- [DONE] **Register mapper + reflection visualizer** -- pure-int `lane x register -> (row,col)`; structured `describe()` + ASCII forward/inverse maps.
- [DONE] **Oracle (headline gate)** -- C encoding field-equals `make_c_warp_dstr_encoding` (16x16x16 + 32x32x8); forward map reproduces the *real* `MfmaAtom.lane_to_output` (via int-eval stub) across the full grid.
- [DONE] **Spec -> builder + public surface + demo** -- `TilingGemmSpec` (tile/atom/order knobs) + `is_valid_spec` + `build_tiling_gemm(spec, ..., arch=)`; value objects `TensorDesc`/`TensorWindow`/`TileDesc`/`Fragment`; generic verbs `fill_fragment`/`load_fragment`/`store_fragment` (dtype-threaded, no baked types); `TileMma` (+ `Tiling`) resolves the atom (shape tuple OR intrinsic name) and **processes the whole wave tile -- owns the M x N x K subtile grid + its iteration + order** (any `MNK` permutation, right-most fastest). All addressing from our encodings, bottoming out at raw `IRBuilder` ops (NO `mfma_gemm_inner`).
- [DONE] **Human-approachable authoring surface** -- `make_tile_desc(shape=, thread_dist=, thread_tile=, thread_order=, thread_broadcast=, block_repeat=, wave_dist=, wave_order=, wave_broadcast=, wave_size=)` authors any distribution as a quantity-major geometric table (columns = axes), returning a `TileDesc`; reproduces ck_tile's `MakeADramTileDistribution` field-exact. Kills the raw-encoding anti-pattern for non-MMA tiles.
- [DONE] **Free-factory surface + thin value objects** -- `make_tensor_desc` / `make_window` / `make_tile_desc` / `make_fragment`; `TensorDesc`/`TensorWindow` (`descriptors.py`) + `TileDesc`/`Fragment` (`fragments.py`) stay dataclasses; `TileMma` exposes `a/b/c_desc` (shape + resolved layout). Desc<->window rank agreement in `__post_init__`; `ptr` binds at the verb.
- [DONE] **Clipping / bounds (Part C, rocWMMA limit #5)** -- `make_window(tensor, origin, bounds)` (clip defaults to the desc `lengths`); load zero-pads (`masked_global_load`, `pad` knob), store drops (`scf_if`); tile-aligned axes skipped (byte-identical). Leading dims (`lda/ldb/ldc`) separate the physical stride from the compute extent.
- [DONE] **Bit-exact on gfx90a** (integer inputs, tol 0): single 16x16x16, K-subtiled 16x16x64, M/N-subtiled 32x32x16, full-grid 32x32x32, `order` variants, atom-by-name; **clipping** ragged 255^3 (OOB edge), 256x256x250 (K-only), and **250^3 into 256-alloc (within a VALID space)** -- masked cells hold real data yet excluded, C tail untouched (NaN). Encoding/mapper/spec/window-rank raises tested offline.
- [TODO] **`c_transpose` + interleaved layouts** -- analyzed (c_transpose matches CK); to build. See the plan's Part D.

---

## 15. Decisions log

| # | Decision | Where |
|---|---|---|
| D1 | CK-inspired, rocke-native; encoding-as-single-source-of-truth (Path #3) | review sec Context |
| D2 | Author-API-first; layout is a **value**, raw encoding never author-visible | sec 9, sec 12 |
| D3 | `Mma` is the target-binding line; authorship target-agnostic | sec 9.8 |
| D4 | `kind` (dense/sparse/scaled) explicit on `Mma`; not derivable from dtypes | sec 14.5 |
| D5 | `atom_shape` + `subtile_order` -> one `tiling=` policy on `Mma` | sec 12.1 |
| D6 | `transform_fragment(input, target_layout)` -- source read off the fragment | sec 10.2 |
| D7 | gfx11 default = duplication (expand-at-load); compression deferred | sec 11.3c |
| D8 | R/B/S are distinct axes; never flatten B into R | sec 11, sec 7.2 |
| D9 | Coherency on view/load-store; StreamK needs separate scope/atomics | sec 7, sec 12.2 |
| D10 | Whole-gamut design-in (dense/sparse/MX); implement dense in M1 | sec 14 |
| D11 | Error handling = project error-handling convention (builtin exceptions, Tier-A, `__post_init__`) | sec 16 |
| D12 | Public/internal boundary designed-in; raw encoding never public | sec 17 |
| D13 | M1 target = gfx90a; MMA traits SOT = the intrinsic support matrix (capture, never guess) | sec 18 |
| D14 | SUPERSEDED: the M1 demo uses NO `mfma_gemm_inner` helpers -- verbs bottom out at raw `IRBuilder` ops (`global_load`/`global_store`/`b.mma`), keeping the new surface fully isolated. (Was: integration seam = `mfma_gemm_inner.mfma_k_loop`.) | sec 4.8, D15 |
| D15 | IRBuilder is the lowering vessel; IR-free layer (traits/layout/calc/reflection) vs IR-emitting verbs (load/store/mma_op, b-first) | review sec 19 |
| D16 | Layout/TileDesc are DTYPE-AGNOSTIC (pure coordinate map). dtype lives on `Fragment` (from the typed view); ONLY `TileMma.__call__` validates operand dtype -- bare load/store trust `view.dtype`. Gap #5 accepted by design (matches CuTe/ck_tile) | sec 17, emit.py |
| D17 | End-to-end flow is **spec -> builder** (rocke convention): `TilingGemmSpec` carries the knobs (tile, atom, order, dtypes); `is_valid_spec(spec, arch)` fail-fast; `build_tiling_gemm(spec, ..., arch=)`. Spec is target-agnostic; arch binds at build. | sec 4.4, demo |
| D18 | `TileMma(wave_shape, tiling=Tiling(atom_shape, order))` OWNS the M x N x K subtile grid: wave layout = atom encoding + `m_iter`/`n_iter`/`k_iter` (subtile-contiguous); the author writes ONE load/mma/store, no atom loop. `order` = any `MNK` permutation, RIGHT-MOST fastest (stride convention); commutative -> bit-exact across orders. | sec 4.2, 4.5 |
| D19 | `Tiling.atom_shape` accepts a **shape tuple** (target-agnostic; resolve by shape) OR an explicit **intrinsic name** `str` (escape hatch; resolve by op_id, target-specific) OR `None` (single MMA). Names are validated (exists / target / dtypes) fail-fast. | sec 4.2 |
| D20 | **Free-factory surface** (ck_tile idiom): `make_tensor_desc` / `make_window` / `make_tile_desc` / `make_fragment` are the public factories; the value objects stay thin dataclasses. NOT methods on the descriptor -- it is a pure memory layout and must not know windows exist; `ptr` binds at the verb. Desc<->window rank agreement enforced in `TensorWindow.__post_init__`. | descriptors.py / fragments.py |
| D21 | **Clipping = window bound + verb, upper/lower box** (Part C, BUILT). `make_window(tensor, origin, bounds)` (clip defaults to the desc `lengths`); predicate `position < bound` per axis (bound = the compute extent, SEPARATE from the leading-dim stride); tile-aligned axes skipped at build time (byte-identical). Load zero-pads via `masked_global_load` (`pad` knob, default 0; `constant(value)` reserved); store drops via `scf_if`. Verified INSIDE a valid space (250^3 into 256-alloc, NaN-tail), not just OOB. C2 buffer-`num_records` wide path is the reserved efficient lowering. | Part C, emit.py |
| D22 | **`c_transpose` = a bool on `TileMma`, realized in the ENCODING** (swap a/b encoders + mma slots + `_transposed(c_enc)` axis swap; matches CK `CTranspose`). Chose the flag over CUTLASS `LayoutC`(row/col) because row/col is 2D but the tiling engine is ND (c_transpose = the 2-axis case of a permutation). View untouched; author threads the C window in (N,M). Analyzed, not yet built. | Part D (plan) |

---

## 16. Open questions

- Exact `custom_layout(...)` algebra signature (reserved; pin before it ships).
- CK handoff scope: which of calculator / reg-mapper / traits / sparse+scale policy structs they hand over, and their gfx11 storage choice.
- Whether C++ calculator is generated from spec or hand-mirrored + exhaustive golden corpus.
- Non-16/32 C-encoding oracle coverage (currently only 16x16 / 32x32 via `make_c_warp_dstr_encoding`).

---

## Appendix A -- Glossary
- **Encoding** -- `TileDistributionEncoding(Rs, Hs, Ps2RHs_major/minor, Ys2RHs_major/minor)`; the coordinate-transform graph (NOT a data descriptor). Internal; never author-visible.
- **Hs / Ps / Ys / Rs** -- hierarchical unmerge / lane-partition / register-item / replication axes.
- **interleave_idx<g,s,c>** -- intra-lane register-array transpose primitive.
- **SOT columns** -- ABK=kABKPerLane, AKN=kAKNumAccess, AR=kARepeat, BKN=kBKNumAccess, BR=kBRepeat, CM=kCMPerLane, CMN=kCMNumAccess.
- **RCR** -- A row-major, B col-major, C row-major (the K-contiguous aligned case).
- **Oracle** -- `make_c_warp_dstr_encoding` / `MfmaAtom.lane_to_output` (rocke substrate), the M1 correctness reference.

---

## Appendix B -- How the encoding actually works (zero-context deep-dive)

This is the core mechanic. Everything above hides it; to work on the internals you must
understand it.

### B.1 The encoding
`TileDistributionEncoding(Rs, Hs, Ps2RHs_major, Ps2RHs_minor, Ys2RHs_major, Ys2RHs_minor)`
is a coordinate-transform graph mapping **(lane in the wave, per-lane register slot)** to a
**matrix element coordinate** (row, col / M, N, K).

- `Hs` -- per X-dim **unmerge** decomposition. `Hs[x]` is a tuple of level sizes, **MSB->LSB**
  (last entry has stride 1). `X_length[x] = product(Hs[x])`.
- `Ps2RHs_*` -- how the **lane** (P) is built by merging chosen `H` (or `R`) buckets.
- `Ys2RHs_*` -- how the **register items** (Y) map to chosen buckets.
- `Rs` -- replication axis (RDNA3 repeat / broadcast). Empty for M1.
- **Major convention:** major `0` = R bucket; major `1..len(Hs)` = X-dim (major-1); minor
  indexes the level within that bucket.
- **Bijection invariant** (`__post_init__`): every H (and R) bucket is referenced by exactly
  one P or Y entry; every H bucket has a contributor. This is the structural correctness net.
- `calculate_x(b, ys, ps)` walks each `Hs[x]` from innermost (stride 1) outward:
  `x = sum(contributor * stride_below)`. It is pure mixed-radix arithmetic and **MMA-agnostic**.

### B.2 Worked example -- gfx90a `mfma_f32_16x16x16f16` C-accumulator
From the substrate oracle (`atoms.py` `_C_WARP_PARAMS[(16,16)] = (kCM0PerLane=1, kCMLane=4,
kCM1PerLane=4, kCNLane=16)`), `make_c_warp_dstr_encoding` builds:

```
Hs             = ((1, 4, 4), (16,))     # X0 = M split (m0=1, m_lane=4, m1=4); X1 = N (n_lane=16)
Ps2RHs_major   = ((1, 2),)              # lane sub0 -> X0 (M), sub1 -> X1 (N)
Ps2RHs_minor   = ((1, 0),)              # ...at M level 1 (m_lane=4) and N level 0 (n_lane=16)
Ys2RHs_major   = (1, 1)                 # both register dims on X0 (M)
Ys2RHs_minor   = (0, 2)                 # ...M level 0 (m0=1) and M level 2 (m1=4)
```

Decoding `calculate_x` for this encoding:
- **row (M)** = `y0*16 + (lane//16)*4 + y1` with `y0  in  [0,1)` (m0=1 => always 0), `y1  in  [0,4)` => **row = (lane // 16) * 4 + y1**
- **col (N)** = `lane % 16`
- per-lane register count = `m0 * m1 = 1 * 4 = 4` = `c_per_lane` (ok); lane extent = `m_lane * n_lane = 4 * 16 = 64` = wave64 (ok).

So lane 0 slot 2 -> (row 2, col 0); lane 17 slot 1 -> (row 5, col 1). That is the MFMA 16x16
accumulator layout. **The M1 oracle test asserts our calculator reproduces exactly this.**

### B.3 Decoding the SOT compact unmerge-merge descriptor
From the intrinsic support matrix, the same op's descriptors are `A: K{4} L{K1M} V{K0}` and
`C/D: M{4} L{M1N} V{M0}`. Read them as:
- `K{4}` -- unmerge K into `K0` (size 4, fastest) and `K1` (size K/4 = 4).
- `L{K1M}` -- lane = merge{K1, M}: `lane = m + k1*M` (M=16) => `[0,64)` = wave64.
- `V{K0}` -- register items = `{K0}`: `v = k0  in  [0,4)` => 4 A elements per lane.
- `M{4}` (C) -- unmerge M into `M0` (4, fastest), `M1` (4). `L{M1N}`: `lane = n + m1*16`;
  `V{M0}`: `v = m0` => `row = (lane//16)*4 + v`, `col = lane%16` -- **identical** to B.2. (ok)

This equivalence (descriptor => encoding => oracle) is the whole M1 correctness story.

---

## Appendix C -- File map & how to run

### Design docs (`helpers/tiling/docs/`)
- `tiling_design_proposal.md` -- **this doc** (top-down design + plan + status).
- `tiling_api_surface.md` -- the how-to-use catalog: every surface, default vs override, runnable examples.
- `tiling_review_report.md` -- full provenance: adversarial teardowns + two expert deep-dives + every decision (sec 0-sec 18; see Appendix D).
- `tiling_design_plan.md` -- the original approved plan. `MIGRATION.md`, `ERROR_HANDLING.md`, `README*.md` also live here.

### Package (`platform/python/rocke/helpers/tiling/`, import `rocke.helpers.tiling`)
- `encoding.py` -- `WarpDistributionEncoding` (the foundational coordinate-transform type; bijection-validated).
- `register_mapper.py` -- `RegisterMapper` / `LaneRegister` (pure-int `lane x register -> coord`).
- `descriptors.py` -- `TensorDesc` + `TensorWindow` (+ `make_tensor_desc` / `make_window`) -- the memory model.
- `fragments.py` -- `TileDesc` + `Fragment` (+ `make_fragment` / `fragment_length`) -- the register model.
- `emit.py` -- b-first verbs (`fill_fragment` / `load_fragment` / `store_fragment`) + `emit_tensor_coordinates` (IR-emitting `calculate_x`).
- `layouts/tile_distribution.py` -- `make_tile_desc` (the quantity-major human-approachable authoring surface).
- `mma/` -- `mma_operation.py` (`TileMma` + `Tiling`), `warp_encoding.py` (A/B/C encoders with `m_iter`/`n_iter`/`k_iter`).
- `traits/` -- `mma_traits.py` (typed loader/catalog) + `data/mma_traits.json`.
- `reflection/layout_visualizer.py` -- `describe()` + ASCII forward/inverse maps.
- `kernels/tiling_gemm_demo.py` (+ `tiling_gemm_manual_demo.py`) -- end-to-end kernels: `TilingGemmSpec` / `build_tiling_gemm` / `run_and_verify` (BUILT, bit-exact on gfx90a; torch-free numpy golden refs).
- Tests mirror the package at `platform/python/rocke/tests/helpers/tiling/` (95/95 pass).

### SOT & reference artifacts (`helpers/tiling/reference_docs/`)
- Internal reference material (the intrinsic support matrix / traits SOT, the interleave-layout transform formulas, and related notes) lives in the **gitignored** `reference_docs/` sibling and is NOT committed.

### rocke substrate seams (imported on integration; do NOT modify)
- `platform/python/rocke/helpers/distribution.py` -- `TileDistributionEncoding`, `make_static_tile_distribution`, `calculate_x`, `shuffle_tile`.
- `platform/python/rocke/helpers/atoms.py` -- `make_c_warp_dstr_encoding`, `c_warp_params` (the **oracle**).
- `platform/python/rocke/core/arch/target.py` -- `ArchTarget`, `MmaOp`, `_MMA_FRAGMENT_INFO`.
- `platform/python/rocke/helpers/mfma_gemm_inner.py` -- `mfma_k_loop`, `store_acc_to_global` (the **integration seam** for M1).
- `platform/python/rocke/core/ir.py` -- coherency consts, `ds_swizzle`, `ds_bpermute`, `permlanex16`, `ds_read_tr`.

### How to run
- Host is **gfx90a**. Tests (from `platform/python` on the path):
  `PYTHONPATH=platform/python <venv>/python -m pytest platform/python/rocke/tests/helpers/tiling/ -q` (95 pass).
- Demos: `PYTHONPATH=platform/python <venv>/python -m rocke.helpers.tiling.kernels.tiling_gemm_demo`
  (and `...tiling_gemm_manual_demo`) -> bit-exact (`max_abs_diff=0.0`).
- `mma_traits.json` is committed package data under `traits/data/`; regenerate it from the intrinsic
  support matrix (do NOT hand-edit).
- CK unification source (via WebFetch raw): `ROCm/rocm-libraries` branch `users/krithalith/ck/unification_all_fixes`, path prefix `projects/composablekernel/include/ck_tile/core/arch/mma/`.

---

## Appendix D -- Provenance report (`tiling_review_report.md`) section index

| sec  | Topic |
|---|---|
| 0 | Headline verdict (the plan built the anti-pattern generator before the thing that hides it) |
| 1 | rocWMMA adversarial teardown (4 limits + N1-N7 + what it got right) |
| 2 | rocke substrate teardown (S1-S9 + strengths) |
| 3 | Plan critique (G1-G10) |
| 4 | Revised direction (author-API-first, real relayout, open transforms, one SOT) |
| 5 | Anti-pattern check |
| 6 | Open questions (first round) |
| 7 | First expert deep-dive (7.1 architect, 7.2 GPU, 7.3 realist) |
| 8 | Converged M1 (first) |
| 9 | Author surface DECIDED (9.1 model, 9.2 must-specs, 9.3 GPU sharpenings, 9.4 resolves, 9.5 M1, 9.6 acceptance, 9.7 interleaved-formula work item, 9.8 Mma = target-binding line) |
| 10 | Fragment transforms & subtile iteration (10.1 interleave_idx, 10.2 transform_fragment, 10.3 dataflow, 10.4 shape-sets-mode, 10.5 validation, 10.6 M1 impact, 10.7 atom_shape knob) |
| 11 | CDNA/RDNA transforms (11.1 RDNA4, 11.2 RDNA3, 11.3 op algebra + cost classes, 11.3b storage tradeoff, 11.3c gfx11 DECISION, 11.4 milestone impact) |
| 12 | Second expert deep-dive -> ready for M1 (12.1 contract fixes, 12.2 HW reframes, 12.3 integration correction, 12.4 updated M1, 12.5 rollup) |
| 13 | Layout debugging / reflection / visualization |
| 14 | Whole-gamut dense/sparse/MX (14.5 = where `kind` lives) |
| 15 | Agentic-workflow-friendly requirement |
| 16 | Error handling = project error-handling convention |
| 17 | Public vs internal API boundary |
| 18 | M1 target gfx90a; MMA traits SOT = the intrinsic support matrix |
