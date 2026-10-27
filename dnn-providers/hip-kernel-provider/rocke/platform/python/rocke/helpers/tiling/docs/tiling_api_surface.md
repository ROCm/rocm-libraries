# rocKE Tiling Primitives -- API Surface, Options & Composability

**Status:** reflects the BUILT M1 + wave-tile subtiling + clipping surface (95/95 tests, bit-exact on gfx90a).
**Companion to:** `tiling_api_contract.md` (the WHY: promotion gate, audience tiers, glass-box mandate,
the clipping/N-D rules). This doc is the **how-to-use** catalog: every API surface, its
**default (MMA-driven) mode**, and its **manual override**, with runnable examples that exercise the
extents of the API. (One-home rule: the *why* lives in the contract, the *how* lives here.)

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
| `make_tensor_desc(lengths, strides, dtype)` | ptr-free strided memory descriptor, global or LDS (sec 3) | no (IR-free) | strides = data layout; `lengths` = valid extent (the default clip); axes in the one convention (sec 3) |
| `make_window(tensor, origin)` / `TensorWindow.with_origin(axis, origin)` | a tile's position on a tensor: one origin per tensor axis (int or i32 IR value; negative allowed) | no | no length of its own: the valid region is the tensor's `lengths`; `with_origin` moves one axis |
| `exit_if_tile_outside(b, origin, length)` -> i32 value | the prologue guard on a block origin: the workgroup returns unless `0 <= origin < length` (half-open; only the origin is guaranteed, not the whole tile) (sec 3) | yes (b-first) | use the value it returns, which carries the fact; `length` is the tensor's own int or length Value |
| `explain_bounds_checks(b, ptr, window, desc)` -> `BoundsCheckPlan` | which bound checks a global access of `desc` through `window` needs, and how they are enforced -- what `load_fragment` / `store_fragment` emit (sec 5b) | yes (reads the IR; emits nothing) | `print(plan)`: one line per tensor axis and side; values `AxisChecks` / `DomainChecks` of `SideCheck`, `VectorValidity`, `AccessPath`, `GlobalForm` (`plan.form`, the `addresses` line) |
| `check_gemm_shapes(a, b, c=None)` | build-time check that one GEMM's tensors agree on their extents (sec 3) | no (emits no IR) | reads A `(M, K)`, B `(N, K)`, C `(M, N)` positionally, batch axes first |
| `make_tile_desc(shape=, thread_dist=, thread_tile=, ...)` | author a logical->register layout (quantity-major) | no | for MMA operands use `mma.a/b/c_desc` instead |
| `TileDesc(shape, layout)` | dtype-free logical->register layout value | no | `mma.a/b/c_desc`, self-composed, or `make_tile_desc(...)` |
| `make_fragment(tile_desc, dtype)` | realized per-lane registers | no (value filled by verbs) | dtype from the operand |
| `fill_fragment(b, frag, 0)` | element-wise materialize | yes (b-first) | -- |
| `load_fragment(b, ptr, window, desc, lane, *, pad=0, lds_swizzle=False)` | memory -> Fragment (zero-pads OOB) | yes | addressing from `desc.layout`; `pad` = clip fill (0); `lds_swizzle` = bank-swizzle POLICY (§5c) |
| `store_fragment(b, ptr, window, frag, lane, *, lds_swizzle=False)` | Fragment -> memory (drops OOB) | yes | cast frag dtype -> desc dtype; `lds_swizzle` = bank-swizzle POLICY (§5c) |
| `TileMma(shape, a,b,c, target, tiling=, style=)` | intrinsic resolver + subtile driver | no | atom/layouts/op_id/wave_size from traits; `style=` picks the operand-layout profile |
| `Tiling(atom_shape=, order=, mac_prio=)` | the MMA object's knobs | no | atom=shape (single MMA), order="MNK", mac_prio=0 (setprio off; §5d) |
| `CanonicalStyle()` / `InterleavedStyle()` | the operand-layout profile (`style=`) | no | default canonical; interleaved = wide free-contiguous loads |
| `InstrClass` / `derive_sched_group_counts(plan, ...)` | `sched_group_barrier` scheduling — class masks + per-trip counts | no (`InstrClass` int mask; verb IR-free) | scheduling cadence is a perf KNOB, off by default (§5d) |
| `tensor_transforms`: `make_transform_graph` + kind builders, `emit_accesses` | coordinate graph (loop coordinates -> tensor coordinates), split element address + derived bound checks | graph: no; `emit_accesses`: yes (b-first) | checks derived from ranges vs the tensor's lengths; import from `rocke.helpers.tiling.tensor_transforms` (sec 3b) |

Free `make_*` factories over thin value objects (`TensorDesc`/`TensorWindow` in `descriptors.py`;
`TileDesc` in `fragments.py`) and one register carrier (`Fragment`); three verbs (`emit.py`); one
driver (`mma/`). The factories are the public surface -- you never call a method on the descriptor
to make a window (the descriptor is a pure memory layout; `ptr` binds at the verb, not on the
descriptor).

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
(The clip is the `TensorDesc` `lengths`; for a full, tile-aligned matrix that is a no-op,
byte-identical to no-clip. A narrower region is a narrower tensor description.)

What you did **not** have to say (all MMA-driven): the concrete intrinsic (`op_id`), the wave
size, the per-lane register counts, the lane->element mapping, the number of atoms, or the
accumulator width. See sec 8 for the full "you didn't have to say it" list.

---

## 3. `TensorDesc` -- ptr-free strided memory descriptor

Built with `make_tensor_desc(lengths, strides, dtype)`; `ptr` binds later at the verb. `lengths`
are the valid (compute) extent and the only clip a window has; `strides` are the physical
layout (a stride-1 axis is the memory-contiguous one). The same description serves global memory
and LDS: an element's index is `sum(position * stride)` in both (see "LDS buffers" below).

### Default: lengths = valid extent, strides = the data layout
```python
a_td = make_tensor_desc((M_LEN, K_LEN), strides=(K_LEN, 1), dtype=F16)   # A (M, K), K-contiguous
```
- `dtype` is the element type of memory; `load_fragment` reads it, `store_fragment` casts to it.
- `strides` ARE the data layout -- one generic verb serves A/B/C; the operand difference is
  entirely the strides + the layout, never the verb.

### One axis convention

This is the one home of the rule; other docs point here. Every tensor is written in one axis order,
and nothing in the tiling layer reads what an axis means (there are no axis labels):

- **Batch axes first, the reduction axis last.** A GEMM's K and a reduction's reduced axis sit last.
- **Memory layout in the strides only.** The axis order never changes with the layout; the stride-1
  axis is the contiguous one.
- **A rank-`r` tile covers the tensor's LAST `r` axes.** For a rank-`R` tensor, tile axis `i` is
  tensor axis `R - r + i`. Each leading axis is positioned by the window origin alone: tile extent 1,
  address term `origin * stride` (both sides of its origin are checked unless proven, as on every
  axis: an origin outside masks the whole tile). A tile whose rank exceeds the tensor's is refused.
- **Another axis order goes through `permute`** -- a strides view of the same bytes. Anything more
  general (merged or split axes, an implicit-GEMM convolution) goes through a transform graph.

```python
a3 = make_tensor_desc((BATCH, M, K), (M * K, K, 1), F16)   # (G, M, K), K-contiguous
a_win = make_window(a3, (batch, m_base, k))                # the batch axis is positioned by its origin
a_frag = load_fragment(b, a_ptr, a_win, mma.a_desc, lane)  # rank-2 (M, K) tile over the last two axes
```

**The GEMM reading.** A `(M, K)`, B `(N, K)`, C `(M, N)`; batched `(G, M, K)`, `(G, N, K)`,
`(G, M, N)`, batch first. It is the order the tile descriptors already use (`mma.a_desc` `(M, K)`,
`mma.b_desc` `(N, K)`, `mma.c_desc` `(M, N)`), GEMM "RCR", and conv NHWC / KYXC. CuTe / CUTLASS 3 and
CK contraction use the same `(N, K)` B; CuTe puts its batch axis last (its `L` mode), rocKE puts it
first.

**Say K-contiguous, N-contiguous, M-contiguous -- never row- or column-major.** "Row-major" is relative
to the order a tensor was written in, so for B it names opposite axes depending on whether B was
written `(K, N)` or `(N, K)`. The contiguous axis is unambiguous in every library. How B translates:

| B in memory | BLAS | Triton TMA | PyTorch | rocWMMA `matrix_b` | CuTe | rocKE |
|---|---|---|---|---|---|---|
| K-contiguous (N rows of K; an `nn.Linear` weight) | `transB=N`, `ldb=K` | `shape=[N,K]`, then `b.T` | `W.T`, strides `(1, K)` | `col_major`, `ldm=K` | `(N,K)`, stride `(ldB,_1)` | `make_tensor_desc((N, K), (K, 1), F16)` |
| N-contiguous (K rows of N) | `transB=T`, `ldb=N` | `shape=[K,N]` | `B`, strides `(N, 1)` | `row_major`, `ldm=N` | `(N,K)`, stride `(_1,ldB)` | `make_tensor_desc((N, K), (1, N), F16)` |

From PyTorch in one line: rocKE's B description is torch's `B.T` view,
`make_tensor_desc(tuple(B.T.shape), tuple(B.T.stride()), dtype)`.

Anti-example: `make_tensor_desc((K, N), ...)` for a GEMM's B. Read positionally, its "K" is the N
extent -- refused by `check_gemm_shapes` when the extents differ, caught only by a bit-exact test when
they do not. Write `(N, K)` and put the memory order in the strides.

### Runtime lengths and strides: parameter Values

A length or a stride is a compile-time int or an i32 `Value` whose facts are stated. The facts have
one home: the kernel parameter's declaration, which `KernelLauncher.from_artifact` checks on the host
before each launch, or the Value `b.gpu_checked_value` returned, which the GPU checks. Pass that Value
itself; nothing wraps it:
```python
M = b.param("M", I32, ge=1, le=1 << 20)
K = b.param("K", I32, ge=1, le=1 << 20)
a_td = make_tensor_desc((M, K), (K, 1), F16)   # one K Value: the length and the row stride
```
- **What a runtime extent needs.** A length is stated `>= 0` with an upper bound (`le=` or `lt=`), the
  number every range proof over it reads. A stride is stated `>= 1`; a relation carries it
  (`ldc = b.param("ldc", I32, ge=N)` with `N` declared `ge=1`). A stride with no upper word has the i32
  limit, so the build may choose 64-bit index forms. The Value is the same for the whole wave: a
  parameter, or a checked value computed from block ids or a `b.readfirstlane` result. A Value with no
  stated facts is refused, and so is one whose facts are too weak; each message names the fix.
- **The host check.** A kernel whose parameters state facts launches through
  `KernelLauncher.from_artifact(compile_kernel(kernel, arch=...))`, which checks every fact before every
  launch and raises `LaunchRefused` naming each failed one. `first_supported([aligned, ragged], args)`
  picks the first kernel that accepts a launch.
- **The contiguous axis** keeps the int stride 1.
- **LDS descriptors are static.** rocKE allocates LDS statically, with a fixed size, so an LDS tensor
  takes int lengths and strides only: `load_fragment` / `store_fragment` on an LDS buffer, the
  cooperative LDS copies and `element_span` refuse a tensor with a runtime length or stride, all with
  one message. A runtime window origin is fine.
- **`check_gemm_shapes`** agrees two runtime extents only when they are one and the same Value; two
  `gpu_checked_value` calls give two different Values, even on one loaded value.

### Pointer alignment: what a global access claims

A wide global access tells the compiler how aligned its address is, and that claim must be true: a
uniform address claimed 4-byte aligned or more may be read by a scalar load, which ignores the low
address bits. The verbs claim only what the facts prove -- the largest power of two dividing the
pointer's alignment and every access's element offset, at most the access's own width:
- **The pointer:** the `align=` of its kernel parameter (`b.param("A", PtrType(F16, "global"),
  align=16)`); a parameter without one, or a pointer computed from one, claims the element's size.
  Declare `align=` on every pointer parameter the host allocates aligned.
- **The offset:** every axis's position (an int origin by its value, a value origin by the multiple-of
  fact the IR range walk proves, e.g. `block_id * 32`), a slice's begin, and every stride (a runtime
  stride by its stated `multiple_of`).

So an f16 tile at an odd column, an odd begin or an odd stride claims 2 bytes whatever its width, and
`lda` stated `multiple_of=8` with `align=16` claims 16. The claim never narrows the access: the width
follows the layout and the strides (sec 3, LDS buffers, "Vector width").

### Block origins: `exit_if_tile_outside`

A block origin such as `block_id_y * TM` is a plain i32 value; nothing in the IR says the tile starts
inside the tensor. `exit_if_tile_outside` makes it true, once, in the kernel prologue:
```python
m0 = exit_if_tile_outside(b, b.mul(b.block_id_y(), b.const_i32(TM)), M)
n0 = exit_if_tile_outside(b, b.mul(b.block_id_x(), b.const_i32(TN)), N)
wave = b.div(b.thread_id_x(), b.const_i32(wave_size))
c_win = make_window(c_td, (b.add(m0, b.mul(b.div(wave, b.const_i32(waves_n)), b.const_i32(warp_m))),
                           b.add(n0, b.mul(b.mod(wave, b.const_i32(waves_n)), b.const_i32(warp_n)))))
```
- **What it guarantees.** Ranges are half-open (`[lo, hi)` holds `lo`, not `hi`). A workgroup whose
  origin is outside `[0, length)` returns before doing anything, so an over-launched grid or a remap
  that yields a negative origin touches no memory. Only the ORIGIN is guaranteed: a tile `[m0, m0 + TM)`
  fits when `length` is a multiple of `TM` (an int, or a Value stated `multiple_of` a multiple of `TM`);
  otherwise its upper side overhangs on the last tile and stays checked.
- **Use the value it returns.** The fact is a range marker on the returned value (written
  `by="exit_if_tile_outside"`); the IR range walk (`tensor_transforms`) reads it from every use, and
  from arithmetic on it such as a wave offset added with `b.add`. The raw origin passed in carries no
  upper end.
- **Rules.** Call it in the kernel's top level, never inside a loop or an `if`. The origin is an i32 IR
  value (an int origin needs no guard) and the length is the tensor's own int or length Value (one
  with stated facts, the same Value the tensor description holds). Both must be the same for the whole
  workgroup -- block ids, kernel parameters, constants, or a `b.readfirstlane` of a load at such an
  index through a parameter declared `readonly=True, noalias=True` -- so every wave leaves together and
  no barrier waits on a departed wave. A batch index is guarded the same way:
  `g = exit_if_tile_outside(b, b.block_id_z(), G)`. State the workgroup size right after creating the
  builder.

### `check_gemm_shapes(a, b, c=None)` -- the tensors of one GEMM agree

A pure build-time check (emits no IR). It reads its tensors positionally as A `(batch..., M, K)`,
B `(batch..., N, K)` and, when given, C `(batch..., M, N)`: equal ranks of at least 2, one extent per
batch axis across the tensors, A's K is B's K, A's M is C's M, B's N is C's N. Two extents agree when
they are equal integers (Python or numpy), or one and the same extent object.
```python
check_gemm_shapes(a_td, b_td, c_td)       # once per GEMM, on the logical tensors
check_gemm_shapes(q_td, k_td)             # c=None: the A / B pair alone (a fused chain's Q*K^T)
```
- **When to call it.** Once per GEMM whose A, B and C are tensors, on the logical tensors: on the
  `(M, N)` C the verbs see (`c_td.permute([1, 0])` where a kernel spells C `(N, M)`), and for a kernel
  that crosses A and B, on the C before the crossed view. With `c=None` on each A / B tensor pair whose
  C is never a tensor. Nothing calls it for you (`TileMma` sees no tensors).
- **Its domain** is GEMMs whose A, B and C are plain-window tensors in that shape family, their batch
  axes shared one to one. Two GEMMs are outside it and make no call:
  - one that reaches its operands through a transform graph (an implicit-GEMM convolution): its
    tensors are the conv's own and the graph defines M, N and K (K = `Y * X * C`), so read positionally
    the wrong axes would be compared -- a correct conv refused, or passed by coincidence;
  - one that sums over a batch axis (A `(G, M, K)`, B `(G, N, K)`, C `(M, N)`): C has no `G`, so the
    rank rule refuses it. Check its A / B pair with `c=None`.

  With no axis labels the check cannot tell such a tensor from a mistake, so respecting the domain is
  the caller's job. A GEMM with a register operand (`P*V`) has no tensor to check.
- **What it catches:** a transposed or mis-sized operand whenever the swapped extents differ, and every
  batch-extent mismatch. **What it cannot:** a square transposed operand (M = N = K) or a forgotten
  call -- bit-exact tests on integer inputs stay the backstop.
- Tensors are compared in logical elements. Mixed-width MMA operands hold the same number of logical K
  elements, so the comparison is right for them; a packed sub-byte tensor is described in elements,
  never bytes or words. Scale tensors are not A or B and are not passed to it.

### Override: express any data layout via strides
```python
a_td = make_tensor_desc((M_LEN, K_LEN), (K_LEN, 1), F16)   # A (M, K), K-contiguous
b_td = make_tensor_desc((N_LEN, K_LEN), (1, N_LEN), F16)   # B (N, K), N-contiguous
c_td = make_tensor_desc((M_LEN, N_LEN), (N_LEN, 1), F16)   # C (M, N), N-contiguous
```
`permute` presents the same bytes in another axis order: on the N-contiguous B above,
`b_td.permute([1, 0])` is its `(K, N)` view with strides `(N_LEN, 1)` -- a view, not the operand order.

### Override: dtype is whatever the pointer is (no dtype baked into verbs)
```python
a_td = make_tensor_desc((M_LEN, K_LEN), (K_LEN, 1), BF16)  # bf16 in
c_td = make_tensor_desc((M_LEN, N_LEN), (N_LEN, 1), F32)   # f32 store (no narrowing cast)
```

### LDS buffers: a flat allocation, described by its descriptor

An LDS buffer is a flat (rank-1) allocation. Its layout lives in an ordinary `TensorDesc` -- logical
axes in the convention order, int lengths, int strides -- and nowhere else. Size the buffer from the
descriptor with core's typed allocation and hand both to the same verbs as global memory:
```python
b_lds_td = make_tensor_desc((N, K), (1, N), F16)               # B staged as [K][N], written (N, K)
smem = b.smem_alloc(b_lds_td.dtype, [b_lds_td.element_span])   # flat, typed, sized in elements
store_fragment(b, smem, make_window(b_lds_td, (n0, k0)), b_frag, lane)
b_frag = load_fragment(b, smem, make_window(b_lds_td, (n0, k0)), mma.b_desc, lane)
```
- **Addressing.** The verbs see an LDS buffer and compute `sum(position * stride)` exactly as for
  global memory: the lane's base index once, plus a compile-time offset per register run. The vector
  width follows the same rule as global: a register run on a stride-1 axis is consecutive elements.
- **`TensorDesc.element_span`** is `1 + sum((length - 1) * stride)`: the elements from the first to the
  last element the description can address. Static (int lengths and strides); a negative stride is
  refused. The LDS budget stays a byte count: the span times the element size.
- **`lds_element_span(td, lds_swizzle)`** is the size to allocate: `element_span`, except that the
  built-in `lds_swizzle=True` moves an element of the last axis within its 64-element group, so the last
  row reaches the end of its group. A `(8, 48)` buffer with strides `(64, 1)` spans 496 elements but
  needs 512 swizzled. A swizzle callable must keep every position inside the descriptor's lengths.
- **Padding** is a larger outer stride: a `(rows, cols)` buffer padded by `pad` elements per row is
  `make_tensor_desc((rows, cols), (cols + pad, 1), dtype)`.
- **Vector width.** The verbs pick a width that divides every stride except the stride-1 axis's,
  leading axes included (in global memory too; a runtime stride by its stated `multiple_of`): a pad or
  batch stride that is not a multiple of the natural vector narrows the access. In global memory the
  alignment the access claims is a separate fact ("Pointer alignment" above).
- **Stages** are a leading axis positioned by the window origin: a `[stage][K][F]` buffer is the
  rank-3 `make_tensor_desc((stages, F, K), (K * F, 1, F), dtype)`, written and read with an unswapped
  rank-2 `(F, K)` tile at origin `(stage, f0, k0)`. Derive both operands' stage origins from one stage
  index.
- **Checks.** Before emitting, the verbs refuse an LDS buffer that is not rank 1, whose element type is
  not the tensor's dtype, or that holds fewer than `lds_element_span` elements; the built-in
  `lds_swizzle=True` needs the stride-1 axis last and every other stride at least the swizzled row
  (the last length rounded up to 64), so rows cannot overlap (otherwise pad the stride or pass a
  swizzle callable). LDS accesses are
  unclipped, so a disagreement is refused rather than masked. Window origins are trusted, as for every
  LDS access.
- **The cooperative-load bridge** (`memory.py`, toolbox). `lds_tile_desc(coop_desc, dtype, *, buffers,
  n_waves, wave_size=64)` returns the full-tile LDS descriptor of a cooperative-load operand in the
  convention order `(free, K)`: lengths `(buffers * free, K)`, strides `(1, buffers * free)`, buffer `i`
  starting at free position `i * free`. `lds_store` / `lds_read` take the cooperative and read tiles
  unswapped, at origins `(free_position, k)`, and check each tile extent against the descriptor's
  lengths. Its stride-1 axis is axis 0, so a swizzle on it is a swizzle callable.
- **Why no allocation verb.** `smem_alloc(dtype, [count])` already takes the two values the descriptor
  gives; a tiling wrapper would be a second home for them.

### Reserved (post-M1)
- per-load `coherency=` (param exists on the verbs; any value but 0 raises `NotImplementedError`).

---

## 3b. Coordinate graphs -- `tensor_transforms`

When a tensor is not a plain strided view of the loop coordinates (an implicit-GEMM convolution's
`k -> (y, x, c)` with a halo, an LDS swizzle), describe the mapping as a coordinate graph and let
`emit_accesses` give each access's split element address and exactly the bound checks that can fail. The
graph holds no tensor and no bound: the `TensorDesc` above stays the one home of lengths and strides. Import
it from `rocke.helpers.tiling.tensor_transforms`. The API (units, kinds, rules, every error) is
[`tensor_transforms_api.md`](tensor_transforms_api.md); worked examples are
[`tensor_transforms_guide.md`](tensor_transforms_guide.md).

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
- `tile_desc`/`dtype` are read-only; `value` is the one writable field (re-bound each K step:
  `accumulator.value = b.mma(...)` for a single-atom tile -- wave shape == atom shape, one
  `b.mma`; a multi-atom tile goes through `TileMma`). The write and read checks are documented on
  `Fragment` (`fragments.py`).

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
of the tile forces tensor padding or a separate edge kernel. Here clipping follows from the
**tensor's lengths** -- no separate remainder path.

- The window carries only the position; the valid region is the `TensorDesc` `lengths` (the one home
  of a length). A narrower region is a narrower tensor description; a negative origin (a convolution
  halo) reads 0 and writes nothing before the tensor start.
- **Default = ZERO-PAD:** an element outside the tensor reads 0 and is not written; no instruction
  touches it. `load_fragment(..., pad=0)` is the default; `pad != 0` (constant fill) is reserved and
  raises `NotImplementedError` on every path.
- **Checked unless proven:** per tensor axis and side, the shared bound-check rule
  (tensor_transforms `required_checks`) asks whether the window's positions -- each origin's range
  as the IR range walk reads it, plus the tile's extent -- can leave `[0, length)`. A side that
  cannot is proven and emits nothing; every other side is checked. Nothing is trusted: an origin
  the IR says nothing about keeps both sides checked, and a tile alignment is not a proof. Guard
  block origins with `exit_if_tile_outside` (sec 3) so a tile-multiple length proves the whole tile.
- **One code for every tile:** the decision is made at build time per window; a ragged kernel runs
  its checked code on interior tiles too. The fast path for tile-multiple shapes is a second kernel
  (lengths stated `multiple_of` the tile), picked on the host with `first_supported`.
- **Loops over a ragged axis:** a loop counter is an origin like any other, its range read from the
  loop's start, stop and step. The two kernels write the K loop differently:
  ```python
  # aligned kernel (K declared multiple_of=TK): the last trip is K - TK, every K tile is whole
  with b.scf_for(b.const_i32(0), K, b.const_i32(TK)) as k: ...          # K side proven
  # ragged kernel: full tiles in the loop, the last partial tile (the tail tile) after it
  k_stop = b.add(K, b.const_i32(1 - TK))                                # proves k + TK <= K
  with b.scf_for(b.const_i32(0), k_stop, b.const_i32(TK)) as k: ...     # K side proven
  k_tail = b.mul(b.div(K, b.const_i32(TK)), b.const_i32(TK))            # K side checked
  ```
  The loop `[0, K)` over a `K` with no `multiple_of` fact keeps the K side checked on every trip
  (its last trip may be partial): correct, slower. When the window origin holds the counter as a
  summand of its own (`k`, `k + c`, or `k` plus a per-lane offset), `print(explain_bounds_checks(...))`
  ends that side's line with the fix, e.g. `-- state multiple_of=16 on K's b.param declaration` for
  `TK = 16` (or, for a length checked on the GPU, the `b.gpu_checked_value` call to state it in).
- **What is emitted:** a proven window is unmasked -- one i32 element index from the pointer when
  every stride is an int and every offset fits i32, else the per-wave address on the 64-bit pointer
  plus a 32-bit per-lane offset (a constant offset outside i32 goes on the 64-bit pointer too: an
  offset is an i32 index only when it fits i32). A checked window keeps its vector width and masks each access by
  its validity: a vector wholly valid is one wide access, a vector that may cross a bound has each
  valid element accessed alone, and an address an invalid lane computes carries no in-bounds promise.
  Every form but the one-index form does arithmetic on the 64-bit pointer, which needs a global
  pointer: a pointer parameter declared `addr_space="constant"` is refused there, at build, by the
  verbs and `explain_bounds_checks` alike (declare it without `addr_space`).

```python
# ragged 255x255x255 over a 256^3 tile: the edge CTAs clip internally, zero-padded load / dropped store.
a_win = make_window(a_td, (m, k))                 # the clip is a_td.lengths (the compute extent)
print(explain_bounds_checks(b, a_ptr, a_win, mma.a_desc))     # which sides are checked, and how
a_frag = load_fragment(b, a_ptr, a_win, mma.a_desc, lane)     # OOB -> 0
store_fragment(b, c_ptr, make_window(c_td, (m, n)), acc, lane)  # OOB writes dropped
```
`explain_bounds_checks` takes the builder, pointer, window and tile the verb is given and reports
exactly what the verb emits; an LDS buffer is refused (LDS accesses are never bound-checked).

---

## 5c. LDS bank-swizzle policy (Part C, BUILT)

`load_fragment`/`store_fragment` take `lds_swizzle`, a **customizable bank-swizzle POLICY** for LDS accesses
(no effect on a global `ptr`). It is `bool | Callable`:

- `False` (default) — no swizzle; the natural contiguous vector width is used.
- `True` — the built-in block-preserving swizzle (`_swizzle_lds_positions`).
- a **callable policy** `(builder, positions) -> positions` that remaps the LDS index. `positions` holds one
  position per tensor axis, leading axes (their window origins) first, so an element gets one address
  whatever the rank of the tile that reaches it. It declares its granularity via a `vw_elems` attribute (the block width, in elements, it keeps contiguous); the emit
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

## 5d. Instruction scheduling — `mac_prio` + `sched_group_barrier` (perf KNOBS, Part C, BUILT)

Two levers control how the compiler schedules the steady-state K-loop. Both are **perf-tuning KNOBS**
— off by default (the recording stays byte-identical), reorder ISSUE not the math (bit-exact), and are
**measured, not derived** (sweep per arch/shape; the wins to date are gfx90a-shaped).

- **`Tiling(mac_prio=0..3)`** — wave issue priority (`s_setprio`) raised after the FIRST matrix atom of
  each cluster and dropped after the last. The **driver owns the placement** (`mma/driver.py`); a
  single-atom cluster emits nothing. `0` = off. Raising *after* the first atom (not before) is the
  measured-stable placement.
- **`InstrClass` + `derive_sched_group_counts(plan, *, tile_m, tile_n, n_waves, vw)`**
  (`scheduling.py`, both package-root exports) — author a `sched_group_barrier` cadence directly.
  `InstrClass` is the instruction-class mask vocabulary (`MFMA` / `DS_READ` / `DS_WRITE` /
  `VMEM_READ` / `VMEM_WRITE` / `VALU`; the value IS the mask). The helper returns the per-class
  instruction counts for one trip — MFMA from `plan.mfma_count`, DS/VMEM from the tile geometry — so
  the count math has ONE home and cannot drift. `vw` (the kernel's wide-mem access width) is
  **required** — a per-kernel choice, not derivable from the dtype. The cadence *structure* stays
  local to each kernel; only the counts are shared. Mutually exclusive with `iglp_opt`.

The demos (`tiling_gemm_{interwave,pipelined,interleaved}_demo.py`) show real, non-trivial cadences.

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
| load/store alignment | the pointer's `align=`, the origins, a slice's begin and the strides (what they prove) | (internal) |

Override any single one without touching the others -- see the composability matrix.

---

## 9. Composability matrix (extents of the API)

Each knob is independent; combine freely. `BUILT` = works today on gfx90a; `RESERVED` =
designed-in seam, not yet implemented.

| Axis | Default | Override options | Status |
|---|---|---|---|
| operand dtype | -- | f16, bf16, f32, f8/bf8 (via desc + `a/b/c=`) | BUILT (f16->f32 proven; casts honest/fail-fast) |
| data layout | -- | any via `make_tensor_desc(..., strides)` + `.permute([...])`, axes in the one convention (sec 3) | BUILT |
| tile distribution | `mma.a/b/c_desc` | author any via `make_tile_desc(...)` (quantity-major) | BUILT |
| atom selection | single MMA (`None`) | shape tuple (agnostic) / intrinsic name (pinned) | BUILT |
| wave tile size | == atom | any integer multiple of atom (M and/or N and/or K) | BUILT |
| subtile order | `"MNK"` | any of 6 `MNK` permutations | BUILT |
| target | -- | any gfx via `target=` / `arch=` (agnostic authorship) | BUILT (gfx90a proven; RDNA resolves) |
| clipping | desc `lengths` (auto) | a narrower `make_tensor_desc` -> zero-pad load / drop store | BUILT (bit-exact incl within a valid space) |
| clip fill | 0 (zero-pad) | `pad=` on load (`constant(value)`) | RESERVED (`pad!=0` raises) |
| C transpose | off | `c_transpose=True` on `TileMma` (-> C^T, `c_desc` = (N,M)) | PLANNED (analyzed; Part D) |
| interleaved layout | `style=` (default `CanonicalStyle`) | `TileMma(..., style=InterleavedStyle())` | BUILT (dense square atoms; see `tiling_interleaving_design.md`) |
| operand layout | `mma.a/b/c_layout` | `make_tile_desc(...)` / `custom_layout(...)` | BUILT (make_tile_desc) / RESERVED (custom_layout) |
| memory space | global | LDS: a `TensorDesc` over a flat `smem_alloc` buffer, same verbs (sec 3, "LDS buffers") | BUILT |
| coherency | default | `CACHE_STREAM` / `NON_TEMPORAL` per load | RESERVED (`coherency!=0` raises) |
| kind | dense | `sparse(...)` / `scaled(...)` (+index/scale operands) | RESERVED |

---

## 10. Worked progression -- one kernel, escalating overrides

The same GEMM, adding exactly one override at a time -- the loop body never changes, only the
`TileMma` construction. Knobs travel as constructor arguments (`shape`, `tiling=`, `style=`).

```python
# (0) DEFAULT: single 16x16x16 atom, f16->f32, MNK order. Say almost nothing.
mma = TileMma((16, 16, 16), a="f16", b="f16", c="f32", target="gfx90a")

# (1) BIGGER K per tile: 4 K-subtiles, MMA owns the inner K loop.
mma = TileMma((16, 16, 64), a="f16", b="f16", c="f32", target="gfx90a",
              tiling=Tiling(atom_shape=(16, 16, 16)))

# (2) COOPERATIVE M/N: a 2x2 output grid, MMA walks it internally.
mma = TileMma((32, 32, 16), a="f16", b="f16", c="f32", target="gfx90a",
              tiling=Tiling(atom_shape=(16, 16, 16)))

# (3) FULL GRID + iteration order: 2x2x2, K outermost.
mma = TileMma((32, 32, 32), a="f16", b="f16", c="f32", target="gfx90a",
              tiling=Tiling(atom_shape=(16, 16, 16), order="KNM"))

# (4) PIN THE EXACT INTRINSIC by name (target-specific escape hatch).
mma = TileMma((32, 32, 16), a="f16", b="f16", c="f32", target="gfx90a",
              tiling=Tiling(atom_shape="mfma_f32_16x16x16f16"))

# (5) SWITCH THE OPERAND-LAYOUT PROFILE: wide free-contiguous loads.
mma = TileMma((32, 32, 16), a="f16", b="f16", c="f32", target="gfx90a",
              tiling=Tiling(atom_shape=(16, 16, 16)), style=InterleavedStyle())
```

Each step is one argument; the loop body (`load_fragment -> mma -> store_fragment`) is unchanged.
(0)-(4) resolve the SAME layouts and are bit-exact; (5) changes the operand register layout -- the
`style` picks the profile, validated identically to a derived one (the accumulator stays derived --
see the C-oracle in `TileMmaPlan`). That is the composability claim, made concrete.

---

## 11. Customising -- your own atom, layout, or style

- **Pin the atom** with `Tiling(atom_shape=...)` -- a shape tuple, an intrinsic name (escape hatch),
  or `None` for a single-MMA wave.
- **Bring your own operand layout** by holding `mma.plan` and building a `Fragment` from your own
  `TileDesc`; the driver validates it identically to a derived one (`mma_operand_layout_sound` +
  `mma_pair_k_aligned` + the SOA atom-contiguity slice guard). Overrides are OPERANDS only -- the
  accumulator is always derived, and checked by `mma_accumulator_flow_consistent`.
- **Add a layout style** by subclassing `LayoutStyle` (`mma/styles/`): implement `operand_desc` +
  `accumulator_desc` composing the public primitives (`cooperative_load_desc`, `TileDesc.swap_dims`
  / `.reorder_registers`, the `lds_*` bridge). See `docs/tiling_api_contract.md` -- "Adding a layout
  style" -- for the interface contract, which gates police it, and the reach-past trigger.

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
| clipping | desc `lengths` (auto zero-pad) | a narrower `make_tensor_desc` |
| target | -- | `target=`/`arch=` (agnostic) |
| coherency | default | `coherency=` (reserved) |
| memory space | global | LDS: `TensorDesc` + flat `smem_alloc(td.dtype, [td.element_span])` |
| kind | dense | `sparse`/`scaled` (reserved) |
