# rocKE Tiling -- Visualization API Surface

**Status:** reflects the BUILT visualization package (`rocke.helpers.tiling.visualization`).
**Companion to:** `tiling_api_surface.md` (the emit/transforms primitives this package renders) and the
`/layout-viz` skill (which owns the *preferences* -- which view + settings for which image; this doc is the
objective **what-you-can-call** catalog, shared by any skill that renders layouts).

> **Location:** `platform/python/rocke/helpers/tiling/visualization/` (package
> `rocke.helpers.tiling.visualization`). Import everything from the package root, not the submodules.
> **matplotlib is LAZY:** importing the package pulls no plotting stack; a figure is only built when a view
> actually renders. `_canvas`, `draw_*`, and the conflict drawers stay internal behind submodules.
> **Placement, not verdicts:** the LDS views show WHERE data lands (banks/depth). Bank-*conflict* claims are
> the empirical `/bank-conflict` path, never inferred from these pictures.

---

## 0. The one idea

Build every view from a few **cell-field components** (a coloured grid: hue = thread, shade = vectorized
transaction time order) and compose them left-to-right on a **linear `Pipeline` spine**. Each component is fed EITHER by a
`WarpDistributionEncoding` OR by an explicit forward map `{(lane,reg)->coord}` (so a DERIVED distribution
that is not a representable encoding still renders). Two render entries:

- `Pipeline.render(...)` -- **shared axes** (one deduped distribution header + legend), stages top-aligned.
- `Pipeline.render_panels(...)` -- **a row of independent panels at a common height** (each with its own
  info box; one shared legend). Serves both wave- and macro-scope flows.

---

## 1. The surface at a glance

| Surface | What it is | Fed by | Produces |
|---|---|---|---|
| `RegisterFileComponent` | physical `tid x vreg` grid, cell = held coord | `dist=` or `fwd_map=` | a stage component |
| `LogicalTileComponent` | logical `MxK`/`NxK`/`MxN` grid, cell = owner/coord | `dist=` or `owner_map=` | a stage component |
| `LdsBankView` | `depth x banks` LDS placement grid | `mp=` + `addr_fn=` (+`flow_map=`) | a stage component |
| `MmaTee` | the A/B -> derived-C tee (one composite figure) | `MmaTee.from_mma(mma, ...)` | standalone figure |
| `WaveStrip` | macro register stage: N per-wave 64-lane files side by side | `WaveStrip.from_fwd_map(...)` | a stage component |
| `FlowStage` / `Pipeline` | the linear spine (name, component, transform per hop) | a tuple of `FlowStage` | `.render` / `.render_panels` |
| flow recipes | prebuilt Pipelines for a hop or a kernel phase | descriptors / encodings | a `Pipeline` |
| adapters | descriptor -> viz inputs | a `TileDesc` / encoding | maps + `addr_fn` |
| colour model | `ACCENTS` / `NACC` / `accent_tint` | -- | RGB for custom draws |
| text reflection | `describe` / `render_forward_map` / `render_inverse_map` | an encoding | strings (no matplotlib) |

---

## 2. Colour model

- `ACCENTS` -- the ordered accent palette; `NACC` = its length (8). `accent_tint(accent_idx, step, nsteps)`
  -> an RGB tint (darker = earlier).
- **Convention (uniform across every view):** **hue = thread** (lane % `NACC`; `color_mode="first8"` colours
  lanes 0-7 and greys the rest, `"full"` colours all), **shade = vectorized transaction time order** -- one shade per
  *vectorized access*, darkest issued first (the shade count per thread = `regs / VW`, not the register index).
  **`VW` here is the STRIDE-DERIVED transaction width** (stride-1 run, widest legal `<= b128`), NOT the recorded
  emit `vw` (which is pre-coalescing, often 1). `objdump` is a validation gate only, never the width source.

---

## 3. Cell-field components

All three share: `color_mode`, `groups`, `shade_map`, `highlight`/`highlight_color`, `scope`
(`"wave"` per-cell detail | `"macro"` one block per wave, hue = wave), `dense_rows` (see below),
and cosmetic `cell_w`/`cell_h`/`font_size`/`col_ticks_side`/`row_ticks_side`/`origin`/`title`.

Group **detail levels** (`CellGroup(members, detail, name)`; `RegGroup`/`LogicalGroup` are aliases):
`"detailed"` (label every cell) · `"grouped"` (bordered anchor cell + one summary label) · `"block"`
(solid fused block, ONE centred summary, no anchor -- the 1-D / macro form) · `"plain"` (border only).

### `RegisterFileComponent(dist=None, fwd_map=None, dims=("M","K"), ...)`
Physical register file (`tid` rows x `vreg` cols by default; `row_axis`/`col_axis` swap). `dtype_bits`
sets the 32-bit register ruler. `dense_rows=40`: a file taller than this collapses each lane row to a
`block` summary (readability at wave64).

### `LogicalTileComponent(dist=None, owner_map=None, dims=("M","K"), row_coord=0, mode="layout", ...)`
The dual: the logical matrix, cell = owner `T{lane}R{reg}` or the coord. `label_coords="logical"` (default)
shows the coord; `"register"` shows the owner. `mode` = `"layout"` | `"thread_tile"` | `"coalescing"`
(auto-derives borders + a vectorization shade). `text_map` makes the grid a POSITION grid with a flowed
label per cell (the C-body trick). `atom`, `addr_fn`, `detail_first` refine borders/shade.

### `LdsBankView(mp=None, addr_fn=None, nbanks=REQUIRED, elem_bytes=REQUIRED, lds_base_bytes=0, ...)`
`depth x banks` LDS placement. Fed by the store register map `mp={(lane,reg)->(row,col)}` + an `addr_fn`
`(row,col)->element address`; `flow_map` supplies the flowed logical label.

- **`nbanks` and `elem_bytes` are REQUIRED (no default)** -- state the arch bank count and the dtype width;
  a silent default is exactly what mislabels f16 as one element per bank. (gfx90a/CDNA -> `nbanks=32`;
  f16 -> `elem_bytes=2`; f32 -> `elem_bytes=4`.)
- **`lds_base_bytes` (default 0)** -- this operand/buffer's LDS byte offset. A bank is 4 bytes, so the
  physical map is `dword = (lds_base_bytes + elem*elem_bytes) // 4`, `(depth, bank) = divmod(dword, nbanks)`;
  `4//elem_bytes` elements pack per bank. The base shifts every element's **bank by `base_dwords mod nbanks`**
  and its **depth by `base_dwords // nbanks`** -- so B / buffer-1 do NOT start at address 0. Default 0 =
  A / buffer-0 / a standalone single-buffer tile. (The `addr_fn`'s row stride must ALSO be the true memref
  stride -- e.g. `_bufs*free` for a double-buffered `[tile_k, _bufs*free]` tile, not `free`.)
- `label_by` = `"flow"` (logical datum) | `"thread"` (`T{lane}R{reg}`). `compact_rows=False` keeps the TRUE
  physical depths (store stride shows as gaps); `True` drops empty rows (denser, hides the stride).
  `dense_rows=40`: above this many depth rows, each occupied row collapses to one `block` summary label.

---

## 4. Composite views

### `MmaTee.from_mma(mma, **overrides)` (or the dataclass with `a_enc`/`b_enc`/`c_enc`/`atom_shape`)
The A (left wing) / B (top wing) -> **derived C** tee in one figure. `from_mma` fills canonical refs +
dtypes from the atom; override `a_enc=`/`b_enc=` to render a non-canonical (interleaved) operand. Inside the
tee the **B and C wings are transposed** (tid horizontal) to align K/N -- this is the ONLY place a register
file is transposed. Knobs: `issue_order` (C register grouping), `show_logical_inputs`/`show_static`,
`trace_a`/`trace_b`, `show_diagnostics` (soundness/K-match banner). `.render(out_dir, name)` -> path;
`.c_mapping()` -> `{(lane,reg)->(m,n)}`.

### `WaveStrip.from_fwd_map(fwd_map, *, dims, wave_size=64, dtype_bits=16, shade_addr=None, ...)`
The macro register stage: N per-wave 64-lane `RegisterFileComponent`s side by side (each with its wave's
flowed data), never a single `n_waves*64`-tall monolith. A drop-in `Pipeline` stage component.

### `render_coalescing(report, out, *, dtype_label, lane_group=16, title="", instruction=0)`
The ADDRESS-SPACE coalescing figure -- fused-vs-scattered made literal. A pure consumer of a
`CoalescingReport` (`analysis.coalescing.analyze_coalescing(fwd, dims, strides, dtype_bits, direction=,
line_bytes=)`, or `coalescing_probe.report_for_transaction(txn, arch=, dims=)` to DERIVE the descriptor
from a recorded transaction). Fixed conventions (reproduce EXACTLY -- do not re-style per case):
- **Per-SIMD grids.** The wave is split into `ceil(nlanes / lane_group)` stacked grids, `lane_group=16` (a
  SIMD is 16 lanes on BOTH CDNA and RDNA) -- so wave64 -> 4 grids, wave32 (RDNA) -> 2. Each SIMD's address
  spread is local, so its own grid reads far tighter than 64 lanes over one axis.
- **Lane (y) axis = GLOBAL lane ids, continued.** SIMD 0 = lanes `0..15`, SIMD 1 = `16..31`, ... (never
  rebased to `0..15` on every grid). Power-of-2 lane tick distance.
- **Address (x) axis = LINEAR, cropped to the TOUCHED cache pages only.** Each touched 128 B line = one
  outlined light-blue column at its REAL word address; empty pages are simply absent (no compression, no
  `//` elision breaks). Power-of-2 address tick distance; ticks carry the real SIMD-relative byte address.
- **Burst = one lane's VW access**, hue = `lane % 8`, width = `VW_elems` words (a small min-width floor keeps
  a scattered 1-word burst visible). Red box = SCATTERED, green = FUSED, per SIMD, from `lines` vs `min_lines`.
- **Wave-level** is the representative scope (all waves are structurally identical). For a cooperative macro
  load, slice `coop_forward_map(desc, n_waves, wave_size)` to wave 0; for the C store use the recorded global
  store transaction. dtype-aware widths are mandatory; ASM-gate via `coalescing_probe` when a kernel exists.

---

## 5. The pipeline spine

### `FlowStage(name, component, source, transform="", info=(), legend=False, dist=None, relabel=False)`
One hop: a `name`, a stage `component` (any cell-field component / `WaveStrip`), a REQUIRED `source` (the
code-object name this panel renders -- the descriptor VARIABLE, e.g. `"load_desc"` / `"store_desc"` /
`"read_desc"` / `"c_store_desc"`; NO default, so every panel is greppable back to the code), the `transform`
that produced it (named on the arrow), optional extra `info` lines + a per-panel `legend` opt-in, and `dist`
(the source encoding for the info box when the component is fed by a forward map). `relabel=True` declares
this edge is an EXPLICIT label change (the ONLY sanctioned one; see §9 and `label_flow_and_transforms.md`).

**Info-box rule (`box_lines`), one distribution per destination:** a panel that is the DESTINATION of a
transition (it has an incoming `transform`) shows the ONE static distribution that was USED TO PLACE THE DATA
INTO ITS STORAGE -- the *store* distribution for an LDS/memory destination, the *load/read* distribution for a
register destination, led by `src: <descriptor variable>`. A SOURCE panel (the first stage, no incoming
transition) gets NO box at all -- the panel already draws its labelled data, so a box would be redundant. A
`FlowStage(reorder=True)` **in-register reorder** intermediary (a within-lane `v_perm` holding the SAME data
reordered -- e.g. the coalesced-read landing before the reorder into MMA order) ALSO gets NO box; the box
lives on the finally-requested destination. So a multi-transition flow marks exactly one distribution per
destination, and the starting/reorder-intermediate states carry none. The reorder itself is DERIVED
(`transforms.reorder_between`) and priced by the §7a cost ladder (`tiling_interleaving_design.md`).

### `Pipeline(stages=(...), title="")`
- `.render(out_path, dpi=200, gap=4.0, ticks=True, scale=None, show_info=True, show_legend=True, max_in=16.0)`
  -- shared-axes strip; one deduped header + legend.
- `.render_panels(out_path, dpi=200, panel_h_in=9.0, pw_min=2.0, pw_max=5.0, gap_in=1.5, panel_info=True,
  legend=True, tall_factor=2.0, ...)` -- a row of common-height panels, each with its own info box + one
  shared legend. A dense `LdsBankView` (many depths) grows to `tall_factor x` and panels top-align so the
  info boxes stay flush. Per-panel width tracks aspect, clamped to `[pw_min, pw_max]`.
- `.trace(origin_coord, color=...)` -- light one datum across every stage (provenance).

---

## 6. Flow recipes (prebuilt Pipelines)

**Primitive (single hop / one operand), in `layout_render`:**
- `flow_mem_to_register(dist, dims=("M","K"))` -- logical tile -> register file (global load).
- `flow_lds_to_register(mp, addr_fn, read_dist, dims, *, nbanks, elem_bytes, lds_base_bytes=0)` -- LDS
  placement -> wave-read register file.
- `flow_kloop_operand(load_dist, store_mp, store_addr, read_dist, dims, *, nbanks, elem_bytes,
  lds_base_bytes=0, name="A")` -- the whole K-loop for one operand: global -> regs -> LDS -> MMA-operand regs.
- `flow_wave_mma(mma, **overrides)` -- the `MmaTee`.

**Kernel PHASE recipes, in `kernel_stages`** (each returns a `Pipeline`; every LDS-bearing one REQUIRES
`nbanks` + `elem_bytes` and accepts `lds_base_bytes=0` and the true `stride`):
- `flow_load_phase(*, load_desc, dims, nbanks, elem_bytes, lds_base_bytes=0, store_desc=None, dest="lds",
  scope="wave", cooperative=False, n_waves=1, wave=0, stride=None, pad=0, swizzle=None, ...)` -- global
  thread-tile -> register file -> LDS (or a register-prefetch target). `scope="macro"` makes the register
  stage a per-wave strip. Pass `stride` = the true memref row stride.
- `flow_mma_phase(mma, *, a_enc=None, b_enc=None, dims_a=("M","K"), dims_b=("N","K"), trace_a=None,
  trace_b=None, **tee_overrides)` -- the tee (A/B wave operands -> derived C).
- `flow_epilogue_phase(mma, *, nbanks, elem_bytes, lds_base_bytes=0, c_store_desc=None, dims_c=("M","N"),
  ...)` -- C register file -> {auto-detected branch: direct | reorder | LDS round-trip} -> final C tile.
  Returns `(Pipeline|None, branch, note)`; `branch=="unknown"` -> `None` (ask the user).
- `flow_lds_store_placement(*, store_desc, dims, nbanks, elem_bytes, lds_base_bytes=0, load_desc=None,
  wave=0, n_waves=1, stride=None, pad=0, swizzle=None, compact=False, ...)` -- registers -> LDS placement.
- `flow_lds_load_placement(*, read_desc, dims, nbanks, elem_bytes, lds_base_bytes=0, flow_desc=None,
  wave=0, n_waves=1, stride=None, ...)` -- the reverse: LDS -> registers.

---

## 7. Descriptor -> viz adapters (`kernel_stages`)

- `field_inputs(desc)` -> `(encoding, forward_map)` for a single-wave field.
- `lds_inputs(store_desc, *, stride, pad=0, swizzle=None)` -> `(store_mp, addr_fn)`; `addr_fn(row,col)` =
  `row*(stride+pad) + free` in ELEMENTS (`free` = the swizzled column when a `swizzle` callable is given).
- `coop_forward_map(desc, *, n_waves, wave_size=64)` -> the FULL cooperative `{(tid,reg)->coord}` across all
  waves (via the real emit; a single-wave mapper mis-reads a coop encoding).
- `classify_epilogue(c_native, c_store_desc)` -> `("direct"|"reorder"|"cross_lane"|"unknown", note)`.

---

## 8. One-shot entry + text reflection

- `render_views(enc, axes=("M","K"), *, views=("logical","register","lds"), dtype_bits=16, layout="col",
  order_by="reg", replicate=(1,1), out_dir=".", name="layout", combined=True)` -- the logical / register /
  LDS panel set for one encoding (individual files + a combined sheet).
- `render_views` and the components render; **`describe(encoding)`, `render_forward_map(encoding)`,
  `render_inverse_map(encoding)`** are pure text (no matplotlib) -- the raw map for a report.

---

## 9. Physical-accuracy contract (facts, not preferences)

These are correctness requirements the surface enforces; the *taste* (which view + which render knobs for
which image) lives in the `/layout-viz` skill, not here.

1. **Dtype drives bank packing.** `elem_bytes` is required; `4//elem_bytes` elements pack per 4-byte bank
   (f16 -> 2, f32 -> 1). Addresses are in ELEMENTS, banks/depths in DWORDS.
2. **Bank count is explicit.** `nbanks` is required (gfx90a -> 32). No silent assumption.
3. **LDS base + true stride.** Feed the operand/buffer's real `lds_base_bytes` and the true memref row
   `stride` (`_bufs*free` for a double-buffered tile); base 0 / single-buffer stride is valid ONLY for
   A / buffer-0 / a standalone single-buffer tile. Read the kernel's real `smem_alloc` layout -- do not guess.
4. **Placement != conflict.** These views place data; conflict-free-ness (half-stripe parity) is the
   empirically-validated `/bank-conflict` path.
5. **Labels flow INVARIANT; only an explicit relabel changes them.** A datum's label is its identity; a
   transform changes destination coordinates, not the label (never derive a label from a position). Arrow
   labels route through `transforms.describe_edge → (kind, why)` (reposition / reorder / cross_lane / relabel);
   `Pipeline.check_label_invariance()` (run by `render`/`render_panels`) raises `LabelMutationError` on a
   mutated label unless the stage is `FlowStage(relabel=True)`. Full model: `label_flow_and_transforms.md`.
6. **SHADE = vectorized transaction time order (the ISA is what we model).** Hue = thread (who); **shade = when**. One
   shade per **vectorized access (transaction)**, `t0` = darkest = every thread's FIRST, lockstep across
   threads; a thread shows `regs/VW` shades (1 contiguous vector → 1 shade; N scalar accesses → N). Derive it
   from `vector_transactions(map, addr_fn, dtype_bits)` — **WIDTH from the STRIDES** (stride-1 → wider, ≤ b128,
   quantized to legal widths), NOT the recorded emit `vw` (pre-coalescing, often 1). Key on **THIS hop's own
   descriptor** (store map for a store, read map for a read — never the MMA-operand map). Colour ONLY via
   `accent_tint(lane%8, t, nt)` (darkest=first). `objdump` VALIDATES, never sources. ✗ shade from register
   index / from `vw`=1 / from the operand map on a read.
6a. **ORDER THE REGISTER AXIS BY MEMORY — the tensor descriptor decides, NOT the encoding index.** A
   memory-facing register panel (a store's registers, a load's registers) MUST lay its vreg axis out in the
   order the DESCRIPTOR's strides give (`addr_fn` from the recorded `strides`), because the encoding/vreg index
   is NOT physical memory order. **C (and any operand) can be row- OR col-major — the descriptor is the only
   authority**: col-major C ⇒ M is stride-1 ⇒ columns run M-fast; row-major C ⇒ N is stride-1 ⇒ columns run
   N-fast. Sort `range(nreg)` by `addr_fn(*fwd[(lane0, r)])` and pass it as `vreg_values`. ✗ NEVER lay the
   register axis out by encoding index — that renders an M-contiguous (col-major) store as N-fast and
   CONTRADICTS the ASM (`objdump` shows `global_store_dwordx2/x4` over M-adjacent elements). This is a
   consequence of "a transaction = contiguous MEMORY run → CONSECUTIVE vregs": the panel must show that
   consecutiveness, so the reader (and the shade) see the real coalescing.
7. **Grouped data must be UNAMBIGUOUS when it spans ≥2 axes.** A compact per-thread group (`grouped`/`block`)
   whose data spans >1 on **two or more** coordinate axes — e.g. `N0-3 K0-7` (not `N0-3 K0`) — cannot show the
   coord ordering inside a 1-D strip; shade helps but is not always enough. So the FIRST such thread renders
   **fully detailed** (per-cell labels) as the `cell→coord` KEY; same-pattern threads stay compact and read off
   it. If any other such thread's internal `(grid-pos → coord)` ordering DIFFERS from the key, a **panel note**
   names those threads (`… DIFFERS at T5 — do not read those off T0`). Enforced in code by
   `CellFieldMixin._disambiguate_ge2_axes` (wave scope, every cell-field view); a single-axis strip is left as a
   compact block.

---

## 10. Recorder — a fully EXTERNAL bolt-on (`tiling_recorder.record_build`)

`record_build(build_fn, *args, **cfg) -> ((kernel, mma), RecordedPipeline)` captures a kernel's transactions/
ops at the **verb boundary** by **decorating the verbs** (wrapping `build_fn.__globals__[verb]` + the `emit.*`/
`__init__` aliases + `TileMma.__call__` on the class), capture-before-delegate so **emitted IR is byte-
identical**.

- **✗ NEVER edit core for a viz concern.** No recorder hook inside `emit.py` / `transforms.py` /
  `mma_operation.py` / `ir.py` — viz is a bolt-on and the tiling verb set is still growing; coupling core to
  viz would constrain new verbs. Reserve core edits for genuine tiling-API needs.
- **Coverage = the verb registry.** The wrapped verbs ARE the documented coverage surface; a new verb opts in
  with one registry line. Movement that bypasses a registered verb is out of scope BY DESIGN.
- **Loud, not silent.** An external `b`-witness counts emitted mem/mma ops and refuses (`CoverageError`) if a
  node's fan-out doesn't reconcile — a wrong/short pipeline never renders quietly. (`b`-decoration is only good
  as the op-COUNT witness: the verbs consume `tile_desc.layout`/`window` via free functions before ever
  calling `b`, so `b`/`kernel.body.ops` see only flattened SSA — read-IR as the semantic source is OUT.)
- `RecordedPipeline`: `.transactions` / `.ops` / `.lds_spaces()` / `.block_diagram(out, title=)`; the driver
  (`auto_pipeline`) adds `resolve_origin`, the gates `verify_lds_roundtrip` / `verify_mma_soundness`, and the
  committed stage renderers.
