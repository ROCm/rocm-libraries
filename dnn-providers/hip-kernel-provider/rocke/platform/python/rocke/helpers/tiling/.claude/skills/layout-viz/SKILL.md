---
name: layout-viz
description: Render and explain MMA (MFMA/WMMA) layout visualizations at any scale — the A/B/C tee, register files, logical tiles, derived/final-C, C store distribution, and LDS view. Answers questions from "how does this distribution map a logical tile to registers?" and "what would this register file look like after a transform?" to "show the wave-tile flow from LDS through the C-shuffle" and "show my entire kernel pipeline dataflow from global load to final store." Interprets soundness / K-match / transform diagnostics in plain language and dispatches the MMA Kernel Expert for layout judgment. Use when designing, comparing, explaining, or debugging matrix-core operand/accumulator layouts, transforms, interleaving, and pipeline dataflow.
argument-hint: <atom/tile spec or encoding> [view] [knobs]
---

# Layout-Viz Skill

You are the **hands** that drive the tiling viz tools; the **MMA Kernel Expert** is the **brain** you dispatch
for every layout decision (WHICH distribution/encoding). Render what the expert/user specifies, **VERIFY in
code that the render matches the request before showing it**, then read each diagnostic back in plain language.
✗ Never invent a distribution or default to canonical. If under-specified or a result is odd, STOP and consult
the expert or ask the user (`../shared/prerequisites.md`, "Consult, Don't Improvise").

## Which path? (pick ONE front door)

- **"Visualize my kernel / show the whole pipeline / dataflow"** → **Workflow A** (automated, whole kernel).
  ✗ never hand-build stages or hand-draw matplotlib.
- **"Render THIS distribution / register file / tee / transform / one view"** → **Workflow B** (single view).
- **A specific question** ("does this coalesce?", "is this sound?", "what's this after `<transform>`?") →
  **Task patterns**.
- **"Is X a bank conflict / how many / why?"** → NOT here → **`/bank-conflict`** (empirical: rocprof + a
  validated simulator).

## What this skill can do

Every row renders a **correct, verified** picture (Constraint 12) + a plain-language reading; every layout
decision is the MMA Kernel Expert's.

| You want to… | Ask / trigger | You get |
|---|---|---|
| **Whole kernel dataflow** (global load → … → C store) | "visualize my kernel pipeline" (+ the build fn) | Workflow A: `record_build` → block diagram → per-flow panels, correctness gates run in code |
| **Kernel structure** (transactions/ops in order) | "show the block diagram" | a Level-0 node/edge **block diagram** (mem↔reg transactions + ops; K-loop as ×N) |
| **MMA wave-tile flow** (A × B → derived C) | "MMA flow" / "wave-tile flow" | the **MmaTee** — A (M×K), B (K×N) → C (M×N) + the static distribution |
| **A distribution → registers** | "how does this dist lay a tile into registers?" | a **RegisterFile** (tid×reg) beside a **LogicalTile** (owner `T{l}R{r}`) + text maps |
| **A register file after a transform** | "what's this after `<transform>`?" | before/after files + tiles, arrow classified (**reposition / reorder / cross_lane**; relabel only on C-reuse) |
| **LDS → C-shuffle flow** | "LDS through the C-shuffle" | LDS view → wave-read A/B → tee → C-shuffle → final C, each hop named |
| **LDS placement** (where a store/read lands) | "LDS placement of `<store/read>`" | an **LdsBankView** (depth × banks) at true addresses (`nbanks`/`elem_bytes`/base/stride) |
| **Coalescing** (or why strided / the C-store) | "does this coalesce? / how bad is the C-store?" | a **thread-tile + phase-0 coalescing overlay**, one image per output major, dtype-aware width |
| **A soundness / K-match / transform verdict** | paste the layout / "is this sound?" | a plain-language verdict + the cheapest fix (via the MMA Expert) |
| **One hand-specified distribution** | give the encoding / forward-map + which view | that single view, verified to match the request |

## Environment & prerequisites

Prereqs (read first): `../shared/prerequisites.md` (path resolution, fail-fast, dispatch) and
`../shared/temporary_file_policy.md` (ask before creating files; track + offer cleanup).

Run from `platform/`, repo venv, `PYTHONPATH=python`:
```
PYTHONPATH=python <repo-root>/.venv/bin/python <script.py>
```
Import viz from the PACKAGE, not submodules (matplotlib is lazy). Modules under `rocke.helpers.tiling`:
- `mma.mma_operation.TileMma((M,N,K), target=, atom_override=)` — atom + canonical `a_layout`/`b_layout`/
  `c_layout`, `atom_shape`, `traits`.
- `register_mapper.RegisterMapper(enc)` — `matrix_coordinates`, `inverse_map`, `num_lanes`, `num_vector_items`.
- `transforms` — `as_forward_map`, `classify_transform`, `describe_edge`, `reorder_between`,
  `derive_c_distribution`, + the diagnostics (see §Diagnostics).
- `visualization` — components (`RegisterFileComponent`/`LogicalTileComponent`/`LdsBankView`/`MmaTee`), spine
  (`Pipeline`/`FlowStage`/`WaveStrip`), `render_views`, colour model (`accent_tint`/`ACCENTS`/`NACC`), flow +
  phase recipes, adapters (`field_inputs`/`lds_inputs`/`coop_forward_map`/`classify_epilogue`).
- `tiling_recorder.record_build(build_fn, *args, **cfg) -> ((kernel, mma), RecordedPipeline)` — records
  transactions/ops at the verb boundary (emitted IR byte-identical). `RecordedPipeline` exposes
  `.transactions`/`.ops`/`.lds_spaces()`/`.block_diagram(out, title=)`.
- `visualization.auto_pipeline` — `resolve_origin`, the gates `verify_lds_roundtrip` / `verify_mma_soundness`,
  the sweep driver `render_sweep` / `plan_flows`, the `view()` dispatcher, the committed renderers.
- `visualization.block_diagram` — `extract_blocks` + `block_diagram(pipe, out)`.

## Workflow A — whole kernel (automated, block-diagram-first)

Drive the COMMITTED automation. Extraction, the two gates, and every render are CODE; your job is **select a
block + ask scope/drivers + interpret**. One committed call per step, identical output every run.

0. **LOCK THE CONFIG FIRST — the #1 failure mode.** Open and re-read the kernel source (its build fn + params).
   ✗ NEVER work from memory, a prior session, or a nearby test `CFG` / shrunk default. STATE the config
   (macro/wave tile, `waves_m×waves_n`, `tile_k`, dtype, arch) and CONFIRM it's the kernel + config the user
   means. A correct render of the WRONG config is still wrong; the block-diagram metadata line surfaces the
   recorded config — verify it matches before trusting any view.
1. **Record:** `(kernel, mma), pipe = tiling_recorder.record_build(build_fn, *args, **cfg)`.
2. **Level-0 map:** `pipe.block_diagram("<out>/block_diagram.png", title=)`. Show it. Each block = one recorded
   node (mem↔vreg transaction or vreg↔vreg op/`mma`); the K-loop body is a `×N` container; A/B/C colour-coded.
   Structure only — no scope pinned; a MAP, never a correctness claim.
3. **User picks a block → YOU ask scope + drivers** (never assume): **macro** → `WaveStrip` + macro LDS
   placement (ask buffer, kb/macro origin); **wave** → single-wave register file + that space's memory view
   (ask wave, buffer, operand); **transform** → before/after files + `transform_note`; **MMA** → `MmaTee`.
4. **Render** with the committed renderers (`auto_pipeline.render_lds_store`, the flow views, …) — dtype-derived,
   Constraint 12 gated in code.
5. **Offer deeper diagnostics, never blind:** coalescing (this skill) and LDS bank-conflict (`/bank-conflict`).
   If you lack driving context, ASK.

Correctness is CODE: `verify_lds_roundtrip` (per LDS buffer-half) + `verify_mma_soundness` run before any
pipeline render.

**Full sweep** (`render_sweep(pipe, out_dir, scope="both")`) — four levels:
- **L0 = end-to-end overview:** `0_0_block_diagram`, `0_1_localization`.
- **L1 = flows (≥2 state panels):** `1_0_prefetch_A`, `1_1_prefetch_B`, `1_2_lds_read_A`, `1_3_lds_read_B`,
  `1_4_compute`, `1_5_epilogue`. LDS-store is SUBSUMED into the prefetch flow (global load → LDS store) — there
  is no standalone `lds_store` flow. Each scope-bearing flow emits `_w0` AND `_macro`; wave-tile-generic views
  (tee/epilogue) render once (no scope token).
- **L2 = detailed ANALYSIS inspections:** **coalescing** and **bank-conflict**. Deep/empirical, owned by their
  skills — the default sweep does NOT draw them; `render_sweep` REPORTS them as redirects and you should
  **OFFER to sweep them**: coalescing → render here (the Coalescing task pattern, one per output major, ASM-
  validated); bank-conflict → **`/bank-conflict`** (rocprof + validated sim — never a verdict from this skill).
- **L3 = individual single-panel inspections, ON REQUEST** (not swept): a standalone register file / LDS bank
  grid / logical tile / final-C tile / C-store distribution. Render via **Workflow B** when the user asks for
  one specific panel.

L0 = how many state panels? overview. L1 ≥2 (a flow). L2 = a named ANALYSIS (coalescing / bank-conflict). L3 =
exactly one plain panel on demand.

Naming (reference): folder = the ISA config, all DERIVED from the recording
(`<kernel>_<out>_<atom><in>_<macro>_<wavetile>_<wavesMxN>_<opts>`, e.g.
`crc_f32_16x16x16f16_256x256x32_64x64x32_4x4_base`); file = `<L>_<seq>_<context>[_<scope>].png` where the scope
token (`_w<N>` / `_macro`) is the ONLY in-file config bit.

**Output location — ASK when ambiguous.** Explicit dir → use it. Else reuse a dir already settled THIS session.
Otherwise ✗ do NOT invent a path — ASK where to write (suggest `helpers/tiling/tmp/<sweep>/` per the temp-file
policy). Run the two gates first; offer cleanup at the end.

## Workflow B — single hand-specified view

1. **Settle the output dir FIRST** (reuse the session dir; if ambiguous, ✗ never pick silently — ASK).
2. **Parse** — atom/tile spec (or a raw encoding/forward-map), which A/B/C dists, which view, knobs, optional
   trace `(m,k)×(n,k)`. An unpinned "interleaved" is a GAP — ✗ don't default to canonical; get the concrete
   encoding.
3. **Resolve** via `rocke.helpers.tiling` (`TileMma`, encodings, `as_forward_map`). Every default here is
   CANONICAL — not a stand-in for the expert's choice.
4. **Dispatch the MMA Kernel Expert** (subagent, `opus`) for the CONCRETE encoding/forward-map for A/B/C + the
   soundness call. You are the hands, not the brain.
5. **Render** the view into the output dir, building exactly the expert's encoding.
6. **Verify before showing** — prove the defining property in code (interleaved ⇒ `a_enc != canonical` AND
   sound; canonical ⇒ `== canonical`; supplied ⇒ equals the map). If it fails, ✗ don't present — fix or re-consult.
7. **Interpret diagnostics** (§Diagnostics).
8. **Return** path(s) + plain-language reading + expert recommendation + a suggested next view.
9. **On completion**, list files created + offer cleanup.

## Task patterns (smallest set of views that answers the question)

Components compose (shared axes; accept an encoding OR a forward-map from another stage). Classify every
inter-stage arrow (`transform_note` → tier + cost); every layout judgment is the expert's.

- **Map a distribution to registers** → `RegisterFileComponent(dist)` (tid×reg, cell=coord) beside
  `LogicalTileComponent(dist, label_coords="register")` (owner `T{lane}R{reg}`) + optional text maps. Expert
  explains via Rs/Hs/Ps/Ys.
- **Transform a register file → new file / tile / result** → `classify_transform(src, tgt)`; render before/after
  register files AND logical tiles; `highlight` one element to trace it. Kinds: **reposition** (free coordinate
  transpose) / **reorder** (in-register, dtype-graded) / **cross_lane** (expensive); relabel only on C-reuse.
- **Wave-tile / MMA flow** (default) → the tee with `show_logical_inputs=True, show_static=True`.
- **LDS → C-shuffle** → LDS bank view → wave-read A/B files → `MmaTee` → C-shuffle (`classify_transform` C→store
  order) → final C. Classify each hop.
- **Whole kernel dataflow** → use **Workflow A**.
- **Coalescing** ("does this coalesce / why strided / how bad is the C-store?") → a **thread-tile + coalescing
  overlay**, one image per output major (transaction order differs). Hue=lane; shade = load/store TIME order
  (`accent_tint`, darkest=first); per-thread-tile borders; coord at each patch origin; **black box around every
  vector in PHASE 0** (first served instruction). Adjacent phase-0 boxes = FULL coalescing; gaps = partial/
  strided (name the `lanes_major/lanes_minor`× extra txns). Do input loads too.
  - **Dtype-aware width, mandatory:** per-lane bytes = `VW_elems × dtype_bytes` → the real instr (`dword`=4B /
    `dwordx2`=8B / `dwordx4`=16B). Label every access with instr + bytes + % of 128-bit max. Derive `VW_elems`
    from the ACTUAL descriptor. ✗ "VW cells" without dtype→byte scaling hides an under-width access.
  - **Validate against compiled ASM** when a kernel exists (`llvm-objdump`: `global_load_dword{,x2,x4}`,
    `ds_read*/ds_write_b*`). Diagram disagrees with ASM ⇒ the diagram is wrong.
  - **Logical orientation** (Constraint 7): dim0↓ dim1→ per tensor; A=M↓K→, B=K↓N→, C=M↓N→. Expert owns the
    cost reasoning (lane-major axis, txn count, the A↔B + `c_transpose` escape). SOT: `tiling_interleaving_design.md §7`.
- **LDS bank-CONFLICT** → NOT hand-rendered → `/bank-conflict`. layout-viz may render a NEUTRAL bank-layout
  (address→bank) when asked, but ✗ never stamp a bank grid with an N-way/"conflict" verdict.

**Pipeline = each stage in its ACTUAL state:** memory → logical tile; after a load → register file; after a
transform → name it + render the NEW register file. At register stages render the register state — ✗ don't
re-project to the memory tile.

## Phase recipes + adapter (render a REAL kernel)

Pick a phase → the primitive-view sequence as a compact linear `Pipeline`/`FlowStage` strip, transform named
between hops. WAVE scope by default; macro opt-in. Any LDS recipe REQUIRES `nbanks` + `elem_bytes`.

| phase | recipe | sequence |
|---|---|---|
| global load / prefetch | `flow_load_phase(load_desc, dims, nbanks=, elem_bytes=, dest=, store_desc=, scope=)` | logical tile → register file → LDS view (`dest="lds"`) or prefetch file |
| LDS store placement | `flow_lds_store_placement(store_desc, dims, nbanks=, elem_bytes=, load_desc=, wave=)` | register file → LDS grid (true depths) — placement only, no conflict verdict |
| LDS load placement | `flow_lds_load_placement(read_desc, dims, nbanks=, elem_bytes=, flow_desc=, wave=)` | LDS grid → register file(s) — inserts the in-register reorder stage IFF the read reorders (see §Defaults) |
| MMA | `flow_mma_phase(mma, a_enc=, b_enc=)` | the `MmaTee` |
| epilogue | `flow_epilogue_phase(mma, nbanks=, elem_bytes=, c_store_desc=)` | C file (f32) → {auto branch} → final C tile |

Epilogue branch is AUTO-DETECTED via `classify_epilogue` → `direct`/`reorder`/`cross_lane`/`unknown`; on
`unknown`, ASK. Macro scope → the register stage is a `WaveStrip`, render with `render_panels`.

**Adapter — ✗ don't hand-write distributions / addr lambdas:** `field_inputs(desc)` → `(encoding, fwd_map)`;
`lds_inputs(store_desc, *, stride, pad=0, swizzle=None)` → `(store_mp, addr_fn)` (a kernel swizzle is replayed
bit-for-bit via `lds_conflict.NumBuilder`); `coop_forward_map(desc, *, n_waves, wave_size=64)` → the full
cooperative map; `classify_epilogue(...)` → the branch. Worked example:
`kernels/tiling_gemm_crc_demo/crc_pipeline_viz.py`.

## Tool catalog (load-bearing knobs; full list in `visualization_api_surface.md`)

- **`MmaTee`** — `from_mma(mma, **overrides)` (canonical refs + dtypes from the atom) OR explicit args. ✗
  `from_mma` with no `a_enc`/`b_enc` renders CANONICAL — for interleaved you MUST pass explicit `a_enc`/`b_enc`
  and VERIFY `!= mma.a_layout` + sound. Interleaved hallmarks: lane owns a contiguous rectangular patch,
  register order ≠ canonical, the transpose is `reorder` not `cross_lane` (`interleave_idx<1,KPT,DPT·KPT>`) AND
  passes soundness + §2b. Conventions wired: A left wing, B top wing (transposed), C body = POSITION grid with
  (M,N) label (`text_map`). `tee.render(out_dir, name=) -> path`; `tee.c_mapping() -> {(lane,reg)->(m,n)}`. Key
  knobs: `a_enc`/`b_enc`/`c_enc`, `*_canon`, `atom_shape`, `in_dtype`/`out_dtype`, `dims_*`, `issue_order`,
  `full_detail`, `color_mode`, `show_logical_inputs`, `show_static`, `trace_a`/`trace_b`, `show_diagnostics`.
- **`RegisterFileComponent`** — `dist=` OR `fwd_map=`. Knobs: `dims`, `row_axis`/`col_axis`, `row_order`/
  `col_order`, tick sides, `groups`, `color_mode`, `dtype_bits`, `shade_map`, `highlight`, `origin`.
- **`LogicalTileComponent`** — `dist=` OR `owner_map=`; `text_map` makes a POSITION grid with the flowed label
  (the C-body trick). Knobs: `dims`, `row_coord`, `label_coords` (`register`/`logical`), `groups`, ticks,
  `highlight`.
- **Standalone:** final-C tile (position grid, cell=flowed (M,N)); C store distribution; a supplied `fwd_map`
  as a register file; `render_views(...)` (logical/register/LDS set).

## Defaults (TASTE — the objective contract is `visualization_api_surface.md`)

- **Composition:** pipeline flows → `render_panels` (common-height, per-panel info, one shared legend); use
  `render` only for a tight same-dist strip. Wave scope default; state MACRO membership. Load-phase flow is a
  SUMMARY/placement view (block/group labels, `panel_h_in≈11`, `pw_min≈2.6`) — reserve element-by-element for
  the tee. Dense LDS → `tall_factor=2`, `compact_rows=False` (show the TRUE striping).
- **Info box — one distribution per DESTINATION, none on a source or reorder-intermediary** (enforced in
  `FlowStage.box_lines`): a transition destination shows the ONE dist used to place data there (store dist →
  LDS, load/read dist → registers), led by `src: <descriptor variable>`. A SOURCE panel (first stage) and a
  reorder-intermediary carry NO box. SOT: `visualization_api_surface.md §5`.
- **In-register reorder is drawn honestly** (never collapsed into the read/write arrow): IFF a load/store's
  coalesced order ≠ the consumer's order, the flow gains an explicit reg→reg stage — coalesced landing (no box)
  → an `interleave_idx(...)` arrow → the requested order (the box). DERIVED via `transforms.reorder_between`
  (the `interleave_idx` params are OUTPUTS, never hardcoded); the panel states the cost (§7a ladder tier +
  `v_perm`/lane — the *price of the wide load*). No reorder (identity) ⇒ no extra panel. Generic for any kernel.
- **Shade — ONE contract** (`visualization_api_surface.md §9.6`): hue=thread, shade = vectorized transaction
  TIME order, width from strides (≤b128) off THIS hop's descriptor, register axis in memory order; colour ONLY
  via `accent_tint(lane%8, t, nt)` (darkest=first). A SOURCE panel or a REORDER-result panel has NO transaction
  order → FLAT single shade (`shade_map={cell: 0}`). Tee: AS-IS (A/B by load time, C by MMA processing order).
- **`color_mode="first8"` always** (colour lanes 0–7, grey the rest) — ✗ never `"full"` (64 lanes on an 8-hue
  cycle read like 8 threads).
- **Grouping = the SUBJECT UNIT, from the MACHINE (canonical ref), never the supplied labels:** logical tile →
  atom sub-tiles; register file → per-MMA-input register blocks; thread-tile → per-lane patches; LDS → per-bank.
  ✗ don't fall back to a hardcoded 16×16.
- **Label mode by INTENT:** distribution inspection → `label_coords="register"` (owner); correctness/flow →
  flowed labels (`text_map`); plain `logical` only for a static memory picture. `show_static=True` by default.
- **Readability floors (GROW or SUMMARISE, ✗ never shrink text/dpi):** cell ≥ ~0.20 in/side, font ≥ 6–7 pt at
  dpi ≥ 200. Over-dense → (1) grow the panel; (2) collapse to `block`/`grouped` (one label/row); (3) reduce
  scope. Working values: standalone `cell_w≈0.42–0.62`, `cell_h≈0.30–0.42`, font 6–7; tee smaller (`cell≈0.4`,
  font≈5). Zoom, don't drop below the floor.
- **Composition conventions (FIXED):** LDS view ALWAYS depth×banks; logical tiles on natural axes (Constraint
  7); register files vertical (Constraint 4, the tee transposes its OWN B/C wings — render via `flow_mma_phase`,
  ✗ never re-orient a file to feed a tee); one labelled arrow per consecutive-stage hop; relative sizing (each
  element from another's `origin`/`grid_size()` + a shared `gap`).

## Diagnostics (pure observers — translate + name the fix)

- `operand_soundness(layout, canon, role=)` → `ok` / `error`. `mma_compatible(layout, canon)` → `ok` /
  `warning` (an in-register reorder makes-it-so — name the permutation) / `error` (cross-lane or none).
  `mma_pair_compatible(a, b, a_canon=, b_canon=)` → both sound AND K-dists match; drives the tee banner.
  `diagnose_k_match(a, b)` → `ok`/`warning`/`error`. `classify_transform(src, tgt)` → `reorder` (cheap,
  dtype-graded) vs `cross_lane` (expensive/deferred).
- Tee banner red ERROR / orange WARNING = the pair diagnostic — say what's wrong, which rule, cheapest remedy
  (free symmetry [reposition/source-swap] → dword reorder → sub-dword → cross-lane last).

## Output format

**Relay everything in text** — the image is a companion. Mirror the tee's info panel + legend (atom, wave,
tile, dtype, issue order, vec axis A/B/C, C orientation + shuffle tier, shade meaning, grouping) + the
diagnostics + the expert's reading. Always lead with the **Summary Table**.

```
## Layout Viz — <what was rendered>
- rendered: <path(s)> | view: <tee|register|logical|final-C|store|lds> | inputs: <atom/tile, A/B/C dists>

### Summary Table
Atom & config: <atom> | wave <N> | <in>→<out> dtype | K/lane <..> | C VW <ACC_VW>,<regs>/lane | tile <M×N×K> |
issue <order> | vec A:<..> B:<..> C:<..> | shade = vectorized transaction time order.

Correctness: soundness A/B · K-match A·B · position vs label (identity/derived) · C orientation (CANONICAL, or
SWIZZLED → classify the shuffle: in-register reorder vs cross-lane).

Vectorization & cost (per operand × stride-1 axis) — a strided vector axis is INVALID at VW>1 → VW=1 (§2b):
| operand | stride-1 axis | coalesced axis | MMA-ready? | transform | tier + cost |

Suggestions: recommended chain / symmetries / cheapest side for the transpose; what to avoid; bank conflicts
are REDUCTION TARGETS (name the lever — pad/swizzle/layout; ✗ never "hidden"/"free" without measurement).

Pipeline stage table (pipeline requests): | stage (domain) | layout / vec axis | VW | transform → next (tier) |
BW / conflict |. BW from VW×dtype vs 128-bit peak (VW 4 f32 / 8 f16 = full; VW 2 f32 = half; VW 1 ≤ ¼).

### Expert reading — <MMA Kernel Expert's judgment + recommended fix>
- next: <suggested follow-up view>
```

## Constraints (must hold — verify, don't vibe)

These are the render decisions the skill enforces. The physics/label LAW they rest on lives in the SOT (below)
and is enforced IN CODE — obey it, don't restate it.

0. **Lock the config first** (Workflow A step 0): re-read the kernel, pin macro/wave/waves/`tile_k`/dtype/arch,
   confirm with the user. ✗ never a test `CFG` or a convenient default.
3. **Every layout decision is the MMA Kernel Expert's** — dispatch for the concrete encoding; render exactly it.
4. **Register file: tid VERTICAL (rows), reg id HORIZONTAL (cols)** (`RegisterFileComponent` default — don't
   override). Only the MMA tee transposes its own **B & C** wings.
6. **Logical / thread-tile views show LOGICAL coords by default** (`label_coords="logical"`); owner
   `T{lane}R{reg}` only when the user asks to inspect ownership.
7. **Logical axes are natural, per tensor: A=M×K, B=K×N, C=M×N.** ✗ never draw B as N×K (B's free N is
   horizontal).
8. **Macro register stage = N per-wave 64-lane files SIDE BY SIDE** (`WaveStrip`), ✗ never one `n_waves*64`
   monolith. Always state whether a tile is part of a MACRO tile.
9. **dpi ≥ 200.** To shrink a big figure, reduce SCOPE (wave scope, fewer stages) — ✗ never lower dpi.
12. **VERIFY IN CODE before presenting** (never eyeball): **(a)** cross-stage consistency — the dist LEAVING a
    stage == the one ENTERING the next; **(b)** each stage uses the kernel's REAL per-hop descriptor; **(c)**
    physical correctness (dtype/banks/base+stride explicit per stage); **(d)** traceability. The automated path
    runs these gates in code; if it can't be verified, STOP and dispatch the expert.
- **SOT-and-code-enforced — obey, don't re-derive:** POSITION ≠ LABEL + labels flow INVARIANT + reposition/
  source-swap ≠ relabel (`label_flow_and_transforms.md`; `Pipeline.check_label_invariance`); the §2b
  vectorization contract; dtype/bank packing + LDS base+stride + shade + ≥2-axis disambiguation
  (`visualization_api_surface.md §9`; enforced in `CellFieldMixin`).

## SOT & scope

Own the facts elsewhere; here we render + point:
- `mma_is_machinery.md` — the sound MAC + POSITION ≠ LABEL.
- `label_flow_and_transforms.md` — labels flow INVARIANT; edge kinds reposition/reorder/cross_lane + source-swap
  (routing); only a C-reuse relabel changes a label.
- `tiling_interleaving_design.md` — vectorization contract (§2b), derived-C + C-store coalescing (§7), cost
  ladder + the in-register reorder as a layout cost (§7a).
- `visualization_api_surface.md` — every component/recipe/knob + the info-box rule (§5) + the physical-accuracy
  contract (§9: dtype bank packing, `nbanks`/`elem_bytes` required, LDS base+stride, label gate, shade §9.6,
  ≥2-axis disambiguation §9.7).

**Scope:** this skill RENDERS + DIAGNOSES. It does not build viz components or change the tee's baked
conventions (core `layout_render` work), and it does not run hardware or give bank-conflict verdicts
(`/bank-conflict`) — a GPU bit-exact confirm is a possible future gated-GEMM add-on, not part of rendering.
