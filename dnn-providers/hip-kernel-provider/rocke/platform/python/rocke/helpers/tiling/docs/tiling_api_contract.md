# rocKE Tiling API -- The Contract (why it is shaped this way)

**Audience:** contributors ADDING to (or reorganizing) the public surface. If you only want to USE the
API, you do not need this -- start with `tiling_api_surface.md` (its sec 2 is a complete minimal kernel).

**Status:** Steps 1-4 have LANDED -- the primitives named below are shipped and re-exported at the package
root: `TileMmaPlan`/`TileMmaDriver`, the `LayoutStyle` seam (`canonical`/`interleaved`),
`cooperative_load_desc`, `TileDesc.swap_dims`/`.reorder_registers`, `lds_tile_desc`, `check_gemm_shapes`, and
the transform observers. Only `TilePipeline` (Step 7) is still `(planned)`. This doc is where each BELONGS,
and the gate every new one passes.

**This is the WHY.** The how-to-use catalog is `tiling_api_surface.md`; domain theory lives in
`mma_is_machinery.md`, `tiling_interleaving_design.md`, `label_flow_and_transforms.md`, `lds_banks.md`.
This doc states the RULES a contribution must satisfy so the public surface stays coherent -- it does
not restate those docs, it points at them.

> **One line:** the tiling public API is a set of composable PRIMITIVES, each with exactly one home,
> organized by AUDIENCE. A name is public only if it passes the promotion gate. Helpers never lock out
> the primitives underneath them.

---

## 1. The promotion gate -- all four, or it stays private

A symbol earns a place on the public surface only if:

1. **Composed, not glue.** Authors/tools build *with* it; it is not one kernel's scheduling closure.
2. **One home.** Not copy-pasted across siblings; a single owner module.
3. **Misuse is unrepresentable, or it can say "unknown."** The signature makes the wrong call fail to
   type/compile, or returns an explicit `N/A`/`UNKNOWN` -- never a value that means two things.
4. **Named audience.** It lands in exactly one tier (below) for a stated reader.

Every `api_proposals/` file **names the gate question it satisfies and the rule it deletes** -- prefer a
signature that makes an error unrepresentable over a docstring that forbids it.

**Fail-fast ordering (cardinal):** guards whose *absence yields a wrong answer* run before guards that
merely raise for another reason. (The reference is `kernels/gemm_tiled_f16/gemm_tiled_f16.py`'s shape
guard ahead of the budget guard.)

---

## 2. The three tiers -- ONE taxonomy (audience). START HERE.

The `__init__.__all__` is grouped by these tiers, front door first. Do not re-introduce the old role- or
catalog-based groupings; a newcomer meets one map.

- **FRONT DOOR** -- what you reach for by default: `TileMma`, `TilePipeline` (planned), the everyday
  factories (`make_tensor_desc`/`make_window`/`make_tile_desc`/`make_fragment`), the IR verbs
  (`load_fragment`/`store_fragment`/`fill_fragment`/`transform_fragment`), and `check_gemm_shapes`.
- **TOOLBOX** -- the finer primitives the front door is *built from*, callable directly when you need
  control: `Tiling` (the `TileMma` atom/order knobs), `TileMmaPlan`/`TileMmaDriver`, the `LayoutStyle`
  seam (`canonical`/`interleaved`), `TileDesc.swap_dims`/`.reorder_registers`, `cooperative_load_desc`,
  the LDS bridge (`lds_tile_desc`/`lds_store`/`lds_read`), and the transform *observers* (surfaced at the root in the transforms split:
  `classify_transform`, `describe_edge`, `mma_pair_k_aligned`, `mma_operand_layout_sound`,
  `mma_accumulator_flow_consistent`, `mma_pair_compatible`, `reorder_between`, `derive_c_distribution`).
- **MACHINERY** -- internal, NOT authoring API, NOT re-exported: the recorder, the transform *solver*
  core (`_classify_maps`, `as_forward_map`), `RegisterMapper`, the `warp_encoding` calculators.

Clean means **legible layering, not fewer names**: the primitives stay on the surface (the mandate below
forbids hiding them); only machinery is kept out.

---

## 3. Glass-box / layered / non-blocking (the customization mandate)

**Every helper is a composition of primitives you can also call yourself. It never hides, locks, or
replaces them.** An author may (a) let the helper do it, (b) take its pieces and customize, or (c) build
from primitives directly -- all three are first-class.

Each front-door/toolbox helper's docstring carries a **`Customizing`** section with four parts:
- **what it does · what primitives it composes** (one line each);
- **how to override** (pin the atom; supply your own descriptor);
- **the override's contract** (what a custom input must satisfy);
- **which gate validates it** -- the soundness gates are IDENTICAL for a derived and a custom input
  (see 5); what differs is only provenance, that the accumulator is always derived, and observability;
- **the "reach past me when..." trigger** -- the symptom, not just the option.

A task-indexed **"Customize X" cookbook** (in `tiling_api_surface.md`) is the reverse index: task ->
object -> its `Customizing` section.

### 3b. Adding a layout style

A **layout style** (`LayoutStyle`, in `mma/styles/`) is the largest customization unit: a per-operand
strategy that produces {the mma-ready register layout + the memory bridge's intra-tile label-flow}.
`canonical` and `interleaved` are the two shipped profiles; a third is added by writing one file that
implements the protocol. The rules a new style MUST honor -- so it stays correct by construction:

- **Compose the primitives; never copy them.** A style orchestrates the public toolbox
  (`cooperative_load_desc`, `TileDesc.swap_dims` / `.reorder_registers`, the warp-encoding
  calculators); it must not re-implement memory addressing. A style is writable by an author entirely
  from the public surface -- if it needs a private copy of a primitive, the primitive is missing a
  home (fix that, don't copy).
- **Stage through LDS via the protocol, not an ad-hoc method.** A style that lands its operand in LDS
  before the MMA exposes the bridge through the base protocol's `lds_bridge(traits, *, role, free_sub,
  k_sub)` -- it returns `(lds_read_landing_desc, mma_ready_desc)`, or `None` for a non-staging style
  (canonical loads MMA-ready directly). It is a declared extension point on `LayoutStyle`, so the kernel
  composes it off the protocol, never off a concrete subclass.
- **The seam owns the correctness envelope, not the style.** The style never sizes the LDS allocation
  (the seam derives it from `desc_extents`), never masks an LDS access (the LDS verbs have no
  `bounds=`), never re-implements predication (that lives in the style-agnostic emit layer). These are
  structurally unreachable from a style -- a new style cannot violate them. *Corollary (clipping x LDS
  staging): because masking an LDS access is signature-unreachable and the allocation is full-tile, the
  Principle-4 property "OOB zeros from a masked global load flow through a FULL-width, unmasked LDS
  store/read and contribute nothing to the MAC" holds by CONSTRUCTION. This is enforced structurally
  (the no-`bounds=` verb signature is locked by test); an isolated empirical probe of clipped LDS
  contents is deferred -- no shipped kernel both clips and stages through LDS yet.*
- **Operands only; C is always derived.** A style influences the accumulator ONLY through the
  K-distribution its operands present to the atom; it must not supply a C descriptor. The runtime
  C-oracle (`c_desc == derive_c_distribution(...)`) catches a style whose C geometry the
  atom-derived descriptor cannot express.
- **Which gates police it:** `mma_operand_layout_sound` + `mma_pair_k_aligned` (per-operand soundness +
  pairwise K-match, against the canonical machine from the `canonical_layouts` helper -- never the style's
  own descriptor), `mma_accumulator_flow_consistent` (the passed accumulator carries the labels the
  machine produces from the operands, so the store writes the right coordinates), the driver's SOA
  contiguity guard (fail-fast on an AOS register order it cannot slice), and the
  coalescing/vectorization diagnostics that price the load.
- **Reach past the two shipped styles when** you need a global-load direction or a C-epilogue coupling
  neither canonical (K-contiguous) nor interleaved (free-contiguous, wide coalesced) expresses.

---

## 4. Derive by default, accept overrides, validate *identically*

Defaults are driven by the resolved atom; every default is a value you can replace (see
`tiling_api_surface.md` "the one idea"). An override runs the **same** soundness gates as the derived
path -- the gates take raw encodings, so they validate an arbitrary custom descriptor. Concretely:

- The `TileMmaDriver` runs one validation **unconditionally** in `__call__` (`_validate_mma_issue`:
  per-operand soundness + pairwise K-match + accumulator store-coordinate consistency + SOA contiguity).
  It is stateless and cannot tell derived from custom; derived inputs pass trivially. This makes
  "validated identically" literally true, and closes the hole where a per-operand-unsound custom operand
  K-matches its partner yet miscompiles -- and the one where a mislabeled accumulator stores to the wrong
  coordinates.
- **Custom overrides are OPERANDS ONLY** -- the accumulator descriptor is always derived. Atom-contiguity
  is a *separate* contract enforced by the mandatory slice descriptor, which fails-fast on any register
  layout it cannot express. The gates carry a `(lanes x regs)` dimension pre-check (clean diagnostic, not
  a `KeyError`).
- Do NOT cite `atom_override` as the exemplar of custom-descriptor safety; the raw-encoding gates are.
- **Validation is as complete as the stage is observable** -- unobservable non-MMA compute routes to the
  golden, not a false "short recording" failure.

MMA soundness = per-operand soundness (rule 2) + pairwise K-match (rule 3); see `mma_is_machinery.md`.

---

## 5. Arbitrary/odd shapes -- clipping + predication (correct by construction)

Clipping is BUILT and bit-exact on the masked path (`tiling_api_surface.md` 5b). The RULES:

- **Predication is INVISIBLE to the layout gates** (it is a runtime value substitution; the encoding is
  unchanged). Predicated-MMA soundness lives at the WINDOW layer as two contracts: **(a)** A's
  contraction-axis clip == B's (else `a*0` silently drops a term); **(b)** the clip pad is the compute
  stage's additive IDENTITY (0 GEMM, -inf max, 1 product) -- `pad` is a first-class per-stage value, and a
  non-identity pad on a contraction axis is refused.
- **Scope:** predicate the **global load (zero-fill) and the C store ONLY**. The cooperative LDS store and
  the wave read stay **full-width and unmasked**, and the **LDS allocation stays full-tile** (a
  clipped-but-unmasked read would otherwise index another workgroup's LDS); zeros flow through LDS and
  contribute nothing to the MAC. "Coalescing verdict provisional" is a global-load-only caveat; LDS
  store/read are invariant under predication (`lds_banks.md`).
- **Trigger:** the wide-load path predicates on **effective-clip-vs-tile-alignment**
  (`(bounds[axis] or lengths[axis]) % tile_extent`), NOT on `bounds is not None`. Refuse
  "odd tensor + vectorized + no predicate." The tile-aligned-bound mask-skip also requires a
  tile-aligned ORIGIN.
- **Wide load is for fully-in-bounds tiles only.** A tile whose free/stride-1 run overhangs the tensor
  **scalarizes** (an unmasked wide load would read OOB). For the interleaved recipe the free dim IS the
  stride-1 axis, so an M/N-ragged edge scalarizes. "No OOB global read" is an emit-level guard, not only a
  golden test.

---

## 6. N-D tensors -- rank-generic descriptors, one axis convention, no axis labels

- Tensor/tile/window/cooperative-load/store descriptors are **rank-generic**; nothing hardcodes rank 2.
- **One axis convention, no labels.** Batch axes first, the reduction axis last, the memory layout only in
  the strides; nothing reads what an axis means. A rank-`r` tile covers the tensor's last `r` axes and
  each leading axis is positioned by the window origin; other orders go through `permute`, anything more
  general through a transform graph. The rule and its GEMM reading (A `(M, K)`, B `(N, K)`, C `(M, N)`)
  live in `tiling_api_surface.md` sec 3, "One axis convention".
- **Why no labels.** A label on a general unit (a descriptor, a window, a verb) is a workload assumption
  -- "contraction" and "free" are GEMM words that a copy, a reduction or a convolution does not have. The
  one GEMM-specific check is named for GEMM and called on purpose: `check_gemm_shapes` reads A, B and C
  positionally, so its domain is stated in its docstring rather than inferred from labels.
- **LDS is a flat buffer; the descriptor is its layout.** The verbs address LDS through the descriptor's
  strides exactly as global memory, so a staged buffer with several stages is a rank-3 descriptor whose
  leading stage axis is positioned by the origin. The served-group property of any new LDS descriptor is
  re-derived by dumping the address map, NOT inherited from another layout (`lds_banks.md`).
- **Batch loop owner, by shape:** independent batch -> the author's outer loop (later a typed pipeline
  stage that emits nothing); batch *inside the contraction* -> the `TileMmaDriver`'s carried accumulator
  (a naive outer loop is silently wrong -- each per-batch MMA is individually sound so no gate fires);
  attention chaining -> deferred compute-fusion.

---

## 7. No monolithic files

One concept per file. The recorder/analysis machinery is organized behind a defined internal seam, not
reached into ad-hoc by tooling.

---

## 8. Coordinate graphs -- `tensor_transforms`

One line per rule; the rules themselves are in `tensor_transforms_api.md` (section in brackets).

- **The graph holds no tensor and no bound.** A length lives only on the `TensorDesc`; a halo is an `offset`,
  never a restated length, and there is no PAD kind (sec 3).
- **Validity comes from coordinates, never from addresses**: a check exists exactly where a coordinate's range
  can leave an input domain, a read domain or its tensor bound (sec 7).
- **No flags.** Whether an axis is checked follows from data (ranges against domains and bounds); there is no
  per-axis switch and no `bounds=` override (sec 7).
- **Structured address invariants**: one uniform term per set of graph inputs, a per-lane part that starts at 0
  and fits i32, a Python-int constant, widths chosen from unconditioned ranges so no lane's arithmetic can
  overflow (sec 9).

---

## Where the domain theory lives (point, never restate)

| topic | owner |
|---|---|
| how-to-use catalog, default-vs-override spectrum | `tiling_api_surface.md` |
| MMA soundness, derived C, canonical machinery | `mma_is_machinery.md` |
| interleaving recipe, C-store coalescing, access width | `tiling_interleaving_design.md` |
| stage classes, edge kinds, bridge gating | `label_flow_and_transforms.md` |
| banks, served groups, swizzle/pad, binding stage | `lds_banks.md` |
| coordinate graphs, split addresses, derived bound checks | `tensor_transforms_api.md` |
