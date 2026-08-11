# Scaling Portable-IR Record+Roll to All Operators and Architectures

This plan scales the proven record+roll prototype
(`rocke/portable_ir/`, C++ VM in `platform/cpp/portable_ir/`) from the unified-attention 2D kernel
to the full instance/helper surface and all supported gfx targets. It is grounded
in a survey of `instances/` (78 `build_*` entry points across ~45 families) and
`helpers/` (46 modules).

## Goal

Ship a **compact, parametric, CPython-free** kernel JIT path:
- author kernels in Python (unchanged production builders),
- **record** each into a portable IR / recipe,
- **roll** compile-time unrolls + spec-scaled constants into one parametric
  recipe per family/arch,
- the C VM expands + lowers + comgr-compiles at runtime, **byte-identical** to
  production.

The win is **storage / distribution** (one rolled recipe vs N growing per-shape
artifacts), not compile time (comgr-bound). Correctness is enforced by a
byte-identical HSACO oracle per (kernel, shape, arch).

## Two operations, two cost profiles

| Operation | Generality today | Effort to scale |
|---|---|---|
| **Record** (KernelDef -> recipe) | works for any kernel (it just serializes the built op stream) | ~0 once a build-time interception recorder exists |
| **Roll** (compress repeated runs + parameterize constants) | validated on attention-2D `head_size` | per-pattern roller work |

So **record is universal and near-free; roll is the incremental value-add.** A
sound rollout records everything first (concrete = portable-IR parity, already
byte-identical), then rolls family-by-family for the storage benefit.

## Status — productization landed this iteration

The plumbing for the CPython-free path is now in place and gated (see `README.md`
for the directory map and how to run each piece):

- **Reorg.** `portable_ir/` is split into `src/` (engine + runtime binding),
  `utils/` (device-free oracle), `examples/` (demo kernels), `drivers/`
  (harnesses), `tests/`.
- **CBOR codec + bundle** (`src/recipe_bundle.py`, C++ `cbor_dom.cpp`): recipes/bundles
  (`rocke.bundle/v1`) ship as compact CBOR the C VM decodes into the same DOM as
  JSON. ~3× smaller than JSON; the bundle packs many recipes keyed by `(key,arch)`.
- **Online in-process path** (`src/online.py` ctypes ↔ C++ `online.cpp`): the Python
  builder hands a serialized recipe/IR to the C backend and gets `.ll` back, no
  subprocess, no pybind build step.
- **Recipe `.ll` is now byte-identical, not just HSACO-equivalent.** Concrete
  recipes (empty `spec`) carry unique Python SSA names; the VM names each value
  verbatim from its bind (mirroring the IR importer), so the recipe-VM `.ll`
  matches the Python lowerer byte-for-byte.
- **Multi-result `outs`** expand path in the C VM is **done** (exercised on a real
  `inline_asm_multi` kernel, HSACO byte-identical).
- **Cross-arch parity gate** (`drivers/parity_matrix.py`): both backend paths
  (engine-import and recipe-VM) vs the
  Python lowerer — **45/45 buildable kernels byte-identical on gfx942 and gfx950**
  (flavor pinned). A `tile.buffer_load_vN`/`store_vN` → `*_f16` opcode **alias** in
  the portable-IR layer (engine core untouched) made the conv kernels round-trip.
- **Multi-axis rolling** (`src/roll_nd.py`, `drivers/roll_nd_coverage.py`): one
  recipe now covers a family's non-reduction axes' **cross product** rather than one
  axis at a time — 4/4 priority families, no VM change. See
  [Multi-axis rolling](#multi-axis-rolling-delivered--one-recipe-per-axis-cross-product)
  and the [pitfalls](#pitfalls--sharp-edges) it surfaced.

## Step 1 — universal capture: build-time interception recorder  ✅ DONE

`RecordingIRBuilder` (`rocke/portable_ir/src/recording_builder.py`, subclass of
`core.ir.IRBuilder`) records each emitted op live by intercepting `_emit`,
`param`, and `push_region`/`pop_region` into a recipe as it is built. Because it
rides `_emit` (the single op choke point) rather than the public op-builder
methods, **new ops are captured automatically**. `record_kernel(build_fn)`
auto-rebinds the `IRBuilder` name across all imported `rocke` modules, so **any**
production `build_*` records with **zero kernel changes** — helper/closure/
dataclass/descriptor logic executes normally; only emitted ops are captured.

- Output: a concrete (per-shape) recipe == portable IR. Byte-identical (it equals
  the byte-identity-proven `kerneldef_to_recipe` walk; also multi-result aware via
  `outs`).
- **Coverage proven:** `drivers/record_coverage.py` drives the recorder off the
  parity spec set — **55/65 emitters record faithfully, 0 recorder failures**,
  spanning small ops and every tier — GEMM, conv, attention, MoE (scalar unified
  2D, tiled gfx942/gfx950 2D/3D, WMMA gfx1151/gfx1201). The 10 skips are bespoke
  multi-kernel / multi-arg *emitter* signatures in the reuse harness, not recorder
  gaps.
- **Drift guard:** `tests/test_recording_builder.py` asserts the live recording
  matches an independent post-hoc walk + the legacy recipe on synthetic and real
  kernels, and a coverage gate asserts 0 recorder failures across the surface — so
  any future `IRBuilder` change that breaks capture trips CI.

## Step 2 — generalize the roller  🚧 IN PROGRESS

The bespoke roller (`drivers/roll_recipe.py`) handled exactly ONE index-progression
unroll + ONE loop-carried accumulator + linearly-spec-scaled int constants,
hard-coded to attention head_size (D=64/128, VEC=8).

**Delivered — a general, safe-by-construction roller:**
- `src/roller.py` — multi-trace structural roller: aligns traces, finds repeated
  runs via insertion-point detection + periodic expansion (robust to tandem
  repeats), and rolls each run into a `static_for` with **multiple** index-ladder
  constants (`v0 + i*delta`), **multiple** loop-carried values (cross-block
  def→use, threaded via the carry-alias trick), spec-scaled in-run constants, and
  **nested** rolling (runs inside `scf.for`/`scf.if` bodies recurse naturally).
- `utils/recipe_expand.py` — a pure-Python recipe VM (mirrors `recipe_vm.c`) +
  `recipes_equiv` (structural equality modulo SSA renaming). This is the
  **device-free oracle**: `expand(parametric, spec) ≡ recorded_concrete(spec)`
  proves byte-identity without comgr (concrete→HSACO byte-identity is already
  established, and α-equivalent op streams lower identically).
- `src/roll.py` — the `roll(build_at, axis, sample_points, holdout_points)` driver:
  records traces from an **unmodified** builder (Step-1 recorder), infers one
  parametric recipe, then **verifies it against every sample AND held-out point**.
  On any failure it returns `(None, reason)` — the caller keeps concrete per-shape
  recipes (graceful degradation; **never a wrong roll**).
- `src/roll_nd.py` — the `roll_nd(build_at, axes={...}, structural_axis=...)`
  driver: the same contract over **several axes at once**, so one recipe covers
  their cross product. See "Multi-axis rolling" below.

**Multi-run per level (delivered).** A single level can hold several independent
runs (GEMM pipeline-nest + CShuffle; two unrolled loops). The roller finds the
first run (divergence-anchored, candidate periods tried smallest-first), rolls
it, and **recurses on the remainder** — validated on a synthetic two-run kernel.

**Validated per-probe status** (`roll_coverage.py`, `tests/test_roller.py`). One
representative kernel per tier, not the tier's full family list — tiers are
numbered per Step 4:

| Tier | Kernel | Axis | Status |
|---|---|---|---|
| T1 GEMM | `gemm_universal` k-atom nest | tile_k | fallback (oracle-rejected) |
| T3 attention | **unified-attention 2D** (Section-3 kernel) | head_size | **rolled 54×** |
| T5 deep fused conv | **`deep_fused_conv_pool`** conv0→conv1→pool | pool tile | fallback (diagnosed) |
| *(untiered)* small op | `qk_block` vec8 dot | head_size | **rolled 78×** |

Also: a synthetic kernel with **two carries + two ladders** rolls, and a
quadratic-constant kernel is correctly **rejected at held-out N** (2-point fit
predicted 14, actual 16) → concrete. The held-out check makes two-trace
inference safe; the oracle makes every reported roll byte-equivalent.

**Variable loop-carry fan (T5) — capability delivered.** A runtime `scf.for`
whose iter-arg count scales with the axis (e.g. `deep_fused_conv_pool`'s per-
output-row mfma accumulators, 2→4 as the pool tile grows) is now representable
and rollable:
- `recipe_expand` supports a **parametric `scf_for`**: rolled iter-args/results
  (a spec-derived *number* of loop-carries) + **format register names**
  (`acc_m{lane}_n0`) + rolled `scf.yield` operands. Validated byte-equivalent
  across lane counts (`tests/test_roller.py::TestParametricFanExpander`).
- The roller **auto-detects** the fan (differing `scf.for` iter arity, in both
  the equal-length and divergent branches), infers the lane count
  (`linear(axis)`), and rolls iter-args + per-lane body run + yield, re-pointing
  the parent's result references. Rolls a clean synthetic fan (verified at
  held-out lane counts).
- The recipe-VM C mirror of parametric iter-args is **delivered** (see "C recipe
  VM — parametric surface complete" below); the Python oracle cross-checks it.

**`deep_fused_conv_pool` diagnosed — the fan is NOT the real blocker.** A
data-dependence analysis of its scf.for body (per-lane backward cones) refuted
the "interleaved fan" hypothesis:
- Each lane has only **~5 private ops** (the final accumulate); the loop-carry
  fan is a sliver.
- The body is dominated by **~687 ops of axis-scaled shared/spatial work** (the
  conv0→conv1→pool spatial-output unroll) that scales with the pool tile
  *independently* of the lane count.
- The shared prologue contains **non-affine constants** (e.g. `3 → 4` as
  `pool_tile_w` goes `4 → 8`, slope `1/4`): the pool tile enters address/size
  math as a **spatial product**, so constants are non-linear in the axis.

So `pool_tile_w/h` are **not clean affine axes** for `deep_fused_conv_pool`;
rolling it would need (a) **multi-axis / polynomial constant inference** and
(b) rolling the **spatial-output unroll**, not interleaved-fan handling. This is
the "non-linear constants stay concrete" limit (below) — `deep_fused_conv_pool`
is a legitimate **concrete-per-shape** kernel under the current affine roller.
Interleaved-fan handling remains useful for kernels whose lanes *genuinely*
interleave (e.g. a GEMM CShuffle `m×n` epilogue) — just not for this one.

**Lane-ref-aware runs (delivered).** A run whose blocks index into a fan's
per-lane values — e.g. a **CShuffle / reduction epilogue over the loop results**
— now rolls: `_roll_run` rewrites operands referencing lane `start+k` to
`fmt{var}` and shifts the static_for to the actual lane range (handling the
**lane offset**, e.g. a reduction that seeds with result 0 and loops 1..N).
Validated on a fan + full reduction-over-results synthetic (`test_roller.py::
TestParametricFanExpander::test_auto_roll_fan_with_reduction`), byte-equivalent
at held-out lane counts.

**GEMM + CShuffle (T1) — body-rolling mechanism delivered; one blocker left.**
`gemm_universal` over `tile_n` (fixed warp grid) is **affine-clean** and a genuine
variable-fan target (K-loop carries `mfmas_m × mfmas_n` accumulators, `16→32→64`).
The fan body is now rolled via the general aligner with the full lane machinery:
- **Output lane-refs** — `_roll_run` aliases each block def that belongs to a
  per-lane family (e.g. a B-load partial, or the mma output for `scf.yield`).
- **Inter-run per-lane value flow** — `_lane_families` detects a run's per-block
  defs that are consumed per-lane by a *downstream* run and registers the family,
  so the consumer's `_roll_run` rewrites them. Validated end-to-end on a phased
  synthetic (`test_roller.py::...test_inter_run_per_lane_flow`): phase-1 produces
  per-lane partials, phase-2 consumes them, plus a reduction epilogue — rolls and
  byte-equivalent at held-out lane counts.

**Real GEMM CShuffle — ROLLS (9.9× over tile_n).** Finishing it required
*data-flow-aware* segmentation plus type/attr parameterization, all now in the
roller:
- **Cone-based lane labeling** (`lane_label_body`) — labels each body op by the
  lane whose result it ultimately feeds (backward from the per-lane yield
  operands), separating the shared **A-tile** from per-lane **B-tiles** even
  though they share opcodes. ("Separate shared from per-item by *meaning*.")
- **Scratchpad matching** (`scratchpad_edges`) — links each `smem_store` to the
  `smem_load` that reads the same buffer+address, so a side-effecting store
  inherits the lane of its consumer *through LDS memory* (no SSA edge). ("Match
  the drop to the pickup by *where on the bench* it landed.")
- **Lane-label body segmentation** (`_segment_by_lane`, `_roll_fan_body`) —
  splits the body into shared regions (merged 1:1) and per-lane PHASES (rolled at
  the fan's lane count; phases split on lane reset), threading inter-phase
  per-lane values as families.
- **Type & attr parameterization** — `_merge_instr` now parameterizes scaling
  *integer attrs* (e.g. a `sched_group_barrier` instruction count) and scaling
  *result types* (`_merge_type`, e.g. an `smem_alloc` buffer `shape:[TN,16]`);
  the expander evaluates intexprs in attrs and type fields.

Verified byte-equivalent at sampled (32/64) and held-out (128/256/96/192)
`tile_n`. (`test_roller.py::TestRollGemmCShuffle`, `::TestLaneAnalysis`.)
Coverage is now **T1–T4 rolled on at least one axis each, T5 fallback**
(conv-pool's non-affine spatial-product constants) — with the caveat recorded in
Step 4 that only T1 `tile_n` and T4 `hidden` compress structurally; T2 and T3 roll
constants only. Note: `tile_k` is a *separate* GEMM axis that still hits a
non-affine LDS-sizing case.

### Multi-axis rolling (delivered) — one recipe per axis cross product

Everything above parameterizes ONE axis, so covering `k` axes cost `k` recipes and
none of them moved together. `src/roll_nd.py` covers the cross product with a
single recipe.

**Why the non-reduction axes first.** A kernel's axes split by what they do to the
trace. The **reduction** axis (GEMM `tile_k`, attention `head_size`) drives the hot
loop, so moving it changes structure — op counts, vector widths, LDS sizing — and
needs structural inference. The **outer** axes (inter-tile problem shape,
intra-tile geometry) frequently leave the instruction sequence completely alone and
move only **constants**: five of the seven gated axes emit an identical op count at
every value. That second group needs no structural inference at all, and a
constant model composes across axes for free, which is what makes the cross
product reachable cheaply.

**Architecture: annotate, then roll.** The aligner is pairwise (`align(la, lb)`)
and the fan/run machinery is built on that, so rather than rewriting it to consume
`T` traces, the axis dependence is pushed into the **concrete** traces before
structural rolling starts:

1. Record the base point, then one **probe** per axis that moves that axis alone.
2. Check each probe is structurally identical to the base — same op at every
   position, and no non-integer field moved (`_nonint_key` guards against a dtype
   string that tracks an axis). A probe that fails this names a *structural* axis,
   which belongs to `structural_axis=` instead.
3. Solve every integer-bearing field — typed-int attrs and integers inside result
   types, the same set `_merge_instr` parameterizes — and rewrite it as an intexpr
   over `{spec: axis}`.
4. Optionally roll a structural axis **on top** of the annotated traces.
5. Verify with the oracle at every cross-product point and every holdout.

Step 4 composes because `merge_intexpr` reconciles two intexpr *trees* that differ
only at integer leaves, fitting each differing leaf in the structural axis. So a
constant may depend on the structural axis and a shape axis at once — the mixed
case emits `{add: [{mul: [{spec: S}, 2]}, {mul: [{spec: N}, 8]}]}` while the run
still compresses to a `static_for`.

**The solve** (`roller.affine_solve` / `affine_intexpr`) is exact — Gaussian
elimination over `Fraction`, no least squares — and is rejected unless it
reproduces every sample point. Per-axis expressibility mirrors `_linear_expr`: an
integer coefficient becomes a `mul`, a unit fraction `1/d` becomes a `div` (the
`count = axis // vec` shape), and any other fractional coefficient is refused
because `div` floors and a non-unit fraction would only agree by luck.

**The probe/verify asymmetry is the safety argument.** Inference reads
`1 + sum(n_j - 1)` traces; verification checks `prod(n_j)` grid points plus
holdouts. That gap is deliberate — see the first pitfall below.

**Measured** (`drivers/roll_nd_coverage.py`, ~0.9 s, no comgr or GPU):

| Family | Axes | Grid | Held out | Traces recorded | `static_for` |
|---|---|---|---|---|---|
| conv_implicit_gemm | `N`, `K` | 4 | 3 | **3** | 0 |
| attention_dense | `seqlen_kv`, `num_query_heads` | 4 | 1 | **3** | 0 |
| fused_moe/gather | `hidden`*, `tokens` | 4 | 1 | **4** | 1 |
| gemm_universal | `tile_n`* | 2 | 2 | **2** | 5 |

`*` = also rolled structurally. The **C VM needed no change**: a multi-axis
constant is still an intexpr over spec values, so `--int N=.. --int K=..` binds
both, and the standalone CLI replays the two-axis conv recipe to byte-identical
`.ll` at all 7 points including `N=64, K=512` (8x the base on *both* axes at once).
Unit tests: `tests/test_roll_nd.py`; CI lanes:
`tests/portable_ir/test_recipe_roller.py::test_roll_nd_cross_product` and
`::test_standalone_cli_replays_multi_axis_recipe`.

## Pitfalls & sharp edges

Each of these bit during development and is now either gated or pinned by a test.
They are recorded because most of them look like success until something checks.

**1. A cross term fits every one-axis probe perfectly.** Three points in general
position determine an affine function of two axes, so a constant scaling with
`N*K` matches the base and both one-axis probes *exactly* and is wrong everywhere
else. This is why verification sweeps the whole grid instead of trusting the fit,
and why the interior points matter as much as the extrapolated ones. Pinned by
`test_cross_term_refused_at_interior_point`; a real instance is conv `C`
(predicted 9, actual 8).

**2. Holdouts inside the sampled box prove almost nothing.** Two-point inference
fails by *extrapolating*, so a holdout must sit outside the sampled range to have
teeth — the quadratic probe is only caught because a held-out `N` predicts 14
against an actual 16. Interpolated holdouts would have passed.

**3. `div` is floor division.** A unit-fraction coefficient agrees only where the
axis is divisible by `d`. Verified points are therefore the contract: a JIT caller
binding an arbitrary spec value can leave the verified set and get a silently
floored constant. Keep spec values on the family's natural multiples, or add the
value to the gate.

**4. The oracle does not check the kernel symbol.** `recipes_equiv` compares
programs only, so a recipe whose `kernel_name_fmt` does not track the axis passes
every oracle check while emitting the *wrong symbol*. Reproduced: a kernel named
after a derived quantity (`g{3N+1}`) rolls "successfully" through `roll()` and
emits `derived_g10` at every `N`, wrong at `N=2` and `N=9`, with
`recipes_equiv == True` at all of them. Two things save this in practice — the
`.ll` SHA gates compare text that contains the symbol, and `roll_nd` verifies the
name at every point and refuses with a precise reason — but **`roll()` plus the
oracle alone is not sufficient** to conclude a rolled recipe is shippable. Pinned
by `test_name_carrying_a_derived_quantity_is_refused`; teaching `recipes_equiv` to
compare names is the real fix.

**5. Sample-point order is load-bearing.** The aligner requires the larger axis
value to record *more* ops; reversed, it refuses with `shorter-at-larger-axis`.
That refusal is also how a genuinely shrinking trace shows up (GEMM `tile_m`).

**6. Non-monotonic axes cannot be rescued by re-sampling.** GEMM `tile_m` moves
the k-loop body 90 -> 81 -> 114 ops over 16/32/64: `tile.mma` scales cleanly
(2 -> 4 -> 8) while the load path *re-vectorizes* under it
(`memref.global_load_vN` 3 -> 2 -> 3, address arithmetic 33 -> 28 -> 42). Sampling
32/64 or 64/128 instead does not help — there is no regime in which it is affine.
Axes like this need opcode/vector-width selection (P3 `static_if`), not a better
constant fit. Do not read a refusal as "sample differently".

**7. An axis that changes nothing looks like coverage.** MoE `tokens` moves **zero**
constants — the kernel is already token-agnostic — so "rolled over 2 axes" there is
really one axis plus a kernel-name suffix. When claiming coverage, check whether
the axis actually moved the body (the gate's `static_for` column and distinct-SHA
count both expose this).

**8. In-run ladder constants that also depend on a constant axis refuse.** After
annotation such a constant is an intexpr, so `_is_const_int` is false and
`_roll_run`'s ladder inference skips it, leaving one block's expression in the
template; the oracle then rejects the roll (predicted 12, actual 8 on a synthetic).
Safe, but it means "structural axis + shape axis" only composes when the shape axis
stays *out* of the unrolled block's per-iteration constants. Fixing it means
teaching the ladder step to fit intexpr leaves in the loop var, the same way
`merge_intexpr` does for the structural axis. Pinned by
`test_shape_axis_inside_the_ladder_refuses`.

**9. Run-boundary phase.** `_run_candidates` anchors at the first divergence and
walks back in whole periods, so when the tail after a run repeats the block's
signatures in a *rotated* order the only candidate offered starts mid-block and
swallows a tail constant. Declines rather than mis-rolls; reproduces identically
through single-axis `roll`, so it is not multi-axis-specific. Pinned by
`test_rotated_run_boundary_refuses_rather_than_mis_rolls`.

**10. A point is a dict, not a value.** `roll` takes bare axis values and
`roll_nd` takes one dict per point; mixing them up (or omitting an axis from a
point) now raises a message naming the mistake rather than failing deep inside
verification. Pinned by `test_point_must_be_a_dict_not_a_bare_value`.

## C recipe VM — parametric surface complete (on-device JIT)

`cpp/portable_ir/recipe_vm.cpp` now expands the rolled recipes on-device (no CPython),
at parity with the Python `recipe_expand` oracle:
- **format register names** (`{var}`/`{spec}` substitution),
- **rolled lists** for `scf_for` iter-args/results and emit operands (the
  variable loop-carry fan + rolled `scf.yield`),
- **intexpr in attrs and in result TYPES** (vector count, `smem` buffer shape),
- **`smem` types** + exact `smem_alloc` LDS naming (so the LDS symbol — which
  affects the object — matches the Python reference),
- **multi-result `outs`**.

Validated end-to-end (build rocke lib + replay CLI + comgr): one rolled recipe →
C VM expands per shape → **HSACO byte-identical to production**, for
`examples/qk_block.py` (head_size), the Section-3 attention kernel
(D64/128/256), and `examples/export_gemm_cshuffle.py`
(GEMM CShuffle, tile_n 32/64/128/256 incl. held-out). This closes Step 6's
"VM expand" path for every kernel family that rolls today.

The roller also now **auto-parameterizes the kernel name** (`g{TN}_..._t16x{TN}x16`)
by diffing the two sample names, so the VM-formatted name matches per shape.

**Other frontier (oracle-reported):**
1. **Type/attr parameterization.** `gemm_universal` over `tile_k`:
   `tile.smem_alloc`'s result *type/size* scales; the roller only parameterizes
   integer *constants* today. Needs spec-parametric types/attrs.
2. **Peeling (prologue/steady/epilogue)** for software-pipelined / double-buffered
   loops with boundary-special iterations.

Still to generalize (the survey's remaining patterns):

1. **Nested rolling.** GEMM/conv/MoE bodies are `for kk in range(k_atoms)`
   nested in `for mi/ni in range(mfmas_*)`. Roll innermost-out, each level a
   `static_for` over a spec-derived bound. CShuffle epilogue
   (`mfmas_m x mfmas_n x c_per_lane` writes) is a rollable nest.
2. **Peeling (prologue/steady/epilogue).** Software-pipelined K-loops
   (`SoftwarePipeline.run_ping_pong`) and double-buffer parity loops have
   boundary-special iterations. Peel the irregular head/tail (keep concrete) and
   roll the uniform core. Preserves byte-identity.
3. **`static_if` case-bodies.** Periodic or class-indexed variation (even/odd,
   `k % P`, first/last) rolls into a `static_for` whose body guards sub-bodies
   with `static_if` on the (compile-time) loop index. The VM picks the right body
   per iteration -> exact ops reproduced.
4. **Multi-axis constant parameterization.** ✅ **DONE** — `src/roll_nd.py`. An
   exact affine solve over N axes (`v = c0 + sum m_j*x_j`, unit fractions rendered
   as `div`) from `1 + sum(n_j - 1)` traces, verified across the whole axis cross
   product plus held-out points, so one recipe replaces one-per-axis. A
   structural axis composes on top. Non-affine constants are refused, not
   approximated: conv `C` (spatial product) and GEMM `tile_m` (non-monotonic,
   re-vectorizing load path) both decline. See P2 under "Phasing & effort".
5. **Runtime-loop conversion (opt-in, NOT byte-identical).** Where a compile-time
   unroll is value- (not structure-) dependent and a looped kernel is acceptable,
   convert to runtime `scf.for`. This changes codegen, so it is a kernel variant,
   gated behind explicit opt-in, never the default.

**Hard limit:** patternless per-iteration structural variation cannot be rolled
(no storage win); those stay concrete per-shape. Expected to be rare.

## Step 3 — architecture strategy

Arch enters three ways (`core/arch` SSOT `arch_specs.json`, `core/isa` backends,
`instances/gfx*` subfolders). Two regimes:

- **Arch-polymorphic builders** (most `common/*`: gemm_universal, conv,
  fmha_mfma): IR differs by arch only in atom `op_id`, waitcnt, datalayout, and
  **wave size / atom shape** (structural). Strategy: carry `arch` (and derived
  `wave_size`, `atom_op_id`) as **recipe spec fields**; where only op_ids/waitcnt
  differ, one recipe covers an arch *family* via `static_if` on catalog
  predicates (mirroring `attention_arch.py`). Where wave32/wave64 or atom shape
  changes the tiling structurally, emit **one recipe per arch-family**
  (CDNA-MFMA vs RDNA-WMMA, plus gfx950's wide atoms) — i.e. ~2-3 families, **not**
  one per gfx.
- **Genuinely divergent builders** (`gfx1151/wmma_*`, `gfx950 fastkv_regp`,
  gfx942-vs-gfx950 tiled attention): separate recordings anyway; record+roll each
  independently.

Net arch multiplier on artifact count: **#arch-families (~2-3)**, not
#gfx-targets. gfx1250 **is** now in `arch_specs.json` (alongside gfx90a, gfx942,
gfx950, gfx1151, gfx1201) with kernels under `instances/gfx1250/`, so it needs no
SSOT work before recording.

## Step 4 — operator tiers (rollout order)

Tiers are ordered by **product priority**, not by difficulty: GEMM, conv,
attention, MoE, then deep-fused conv. Attention deliberately precedes MoE even
though it is the hardest tier, which has a scheduling consequence — the
`static_if` flag/case work that both tiers need is pulled earlier than a
difficulty-ordered plan would put it (see Phasing).

| Tier | Families | Loop profile | Roller needs | Risk | Measured status |
|---|---|---|---|---|---|
| **T1: GEMM** | gemm_universal (+batched/grouped/streamk/mfma/mx/block_scale/multi_d) | `k_atoms x mfmas` nests + 1 runtime K-`scf.for`; pipeline + CShuffle | nested roll + peel; helper-expansion macros | medium | **rolls structurally** on `tile_n` (195 vs 322 ops); `tile_m` non-monotonic, `tile_k` refused |
| **T2: conv** | conv_implicit_gemm, conv_direct, img2col, pooling | implicit-GEMM nests over spatial windows | T1 + non-affine spatial constants | medium | **one recipe covers the `(N,K)` cross product** from 3 traces — constants only, op count flat; `C` non-affine; `tile_m`/`tile_n` refused |
| **T3: attention** | scalar unified 2d/3d/reduce (DONE for 2d), FMHA fwd/bwd/varlen/paged/splitkv/headgroup/fp8, sage/sparse, tiled gfx942/gfx950 | runtime KV `scf.for` + huge spec-`if` flag forks + compile-time tile nests | full roller + `static_if` flag handling; per-arch | high | **one recipe covers `(seqlen_kv, num_query_heads)`** from 3 traces — constants only; `head_size`/`block_n` refused |
| **T4: MoE** | fused_moe (5), moe_gemm_fused (4), moe_fused_mega(+fp8), moe_sorting (4), moe_smoothquant | GEMM skeleton + phase routing (`static_if`) | T1 + multi-phase | medium | **rolls structurally** on `hidden` (21 vs 27 ops) **and carries `tokens` in the same recipe**; `hidden@128` refused |
| **T5: deep-fused conv** | deep_fused_conv_pool | conv0→conv1→pool; pool tile enters address math as a spatial product | multi-axis / polynomial constant inference **and** spatial-output unroll rolling | high | refused (diagnosed, see Step 2) |
| *(untiered)* small ops | elementwise, reduce, layernorm2d, rmsnorm2d, transpose, permute, topk_softmax, smoothquant | shallow `range(vec)` + reductions | index roll + peel | low | the pattern the roller handles best (probe `qk_block` rolls 78×); not competing for priority |

Two notes on reading that status column.

**"Rolls" does not imply structural compression.** Of the seven axes gated today,
only two — GEMM `tile_n` and MoE `hidden` — actually emit a `static_for` and
shrink the op count. The other five produce a recipe with *exactly* the concrete
op count: the kernel emits the same instructions at every axis value and only the
**constants** move, so rolling buys shape coverage (one artifact serves the
family, held-out values included) without buying storage. Both are wins, but only
the first is the "storage payoff" the phasing language promises, and conv is
currently in the second group.

That said, the constants-only axes are no longer *separate* recipes. Multi-axis
rolling (P2, `src/roll_nd.py`) folds a family's non-reduction axes into ONE recipe
covering their cross product, so the artifact count is now per family rather than
per family-times-axis, and the axes move together instead of one at a time.

**Difficulty no longer tracks tier order.** The high-expansion helpers
(`SoftwarePipeline`, `CShuffleEpilogue`, `AsyncTileLoader`,
`mfma_attention_*_inner_body`) concentrate in T1, T3, and T5 under this ordering,
so effort is front- and back-loaded rather than ramping.

## Step 5 — validation harness (the safety net)

A matrix regression that, for each (kernel family, shape set, arch):
1. builds production `KernelDef` and lowers -> reference HSACO,
2. record+rolls -> one recipe, VM-expands per shape -> HSACO,
3. asserts **byte-identical** HSACO.

This makes rolling safe-by-construction: any mis-roll fails the comparison.

**Delivered (concrete tier):** `drivers/parity_matrix.py` is the
parameterized matrix runner over every parity-emitter kernel × arch,
checking BOTH backend paths
(engine import + recipe VM) against the Python lowerer at the `.ll` level
(byte-identical) with one flavor pinned — **46/46 buildable kernels byte-identical
on gfx942 and gfx950**.

**Delivered (rolled tier):** `drivers/roll_hsaco_parity.py` closes the two gaps
that leaves open — it exercises the *rolled* parametric recipe rather than a
concrete trace, and compares final **HSACO** rather than IR text: 22/22 points
byte-identical across gemm/conv/attention_dense/fused_moe on gfx950, 8 of them
at held-out axis values. Rolled `.ll` is byte-identical too, at all 22 points:
a rolled recipe cannot replay names verbatim the way a concrete one does, but
the recipe carries Python's per-op `result_name_hint` and the roller keeps
Python's lane naming for loop-carry fans, so both engines mint the same names
off the same counter. A checksum therefore suffices to compare the rolled path.

**Delivered (multi-axis tier):** `drivers/roll_nd_coverage.py` gates the axis
*cross product* — 4/4 families, every grid point and every holdout verified
against its own concrete recording, plus the kernel name at each point (pitfall 4).
It needs neither librocke nor comgr, so unlike the lanes above it cannot skip;
`--ll` adds an `.ll` SHA report in the same currency as the rolled gate. The driver
also re-probes the axis pairs it *refuses* and fails if one starts working, so the
frontier recorded in this document cannot drift silently.

Extend all three drivers over `instances/SUPPORT_MATRIX.md` for the full shape grid.

## Step 6 — productization

- `record_kernel(build_fn) -> (kernel, recipe)` (interception recorder) — **done**
  (`src/recording_builder.py`).
- `roll(build_at, axis, ...) -> parametric recipe` (multi-trace driver + roller) —
  **done** (`src/roll.py`); `roll_nd(build_at, axes={...})` for an axis cross
  product — **done** (`src/roll_nd.py`).
- A bundle writer emitting CBOR recipes keyed by `(key, arch)` — **done**
  (`src/recipe_bundle.py`, schema `rocke.bundle/v1`; the C VM serves a recipe by
  key via `rocke_recipe_run_from_bundle_cbor`). zstd compression is the remaining
  wrapper.
- In-process online lowering binding — **done** (`src/online.py` ctypes ↔ C
  `online.c`).
- The provider's `ArtifactStore` recipe path (VM expand alongside `.hsaco`/`.ll`,
  gated by a provider-side C-JIT flag) — **pending integration**.
- CI running the parity matrix, the rolled-recipe HSACO gate, and the device
  replay gate (`tests/portable_ir/test_portable_ir.py`,
  `drivers/roll_hsaco_parity.py`, `drivers/gpu_replay.py`) on every kernel
  change — **not wired yet**; no workflow references rocke. These gates are
  run by hand today. See
  [`portable_ir_production_readiness.md`](../../../dsl_docs/architecture/portable_ir_production_readiness.md).

## Onboarding a new instance — what code (if any) is required?

Steady-state question: once the roller is scaled, does a newly developed instance
need bespoke record/roll code? **For the concrete / CPython-free path, never. For
rolling, usually just a small spec-axes declaration; occasionally a one-time
roller extension that then covers all future instances of that pattern.**

| New instance | Record (concrete / CPython-free) | Roll (storage win) |
|---|---|---|
| uses existing primitives, parametrization already in the roller's pattern library (single-axis index rolls incl. runs inside a region, loop-carry fans, affine multi-axis constants, spec-parametric types/attrs — **not** peel or `static_if` cases, see P2/P3) | **0 code** — records + lowers byte-identically automatically | **name the axes and their sample values** (`roll_nd(build_at, axes={…}, structural_axis=…)`, or `roll(…)` for a single axis) — 0 code |
| uses a genuinely new structural-variation pattern | **0 code** | **extend the roller once**; the extension is amortized across every future instance with that pattern |
| introduces a brand-new IRBuilder op (primitive) | **0 code** (captured as a generic op) | unaffected, *but* see caveat below |

Why record is free: `RecordingIRBuilder` intercepts `_emit` (the single op choke
point), `param`, and `push_region`/`pop_region` — not the public op-builder
methods — so it captures *any* emitted op stream (helpers/closures/descriptor math
just execute). `record_kernel(build_fn)` auto-rebinds the `IRBuilder` name across
all imported `rocke` modules, so it works no matter where the builder constructs
its `IRBuilder`. Proven: 55/65 parity emitters record faithfully, 0 recorder
failures, across all tiers (see Step 1).

### Caveats & corner cases

- **New primitive ⇒ C VM lowering owed.** Record captures a new op generically,
  but the C recipe VM must know how to *lower* that opcode. That lowering is work
  owed to *any* C JIT backend regardless of record+roll, and the byte-identical
  oracle catches a missing/wrong lowering. Two guardrails already exist: the
  recorder **deliberately raises** on a new *region-bearing* op (loud "extend me"
  signal — only `scf.for`/`scf.if` are modeled), and the `RecordingIRBuilder`
  drift tests fire if `IRBuilder`'s op/region/param surface changes.
- **Graceful degradation, never breakage.** If the roller finds no pattern in a
  region, it keeps that region **concrete per-shape** — still correct, just less
  compact. A new instance is never *broken* by missing roller support; it only
  loses some compression. (See "Helper-expansion bloat" below.)
- **Multi-result ops.** The recorder is multi-result aware (emits `outs` for N>1
  result ops, e.g. `inline_asm_multi`); the C VM's N-result expand path is **done**
  (validated HSACO byte-identical on a real `inline_asm_multi` kernel).
- **Multi-kernel / multi-arg builders.** A few builders emit several kernels per
  call or take non-standard build signatures (e.g. `(spec, arch, ...)` tuples).
  `record_kernel` matches the returned `KernelDef` by identity, so single-kernel
  returns are fine; multi-kernel emitters need one `record_kernel` call per
  returned kernel. (These are the 3 `record_coverage.py` skips — harness plumbing,
  not recorder gaps.)
- **Non-linear / data-dependent constants.** The multi-axis solver infers *affine*
  spec-scaled constants only. A non-affine constant makes the whole roll decline
  (the caller keeps concrete per-point recipes) rather than being approximated —
  and because verification spans the axis cross product, a constant that looks
  affine along each axis alone but carries a cross term is caught too. Value- (not
  structure-) dependent unrolls are only convertible via the opt-in runtime-loop
  path, which is **not** byte-identical and is never the default.
- **Arch SSOT.** New instances inherit arch handling for free *if* the arch is in
  `arch_specs.json` + the backend registry; a genuinely new target must be added
  there first, and recipes carrying `arch` spec fields regenerated. (All targets
  currently in the plan's scope, gfx1250 included, are already registered.)

## Phasing & effort

Status below is measured, not planned; each claim names the gate that produced it.

### P1 — recorder + parity oracle · ✅ DONE

Interception recorder plus the matrix oracle, unlocking the CPython-free path
(no storage win). Gates: `parity_matrix` 46/46 byte-identical `.ll` on gfx942 and
gfx950 across both replay paths; `hsaco_parity` 45/45 gfx950 and 32/32 gfx942
byte-identical HSACO; `record_coverage` 0 recorder failures across the surface.

The original wording promised *all 78 kernels*. The measurable surface is smaller:
56 kernels reach the comparison, 12 are build-skipped, and 10 are correctly
declined by the lowerer as wrong-arch. The **mechanism** is universal; the 78
figure is not a gate anyone runs.

### P2 — generalize the roller · 🟡 PARTIAL (outcome ahead of mechanisms)

Target was rolling T1 (GEMM) + T2 (conv). The tier goals are partly met, but
through different machinery than this phase names — worth separating so the
remaining work is not read as done:

*Delivered:* single-axis run rolling, including runs nested **inside** an
`scf.for`/`scf.if` body; variable loop-carry fan rolling with lane-label analysis;
linear spec-scaled constants; spec-parametric **types and attrs**; automatic
kernel-name parameterization.

**Multi-axis inference — landed for the non-reduction axes** (`src/roll_nd.py`,
gated by `drivers/roll_nd_coverage.py`). One recipe now covers a *cross product*
of axes instead of one recipe per axis, which is the coverage the tier table
promises. Every integer-bearing field is fitted as an exact affine function of
several axes at once, `v = c0 + sum_j m_j*x_j`, with a unit-fraction coefficient
rendered as `div` (the `count = axis // vec` shape). A structural axis may be
rolled *on top* of that model, so a constant can depend on the structural axis and
a shape axis simultaneously.

Measured, oracle-verified at every grid point and every held-out point:

| Family | Axes | Grid | Held out | Traces recorded | `static_for` |
|---|---|---|---|---|---|
| conv_implicit_gemm | `N`, `K` | 4 | 3 | **3** | 0 |
| attention_dense | `seqlen_kv`, `num_query_heads` | 4 | 1 | **3** | 0 |
| fused_moe/gather | `hidden`*, `tokens` | 4 | 1 | **4** | 1 |
| gemm_universal | `tile_n`* | 2 | 2 | **2** | 5 |

`*` = also rolled structurally. The recording asymmetry is the point: inference
reads `1 + sum(n_j - 1)` traces (3 for a 2x2 grid) while verification checks
`prod(n_j)` grid points plus the holdouts, so a **cross term** — a constant
scaling with `N*K` — fits every one-axis probe exactly and is still caught at the
first interior point. `tests/test_roll_nd.py` pins that case.

The C VM needed **no change**: a multi-axis constant is still an intexpr over spec
values, so `--int N=.. --int K=..` binds both. The standalone CLI replays the
two-axis conv recipe to byte-identical `.ll` at all 7 points, including
`N=64, K=512` (8x the base on both axes at once).

*Still open — the two other mechanisms this phase lists, plus one newly
characterized:*
- **Nested rolling** in the intended sense (a *nest* of loops rolled
  innermost-out, each level its own `static_for`). The roller emits **no nested
  `static_for`** on any family today; what works is one run per level.
- **Peeling.** Not implemented. Boundary-special iterations are only *named* as a
  hypothesis in a refusal message; there is no peel code path.
- **Multi-axis *structural* inference.** The landed work models constants across
  axes; it does not discover a *loop nest* whose trip counts belong to different
  axes. GEMM's mma nest is exactly that shape (`mfmas_m x mfmas_n`), so it is the
  natural next step — but see the `tile_m` finding below for why it is not just an
  inference gap.
- **Run-boundary phase.** `_run_candidates` anchors block boundaries at the first
  divergence and walks back in whole periods, so when the tail following a run
  repeats the block's signatures in a rotated order, the only candidate offered
  starts mid-block and swallows a tail constant. The oracle catches it and the
  roll is declined, so this costs compression rather than correctness. Reproduced
  by `test_rotated_run_boundary_refuses_rather_than_mis_rolls`, and it reproduces
  identically through single-axis `roll`, so it is not multi-axis-specific.

Two axes are refused for reasons worth recording, because neither is fixed by more
inference:
- **conv `C`** enters the spatial-product constants non-affinely (predicted 9,
  actual 8 at a held-out point) — it needs polynomial, not affine, inference.
- **GEMM `tile_m`** is **non-monotonic**: over `tile_m` = 16/32/64 the k-loop body
  goes 90 -> 81 -> 114 ops, because `tile.mma` scales cleanly (2 -> 4 -> 8) while
  the load path *re-vectorizes* (`memref.global_load_vN` 3 -> 2 -> 3, address
  arithmetic 33 -> 28 -> 42). No affine model exists in any regime — sampling
  32/64 or 64/128 instead of 16/32 does not help. Covering the GEMM tuning space
  therefore needs opcode/vector-width selection (P3-style `static_if`), not a
  better constant fit.

Consequently T1 rolls on `tile_n` but not `tile_m`/`tile_k`; T2 and T3 now cover
their shape cross products, still with zero `static_for`.

### P3 — `static_if` flag/case handling · ❌ NOT STARTED

Needed for T3 attention flag forks and T4 MoE phase routing. The VM executes
`static_if` and hand-authored recipes use it (`examples/mini_attn.py` forks on
`use_norm`), but **nothing infers it**: the roller emits zero `static_if` across
every family that rolls today. Flag-fork coverage is therefore manual authoring
only. Under the reprioritized tiers this phase gates the #3 priority rather than
the #4, so it moves earlier than the old ordering implied.

### P4 — targets, packaging, integration · 🟡 PARTIAL

- **gfx1250 in the arch SSOT + backends — done**, though as general arch work
  rather than portable-IR work: `arch_specs.json` carries it alongside gfx90a,
  gfx942, gfx950, gfx1151, gfx1201, and there are `instances/gfx1250/` kernels.
- **CBOR bundle — done** and gated: codec, `(key, arch)` container, C-side lookup,
  exact round-trip.
- **zstd compression — not started.**
- **Provider `ArtifactStore` recipe path — not started**; the symbol does not
  appear in the source. Tracked as gap 7 of the production readiness review.
- **Ship — blocked**, on the P0 shared-library linker collision (gap 1).

## Risks & fallbacks

- **Flag-branch explosion (tiled attention):** hundreds of build-time `if`
  forks. Fallback: keep flags as **separate recipes** per config (still rolled
  over head_size/tiles) rather than `static_if`-ing every flag; or `static_if`
  only the high-fanout flags.
- **Helper-expansion bloat:** if rolling can't compress a helper's one-shot
  subgraph, the recipe approaches concrete size for that region — acceptable
  (correct, just less compact); revisit with recipe "macro ops" if needed.
- **Non-byte-identical only acceptable** for the opt-in runtime-loop conversion;
  everything else must pass the oracle — noting that the oracle compares *programs*
  and not the kernel symbol, so it is necessary and not sufficient (pitfall 4).
- **Arch SSOT drift:** recipes carry `arch` spec fields; regenerate when
  `arch_specs.json` changes; the matrix oracle catches divergence.

These are project-level risks. For the engineering traps that bit while building
the roller — cross terms that fit every probe, floor-division extrapolation, an
axis that changes nothing — see [Pitfalls & sharp edges](#pitfalls--sharp-edges).

## Bottom line

Record is universal and cheap (interception recorder), and it is **done** — the
CPython-free path is available for every kernel that builds, proven byte-identical
at both `.ll` and HSACO. Roll is the incremental win and is **partially done**:
every priority tier except deep-fused conv rolls, and each one now covers its
non-reduction axes' cross product in a single recipe — but only GEMM and MoE
compress *structurally*, and the mechanisms that would broaden this further
(nested loop-nest rolling, peeling, multi-axis **structural** inference) plus
`static_if` case inference are still open (P2/P3). Arch stays a ~2-3 recipe
family multiplier rather than one artifact per gfx target. The byte-identical
oracle is what keeps the whole effort safe: a roller that cannot prove a pattern
declines and the concrete path still ships, so unfinished phases cost coverage,
never correctness.
