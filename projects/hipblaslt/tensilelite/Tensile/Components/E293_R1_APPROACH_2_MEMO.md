# e293 R1 — Approach 2: Source-Module Index Identity Discriminator

Investigator: Approach 2 designer (Claude Opus 4.7, 2026-05-12).
Branch base: `users/alvasile/validator_long_term_plans` (validation tip
`a6aeafc6dc`). **Investigation only — no production code changes.**
Output is design-memo-only; consumed by a downstream comparison agent.

---

## §1. Mechanism summary

Two physically distinct kernel-writer emissions of
`s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]` (one per
tensor side, from `globalReadIncrementAB` at `KernelWriterAssembly.py:9173/9181`)
collapse to byte-identical canonical render-text. Post-hdem identity is
`(canonical_render, emission_ordinal)` (per
`ScheduleCapture.py:439-502`). `emission_ordinal` is assigned at
`LoopBodyCaptureBuilder.finalize` time by `assign_emission_ordinals`
(`ScheduleCapture.py:729-760`) using a per-`(canonical_render)` counter
walked in `(slot.mfma_index, slot.sequence)` lex sort order. Because
the lex key is **scheduler-assigned** (CMS picks one slot order, default
SIA3 picks another) and there is no data-flow dependency forcing a
specific order between the A-side and B-side cmp emissions, the two
builds disagree on which physical instruction wins ordinal=0 vs
ordinal=1. Cross-build identity matching in `compare_graphs` then
pairs the wrong physical instructions, surfacing 12 false-positive
`OverriddenInputFailure(SCC)` residuals across 3 fixtures (per the
e293 investigation).

---

## §2. Approach 2 design (Q1–Q7 answers)

**Theoretical premise.** The kernel writer assigns each emission to a
deterministic source `Module` (e.g. `globalReadIncACode` named
`globalReadIncrementA` at `KernelWriterAssembly.py:9170`,
`globalReadIncBCode` named `globalReadIncrementB` at `:9178`,
`LRCodeAAllIters[u]`, `PackCodeAAllIters[u]`, etc.). `build_idmap`
(`ScheduleCapture.py:972-1035`) records `{category: source_module}`
and is invoked with the SAME source modules on both real builds
(`KernelWriter.py:4591-4607` for non-CMS, `:4671-4687` for CMS). If
we can derive a stable, rocisa-shaped, cross-build-deterministic
identifier per source module (NOT a CMS-shaped category string), we
can put that into identity and break the A-vs-B aliasing without
re-leaking CMS-shape into identity.

### Q1 — How to derive a source-module index

| Candidate | Cross-build stable? | CMS-shaped? | Implementation cost |
|---|---|---|---|
| (a) Python `id()` of source `Module` | NO — different Python process / build = different objects | NO | Trivial but unusable |
| (b) Position in a deterministic kernel-writer-side enumeration of all source modules | YES if the enumeration order is fixed by code (not by data) | NO | Moderate — needs a single enumeration site |
| (c) Content hash of the source module's canonical-rendered children | YES IF modules are byte-identical across builds | NO (hash of rocisa render-text) | Moderate — defines identity stability via hash collision-freedom |
| (d) Source module's rocisa `name` attribute (e.g. `"globalReadIncrementA"`) | YES — name is set in the constructor at `KernelWriterAssembly.py:9159, 9161, 9170, 9178`; same code path runs in both builds | NO — `Module.name` is a rocisa attribute, set by the kernel-writer-side code that constructs the Module, not by the CMS scheduler | Trivial — already exists |

**Recommended: candidate (d)** — use the source `Module.name`
attribute as the source-module identifier. Cross-build stability
follows from: both builds invoke the same kernel-writer code path
(`globalReadIncrementAB`, `_loopBody`, etc.) which constructs Modules
with deterministic `name` arguments. The kernel-writer-side construction
predates CMS scheduling — the CMS scheduler reorders instructions WITHIN
a Module's flatitems sequence but does not rename Modules.

Candidate (b) (positional enumeration) is a viable fallback if module
names ever turn out to be non-unique (e.g., multiple unnamed
intermediate Modules). The enumeration would walk
`build_idmap`'s output in a fixed order (which the helper already
produces deterministically per `_shared_category_assignments`,
`ScheduleCapture.py:915-969`) and assign `0, 1, 2, ...` to each
distinct source module. Both candidates are rocisa-derived (the
Module instance is a rocisa object).

Candidate (c) (content hash) over-reaches: the SAME source module might
hash differently across builds if any child instruction's
operand-name resolution differs (e.g., a vgpr that gets allocated to a
different numeric index across builds). Skip.

**Critical caveat.** Both `Module.name` and the positional candidate
collapse on UNNAMED Modules (some intermediate Modules in
`globalReadIncrement` are unnamed). A robust scheme uses a
**path tuple** of nested Module names from the closest *named ancestor*
of each leaf instruction, walking the tree the kernel writer constructs
via `Module.add(...)`. For the e293 emissions, the relevant named
ancestors are `globalReadIncrementA` and `globalReadIncrementB` — both
named, both deterministic. Use Module.name if the leaf has a named
ancestor; fall back to the positional enumeration over `build_idmap`'s
deterministic emission order otherwise.

### Q2 — Where to put it in identity

Two shapes considered:

**(a)** `(canonical_render, source_module_index, emission_ordinal)` —
add a third slot. emission_ordinal becomes per-`(canonical_render,
source_module_index)` instead of per-`(canonical_render)`. This means:
the A-side cmp and the B-side cmp each get `source_module_index =
"globalReadIncrementA"` resp. `"globalReadIncrementB"`, and each
counter restarts at 0 within its source module. Both builds derive
identical ordinals because each side's source module emits exactly one
cmp per body.

**(b)** `(canonical_render, source_module_index)` — drop
emission_ordinal entirely; rely on source_module_index for
disambiguation.

**Recommended: (a)**. emission_ordinal is still load-bearing for the
case where a single source module legitimately emits the same
canonical render multiple times (e.g., LRA emissions for multiple
inner-unroll indices land in `LRCodeAAllIters[u]` and may share render
text within that one source). Without emission_ordinal, those legitimate
within-source duplicates re-collide. (a) preserves the post-hdem
guarantee for within-source disambiguation while adding cross-source
discrimination on top.

Shape (b) breaks the within-source disambiguation that
`EMISSION_ORDINAL_DESIGN.md §3.2-3.3` makes load-bearing for the
pack-MFMA case (two `v_mfma_*` emissions of the same render in one
source). Reject.

### Q3 — Pipeline impact

`assign_emission_ordinals` runs at `LoopBodyCaptureBuilder.finalize`
(`ScheduleCapture.py:837-884`). Today, the source module is NOT
attached to the `TaggedInstruction` — only `category` (CMS-shaped
string) is. The information flow currently is:

1. Kernel writer constructs `globalReadIncACode = imod.add(Module("globalReadIncrementA"))`
   and `globalReadIncBCode = imod.add(Module("globalReadIncrementB"))`.
2. `build_idmap` records `{'GRIncA': globalReadIncACode, 'GRIncB': globalReadIncBCode, ...}`.
3. `invert_idmap_to_id_to_category` produces `{id(item): category}` where
   category is the CMS string.
4. `_captureSubIterToBuilder` (`KernelWriter.py:2685-2730`) walks
   `iterCode.flatitems()`, looks up `category = id_to_category.get(id(item))`,
   and `capture.append(inst=item, category=category, ...)`.

The source module identity is **lost at step 3** — `invert_idmap`
flattens the per-Module tree to `id(item) → category_string`. To
preserve source-module info, two changes are needed:

(i) Add a parallel inverter `invert_idmap_to_id_to_source_module(idmap)`
that returns `{id(item): source_module_name_or_path}`. This walks the
same tree as `invert_idmap_to_id_to_category` but records the rocisa
`Module` reference (or its name path) instead of the CMS category.

(ii) Extend `LoopBodyCaptureBuilder.append` to accept a
`source_module_id` kwarg, and store it as a new
`TaggedInstruction.source_module_id` field
(`ScheduleCapture.py:388-424`).

(iii) Update `assign_emission_ordinals` to key the counter on
`(canonical_render, source_module_id)` instead of `(canonical_render,)`.

(iv) Update `TaggedInstruction.identity_for`
(`ScheduleCapture.py:439-502`) to return the new tuple shape.

Both real-build paths and the CMS macro expander (`expand_cms_macro`
at `ScheduleCapture.py:2297-2414`) must thread the new
`source_module_id`. The macro expander uses `tag_by_origin_id` for the
category lookup; a parallel `source_module_by_origin_id` map (built
the same way from `build_idmap`'s output) handles source modules
symmetrically.

### Q4 — Cross-build stability proof

**Claim.** For any leaf instruction `inst` emitted in body `B`, the
`source_module_id(inst)` derived at capture time is identical in both
real builds (CMS-emit and non-CMS-emit) of the same YAML.

**Proof sketch.**
1. Both real builds invoke the same kernel-writer code path —
   specifically `KernelWriter._loopBody`, `globalReadIncrementAB`,
   `localReadDo`, `localWriteDo`, `packCode...` — to PRODUCE the
   source modules (`globalReadIncACode`, `LRCodeAAllIters[u]`, etc.).
   Verified by reading `KernelWriter.py:4585-4713`: both branches call
   `build_idmap(num_loop_iter, LRCodeA=LRCodeAAllIters, ...)` with the
   SAME source-module references. The kernel-writer-side construction
   does NOT depend on `UseCustomMainLoopSchedule`.
2. Each Module's `name` is set in the kernel-writer's
   `Module("globalReadIncrementA")` constructor call at
   `KernelWriterAssembly.py:9170` (and similar). The constructor call
   is unconditionally executed in both builds.
3. `build_idmap`'s output dict ordering is deterministic
   (`_shared_category_assignments` yields keys in fixed code order,
   per `ScheduleCapture.py:915-969`). The positional fallback (Q1
   candidate (b)) is therefore deterministic.
4. The post-step (`invert_idmap_to_id_to_source_module`) produces
   `{id(item): source_module_name}` deterministically given the same
   input. Since the input modules are byte-identical (the SAME Python
   objects flow through both builds via `KernelWriter.py:4591-4607`
   and `:4671-4687`), the inverted map's contents are identical.
5. `_captureSubIterToBuilder` and `expand_cms_macro` look up
   `source_module_id` by `id(item)`. The looked-up values are identical
   across builds because the upstream maps are identical.

**Therefore:** for any leaf `inst` that appears in BOTH builds (which
is the precondition for cross-build identity matching at all), the
`source_module_id(inst)` is the same. The new identity tuple
`(canonical_render, source_module_id, emission_ordinal)` is therefore
cross-build-stable when the kernel writer's source-module construction
is deterministic — which is a precondition that already holds for
every other identity-based comparison the validator performs.

**Assumption to validate empirically.** The kernel writer never
constructs DIFFERENT named source modules in the two builds for the
same leaf. Spot-check via `git grep "Module(\"" Tensile/KernelWriter*.py`
to confirm Module construction is unconditional / not gated on
`UseCustomMainLoopSchedule`. From the readings done here, no such
gating was observed in `globalReadIncrement`, `globalReadIncrementAB`,
or `_loopBody`'s `build_idmap` invocations — but a comprehensive
grep before implementation is appropriate.

### Q5 — Coverage

**SUB-CONDITION Y SHARPENING (added 2026-05-13 per reviewer caveat
and Approach 2 vs 3 comparison memo §3 Axis 1).**

The cross-build stability proof in §Q4 (and the coverage claims
below) apply ONLY to source modules constructed via SHARED
kernel-writer code paths — i.e., code paths that execute
unconditionally with respect to `UseCustomMainLoopSchedule`. The
`globalReadIncrementA` / `globalReadIncrementB` Modules at
`KernelWriterAssembly.py:9170/9178` are constructed inside
`globalReadIncrementAB`, which is invoked unconditionally from
`_loopBody`'s shared prelude. Their source-module identifiers are
therefore cross-build-stable.

The `LR*AllIters` / `Pack*AllIters` source modules
(`LRCodeAAllIters`, `LRCodeBAllIters`, `LRCodeMAllIters`,
`PackCodeAAllIters`, `PackCodeBAllIters`, `PackCodeMAllIters`,
`LRMXSACodeAllIters`, `LRMXSBCodeAllIters`, `LRMetadataCodeAllIters`)
are populated through `_loopBody`'s `UseCustomMainLoopSchedule`
branches. Those branches are ASYMMETRIC — the CMS branch and the
non-CMS branch take different code paths through
`_localReadIncCode...`, `_lwaTileAssignmentA/B`, and the iteration-
unroll dispatch. The same physical emission may be assigned to
different intermediate Module containers (or constructed in different
container nesting depth) across the two builds. Approach 2's
source-module discrimination on these emissions is NOT guaranteed to
be cross-build-stable.

**Implication for in-scope vs out-of-scope cases:**

- **In scope (covered):** e293 SCC `s_cmp_eq_u32 LoopCounterL,
  StaggerUIter` (GRIncA vs GRIncB); p39d's GR-side `s_add_u32 m0,
  m0, 4224` family (GRA vs GRB) IF its source modules are also from
  the shared `globalReadIncrement*` path; future m0-write or
  singleton-resource collisions originating in GRA/GRB or other
  unconditional kernel-writer paths.

- **Out of scope (NOT covered by Approach 2):** any cross-build
  identity divergence whose source modules are LR*AllIters or
  Pack*AllIters (e.g., LRA / PackA cases). These cases (if/when they
  emerge — e.g., a future per-iter LR same-render twin, or a
  Pack-MFMA cross-side aliasing) must be addressed by a separate
  mechanism (likely defense-in-depth via Approach 3's comparator
  gate, or further structural work in the LR/Pack capture path to
  make those Modules' construction symmetric across the two
  `UseCustomMainLoopSchedule` branches).

Approach 2 fixes (within the in-scope envelope above):
- **e293 SCC same-render-different-side case.** YES. The A-side cmp
  and B-side cmp now have distinct `source_module_id`
  (`"globalReadIncrementA"` vs `"globalReadIncrementB"`), giving them
  distinct identities even when their canonical render and ordinal-N
  match.
- **Other potential cross-build z012-class issues confined to GR-side
  shared paths.** Any case where the cross-build collision is "same
  canonical_render, different source module" AND both source modules
  are from the SHARED kernel-writer code path is fixed. This covers
  the e293 SCC case (`s_cmp_eq_u32 LoopCounterL, StaggerUIter` from
  GRIncA vs GRIncB) and the GR-side m0-write twins (`s_add_u32 m0,
  m0, 4224` from GRA vs GRB documented at `EMISSION_ORDINAL_DESIGN.md
  §1.2`) when those originate in `globalReadIncrement*`. The LCC twin
  (`s_sub_u32 LoopCounterL, ...` from `loopCounterCode`) is in scope
  IFF `loopCounterCode` is constructed unconditionally; otherwise out
  of scope.
- **Wait-coverage residuals.** No effect. `validate_edge_wait_coverage`
  operates on edge resources, not identities. (3ija reports zero
  wait-coverage residuals across all fixtures regardless.)

Approach 2 does NOT fix:
- **LR/Pack source-module-asymmetry cases** (per sub-condition Y
  above). Cases involving `LRCodeAAllIters` / `PackCodeAAllIters` /
  etc. as the source-module discriminator are NOT closed by Approach
  2 because the construction of those Modules is asymmetric across
  CMS vs non-CMS builds. e293 + p39d (GR-side) + future m0 cases are
  in scope; LRA/PackA cases are NOT.
- **(b) Q2-expected per-tile schedule mutations.** Cases like
  `UsePLRPack` flip (Pack code moving from per-iter to prologue) are
  cross-BODY divergence, not cross-side intra-body collision.
  Body-label-tolerance work (`oram.1`) handles that orthogonally.
- **Cross-build identity divergence where the same source module
  legitimately emits the same render multiple times in different
  positions.** These collide on emission_ordinal alone today — and
  Approach 2 inherits that risk because `emission_ordinal` becomes
  per-`(canonical_render, source_module_id)` rather than per-`(canonical_render)`,
  meaning within-source ordering must be stable across builds. This
  IS the existing post-hdem assumption restricted to a smaller scope;
  it is strictly weaker (easier to satisfy) than the current
  assumption, so Approach 2 is a strict improvement on this axis too.
- **CaptureUnknownInstructionError / structural-divergence residuals
  surfaced by 3ija §3.C-F.** These are pre-comparator failures, not
  identity-collision symptoms.

### Q6 — Test impact

Identity-shape pinning tests that hard-code the 2-tuple shape break.
Current count of files referencing identity (`grep -rn "identity_for\|\.identity\["`):

```
test_pre_post_loop_label_disambiguation.py
test_dataflow_graph_comparison.py
test_ScheduleCapture.py
test_dataflow_graph_hdem.py
test_dataflow_graph_register_gaps.py
test_dataflow_graph_emission_ordinal.py
_dump_carveout_assembly.py  (tooling, not test)
```

Six test files. Comparable in scope to 4up4's migration (which also
touched ~6 identity-shape tests per `EMISSION_ORDINAL_DESIGN.md` §4.5).
Migration cost: each test that asserts a specific identity tuple shape
needs the new third slot accommodated; tests that index `identity[0]`
or `identity[1]` for the render text need to use named accessors or
shift indices. A NamedTuple-shaped identity (instead of a bare tuple)
would prevent index-shift fragility for future migrations — flagged
as an opportunistic refactor.

The conformance test fixture `dataflow_fixtures.make_capture` (used
across tests) needs to thread `source_module_id` through; default to a
sentinel (`None` or `"<synthetic>"`) for tests that construct
TaggedInstructions without a real source module.

### Q7 — xqj3 alignment check

xqj3 is "validator categorization layer must move from CMS-shaped
strings to rocisa-derived properties." Approach 2's
`source_module_id`:

- **Is rocisa-derived.** A `Module` is a rocisa object
  (`from rocisa.code import Module` per `ScheduleCapture.py:1092`).
  `Module.name` is a rocisa attribute set by the kernel writer's
  rocisa-side code, not by the CMS scheduler. The fact that
  `build_idmap` happens to ALSO key on a CMS-shaped string
  (`'GRIncA'`) is incidental — Approach 2 ignores that string and uses
  the underlying Module's name (or its position in the deterministic
  enumeration), bypassing the CMS-shape entirely.

- **Does not re-introduce 4up4/hdem/hdu1 coupling.** 4up4 dropped
  loop_index from identity (Approach A); hdem dropped class_tag from
  identity. Both changes removed CMS-shaped slots. Approach 2 ADDS a
  rocisa-shaped slot — not a CMS-shaped one. The xqj3 epic's framing
  is "identity must ground in rocisa-derived properties"; the source
  Module is exactly such a property.

- **Open question on naming convention.** If the source-module
  identifier is derived from `Module.name` (which the kernel writer
  CHOOSES to set), there's a residual coupling: a kernel-writer
  refactor that renames `globalReadIncrementA` → `gri_A` would change
  the identity stability across versions. This is no different from
  any other naming-stable identity convention in the codebase
  (`canonical_render` is render-text-stable; `source_module_id` becomes
  Module-name-stable). If the bare `Module.name` is judged too
  fragile, fall back to the positional enumeration over
  `build_idmap`'s deterministic order — that ties identity to the
  enumeration's structural shape rather than to chosen string names.

---

## §3. Implementation sketch

### File-by-file change list

1. **`Tensile/Components/ScheduleCapture.py`**
   - `~750`: `assign_emission_ordinals` — change counter key from
     `render` to `(render, source_module_id)`.
   - `~388-424`: `TaggedInstruction` — add
     `source_module_id: Optional[str] = None` field.
   - `~439-502`: `identity_for` — return
     `(canonical_render, source_module_id, emission_ordinal)`.
   - `~821-835`: `LoopBodyCaptureBuilder.append` — accept and store
     `source_module_id` kwarg.
   - `~1038-1063`: new `invert_idmap_to_id_to_source_module(idmap)`
     helper, walking the same tree as `invert_idmap_to_id_to_category`
     but emitting `{id(item): source_module_id}` where
     `source_module_id` is the closest-named-ancestor's name from the
     idmap value (or, for fallback, a positional index assigned in
     idmap-iteration order).
   - `~2297-2414`: `expand_cms_macro` — accept and thread
     `source_module_by_origin_id` kwarg; pass to `builder.append`.

2. **`Tensile/KernelWriter.py`**
   - `~2643-2730`: `_captureSubIterToBuilder` — accept and thread
     `source_module_id` kwarg in addition to `id_to_category`; pass to
     `capture.append`.
   - `~4585-4713` (both real-build branches) and `~3699-3731` (NLL/NGL
     path): construct `id_to_source_module` alongside `id_to_category`
     via `invert_idmap_to_id_to_source_module(capture_idmap)`; pass
     through.

3. **`Tensile/Components/CustomSchedule/dispatch.py`**
   - `~136-149` and `~192-194`: build
     `source_module_by_origin_id` alongside `tag_by_origin_id`; pass
     to `expand_cms_macro` invocations.

4. **`Tensile/Components/CMSValidator.py`**
   - `~1962-2023`: no logic change, but `nodes_by_identity` keys are
     now 3-tuples instead of 2-tuples — verify no consumers index
     `identity[N]` positionally.
   - `~3349-3354` and `~1238-1241`: `edge_keys` — already migrated to
     consume `producer.identity` / `consumer.identity` opaquely under
     hdem (Approach E). No change.

5. **Test migrations** (6 files): rewrite identity-shape pins from
   `(canonical_render, ordinal)` to `(canonical_render,
   source_module_id, ordinal)`. Use named accessors via NamedTuple if
   possible.

### Pseudocode for the non-trivial logic

```python
# ScheduleCapture.py — new helper
def invert_idmap_to_id_to_source_module(idmap):
    """{id(item): source_module_identifier} from {category: module}.

    The identifier is the source Module's `name` attribute (rocisa-
    side, set by the kernel-writer's Module() constructor calls). For
    leaves whose immediate parent Module is unnamed, falls back to the
    nearest named ancestor — walking up from the leaf via the captured
    parent chain, stopping at the first non-empty `name`.

    If even the top-level Module has an empty name, falls back to a
    positional index derived from idmap iteration order. (Both builds
    iterate idmap in the same code-determined order, so the positional
    fallback is also cross-build-stable.)
    """
    out = {}
    for pos, (cat, mod) in enumerate(idmap.items()):
        if mod is None:
            continue
        items = mod.flatitems() if hasattr(mod, 'flatitems') else mod
        # Use mod.name if non-empty; otherwise use positional index.
        src_id = mod.name if (hasattr(mod, 'name') and mod.name) else f"#idmap_pos_{pos}"
        for item in items:
            out[id(item)] = src_id
    return out

# ScheduleCapture.py — modified ordinal assignment
def assign_emission_ordinals(instructions: list) -> None:
    sorted_tis = sorted(
        instructions,
        key=lambda ti: (ti.slot.mfma_index, ti.slot.sequence),
    )
    counter: Dict[Tuple[str, Optional[str]], int] = {}
    for ti in sorted_tis:
        render = WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst)
        key = (render, ti.source_module_id)
        ord_idx = counter.get(key, 0)
        ti.emission_ordinal = ord_idx
        counter[key] = ord_idx + 1

# ScheduleCapture.py — new identity tuple
def identity_for(self, body_label: str) -> tuple:
    inst = self.wrapped.rocisa_inst
    return (WrappedInstruction.canonical_str(inst),
            self.source_module_id,
            self.emission_ordinal)
```

### Estimated LOC delta (sizing input only)

- New `invert_idmap_to_id_to_source_module`: ~25 LOC (mirrors existing
  `invert_idmap_to_id_to_category`).
- Plumbing through `_captureSubIterToBuilder` /
  `LoopBodyCaptureBuilder.append` / `expand_cms_macro`: ~30 LOC across
  call sites (one new kwarg threaded through three layers).
- `TaggedInstruction.source_module_id` field + identity_for change:
  ~10 LOC.
- Counter-keying change: ~3 LOC.
- Test migrations: ~50–100 LOC across 6 files (most are signature-
  shape line edits).
- New regression tests (cross-build determinism for source_module_id;
  e293 collision regression): ~80 LOC.
- **Net:** ~200–250 LOC change set, comparable to the 4up4 / hdem
  migrations.

---

## §4. Trade-offs

### Preserved
- **Body-blindness from hdem (Approach A).** loop_index stays out of
  identity. Cross-body PLR-pack movement remains absorbed.
- **Within-source ordinal disambiguation.** Two emissions of the same
  render from the SAME source module still get distinct ordinals, so
  pack-MFMA / main-loop-MFMA same-render disambiguation per
  `EMISSION_ORDINAL_DESIGN.md §3.2` continues to work.
- **rocisa-derived identity layer.** No CMS-shaped string re-enters
  identity. xqj3-clean.
- **edge_keys body-blindness from hdem (Approach E).** edge_keys
  consume identity opaquely; the new third slot rides through
  unchanged.

### Broken / migrated
- **Identity-shape pinning tests.** Six test files need shape
  migration. Migration cost is comparable to 4up4/hdem precedent.
- **Synthetic test fixtures** (`make_capture` etc.) need a default
  `source_module_id` (e.g. `None` or `"<synthetic>"`); fixtures that
  test cross-source disambiguation must thread real values.

### Risk surfaces

1. **Module-name uniqueness assumption.** If two distinct source
   modules in `build_idmap`'s output happen to share the same `name`
   string (shouldn't happen — `globalReadIncrementA` and
   `globalReadIncrementB` are unique — but worth verifying), they
   re-collide. Mitigation: positional-index fallback at the
   `invert_idmap_to_id_to_source_module` layer; key the source-module
   ID on `(name, idmap_position)` rather than just `name` for
   guaranteed uniqueness.

2. **Unnamed intermediate Modules.** Some `globalReadIncrement` paths
   add unnamed sub-Modules. The closest-named-ancestor walk must
   correctly attribute leaves to `globalReadIncrementA` /
   `globalReadIncrementB` even when intermediate wrappers are
   unnamed. Verified for `globalReadIncrementAB`'s top-level structure
   (`KernelWriterAssembly.py:9159, 9161, 9170, 9178` — all top-level
   side modules are named). Need to verify deeper for sparse-MX
   paths.

3. **Future kernel-writer refactor.** A rename of
   `globalReadIncrementA` → `gri_A` (or any source-module name change)
   would invalidate identity stability across Tensilelite versions.
   Identity is intra-version (one CMS build vs one non-CMS build of
   the same source tree), so this is acceptable — but worth
   documenting as a contract.

4. **Determinism of `build_idmap` iteration order.** Python dict
   preserves insertion order (3.7+); `_shared_category_assignments`
   yields in fixed code order. Verified: deterministic. The
   positional-fallback strategy is therefore safe.

5. **Counter-keying scope shrinkage.** Moving from
   per-`(canonical_render)` to per-`(canonical_render,
   source_module_id)` ordinal scope shrinks the within-counter pool,
   which strictly weakens the within-counter ordering assumption — no
   new hazard introduced. The post-hdem assumption (within-counter
   order is stable across builds) was already required and holds for
   the same kernel-writer-source reasoning.

6. **Failure-mode under invariant breakage.** If a future build path
   conditionally emits an extra instance from one source module on one
   build (a real divergence), the per-source ordinal assignment makes
   the divergence VISIBLE as identity-set mismatch — same desired
   behavior as today, just at finer granularity. Not a regression.

---

## §5. Open questions for the user

1. **Should `source_module_id` derive from `Module.name` directly, or
   from the positional enumeration over `build_idmap`'s output?** The
   former is more readable in failure messages
   (`("s_cmp_eq_u32 ...", "globalReadIncrementA", 0)`); the latter is
   more refactor-stable but uses opaque integers. The hybrid
   `(name, position)` form sidesteps both concerns at the cost of
   slightly noisier identity tuples.

2. **Is a NamedTuple identity preferable to a positional 3-tuple?**
   `EMISSION_ORDINAL_DESIGN.md` left identity as a bare tuple. Adding
   a third slot is a natural moment to migrate to a NamedTuple
   (`Identity(canonical_render=..., source_module_id=...,
   emission_ordinal=...)`) for consumer ergonomics and migration
   resilience. Out of scope of e293-fix as such, but the slot
   addition is the natural place.

3. **Should the source-module slot be conditionally populated or
   always required?** Synthetic test fixtures may not have a real
   source module. A `None` sentinel works mechanically (counter just
   keys on `(render, None)` for synthetics), but it weakens the
   invariant that "every production identity carries a source-module
   slot." Acceptable trade-off; flag if the user prefers stricter
   typing.

4. **Sparse-MX `LRMXSA` / `LRMXSB` / `LRMetadata` source modules.**
   Verified at the schema level
   (`ScheduleCapture.py:919-967`) but not exercised in the e293
   fixture surface. If sparse-MX paths surface a different module-
   nesting structure (unnamed intermediate Modules), the closest-
   named-ancestor walk needs sparse-MX coverage in the test suite.

5. **Migration sequencing relative to ongoing 71hw/oram.1 work.**
   Approach 2 lands as a single PR per the same design philosophy as
   hdem. Should it land BEFORE or ALONGSIDE oram.1 (body-label
   tolerance)? The two are layer-orthogonal (identity composition vs
   identity-divergence interpretation), so independent. Not a blocker.

---

## §6. Citations (verified by reading file:line)

- `Tensile/Components/E293_SCC_CARRY_CHAIN_INVESTIGATION.md` §Verdict,
  §Q1, §Q4 — mechanism trace.
- `Tensile/Components/EMISSION_ORDINAL_DESIGN.md` §2.6, §3, §4 —
  determinism guarantee that 4up4 made; migration of class_tag
  consumers; the post-hdem identity shape this design extends.
- `Tensile/Components/HDEM_IMPLEMENTATION.md` "Approach A" / "Approach
  E" — current 2-tuple identity shape and body-blind edge_keys.
- `Tensile/Components/SVB1_IMPLEMENTATION.md` — `_shared_category_assignments`
  deterministic emission order.
- `Tensile/Components/ScheduleCapture.py:388-424` — `TaggedInstruction`
  dataclass.
- `Tensile/Components/ScheduleCapture.py:439-502` — `identity_for`
  current 2-tuple.
- `Tensile/Components/ScheduleCapture.py:729-760` —
  `assign_emission_ordinals` per-render counter.
- `Tensile/Components/ScheduleCapture.py:817-884` —
  `LoopBodyCaptureBuilder.append/finalize`.
- `Tensile/Components/ScheduleCapture.py:915-969` —
  `_shared_category_assignments` deterministic yield order.
- `Tensile/Components/ScheduleCapture.py:972-1063` — `build_idmap` and
  `invert_idmap_to_id_to_category`.
- `Tensile/Components/ScheduleCapture.py:1105-1207` —
  `build_id_to_category_per_iter`.
- `Tensile/Components/ScheduleCapture.py:2297-2414` —
  `expand_cms_macro` (CMS-side capture funnel; uses `tag_by_origin_id`).
- `Tensile/KernelWriter.py:2643-2730` — `_captureSubIterToBuilder`
  (default-side capture funnel; consumes `id_to_category`).
- `Tensile/KernelWriter.py:4585-4713` — both real-build branches
  invoke `build_idmap` with the SAME source modules.
- `Tensile/KernelWriterAssembly.py:9149-9186` —
  `globalReadIncrementAB` constructs Modules with deterministic
  names (`"globalReadIncrementA"`, `"globalReadIncrementB"`).
- `Tensile/Components/CustomSchedule/dispatch.py:136-194` —
  `tag_by_origin_id` construction; CMS-side category routing.
- `Tensile/Components/CMSValidator.py:1571, 1962-2023` —
  `nodes_by_identity` accumulation; `_NO_DATAFLOW_IDENTITY_CATEGORIES`
  filter (rocisa-derived, identity-independent).
- `Tensile/Components/3IJA_RESIDUAL_TRIAGE.md` — the 12-failure
  residual surface this design fixes.
