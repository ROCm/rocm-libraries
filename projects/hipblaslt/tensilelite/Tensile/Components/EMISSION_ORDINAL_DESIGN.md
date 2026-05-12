# Per-Emission Ordinal Identity: Architectural Design Memo

Investigator: emission-ordinal-investigator (Claude Opus 4.7, 2026-05-12).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/emission-ordinal-investigation/`.
Umbrella: `rocm-libraries-xqj3` (P0 epic — categorization → rocisa-derived).
Sub-bead: filed under xqj3 (see §7).

This memo answers the user's two coupled questions on `xqj3`:
1. Replace CMS-shaped categorization in `identity_for`.
2. Decide whether `class_tag` belongs in identity at all.

It supersedes Approach 11 on `rocm-libraries-on0t` (selective un-collapse
in `class_tag_for_category`) — the user has rejected CMS-coupled fixes.

---

## §1. Mechanism — what's broken today (verified end-to-end)

### §1.1 The identity tuple

`TaggedInstruction.identity_for` (`Tensile/Components/ScheduleCapture.py:507-536`)
returns

```python
(class_tag, loop_index, canonical_render)
```

where `class_tag = WrappedInstruction.class_tag_for_category(self.category, inst)`
(line 535) consults a CMS-shaped collapse table at
`ScheduleCapture.py:374-426`. That table maps:

| Category prefix or value (CMS-derived) | class_tag bucket |
|---|---|
| `LRA*`, `LRB*`, `LRMXSA*`, `LRMXSB*`, `LRMetadata*` | `'LR'` |
| `LRS*` | `'LRS'` |
| `LWS*` | `'LWS'` |
| `LW*` | `'LW'` |
| `GRInc*` | `'GRINC'` |
| `GR*` | `'GR'` |
| `Pack*` | `'PACK'` |
| `LCC` | `'LCC'` |
| `SYNC` | isinstance fallback (SWAIT or SBARRIER) |
| `SNOP` / `SSETPRIO` / `BARRIER` / `MFMA` | matching tag |
| `UNKNOWN` / `None` | `class_tag(inst)` (isinstance fallback) |

The buckets are CMS-shaped because their categories are CMS-shaped:
`build_idmap` (`ScheduleCapture.py:908-947`) is the schema, written
when CMS was the only consumer.

### §1.2 The collision pattern (the on0t failure mode)

`build_dataflow_graph` Phase 1 (`CMSValidator.py:1789`) does
last-writer-wins: `nodes_by_identity[node.identity] = node`. Two
physically distinct kernel-writer emissions sharing the same identity
collapse to one node; the surviving stream-position is unstable across
captures whose interleaving differs.

Verified collisions on `example.yaml` against
`_get_schedule_160x128x64_TF32` (8 false `OrderInvertedFailure`s,
recorded in `EXAMPLE_YAML_DEFECT_INVESTIGATION.md §1-2`):

| Body × bodies | Resource | Render-string | Source |
|---|---|---|---|
| ML-1 / ML / NGL (×4) | `regType=scc` | `s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]` | `KernelWriterAssembly.py:9051-9052` (`globalReadIncrement`) — called once per tensor side from `globalReadIncrementAB` (`:9173, :9181`); A-side and B-side carry IDENTICAL canonical render (no operand discriminates A from B). Tagged `GRIncA` and `GRIncB` respectively → both collapse to `'GRINC'`. |
| ML-1 / ML (×3) | `regType=m` | `s_add_u32 m0, m0, 4224` | `globalReadIncrementAB` per-side GR group. Tagged `GRA` and `GRB` → both collapse to `'GR'`. |

The render-strings are byte-identical because neither A-side nor B-side
emission carries a side-discriminating operand; the discriminator lives
only in the categorization tag, which is then collapsed.

### §1.3 Per-tensor / Pack collisions absorbed by collapse but pending

`Z012_RESIDUAL_INVESTIGATION.md §1` enumerates the LRSwap case:
default-side tags pointer items bare `'LRS'`/`'LWS'`
(`KernelWriter.py:976,979`); CMS-side tags per-tensor
`LRSA/LRSB/LWSA/LWSB`. Both collapse to `'LRS'`/`'LWS'` so the
divergence is hidden by the identity layer today — but if we extend
identity in any direction (per-side suffix, per-emission ordinal,
anything beyond render-string) the divergence comes back.

`EXAMPLE_YAML_DEFECT_INVESTIGATION.md §6`: 186 Pack instructions tagged
`PackA0`/`PackB0` on default but `PackA1`/`PackB1` on CMS — same physical
instructions. Currently absorbed (both collapse to `'PACK'`); same
re-emergence risk under any extension.

### §1.4 What's load-bearing about class_tag in identity today

Three downstream consumers depend on `identity[0]`:

1. **`_NO_DATAFLOW_IDENTITY_CATEGORIES` exclusion** at `CMSValidator.py:1788`:
   ```python
   if _category(inst) not in _NO_DATAFLOW_IDENTITY_CATEGORIES:
       nodes_by_identity[node.identity] = node
   ```
   This filters BEFORE adding to the cross-graph dict; it consults
   `_category(inst)` (the rocisa class-name registry, NOT
   `node.identity[0]`). So this consumer is not actually identity-coupled
   — flagging it explicitly because the on0t bead description was
   imprecise about which side does the filtering. The filter set is
   `{SWAIT, SBARRIER, SNOP, SSETPRIO}`; class-name dispatch via
   `_CLASS_NAME_TO_CATEGORY` resolves them without identity.

2. **`_DATA_FLOW_KINDS` filter** at `CMSValidator.py:3301-3304`
   (entry-time identity-set parity check):
   ```python
   _DATA_FLOW_KINDS = ("LR", "LW", "GR", "MFMA")
   def _data_flow_ids(graph):
       return {k for k in graph.nodes.keys() if k and k[0] in _DATA_FLOW_KINDS}
   ```
   Restricts the cross-graph identity-coverage check to data-flow
   identities. Reads `identity[0]`. Is identity-coupled.

3. **`DataflowGraph.edge_keys` role positions** at `CMSValidator.py:1238-1241`:
   ```python
   return {(e.producer.identity[0], e.producer.position, e.src_operand_slot,
            e.consumer.identity[0], e.consumer.position, e.sink_operand_slot,
            e.edge_kind, e.intra_operand_byte_offset)
           for e in self.edges}
   ```
   Same shape replicated at `:3349-3354` for ref-edge lookup. Reads
   `identity[0]` as the `src_role`/`sink_role` slot in the edge identity
   tuple. Is identity-coupled.

The `_summary_by_class` diagnostic helper at `:3314-3319` also reads
`ident[0]`, but it's only for error-message rendering — not load-bearing.

`cms_node_label` (`:3202-3252`) and `CmsLabelRenderer.render`
(`cms_to_timeline.py:112-120`) both consume `node.category` (the raw
string), NOT `identity[0]`. Failure rendering is independent of
identity shape.

### §1.5 Why Approach 11 perpetuates CMS coupling

The on0t bead's "Approach 11" recommendation un-collapses GRInc/GR but
keeps the CMS category strings flowing through identity. Even after
the un-collapse, identity buckets remain CMS-derived (`'GRINCA'`,
`'GRINCB'`, `'GRA'`, `'GRB'`). Per the user's xqj3 comment 76: this
inheritance bakes CMS coupling into identity itself, regardless of
which derivation path produces the bucket name. The cheap mechanical
fix is right; the architectural direction is wrong.

---

## §2. Design — per-emission ordinal identity

### §2.1 Identity tuple shape

Proposed:

```python
identity = (loop_index, canonical_render, emission_ordinal)
```

`class_tag` is dropped entirely. See §3 for the defense.

`emission_ordinal` is a per-`(body_label, canonical_render)` monotonic
counter computed at capture time. Two physically distinct emissions
of the same canonical render-text in the same body get distinct
ordinals (0, 1, 2, ...). Two physical emissions in different bodies
get ordinal 0 each (the body_label, encoded via `loop_index`,
disambiguates them).

### §2.2 Where the counter lives — at capture time, on the builder

The counter is a per-`(body, canonical_render)` dict on the
`LoopBodyCaptureBuilder` (`ScheduleCapture.py:796-877`). It increments
inside `LoopBodyCaptureBuilder.append` (`:821-835`). The append site is
the single funnel through which both default-side capture
(`KernelWriter._captureSubIterToBuilder`, `:2601-2688`) and CMS-side
capture (`expand_cms_macro`, `:2186-2303`) feed instructions into the
capture stream — making it the only point where the ordinal is assigned
per emission.

`LoopBodyCapture` (`ScheduleCapture.py:540-559`) carries no global
state; the builder owns lifecycle. The ordinal assignment is local to
one builder's lifetime, which is one body's worth of capture. That's
correct: the ordinal is body-scoped, not kernel-scoped.

The counter key is `canonical_render` (already computed today via
`WrappedInstruction.canonical_str`). NO category, NO class_tag — only
the rocisa-rendered text.

### §2.3 Why the counter must NOT be keyed on `(body, class_tag, canonical_render)`

A naive variant: key on `(body, class_tag, canonical_render)`. Rejected,
because class_tag is itself CMS-derived. Keying the disambiguator on
the same CMS-derived bucket we're trying to drop reintroduces the
coupling at the counter level. Same render-text under different CMS
categories would get separate ordinal sequences — defeating the goal of
identity that doesn't depend on category names.

### §2.4 Why the counter must NOT be keyed on `(body, canonical_render, category)`

Same objection, harder. Even if we run categories through the future
rocisa-derived classifier first (§4 below), category is still a
scheduler-role discriminator (LR vs PACK vs MFMA). Two emissions of the
same render-text under different roles would NEVER collide as ordinals
under this key — but the whole point of the ordinal is to disambiguate
emissions whose render-text is identical. If their render-text is
identical and they're in different roles, the role discriminator
suffices and the ordinal is unnecessary; if the roles are identical
(the on0t case: GRIncA-cmp and GRIncB-cmp both `GRINC`), the ordinal
is what distinguishes them. Keying the counter on category buys us
nothing in the only case the ordinal needs to handle.

### §2.5 When the counter increments — at `append` time, not at `_make_node` time

`_make_node` (`CMSValidator.py:1587-1617`) runs at graph-build time,
after capture finalize. By that point the per-body capture order is
fixed (the `LoopBodyCaptureBuilder._instructions` list is the canonical
sequence). Computing the ordinal here is mechanically equivalent to
computing it in append.

Append-time computation is preferred for two reasons:
- The builder already owns mutable per-append state (`_seq_counter` for
  slot sequencing); adding another dict is symmetric.
- The ordinal becomes a property of the `TaggedInstruction` (or its
  `WrappedInstruction`), making it inspectable BEFORE `build_dataflow_graph`
  runs. That matters for tests (assert per-render counts in
  `LoopBodyCapture`) and for any future tooling that operates on captures
  before the graph stage.

### §2.6 Determinism guarantee — the kernel-writer-shared-source axiom

Both real builds (CMS-emit-build and non-CMS-emit-build under
Approach A, `rocm-libraries-71hw`) consume IDENTICAL kernel-writer
source modules: `LRCodeAAllIters[u]`, `PackCodeAAllIters[u]`,
`globalReadA`, `globalReadB`, `globalReadIncACode`, `globalReadIncBCode`,
etc. These are populated by `KernelWriter._loopBody`'s `localReadDo` /
`localWriteDo` / `globalReadIncrementAB` calls BEFORE either capture
path runs.

For the per-emission ordinal to be deterministic across the two
builds, ONE invariant must hold: **for each body and each canonical
render-text, both real builds emit the same number of instances of
that render-text in the same relative order against other emissions
of the same render-text.**

This invariant DOES NOT require both builds to:
- Emit the same ABSOLUTE number of total instructions in the body
  (CMS interleaves, default doesn't).
- Emit the same relative order of DIFFERENT render-texts (CMS reorders
  pack vs MFMA vs LR).
- Emit the same scheduler-control instructions (SWaits, SBarriers,
  SNops can differ — they're excluded from `nodes_by_identity` by
  `_NO_DATAFLOW_IDENTITY_CATEGORIES`).

It DOES require: if default emits the `s_cmp_eq_u32 LoopCounterL,
StaggerUIter` from incCodeA before the one from incCodeB, CMS must do
the same. **And it does.** Both captures consume the SAME
`globalReadIncACode` and `globalReadIncBCode` modules; both walk
`incCodeA.flatitems()` before `incCodeB.flatitems()` because that's
the order `globalReadIncrementAB` (`KernelWriterAssembly.py:9170-9181`)
adds them to the parent module. The CMS macro expander reads the same
module references; SIA3 packs them into per-iter combined modules in
the same order.

### §2.7 Edge case — when one build emits an instance the other doesn't

Concrete examples drawn from `2LZD_INVESTIGATION.md §1`:

- `UsePLRPack` flip moves Pack instructions from per-iter modules
  (default-build) to the prologue (CMS-build). Same `v_cvt_pk_bf16_f32
  v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0], v[vgprValuA_T0_I0+1]` text,
  different body. Under the per-emission ordinal scheme, both ordinal
  sequences start at 0 in their respective bodies — and `loop_index`
  differs — so identity differs. This is correctly handled by `oram.1`
  (`rocm-libraries-oram.1`)'s body-label-tolerance work; it's not the
  per-emission ordinal's job to absorb cross-body movement.

- A schedule that emits a `s_nop` only on the CMS side (scheduler
  insertion). SNop is in `_NO_DATAFLOW_IDENTITY_CATEGORIES` and never
  enters `nodes_by_identity`. The ordinal on each side starts at 0 for
  the next non-control emission; the SNop discrepancy doesn't shift
  ordinals. ✓

- A schedule that emits an extra `v_mfma` on the CMS side because of a
  carve-out. The CMS side has an extra ordinal; the comparison sees
  identity-set divergence (more nodes on one side) and surfaces it via
  `compare_graphs`'s entry-time check at `:3306-3336`. This is the
  desired behavior — emit-count divergence IS a real defect, not
  something to absorb.

The invariant restated: **two builds that consume the same
kernel-writer source-module references will see identical per-render
emission orders, modulo scheduler-control differences excluded from
identity construction.** Per-emission ordinal identity is correct
exactly under this invariant.

### §2.8 What if the invariant breaks in some future build path

Two breakage modes worth pre-empting in the design:

1. **A future kernel-writer change conditionally emits the same
   render-text under one build path and not the other** (e.g., a
   `if UsePLRPack: emit_extra_pack_instance(...)`). Per-emission ordinal
   becomes a divergence detector: the two builds' ordinal-N entries
   no longer correspond to the same logical instruction, and
   `compare_graphs` surfaces it. This is the correct behavior — the
   user explicitly wants such changes to be CAUGHT (Q2 framing in
   `2LZD_INVESTIGATION.md §6.2`).

2. **The kernel-writer call sequence becomes nondeterministic** (e.g.,
   dict iteration order leaks). Already a precondition for ANY
   identity-based comparison; not specific to the ordinal scheme. Mitigation
   is the existing `assign_stream_indices_for_body` lex-sort (which
   re-derives the canonical order from `(slot.mfma_index, slot.sequence)`
   if `_instructions` happen to be out of order). The ordinal can use
   the same canonical lex-sort: assign ordinals after sorting, in the
   sorted order, so `_seq_counter` ordering doesn't itself become
   load-bearing. This is the recommended implementation (§5.1 below).

---

## §3. Should class_tag be in identity? — drop it

### §3.1 The defense

Three downstream consumers read `identity[0]` (per §1.4):
- `_DATA_FLOW_KINDS` filter at `CMSValidator.py:3301-3304`.
- `edge_keys` role positions at `CMSValidator.py:1238-1241`.
- `_summary_by_class` diagnostic at `:3314-3319` (rendering only).

Every single one of them can be rewritten to consult the rocisa
instance directly via `WrappedInstruction.is_lr` / `is_lw` / `is_gr` /
`is_mfma` / `is_swait` / `is_sbarrier` (`ScheduleCapture.py:235-298`),
which dispatch through the centralized `_CLASS_NAME_TO_CATEGORY`
registry (`InstructionCategory.py:111-227`). That registry IS the
rocisa-derived classifier the umbrella xqj3 wants identity to
ground in.

Concretely, `_DATA_FLOW_KINDS` becomes a `WrappedInstruction.is_data_flow`
property (or a new `is_data_flow` predicate on `GraphNode`): true iff
the underlying rocisa instance's category is one of `{LR, LW, GR,
MFMA, MIDDLE_PACK, CVT_PACK}`. The `MIDDLE_PACK` / `CVT_PACK` cases
are a refinement on the current `("LR", "LW", "GR", "MFMA")` tuple —
they're TF32-emulation rocisa instances that participate in the
data-flow set today (because the CMS scheduler categorizes them as
Pack but `class_tag_for_category` doesn't currently put them in
`_DATA_FLOW_KINDS`). The current filter loses them, which is a defect
the rocisa-derived rewrite happens to fix.

Edge-key role positions become per-rocisa-derived discriminators:
`(producer_role, consumer_role)` where `role` is computed from
`producer.rocisa_inst` via `_category(...)`. The result is the same
LR/LW/GR/MFMA/PACK vocabulary as today, but produced from the rocisa
classifier — same shape, no CMS-string indirection.

### §3.2 The disambiguator role

The original argument for keeping class_tag was disambiguation: two
instructions with the same canonical render but different categories
should have different identities. Today this is needed for:
- TF32 pack-MFMAs: real `MFMAInstruction` rocisa objects categorized
  as `PackA*`/`PackB*`. Their class_tag is `'PACK'` (not `'MFMA'`),
  preventing them from masquerading as main-loop MFMAs. (Per the
  `class_tag_for_category` docstring at `ScheduleCapture.py:380-393`.)

Under per-emission ordinal identity, the disambiguation comes from a
DIFFERENT axis: the ordinal counts emissions of the same render-text
in the same body REGARDLESS of category. A pack-MFMA emitting
`v_mfma_f32_4x4x4_16b_bf16 v[...], v[...], v[...]` and a main-loop
MFMA emitting the same render-text would BOTH increment the same
ordinal counter, getting consecutive ordinals (0 and 1). Both end up
as distinct identities, both correctly mapped to distinct nodes.

The behavior under cross-build comparison: if both builds emit one
pack-MFMA and one main-loop MFMA with that render-text in the same
body, both builds derive the same two ordinals (0 and 1), and the
identities match. If only one build emits the pack-MFMA (a real
divergence), the identity sets differ and `compare_graphs` surfaces it
correctly.

For the role-aware downstream uses (e.g., `is_mfma_pack_producer` at
`CMSValidator.py:960-986`, which discriminates pack-MFMAs from real
MFMAs for the quad-cycle gap branch), the discriminator stays available
on the GraphNode via `node.category` (the raw string) and on
`node.rocisa_inst` (for rocisa-derived classification). NEITHER needs
identity[0]. The role discrimination consumers are independent of
identity discrimination consumers.

### §3.3 Verification — is canonical_render-text alone unique per-body?

Spot-checking the `_CLASS_NAME_TO_CATEGORY` registry: render-strings
embed the rocisa class's print layout, which is opcode-specific. Two
DIFFERENT rocisa classes (e.g., `DSLoadB128` vs `BufferLoadB128`)
produce different render-strings (`ds_read_b128 ...` vs `buffer_load_b128
...`). Two instances of the SAME rocisa class with the same operands
produce the same render — and that's exactly what the ordinal
discriminates.

Cross-class same-render is implausible given rocisa's render
conventions. The registry has 13 buckets; their opcode prefixes
(`ds_read_*`, `ds_store_*`, `buffer_load_*`, `global_load_*`, `v_mfma_*`,
`s_waitcnt`, `s_barrier`, `s_nop`, `s_setprio`, `v_cvt_pk_bf16_f32`,
`v_cvt_f32_bf16` / `v_sub_f32` / `v_dot2c_f32_bf16`, `s_load_*`,
`flat_*` / `buffer_store_*` / `global_store_*`) are all distinct.
A future rocisa addition that produces a render-string colliding with
an existing class's render would be a rocisa correctness bug, not a
validator concern.

The pack-MFMA case: same rocisa class as main-loop MFMA, identical
render. The ordinal handles this because two emissions of the same
render-string in the same body get distinct ordinals. ✓

### §3.4 Verdict

**Drop class_tag from identity.** The disambiguator role is taken over
by the ordinal; the filter / edge-key roles migrate to rocisa-derived
predicates that consult `node.rocisa_inst` directly. No surviving
need for a category-string slot in identity.

---

## §4. Migration of downstream consumers

For each `identity[0]` consumer, the rocisa-derived replacement:

### §4.1 `_DATA_FLOW_KINDS` filter (`CMSValidator.py:3301-3304`)

Today:
```python
_DATA_FLOW_KINDS = ("LR", "LW", "GR", "MFMA")
def _data_flow_ids(graph):
    return {k for k in graph.nodes.keys() if k and k[0] in _DATA_FLOW_KINDS}
```

After:
```python
_DATA_FLOW_CATEGORIES = frozenset({
    InstructionCategory.LR, InstructionCategory.LW,
    InstructionCategory.GR, InstructionCategory.MFMA,
})
def _data_flow_ids(graph):
    return {n.identity for n in graph.nodes.values()
            if _category(n.rocisa_inst) in _DATA_FLOW_CATEGORIES}
```

`graph.nodes` is currently `{identity: GraphNode}` (see
`build_dataflow_graph:1789`); keep the dict shape, iterate values to
inspect rocisa, return identities matching the predicate. Same
semantics, no `identity[0]`.

`_summary_by_class` (`:3314-3319`) similarly migrates to:
```python
def _summary_by_class(ids, nodes_by_id):
    counts = {}
    for ident in ids:
        node = nodes_by_id[ident]
        cat = _category(node.rocisa_inst)
        key = cat.value if cat else "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    return counts
```

Threading `nodes_by_id` (= `graph.nodes`) into the call site is a
2-line plumbing change.

### §4.2 `edge_keys` role positions (`CMSValidator.py:1238-1241`)

Today:
```python
return {(e.producer.identity[0], e.producer.position, e.src_operand_slot,
         e.consumer.identity[0], e.consumer.position, e.sink_operand_slot,
         e.edge_kind, e.intra_operand_byte_offset)
        for e in self.edges}
```

After:
```python
def _role(node):
    cat = _category(node.rocisa_inst)
    return cat.value if cat else "UNKNOWN"

return {(_role(e.producer), e.producer.position, e.src_operand_slot,
         _role(e.consumer), e.consumer.position, e.sink_operand_slot,
         e.edge_kind, e.intra_operand_byte_offset)
        for e in self.edges}
```

The role vocabulary becomes rocisa-derived (`InstructionCategory` enum
values: `"LR"`, `"LW"`, `"GR"`, `"MFMA"`, `"CVT_PACK"`, `"MIDDLE_PACK"`,
`"SWAIT"`, etc.). Edge-set semantics are unchanged: same logical roles,
sourced from rocisa instead of from the CMS category string.

The same shape replicates at `CMSValidator.py:3349-3354` (ref-edge
lookup); apply the same rewrite there.

The PACK case under the rocisa-derived role: pack-MFMAs are real
`MFMAInstruction` rocisa objects with `_category()` returning
`InstructionCategory.MFMA`. So `_role(pack_mfma_node)` yields `"MFMA"`,
NOT `"PACK"`. The edge-key vocabulary loses the PACK slot — ALL MFMAs
share the same role token regardless of whether they're pack-MFMAs or
main-loop MFMAs.

This is a deliberate semantic shift, not a defect. The edge-key tuple's
purpose is to discriminate edges across graphs; it does not need a
scheduler-role distinction beyond what the rocisa classifier provides.
The pack-MFMA / main-loop-MFMA discrimination IS still needed at the
TIMING-DISPATCH layer (`is_mfma_pack_producer` at `CMSValidator.py:960-986`)
where `node.category.startswith("Pack")` controls quad-cycle gap routing
— that consumer is independent of identity and is unchanged.

### §4.3 The `_NO_DATAFLOW_IDENTITY_CATEGORIES` exclusion (`CMSValidator.py:1788`)

Already rocisa-derived today (consults `_category(inst)`, NOT
`node.identity[0]`). No migration needed. Flagged in §1.4 because the
on0t bead description was imprecise.

### §4.4 Diagnostic / rendering consumers

`cms_node_label` / `CmsLabelRenderer` / `FailureNodeLabel` all consume
`node.category` (the raw CMS string), NOT identity. Failure rendering
is unaffected.

The render-string semantics that users see in failure messages
(`PackA0[3]`, `LRA0[2]`, etc.) come from `CmsLabelRenderer.render()` at
`cms_to_timeline.py:112-116` and depend ONLY on `tagged_inst.category`
and `name_idx`. Both are unchanged. Pinning tests on render output
keep passing.

### §4.5 Tests

`test_dataflow_graph_lcc.py:128`:
```python
assert node.identity[0] == "LCC", node.identity
```
Migrate to the rocisa-derived role check or to a category check on the
TaggedInstruction. The LCC category lacks a rocisa-derived classifier
today (LCC instructions are `SSubU32` / `SCmpEQI32` rocisa classes,
which currently aren't in `_CLASS_NAME_TO_CATEGORY`). The test can
either:
- Assert `node.category == "LCC"` (CMS-string check at the test layer
  only — acceptable because tests SHOULD pin category-string semantics
  separately from identity semantics).
- Assert `_category(node.rocisa_inst) is None and node.category == "LCC"`
  (combined check that pins both layers).

Recommended: the first form. `category` on GraphNode is the public
display attribute; identity is the comparison key. Tests of identity
shape must be rewritten to the new shape regardless.

`test_dataflow_graph_builder.py:590`:
```python
assert pack_nodes[0].identity[0] == "PACK"
```
Must rewrite. The per-emission-ordinal identity has no `identity[0]`
slot for class_tag. The intent of the test (pack producers exist in the
expected place) becomes:
```python
assert pack_nodes[0].category.startswith("Pack")
```
or migrate to a rocisa-derived check.

`test_dataflow_graph_comparison.py:586-609`: the entire
`test_category_overrides_isinstance_class_tag` test (lines 550-609)
becomes obsolete — it pins `class_tag_for_category` behavior, and
`class_tag_for_category` is no longer consulted by identity. The
function itself can stay (it's still useful as a category-collapse
helper for any consumer that wants the historical buckets), but its
identity-coupling tests come out.

The 4 fixture sites at `:300, :308, :320, :328, :347, :354` that
construct synthetic `category="GRA"` TaggedInstructions still work
(they go through the same `_make_node` path; identity now uses the
ordinal instead of `class_tag_for_category("GRA", inst)`). No fixture
rewrite needed for those.

---

## §5. Implementation sketch (capture-side)

### §5.1 LoopBodyCaptureBuilder — ordinal counter

`Tensile/Components/ScheduleCapture.py:796-877`:

```python
class LoopBodyCaptureBuilder:
    def __init__(self):
        self._instructions = []
        self._seq_counter = {}
        # NEW: per-(canonical_render) emission counter, body-scoped.
        # Keys are pure render-strings; no category, no class_tag.
        self._render_ordinal_counter = {}

    def append(self, inst, category, subiter, slot_kind=SLOT_KIND_MFMA, mfma_index=-1):
        # ... existing slot-key computation ...
        wrapped = WrappedInstruction(inst)
        # NEW: assign emission_ordinal at append time. Capture the
        # canonical render BEFORE finalize() runs the per-rule
        # populator so the ordinal is stable.
        render = WrappedInstruction.canonical_str(inst)
        ord_idx = self._render_ordinal_counter.get(render, 0)
        self._render_ordinal_counter[render] = ord_idx + 1
        self._instructions.append(TaggedInstruction(
            wrapped=wrapped, category=category, slot=slot,
            emission_ordinal=ord_idx,  # NEW
        ))
```

`TaggedInstruction` (`:471-536`) adds an `emission_ordinal: int` field.

### §5.2 identity_for — new tuple shape

```python
def identity_for(self, body_label: str) -> tuple:
    from Tensile.Components.CMSValidator import BODY_LABEL_TO_LOOP_INDEX
    inst = self.wrapped.rocisa_inst
    loop_idx = BODY_LABEL_TO_LOOP_INDEX[body_label]
    return (loop_idx, WrappedInstruction.canonical_str(inst), self.emission_ordinal)
```

Three slots: loop_index, canonical_render, ordinal. No class_tag.

### §5.3 Determinism guard — assign ordinals after canonical sort

To make ordinal assignment robust against `_instructions` insertion
order (the `_seq_counter` invariant assumes natural append order
matches lex-sort, but per
`assign_stream_indices_for_body:744-760`'s defensive re-sort comment,
we should not rely on this), the recommended implementation moves the
ordinal computation to `LoopBodyCaptureBuilder.finalize` AFTER sorting:

```python
def finalize(self):
    # Sort once, deterministically, before assigning ordinals.
    sorted_tis = sorted(
        self._instructions,
        key=lambda ti: (ti.slot.mfma_index, ti.slot.sequence),
    )
    counter = {}
    for ti in sorted_tis:
        render = WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst)
        ti.emission_ordinal = counter.get(render, 0)
        counter[render] = ti.emission_ordinal + 1
    # ... existing rocisa-wiring / SMEM / FLAT / store guards ...
    return LoopBodyCapture(instructions=list(sorted_tis))
```

This makes ordinal a function of the canonical sort order, NOT of
append timing. Both real builds get identical ordinals as long as the
sorted order is identical — which holds because both builds derive
`(slot.mfma_index, slot.sequence)` from the same kernel-writer state
via the same builder.

(`TaggedInstruction` becomes mutable in `emission_ordinal` only;
alternatively, freeze it and use `dataclasses.replace`. Same effect.)

### §5.4 Removing class_tag_for_category from the identity path

`class_tag_for_category` (`ScheduleCapture.py:374-426`) becomes
unreferenced by identity construction. Keep the function as a public
helper (other consumers may want the historical bucket vocabulary —
see `_summary_by_class` in §4.1 for one). Remove the `class_tag`
positional consumer at `identity_for:535`. Update the docstring at
`:507-529` to describe the new tuple shape.

`WrappedInstruction.class_tag` (`:336-372`) similarly stays as a public
API for any downstream that wants the rocisa-derived class tag (it's
already rocisa-derived via the `_category` registry — it's just not
consulted by identity anymore).

---

## §6. Migration sequencing

### §6.1 Ordering relative to svb1 and 71hw

**ALONGSIDE svb1.** `rocm-libraries-svb1` (idmap-factory unification)
operates on the categorization layer (`build_idmap` /
`build_id_to_category_per_iter`); per-emission-ordinal identity
operates on the identity-construction layer downstream of
categorization. Neither blocks the other; both can land independently
once xqj3's design (this memo) is approved. svb1's hybrid A+C
recommendation may become OBSOLETE under xqj3 if the categorization
layer is restructured into rocisa-derived role producers — but that's
xqj3's concern, not svb1's. For sequencing: land per-emission ordinal
when ready; svb1 lands when ready; neither needs to wait for the
other.

**BEFORE 71hw's full Approach A implementation.** The per-emission
ordinal identity must be in place before 71hw lands a real-build
non-CMS-side capture path that compare_graphs consumes as
`reference`. Reason: the on0t collisions surface against the SHADOW
capture today; under Approach A they'd surface against a real
non-CMS-side capture that suffers from the same collision pattern (per
the on0t bead's mechanism description: "intra-capture; both real builds
independently apply the last-writer-wins clobber"). Without the
ordinal, Approach A inherits the spurious-failure surface; with it,
Approach A starts clean.

Operationally: stage per-emission ordinal as a small, focused PR before
71hw's larger second-build PR. Run the test suite under the new
identity to confirm the existing unit corpus passes. Then proceed
with 71hw.

**ALONGSIDE oram.1.** `rocm-libraries-oram.1` (body-label-tolerance for
cross-body PLR-pack movement) is orthogonal to identity composition:
oram.1 changes how `compare_graphs` interprets cross-body identity
divergence; per-emission ordinal changes how identity is constructed.
They compose. Recommended order: per-emission ordinal first (smaller
blast radius, clean baseline), oram.1 second (consumes the new
identity shape).

### §6.2 Test strategy — proving cross-build determinism BEFORE downstream consumers depend on it

The determinism invariant (§2.6) is the load-bearing claim. Validate it
directly in the unit suite BEFORE migrating any consumer:

1. **Per-render emission count assertion.** Add a fixture that builds
   one CMS-emit and one non-CMS-emit capture for a representative
   schedule (e.g., `_get_schedule_160x128x64_TF32`). Assert that for
   every (body, canonical_render) pair, both captures' counts match.
   Failure here means the determinism invariant doesn't hold for that
   schedule — fix or scope-out before proceeding.

2. **Per-render ordinal assignment assertion.** Same fixture, assert
   that for every (body, canonical_render, ordinal) tuple appearing in
   either capture, both captures have the same TaggedInstruction's
   underlying category (or both have it tagged identically up to
   class_tag-equivalent). This tests that the ordinal-N-th emission
   refers to the same logical instruction on both sides.

3. **Z012 collision regression.** Add a test that synthesizes a
   capture with two `s_cmp_eq_u32 LoopCounterL, StaggerUIter`
   emissions in the same body — assert they get distinct identities
   under the new scheme. Regression-pin against the on0t failure
   mode.

4. **example.yaml end-to-end.** Run `example.yaml` against
   `_get_schedule_160x128x64_TF32` after the migration; assert that
   the 8 spurious `OrderInvertedFailure`s from
   `EXAMPLE_YAML_DEFECT_INVESTIGATION.md §1` no longer fire.

5. **Full unit suite under
   `pytest --ignore=tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py`**
   stays green. Tests pinning `identity[0] == "..."` get rewritten per
   §4.5; structural identity tests get rewritten to the new tuple
   shape.

### §6.3 Migration as one PR

Per §5, the change set is:
- `ScheduleCapture.py`: add `emission_ordinal` to `TaggedInstruction`,
  add ordinal assignment in `LoopBodyCaptureBuilder.finalize`, change
  `identity_for` to the new tuple.
- `CMSValidator.py`: rewrite `_data_flow_ids` and `_summary_by_class`
  to consult `node.rocisa_inst`; rewrite the two `edge_keys`-shaped
  tuple constructions (`:1238` and `:3349`) to compute `_role` from
  rocisa.
- Tests: rewrite the four pinning tests called out in §4.5; add the
  five regression tests in §6.2.

This is one cohesive PR. No staged half-state where the new identity
exists alongside the old class_tag in identity. The risk of a partial
migration is exactly the consumer-coupling problem the umbrella xqj3
exists to prevent: every half-state preserves CMS shape in some slot.

---

## §7. Sub-bead under xqj3

Filed as `rocm-libraries-9c7y` (see §0 of this commit's br state).
Title: "Replace identity_for with (loop_index, canonical_render,
emission_ordinal) — drop class_tag from identity".
Type: `task`. Priority: `P1`. Wired under `xqj3`.

Comment posted on `rocm-libraries-on0t` recording that this design
supersedes Approach 11 once it lands.

---

## §8. Discipline check — what this design does NOT contain

- **No CMS-derived fallback.** The emission_ordinal counter keys on
  pure canonical_render-text. The migration of `_DATA_FLOW_KINDS` and
  `edge_keys` consults the rocisa-derived `_CLASS_NAME_TO_CATEGORY`
  registry directly. No `category.startswith(...)` branch survives in
  identity composition or its load-bearing consumers.

- **No tactical patch.** Approach 11 (selective un-collapse in
  `class_tag_for_category`) is explicitly NOT recommended. The cheap
  mechanical fix is correct in isolation but perpetuates the CMS
  coupling at the identity layer; the user has rejected this framing.

- **No half-state.** The migration is one PR (§6.3). No
  per-consumer migration that leaves identity_for half-rocisa-derived,
  half-CMS-coupled.

- **No untested determinism claim.** §6.2 specifies the test fixtures
  that validate the determinism invariant BEFORE downstream consumers
  depend on it. If the invariant doesn't hold for some schedule, the
  test catches it before the migration breaks production.

- **No render-text uniqueness claim beyond what canonical_str
  guarantees.** §3.3 verifies that across rocisa class boundaries,
  render-text collisions are implausible given current rocisa
  conventions. Same-class collisions (the pack-MFMA case) ARE handled
  by the ordinal disambiguator. Cross-class collisions, if they ever
  appear, would be a rocisa correctness regression and surface as a
  test failure in the per-render-count assertion of §6.2.

---

## §9. Citations (verified by reading file:line this investigation)

- `Tensile/Components/ScheduleCapture.py:374-426` —
  `class_tag_for_category` collapse table.
- `Tensile/Components/ScheduleCapture.py:507-536` — `identity_for`,
  current 3-tuple shape `(class_tag, loop_index, canonical_render)`.
- `Tensile/Components/ScheduleCapture.py:540-559` — `LoopBodyCapture`.
- `Tensile/Components/ScheduleCapture.py:796-877` —
  `LoopBodyCaptureBuilder.append` / `finalize`.
- `Tensile/Components/ScheduleCapture.py:744-760` —
  `assign_stream_indices_for_body` lex-sort by
  `(slot.mfma_index, slot.sequence)`.
- `Tensile/Components/ScheduleCapture.py:235-298` —
  `WrappedInstruction.is_lr` / `is_lw` / `is_gr` / `is_mfma` /
  `is_swait` / `is_sbarrier` / `is_snop` / `is_ssetprio` predicates.
- `Tensile/Components/ScheduleCapture.py:303-333` —
  `WrappedInstruction.canonical_str` (the per-render input to ordinal
  keying).
- `Tensile/Components/ScheduleCapture.py:908-947` — `build_idmap` (the
  CMS schema this design targets for migration).
- `Tensile/Components/ScheduleCapture.py:1017-1117` —
  `build_id_to_category_per_iter` (the SIA3 default-side schema).
- `Tensile/Components/ScheduleCapture.py:2186-2303` —
  `expand_cms_macro` (CMS-side capture funnel).
- `Tensile/Components/InstructionCategory.py:55-99` —
  `InstructionCategory` enum.
- `Tensile/Components/InstructionCategory.py:111-227` —
  `_CLASS_NAME_TO_CATEGORY` registry (the rocisa-derived classifier).
- `Tensile/Components/CMSValidator.py:893-957` — `GraphNode` dataclass
  (note: docstring at `:913` says
  `(rocisa_class_name, loop_index, signature_tuple)` — stale; actual
  shape is `(class_tag, loop_index, canonical_render)`).
- `Tensile/Components/CMSValidator.py:1238-1241` — `edge_keys`
  consumes `identity[0]` as `src_role`/`sink_role`.
- `Tensile/Components/CMSValidator.py:1509-1514` —
  `_NO_DATAFLOW_IDENTITY_CATEGORIES` (rocisa-derived already; not
  identity-coupled).
- `Tensile/Components/CMSValidator.py:1587-1617` — `_make_node`
  (graph-build-time identity construction site).
- `Tensile/Components/CMSValidator.py:1789` — last-writer-wins
  collapse `nodes_by_identity[node.identity] = node`.
- `Tensile/Components/CMSValidator.py:3275-3357` — `compare_graphs`
  with `_DATA_FLOW_KINDS` filter at `:3301-3304` and ref-edge
  `identity[0]` lookup at `:3349-3354`.
- `Tensile/Components/CMSValidator.py:3202-3252` — `cms_node_label`
  (consumes `node.category`, NOT `identity[0]`).
- `Tensile/Components/cms_to_timeline.py:81-143` — `CmsLabelRenderer`
  and `_name_idx_for` (per-category-stream index for failure rendering;
  similar in spirit to per-emission ordinal but keyed on category, NOT
  on render-text — kept as-is for failure rendering, not consulted by
  identity).
- `Tensile/KernelWriter.py:2601-2688` — `_captureSubIterToBuilder`
  (default-side capture funnel into `LoopBodyCaptureBuilder.append`).
- `Tensile/KernelWriterAssembly.py:9024-9148` — `globalReadIncrement`
  (the `SCmpEQU32(loopCounter, sgpr("StaggerUIter"))` emission).
- `Tensile/KernelWriterAssembly.py:9149-9187` — `globalReadIncrementAB`
  (per-side dual call).
- `Tensile/Components/EXAMPLE_YAML_DEFECT_INVESTIGATION.md §1-2` — the
  8 production failures the new identity tuple must extinguish.
- `Tensile/Components/Z012_RESIDUAL_INVESTIGATION.md` — the on0t
  framing (now superseded by this memo's §3-§5).
- `Tensile/Components/2LZD_INVESTIGATION.md §1` — the cross-build
  count-divergence catalog (Pack +60, LRS/LWS net 0 / +4 cat-mismatch
  edges) that defines the determinism invariant's edge cases.
- `Tensile/Components/PRELOOP_CAPTURE_PHASE1.md §2` — body-label
  identity-sensitivity surface (the oram.1 work that lands alongside
  this).

---

## §10. Reporting summary

- **Class_tag verdict:** drop entirely from identity. Disambiguator
  role taken by emission_ordinal; filter / edge-key roles migrated to
  rocisa-derived predicates that consult `node.rocisa_inst` via the
  centralized `_CLASS_NAME_TO_CATEGORY` registry.
- **Determinism guarantee:** both real builds consume identical
  kernel-writer source modules, so for each `(body,
  canonical_render)` they emit the same number of instances in the
  same relative order; per-emission ordinals therefore match cross-
  build by construction.
- **Sequencing:** ALONGSIDE svb1 (independent layers); BEFORE 71hw's
  Approach A real-build PR (so the new validator path starts clean);
  ALONGSIDE oram.1 (body-label tolerance composes orthogonally on the
  new identity).
- **Framing corrections to the dispatching brief:**
  - `_NO_DATAFLOW_IDENTITY_CATEGORIES` (`CMSValidator.py:1788`) is
    NOT identity-coupled today — it consults `_category(inst)`, not
    `node.identity[0]`. The brief implied the opposite.
  - `cms_node_label` and the CMS-side `FailureNodeLabel` providers
    consume `node.category`, NOT `identity[0]`. The brief flagged
    them as identity-coupled; they are not.
  - `GraphNode.identity` docstring at `CMSValidator.py:913` is stale
    (claims `(rocisa_class_name, loop_index, signature_tuple)` — the
    actual shape is `(class_tag, loop_index, canonical_render)`).
    Worth correcting in the same PR.
