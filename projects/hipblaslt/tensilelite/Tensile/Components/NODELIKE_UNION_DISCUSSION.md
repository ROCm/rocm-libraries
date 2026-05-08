# NodeLike Union pattern: discussion memo

Bead: `rocm-libraries-x4ef`. Branched off
`users/alvasile/validator_long_term_plans` at `bb96c76a19`.

This is a doc-only deliverable. No production code or tests are
modified. Pin everything to file:line as of that base SHA.

## 0. Background

`cms_node_label` (`Tensile/Components/CMSValidator.py:2283`) constructs
the per-node `FailureNodeLabel` carried by every typed `Failure`. Its
parameter `node` is typed as

    NodeLike = Union["GraphNode", "ValidatorInstruction"]

declared at `Tensile/Components/CMSValidator.py:302`. Both shapes flow in
because the helper is used by two unrelated source paths:

- **Graph-side**: the dataflow comparison pipeline
  (`compare_graphs`, `diagnose_missing_edge`,
  `validate_edge_wait_coverage`, `validate_middle_pack_pair_interleaving`)
  hands it a `GraphNode` (defined at `CMSValidator.py:305`). A
  `GraphNode` carries `position`, `category`, `tagged_inst`, and
  `body_label` natively (the `_make_node` builder populates them
  unconditionally — see `CMSValidator.py:832-851`).

- **Structural-side**: per-instruction `validate()` overrides on
  `GlobalRead._validate_needed_by`
  (`Tensile/Components/CMSValidator.py:3307-3327`) — the only
  remaining caller of this kind. It has the `ValidatorInstruction`
  (`CMSValidator.py:3152`) it is validating in scope; no
  `GraphNode` is available because no dataflow graph has been
  constructed at that point in the structural pipeline.

The helper discriminates with `getattr` probes: `category` exists on
both shapes (a field on `GraphNode` at `:329`, a property on
`ValidatorInstruction` at `:3165-3173`), `tagged_inst` and `body_label`
exist only on `GraphNode` and are read with `getattr(..., None)` at
`CMSValidator.py:2306` and `:2329`. `_node_position`
(`CMSValidator.py:1965-1970`) does the same dance: `getattr(node,
"position", None) or node.issued_at`.

The Union therefore pushes the two-shape complexity into one helper
rather than spreading it across N call sites, but the tradeoff is real:
`getattr` probes are an admission that the type signature is wider than
the helper actually wants to handle.

---

## 1. Three options for the abstraction

### Option A — Keep the Union (status quo)

`cms_node_label(node: Union[GraphNode, ValidatorInstruction], body_capture)`
with internal `getattr` discrimination.

**Pros**
- Zero churn. Every existing call site
  (CMSValidator.py:2548, :2549, :2595, :2596, :2599, :2624, :2625,
  :2644, :2645, :2664, :2665, :2698, :2699, :2896, :2897, :3141,
  :3142, :3145, :3326, :3327) continues to compile unchanged.
- Confines the two-shape bookkeeping to two files (the helper plus
  `_node_position`). New `Failure`-emitting code in either source can
  call the same helper with no boilerplate.
- The `getattr` probes are bounded — they look for two attributes
  (`tagged_inst`, `body_label`, `position`) and have explicit
  fallbacks. There is no defensive open-ended duck-typing.

**Cons**
- The type alias lies about the contract. A `ValidatorInstruction`
  passed in cannot ever produce a `[N]`-suffixed primary because
  `body_capture` is forced to `None` by the only structural-side
  caller (`CMSValidator.py:3326-3327`). The type accepts both shapes
  but the rendering is asymmetric.
- `getattr(node, "position", None) or node.issued_at`
  (`CMSValidator.py:1970`) silently flips behavior between shapes;
  any future field rename in `GraphNode` (e.g. dropping `position`
  for a different name) becomes a silent regression on the
  structural side.
- New developers reading the type alias have to learn the dispatch
  rule from the docstring rather than the type system.

**Call-site impact if adopted (kept)**: 0 changes.

### Option B — `ValidatorInstruction.to_node_label()` method

Move the bare-category fallback into a method on `ValidatorInstruction`
itself; have `cms_node_label` accept only `GraphNode`. The structural
caller becomes
`producer_label = self.to_node_label(); consumer_label = self.needed_by.to_node_label()`.

**Pros**
- `cms_node_label`'s type signature shrinks to a single shape; the
  `getattr` probes for `tagged_inst` and `body_label` go away.
- `_node_position`'s `getattr` probe also goes away — split into two
  helpers, one per shape, or replaced with direct attribute reads.
- Each shape owns its own label-construction logic. If a future
  `ValidatorInstruction` subclass wants to plumb richer information
  into its label (see Topic 2), the method is the natural extension
  point.
- Eliminates the type-system lie: the API now says exactly what it
  accepts.

**Cons**
- `ValidatorInstruction` (in CMSValidator.py) gains a dependency on
  `FailureNodeLabel` (also in CMSValidator.py — same file, so no
  import cycle, but a layering inversion: instructions used to be
  pure data and validation logic). Today
  `ValidatorInstruction.validate()` returns `Optional[Failure]`, and
  Failures already reference `FailureNodeLabel`s, so the dependency
  is conceptually already present; this just makes it explicit.
- The method body is essentially the bare-category branch of
  `cms_node_label`. There is a small duplication risk if the
  graph-side helper's bare-category fallback drifts away from the
  structural-side method.
- Two structural call sites to update
  (`CMSValidator.py:3326`, `:3327`). Plus the helper signature
  narrows, so every graph-side caller's argument type changes (no
  call-site code changes — the graph-side already passes
  `GraphNode`s — but type-checker errors surface anywhere a
  `ValidatorInstruction` was incidentally being passed).
- One test pinning the helper's behavior on a `GraphNode`
  (`test_dataflow_graph_register_gaps.py:3696`) is unaffected. No
  test pins the `ValidatorInstruction` shape directly today, so the
  method-side migration adds no test changes.

**Call-site impact if adopted**: 2 structural-side call sites
rewritten; 19 graph-side sites unchanged (just retyped); 0 tests
modified.

### Option C — Thin GraphNode-shim wrapper

Wrap `ValidatorInstruction` in a minimal `GraphNode`-shaped adapter at
the structural call site:

    producer_label = cms_node_label(_shim_node(self), None)

where `_shim_node` returns a tiny dataclass with `category`, `position`
(=`issued_at`), `tagged_inst=None`, `body_label=None`.

**Pros**
- `cms_node_label` accepts a single shape (just `GraphNode` or a
  GraphNode-protocol).
- Minimal change to `ValidatorInstruction`'s public surface.

**Cons**
- The shim is strictly worse than Option B. It introduces a new
  ephemeral object whose only purpose is to satisfy the helper's
  type signature — exactly the "thin wrapper for shape conversion"
  pattern that accumulates kludge.
- The `body_label=None` and `tagged_inst=None` fields encode the
  same body-context loss as Option A but now the loss is hidden
  inside the shim rather than visible at the call site.
- No clear extension story. If we later want to plumb body context
  into the structural side (Topic 2), the shim would have to grow
  the same fields anyway — at which point it is just a parallel
  `GraphNode` with no clear ownership.
- Adds a new file-level helper for nothing.

**Call-site impact if adopted**: 2 structural-side sites rewritten,
plus a new shim helper. Strictly more code than Option B with no
abstraction gain.

### Recommendation for Topic 1

**Option A (keep the Union), with two doc tweaks**: see Topic 4.

The reason to NOT pick Option B today is purely a matter of
priority-vs-payoff. The structural-side caller path is currently
dead-code in production (Topic 2 result), so the type-system
asymmetry the Union creates has no observable user-facing
consequence. Option B is the cleaner abstraction — and it is the
right answer the moment a SECOND structural-side path needs to
construct a label, or the moment we plumb body context into the
structural side. Until then, adopting Option B trades a small
churn for an abstraction win that no caller currently exercises.

---

## 2. Body-context loss on the structural side

The structural-side path passes `body_capture=None` at
`CMSValidator.py:3326-3327`:

    producer_label = cms_node_label(self, None)
    consumer_label = cms_node_label(self.needed_by, None)

Inside the helper at `CMSValidator.py:2308`, `body_capture is None`
forces the bare-category branch:

    if body_capture is not None and tagged is not None:
        same_cat = [...]
        try:
            idx = same_cat.index(tagged)
            primary = f"{cat}[{idx}]"
        except ValueError:
            pass

Result: a structural-side `MissingWaitFailure` for the third
`GRA` in flight renders as `GRA` rather than `GRA[2]`. The graph-side
path renders the same failure as `GRA[2]` because it passes
`_body_for_node(subj_graph, p_node)` through.

### Is this loss-of-precision currently affecting any failure rendering?

**No.** The structural caller is currently dead.

Evidence:

1. The only way into `GlobalRead._validate_needed_by`
   (`CMSValidator.py:3307`) is via `validate_timeline`
   (`CMSValidator.py:3896-3918`). Production callers of `isValid`
   (the production validation entry point at
   `CMSValidator.py:3996`) do NOT invoke `validate_timeline` — see
   `CMSValidator.py:3996-4068`, which only runs the graph-side
   `compare_graphs` + `validate_edge_wait_coverage` pipeline.

2. The only caller of `validate_timeline` anywhere in the tree is
   `Tensile/Tests/unit/cms_validation_base.py:106`. Production has
   one transitive `cmsv.isValid` call from
   `Tensile/Components/CustomSchedule.py:376`; that path is
   graph-side only.

3. `GlobalRead.needed_by` defaults to `MFMA(name="MFMA",
   issued_at=POSITION_INF)` (`CMSValidator.py:3290`). The first
   line of `_validate_needed_by` at `:3310` early-returns when
   `needed_by.issued_at == POSITION_INF`. A grep for
   `\.needed_by\s*=` across the entire codebase finds no
   production assignment that overrides the default — only the
   default-factory line and self-references inside the helper
   itself (`CMSValidator.py:3310, :3313, :3318, :3327, :3328,
   :3349, :3356`). No test assigns `needed_by` either.

So the structural-side label-construction code at `:3326-3327`
exists but is unreachable today; the `body_capture=None` fallback is
not surfacing a "GR" label to any user — production or test.

This makes Topic 2 a **theoretical** concern under the current
tree. The right disposition is therefore the cheapest one:

### Recommendation for Topic 2

**Accept the loss with a docstring note**, AND add a TODO
referencing this memo if the structural-side path is ever
revived.

Concretely (illustrative — no edit applied):

In `cms_node_label`'s docstring (currently at
`CMSValidator.py:2287-2301`), the existing line

    `body_capture` is the LoopBodyCapture for the body that
    emitted this node (resolved by the caller from
    `node.body_label`). When `body_capture` is None or the node
    has no `tagged_inst` recorded in that body, the helper falls
    back to a bare `category` primary

is already the docstring half. The missing half is to spell out
that the structural-side caller (`GlobalRead._validate_needed_by`)
deliberately passes `body_capture=None` because no body context is
in scope, and that this PATH IS CURRENTLY DEAD CODE in both
production and tests — so the loss-of-precision is not affecting
any rendered failure.

Plumbing the body context into the structural side (the alternative
route) is technically possible. Today `ValidatorInstruction` has no
`body_label` or `tagged_inst` field (`CMSValidator.py:3152-3163`).
The body-mapping data is held by the `Timeline` that
`validate_timeline` walks (`CMSValidator.py:3906-3908`). Plumbing
it would require either:

- Adding `body_label`/`tagged_inst` fields to `ValidatorInstruction`
  (forces every constructor in
  `ScheduleCapture._instruction_for_tagged` and the test fixture
  helpers in `_gr`, `make_swait`, `make_sbarrier` to populate the
  fields), then having the structural caller pass
  `_body_for_node(graph_or_None, self)` — but no graph exists at
  that callsite.

- Or threading the body capture through `validate_timeline` ->
  `instruction.validate()` (changes the abstract method signature
  on `ValidatorInstruction` at `:3175`).

Both options are non-trivial and would only matter if the
structural-side path is revived. Punting them to a follow-up bead
keeps scope contained.

**Follow-up bead recommendation (do not file)**: if/when
`GlobalRead._validate_needed_by` is reactivated as part of a
structural-validation revival, file a bead to plumb `body_capture`
into the structural caller. Likely shape: extend
`ValidatorInstruction.validate()` to accept an optional
`body_capture` parameter and propagate from `validate_timeline`.

---

## 3. All callers of `cms_node_label`

A grep across the whole tree
(`grep -rn "cms_node_label" projects/hipblaslt/tensilelite --include="*.py"`)
returns the following call sites. Twenty production callers, one
test caller. Each is annotated graph-side (G) or structural-side
(S).

| File:line | Side | Context |
|---|---|---|
| `CMSValidator.py:2548` | G | `OrderInvertedFailure` producer in `diagnose_missing_edge`. |
| `CMSValidator.py:2549` | G | `OrderInvertedFailure` consumer, same site. |
| `CMSValidator.py:2595` | G | `OverriddenInputFailure` producer (SCC clobber). |
| `CMSValidator.py:2596` | G | `OverriddenInputFailure` consumer (SCC clobber). |
| `CMSValidator.py:2599` | G | `OverriddenInputFailure.intervening_writer` (SCC clobber). |
| `CMSValidator.py:2624` | G | `TimingTooCloseFailure` producer (PackMFMA -> CVTPack). |
| `CMSValidator.py:2625` | G | `TimingTooCloseFailure` consumer, same site. |
| `CMSValidator.py:2644` | G | `TimingTooCloseFailure` producer (MFMA -> any). |
| `CMSValidator.py:2645` | G | `TimingTooCloseFailure` consumer, same site. |
| `CMSValidator.py:2664` | G | `TimingTooCloseFailure` producer (CVTPack -> MFMA). |
| `CMSValidator.py:2665` | G | `TimingTooCloseFailure` consumer, same site. |
| `CMSValidator.py:2698` | G | `p_label` for Phase-2 wait-coverage emit. |
| `CMSValidator.py:2699` | G | `c_label` for Phase-2 wait-coverage emit. |
| `CMSValidator.py:2896` | G | `p_label` in `validate_edge_wait_coverage`. |
| `CMSValidator.py:2897` | G | `c_label` in `validate_edge_wait_coverage`. |
| `CMSValidator.py:3141` | G | MiddlePack pair-interleaving producer. |
| `CMSValidator.py:3142` | G | MiddlePack pair-interleaving consumer. |
| `CMSValidator.py:3145` | G | MiddlePack pair-interleaving intervening_writer. |
| `CMSValidator.py:3326` | **S** | `GlobalRead._validate_needed_by` producer; `body_capture=None`. |
| `CMSValidator.py:3327` | **S** | `GlobalRead._validate_needed_by` consumer; `body_capture=None`. |
| `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py:3696` | G (test) | Pins SSetPrior label primary; passes a synthesized `GraphNode`. |

**Twenty production call sites + one test call site = twenty-one
total. Two of the production sites are structural; nineteen are
graph-side. No third shape exists** — every caller passes either a
`GraphNode` or a `ValidatorInstruction`.

Two facts worth noting:

1. The structural-side count is asymmetric: nineteen graph-side
   versus two structural-side. The Union exists for two call sites
   in one method.
2. Both structural sites are inside the same dead-code path
   (Topic 2). If the path stays dead, the Union is justified by
   exactly zero live callers on the structural side.

---

## 4. Synthesis

**Recommendation: Delete the dead structural-side path and collapse
`cms_node_label` to a single `GraphNode` shape.** Concretely: remove
`validate_timeline` and the unreachable body of
`GlobalRead._validate_needed_by` past the `:3310` early-return; narrow
`cms_node_label` to take `GraphNode` only (no Union, no `getattr`
discrimination, no `body_capture=None` precision-loss path); rename or
delete the `NodeLike` alias.

Justification (5 sentences):

1. The structural-side caller (`GlobalRead._validate_needed_by`) is
   unreachable in both production
   (`CustomSchedule.py:376` -> `cmsv.isValid` -> graph-side only) and
   tests (no test assigns `GlobalRead.needed_by`; the only
   `validate_timeline` invocation is in `cms_validation_base.py:106`
   and that test path is itself unused by any production validation
   call). The Union exists to support code that nothing exercises.

2. The "structural-side validation revival" justification for keeping
   the Union is purely speculative — no plan, bead, or design
   discussion in the project cites a concrete trigger. Codebase
   discipline elsewhere on this branch (wx9.3 phase 3 explicitly
   forbade fallback paths and parallel old/new APIs; the
   `_populate_wrapper` 2-tuple fallback was deleted on the same
   grounds even though only one test fixture exercised it) argues
   against keeping unused code "just in case."

3. Deleting the dead path eliminates THREE concrete sources of
   complexity simultaneously: (a) the `Union` and its `getattr`
   discrimination at `:2306` / `:2329` / `:1970`; (b) the
   `body_capture=None` precision-loss path in `cms_node_label` and
   the bare-category fallback wording in failure formatters; (c) the
   `_validate_needed_by` body that is the only consumer of
   `POSITION_INF` semantics — all in one focused change.

4. Option B (refactor to `ValidatorInstruction.to_node_label()`) is
   strictly inferior to deletion under the dead-code finding: it
   preserves the structural-side surface area without any caller
   benefiting from it, just dressed in a different abstraction.
   Option C (GraphNode-shim) is dominated by both deletion and B.

5. If a future use case genuinely needs structural-side validation
   (currently no such use case exists), reintroducing it then with
   `to_node_label()` and proper body-context plumbing is correct
   work — but it MUST be triggered by a real consumer, not by
   defending dead code. The git history preserves what was here for
   anyone needing to revive it.

### Recommended follow-up beads (recommendation only — not filed)

- **NEW (P2, code):** Implement the deletion described in this
  synthesis: remove `validate_timeline`, the unreachable body of
  `GlobalRead._validate_needed_by` past `:3310`, the
  `cms_validation_base.py:106` test caller (or its test if no other
  reason for it to exist), and narrow `cms_node_label` to a single
  `GraphNode` shape. Drop the `NodeLike` alias. Update
  VALIDATOR_DESIGN.md §7.4 to reflect the simplified shape and
  reference this memo for the deletion rationale. Pytest must remain
  672/2/1 (or higher if the dead test caller's removal is its own
  net-zero change).
