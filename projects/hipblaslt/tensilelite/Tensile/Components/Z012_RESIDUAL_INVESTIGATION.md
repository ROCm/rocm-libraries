# Z012 residual investigation: post-2lzd / svb1 / aprv

Investigator: z012-investigation-meta (Claude Opus 4.7, 2026-05-12).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/z012-investigation-meta/`.
Bead filed: `rocm-libraries-on0t` (z012-collision), under meta `rocm-libraries-71hw`.

This memo records two verdicts for the z012 follow-up after the
shadow-rejection decision (`2LZD_INVESTIGATION.md §6`) and the Q6
sibling investigations (svb1 — idmap-factory; aprv — PRE_LOOP /
POST_LOOP @-1):

1. **z012-tagging** (cross-build category-tagging consistency for
   LRSwap and other families): **empty under Approach A**. No bead
   filed. See §1.
2. **z012-collision** (intra-capture identity collisions for state
   registers): real, fileable, has a viable cheap fix. Bead
   `rocm-libraries-on0t`. See §2.

---

## 1. z012-tagging is absorbed by `class_tag_for_category`

### Claim from the brief

> `Z012_ALTERNATIVE_FIXES_INVESTIGATION.md §6` flagged that LRSwap
> (LRSA/LRSB) exhibits the same default-vs-CMS tagging divergence
> pattern as Pack (PackA0/PackA3): under shadow comparison, CMS-side
> tags as `LRSA`/`LRSB` while default-shadow tags as `LRS`. Under
> Approach A this remains a cross-build divergence: CMS-emit-build
> uses one tagging scheme; non-CMS-emit-build uses another.

The brief asks: under Approach A, does this surface as graph-shape
differences in `compare_graphs`?

### What `class_tag_for_category` does today (verified line-by-line)

`WrappedInstruction.class_tag_for_category` at
`ScheduleCapture.py:374-426`. Full collapse table:

| Category prefix or value | Returned class_tag |
|---|---|
| `category is None` | `class_tag(inst)` (isinstance fallback) |
| `LRA*`, `LRB*`, `LRMXSA*`, `LRMXSB*`, `LRMetadata*` | `'LR'` |
| `LRS*` (matches `LRS`, `LRSA`, `LRSB`) | `'LRS'` |
| `LWS*` (matches `LWS`, `LWSA`, `LWSB`) | `'LWS'` |
| `LW*` (matches `LW`, `LWA`, `LWB`) | `'LW'` |
| `GRInc*` (matches `GRIncA`, `GRIncB`) | `'GRINC'` |
| `GR*` (matches `GR`, `GRA`, `GRB`) | `'GR'` |
| `Pack*` (matches `PackA{0..n}`, `PackB{0..n}`) | `'PACK'` |
| `LCC` | `'LCC'` |
| `SYNC` | `class_tag(inst)` (disambiguates SWaitCnt vs SBarrier) |
| `SNOP` | `'SNOP'` |
| `SSETPRIO` | `'SSETPRIO'` |
| `BARRIER` | `'SBARRIER'` |
| `MFMA` | `'MFMA'` |
| `UNKNOWN` or unknown text | `class_tag(inst)` |

`identity_for` (`ScheduleCapture.py:507-536`) calls
`class_tag_for_category(self.category, inst)` at line 535, then
assembles the tuple `(class_tag, loop_index, canonical_render)`. The
class_tag is the ONLY identity component derived from category; the
raw category string never reaches the identity tuple.

### LRSwap trace

Default side: `KernelWriter._makeSubIterSchedule` at
`KernelWriter.py:974-979`:

```python
if pointerLRCode is not None:
  for item in pointerLRCode.flatitems():
    capture_id_to_category.setdefault(id(item), "LRS")
if pointerLWCode is not None:
  for item in pointerLWCode.flatitems():
    capture_id_to_category.setdefault(id(item), "LWS")
```

Tags pointer-flip items with bare `'LRS'` / `'LWS'`.

CMS side: `build_idmap` at `ScheduleCapture.py:934-937`:

```python
'LRSA':   LRSwapA,
'LRSB':   LRSwapB,
'LWSA':   LWSwapA,
'LWSB':   LWSwapB,
```

Per-tensor `LRSA/LRSB/LWSA/LWSB`.

Two captures, two TaggedInstructions for the same logical instruction
(say, `v_xor_b32 v[vgprLocalReadAddrA], 0x10000, v[vgprLocalReadAddrA]`):

| Capture | category | class_tag (via collapse) | identity[0] |
|---|---|---|---|
| Default | `'LRS'` | `'LRS'` | `'LRS'` |
| CMS     | `'LRSA'` | `'LRS'` | `'LRS'` |

`canonical_str` is the same (same writer state ⇒ same render). Both
identity tuples are `('LRS', loop_idx, canonical_render)`. **They
collide. compare_graphs sees one identity on each side, not two
diverging ones.** No `CaptureConsistencyError`, no
`OrderInvertedFailure` from this pattern.

The 2lzd memo's empirical evidence — `LRS:2 + LWS:2` in default vs
`LRSA:1, LRSB:1, LWSA:1, LWSB:1` in CMS, net 0 count, net +4
category-mismatch edges — was measured against the SHADOW capture's
INSTRUCTION-LIST, not the post-`identity_for` graph. The shadow
emitted raw category strings into a per-instruction count map; that
view is upstream of `class_tag_for_category` and saw the divergence.
The graph view does not.

### Pack trace

Default side: `build_id_to_category_per_iter` at
`ScheduleCapture.py:1108-1112`: `tag_module(sub, f"Pack{side}{subiter}")`
— per-subiter `PackA0/PackB0/PackA1/PackB1/...`.

CMS side: `build_idmap` at `ScheduleCapture.py:943-944`:
`idmap[f"PackA{u}"] = PackCodeA[u]` — same `Pack{side}{u}` shape.

If the two paths happen to assign different subiter indices to the
same physical Pack instruction (the `EXAMPLE_YAML_DEFECT_INVESTIGATION.md
§6` finding: 186 diverging Packs where default tagged `PackA0/PackB0`
but CMS tagged `PackA1/PackB1`), both still collapse to `'PACK'`. No
identity-level divergence.

### Verdict

**z012-tagging has no surface in `compare_graphs` under Approach A.**
The divergences exist at the raw-category level (visible in
`build_idmap` outputs and instruction-count maps) but are absorbed
before they reach the identity tuple. No bead filed.

If a future change extends the identity tuple to include the raw
category text (the rejected "Approach 9" line of work), the LRSwap
and Pack divergences would re-surface. That change would need to be
preceded by a tagging-reconciliation pass that aligns the
default-side bare-`'LRS'`/`'LWS'`/`'GR'` setdefault paths with the
CMS-side per-tensor scheme. Approach 9's previous failure
(`Z012_ALTERNATIVE_FIXES_INVESTIGATION.md §3 Approach 9`) was the
empirical data point that prompted the collapse-layer addition in the
first place. Today the layer is in place; tomorrow's Approach 9
would not need it if the upstream tagging is unified
(`rocm-libraries-svb1` may be the right place for that work).

---

## 2. z012-collision is real and has a cheap fix

Bead `rocm-libraries-on0t` carries the full mechanism, approach
catalog, and recommendation. This memo summarizes for quick reference;
the bead description is authoritative.

### Mechanism

`nodes_by_identity[node.identity] = node` at `CMSValidator.py:1789`
is last-writer-wins. When two physical kernel-writer emissions share
the same canonical render-string AND the same class_tag (post-collapse),
they collapse to one node. The surviving stream-position depends on
which capture saw which physical emission last — unstable across
captures with different scheduling.

The 8 production failures from `EXAMPLE_YAML_DEFECT_INVESTIGATION.md
§2` are the symptom: `s_cmp_eq_u32 LoopCounterL, StaggerUIter`
(emitted once per A-side / B-side GR-inc lowering — verified at
`KernelWriterAssembly.py:9051-9052`, called for both tensors via
`globalReadIncrementAB` at `:9173, :9181`) and `s_add_u32 m0, m0, 4224`
(emitted once per per-side GR group).

### Why Approach A doesn't fix it

The collision pattern is INTRA-CAPTURE. Both real builds (CMS-build
and non-CMS-build) independently apply the last-writer-wins clobber.
Comparing two captures preserves the artifact regardless of which is
shadow vs real. oram.1 is body-label work; z012-collision is within
a body.

### Why slot_kind doesn't help

Per `rocm-libraries-aprv`: `slot_kind` is identity-stable (not in
`identity_for`). But both GRIncA-cmp and GRIncB-cmp share the same
`slot_kind` (both inside the loop body, slot_kind=MFMA), so adding
slot_kind to identity wouldn't disambiguate them.

### Recommended fix (Approach 11 — new in this investigation)

Selective per-side preservation in `class_tag_for_category` ONLY for
categories where both default and CMS paths derive the per-side
string from the SAME source modules:

```python
if category.startswith("GRInc"):
    return category  # 'GRIncA' or 'GRIncB'
if category.startswith("GR"):
    return category if category in ("GRA", "GRB") else "GR"
```

GRInc and GR satisfy the per-tensor stability property:
- `globalReadIncACode` / `globalReadIncBCode` are the same Python
  modules consumed by both `build_idmap:928-929` (CMS) and
  `build_id_to_category_per_iter:1074-1077` (default).
- `globalReadA` / `globalReadB` ditto via `:930-931` and `:1078-1081`.

LRS / LWS do NOT satisfy this — `KernelWriter.py:976,979` tags
pointer items bare `'LRS'`/`'LWS'` on default while CMS uses per-tensor
`LRSA/LRSB/LWSA/LWSB`. Un-collapsing LRS/LWS would re-introduce the
asymmetry that the collapse layer was added to absorb.

**Cost.** ~6 LoC + one test-fixture update (`test_dataflow_graph_comparison.py:593-594`).

**Watchpoint.** Default-side fallback at `ScheduleCapture.py:1099`
(`id_to_category.setdefault(id(item), "GR")`) tags items in
`globalReadCode` not also in `globalReadA/B` with bare `'GR'`. The
new `class_tag_for_category` returns `'GR'` for those (they don't
match the `("GRA", "GRB")` set) — but the CMS side's `build_idmap`
has no bare `'GR'` bucket, so any such item would not appear with the
same class_tag on the CMS side. Pre-landing checklist (in the bead):
add a sentinel assertion that bare-'GR' setdefault never fires for
items not also in A/B; if it never fires on the cms_to_default
corpus, GR un-collapse is safe.

---

## 3. Framing-changing observations

### Observation 1 — the original z012 memo's §6 finding pre-dates `class_tag_for_category`

`EXAMPLE_YAML_DEFECT_INVESTIGATION.md §6` flagged 186 diverging Pack
canonical-renders and recommended "no bead filed yet — flagging here
for awareness." That recommendation was correct at the time but the
collapse layer (introduced sometime between the z012 investigation
and the nn0 refactor — `WrappedInstruction.class_tag_for_category`
landed as part of the 009 re-scoping, per the docstring at
`ScheduleCapture.py:380-393`) has since absorbed it. The z012 memo's
§6 should be marked RESOLVED-by-collapse, not "potential follow-up
bead."

### Observation 2 — Approach 9 is not viable today, but for a different reason than the original catalog

The original catalog (`Z012_ALTERNATIVE_FIXES_INVESTIGATION.md §3
Approach 9`) rejected Approach 9 because it broke 10 LRSwap tests
under the OLD (no-collapse) class_tag scheme. With
`class_tag_for_category` in place today, those 10 tests would no
longer fail because the LRSwap divergence is absorbed before identity
is built. So Approach 9's empirical-failure justification is stale.

Approach 9 still doesn't WORK as written, though, because it adds a
4th tuple element from raw category text — and the raw category
divergence (LRS vs LRSA/LRSB) STILL exists at the upstream
`build_idmap` / `build_id_to_category_per_iter` level. Approach 9
would re-surface the divergence the collapse layer was added to
hide. The fix would be to run the raw category through
`class_tag_for_category` before adding it to identity — but then it's
just a duplicate of the existing class_tag. Approach 11 (selective
un-collapse) is the surviving meaningful variant.

### Observation 3 — the brief's Approach 1 & 2 catalog still apply

The brief lists three candidate approaches for z012-collision:
Approach 1 (per-render emission ordinal), Approach 2 (source-side
discriminator threaded through lowering), Approach 3 (multi-valued
container). All three are equivalent in cost-and-blast-radius
terms to Approach 11 or worse:

- **Approach 1** = the original investigation's "Approach 3
  (occurrence-index)". Already proven to fail because occurrence
  itself is stream-order-dependent.
- **Approach 2** = exactly what Approach 11 implements at the
  collapse-layer site instead of plumbing through every lowering
  function. Same effect, smaller blast radius.
- **Approach 3** = the original investigation's "Approach 2
  (no-collapse at construction)". 1:1 identity→node assumption is
  load-bearing across the comparator; rewrite cost is large.

So the recommendation in the bead — Approach 11 — is the cheapest
viable disambiguator and the one that exploits svb1's collapse-layer
finding most directly.

---

## 4. Summary table

| Concern | Status | Bead | Notes |
|---|---|---|---|
| z012-tagging (LRSwap / Pack cross-build) | Empty under Approach A | none filed | Absorbed by `class_tag_for_category`. See §1. |
| z012-collision (state-register intra-capture) | Real, has cheap fix | `rocm-libraries-on0t` | Approach 11: selective un-collapse for GRInc / GR. See §2. |

