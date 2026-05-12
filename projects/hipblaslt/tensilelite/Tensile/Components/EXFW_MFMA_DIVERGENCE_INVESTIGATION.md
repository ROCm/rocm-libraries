# EXFW — Whole-kernel MFMA-count divergence between CMS and default captures

**Bead:** `rocm-libraries-exfw`
**Date:** 2026-05-12
**Status:** Investigation only. No production code changes.
**Trigger surface:** `4up4 + F3` layered, post-cherry-pick onto
`users/alvasile/validator_long_term_plans` tip `03ca4ef0d30`.
**Failing test:** `projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_prologue_capture.py::test_whole_kernel_useplrpack_cms_matches_both_defaults`.

## TL;DR

The 16-MFMA divergence is **mechanism (a) — real CMS-vs-default emission
asymmetry** of pack-MFMAs (TF32 emulation `v_mfma_f32_4x4x4_16b_bf16`
instructions categorized as `PackA3`/`PackB3`). The CMS macro emits
plrIdx=3 prefetch-pack chains inside the main_loop body; the default
SIA3-shadow emission does not. Both kinds of emissions live under the
same rocisa class (`MFMAInstruction`) and therefore both pass the
`_data_flow_ids` filter at `compare_graphs` entry, but only the
CMS-side has them in ML-1 / ML.

The 4up4 design memo (`EMISSION_ORDINAL_DESIGN.md` §2.7 third bullet)
explicitly anticipated this: "A schedule that emits an extra v_mfma on
the CMS side because of a carve-out. The CMS side has an extra
ordinal; the comparison sees identity-set divergence... **This is the
desired behavior — emit-count divergence IS a real defect.**"

The exfw briefing's "categorically different from leftover-walk"
classification needs a partial correction: the 16 instructions ARE
pack-content (Pack-categorized), but their rocisa class is
`MFMAInstruction`, so the divergence summary's `'MFMA': 16` count
appears even though the leftover walk only emitted Pack-categorized
content. This is what the F3 author hinted at when they observed the
test "still fails ... independent of the leftover-pack walk." It is
the **same defect family** as flpk/xbi0 (pack-MFMA accounting
asymmetry between CMS and default captures), but a different
instance: flpk/xbi0 surfaced as default-side over-tagging from the
leftover walk; exfw surfaces as default-side under-emission of CMS-only
pack content. F3 closed the over-tagging arm; exfw is the remaining
under-emission arm.

**Recommended fix: fold this into rocm-libraries-71hw Approach A** (a
true non-CMS reference build under `UseCustomMainLoopSchedule=0`),
which by construction emits the same pack chain layout the production
non-CMS path emits. Until 71hw Approach A lands, no clean tactical fix
exists that doesn't either special-case pack-MFMAs (CMS-coupled,
violates the user's standing rule) or paper over the divergence by
weakening the `_data_flow_ids` filter (architectural regression). The
`xfail` route is also viable as a temporary marker until 71hw lands.

---

## §1. The failing test (Q1)

### What the test does

`test_whole_kernel_useplrpack_cms_matches_both_defaults`
(`projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_prologue_capture.py:279-352`)
builds two `KernelWriter` captures via `_build_capture(isa_infrastructure,
force_use_plr_pack=True/False)`. Each capture populates BOTH
`writer._capture_context.default` and `writer._capture_context.cms`
under the SAME kernel emission run (the default-side via the SIA3
shadow path inside `_loopBody`; the CMS-side via the deferred
`build_cms_four_part_capture` expansion in `kernelBody`). The test
then compares `(default_cap, cms_cap)` whole-kernel for each
`UsePLRPack` setting via `_explicit_validate`
(`test_prologue_capture.py:137-162`).

### subject vs reference

In `_explicit_validate(default_cap, cms_cap)`:
- `ref_graph = build_dataflow_graph(default_cap)` — DEFAULT is reference.
- `subj_graph = build_dataflow_graph(cms_cap)` — CMS is subject.

The error
> in subject but not reference: 16 identities (`{'MFMA': 16}`)

means **CMS has 16 MFMA-class identities the default does not**.

### Kernel configuration

`_CMS_CONFIG` at `test_prologue_capture.py:68-80`:
- `OperationType=GEMM`, `DataType=S` (F32), `F32XdlMathOp=X` (TF32 emulation).
- `MatrixInstruction=[16, 16, 32, 1, 1, 4, 4, 2, 2]`.
- `DepthU=32`, `PrefetchGlobalRead=2`, `PrefetchLocalRead=1`,
  `LocalReadVectorWidth=4`, `UseCustomMainLoopSchedule=1`.
- `UsePLRPack` is forced TRUE in this test invocation.

This is a **registered CMS-eligible** schedule shape. Both the default
SIA3 shadow and the CMS macro expansion are exercised (the test is not
synthetic).

### Failure shape repro

```
CaptureConsistencyError: compare_graphs: data-flow node identity sets
differ. in subject but not reference: 16 identities ({'MFMA': 16});
first 3: [(0, 'v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:...], v[74:75],
v[vgprValuA_X0_I0+0:...], v[vgprValuA_T0_I0+0:...]', 0), ...]
```

The `(loop_index=0, render, emission_ordinal=0)` shape is consistent.
All 16 identities have `emission_ordinal=0`: 8 at `loop_index=0` (ML-1)
and 8 at `loop_index=1` (ML).

---

## §2. Forensic findings (Q2)

The bead's framing about "16 distinct emission ordinals at the same
loop_index" is **wrong**. Forensic instrumentation (a temporary
`test_zzz_forensic_exfw.py` test that wraps `_build_capture` and dumps
the captures) shows:

### Per-body LoopBodyCapture MFMA-category accounting

Both default and CMS report **identical** numbers at the
`LoopBodyCapture.instructions` (category-MFMA) level:

| Body | DEFAULT MFMA insts | CMS MFMA insts | distinct renders | distinct identities |
|---|---|---|---|---|
| PRO    | 0  | 0  | 0  | 0  |
| ML-1   | 48 | 48 | 48 | 48 |
| ML     | 48 | 48 | 48 | 48 |
| NGL    | 48 | 48 | 48 | 48 |
| NLL    | 48 | 48 | 48 | 48 |

All emission ordinals are 0 (no within-body render-text collisions
among MFMA-category instructions). At the `category=="MFMA"` level the
two sides are identical.

### Per-body category breakdown — the asymmetry source

```
DEFAULT ML-1[0]: 150 insts;
  top cats=[('MFMA', 48), ('PackA0', 20), ('PackB0', 20),
            ('GRIncA', 9), ('GRIncB', 9), ('GRA', 8)]
CMS     ML-1[0]: 194 insts;
  top cats=[('MFMA', 48), ('PackA0', 20), ('PackB0', 20),
            ('PackB3', 20), ('PackA3', 20), ('SYNC', 10)]
```

CMS's ML-1 contains **40 extra Pack-categorized instructions** (20
PackA3 + 20 PackB3) that DEFAULT's ML-1 does not. The same is true
for ML. NGL and NLL match on both sides.

### Cross-body MFMA-rocisa-class node count in the graph

`build_dataflow_graph` (`CMSValidator.py:1900-1901`) admits a node
into `nodes_by_identity` when its rocisa class is in
`_DATA_FLOW_CATEGORIES = {LR, LW, GR, MFMA}`. The category-string
("PackA3") is irrelevant here — only the rocisa class
(`MFMAInstruction`) is consulted via `_category(node.rocisa_inst)`.

| loop_index | DEFAULT MFMA-class nodes | CMS MFMA-class nodes |
|---|---|---|
| -1 (PRO) | 8  | 8  |
|  0 (ML-1) | 56 | 64 |
|  1 (ML)   | 56 | 64 |
|  2 (NGL)  | 64 | 64 |
|  3 (NLL)  | 64 | 64 |
| **total** | **248** | **264** |

The 16-identity gap is `(64-56)*2 = 16` at the ML-1 + ML rows.

The distinct MFMA-rocisa-class render set per side at ML-1:
- DEFAULT distinct renders: 56 (= 48 main MFMA + 8 cvt-MFMA-in-pack
  shared with CMS).
- CMS distinct renders: 64 (= 48 main MFMA + 8 shared cvt-MFMA + 8
  CMS-only PackA3/PackB3 cvt-MFMA).
- `CMS \ DEFAULT renders = 8`; `DEFAULT \ CMS renders = 0` per body.

### Per-node category of the CMS-extras

For each of the 16 CMS-only identities, the node's `tagged_inst.category`
is one of `PackA3` (8 of them) or `PackB3` (8). Specifically:

```
loop=0 ord=0 category=PackA3 render=v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:...]
loop=0 ord=0 category=PackA3 render=v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+4:...]
loop=0 ord=0 category=PackA3 render=v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:...]
loop=0 ord=0 category=PackA3 render=v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+4:...]
loop=0 ord=0 category=PackB3 render=v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+0:...]
loop=0 ord=0 category=PackB3 render=v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_T0_I0+4:...]
loop=0 ord=0 category=PackB3 render=v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+12:...]
loop=0 ord=0 category=PackB3 render=v_mfma_f32_4x4x4_16b_bf16 v[vgprValuB_X0_I0+4:...]
... (same 8 renders repeat at loop=1)
```

These are exactly the F32X TF32-emulation pack-MFMAs documented in
`ScheduleCapture.py:378-386`: *"F32X TF32 emulation MFMAs in the pack
path are real `MFMAInstruction` objects but are categorized as
`PackA{u}`/`PackB{u}`. Treating them as `cls='MFMA'` in the identity
tuple causes them to appear as missing main-loop MFMAs in compare_graphs
when the two captures see different counts of pack-MFMAs."*

The exfw failure is exactly the scenario that 380-line docstring
warns about, surfaced now because:
- 4up4 dropped `class_tag` from identity (`(class_tag, render, ...)` →
  `(loop_idx, render, ord)`) — pack-MFMAs are no longer disambiguated
  from main-loop MFMAs at the identity level.
- F3 deleted the leftover-pack walk that previously injected
  default-side `PackA3`/`PackB3` content into ML-1/ML to match CMS's
  shape.

---

## §3. Mechanism trace (Q3)

### CMS-side emission of `PackA3`/`PackB3` MFMAs in main_loop

The CMS macro is built in `customMainLoopSchedule`
(`projects/hipblaslt/tensilelite/Tensile/Components/CustomSchedule/dispatch.py`).
Per the docstring around `:185-205`, the macro emits ALL of
`mfmaCode[miIndex]` plus all the per-`miIndex` scheduled `Module`s
across all CMS keys (LRA/LRB/PackA{u}/PackB{u}/GRA/GRB/...). Pack
emissions enter via `emit_instructions` (`dispatch.py:270-289`) under
the `\\usePLR == 1` guard for `LRA{lastIter}/LRB{lastIter}/LRSA/LRSB`
and unconditionally for other keys. Tagging is uniform: `category=k`
where `k` is the InstStream key (e.g., `PackA3`, `PackB3`). At
`expand_cms_macro(macro, useGR=1, usePLR=1, useGRInc=1, useLoop=1)`
(`ScheduleCapture.py:2257-2374`) used for the main_loop expansion in
`build_cms_four_part_capture` (`ScheduleCapture.py:2459-2469`), all
guards pass and the full pack chain is emitted into `main_loop[0]`.

### Default-side SIA3 shadow emission of pack content

The default-side capture is the SIA3 shadow built inside `_loopBody`
(`KernelWriter.py:4450-4529`). At each iteration `u`,
`_makeSubIterSchedule` is invoked with `pack=pack[packIdx]` and
`packPre=packPre[packPreIdx]` for **only that iteration's PLR slot**.
`pack[packIdx]` and `packPre[packPreIdx]` are reset to empty `Module()`
after each iteration (`KernelWriter.py:4533-4534`):

```python
pack[packIdx] = Module()
packPre[packPreIdx] = Module()
```

So the shadow capture sees PackA0/PackB0 (the `packIdx` for the
current iteration's PLR) but NOT PackA3/PackB3 (which would be the
plrIdx=3 slot). The plrIdx=3 pack content lives in `pack[3]` and is
naturally consumed only by NGL/NLL paths (`pack` index 3 is the
prefetch-side pack consumed at the next macro tile boundary).

CMS, however, accumulates pack chains into the macro across all
plrIdx values because its scheduling is global — `customMainLoopSchedule`
walks `PackCodeAAllIters` / `PackCodeBAllIters` for all `u` and feeds
them to `scheduleInst` per `miIndex`, tagging by per-iteration
PackA{u}/PackB{u} category.

### What changed at F3

Pre-F3, the `KernelWriter.py:4571-4630` leftover-pack walk re-injected
unconsumed `pack[*]`/`packPre[*]` content into the SIA3 shadow's
final builder, including the plrIdx=3 content, **so the default-side
ML-1/ML capture matched CMS's** in pack composition. The walk was
buggy in two ways (xbi0 same-id double-append, flpk per-iter
cross-tagging), so F3 deleted it. The deletion was correct *under
71hw Approach A's premise* (the default side stops being a shadow and
becomes a real non-CMS kernel build, which would have the production
non-CMS pack layout naturally). Under the current shadow-based default
capture, the deletion leaves the asymmetry exposed.

### Edge formation — does this affect downstream comparison?

The 16-identity divergence trips `compare_graphs` at the entry-time
identity-set check (`CMSValidator.py:3427-3463`) BEFORE edge-by-edge
comparison runs. Edge-comparison effects are not a separate signal
here — the test fails at the entry gate.

---

## §4. Mechanism classification (Q4)

**Mechanism (a) — Real CMS-vs-default emission divergence.** Both
captures compute valid LoopBodyCaptures from their respective build
paths; both feed into `build_dataflow_graph` correctly. The 16
PackA3/B3 cvt-MFMAs are physically emitted into the CMS-side
main_loop body; they are NOT physically emitted into the default-side
main_loop body. The `_data_flow_ids` filter (rocisa-class-derived,
not category-derived) admits both as MFMA-class identities. Default
has 0 of these renders; CMS has 8 per body. The graph faithfully
represents that asymmetry.

This is NOT mechanism (b) (capture pipeline asymmetry) — the
captures are accurate snapshots of what each side actually emits.

This is NOT mechanism (c) (validator-side logic divergence) — the
filter and identity construction are symmetric across the two graphs.

This is NOT mechanism (d) — the asymmetry has a clean upstream
explanation in the CMS macro builder vs. SIA3 shadow scheduler design
gap.

### Why the bead briefing's classification was off

The briefing said "These are MFMA computational instructions, not
Pack instructions. The leftover walk only emitted Pack* content.
Therefore the divergence cannot be in the leftover walk." The first
clause (rocisa-class MFMA) is true; the second clause (categorized as
MFMA) is false — the 16 are categorized as `PackA3`/`PackB3` but the
underlying rocisa class is `MFMAInstruction`. The leftover walk
**did** emit pack content of this exact shape (per `flpk`'s table at
`FLPK_CROSS_TAGGING_INVESTIGATION.md:93-94`: "post_loop/PackA3=40,
post_loop/PackB3=20"). With F3's deletion of the walk, those Pack-
class-with-MFMA-rocisa instances stopped flowing into the default
shadow — and that is precisely what exfw measures.

So exfw is in the same defect family as flpk/xbi0 (CMS-vs-default
pack-MFMA accounting asymmetry through the SIA3-shadow path). It is
not in a different file region — it is the absence of the file region
that F3 just deleted. The reviewer's "categorically different
mechanism" classification was based on the assumption that F3's
deletion was unconditionally correct under all current pipeline
states; that assumption holds **under 71hw Approach A**, but that
approach has not yet landed.

---

## §5. Recommended fix approach (Q5)

Three principled paths. The user must weigh in between them.

### §5.1 Fix Path 1 (recommended): fold under 71hw Approach A

**Where the change lives:** `KernelWriter.py` whole-kernel emission
plus capture wiring (the 71hw work, not yet implemented).

**What invariant it establishes:** The default-side capture is built
from a SECOND, isolated, real non-CMS kernel build with
`UseCustomMainLoopSchedule=0` — not from a SIA3 shadow inside the CMS
build. Both captures are then snapshots of EMITTABLE+RUNNABLE
kernels, and the pack-MFMA accounting on each side is the production
asymmetry between CMS and non-CMS scheduling, not an artifact of the
shadow path.

**CMS-coupling rule-out:** none — Approach A is the architectural
move that *eliminates* CMS-shape leakage into the default side.

**Architectural trade-off:** doubles per-kernel build cost during
validation (the "Approach A doubles per-build cost" note in
`2LZD_INVESTIGATION.md:781`). Recorded as accepted in 2lzd's decision
log (`2LZD_INVESTIGATION.md:886-888`).

**Why this is the principled fix:** the entire shadow-capture
architecture was scaffolding to make a synthetic mirror that looked
"comparable" to CMS. Every defect surfaced so far in the
shadow-vs-CMS framing (xbi0, flpk, vybd/F3, exfw, and arguably the
two still-XFAIL tests in `test_dataflow_graph_emission_ordinal.py`)
has a common root: the shadow is forced into a body shape the SIA3
scheduler would not naturally produce. Approach A removes the source.

**What the post-A behavior looks like for the test:**
`UseCustomMainLoopSchedule=0` natively does NOT consume PackA3/B3 in
ML-1/ML (those plrIdx=3 packs land in NGL/NLL where they belong on
both sides). Both captures' main_loop bodies have the same MFMA-class
identity sets. compare_graphs passes. The test would also need to
move from "force `UsePLRPack` flag at prologue time" to a more
natural setup, but that's a test-side adjustment.

### §5.2 Fix Path 2: temporary `xfail` until 71hw lands

**Where the change lives:** `test_prologue_capture.py:279-352`, add a
`pytest.mark.xfail(reason="exfw — fold under 71hw Approach A; CMS
emits PackA3/B3 cvt-MFMAs in main_loop that SIA3-shadow does not")`.

**What invariant it establishes:** explicit acknowledgment that the
test is pinning a property the current architecture cannot satisfy.
Sister to the two existing XFAILs in
`test_dataflow_graph_emission_ordinal.py:376-471` (which also block
on shadow-vs-CMS divergences that 71hw will remove).

**CMS-coupling rule-out:** none — `xfail` is a test-marker, not a
fix.

**Architectural trade-off:** does not block the validation-branch
clean count; documents the gap; depends on 71hw landing to remove the
marker.

**When this is the right call:** if 71hw's timeline is short enough
that a temporary marker is preferable to bigger fixes. This is the
status quo's natural extension — F3's verification table already
notes the same test is "out of scope for vybd F3." Marking it xfail
formalizes that.

### §5.3 Fix Path 3: don't take this path — listed only to rule out

A category of fixes the user's standing rules forbid:

- **Adding `PackA*`/`PackB*` exclusions to `_data_flow_ids`** would
  mean the validator pretends the 8-per-body cvt-MFMAs aren't real
  data-flow producers, weakening the cross-graph identity check on
  every kernel. That defeats the purpose of the post-4up4 design.
- **Special-casing pack-MFMAs in identity construction** (e.g.,
  re-introducing class_tag for the `Pack` category only) would
  re-couple identity to CMS-shape categories, exactly what
  `EMISSION_ORDINAL_DESIGN.md` §3 argued against.
- **Filtering cvt-MFMAs in `compare_graphs` only on the CMS side**
  introduces CMS-coupling in the validator filter, violating
  "no CMS-shaped category checks anywhere in the data-flow filter".
- **Adjusting the test to ignore the 16 MFMAs** without understanding
  the cause (which we now do) hides a real asymmetry.

These are listed only to confirm they were considered and rejected.

### Summary table

| Path | Where | CMS-coupled? | Lands when | User decision needed? |
|---|---|---|---|---|
| §5.1 fold under 71hw A | `KernelWriter` capture wiring | no | with 71hw | yes — confirm timing |
| §5.2 temporary xfail | the test only | no | now | yes — accepts marker |
| §5.3 (rejected) | various | yes / yes / yes / no but masks | n/a | no |

**Default recommendation if forced to pick one: §5.2 now + §5.1 with 71hw.**

---

## §6. Test surface (Q6)

Once §5.1 lands (71hw Approach A), the regression-pinning assertion
shape is:

> **Whole-kernel CMS-vs-(true non-CMS) MFMA-class identity sets are
> equal for any registered F32X TF32 GEMM schedule that exercises
> PrefetchLocalRead > 0.**

Concrete shape:

```python
# In test_prologue_capture.py (or a new test_capture_pipeline_checks.py
# section), after both captures are produced from REAL builds (not
# shadow):
g_default = build_dataflow_graph(default_cap)
g_cms     = build_dataflow_graph(cms_cap)
default_mfma_ids = {n.identity for n in g_default.nodes.values()
                    if _category(n.rocisa_inst) is InstructionCategory.MFMA}
cms_mfma_ids     = {n.identity for n in g_cms.nodes.values()
                    if _category(n.rocisa_inst) is InstructionCategory.MFMA}
assert default_mfma_ids == cms_mfma_ids, (
    f"Whole-kernel MFMA-class identity sets differ: "
    f"default-only={len(default_mfma_ids - cms_mfma_ids)}, "
    f"cms-only={len(cms_mfma_ids - default_mfma_ids)}. "
    f"This includes both real main-loop MFMAs and TF32-emulation "
    f"pack-MFMAs; both must be present on both sides under Approach A."
)
```

The current `_explicit_validate` already pins this transitively (any
identity-set divergence trips `compare_graphs`'s entry gate). After
71hw the test passes by virtue of the pre-existing assertion;
`_explicit_validate` becomes the regression test. No new test is
required — the existing
`test_whole_kernel_useplrpack_cms_matches_both_defaults` becomes the
pin.

What does NEED to happen: the assertion in the test today is
correctly written; it fails because the property doesn't hold yet.
Removing the failure means landing the underlying fix, not changing
the assertion.

If §5.2 (xfail) is taken as an interim, the test marker should
include a STRICT `xfail` that becomes `xpass`-failing once §5.1 is
done — that way the marker self-removes when 71hw lands and someone
forgets to clean it up.

---

## §7. Test-suite impact (Q7)

### Currently-passing tests on the validation branch (post-F3)

**No impact** from EITHER §5.1 or §5.2 to the F3 baseline (which
already passes 1005 / 3 skipped / 1 xfailed). `4up4 + F3` brings the
exfw test back to failing; that is the bead's surface.

§5.2 would change the count to: same passing count, exfw becomes
xfailed (or xpassed-then-failing once §5.1 lands).

§5.1 (71hw Approach A) would land alongside its own test changes that
re-baseline the entire validation suite around real non-CMS reference
captures; that is 71hw's scope, not exfw's.

### XFAIL-marked tests in `test_dataflow_graph_emission_ordinal.py`

The two XFAILs (`test_real_kernel_per_render_counts_match`,
`test_real_kernel_per_ordinal_logical_instruction_matches`) per F3's
table are both **shadow-vs-CMS divergences** in different categories
(SYNC/SWaitCnt/SBarrier counts). They are part of the same family
(architectural shadow path leaking into the comparison) but in
DIFFERENT instruction classes. They are NOT closed by exfw's fix and
not affected by it; they will be closed by the same 71hw Approach A
move.

### Other tests

The forensic instrumentation found no other test that:
- Reads the 16 specific MFMA identities directly.
- Uses `_data_flow_ids` against a CMS+default capture pair.
- Has hard-coded counts of MFMA nodes per body that would shift.

The full pytest run `python -m pytest tensilelite/Tensile/Tests/unit/
--ignore=tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py`
on `4up4 + F3` shows exactly the one failure (exfw) plus the two
xfails (per F3's verification table). That accounting is unchanged by
this investigation.

---

## §8. References

- `EMISSION_ORDINAL_DESIGN.md` §2.7 (anticipates this exact scenario:
  "A schedule that emits an extra v_mfma on the CMS side because of a
  carve-out... This is the desired behavior — emit-count divergence IS
  a real defect").
- `EMISSION_ORDINAL_DESIGN.md` §2.8 (kernel-writer-shared-source
  invariant: "two builds that consume the same kernel-writer
  source-module references will see identical per-render emission
  orders" — exfw is exactly the case where the two builds DON'T
  consume the same source-module reference because the CMS macro
  walks all PackCodeAAllIters[u] but SIA3 only consumes pack[packIdx]
  per-iter).
- `EMISSION_ORDINAL_DESIGN.md` §3.2 (rationale for dropping class_tag,
  including the pack-MFMA case — explicitly accepts that pack-MFMA
  divergences will surface as identity-set divergences).
- `VYBD_F3_IMPLEMENTATION.md` §"Architectural alignment with
  rocm-libraries-71hw Approach A" — states the shadow framing's
  obsolescence under Approach A.
- `VYBD_F3_IMPLEMENTATION.md` §Verification "Residual failure" — F3
  acknowledged the exfw failure as out-of-scope and "independent of
  the leftover-pack walk."
- `FLPK_CROSS_TAGGING_INVESTIGATION.md` (sister bead — same
  pack-MFMA-accounting family, different surface).
- `XBI0_CAPTURE_DEFECT_INVESTIGATION.md` (sister bead — also closed
  by F3).
- `2LZD_INVESTIGATION.md` §6, §"Approach A — true non-CMS reference
  build" (the 71hw decision log; documents the cost vs. correctness
  trade-off and records Approach A as the chosen path).
- `ScheduleCapture.py:378-393` (`class_tag_for_category` docstring on
  pack-MFMAs — the inline explanation of why this case is delicate).
- `CMSValidator.py:3387-3463` (`compare_graphs` entry-time
  identity-set check, the actual failure site).
- `KernelWriter.py:4400-4534` (SIA3 shadow `_loopBody` emission, the
  default-side under-emission source).
- `Components/CustomSchedule/dispatch.py:185-308` (CMS macro
  construction, the CMS-side over-emission source).
