# Quad-cycle dispatch audit (rocm-libraries-o0ei) — REDO

Investigation memo for the order-dependent dispatch in
`_classify_edge_coverage` and `diagnose_missing_edge`. The original o0ei
memo (now-closed) recommended a 3-helper dispatch keyed on CMS-scheduler
category strings (`PackMFMA`, `CVTPack`, `MFMA`). That recommendation
materialized through `rocm-libraries-s5g1` (`_DISPATCH` table over
`(_ProducerRole, _ConsumerRole)`) and then `rocm-libraries-vmua`
(`gap_rules: dict[(InstructionShape, InstructionShape), List[GapRule]]`).
Both incarnations still resolve their dispatch keys via
`shape_of(node)` — and `shape_of()` reads `node.category` (a CMS-scheduler-
side string) FIRST, falling back to rocisa-class-derived data only when
no category is present.

> **Why this redo.** Verbatim from the user dispatch:
> *"it's basing those lookup and the naming based on CMS concepts. it
> need to be redone based on the underlying rocisa constraint. This need
> to keep working once the bead to extend to arbitrary schedules has
> landed."* The "bead to extend to arbitrary schedules" is
> `rocm-libraries-zq3` ("Extend validator coverage beyond 4 CMS bodies"),
> which broadens validator input from the four CMS bodies (ML-1, ML, NGL,
> NLL) to the full kernel — pre-loop preamble, post-loop epilogue, tail
> loops, prefetch loops. None of those regions carry CMS category strings;
> the dispatch as it stands today will silently degrade on every node it
> sees from outside the CMS scheduler.

> Status: investigation only in this commit. Implementation deferred to a
> follow-up bead — see §8. The original o0ei memo's date / commit
> reference is preserved for traceability: original audit was authored
> against `_classify_edge_coverage` at `CMSValidator.py:2916` (line
> numbers shifted by subsequent landings of s5g1 / vmua / 009).

---

## 1. Summary

`_classify_edge_coverage` (now `CMSValidator.py:3724`) and
`diagnose_missing_edge` (`CMSValidator.py:3361`) both run their quad-
cycle gap classification through `_dispatch_quad_cycle_check`
(`CMSValidator.py:2623`). The dispatch consults
`arch_profile.gap_rules[(p_shape, c_shape)]` where
`p_shape, c_shape = shape_of(p_node), shape_of(c_node)`.
`shape_of()` (`InstructionShape.py:154`) is implemented as:

```
if cat.startswith("Pack"):       # CMS-scheduler-assigned string
    inst_cat = _category(inst)
    if inst_cat is MFMA:        return MFMA_4x4
    if inst_cat is CVT_PACK:    return CVT_PACK
    if inst_cat is MIDDLE_PACK: return MIDDLE_PACK
    return ALU
if cat == "MFMA":                # CMS-scheduler-assigned string
    return MFMA_STANDARD
# Fallback: rocisa-derived
...
```

The first two discriminators are CMS-string-keyed. They route the same
PackMFMA rocisa instance to a 5-quad-cycle settle window
(`MFMA_4x4 → CVT_PACK`) only because the CMS scheduler tagged it
`PackA0` / `PackB0`. The same rocisa MFMAInstruction emitted by ANY
non-CMS source (a pre-loop preamble's lone v_mfma_f32_4x4x4_*, an asm-
captured tail loop) lands in `MFMA_STANDARD` instead — wrong family,
wrong cycle count, silently.

**The fragility the original memo described as "branch order in
`_classify_edge_coverage`" was not eliminated by s5g1 / vmua — it was
relocated.** Today it lives in `shape_of()`'s if-cascade
(`InstructionShape.py:178-225` — the full pre-fallback cascade,
covering both the Pack/MFMA early-returns at :182-197 and the
rocisa-instance fallback at :206-225): the `cat.startswith("Pack")`
branch MUST run before the `cat == "MFMA"` branch (a Pack-categorized
MFMAInstruction satisfies both pre-checks). And the entire if-cascade
is keyed on CMS scheduler labels that won't exist post-zq3.

**Top recommendation:** rewrite `shape_of()` to discriminate on rocisa-
observable instruction shape only — opcode family (e.g. v_mfma_4x4 vs
v_mfma_16x16 vs v_mfma_32x32; v_cvt_pk_bf16_f32 vs other VCvt; standard
VALU vs MFMA) — and keep `node.category` as a *hint* for the small
number of cases where rocisa cannot disambiguate by class alone (mostly
test fixtures where `rocisa_inst is None`). Most of this work is
**rocisa-side exposure**: §5 enumerates exactly what nanobind needs to
expose (`MFMAInstruction.variant`, `VCvtInstruction.cvtType`) before
the dispatch can be made fully rocisa-keyed. Without those exposures
we are stuck distinguishing 4x4 vs 16x16 MFMA via `getIssueLatency()`
(works) and v_cvt_pk_bf16_f32 vs other VCvts via `type(inst).__name__`
string match (works but couples to class naming).

The follow-up bead can land in ~120 LoC if rocisa exposures are in
place; ~60 LoC if it relies on the existing `getIssueLatency()` +
class-name discriminators. Either path eliminates `shape_of()`'s
CMS-string dependency and survives zq3 by construction.

---

## 2. Current dispatch (CMS-string-keyed)

### 2.1 The dispatch chain today

The validator's quad-cycle classification flows through three layers:

1. **`_dispatch_quad_cycle_check`** (`CMSValidator.py:2623`) — the
   shared entry point both `_classify_edge_coverage` and
   `diagnose_missing_edge` call. Computes
   `(p_shape, c_shape) = (shape_of(p_node), shape_of(c_node))` and
   looks up `profile.gap_rules.get((p_shape, c_shape))`. Returns one
   of `_PASSTHROUGH` / `TimingCheck` / `None`.
2. **`shape_of(node)`** (`InstructionShape.py:154`) — assigns one of
   16 `InstructionShape` enum values per node. **This is where the
   CMS-string dependency lives.** See §2.2 below.
3. **`gap_rules`** (`CMSValidator.py:490`, `_build_cdna4_gap_rules`) —
   the `(p_shape, c_shape) -> List[GapRule]` table. Keys on
   `InstructionShape` only; carries no CMS-side knowledge directly.

The gap rules table itself is rocisa-clean. The dispatch keys feeding
it are not.

### 2.2 `shape_of()`'s CMS-string dependency

`shape_of(node)` (`InstructionShape.py:154-239`) discriminates in two
phases. The phase boundary matters: the second phase only fires when
`rocisa_inst is None`, which is a test-fixture-only condition under
the current builders. Production captures always populate
`rocisa_inst` at append time, so production traffic is entirely served
by Phase 1. (zq3 captures will also carry `rocisa_inst` — see §3.1.)

**Phase 1 — rocisa-present cascade (`InstructionShape.py:178-225`).**
Fires for every node with `rocisa_inst is not None` (i.e. every
production node today, every zq3 capture node post-broadening, and
test fixtures that wire a real or stand-in rocisa instance).

| Step | Predicate | Result | Code lines |
|------|-----------|--------|------------|
| 1 | `cat.startswith("Pack")` AND `_category(inst) is MFMA` | `MFMA_4x4` | :182-185 |
| 2 | `cat.startswith("Pack")` AND `_category(inst) is CVT_PACK` | `CVT_PACK` | :186-187 |
| 3 | `cat.startswith("Pack")` AND `_category(inst) is MIDDLE_PACK` | `MIDDLE_PACK` | :188-189 |
| 4 | `cat.startswith("Pack")` (else branch — Pack-cat with non-MFMA/non-CVT/non-MIDDLE rocisa) | `ALU` | :193 |
| 5 | `cat == "MFMA"` (any rocisa class) | `MFMA_STANDARD` | :196-197 |
| 6 | `inst is not None` AND `_category(inst)` in `_CATEGORY_TO_SHAPE` (LR/LW/GR/SWAIT/SBARRIER/SNOP/SSETPRIO/SMEM/FLAT/VECTOR_STORE) | direct shape | :208-211 |
| 7 | `inst is not None` AND `_category(inst) is MFMA` (no Pack/MFMA category) | `MFMA_STANDARD` | :214-215 |
| 8 | `inst is not None` AND `_category(inst) is CVT_PACK` (no Pack category) | `CVT_PACK` | :216-217 |
| 9 | `inst is not None` AND `_category(inst) is MIDDLE_PACK` (no Pack category) | `MIDDLE_PACK` | :218-219 |
| 10 | `inst is not None` AND `_category(inst) is None` (rocisa class not in any bucket — SMov, SAdd, VXor, etc.) | `ALU` | :225 |

Steps 1-5 read `node.category` BEFORE consulting `node.rocisa_inst`.
Steps 6-10 are the rocisa-instance-driven fallback — they fire when no
Pack* / MFMA category short-circuit applied earlier in Phase 1.

**Phase 2 — rocisa-None category fallback (`InstructionShape.py:227-239`).**
Fires only when `rocisa_inst is None` (Phase 1's `if inst is not None:`
guard at :206 was false). Today this is reached only by test fixtures
that explicitly construct a node without a rocisa instance.

| Step | Predicate | Result | Code lines |
|------|-----------|--------|------------|
| 11 | `cat in {"LRA0".."LRB3","LWA","LWB"}` | `LR` / `LW` | :232-235 |
| 12 | `cat in {"GRA","GRB","GR"}` | `GR` | :236-237 |
| 13 | catch-all | `OTHER` | :239 |

The same `MFMAInstruction` rocisa instance produces:

- `MFMA_4x4` if `category=="PackA0"` (CMS scheduler tagged it as a
  TF32 pack chain member)
- `MFMA_STANDARD` if `category=="MFMA"` (CMS scheduler tagged it as
  a main-loop MFMA)
- `MFMA_STANDARD` if `category==""` AND the rocisa instance has a 4x4
  variant (post-zq3 case — falls through step 5 and lands in step 7,
  losing the 4x4 discrimination)

The third row is the zq3-introduced silent-degradation. See §3.

### 2.3 Carve-outs preserved by enum-distinctness

The s5g1 commit replaced the original four-branch `if/elif` dispatch
with `_DISPATCH: dict[(_ProducerRole, _ConsumerRole), helper]`, and the
vmua commit then re-keyed it on `(InstructionShape, InstructionShape)`.
The original memo's "Must run BEFORE" carve-outs were absorbed in three
places:

1. **PACK_MFMA vs MFMA disambiguation.** The s5g1 enum split
   `_ProducerRole.PACK_MFMA` from `_ProducerRole.MFMA`; the vmua shape
   split `MFMA_4x4` from `MFMA_STANDARD`. `shape_of()` step 1 (Pack*
   category + MFMA rocisa) routes to `MFMA_4x4`, step 5 (cat=="MFMA")
   routes to `MFMA_STANDARD`. The dispatch can no longer downgrade a
   PackMFMA → CVTPack edge to the standard finish-cycle rule because
   the producer never reaches the standard rule's key.
2. **CVT_PACK vs ALU disambiguation.** The s5g1 enum split
   `_ProducerRole.CVT_PACK` from `_ProducerRole.ALU`; the vmua shape
   split `CVT_PACK` from `ALU`. `shape_of()` step 2 routes
   v_cvt_pk_bf16_f32 producers to `CVT_PACK`; step 4 routes other
   Pack-categorized non-MFMA producers to `ALU`.
3. **PackMFMA carve-out from ALU.** `shape_of()` step 1 fires before
   step 4, so a Pack-categorized MFMAInstruction reaches `MFMA_4x4`
   instead of `ALU`. This is the same load-bearing-order property the
   original memo flagged — but it's now scoped to a 4-line if-cascade
   rather than a 50-line dispatch.

### 2.4 Where CMS strings come from

`node.category` originates from `TaggedInstruction.category`
(`ScheduleCapture.py:472`), which is *"determined at emission time by
the source list/module the instruction was popped from"*
(`ScheduleCapture.py:475`). The CMS scheduler emits each instruction
into a named bucket (`MFMA`, `PackA0`, `PackB1`, `LRA0`, `GRA`,
`SYNC`, `SNOP`, ...); the bucket name becomes the category string. The
default-side capture builder uses an analogous tagging
(`build_id_to_category_per_iter`, `ScheduleCapture.py:1017`).

**Both the CMS pipeline and the default-side capture builder are
*scheduler-side concepts*.** The category is a label about *which
scheduling slot the instruction was emitted from*, not about *what the
instruction is*. The same rocisa `MFMAInstruction` carries category
`"PackA0"` if emitted as part of a TF32 pack chain or `"MFMA"` if
emitted as a main-loop MFMA — same instruction, same hardware shape,
different category string because of *where it sits in the
scheduler's output*.

### 2.5 Test surface

`Tests/unit/test_dataflow_graph_register_gaps.py` (89 refs),
`test_arch_profile_gap_rule_table.py`,
`test_arch_profile_gap_rule_snapshot.py`, and
`test_dataflow_graph_register_gaps.py` import the helpers and
construct nodes via fixtures that DO populate `category`. None of the
existing test fixtures exercise a node where `category=""` AND
`rocisa_inst is not None` — i.e. no test fixture exercises the post-
zq3 shape today. This is itself a gap: the rocisa-only fallback paths
in `shape_of()` (steps 6-11) are exercised only by test fixtures that
deliberately set `rocisa_inst=None`, and not by any "rocisa instance
present, category absent" combination.

---

## 3. Why CMS-string-keyed is wrong post-zq3

### 3.1 What zq3 changes

`rocm-libraries-zq3` (P3, `open`) extends validator input from the
four CMS bodies to the full kernel:

> *"Today the validator's input is a `ScheduleCapture` of four bodies —
> ML-1, ML, NGL, NLL — which are the bodies the CMS scheduler operates
> on. Anything **outside** those four bodies is invisible to the
> validator: pre-loop (kernel preamble), post-loop (epilogue), other
> end / tail loops, peeled iterations, prefetch-only loops..."*

zq3's Phase-2 design has two candidate substrates:

- **CMS-path extension** — add `LoopBodyCapture`-like containers for
  PRE_LOOP / EPILOGUE / TAIL regions. These containers would need
  category-tagging logic for their instructions, but the *CMS
  scheduler doesn't operate on these regions* (zq3's own "Risks" §
  notes: *"if the CMS scheduler doesn't operate on those regions
  (just emits them), the asm path is the cleaner solution"*). So
  category strings on PRE_LOOP / EPILOGUE / TAIL nodes would have to
  be *invented* by zq3 — they wouldn't reflect any scheduler-side
  emission bucket. This invented tagging would be a parallel
  divergence to the cms-vs-default Pack-tagging divergence already
  documented in `EXAMPLE_YAML_DEFECT_INVESTIGATION.md` §6 and
  `Z012_ALTERNATIVE_FIXES_INVESTIGATION.md` §6.
- **Asm-path** — route everything through `assembly_to_timeline` and
  tag each event with its region. The asm path has *no scheduler-
  emission bucket* available; the tag is "what region the asm address
  falls in," NOT "what CMS bucket emitted this instruction." So
  `node.category` would be either empty or repurposed (`PRE_LOOP`,
  `EPILOGUE`, `TAIL_LOOP_3`, ...) — neither of which match any
  category string `shape_of()` discriminates on today.

In both cases, `shape_of()`'s `cat.startswith("Pack")` and
`cat == "MFMA"` branches cease to fire for nodes in the new regions.

### 3.2 What breaks in `shape_of()` post-zq3

zq3 captures carry `rocisa_inst` (asm-path emits via the rocisa
stream; CMS-extension-path constructs WrappedInstruction at append
time) but no Pack* / MFMA category string. So the rocisa-present cascade
of Phase 1 still fires — but Phase 1 steps 1-5 (the Pack/MFMA
category-keyed early-returns) all fall through, and the node lands in
Phase 1 steps 6-10 (the rocisa-instance-driven fallback). Per row:

| Today's production routing | Post-zq3 (no CMS category) routing | Result |
|----------------------------|------------------------------------|--------|
| PackMFMA → step 1 → `MFMA_4x4` | step 1 fails (no Pack cat); step 5 fails (no MFMA cat); MFMAInstruction reaches step 7 | every MFMA → `MFMA_STANDARD` (4x4 discrimination LOST) |
| CVTPack → step 2 → `CVT_PACK` | step 2 fails; VCvtPkF32toBF16 reaches step 8 | `CVT_PACK` (preserved via rocisa class) |
| MiddlePack → step 3 → `MIDDLE_PACK` | step 3 fails; PVCvtBF16toFP32 / VSubF32 / etc. reach step 9 | `MIDDLE_PACK` (preserved) |
| Other Pack (generic VALU with Pack* cat) → step 4 → `ALU` | step 4 fails; rocisa class is unregistered (`_category(inst) is None`) → step 10 at line :225 | `ALU` (preserved via the unregistered-rocisa → ALU fallback) |
| Main-loop MFMA → step 5 → `MFMA_STANDARD` | step 5 fails; reaches step 7 | `MFMA_STANDARD` (preserved by happy accident — 4x4 vs standard collision) |

The single regression is **the 4x4 vs standard MFMA discrimination
loss** (row 1). Step 7 maps every `MFMAInstruction` →
`MFMA_STANDARD` regardless of variant, with no way to detect a
4x4-variant MFMA from the `_category(inst)` lookup alone. A pre-loop
v_mfma_f32_4x4x4_* would route to the 3-quad-cycle standard finish
rule instead of the 1-quad-cycle 4x4 rule — over-strict by 2 cycles,
spurious failures.

The "Other Pack → ALU" row (row 4) survives by an artifact of the
current Phase 1 cascade: when `_category(inst)` returns None for an
unregistered rocisa class, line :225 returns `InstructionShape.ALU`
unconditionally. The OTHER catch-all at line :239 is reachable ONLY in
Phase 2 (`rocisa_inst is None`), which zq3 captures do not enter.
Documented here so a future change to the Phase 1 fallback (e.g.
returning OTHER instead of ALU for unregistered rocisa) doesn't
silently regress this row.

(In practice the rocisa-fallback path is rarely exercised in
production today because every production node carries a CMS category.
zq3's broadening exercises it as the dominant path.)

### 3.3 What breaks in the dispatch table itself

The `gap_rules: dict[(p_shape, c_shape), ...]` table is keyed on
`InstructionShape` only — no CMS-string surface in the keys
themselves. **The table is structurally fine.** The breakage is
upstream of the table, in `shape_of()`. Once `shape_of()` returns the
right shape for an arbitrary capture source, the table produces the
right rule.

The downstream consequence: the existing 5 carve-outs (MFMA_STANDARD,
MFMA_4x4, CVT_PACK → MFMA, MFMA_4x4 → CVT_PACK, ALU passthrough)
continue to fire for the existing CMS-categorized nodes. The new
nodes (pre-loop preamble's lone MFMA, an epilogue's CVT for store-
back, tail-loop MFMAs) silently miss the carve-outs unless `shape_of()`
classifies them rocisa-only.

### 3.4 The `cms_from_default` divergence as a leading indicator

`Z012_ALTERNATIVE_FIXES_INVESTIGATION.md` §6 documented that the CMS
scheduler and the default-side capture pipeline assign DIFFERENT
category strings to functionally-identical rocisa instances:

- The TF32 emul Pack chain on the CMS side is tagged `PackA0`,
  `PackA1`, `PackB0`, `PackB1`, `PackA3`, `PackB3` etc. by the macro
  expansion in `expand_cms_macro` (`ScheduleCapture.py:2186`).
- The same TF32 emul Pack chain on the default side is tagged
  `PACKA[0]`, `PACKB[0]`, ... by `build_id_to_category_per_iter`
  (`ScheduleCapture.py:1017`).
- The LRSwap / LRSA / LRSB tagging diverges similarly
  (`Z012_ALTERNATIVE_FIXES_INVESTIGATION.md` §7 second bullet).

The Z012 memo concluded that the divergence is real and that any
identity-or-dispatch refactor that depends on category-string equality
across capture sources is fragile. **zq3 introduces a third capture
source** (asm-path or PRE_LOOP/EPILOGUE/TAIL CMS extensions) which
will add its own category-tagging conventions, multiplying the
divergence surface.

The fix is the same fix Z012 recommended: dispatch on facts derivable
from the rocisa instruction itself, not on scheduler-assigned labels.

---

## 4. Proposed dispatch (rocisa-constraint-keyed)

### 4.1 Design principle

`shape_of(node)` must compute its result from `node.rocisa_inst`
alone. `node.category` may participate ONLY as a tiebreaker for cases
where `rocisa_inst is None` (pure synthetic test fixtures).

**Prerequisite: MFMA-family registry coverage.** The current
`_CLASS_NAME_TO_CATEGORY` registry (`InstructionCategory.py:111-227`)
binds `"MFMAInstruction" → MFMA` (line :156) but does NOT bind
`"MXMFMAInstruction"` or `"SMFMAInstruction"`. Production code uses
MXMFMA via `Tensile/Components/CustomSchedule/dispatch.py:330-331`
(added to the `mfma_classes` tuple via a `try: from rocisa.instruction
import MXMFMAInstruction` import). Under today's `shape_of()` an
MXMFMA node with `category=="MFMA"` lands in `MFMA_STANDARD` via
Phase 1 step 5, because step 5 ignores `_category(inst)`. Under
§4.3's rewrite — which keys solely on `_category(inst)` —
`_category(MXMFMA_instance)` returns None and the node would fall to
`InstructionShape.ALU`, a silent regression.

**Correction (supersedes the prior fixup of this section).** The
prior fixup of this memo claimed that adding `"MXMFMAInstruction" →
MFMA` to the registry was sufficient because MXMFMA already exposes
`getIssueLatency`. That premise is wrong. Verification of
`rocisa/rocisa/src/instruction/mfma.cpp` (234 lines total) shows two
`def("getIssueLatency", ...)` bindings only:
  - `mfma.cpp:157` — `MFMAInstruction.getIssueLatency`.
  - `mfma.cpp:229` — `SMFMAInstruction.getIssueLatency`.
The `MXMFMAInstruction` `nb::class_` block at `mfma.cpp:163-200` has
no `def("getIssueLatency", ...)` line. Verification of
`rocisa/rocisa/include/instruction/mfma.hpp` (861 lines total) shows
two `getIssueLatency` C++ overrides only:
  - `mfma.hpp:446` — `MFMAInstruction::getIssueLatency`.
  - `mfma.hpp:853` — `SMFMAInstruction::getIssueLatency`.
The `MXMFMAInstruction` struct at `mfma.hpp:455-720` declares no
`getIssueLatency` method. Therefore the docstring at
`CMSValidator.py:376` ("`MXMFMAInstruction` does not carry the
binding") is **correct, not stale**.

Consequence: routing MXMFMA into the §4.3 sketch's `inst_cat is
InstructionCategory.MFMA` branch with a registry entry alone is
broken. The branch's `getattr(inst, "getIssueLatency", None)` guard
returns `None` for an MXMFMA instance, the `<=2` test is skipped,
and the function falls past the early return into the
`InstructionShape.ALU` tail — the very regression the registry
edit was supposed to prevent.

**Resolution chosen: option (c) — document the residual
category-string dependency for MXMFMA only.** Add only
`"SMFMAInstruction" → MFMA` to `_CLASS_NAME_TO_CATEGORY` (SMFMA has
`getIssueLatency` exposed at `mfma.cpp:229`, so it works in the §4.3
sketch). MXMFMA continues to read from `node.category == "MFMA"` as
the **single residual CMS-string dependency**, scoped to one rocisa
class, until the rocisa-exposure follow-up adds an
`MXMFMAInstruction::getIssueLatency` C++ override and the matching
`def("getIssueLatency", ...)` binding. See §5.3 for the exposure
prerequisite and §8.1 for bead acceptance.

Why option (c) over the alternatives:
  - **(a) registry-only edit** — broken for MXMFMA per the
    correction above.
  - **(b) `isinstance(inst, (MFMAInstruction, MXMFMAInstruction,
    SMFMAInstruction))` in the §4.3 sketch with a class-name probe
    fallback for MXMFMA** — re-introduces a rocisa import into
    `InstructionShape.py` (the whole rationale for the
    `_CLASS_NAME_TO_CATEGORY` registry was to keep surrounding
    modules rocisa-import-free; see `InstructionCategory.py:23-47`
    module docstring). The class-name probe
    (`type(inst).__name__ == "MXMFMAInstruction"`) is the same
    fragility class as the category-string check it replaces.
  - **(c) explicit MXMFMA-only category fallback** — keeps the
    rocisa-exposure work explicit in §5 as a true prerequisite for
    full coverage, doesn't introduce class-name string checks, and
    is honest about what the §4.3 design can and cannot do today.
    The CMS-string dependency for MXMFMA is bounded to ONE class
    and tracked to closure by the rocisa-exposure follow-up.

### 4.2 The translation table

For each `shape_of()` step in §2.2, the rocisa-constraint replacement:

| Old (CMS-categorized) name + predicate | New (rocisa-constraint) name + predicate | Notes / rocisa exposure needed |
|----------------------------------------|------------------------------------------|---------------------------------|
| `MFMA_4x4` ← Pack* + MFMA rocisa | `MFMA_4x4_PASS` ← `MFMAInstruction` AND `getIssueLatency() <= 2` | Already works; `getIssueLatency` exposed (`mfma.cpp:157`). The `<=2` threshold is documented at `CMSValidator.py:393` and partitions the 4x4 family from 16x16/32x32/SMFMA cleanly. |
| `MFMA_STANDARD` ← cat=="MFMA" | `MFMA_STANDARD` ← `MFMAInstruction` AND `getIssueLatency() > 2` | Same exposure; the partition is total over MFMAInstruction. |
| `CVT_PACK` ← Pack* + VCvtPkF32toBF16 rocisa | `CVT_PACK` ← `isinstance(inst, VCvtPkF32toBF16)` (or class-name string match if rocisa import is undesired) | Class-name string match works today. **Or** add `VCvtInstruction.cvtType` to nanobind (currently NOT exposed; constructor takes it but field is not `def_rw`); see §5. |
| `MIDDLE_PACK` ← Pack* + MIDDLE_PACK rocisa | `MIDDLE_PACK` ← `_category(inst) is MIDDLE_PACK` | Already works; `InstructionCategory` lookup is class-name keyed and rocisa-import-free per `InstructionCategory.py:32-46`. |
| `ALU` ← Pack* (else) | `ALU` ← `inst` is a generic VALU/SALU op (not MFMA, not CVT_PACK, not MIDDLE_PACK, not LR/LW/GR/etc) | Already works; same `_category(inst)` lookup. The "Pack* else" branch is purely category-driven today and reduces to "rocisa class is not in any non-ALU bucket." |
| `LR` / `LW` / `GR` ← `inst is None` AND cat hint | `LR` / `LW` / `GR` ← `_category(inst) is LR/LW/GR` | Already works for non-None inst. The `inst is None` branch (test-fixture-only) MAY remain category-keyed; flag it as a fixture-only escape hatch. |

The translation deletes the `cat.startswith("Pack")` and `cat ==
"MFMA"` early-return branches in `shape_of()` and replaces them with a
single rocisa-class-driven discrimination chain.

### 4.3 Python sketch

```python
# Tensile/Components/InstructionShape.py — proposed rewrite

def shape_of(node) -> InstructionShape:
    """Return the InstructionShape of `node`.

    Discrimination is based exclusively on properties of
    `node.rocisa_inst` (the underlying rocisa instruction); CMS scheduler
    category strings are consulted ONLY when `rocisa_inst is None`
    (test-fixture-only escape hatch).

    Survives arbitrary capture sources (CMS scheduler, default-side
    capture, post-zq3 pre/post/tail-loop captures, asm-path captures)
    because the rocisa instance is the same regardless of which
    pipeline emitted it.
    """
    inst = getattr(node, "rocisa_inst", None)

    if inst is not None:
        inst_cat = _category(inst)
        # MFMA family discrimination via rocisa's own getIssueLatency.
        # The 4x4 family yields issueLatency in {1, 2}; standard
        # (16x16 / 32x32 / DGEMM / SMFMA) yields >= 4. Threshold 3
        # cleanly partitions per CMSValidator.py:393 docstring.
        # Prerequisite registry patch: SMFMAInstruction must be bound
        # to InstructionCategory.MFMA in `_CLASS_NAME_TO_CATEGORY` so
        # this branch fires for SMFMA; MFMAInstruction is already
        # registered (InstructionCategory.py:156); getIssueLatency is
        # exposed on both (mfma.cpp:157, :229).
        if inst_cat is InstructionCategory.MFMA:
            get_il = getattr(inst, "getIssueLatency", None)
            if get_il is not None and get_il() <= 2:
                return InstructionShape.MFMA_4x4
            return InstructionShape.MFMA_STANDARD
        # Residual MXMFMA fallthrough — option (c) per §4.1. MXMFMA
        # is NOT registered in `_CLASS_NAME_TO_CATEGORY` and does NOT
        # expose `getIssueLatency` in nb (mfma.cpp:163-200 has no
        # def("getIssueLatency", ...); mfma.hpp:455-720 declares no
        # such method). Until the rocisa-exposure prerequisite (§5.3)
        # adds both, MXMFMA reads `node.category == "MFMA"` as the
        # single residual CMS-string dependency; it routes to
        # MFMA_STANDARD unconditionally (the M=4 case is not produced
        # for MXMFMA in current emission). Remove this block once the
        # exposure lands and add MXMFMAInstruction to the registry.
        if (getattr(node, "category", "") or "") == "MFMA":
            return InstructionShape.MFMA_STANDARD
        if inst_cat is InstructionCategory.CVT_PACK:
            return InstructionShape.CVT_PACK
        if inst_cat is InstructionCategory.MIDDLE_PACK:
            return InstructionShape.MIDDLE_PACK
        # Direct shape for non-MFMA non-CVT non-MIDDLE buckets
        # (LR / LW / GR / SWAIT / SBARRIER / SNOP / SSETPRIO / SMEM /
        # FLAT / VECTOR_STORE).
        if inst_cat is not None:
            direct = _CATEGORY_TO_SHAPE.get(inst_cat)
            if direct is not None:
                return direct
        # rocisa instance whose class is not registered in
        # InstructionCategory: treat as ALU (matches `_is_alu_producer`
        # semantics — `_category(inst) is None` falls through the
        # `_NON_ALU_CATEGORIES` check and the predicate returns True).
        return InstructionShape.ALU

    # rocisa_inst is None — test-fixture-only escape hatch. Falls back
    # to the category hint. Production code paths always carry a
    # rocisa_inst (LoopBodyCaptureBuilder.append constructs the
    # WrappedInstruction at append time).
    cat = getattr(node, "category", "") or ""
    _LDS_LIKE = ("LRA0", "LRA1", "LRA3", "LRB0", "LRB1", "LRB3",
                 "LWA", "LWB")
    _GR_LIKE  = ("GRA", "GRB", "GR")
    if cat in _LDS_LIKE:
        return InstructionShape.LR if cat.startswith("LR") else InstructionShape.LW
    if cat in _GR_LIKE:
        return InstructionShape.GR
    return InstructionShape.OTHER
```

Naming reframe (per the brief's instruction to phrase predicates at
the rocisa-observable level rather than CMS scheduling pattern):

- `MFMA_4x4` documents what it is at the hardware level: a 4x4
  matrix-instruction family producer with a 1-quad-cycle finish
  window. The CMS-side name "PackMFMA" reflects only that the CMS
  scheduler grouped it with the surrounding pack chain; the
  *underlying constraint* is "MFMA producer with M=4 → small finish
  window."
- `CVT_PACK` documents that the producer is a `v_cvt_pk_bf16_f32`-
  family instruction whose write-to-VGPR settles after 2 quad-cycles
  before a downstream MFMA can read the converted bf16 value. The
  CMS-side name "CVTPack" reflects only that the CMS scheduler
  emitted it as part of a pack chain; the *underlying constraint* is
  "v_cvt_pk_bf16_f32 producer → MFMA reader of the converted output
  needs 2 quad-cycles."
- The `MFMA_4x4 → CVT_PACK` 5-quad-cycle gap reframes from "PackMFMA
  → CVTPack settle window" to "wide-register-write MFMA producer
  feeds CVT-pack consumer reading the same accumulator slot →
  accumulator settle = 5 quad-cycles." The constraint is on the
  hardware pipeline (accumulator visibility), not on the CMS pack-
  chain pattern.

### 4.4 What still needs the category, and why

Two cases legitimately need `node.category` even after the rewrite:

1. **`rocisa_inst is None` test fixtures.** These are test-only stubs
   that synthesize a node with a category string but no rocisa
   instance. The `if inst is not None:` block above returns early in
   production; the category-fallback at the bottom of `shape_of()`
   handles the synthetic case. Documented as fixture-only. zq3
   captures all carry rocisa instances (asm-path goes via the rocisa
   stream by definition; CMS-extension-path constructs
   WrappedInstruction at append time).
2. **The DTL m0 setter** (category=`"GRA"` / `"GRB"` but rocisa class
   is `SMov` / `SAddU32`). `_is_alu_producer`
   (`CMSValidator.py:2473-2525`) handles this case via
   `_NON_ALU_CATEGORIES` (`:2519`) — the rocisa class
   (SMov, SAddU32) is not registered in `_CLASS_NAME_TO_CATEGORY`, so
   `_category(inst)` returns None, `None not in _NON_ALU_CATEGORIES`,
   the predicate returns True. Under §4.3's sketch the same node
   walks: `_category(SMov_instance)` returns None →
   `inst_cat is None` → falls past the `if inst_cat is not None:`
   guard at line :382 of the sketch (the `_CATEGORY_TO_SHAPE` /
   MFMA / CVT_PACK / MIDDLE_PACK branches all require `inst_cat is
   not None`) → reaches `return InstructionShape.ALU` at line :390
   of the sketch. Same shape (`ALU`), reached via the same
   "unregistered rocisa class" path the predicate uses. The DTL m0
   setter's `category="GRA"` / `"GRB"` is never consulted in either
   walk.

The rewrite preserves both cases without reading category strings in
production paths.

### 4.5 Carve-outs in the new dispatch

The 5 carve-outs from §2.3 are preserved as follows:

- **PACK_MFMA vs MFMA disambiguation.** Both shapes still exist
  (`MFMA_4x4` vs `MFMA_STANDARD`); discrimination is now via
  `getIssueLatency()` instead of category prefix. Functionally
  identical: a CMS-categorized PackMFMA has `getIssueLatency() <= 2`,
  so it lands in `MFMA_4x4`; a CMS-categorized main-loop MFMA has
  `getIssueLatency() >= 4`, so it lands in `MFMA_STANDARD`. Same
  partition, derived from rocisa instead of category.
- **CVT_PACK vs ALU disambiguation.** `_category(inst) is CVT_PACK`
  partitions VCvtPkF32toBF16 from generic ALU. The
  `_CLASS_NAME_TO_CATEGORY` registry binds VCvtPkF32toBF16 →
  CVT_PACK (`InstructionCategory.py:176`). No ordering required.
- **PackMFMA carve-out from ALU.** A Pack-categorized MFMAInstruction
  reaches the `_category(inst) is MFMA` branch (step 1 of §4.3's
  sketch) before any ALU classification. Same load-bearing-order
  property, but now the order is "MFMA-shaped before generic VALU"
  — a property of the discrimination order in `shape_of()`'s rocisa
  cascade, NOT of any CMS scheduler tagging.

### 4.6 Symmetry with `_is_alu_producer`

`_is_alu_producer` (`CMSValidator.py:2473`) and the proposed
`shape_of()` rewrite need to agree on discrimination order so a
later collapse (`_is_alu_producer = shape_of(node) is ALU`) is a
mechanical refactor.

**Decision: rocisa-first is the canonical order.** Both predicates
shall consult `_category(inst)` on the rocisa instance *before*
inspecting `node.category`. The `node.category` value participates
ONLY when `rocisa_inst is None` (test-fixture-only escape hatch).
This matches the design principle in §4.1 and the §4.3 sketch.

`_is_alu_producer`'s actual code today is **category-first** by
textual order: lines :2508-2516 short-circuit on
`cat.startswith("Pack")` and `cat == "MFMA"` BEFORE consulting
`rocisa_inst`; only when neither category check fires does the code
fall through to :2517-2522 to read `inst = getattr(producer,
"rocisa_inst", None)` and check `_category(inst) in
_NON_ALU_CATEGORIES`. The docstring at :2492-2494 ("classify by
category_first (Pack* / PackMFMA → ALU)") **accurately describes
this order** — it is not misleading. The classifier today is
authoritatively category-first.

The §4.6 decision (rocisa-first canonical) is therefore a
**code-change** going forward, not a docstring fix. Once
`shape_of()` adopts rocisa-first per §4.3, a sibling refactor of
`_is_alu_producer` to mirror that order would update both the code
AND the docstring together. A docstring-only change today would
make the docstring lie about the code.

A natural follow-up: rewrite `_is_alu_producer` to use `shape_of()`
once `shape_of()` is rocisa-keyed. The two predicates collapse to
`shape_of(node) is InstructionShape.ALU`, and the docstring is
updated as part of that rewrite. Out of scope for this bead but
called out in §6.

### 4.7 Performance angle

`shape_of()` is invoked twice per dispatched edge in
`_dispatch_quad_cycle_check` (`CMSValidator.py:2623`). Under the
rewrite, the only edges that take the nanobind round-trip are MFMA
producers (the `inst_cat is InstructionCategory.MFMA` branch in
§4.3's sketch calls `inst.getIssueLatency()`). Every other shape
discriminator stays Python-side (`_category(inst)` is a
`type(inst).__name__` dict lookup; no nanobind call).

MFMA edges are O(K) per body where K is the number of MFMAs (small
relative to total edges — a typical body has 8-32 MFMAs and hundreds
of LR/LW/ALU/CVT edges). The nanobind round-trip cost is also
amortizable: `shape_of()` results can be memoized per-node if a
profile pass shows the call is hot. No memoization is proposed for
the initial bead — measure first. Expectation: zero detectable
delta on validator wall-clock, since the existing
`mfma_finish_cycles_for` (`CMSValidator.py:355`) already calls
`getIssueLatency()` once per MFMA gap check; the rewrite adds at
most one extra call per MFMA edge participating in dispatch.

If a profile shows `getIssueLatency()` is hot, the cleanest fix is
the §5.1 `MFMAInstruction.variant` exposure (`def_rw`) — direct
field access has no `getMFMAIssueLatency` template instantiation
cost. That's an Option C cleanup, not an Option A blocker.

---

## 5. Rocisa-exposure prerequisites

The §4 design works *today* without new rocisa exposures, by relying
on `getIssueLatency()` (already exposed) for MFMA family
discrimination and `_category(inst)` (Python-side class-name lookup,
no nanobind work needed) for CVT_PACK / MIDDLE_PACK / ALU
discrimination. **The dispatch can land without any rocisa changes.**

That said, two rocisa-side exposures would make the dispatch more
robust and cleaner:

### 5.1 `MFMAInstruction.variant` (priority: high)

Cross-reference: `ISA_GAP_GENERALIZATION_AUDIT.md` §6 Q5 (load-bearing).

Current state: `MFMAInstruction::variant` is a `std::vector<int>` C++
field (`mfma.hpp:71`) holding `[M, N, K, B]` for the matrix
dimensions. The constructor accepts it (`mfma.cpp:144`); the field
itself is NOT exposed via `def_rw`. The validator currently infers
"4x4-family vs standard" via `getIssueLatency() <= 2`, which works
for every CDNA variant rocisa knows how to emit but is indirect:
"the issue latency is small *because* M=4."

What exposure unlocks: direct family discrimination
(`inst.variant[0] == 4 → MFMA_4x4`) instead of cycle-count proxy.
Eliminates the "threshold 3 cleanly partitions" assertion in
`mfma_finish_cycles_for` (`CMSValidator.py:355-395`); reduces it to a
single integer comparison. Also unlocks finer-grained shape
discriminators (16x16 vs 32x32) which the audit-memo §6 Q2 flags as
deferred.

Cross-reference to `rocm-libraries-4t0`: 4t0 ("Add OperationShape
enum to rocisa for swap and no-dst opcodes") is the closest related
work but does NOT cover MFMA variant exposure — it's about VSwapB32
and SCC-only opcodes. So 5.1 is a separate exposure item not covered
by 4t0.

Cross-reference to `rocm-libraries-qzpa`: qzpa ("Add uniform
RegisterContainer factory_for entry point to rocisa") covers
RegisterContainer factory exposure; also unrelated to MFMA variant.

LoC estimate: ~10 LoC in `mfma.cpp` (one `def_rw` line per MFMA
class — MFMA, MX, S — plus regenerate the bindings).

### 5.2 `VCvtInstruction.cvtType` (priority: medium)

Current state: `VCvtInstruction::cvtType` is a `CvtType` enum field
(`cvt.hpp:30`) holding `CvtType::CVT_PK_F32_to_BF16` for
v_cvt_pk_bf16_f32. The constructor accepts it (`cvt.cpp:37`); the
field is NOT exposed via `def_rw`. The validator currently infers
"is this a v_cvt_pk_bf16_f32" via class-name string match
(`type(inst).__name__ == "VCvtPkF32toBF16"` indirectly, through the
`_CLASS_NAME_TO_CATEGORY` registry).

What exposure unlocks: structural discrimination
(`inst.cvtType is CvtType.CVT_PK_F32_to_BF16`) instead of class-name
string match. Robust to future renames of the C++ class. Allows
lookups by enum identity rather than class identity.

LoC estimate: ~4 LoC — one `def_rw` line on `VCvtInstruction`, plus
regenerating the `CvtType` enum binding (already exposed at
`enum.cpp` if present; nanobind enums are auto-bound when used in
constructor signatures, but `def_rw` access requires explicit
binding).

### 5.3 What rocisa already exposes (and what's missing)

What's already accessible from Python today:

- `MFMAInstruction.getIssueLatency()` — Yes
  (`mfma.cpp:157`; C++ override at `mfma.hpp:446`). Sufficient for
  4x4 vs standard discrimination.
- `SMFMAInstruction.getIssueLatency()` — Yes
  (`mfma.cpp:229`; C++ override at `mfma.hpp:853`). Sparse-MFMA
  path always uses mi_divisor=4 per the C++ override.
- `MFMAInstruction.a` / `.b` / `.acc` / `.acc2` — Yes
  (`mfma.cpp:152-155`). The four operands are accessible as
  `RegisterContainer` shared_ptrs, allowing register-type inspection.
- `RegisterContainer.regType` — Yes (`container.cpp:543`).
  Returns `'v'` / `'s'` / `'acc'` / `'m'` / `'vcc'` / `'scc'`
  string. Lets us read the destination register type to confirm
  "MFMA writes to AccVgpr" structurally.
- `type(inst).__name__` — Always available (Python primitive). Used
  by `_category(inst)` (`InstructionCategory.py:230`) for the
  class-name registry lookup. Robust to nanobind class hierarchy
  (the binding generates `__name__` from the C++ class name).

What's **missing** and required for full §4 coverage:

- `MXMFMAInstruction.getIssueLatency()` — **No.** Verified at
  `mfma.cpp:163-200` (the `nb::class_<MXMFMAInstruction>` block has
  no `def("getIssueLatency", ...)` line) and `mfma.hpp:455-720`
  (the `MXMFMAInstruction` struct declares no such method).
  **Medium-priority prerequisite.** Without this exposure, MXMFMA
  must use the option-(c) category-fallback in §4.1 / §4.3.
  Closing this gap requires:
    1. Add a `getIssueLatency() const override` body inside the
       `MXMFMAInstruction` struct in `mfma.hpp` (parallel to
       `mfma.hpp:446` / `:853`; choose the appropriate
       `getMFMAIssueLatency<...>` template instantiation for MX
       data types).
    2. Add a matching `.def("getIssueLatency",
       &rocisa::MXMFMAInstruction::getIssueLatency)` line inside
       the `nb::class_<MXMFMAInstruction>` block in `mfma.cpp`
       (parallel to `mfma.cpp:157` / `:229`).
    3. Add `"MXMFMAInstruction" → InstructionCategory.MFMA` to
       `_CLASS_NAME_TO_CATEGORY` in `InstructionCategory.py`.
    4. Remove the residual `node.category == "MFMA"` MXMFMA
       fallthrough block in §4.3's sketch.
  Cross-reference: `ISA_GAP_GENERALIZATION_AUDIT.md §6 Q5`
  (load-bearing for §5.1 variant exposure; the
  `MXMFMAInstruction.getIssueLatency` exposure is a sibling task in
  the same nb-binding module).

The §4 design needs none of `variant` / `cvtType` /
`MXMFMAInstruction.getIssueLatency` to **function** today (option (c)
covers MXMFMA via category fallback). The MXMFMA exposure is the only
prerequisite to fully remove the residual CMS-string dependency. The
`variant` / `cvtType` exposures are quality-of-life improvements that
eliminate proxy discrimination (`getIssueLatency() <= 2` standing in
for "M=4").

### 5.4 Cross-reference summary

- `rocm-libraries-4t0` (`open`): rocisa OperationShape enum for VSwap
  / no-dst opcodes. Unrelated to this dispatch redesign.
- `rocm-libraries-qzpa` (`open`): rocisa RegisterContainer factory_for
  entry point. Unrelated.
- `rocm-libraries-009` (closed, `InstructionCategory.py` shipped):
  centralizes the rocisa-class-name → category map; **already
  in-place and consumed by §4's design**. The §4 rewrite leans on
  `_category(inst)` from this module.
- `ISA_GAP_GENERALIZATION_AUDIT.md` §6 Q5: the MFMAInstruction.variant
  exposure question. **Load-bearing** for §5.1.

---

## 6. Refactor options (revised)

The original memo's Option 1 / 2 / 3 / 4 catalog was constructed for
the **pre-009, pre-s5g1, pre-vmua world** where the dispatch was a
4-branch if/elif chain in `_classify_edge_coverage`. That world no
longer exists. The dispatch is already a table; the question now is
*where in the existing table-driven pipeline does the CMS-string
dependency live, and how do we excise it*.

The relevant axis is therefore: **how invasive is the rewrite of
`shape_of()`?** Three options:

### 6.1 Option A — Minimal surgery (recommended)

Rewrite `shape_of()` per §4.3. Delete `_LDS_LIKE` / `_GR_LIKE`
fixture-fallback constants if no fixture relies on them (verify via
test-suite run). Keep the rest of `InstructionShape.py` and all of
`gap_rules` untouched.

LoC: ~50 LoC delta in `InstructionShape.py` (~80 LoC current →
~50 LoC after). Tests: re-run `test_instruction_shape*.py` (if any),
`test_arch_profile_gap_rule_table.py`,
`test_dataflow_graph_register_gaps.py`. Expected delta: zero failures
on existing fixtures (all currently set both `category` and
`rocisa_inst`); add ~20 LoC of new fixture tests covering the
`category="" + rocisa_inst is not None` zq3 case.

### 6.2 Option B — Rewrite + collapse `_is_alu_producer`

Option A plus: rewrite `_is_alu_producer` (`CMSValidator.py:2473`)
as `return shape_of(producer) is InstructionShape.ALU`. Removes ~50
LoC of duplicated discrimination logic (the same Pack-vs-non-Pack-
vs-rocisa-class cascade as `shape_of()`).

LoC: ~50 LoC delta in `InstructionShape.py` + ~−50 LoC in
`CMSValidator.py`. Net ~0 LoC, but 1-source-of-truth for the
discrimination chain. Tests: `_is_alu_producer` is referenced at
`CMSValidator.py:3448` (cross-subiter ALU artifact branch); same call
shape, same return type, no signature change. Test impact: re-run
`test_dataflow_graph_register_gaps.py`'s ALU exemption tests.

### 6.3 Option C — Rocisa-side exposures (deferred)

Add `MFMAInstruction.variant` and `VCvtInstruction.cvtType` to
nanobind (§5.1, §5.2). Rewrite `shape_of()` to use the structural
exposures instead of `getIssueLatency()` / class-name string match.
This is two beads' worth of work (rocisa-side bindings + Python-side
`shape_of()` rewrite) and produces marginal-cycles improvement over
Option A.

LoC: ~14 LoC in rocisa C++ + ~30 LoC delta in `InstructionShape.py`
(slightly cleaner than Option A because direct enum comparisons
replace `getIssueLatency() <= 2`). Tests: rocisa rebuild + same
Python test surface as Option A.

### 6.4 Re-ranking

The original memo ranked Option 4 (Producer-classify-once + tiny
dispatch) > Option 1 (Single dispatch table) > Option 2 (match) >
Option 3 (visitor). Under the new CMS-vs-rocisa framing, the original
options are **all equivalent at the dispatch layer** (the dispatch is
already a table — that work is done). The discriminator is **how the
dispatch keys are derived**.

Re-ranked:

1. **Option A (minimal surgery)** — does the work in one bead, lands
   in ~50 LoC, requires no rocisa changes, survives zq3. **Recommended.**
2. **Option B (collapse `_is_alu_producer`)** — extra refactor on
   top of A; nice cleanup but not required for zq3 survival.
3. **Option C (rocisa exposures)** — better long-term but blocks on
   rocisa-side work; defer until the small wins stack up (also
   unblocks ISA_GAP_GENERALIZATION_AUDIT §6 Q5's deferred items).

---

## 7. Compatibility with zq3 and 009

### 7.1 zq3 survival

Under §4's design, `shape_of(node)` works for any node carrying a
rocisa instance, regardless of capture source. zq3's two candidate
substrates both produce nodes with rocisa instances:

- **CMS-extension path** — new `LoopBodyCapture`-like containers for
  PRE_LOOP / EPILOGUE / TAIL would still construct `WrappedInstruction`
  via `LoopBodyCaptureBuilder.append`, populating
  `WrappedInstruction._rocisa_inst`. The new containers have no need
  to invent category strings because `shape_of()` no longer reads
  them in production paths.
- **Asm-path** — `assembly_to_timeline` constructs nodes from the
  rocisa stream directly. Same `WrappedInstruction` shape; same
  `_rocisa_inst` field; `shape_of()` produces the same shape it
  produces for the CMS-path equivalent instruction.

The `rocisa_inst is None` fallback at the bottom of §4.3's sketch
fires only for synthetic test fixtures, never for production zq3
captures.

### 7.2 Integration with 009

`rocm-libraries-009` (closed) shipped `InstructionCategory.py`'s
`_CLASS_NAME_TO_CATEGORY` registry, the central rocisa-class →
`InstructionCategory` map. §4's design **leans on this registry** —
the proposed `shape_of()` consults `_category(inst)` for
classification instead of category strings. So the redo is *strictly
in line with* 009's centralization:

- 009 centralized the rocisa→category mapping.
- This bead pushes the dispatch to consume that centralized mapping
  *instead of* CMS scheduler labels.
- After both, `shape_of()` is a thin wrapper over `_category(inst)` +
  `getIssueLatency()` — fully rocisa-derivable.

009 alone did NOT eliminate the CMS-string layer from the dispatch
keys. It centralized one half of the discrimination (rocisa-class →
category) but `shape_of()` still consults `node.category` first. This
bead closes that loop.

### 7.3 Compatibility with the existing gap_rules table

The `gap_rules: dict[(InstructionShape, InstructionShape), List[GapRule]]`
table keys on `InstructionShape` enum values, NOT on category strings.
**The table is unchanged.** §4's rewrite changes only how shapes are
*derived* from nodes, not what shapes the table uses for keys.

---

## 8. Recommendation

**File a follow-up bead for Option A (minimal surgery).** Implement
the §4.3 `shape_of()` rewrite. Land before zq3 starts so that zq3's
Phase-2 design has a guarantee that `shape_of()` works on its new
capture surfaces.

### 8.1 Follow-up bead spec

Title: *"Replace CMS-string-keyed `shape_of()` discrimination with
rocisa-derivable predicates"*

Body: see §4 of this memo. Acceptance:

- Replace the full pre-fallback cascade at `InstructionShape.py:178-225`
  (Pack/MFMA early-returns at :182-197 plus the rocisa-fallback at
  :206-225) with a single rocisa-class-driven discrimination chain
  (§4.3 sketch). The fixture-only `rocisa_inst is None` tail at
  :227-239 is preserved.
- Preserve the `rocisa_inst is None` fallback for test fixtures.
- **Prerequisite registry patch (§4.1):** add
  `"SMFMAInstruction" → InstructionCategory.MFMA` to
  `_CLASS_NAME_TO_CATEGORY` in `InstructionCategory.py:111-227`.
  Without this, SMFMA nodes regress from `MFMA_STANDARD` (today's
  behavior via category-string short-circuit) to `ALU` under the
  rewrite. SMFMA already exposes `getIssueLatency` (`mfma.cpp:229`)
  so it works in §4.3's MFMA branch.
- **MXMFMA handling deferred under option (c) per §4.1.** The §4.3
  sketch carries a single residual `node.category == "MFMA"`
  fallthrough scoped to MXMFMA, with a TODO marker referencing
  §5.3. Closing the residual requires a separate rocisa-exposure
  follow-up bead (see §5.3) that adds:
    1. `MXMFMAInstruction::getIssueLatency` C++ override in
       `mfma.hpp` (parallel to `mfma.hpp:446` / `:853`).
    2. `.def("getIssueLatency", ...)` binding in `mfma.cpp` inside
       the `nb::class_<MXMFMAInstruction>` block (parallel to
       `mfma.cpp:157` / `:229`).
    3. `"MXMFMAInstruction" → InstructionCategory.MFMA` registry
       entry.
  Once that bead lands, the MXMFMA fallthrough block is removed
  from `shape_of()` in a 1-line follow-up commit.
- All existing tests pass byte-for-byte (every existing fixture sets
  both `category` and `rocisa_inst` consistently; the rewrite returns
  the same shape for them).
- Add new fixture tests for the `category="" + rocisa_inst is not None`
  case with each of: 4x4 MFMAInstruction, 16x16 MFMAInstruction,
  SMFMAInstruction, VCvtPkF32toBF16, generic VALU, MIDDLE_PACK,
  etc. Verify each routes to the same shape its CMS-categorized
  counterpart routes to. MXMFMA fixture tests use the existing
  `category="MFMA"` convention until the rocisa-exposure follow-up
  lands.
- Run the full pytest suite (skip the slow
  `test_MatrixInstructionConversion.py` per the agent-side directive).
  Pytest count delta: at most ±5.

Dependencies: 009's registry is shipped; this bead patches it
in-place (the 1 new SMFMA entry above) as part of the same change.
The MXMFMA rocisa-exposure follow-up is a sibling bead (deferred,
medium priority).

Effort: ~50 LoC in `InstructionShape.py` + 1 LoC in
`InstructionCategory.py` + ~40 LoC tests = 91 LoC. Single PR,
~3 hours of focused work.

Ordering vs zq3: this bead should land BEFORE zq3 Phase 2 starts. zq3
Phase 1 (the inventory pass) is unblocked by this bead's status. zq3
Phase 2's Timeline-construction design will benefit from this bead's
clarification that `shape_of()` is rocisa-only — it removes a
constraint zq3's design would otherwise have to work around (or
silently bake in the divergence).

### 8.2 What this commit DOES land

- This memo (`Tensile/Components/QUAD_CYCLE_DISPATCH_AUDIT.md`),
  rewritten end-to-end.

No code changes. Investigation only.

---

## 9. Open questions for the user

1. **Procedural — `_is_alu_producer` docstring update timing.** The
   §4.6 decision is that rocisa-first is the canonical order across
   `shape_of()` and `_is_alu_producer`. `_is_alu_producer`'s actual
   code today is **category-first** by textual order
   (`CMSValidator.py:2508-2516` short-circuits on
   `cat.startswith("Pack")` and `cat == "MFMA"` BEFORE consulting
   `rocisa_inst` at :2517-2522). The docstring at :2492-2494
   ("classify by category_first") matches the code accurately and
   is therefore NOT misleading today. Switching to the rocisa-first
   canonical order is a code change, not a docstring fix; the
   docstring will be updated as part of that future code change
   (sibling refactor of `_is_alu_producer` once `shape_of()` adopts
   rocisa-first per §4.3). A docstring-only edit today would make
   the docstring lie about the code.

2. **Should the test-fixture-only `rocisa_inst is None` fallback
   stay?** §4.3 keeps it for backwards-compat with synthetic test
   fixtures. An alternative is to delete it and require all fixtures
   to construct a real `WrappedInstruction(inst)` with a rocisa
   instance. Cost: ~5-10 fixtures across `test_dataflow_graph_*` need
   to construct real (or stand-in real-shaped) rocisa instances.
   Default if no answer: keep the fallback; flag it as fixture-only
   in the docstring.

3. **`MFMAInstruction.variant` exposure — go or no-go?** §5.1
   recommends adding `def_rw("variant", ...)` to the rocisa nanobind
   binding. Option A works without it (uses `getIssueLatency()` as
   proxy); Option C unlocks it. The user has visibility into rocisa-
   side maintenance preferences (some teams prefer minimal nanobind
   surface, others prefer full field exposure). If exposed, also
   include `MXMFMAInstruction.variant` and `SMFMAInstruction.variant`
   for symmetry. Default if no answer: defer to Option C (file as
   sibling bead, do not bundle with the dispatch rewrite).

4. **`VCvtInstruction.cvtType` exposure — go or no-go?** §5.2 same
   shape as Q3. Smaller win; class-name discrimination via the
   `_CLASS_NAME_TO_CATEGORY` registry is robust and already in use.
   Default if no answer: defer indefinitely; not worth the rocisa
   churn for a class-name-string match alternative that already
   works.

5. **Should this bead block on zq3 Phase 1's region-inventory output?**
   zq3 Phase 1 will surface "what region kinds exist beyond the
   obvious five" — there may be region-specific category-tagging
   conventions that zq3 wants to introduce (e.g. `PRE_LOOP_VALU`,
   `EPILOGUE_STORE`, etc.) for its own purposes. If zq3 plans to
   use category strings for region tagging, this bead's recommendation
   to demote category from production discrimination still holds —
   but the new region-tagging convention should be documented as
   *informational only* (failure messages, diagnostics), NOT as
   dispatch input. Default if no answer: this bead does not block on
   zq3 Phase 1; the recommendation is to land before zq3 Phase 2 so
   zq3 doesn't need to design around the CMS-string-keyed dispatch.

6. **`_is_alu_producer` collapse (Option B) — bundle with this bead
   or sibling?** Option B is clean cleanup but not zq3-required.
   Bundling adds ~50 LoC of additional change to the same PR;
   sibling-beading keeps the dispatch rewrite minimal. Default if no
   answer: sibling bead, file after Option A lands.

7. **Are there other CMS-string-keyed dispatches lurking elsewhere?**
   This memo focused on `_classify_edge_coverage` /
   `diagnose_missing_edge` per the original o0ei scope. A grep for
   `cat.startswith(` and `category ==` across `Tensile/Components/`
   would surface any siblings. Default if no answer: not in scope for
   this bead; file as a sibling investigation if any are found.
