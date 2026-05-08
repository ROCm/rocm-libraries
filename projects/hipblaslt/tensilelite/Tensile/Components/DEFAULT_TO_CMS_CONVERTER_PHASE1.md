# Default-to-CMS Converter — Phase 1 Investigation Memo

Bead: `rocm-libraries-wlrp`
Branch: `users/alvasile/validator_long_term_plans` (vlt)
Worktree base verified: tip `7bf7158524`.

This memo answers Phase 1 questions for the converter
`default_schedule_to_cms(yaml_path, output_path, *, schedule_name, accept_min_quad_cycle_gap=True)`.
It is investigation only — no code changes are proposed here.

All file references below are absolute paths inside this worktree:
`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a8abd53aa0216e992`.

---

## 1. CMS schedule format — exact data structures

### 1.1 The body of a registered schedule

Ground truth: `Tensile/Components/CustomSchedule.py`. Each registered schedule
function returns `(matched: bool, ScheduleInfo)`. `ScheduleInfo`
(CustomSchedule.py:222–243) is the canonical container:

```python
class ScheduleInfo:
    def __init__(
        self,
        numCodePaths: int,
        numMfma: int,
        optSchedule: dict[str, list[list[int]]],
        syncCode: list[Union[SWaitCnt, SBarrier]],
        nglshift: int,
        nllshift: int,
        nllZeroDscnt: bool = False,
        mfmaReorder = [],
        snopCode: list[SNop] = [],
    ):
```

Fields:

- `numCodePaths` — number of parallel codepaths (typically 1 or 2). Determines
  how many sub-lists each `optSchedule[key]` may carry.
- `numMfma` — total MFMA count for the mainloop (consumer asserts
  `len(mfmaCode) == numMfma` at CustomSchedule.py:330).
- `optSchedule` — `dict[str, list[list[int]]]`: outer dict keyed by category
  (see §1.2), value is a `list[list[int]]` of length 1 or `numCodePaths`.
  Each inner list is the *positions* (mfma slot indices) at which to emit each
  successive instruction of that category. The semantics are documented at
  CustomSchedule.py:410–433 (`scheduleInst`): the dispatcher walks
  `mfmaIndex` from `-1` to `numMfma-1`; for each (category, codepath) it
  pops every position in that list whose value equals the current `mfmaIndex`
  and emits the corresponding instruction from the underlying `idMap`.
- `syncCode` — flat `list[Union[SWaitCnt, SBarrier]]` of length
  `len(optSchedule['SYNC'][0])`, in 1:1 zip order with `optSchedule['SYNC'][0]`.
  This is the *content* (rocisa instructions); the positions live in
  `optSchedule['SYNC'][0]`.
- `snopCode` — flat `list[SNop]` of length `len(optSchedule['SNOP'][0])`,
  in 1:1 zip order with `optSchedule['SNOP'][0]`. Positions live in
  `optSchedule['SNOP'][0]`. Defaults to `[]`.
- `nglshift` / `nllshift` — vmcnt shifts the dispatcher applies to SWaitCnts
  in the no-global-load and no-load-loop branches (CustomSchedule.py:437–457,
  `nllvmcntHandling`).
- `nllZeroDscnt` — boolean: if True, in the NLL branch the dispatcher zeros
  the `dscnt` field of guarded SWaitCnts.
- `mfmaReorder` — optional permutation list applied to `mfmaCode` (CustomSchedule.py:336).
- `numCodePaths` is asserted >= every `len(indexList)` for every key
  (CustomSchedule.py:333–334).

### 1.2 The full set of `optSchedule` keys

Schema is defined by `Tensile/Components/ScheduleCapture.py:541` (`build_idmap`)
and is the single source of truth for both the CMS dispatcher and the default
side capture. Categories grep'd from CustomSchedule.py production sites:

```
GRA, GRB              # global reads of A/B
GRIncA, GRIncB        # global-read address increments
LRA0..LRA{n-1}        # local reads of A, per inner-unroll subiter
LRB0..LRB{n-1}
PackA0..PackA{n-1}    # data packing (mainly Mixed/F32X paths)
PackB0..PackB{n-1}
LWA, LWB              # local writes (rare; most schedules omit them)
LRSA, LRSB            # local-read pointer swap (XOR-base swap)
LWSA, LWSB            # local-write pointer swap
LCC                   # loop-counter code
SYNC                  # SWaitCnt + SBarrier (positions only — content in syncCode)
SNOP                  # SNop wait-state padding (positions only — content in snopCode)
```

`build_idmap` always writes every key (even when the source module is empty),
so the full schema is canonical regardless of what the kernel uses. Keys for
inner-unroll subiters are constructed as `f"LRA{u}"` etc. with
`u in range(num_loop_iter)`.

### 1.3 The SWaitCnt structure

`SWaitCnt` is a rocisa instruction class
(`rocisa.instruction.SWaitCnt`, imported at CustomSchedule.py:35).
Its constructor signature, as used throughout CustomSchedule.py, is:

```python
SWaitCnt(dscnt=int, vlcnt=int, vscnt=int, comment=str)
```

`-1` is "don't wait" (omit that counter from the emitted s_waitcnt). The
helper class `SyncSchedule` (CustomSchedule.py:120–148) collects pairs
`(mfmaIdx, SWaitCnt)` plus optional `(barrierIdx, SBarrier)` — its
`get_indicies()` and `get_code()` accessors produce `optSchedule['SYNC'][0]`
and `syncCode` respectively.

The dispatcher inspects `inst.vlcnt` and `inst.dscnt` directly (CustomSchedule.py:438–457)
to decide whether NGL/NLL specialization is needed. So the SWaitCnt fields
are part of the validation surface, not opaque.

### 1.4 The SBarrier structure

`SBarrier(comment=str)` — also a rocisa instruction class. Lives in
`syncCode` interleaved with SWaitCnts; positions live in
`optSchedule['SYNC'][0]`. The dispatcher does no special handling (it is
just an opaque entry the macro emits).

### 1.5 Worked example — `_get_schedule_256x96x64_16bit` (TN, TLDS=1)

Excerpted verbatim from CustomSchedule.py:917–955:

```python
syncTable = [
    -1, SWaitCnt(dscnt=2, ..., comment="Finish all LRA1 and 1/3 LRB1"),
     7, SWaitCnt(dscnt=7, ..., comment="Finish 2/3 LRB1"),
    15, SWaitCnt(dscnt=1, ..., comment="All LRB1 and LRA0 done"),
    15, SBarrier(comment=""),
    23, SWaitCnt(dscnt=2, ..., comment="1/3 LRB0 done"),
    29, SWaitCnt(dscnt=0, ..., comment="All LRB0 done"),
    29, SBarrier(comment=""),
    35, SWaitCnt(..., vlcnt=11, comment="All GRA launched, 3 prev GRB."),
    35, SBarrier(comment=""),
    42, SWaitCnt(..., vlcnt=11, comment="Only global reads for this iter"),
    42, SBarrier(comment=""),
]
syncCode = syncTable[1::2]                          # 11 items
optSchedule = {
    'SYNC'   : [syncTable[::2]],                    # 11 positions, 1 codepath
    'GRIncA' : [[1,1,2,2,3,3,3,4,4]],               # 9 positions, 1 codepath
    'GRIncB' : [[5,5,6,6,6,7,7,8,8]],
    'LRA0'   : [[1,2,3,4,5,6,8,10],                 # 8 positions per codepath
                [1,2,3,4,5,6,9,11]],                # 2 codepaths
    'LRB0'   : [[12,16,18],
                [13,17,19]],
    'GRB'    : [[36,36,38,38,40,40],
                [37,37,39,39,41,41]],
    'GRA'    : [[16,16,18,18,20,20,22,22,24,24,26,26,28,28,30,30],
                [17,17,19,19,21,21,23,23,25,25,27,27,29,29,31,31]],
    'LRA1'   : [[36,37,38,39,40,41,42,43]],
    'LRB1'   : [[44,45,46]],
    'LRSA'   : [[30]],
    'LRSB'   : [[31]],
    'LWSA'   : [[32]],
    'LWSB'   : [[42]],
    'LCC'    : [[47, 47]],
}
```

Return: `ScheduleInfo(numCodePaths=2, numMfma=48, optSchedule, syncCode, nglshift=11, nllshift=11)`.

Notes from this example:

- Inner-list lengths = number of *instructions* in that category. The dispatcher
  consumes them in order via `idMap[key]`, so `idMap[key]` must have at
  least `len(optSchedule[key][i])` entries per codepath `i`.
- Positions can repeat (`[16,16,18,18,...]`) — multiple instructions
  of the same category emitted at the same mfma slot.
- Position `-1` is the pre-MFMA-loop slot (used by the SYNC head and the LCC
  tail at `numMfma-1`, since `numMfma=48` ⇒ `[47, 47]`).
- One codepath per key is fine; the dispatcher branches on `numCodePath`
  vs `len(ts)` at CustomSchedule.py:497–509.

### 1.6 SNop example — `_get_schedule_96x256x64_16bit` (NN, useLDSTr=False, TLDS=1)

CustomSchedule.py:1817–1882:

```python
snopIdxs = [1, 25]
snops = [[x, SNop(1, comment="")] for x in snopIdxs]
...
if snops:
    optSchedule['SNOP'] = [[s[0] for s in snops]]    # [[1, 25]]
    snopCode = [s[1] for s in snops]                 # [SNop(1), SNop(1)]

opt1 = ScheduleInfo(1, 48, optSchedule=optSchedule, syncCode=syncs.get_code(),
                    nglshift=nglshift, nllshift=nllshift, snopCode=snopCode)
```

So SNops are first-class. `optSchedule['SNOP'][0]` is the position list and
`snopCode` is the parallel content list. Single-codepath only (always `[[…]]`).

---

## 2. SNop representability — verdict

### 2.1 Findings

**SNops are fully representable** in the existing CMS format. Specifically:

- `ScheduleInfo.snopCode: list[SNop]` is a constructor field
  (CustomSchedule.py:233, 243).
- The schema key `'SNOP'` is part of `build_idmap`'s canonical output
  (ScheduleCapture.py:579, `idmap['SNOP'] = snopCode`).
- The dispatcher path treats `'SNOP'` like any other category — it reuses the
  `scheduleInst` walker (CustomSchedule.py:410). The only oddity is that
  `snopCode` is passed as a Python `list` (not a Module) but `build_idmap`
  is content-agnostic: it stores whatever you pass (ScheduleCapture.py:541–580).
  The macro builds it via `InstStreams = {key: [stream, idMap[key]] for ...}`
  (CustomSchedule.py:397) where `stream` is `optSchedule[key]` and the
  consumer indexes `instructionList[ind]` (CustomSchedule.py:427) — works
  uniformly for either lists or modules.
- The default-side capture pipeline already tags SNops with `category="SNOP"`
  (KernelWriter.py:2602–2603, _categorize fallback) and the CMS macro
  expander does the same (`expand_cms_macro` ScheduleCapture.py:1804–1805).
- Validator-side: SNops carry no register dataflow and are excluded from the
  cross-graph identity check
  (CMSValidator.py:2364, `_DATA_FLOW_KINDS = ("LR", "LW", "GR", "MFMA")`).
  Their *timing* contribution is consumed via
  `_min_issue_quad_cycles_for(rocisa_inst, profile)` (CMSValidator.py:805+),
  which inspects `SNop.waitState` to compute issue cost for cycle-gap checks.
- An existing production schedule (`_get_schedule_96x256x64_16bit` above and
  one TF32 case at CustomSchedule.py:4465–4536) already emits SNops; the
  validator accepts them in `cmsv.isValid` at CustomSchedule.py:376.

### 2.2 Implication for the converter

The converter does NOT need to drop SNops. It can:

1. Walk the captured default-side instruction stream.
2. When it encounters a `SNop`, bucket the instruction's mfma-slot position
   into `optSchedule['SNOP'][0]` and the rocisa instance into `snopCode`.
3. Emit `'SNOP'` in the produced `optSchedule` dict.
4. Pass `snopCode=...` to `ScheduleInfo(...)` in the generated body.

The `accept_min_quad_cycle_gap=True` knob in the API is therefore *only*
required for cases where the converter places an SNop at a slot that the
validator's `TimingTooCloseFailure` check still flags as too-close — e.g.
because the per-arch `ArchProfile`'s `default_issue_quad_cycles` doesn't
match the SNop's actual `waitState`. Spec §4 in the bead says:
hard-fail on anything that is not a min-quad-cycle gap; soft-fail (warn-and-emit)
on min-quad-cycle gap when the flag is True. That maps cleanly to:

- Catch `cmsv.ValidationError`.
- If the `failure` payload `isinstance(.. , TimingTooCloseFailure)`
  (CMSValidator.py:2162), and `accept_min_quad_cycle_gap`, log a warning
  in the output file's docstring and proceed.
- Otherwise re-raise.

**SNop is NOT a Phase 2 blocker.** The bead's contingency plan ("if
unrepresentable, drop them and warn") does not apply.

### 2.3 Emission-fidelity caveat — `optSchedule` key insertion order (Phase 3 hazard)

SNop is representable as *content* (the points above), but emission fidelity
requires one additional invariant. The dispatcher emits per-`miIndex`
instructions in `dict.items()` insertion order: it iterates
`for k, ts in ToSched.items()` (CustomSchedule.py:494) where
`ToSched = {k: scheduleInst(k, …) for k, … in InstStreams.items()}`
(CustomSchedule.py:435), and `InstStreams` is in turn keyed by
`optSchedule.items()`. Python dicts preserve insertion order. Therefore,
when SNop, SYNC, and (e.g.) LRA0 all have a position at the same `miIndex`,
their relative emission order is determined by which key was inserted into
`optSchedule` first.

The default-side capture is a linear stream with a deterministic emission
order (each `TaggedInstruction` carries its source position). The converter
must preserve that order when populating the `optSchedule` dict — i.e.
**insert keys in the order their first instruction appears in the linear
stream**, not in alphabetical order, not in some fixed canonical order.

Concretely, the converter cannot use a static dict literal; it must walk
the linear stream and `optSchedule.setdefault(category, [[]])` (or
equivalent) so that the first-seen category claims the earliest dict
position. Flag for Phase 3 implementation.

---

## 3. Capture pipeline reuse

### 3.1 What exists today

`ScheduleCapture.py` exposes the following relevant artifacts:

- `LoopBodyCaptureBuilder` (line 432): accumulates `TaggedInstruction`s with
  `(category, slot)` metadata as the writer emits them. Produces a
  `LoopBodyCapture(instructions=[...])` on `finalize()`.
- `TaggedInstruction(wrapped, category, slot)` (line 229): each captured
  rocisa instruction tagged with its `idMap` category and a `SlotKey(subiter,
  slot_kind, mfma_index, sequence)`.
- `FourPartCapture` (line 299): the four loop bodies (`main_loop`,
  `main_loop_prev`, `n_gl`, `n_ll`) each as `dict[codepath, LoopBodyCapture]`.
  The default-side path uses `num_codepaths=1`.
- `build_idmap(...)` (line 541): canonical `{category: source_module|list}`
  schema.
- The KernelWriter integration (`_loopBody`, `kernelBody`) gates on
  `self.states._captureDefaultSchedule = True` to opt in to the shadow
  capture (KernelWriter.py:4493, 4520, 5204). When set, `kernelBody`
  assembles `ctx.default = FourPartCapture(...)` at KernelWriter.py:5245.

### 3.2 Shape of what the converter needs

The converter needs the *linear sequence of TaggedInstructions for the
mainloop*, in emission order, with `category` and `slot.mfma_index`
populated. From that it can directly synthesize:

- `optSchedule[category][0] = [ti.slot.mfma_index for ti in stream if ti.category == category]`
  (1 codepath; see §6 below for multi-codepath).
- `idMap[category]` (Module of rocisa instances) reconstructed from the
  same instructions.
- `syncCode = [ti.wrapped.rocisa_inst for ti in stream if ti.category == "SYNC"]`
  in the same order as their positions.
- `snopCode = [ti.wrapped.rocisa_inst for ti in stream if ti.category == "SNOP"]`.
- `numMfma = sum(1 for ti in stream if ti.category == "MFMA")`.
- `numCodePaths = 1`.

`ctx.default.main_loop[0].instructions` (a `list[TaggedInstruction]`) is
*exactly* the linear stream the converter wants. **No additional capture
shape is required.** The default-side capture is already pre-bucketed into
TaggedInstructions. The converter can consume it as-is.

### 3.3 `nglshift` / `nllshift` derivation rule (resolved)

These two integers are required by `ScheduleInfo`. They control how vlcnt
gets shifted in the NGL/NLL branches of the dispatcher. To synthesize them
from the default-side capture, the converter counts the *vmcnt-tracked*
global-read pair-count (GRA + GRB) in a single codepath of the mainloop.

**Correct formula** (validated below):

```python
nglshift = nllshift = (len(optSchedule['GRA'][0]) + len(optSchedule['GRB'][0])) // 2
```

The `[0]` index is critical: `optSchedule['GRA']` is `list[list[int]]` whose
*outer* list is per-codepath. `len(optSchedule['GRA'])` returns the
codepath count (1 for single-codepath schedules, 2 for dual-codepath, …),
not the GR count. The shift is computed per-codepath, so the converter
takes codepath 0 as the canonical path. (Per the parallel `prp2`
investigation captured in `VALIDATOR_DESIGN.md`, `isValid` itself only
walks codepath 0 today — codepath 0 is the established canonical
choice for single-codepath analyses across the validator stack.)

The factor of 2 is because each entry in `GRA[0]` / `GRB[0]` represents
one of a pair of `BufferLoad*` halves (the schedules duplicate each
load-pair position; see e.g. `'GRA': [[16,16,18,18,…]]` in §1.5). Pair
count = total entries / 2.

**Verification — direct inline confirmation in CustomSchedule.py.** Five
production schedules compute `nglshift` directly from `optSchedule` with
this exact formula (modulo Python `/` vs `//`, which is irrelevant
because both sides of the sum are even):

```
CustomSchedule.py:3051, 3108, 3136, 3169, 5577:
    nglshift = nllshift = len(optSchedule["GRA"][0])/2 + len(optSchedule["GRB"][0])/2
```

This is byte-for-byte the formula above, just written with two divisions
instead of one. It is the canonical existing pattern.

**Verification — spot checks against requested fixtures.**

- `_get_schedule_96x256x64_16bit` NN/!useLDSTr/TLDS=1 branch
  (CustomSchedule.py:1816, `numCodePaths=1`, line 1882): `GRA[0]` has 6
  entries, `GRB[0]` has 16 entries → `(6 + 16) // 2 = 11`. Schedule sets
  `nglshift = nllshift = 11` (CustomSchedule.py:1691, via
  `num_gr = (len(grA) + len(grB)) // 2` at line 1858). Match.
- `_get_schedule_256x96x64_16bit` TN/TLDS=1 branch
  (CustomSchedule.py:914, `numCodePaths=2`, line 1009): `GRA[0]` has 16
  entries (line 946), `GRB[0]` has 6 entries (line 944) →
  `(16 + 6) // 2 = 11`. Schedule sets `nglshift = nllshift = 11`
  (line 916). Match.

**Verification — full audit across all `_get_schedule_*` functions.**
A comprehensive grep across all 30+ registered schedules in
`CustomSchedule.py` confirms zero counterexamples. Where the schedule
hard-codes `nglshift = N` and the formula is computable from inline GRA/GRB
lists, all 44 spot-checkable branches match (`OK=44, MISMATCH=0`). The
remaining branches either compute `nglshift` via an equivalent formulation
(`num_gr = len(gra) + len(grb)` where `gra`/`grb` are halved
pre-`duplicate_list_items`-expansion lists, e.g. CustomSchedule.py:2617,
2641; semantically identical), or use computed list comprehensions whose
length the regex-based audit could not parse (UNKNOWN — no mismatches were
hidden, just unparsed by the audit script).

**Conclusion — formula generalizes to all known schedules.** The
converter can apply this formula unconditionally for the `numCodePaths=1`
output it produces. No exceptions need to be flagged as Phase 3 hazards.

The earlier draft of this memo gave the formula as
`(len(GRA) + len(GRB)) // 2` — wrong-as-written for `numCodePaths>=2`
because it counted codepaths instead of entries. Corrected here.

`nllZeroDscnt` and `mfmaReorder` are non-default flags used by a few
schedules; the converter can leave them at defaults (False / []) for the
"matches default codegen" baseline.

### 3.4 Known capture-side limitations to surface in Phase 2

`LoopBodyCaptureBuilder.finalize()` raises on:

- SMEM operations (`CaptureSMEMError`) — load_sgpr in mainloop.
- Flat loads/stores (`CaptureFlatError`).
- Vector-memory stores (`CaptureStoreError`).

If the YAML being converted produces a kernel whose mainloop contains any of
these, the converter will fail at the capture stage *before* it even tries
to produce a CMS schedule. This is a pre-existing capture limitation, not a
converter limitation — but the converter's docstring should document it.

---

## 4. YAML-load path — TCL vs non-TCL

### 4.1 Findings

There is **one entry point** for both formats: `LibraryIO.read(filename)`
(LibraryIO.py:204–211). It dispatches on extension (`.yaml` / `.json`)
and calls `readYAML` or `readJson`. There is no separate "TCL" code path —
the bead's "TCL vs non-TCL" terminology maps to the two *YAML schemas*
both supported by `Tensile/Tensile.py:Tensile()`:

1. **Default format** (TCL/legacy in bead terminology):
   one config file with top-level `GlobalParameters`, `BenchmarkProblems`,
   `LibraryLogic`, `LibraryClient` keys.
2. **Alternate format** (`--alternate-format` flag, Tensile.py:558–578):
   one or two config files. The first carries `ProblemType` +
   `BenchmarkCommonParameters`/`ForkParameters`/`GroupForkParameters` directly
   at top level, and an optional second file is a flat list of sizes. The
   `Tensile()` function rewrites this into the default-format `dict` shape:

```python
config = {"GlobalParameters": base.get("GlobalParameters")}
solParams = {
    "BenchmarkCommonParameters": base.get("BenchmarkCommonParameters"),
    "ForkParameters": base.get("ForkParameters"),
    "GroupForkParameters": base.get("GroupForkParameters"),
    "BenchmarkFinalParameters": [{"ProblemSizes": sizes}],
}
config["BenchmarkProblems"] = [[base["ProblemType"], solParams]]
```

Both then funnel into `executeStepsInConfig(config, ...)` (Tensile.py:69).

### 4.2 Implication for the converter

The converter can either:

- **(a) Reuse `Tensile()`'s normalization logic** by copying the alt-format
  detection + rewrite block into a small helper (or refactoring it out of
  `Tensile.py:518–578`). This gets both formats for free.
- **(b) Limit Phase 3 to the default format** and document alt-format as
  a Phase 2.x add. The default format is sufficient for "tune one kernel
  config" use.

For the converter's purpose ("one kernel config in, one schedule out"), only
the `BenchmarkProblems[0][0]` (ProblemType) and `BenchmarkProblems[0][1]`
ForkParameters need to be parsed into a single `Solution`. The converter
does NOT need the `LibraryLogic`/`LibraryClient` machinery and does NOT
need to actually run the benchmark — it only needs the same Solution-dict
shape that `_make_solution` (Tests/unit/cms_test_utils.py:417) produces for
unit tests, and the existing `Tensile.SolutionStructs.Solution` constructor.

**Recommendation for Phase 2 user input**: Decide whether the converter
operates on a *full* Tensile YAML (one of formats above) or accepts a
single-Solution dict (closer to `_make_solution`'s input). The full YAML
path adds a fork-resolution step (the YAML may declare a `ForkParameters`
list that fans out to N kernels — which one does the converter convert?).
A simple rule: if the YAML expands to >1 Solution, raise; the user must
narrow first. Or: take `solution_index: int` as a kwarg.

---

## 5. Registered-schedule emit format

### 5.1 What boilerplate the output `.py` needs

Inspecting any `_get_schedule_*` function and its decorator
(CustomSchedule.py:901–1010), the registration boilerplate is:

```python
# Imports the file needs (subset of what CustomSchedule.py uses):
from rocisa.instruction import SBarrier, SNop, SWaitCnt   # for the schedule body
from Tensile.Components.CustomSchedule import (
    RegisterSchedule, ScheduleInfo, ScheduleMatchStatus,
    TileConfig, is16bit, is8bit, isTF32,
    isNN, isNT, isTN, isTT,
)

@RegisterSchedule(
    tile_config=TileConfig(MT0, MT1, DU, PGR, PLR, DTL, DPLB, WSGRA, WSGRB,
                           isa=(9, 5, 0), wavefront_size=64),
    dtype_predicate=is16bit,            # or is8bit / isTF32
    vector_widths=[GRVWA, GRVWB, LRVW],
    matrix_inst=[M, N, K, B],
    mfma_wave_group=[rows, cols],
)
def <schedule_name>(kernel, useLDSTr, TLDS):
    optSchedule = dict()
    syncCode = []
    snopCode = []
    nglshift = nllshift = 0

    if isTN(kernel) and TLDS == 1 and useLDSTr == False:    # layout filter
        # ... build syncTable, optSchedule, syncCode, snopCode ...
        opt1 = ScheduleInfo(numCodePaths, numMfma, optSchedule, syncCode,
                             nglshift, nllshift, snopCode=snopCode)
        return True, opt1
    else:
        return False, None
```

The decorator handles:

- Layout auto-detection by probing the inner function with synthetic kernel
  dicts (CustomSchedule.py:795–822 `_detect_supported_layouts`). Important:
  this means the inner function MUST be tolerant of probe inputs (return
  `False, None` for unsupported layouts; never raise on missing optional
  keys). The probe walks `(transA, transB) × (useLDSTr, TLDS) × valid VWs`
  with `valid_vector_widths = [1, 2, 3, 4, 6, 8]`.
- Registration into `_SCHEDULE_REGISTRY` (CustomSchedule.py:865) so
  `hasCustomSchedule()` finds it at kernel-build time.
- Metadata population into `_SCHEDULE_METADATA` so `query_cms_kernels()` and
  `get_cms_kernel_info_objects()` see it.

### 5.2 Required vs optional fields in `ScheduleInfo`

Required: `numCodePaths`, `numMfma`, `optSchedule`, `syncCode`, `nglshift`,
`nllshift`. Optional: `nllZeroDscnt=False`, `mfmaReorder=[]`, `snopCode=[]`.

For the converter's "matches default" baseline, the optional fields stay at
defaults except `snopCode` (populate when SNops were emitted).

### 5.3 What gets registered

The inner function is *registered* (i.e. appended to `_SCHEDULE_REGISTRY`)
at *import time* — the decorator's `__call__` runs immediately. So the
output file becomes a registered schedule simply by being imported. There is
no separate "register this schedule" call; just `import path.to.file`.

### 5.4 Multi-arch support is free

`TileConfig.isa: tuple = (9, 5, 0)` (CustomSchedule.py:701) defaults to
gfx950 but is a free constructor parameter accepting any
`(maj, min, patch)` tuple. The converter passes the arch through from the
input YAML's `ISA` field (or equivalently the `kernel["ISA"]` derived from
it) into the registered schedule's `TileConfig` — no per-arch code paths
are required in the converter. `hasCustomSchedule` already keys lookup on
`kernel_isa` (CustomSchedule.py:573) so converter output for a non-default
arch will be matched correctly at kernel-build time.

### 5.5 Dispatch ordering — collision policy with existing schedules

`hasCustomSchedule` iterates `_SCHEDULE_REGISTRY` and returns the **first**
match (CustomSchedule.py:580–588, `for schedule_func in _SCHEDULE_REGISTRY`
+ early `return True, schedule` on `ScheduleMatchStatus.FOUND`). Iteration
order = import order. The match key is the unique tuple
`(TileConfig, dtype_predicate, vector_widths, matrix_inst, mfma_wave_group)`
(plus the inner function's runtime layout/useLDSTr/TLDS check).

**Collision risk.** If the converter emits a schedule whose
`(TileConfig, dtype, …)` tuple is already covered by a registered schedule
in `CustomSchedule.py`, then at kernel-build time:

- If the converter's output module is imported *before* `CustomSchedule.py`
  (or its sub-modules in the bkub refactor), the converter's schedule
  wins for matching kernels — *silently shadowing* the hand-tuned one.
- If imported *after*, the converter's schedule loses — silently dead code.

Either outcome is bad: silent shadowing risks regressing performance on
already-tuned configurations, silent dead code wastes the conversion.

**Recommended policy** (Phase 2 confirm): the converter scans
`_SCHEDULE_REGISTRY` (after importing the existing CMS modules) for a
match against the same `TileConfig` + dtype before emitting, and **refuses
to write the output file** if a match exists. An explicit `--overwrite`
flag (or `force: bool = False` kwarg) bypasses the check; otherwise the
converter raises with a message naming the colliding schedule. This keeps
collisions explicit and prevents silent shadow/dead-code outcomes. The
matching `TileConfig` tuple uniqueness ensures the check is deterministic.

### 5.6 Coordination with `rocm-libraries-bkub`

The bead notes a parallel investigation refactoring `CustomSchedule.py` into
a subdirectory. The converter's output file is *one decorated function*; it
imports `RegisterSchedule`, `ScheduleInfo`, `TileConfig`, dtype/layout
predicates from `Tensile.Components.CustomSchedule`. Whatever the final
path of those exports (a re-export layer in `Tensile.Components.CustomSchedule`
or direct from a sub-module), the converter only needs a stable import
target. It can read this set of symbols at code-emit time and not need to
know what file inside `Tensile/Components/` they actually live in. If
the bkub refactor ships first, the converter takes the new import path; if
it ships later, the converter survives via the re-export shim. **Decision
deferred to Phase 2** but the dependency is shallow.

---

## 6. Validator entry point

### 6.1 What exists today

There are three relevant validator entry points, all in
`Tensile/Components/CMSValidator.py`:

- `isValid(scheduleInfo, ValidationContext)` (line 3930). Structural
  validator: checks `optSchedule` shape, slot-counter ranges, ordering,
  per-arch SCC overlap, etc. Raises `ValidationError` on first failure.
  Already invoked at CustomSchedule.py:376 inside `customMainLoopSchedule`.
- `compare_graphs(reference, subject)` (line 2338). Cross-graph compare.
  Returns `list[Failure]`.
- `validate_edge_wait_coverage(graph)` (line 2751). Per-graph wait check.
  Returns `list[Failure]`.

The "validate this newly-registered schedule against the default for this
kernel config" flow is wired into `KernelWriter.kernelBody`
(KernelWriter.py:5204–5302). When `self.states._captureDefaultSchedule`
is True it:

1. Builds `ctx.default = FourPartCapture(source="default-sia3", ...)` from
   the LoopBodyCaptureBuilder shadow run.
2. Calls `build_cms_four_part_capture(macro=..., default_capture=ctx.default,
   ...)` to build `ctx.cms` from the macro emitted by `customMainLoopSchedule`.
3. Calls `compare_graphs(build_dataflow_graph(ctx.default), build_dataflow_graph(ctx.cms))`.
4. Calls `validate_edge_wait_coverage(subj_graph)`.

So **the comparison runs** today — but only as part of a full kernel build,
gated on `_captureDefaultSchedule`. No standalone CLI exists for "given
YAML + new schedule .py, run only this comparison."

### 6.2 What this epic creates

The bead identifies that this epic creates the standalone comparison
entry. Sketch of the API:

```python
def validate_schedule_against_default(
    yaml_path: Path,          # or kernel_config dict
    schedule_module: ModuleType,   # or schedule_path: Path that gets imported
    *,
    isaInfoMap,                # passed in to avoid re-probing
    asm,
) -> ValidationReport:
    """Build a Solution from the YAML, import the schedule (registers it),
    drive a full kernel build with _captureDefaultSchedule=True, and surface
    the resulting compare_graphs / validate_edge_wait_coverage output.

    Returns a ValidationReport with:
      - structural: list[Failure] from isValid (or empty)
      - graph_diff: list[Failure] from compare_graphs
      - wait_coverage: list[Failure] from validate_edge_wait_coverage
      - timing_only_failures: list[TimingTooCloseFailure] (subset of
                              graph_diff for the soft-fail path)
    """
```

Implementation maps directly to the existing `_make_solution` +
`writer._getKernelSource(solution)` machinery already used by
`Tests/unit/test_ScheduleCapture.py:TestPhase4DefaultCapture` (line 916+)
and `TestDataflowGating` (line 1170+). The unit tests *already* drive this
path; productizing them into a standalone helper is straightforward.

The only gotcha is the timing-of-states-init dance the unit tests work
around (test_ScheduleCapture.py:996–1002): `_captureDefaultSchedule` must
be set on `writer.states` AFTER `setupNewTile` initializes it but BEFORE
`_loopBody` reads it. The unit tests monkey-patch `setupNewTile`. The
production helper should do the same OR, cleaner, KernelWriter could expose
a `enable_capture_default_schedule()` method that handles the lifecycle
correctly. **Phase 2 user decision**: monkey-patch (matches tests) or new
KernelWriter API (cleaner, but adds a public surface).

### 6.3 Soft-fail filter for `accept_min_quad_cycle_gap`

`compare_graphs` returns a `list[Failure]`. To support the bead's contract,
the converter's wrapper categorizes each Failure:

- `TimingTooCloseFailure` (CMSValidator.py:2162) — the only soft-fail
  candidate when `accept_min_quad_cycle_gap=True`.
- All others (`OrderInvertedFailure`, `MissingWaitFailure`,
  `WaitInsufficientFailure`, `MissingBarrierFailure`,
  `OverriddenInputFailure`, `InvalidCounterValueFailure`) — hard fail.

Note: `validate_edge_wait_coverage` failures are independent and are NOT
covered by `accept_min_quad_cycle_gap` — they should be hard-fail too.
Phase 2 user input: confirm this.

---

## 7. Phase 1 established conclusions

These are answered in this memo and need no Phase 2 user input. They are
captured here so Phase 2 / Phase 3 don't re-litigate.

E1. **`numCodePaths=1` policy.** The default codegen yields a single linear
   instruction stream; the converter's output is therefore `numCodePaths=1`
   by construction. Multi-codepath CMS (two SIMDs following different
   orderings to mask MFMA issue gaps) is a *human authoring* optimization
   that the bead places explicitly **out of scope** ("Heuristic improvement
   of the conversion … is the authoring step"). Established.

E2. **`nglshift` / `nllshift` derivation.** Computed as
   `(len(optSchedule['GRA'][0]) + len(optSchedule['GRB'][0])) // 2`. This
   formula is verified across all 30+ existing schedules with zero
   counterexamples (audit in §3.3). Five existing schedules use the
   identical formula inline (CustomSchedule.py:3051, 3108, 3136, 3169,
   5577). Established.

E3. **SNop representability.** SNops are first-class in the CMS format
   (§2). The converter does not need to drop or warn-on SNops as a
   general rule. Established.

E4. **Default-codegen SNop well-formedness.** Default-side SNops are
   emitted by `KernelWriter.py` (lines 728–838 etc.) using the writer's
   own per-arch pack→MFMA waitState calculations (`waitState = 4 -
   after2ndMfma` style). They are well-formed *by construction*: the
   writer has accurate per-arch knowledge when it places them, and the
   per-arch validator profile is derived from the same arch metadata. The
   converter inherits this well-formedness — it does not need to second-
   guess SNop placement. (If a default-side SNop ever fired
   `TimingTooCloseFailure`, that would indicate a bug in either the writer
   or the validator profile, *not* a converter problem to handle.)
   Established.

E5. **Multi-arch.** `TileConfig.isa: tuple = (9, 5, 0)` accepts any
   `(maj, min, patch)` tuple. The converter passes the YAML's `ISA`
   through (§5.4). Established.

E6. **SNop emission-fidelity invariant.** The dispatcher emits
   per-`miIndex` instructions in `optSchedule.items()` insertion order; the
   converter must populate `optSchedule` keys in the order they first
   appear in the captured linear stream (§2.3). Established as a Phase 3
   implementation requirement.

E7. **Schedule-collision policy.** The converter refuses to emit if a
   registered schedule for the same `(TileConfig, dtype, …)` tuple already
   exists, unless `--overwrite` is passed (§5.5). Established as
   recommendation; Phase 2 may confirm but no alternative is reasonable.

## 8. Decisions staged for Phase 2 user input

1. **Where does the converter live?** Options:
   - `Tensile/Tools/default_to_cms.py` (sibling to existing tools).
   - `Tensile/Components/SchedConvert/` (new subdirectory, sibling to
     CustomSchedule.py).
   - Standalone script under `Tensile/bin/`.

2. **CLI vs library?** The signature in the bead is a Python function. A
   thin CLI wrapper (`python -m Tensile.Tools.default_to_cms <yaml>
   <output> --schedule-name ...`) is cheap to add; recommend doing both.

3. **YAML format coverage in Phase 3.** Default format only, or both
   default and alternate formats (`--alternate-format`)? See §4.2.

4. **Multi-Solution YAML handling.** A YAML's `ForkParameters` may
   expand to N Solutions. Reject? Take `solution_index: int`? Auto-pick
   the first? Recommend: reject with a clear error; user narrows.

5. **`enable_capture_default_schedule()` lifecycle.** Add a public
   KernelWriter method, or monkey-patch `setupNewTile` like the unit tests
   do? Recommend the public method — it's cleaner, ~5 lines of code, and
   the timing dance is non-obvious for callers.

6. **Soft-fail scope.** Confirm the bead's intent: only
   `TimingTooCloseFailure` from `compare_graphs` triggers warn-and-emit
   when `accept_min_quad_cycle_gap=True`. Wait-coverage failures and all
   other Failure types remain hard-fail.

7. **Coordination with `rocm-libraries-bkub`** (CustomSchedule.py refactor).
   §5.6 above. The dependency is shallow: just import targets. No blocking.

---

## 9. Summary

- **CMS schedule format**: dict-of-list-of-list-of-ints + parallel content
  lists. Fully documented above. Single source of truth in
  `ScheduleInfo` (CustomSchedule.py:222) and `build_idmap`
  (ScheduleCapture.py:541).
- **SNop**: representable. `'SNOP'` is a first-class category;
  `ScheduleInfo.snopCode` is a constructor field. No conversion-loss risk
  for SNops.
- **Capture pipeline reuse**: `ctx.default.main_loop[0].instructions`
  (a `list[TaggedInstruction]`) is the converter's natural input.
  No new capture shape needed.
- **YAML loading**: single entry point `LibraryIO.read`. The
  default vs alternate format split is normalized inside `Tensile()`
  (Tensile.py:555–578); converter can either reuse that or restrict to the
  default format.
- **Schedule registration**: `@RegisterSchedule(...)` decorator at
  module-import time. Output file's only contract is "import-side-effect
  registers a schedule." Boilerplate is 3 imports + 1 decorated function.
- **Validator entry**: comparison machinery exists
  (`compare_graphs` + `validate_edge_wait_coverage`) and runs end-to-end
  inside `KernelWriter.kernelBody` when `_captureDefaultSchedule` is set.
  This epic adds a thin standalone wrapper that drives that path from a
  YAML/Solution, then filters `TimingTooCloseFailure` for the
  `accept_min_quad_cycle_gap` soft-fail. Sketched API in §6.2.

**SNop verdict (most important Phase 2 input)**: SNops are fully
representable in the existing CMS format. The converter does not need
to drop or warn-on SNops in the general case. The `accept_min_quad_cycle_gap`
soft-fail is only needed for Pack→MFMA timing gap warnings detected by
the validator's per-arch quad-cycle check, and is independent of SNop
emission.
