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

### 3.3 `nglshift` / `nllshift` derivation rule — DTL-aware

These two integers are required by `ScheduleInfo`. They control how vlcnt
gets shifted in the NGL/NLL branches of the dispatcher.

**User correction (recorded post-investigation).** An earlier draft of
this section claimed the halving formula
`(len(GRA[0]) + len(GRB[0])) // 2` generalizes to all schedules. **It
does not.** The halving is only valid in **DTL** (DirectToLDS, i.e.
`TileConfig.direct_to_lds == 1`) cases, because in DTL mode every other
entry in `GRA[0]` / `GRB[0]` is a *pointer increment*, not a load — so
the entry count is doubled relative to the load-pair count. In non-DTL
cases, the entries are not doubled and the formula is different.

**DTL-aware shape** (Phase 2 must implement both branches):

```python
if tile_config.direct_to_lds == 1:
    # entries are pairs of (load, ptr-increment); halve to get pair count
    nglshift = nllshift = (len(optSchedule['GRA'][0]) + len(optSchedule['GRB'][0])) // 2
else:
    # non-DTL: entries are loads only; no halving
    # exact formula TBD in Phase 2 (likely just len(GRA[0]) + len(GRB[0]),
    # but verify against known non-DTL schedules before locking)
    nglshift = nllshift = ...   # see Phase 2 directive below
```

**Why the prior audit said "0 mismatches".** The 44 spot-checkable
branches in the prior audit were almost certainly all DTL cases (the
example.yaml fixture is DTL — `DirectToLds: [1]`; the worked example in
§1.5 is DTL — `direct_to_lds=1` in its `TileConfig`). The audit lacked
a DTL/non-DTL split and therefore could not see the divergence.

**Phase 2 implementation directive — non-DTL formula derivation.** Before
the converter ships, the implementer must:

1. Enumerate the registered schedules with `direct_to_lds == 0` (filter
   `_SCHEDULE_REGISTRY` by `tile_config.direct_to_lds`).
2. For each, read the inline `nglshift`/`nllshift` literal and compare
   against `len(GRA[0]) + len(GRB[0])` (no halving). Confirm match — or
   discover the actual non-DTL formula.
3. Implement the DTL-aware branch in the converter. Add a regression
   test using a non-DTL schedule from this enumeration.

The `[0]` indexing remains correct (single-codepath canonical choice) —
that part of the prior text stands.

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

### 5.4 Multi-arch support — explicit isa required (no default)

`TileConfig.isa` (CustomSchedule.py:708) currently defaults to `(9, 5, 0)`.
**User direction (recorded post-investigation): the default must be
removed.** Every `TileConfig` construction must specify `isa` explicitly.
Rationale: a silent gfx950 default risks emitting a schedule that
matches a kernel on a different arch by accident; explicit isa makes
arch coverage auditable.

**Phase 2 implementation directives:**

1. **Drop the `isa = (9, 5, 0)` default in `TileConfig`** (CustomSchedule.py:708).
   Make `isa` a required positional or required keyword field of the
   frozen dataclass. After the change, every `TileConfig(...)` call site
   in `CustomSchedule.py` must pass `isa=` explicitly.
2. **Coordinate with bkub.** bkub is splitting CustomSchedule.py into
   per-schedule files (one `_get_schedule_*` per file in `gfx950/`).
   Each existing registration that relied on the default must be
   updated to `isa=(9, 5, 0)` explicitly. This update lands either:
   - inside bkub's split (bkub's implementer adds `isa=(9, 5, 0)` to
     every `TileConfig(...)` it moves), OR
   - as a follow-up to bkub before wlrp Phase 2 ships.
   Either ordering is fine; flag at merge-time.
3. **Converter behavior.** The converter reads the input YAML's `ISA`
   field (or `kernel["ISA"]`) and passes it through as
   `isa=(<maj>, <min>, <patch>)` in the emitted `TileConfig(...)`. It
   never falls back to a default; if the YAML has no ISA, the converter
   raises with a clear error.

`hasCustomSchedule` already keys lookup on `kernel_isa`
(CustomSchedule.py:573) so the dispatch side is unaffected by the
default removal.

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

**User question (recorded post-investigation):** "Why is
`validate_schedule_against_default` needed?"

**Answer.** Strictly speaking, it isn't *required* — the comparison
already runs as a side effect of `KernelWriter.kernelBody` when
`_captureDefaultSchedule=True`. A new entry point earns its keep on
three grounds:

1. **Single kernel build.** The converter must drive ONE kernel build
   to obtain `ctx.default` (its input). If validation requires a
   *second* kernel build (registering the emitted schedule and
   re-driving with `_captureDefaultSchedule=True`), that's wasted work.
   `validate_schedule_against_default` lets the converter validate
   **in-process** against an already-built `ctx.default` + a freshly
   constructed `ctx.cms` from the emitted schedule.
2. **Structured return value.** The existing path emits failures via
   kernel-writer logging. The converter pipeline needs a typed
   `ValidationReport` it can branch on programmatically — not a log
   string to scrape.
3. **Pipeline composability.** The converter is ONE caller; future
   tooling (the `prp2` skill, CI auto-comparison, hand-tuning workflows)
   can reuse the same standalone entry. Each new caller should not
   duplicate the kernel-writer plumbing.

If the existing path's outputs were exposed via a `ValidationReport`
return value at the existing call site (KernelWriter.py:5204–5302),
the standalone entry would be unnecessary. **Phase 2 implementer's
choice:** either add `validate_schedule_against_default` as a thin
in-process wrapper (recommended; ~30 LOC), or refactor the existing
path to return a `ValidationReport` and have the converter call
`KernelWriter.kernelBody` directly with capture enabled. Either is
defensible; the wrapper is the smaller change.

Sketch of the API:

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

### 6.3 Soft-fail policy — always accept timing-too-close

**User decision (recorded post-investigation): the converter ALWAYS
soft-fails `TimingTooCloseFailure` — there is no opt-in flag.** The
prior `accept_min_quad_cycle_gap=True` kwarg in the proposed API is
removed. Rationale: the converter's job is to convert what default
codegen emitted; timing-too-close on the converted result is by
definition a property the default already had (default codegen produced
the source instructions, including any min-quad-cycle Pack→MFMA gap
that triggers the validator). Refusing to emit a converted schedule
because the default would have triggered the same warning is not
useful.

Categorization the converter applies to `compare_graphs` output:

- `TimingTooCloseFailure` (CMSValidator.py:2162) — **always soft-fail**:
  log a warning into the emitted file's docstring, proceed.
- All others (`OrderInvertedFailure`, `MissingWaitFailure`,
  `WaitInsufficientFailure`, `MissingBarrierFailure`,
  `OverriddenInputFailure`, `InvalidCounterValueFailure`) — **hard
  fail**: the converter raises and emits nothing.
- `validate_edge_wait_coverage` failures — **hard fail** (independent
  of timing). These signal missing waits in the emitted schedule, not
  timing tightness; soft-failing them would emit known-broken code.

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

E2. **`nglshift` / `nllshift` derivation — DTL-aware.** The
   `(len(GRA[0]) + len(GRB[0])) // 2` halving formula is valid only for
   DTL (`direct_to_lds == 1`) cases, because in DTL mode every other
   GRA/GRB entry is a pointer increment rather than a load. Non-DTL
   formula needs to be derived in Phase 2 from a non-DTL schedule
   audit. See §3.3 for the corrected DTL-aware shape and the Phase 2
   directive. (The earlier "0 mismatches" audit was DTL-only.)

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

E5. **Multi-arch — explicit isa, no default.** `TileConfig.isa` must
   become a required field; the existing `(9, 5, 0)` default is
   removed. The converter reads ISA from the YAML and passes it
   through explicitly; missing ISA in input → hard error. See §5.4 for
   the Phase 2 directive (coordinated with bkub).

E6. **SNop emission-fidelity invariant.** The dispatcher emits
   per-`miIndex` instructions in `optSchedule.items()` insertion order; the
   converter must populate `optSchedule` keys in the order they first
   appear in the captured linear stream (§2.3). Established as a Phase 3
   implementation requirement.

E7. **Schedule-collision policy.** The converter refuses to emit if a
   registered schedule for the same `(TileConfig, dtype, …)` tuple already
   exists, unless `--overwrite` is passed (§5.5). Established as
   recommendation; Phase 2 may confirm but no alternative is reasonable.

## 8. Phase 2 decisions — RESOLVED

User decisions recorded; Phase 2 is unblocked.

1. **Converter file location → `Tensile/Components/CustomSchedule/cms_from_default.py`.**
   The converter lives **inside the bkub package** (alongside
   `dispatch.py` / `shared.py` / `gfx950/`). Not `Tensile/Tools/`,
   not `Tensile/bin/`, not a sibling subdirectory. This makes the
   converter a first-class component of the CMS subsystem and gives it
   short relative imports for `RegisterSchedule`, `ScheduleInfo`,
   `TileConfig`, dtype/layout predicates.

2. **CLI → yes (in addition to the library function).** Phase 2 ships
   both:
   - Python API: `cms_from_default.default_schedule_to_cms(...)` (importable).
   - CLI: `python -m Tensile.Components.CustomSchedule.cms_from_default
     <input.yaml> <output.py> --schedule-name <name> [--isa <maj.min.patch>]`.
   The CLI is a thin wrapper over the API.

3. **YAML format → see `Tensile/Components/example.yaml` (committed
   alongside this memo).** The example.yaml is the canonical input
   format the converter must accept. It is the **default Tensile YAML
   format** (top-level `GlobalParameters` + `BenchmarkProblems` with
   one `[ProblemType, BenchmarkProblemSizeGroup]` pair under
   `BenchmarkProblems[0]`). Phase 2 does NOT need to support the
   alternate (`--alternate-format`) shape; if the user wants to convert
   an alternate-format YAML, they normalize it via Tensile's existing
   `Tensile()` rewrite path first.

4. **Multi-Solution YAML handling → reject with a clear error.** When
   the input YAML's `ForkParameters` expands to >1 Solution, the
   converter raises with a message naming the fork dimensions and
   suggesting which keys to narrow. **No `solution_index` kwarg.** The
   user must provide a YAML that resolves to exactly one Solution.

5. **`enable_capture_default_schedule()` lifecycle → public method.**
   Add a public method to `KernelWriter` that handles the
   `_captureDefaultSchedule` lifecycle dance internally (the
   setupNewTile-after, _loopBody-before timing). The unit-test
   monkey-patch pattern is replaced by callers using this method. The
   converter and the test suite both consume the new public surface.

6. **Soft-fail policy → ALWAYS accept timing-too-close.** The
   converter unconditionally soft-fails `TimingTooCloseFailure` from
   `compare_graphs`. The previously proposed
   `accept_min_quad_cycle_gap` kwarg is **removed** — there is no
   opt-out, no opt-in. All other failure types (incl.
   `validate_edge_wait_coverage` failures) remain hard-fail. See §6.3
   for rationale.

7. **Coordination with `rocm-libraries-bkub`** — already implied by
   decision 1 (the converter file lives in bkub's package). The
   converter is added in the same package layout bkub establishes; if
   bkub ships first, the converter takes the new import path; if the
   converter ships first, it is the first non-`gfx950/` file in the
   new package. No blocking either direction.

### Phase 2 implementation directives (derived from above)

- **API signature** (final, with kwargs that survived after decision 6):
  ```python
  def default_schedule_to_cms(
      yaml_path: Path,
      output_path: Path,
      *,
      schedule_name: str,
      isa: tuple[int, int, int] | None = None,   # override; falls back to YAML's ISA
  ) -> ValidationReport:
      """Convert a default-codegen schedule into a registered CMS schedule.

      Always soft-fails TimingTooCloseFailure (warning written to output
      file's docstring); hard-fails all other validator failures.

      Reads the input YAML as the default Tensile format. Raises if the
      YAML expands to more than one Solution; user must narrow ForkParameters
      to a single config. Raises if the YAML lacks an ISA and `isa` is None.
      """
  ```
  No `accept_min_quad_cycle_gap` parameter.

- **CLI signature:**
  ```
  python -m Tensile.Components.CustomSchedule.cms_from_default \
      <input.yaml> <output.py> \
      --schedule-name <name> \
      [--isa <maj.min.patch>] \
      [--overwrite]
  ```
  `--overwrite` bypasses the schedule-collision check (§5.5).

- **Schedule-collision policy** (§5.5 unchanged): converter scans
  `_SCHEDULE_REGISTRY` for matching `(TileConfig, dtype_predicate, ...)`
  and refuses to write when a match exists, unless `--overwrite` /
  `force=True`.

- **Multi-arch** (§5.4): converter passes `isa=(<maj>, <min>, <patch>)`
  explicitly to `TileConfig`. `TileConfig.isa` default removed.
  Coordinate with bkub on the default removal (either bundles into
  bkub's split, or lands as a follow-up before wlrp Phase 2 ships).

- **`nglshift` / `nllshift`** (§3.3): DTL-aware branch in the
  converter. Phase 2 implementer enumerates non-DTL schedules,
  derives the non-DTL formula, implements both branches, adds a
  regression test using a non-DTL schedule.

- **`validate_schedule_against_default`** (§6.2): implement as a thin
  in-process wrapper that runs ONE kernel build (with capture enabled
  via the new public method from decision 5), runs `compare_graphs` +
  `validate_edge_wait_coverage` against the default capture and a
  fresh CMS capture from the emitted schedule, returns a
  `ValidationReport` with categorized failure lists. Do not refactor
  the existing kernel-writer logging path; keep the new wrapper
  separate.

- **Multi-Solution rejection error message** (decision 4): name the
  fork dimensions that produced >1 Solution and suggest which keys to
  narrow. Example error wording the implementer should produce:
  ```
  Input YAML expands to N Solutions (fork dimensions: MatrixInstruction[3],
  PrefetchGlobalRead[2]). Narrow ForkParameters to a single config and re-run.
  ```

- **`example.yaml` is committed** alongside this memo as the canonical
  input format. Phase 2 implementer uses it as the integration-test
  fixture for the converter.

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
