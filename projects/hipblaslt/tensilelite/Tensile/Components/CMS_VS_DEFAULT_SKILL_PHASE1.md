# CMS-vs-Default Diff Skill — Phase 1 Investigation

Bead: `rocm-libraries-prp2`
Branch: `users/alvasile/validator_long_term_plans` (vlt)
Phase 1 scope: investigation only. No code changes; no skill created.

This memo answers the six investigation questions in the bead and stages
the decisions Phase 2 needs from the user.

---

## 1. Schedule corpus inventory

### Where the corpus lives

A single file: `projects/hipblaslt/tensilelite/Tensile/Components/CustomSchedule.py`
(5,725 lines). All registered CMS schedules are defined as module-level
`@RegisterSchedule(...)`-decorated functions in this one file.

### Counts

- `@RegisterSchedule` decorated functions: **38** (one per
  `_get_schedule_*` function).
- `ScheduleInfo(...)` constructions inside those functions: **51**. The
  delta (51 − 38 = 13) is the count of multi-branch schedule functions —
  a single registered function can contain multiple `if isTN/isNN/isNT/isTT
  ... elif ...` branches that each construct their own `ScheduleInfo`.
  44 occurrences of `elif is(TN|NN|NT|TT)(kernel)` confirm the multi-
  layout-per-function pattern.

### Grouping axes

Each schedule is identified by a `CMSKernelInfo` dataclass
(`CustomSchedule.py:69-115`). The grouping axes attached to every
registered schedule are:

| Axis                           | Source                                                                  |
| ------------------------------ | ----------------------------------------------------------------------- |
| `name`                         | function name (e.g. `_get_schedule_256x96x64_16bit`)                    |
| `dtype`                        | `dtype_predicate` mapped to a string ("16bit", "8bit", "TF32")          |
| `(MacroTile0, MacroTile1, DepthU)` | `TileConfig`                                                       |
| `PrefetchGlobalRead, PrefetchLocalRead` | `TileConfig`                                                   |
| `DirectToLds, DtlPlusLdsBuf`   | `TileConfig`                                                            |
| `WaveSeparateGlobalRead{A,B}`  | `TileConfig`                                                            |
| `GlobalReadVectorWidth{A,B}, LocalReadVectorWidth` | `vector_widths`                                     |
| `MatrixInstruction[M, N, K, B]` | `matrix_inst`                                                          |
| `MIWaveGroup[rows, cols]`      | `mfma_wave_group`                                                       |
| `LDSTrInst, TransposeLDS`      | auto-detected by `_detect_supported_layouts` probing the inner function |
| `TransposeA, TransposeB`       | auto-detected (one row per supported (transA, transB, useLDSTr, TLDS))  |

**Important consequence for the skill:** `_SCHEDULE_METADATA` (a list of
`CMSKernelInfo`) is populated at registration time and is the canonical
catalog. The skill should iterate `_SCHEDULE_METADATA` (via
`get_cms_kernel_info_objects(...)`) for arch/dtype/layout filtering,
and dispatch one subagent per `CMSKernelInfo` row — NOT one per
function. A function with 3 layout branches produces 3 catalog rows
and should produce 3 separate analyses.

### Arch coverage today

`hasCustomSchedule` (`CustomSchedule.py:556-589`) gates CMS by
`kernel["ISA"] in (IsaVersion(9,5,0), IsaVersion(11,5,1))` — i.e.,
**gfx950 (CDNA) and gfx1151 (RDNA) only**. The TileConfig dataclass
defaults to `isa=(9, 5, 0)` so the vast majority of schedules are
gfx950. There is no gfx942/gfx940 CMS surface today. Phase 2 should
treat `--arch` filtering as a thin wrapper over the ISA tuple — for
the current corpus, "gfx950" returns essentially everything, "gfx1151"
returns the small RDNA subset.

### Dtype distribution (rough)

Counted from `dtype_predicate` arguments at `@RegisterSchedule(...)`
call sites (see `is16bit\b` / `is8bit\b` / `isTF32\b` predicates):

- `is16bit`: ~24 schedules (the dominant class — fp16 / bf16 GEMM)
- `is8bit`:  ~15 schedules
- `isTF32`:  ~1  schedule

This is the default per-arch / per-dtype frequency-skew the aggregate
report should account for: any "X% of schedules do Y" claim must be
weighted by dtype to avoid 16-bit dominance crowding out 8-bit/TF32
patterns.

---

## 2. Multi-codepath handling — design recommendation

This is the central design question. Findings:

### How the multi-codepath dispatch works in practice

The schedule dict `optSchedule` is keyed by category (`'GRA'`, `'LRA0'`,
`'SYNC'`, …). Each value is a `list[list[int]]` where the outer list
indexes the codepath (length 1 or `numCodePaths`) and the inner list
holds the mfma-slot index for each instruction in that category's
stream. Example from `_get_schedule_224x128x64_16bit` NT branch
(`CustomSchedule.py:2397-2455`, `numCodePaths=2`):

```python
'GRA': [[14,15, 17,18, 20,21, 23,24],
        [15,16, 18,19, 21,22, 24,25]],   # codepath 1 lags codepath 0 by ~1 mfma slot
'LRA1': [[43, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]],  # single list → all codepaths share
'LWSA': [[48]],                                                       # single list → all codepaths share
```

The macro emitter (`customMainLoopSchedule` →
`scheduleInst` → `emit_instructions`, `CustomSchedule.py:494-509`)
reads the per-codepath list and wraps emission in
`ValueIf("\\ID == 0")` / `ValueElseIf("\\ID == 1")` blocks. At runtime
the workgroup launches with a specific `\ID` value (see
`simdSpecDispatch` in `KernelWriterAssembly.py`) and only one branch
executes per wave.

The default codegen has no analogue: it produces a single
`_makeSubIterSchedule` body that every codepath uses. So strictly
speaking the comparison is N vs 1 (N CMS codepaths, 1 default).

### What genuinely differs vs what's boilerplate (representative case)

For `_get_schedule_224x128x64_16bit` NT branch (`numCodePaths=2`):

- **Genuinely different per codepath** (multi-element outer list):
  `LRB0`, `LRA0`, `GRA`, `GRB`, `LRB1`. All by exactly 1 mfma slot —
  this is the dual-issue / address-update + buffer-load pair pattern
  documented in the comment at line 2399 ("Each GR has two instructions
  (addr update + buffer_load), so we list them explicitly as two
  adjacent MFMA indices per GR").
- **Identical across codepaths** (single-element outer list): `SYNC`,
  `GRIncA`, `GRIncB`, `LRA1`, `LRSA`, `LRSB`, `LWSA`, `LWSB`, `LCC`.

So in practice the codepaths are **two phases of the same
instruction-issue rhythm** offset by one slot, not semantically
distinct schedules. This pattern repeats across every multi-codepath
schedule I sampled (`_get_schedule_256x96x64_16bit`,
`_get_schedule_192x256x64_16bit`, etc.).

### Dispatch glue in `KernelWriter.py`

- `customMainLoopSchedule` is invoked at `KernelWriter.py:4581`,
  returns `(optSchedule, numCodePath)`.
- `numCodePath` is consumed by the deferred CMS-side capture builder
  `build_cms_four_part_capture` (`ScheduleCapture.py:1874`), which
  expands the macro once per codepath:

  ```python
  for cp in range(num_codepaths):
      body = expand_cms_macro(macro, id_value=cp, useGR=1, usePLR=1, ...)
      main_loop[cp] = body
      main_loop_prev[cp] = clone_loop_body(body)
  ```

- The validator (`CMSValidator.isValid`) runs the full validation
  pipeline once per codepath (per the per-codepath comments in
  `VALIDATOR_DESIGN.md` §3). Both codepaths converge to identical
  architectural state at the loop boundary, so n_gl/n_ll only ever use
  `\ID=0` (`_emitNoLoadLoopBodyCMSMacro` docstring).

### Three handling options

**Option (a) — compare default to each CMS codepath separately and
aggregate.** Run the diff N times per multi-codepath schedule. Pros:
no information loss; surfaces per-codepath quirks if any. Cons: most
codepath differences are 1-slot offsets that produce duplicated noise
("CMS spreads LR_A by N MFMAs… in codepath 0… in codepath 1 too").

**Option (b) — compare default to a 'canonical' CMS codepath.**
Canonical = codepath 0 (the `\ID == 0` branch). Justified because:
- `n_gl` and `n_ll` already hard-code `\ID=0` (per
  `_emitNoLoadLoopBodyCMSMacro`), so the codebase already treats
  codepath 0 as the canonical choice for tail loops.
- The per-codepath differences are a 1-slot dual-issue alternation —
  semantically the same schedule with a phase shift — so codepath 0
  fully captures the design intent. Codepath 1 is the same schedule
  shifted by 1 mfma; the diff vs default would be ~identical to
  codepath 0's diff modulo a constant slot offset.

**Option (c) — diff codepath 0 against default; emit codepath
divergence as a separate, bounded annotation.** Hybrid: do the main
diff against codepath 0 only, but also compute the "max instruction
slot delta between codepath 0 and codepath i" per category and report
it as a secondary signal ("this schedule uses dual-issue with
1-slot offset" vs "this schedule uses no dual-issue"). This is one
extra line per multi-codepath schedule, no per-codepath subagent
duplication.

### Recommendation: **Option (c)**.

Reasoning:
- (b) loses an observable architectural fact (whether the schedule
  uses dual-issue codepaths at all) and that fact is itself a
  diff-worthy "theme" — default codegen never produces multi-codepath
  schedules.
- (a) triples the subagent budget on multi-codepath schedules for
  ~zero additional information.
- (c) keeps the per-schedule analysis cost at exactly 1 subagent and
  produces an extra one-line annotation when warranted. The extra
  annotation feeds directly into a candidate aggregate theme:
  *"M/N CMS schedules use dual-codepath dispatch; the default codegen
  has no dual-codepath path at all."*

Phase 2 should confirm (c) with the user and lock the canonical
codepath as `0` (matching `_emitNoLoadLoopBodyCMSMacro`).

---

## 3. Timeline-extraction pipeline

### Current state — `ScheduleCapture.py`

The capture pipeline is mature for validator use. Key data shapes:

- `WrappedInstruction` (`ScheduleCapture.py:148-201`): proxy around a
  rocisa instruction with `reads`/`writes` resource sets.
- `TaggedInstruction` (`ScheduleCapture.py:228-249`): wraps a
  `WrappedInstruction` plus its CMS-style `category` ("LRA0", "GRA",
  "SYNC", …) and `SlotKey(subiter, slot_kind, mfma_index, sequence)`.
- `LoopBodyCapture(instructions: list[TaggedInstruction])`: ordered
  flat stream — already a linear timeline.
- `FourPartCapture(main_loop, main_loop_prev, n_gl, n_ll, num_mfma,
  num_codepaths)` — `main_loop` and `main_loop_prev` are dicts keyed
  by codepath; `n_gl` and `n_ll` always `{0: body}`.

### CMS-side timeline (already linear, already shaped right)

`expand_cms_macro` (`ScheduleCapture.py:1722`) walks the CMS
`MAINLOOP` macro, evaluates the `\ID` / `\useGR` / `\usePLR` /
`\useGRInc` / `\useLoop` guard expressions, and emits a flat
`LoopBodyCapture` for the active branch. Calling it with
`id_value=0, useGR=1, usePLR=1, useGRInc=1, useLoop=1` produces
**exactly the linear instruction timeline the diff skill wants** for
codepath 0 of the main-loop body.

### Default-side timeline (also already linear)

For SIA3 only — the only schedule alg captureable today — the
`_makeSubIterSchedule` capture path
(`KernelWriter.py:4396, 4467`) appends each instruction to a
`LoopBodyCaptureBuilder` as it's emitted, in the same flat stream.
The result is also a `LoopBodyCapture` with the same
`TaggedInstruction` shape.

### Bottom line for Phase 2

**The existing capture pipeline already produces timelines in the
right shape for diff analysis.** The skill does NOT need to build a
new (lighter) timeline extractor. Concretely the skill needs:

1. A way to drive the existing pipeline for a given `CMSKernelInfo` →
   produce both the default-side `LoopBodyCapture` and the CMS-side
   `LoopBodyCapture` for codepath 0.
2. A simple flat representation: list of
   `(category, slot_kind, mfma_index, sequence, instruction_str)`
   tuples extracted from `TaggedInstruction.slot` and a canonical
   render of the rocisa inst. Conversion is ~10 lines.

### Coordination with `rocm-libraries-wlrp`

`rocm-libraries-wlrp` (default→CMS converter) needs to walk the same
default-side `LoopBodyCapture` and bucket it by category to emit a
schedule dict. If wlrp lands first, this skill can reuse its
"linearise default-side capture" helper — but the helper is small
enough that it's not a hard dependency. Both paths viable; this
skill can start independently.

**Scope delta (wlrp-stable):**

- *If wlrp lands first:* the skill imports wlrp's
  `default_schedule_to_cms` (or whatever name the converter exposes)
  as the linearization primitive. The skill's per-schedule analysis
  becomes "convert default to CMS form via wlrp → diff against
  captured CMS form." No bespoke converter needed in the skill.
- *If wlrp doesn't land:* the skill builds a lighter inline converter
  (~10-30 lines) that walks `ctx.default.main_loop[0].instructions`
  and groups by `slot.mfma_index`. No registration emit needed (the
  skill doesn't write CMS schedule files; it just compares forms
  in-memory).

Either way the skill ships; only the internals shift.

### Edge cases the skill must handle

- **Non-SIA3 kernels.** Default-side capture asserts
  `scheduleIterAlg == 3`. CMS only exists for kernels that already use
  SIA3 (default codegen would have set it that way), so all CMS
  schedules' matching default-side configs will be SIA3. Should be a
  non-issue; document the assumption.
- **N_GL / N_LL bodies.** Currently the diff focus is the main loop;
  `n_gl` and `n_ll` are interesting but lower-impact. Phase 2 decision:
  scope to main-loop only initially, and revisit n_gl/n_ll once the
  main-loop diff narrative stabilizes.

---

## 4. Difference taxonomy (initial proposal)

Based on reading three schedules in detail
(`_get_schedule_256x96x64_16bit` TN/TLDS=1,
`_get_schedule_224x128x64_16bit` TN+NN+NT,
`_get_schedule_256x256x64_16bit`) plus comparing against the SIA3
heuristics in `SIA.py` and `_makeSubIterSchedule`:

### Tier 1 — high-impact, easily identifiable

| Theme                         | Detection signal                                                                                | Default-codegen pivot                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **LR0 spread-vs-cluster**     | Inter-LR0 mfma-slot spacing distribution. CMS often spreads (`LRA0=[1,2,3,4,5,6,8,10]`); default `_makeSubIterSchedule:1664-1675` evenly distributes via `readLeftLREven`. | `KernelWriter.py:1664-1675` (`readLeftLREven` policy)  |
| **GR clustering / batching**  | Inter-GR mfma-slot spacing. CMS examples cluster GRs (`GRA=[16,16,18,18,20,…]` — pairs at same slot); default `SIA.py:721-767` spreads GRs evenly via `numGlobalReadInsPerMfma`. | `SIA.py:schedGlobalRead` + `calculateGRPMandLWPM`      |
| **GR/LR interleaving order**  | Whether GRA appears before/between/after LR0 stream. CMS often emits LR0 fully before any GR0 (e.g. `_get_schedule_256x96x64_16bit` TN: LR0 ends at slot 19, GR starts slot 16). | `_makeSubIterSchedule` SIA3 main loop, `i in range(numMfmaPerIter)` interleaving |
| **LRSwap placement**          | The mfma slot of `LRSA`/`LRSB` (single integer). CMS picks specific slots like `LRSA=[30]`; default places after final LR0 by construction. | `_makeSubIterSchedule:1840` (`pointerLRCode` placement) |
| **MFMA reorder**              | `mfmaReorder` permutation list non-empty. CMS sometimes reorders the MFMA stream itself (e.g. `_get_schedule_256x256x64_16bit:1400`); default never reorders MFMAs. | The MFMA emit order itself is currently fixed; theme would surface as "default could reorder MFMAs by tile-index permutation P". |

### Tier 2 — synchronization-shape themes

| Theme                          | Detection signal                                                                                | Default-codegen pivot                                  |
| ------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **SWaitCnt count reduction**   | Number of SYNC entries per loop. CMS schedules often have 4–9 SWaitCnts; default emits a SWaitCnt per LR/GR consumer (typically more). | `_makeSubIterSchedule` waitCode emission per iter      |
| **SWaitCnt tightening**        | `dscnt`/`vlcnt`/`vscnt` values in the SYNC stream. CMS uses tight specific counts ("Finish 2/3 LRB1"); default tends to over-wait. | `_wait` calls in `_makeSubIterSchedule`                |
| **SBarrier count reduction**   | Number of `SBarrier` entries in `syncCode`. CMS aims for one barrier per logical phase; default may emit more. | `syncPlrMfmaIndex` barrier insertion point in `_makeSubIterSchedule:1845-1852` |
| **Dual-codepath usage**        | `numCodePaths > 1` — schedule uses dual-issue / phase-offset codepaths. Default never does. | None — would require new mechanism in default codegen. |

### Tier 3 — refinements likely to emerge from corpus run

- **GRInc clustering** — CMS often groups all GRIncA, then all GRIncB
  in early slots; default interleaves with GRs.
- **Per-codepath slot offset magnitude** — for multi-codepath
  schedules, the average slot offset between codepaths (1, 2, …).
- **Pre-loop fence shape** — the SYNC entry at `idx=-1` (pre-loop wait
  for prev-iter LRs) is a CMS pattern; default may not emit a single
  combined pre-loop wait.
- **LWSwap placement relative to last GR** — CMS comments often say
  "swap after last gr a" indicating an explicit policy.

The taxonomy is intentionally larger than the bead's suggested
starter set ("LR placement, GR clustering, MFMA interleaving, barrier
reduction, SWaitCnt tightening"). Phase 3 (corpus run) will narrow it
to the themes that actually fire on >X% of schedules.

---

## 5. Default-codegen surface map

The default-codegen path the skill needs to feed back into has three
main intervention surfaces:

### Surface A — `_makeSubIterSchedule` SIA3 branch

`KernelWriter.py:867-2549` (the SIA3 elif spans lines 1131-2549). This
is the per-iteration scheduler. It walks `for i in range(numMfmaPerIter)`
and, at each MFMA index, emits in fixed order:
- local reads (with `readLeft = checkLocalReadFIFOFull(...)`)
- global reads (`globalReadCode.popFirstNItems(...)`)
- 1LDSBuffer barrier (conditional)
- local writes (between `lwStartMfmaIndex` and `lwEndMfmaIndex`)
- pointer LW / pointer LR (final mfma)
- sync code (at `syncPlrMfmaIndex`)
- next-loop local reads (after `isBarrier`)
- wait for local reads
- pack code
- the MFMA itself

**Key heuristic intervention points** (line numbers from `KernelWriter.py`):

| Heuristic                                  | Lines       | Tunes                                      |
| ------------------------------------------ | ----------- | ------------------------------------------ |
| `readLeftLROPT` + `readLeftLREven`         | 1659-1675   | LR0 spread vs cluster (Tier-1 theme)       |
| `getMFMAs(...)` + dependency reorder       | 1678-1701   | LR-MFMA interleaving (numItersPLR==0 path) |
| `globalReadCode.popFirstNItems(...)`       | 1751-1764   | GR clustering (Tier-1 theme)               |
| `lwStartMfmaIndex` / `lwEndMfmaIndex`      | 1797-1826   | LW window                                  |
| `syncPlrMfmaIndex` barrier insertion       | 1845-1852   | Barrier placement (Tier-2 theme)           |
| `pointerLRCode` placement                  | 1840        | LRSwap placement (Tier-1 theme)            |

### Surface B — `SIA.py` (slot-budget calculations)

`projects/hipblaslt/tensilelite/Tensile/Components/SIA.py` (1,231 lines).
Sets the slot budgets that `_makeSubIterSchedule` consumes:

| Function                              | Lines     | Sets                                  |
| ------------------------------------- | --------- | ------------------------------------- |
| `getLocalWriteMFMAEnd`                | 201-307   | `lwEndMfmaIndex`, `syncPlrMfmaIndex`  |
| `getLocalWriteMFMAStart`              | 309-400   | `lwStartMfmaIndex`                    |
| `getNumLocalWritePerMfma`             | 401-437   | `numLocalWriteModPerMfma`             |
| `calculateGRPMandLWPM`                | 438-466   | `numGlobalReadInsPerMfma`             |
| `getSchedNumForIter0SIA3`             | 651-700   | iter-0 GR scheduling budget           |
| `schedGlobalRead`                     | 721-767   | actual per-iter GR slot assignment    |
| `prepareLWInstToSched` / `schedLocalWrite` | 802-…    | LW slot assignment                  |

### Surface C — sync emission (`_wait`)

`KernelWriter._wait(...)` (called at multiple sites in
`_makeSubIterSchedule`) emits the SWaitCnt at sync points. The
sync-tightening theme would target the dscnt/vlcnt argument
calculations at the call sites.

### Concrete actionable-feedback shape

For each theme the skill identifies, the actionable-feedback report
should produce an entry like:

```
Theme: LR0 spread (observed in N/M schedules)
  CMS pattern:    LRA0 instructions distributed across mfma slots
                  with 2-3 slot gaps (e.g. [1,2,3,4,5,6,8,10]).
  Default:        readLeftLREven distributes evenly within numMfmaPerIter
                  - i (KernelWriter.py:1665, line 1673).
  Code site:      KernelWriter.py:1664-1675
  Suggested change: When numItersPLR > 0 and numReadsInst > numMfmaPerIter/2,
                    bias readLeftLREven toward fewer-LR-per-mfma at the
                    start of the iteration, allowing trailing slots to
                    cluster GRs (see Theme "GR clustering").
  Risk:           N existing CMS schedules already override this; verify
                  the proposed default doesn't regress them.
```

---

## 6. Skill packaging

### Location

`/home/alvasile/.claude/skills/cms-vs-default/` — matches the existing
`/home/alvasile/.claude/skills/{br,split-branch}/` convention.

### Tools needed

- `Bash` — drive `br show`, `git`, optionally invoke `pytest` for
  capture-pipeline smoke tests.
- `Read` — read schedules in `CustomSchedule.py`, read `SIA.py` /
  `KernelWriter.py` for the actionable-feedback step.
- `Agent` (Opus 4.7 dispatch) — one subagent per `CMSKernelInfo`
  catalog row. Each subagent: (1) extracts the CMS timeline via the
  capture pipeline, (2) extracts the matching default timeline, (3)
  produces a per-schedule semantic-diff narrative.
- `Write` — only for the final aggregate report. Per-schedule
  narratives can be returned as text from subagents and aggregated
  in-memory; no per-schedule files needed unless the user opts in.

The skill is **read-only with respect to the codebase** (per the bead's
Hard Rules). It writes only its own report file.

### Input-arg parsing

```
/cms-vs-default <arg> [<arg2> ...]
```

Where each `<arg>` is one of:
- An architecture string: `gfx950`, `gfx1151`. Resolves to ISA tuple
  → filters `_SCHEDULE_METADATA` by ISA.
- A path to a Python file containing CMS schedule definitions: parse
  filename → if it matches `CustomSchedule.py` (or its post-refactor
  per-schedule successors), `import` it and iterate registered
  schedules from the just-loaded module.
- A path to a TCL YAML file: parse the kernel config from YAML, find
  the matching `CMSKernelInfo` (or report no match).
- A dtype name: `16bit`, `8bit`, `TF32` → filter by dtype.
- A layout: `TN`, `NN`, `NT`, `TT` → filter by layout.
- Combinations: `gfx950 16bit TN` → conjunction.

The arg parser should:
1. Bucket args by recognized type (arch / dtype / layout / path).
2. Build a single filter predicate.
3. Apply to `get_cms_kernel_info_objects(...)` (already supports
   dtype + layout natively; arch + path need wrapping).
4. Return the resulting list of `CMSKernelInfo` rows. Print the count
   and a brief preview before dispatching subagents (so the user sees
   "12 schedules will be analyzed" before any work starts).

### SKILL.md sketch

```yaml
---
name: cms-vs-default
description: >-
  Diff-summarize TensileLite CMS schedules against the default-codegen
  output to identify themes for improving the default scheduler.
  Use when investigating why CMS schedules outperform default codegen,
  proposing default-codegen improvements, or auditing the CMS corpus
  for shared patterns.
license: MIT
domain: tensilelite-codegen
role: specialist
scope: analysis
output-format: markdown-report
triggers:
  - cms vs default
  - schedule diff
  - default codegen feedback
  - cms theme analysis
metadata:
  author: alvasile
  version: 0.1.0
---

# /cms-vs-default — CMS vs Default schedule diff skill

## When to use
- Investigating why a CMS schedule beats default codegen.
- Proposing improvements to the default scheduler (`scheduleSubIter` + SIA.py).
- Auditing the CMS corpus for shared patterns / themes.

## Inputs
`/cms-vs-default <filter> [<filter> ...]`
where each filter is an arch (`gfx950`), a dtype (`16bit`, `8bit`, `TF32`),
a layout (`TN`/`NN`/`NT`/`TT`), or a path to a `.py` schedule file or
`.yaml` Tensile YAML.

## Hard rules
- READ-ONLY w.r.t. the codebase. Skill writes only the final report.
- Per-schedule analysis dispatches one Opus 4.7 subagent per
  `CMSKernelInfo` catalog row (NOT per `_get_schedule_*` function —
  multi-layout functions produce multiple rows).
- Multi-codepath handling: compare against codepath 0; report
  codepath divergence as a one-line annotation.
- Output destination: stdout summary + markdown report file (path
  returned to the user). This parallels the user-facing UX of the
  existing `br` and `split-branch` skills, which both produce
  structured stdout output alongside any persisted artifacts.

## Workflow
1. Parse args → filter predicate → resolve to list of CMSKernelInfo rows.
2. For each row, dispatch a subagent (Opus 4.7) to:
   a. Extract the CMS-side LoopBodyCapture for codepath 0 via
      Tensile.Components.ScheduleCapture.expand_cms_macro.
   b. Extract the default-side LoopBodyCapture for the matching kernel
      config via the standard SIA3 capture path.
   c. Produce a per-schedule diff narrative using the taxonomy in
      Tensile/Components/CMS_VS_DEFAULT_SKILL_PHASE1.md §4.
3. Aggregate per-schedule narratives into theme-frequency table.
4. For each major theme, walk the default-codegen surface
   (Tensile/Components/CMS_VS_DEFAULT_SKILL_PHASE1.md §5) and produce
   actionable feedback with cited code locations.
5. Write final report; print path to user.

## References
- Phase 1 memo: projects/hipblaslt/tensilelite/Tensile/Components/CMS_VS_DEFAULT_SKILL_PHASE1.md
- Schedule corpus: projects/hipblaslt/tensilelite/Tensile/Components/CustomSchedule.py
- Capture pipeline: projects/hipblaslt/tensilelite/Tensile/Components/ScheduleCapture.py
- Default scheduler: projects/hipblaslt/tensilelite/Tensile/KernelWriter.py
  (_makeSubIterSchedule SIA3 branch)
- SIA heuristics: projects/hipblaslt/tensilelite/Tensile/Components/SIA.py
```

---

## Phase 2 decisions staged for user

1. **Multi-codepath handling policy.** Phase 1 recommends Option (c):
   diff codepath 0; report codepath divergence as one-line annotation.
   Confirm or override.
2. **Difference taxonomy.** Phase 1 proposes Tier 1 (5 themes), Tier 2
   (4 themes), Tier 3 (4+ to-be-discovered). Confirm starting set or
   prune.
3. **Skill UX:** per-schedule reports separate or aggregate-only?
   Recommendation: aggregate-only by default; add `--per-schedule` flag
   for verbose mode.
4. **Scope of bodies analyzed.** Recommendation: main-loop only in v1;
   revisit n_gl/n_ll in v2 once main-loop diff narrative stabilizes.
5. **Coordination with `rocm-libraries-wlrp`.** Phase 1 confirms the
   skill can start independently. If wlrp lands first, the skill
   reuses its "linearise default-side capture" helper. Confirm: start
   independently, or wait for wlrp?
6. **Comparison-baseline source.** Should the skill auto-derive the
   matching default-codegen output by re-running codegen with
   `hasCustomSchedule(kernel) → False`, or should the user supply a
   pre-captured default-side dump? Recommendation: **auto-derive**, so
   the skill becomes one-input — pass it a `CMSKernelInfo` (or a
   YAML/path arg that resolves to one) and it reconstructs both sides
   from the same kernel config. Avoids drift between captured baselines
   and the live default scheduler.
7. **Host-arch availability.** A CMS schedule may target `gfx950` while
   the host machine is `gfx90a` or `gfx942`. The capture pipeline runs
   at codegen time and does not require a matching GPU, so analysis
   still succeeds. Document explicitly in SKILL.md: *the skill operates
   on captured codegen output, not runtime traces; host arch is
   irrelevant.* Confirm assumption.

---

## Files referenced

- `projects/hipblaslt/tensilelite/Tensile/Components/CustomSchedule.py`
  — corpus (5,725 lines, 38 registered functions, 51 ScheduleInfo
  constructions).
- `projects/hipblaslt/tensilelite/Tensile/Components/ScheduleCapture.py`
  — timeline-extraction primitives (1,952 lines).
- `projects/hipblaslt/tensilelite/Tensile/Components/CMSValidator.py`
  — existing per-codepath validator (4,078 lines).
- `projects/hipblaslt/tensilelite/Tensile/Components/SIA.py`
  — default-scheduler slot-budget heuristics (1,231 lines).
- `projects/hipblaslt/tensilelite/Tensile/KernelWriter.py`
  — `_makeSubIterSchedule` SIA3 branch starts at line 1131.
- `projects/hipblaslt/tensilelite/Tensile/Components/VALIDATOR_DESIGN.md`
  — already documents the per-codepath validator architecture
  (relevant for §2 dispatch decisions).
