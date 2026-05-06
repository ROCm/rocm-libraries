# LR divergence investigation: counts + WHY analysis

**Bead:** `rocm-libraries-ffg`
**Scope:** investigation only — no validator or schedule changes proposed.
**Validator response:** filed under sibling bead `rocm-libraries-p56` and intentionally
left out of this report.

## 1. Executive summary

For the 2 CMS=1 kernels in the gfx1151 HHS yaml, the default-side `FourPartCapture`
contains substantially more instructions than the CMS-side `FourPartCapture`. The
divergences fall into **five named classes** (see typology in §6): **un-scheduled
source-module category** (GRIncA/GRIncB), **macro-guard suppression**
(LRA{lastIter}/LRB{lastIter} in n_ll under PLR=0), **scheduler-injected sync inflation**
(SYNC), **CMS-only loop-counter machinery** (LCC), and — for MT64x32x64 only —
**incomplete miIndex coverage** in the CMS schedule (LRA0..3, LRB0..3, LWA, LWB
emit only the first 1–4 of N source instructions). The 12-LR `CaptureConsistencyError`
in the original bug report is class **macro-guard suppression** (LRA3 + LRB3 in n_ll
for MT128x64x64). **No "missing-coverage" bug was found in the MT128x64x64 main loop**:
default-side and CMS-side LR counts there match exactly per (LRA{u}, LRB{u}) bucket.
The MT64x32x64 case looks like a real CMS schedule under-specification (its
`optSchedule` references only a small subset of the actual instruction streams).

## 2. Per-(kernel, body, category) count tables

Source: extracted from `/tmp/lr_divergence_probe_report.json`, produced by
`Tensile/Components/GFX1151_AUDIT/lr_divergence_capture_probe.py` (this bead's
probe extension), with `--disable-validator` semantics in effect (validator entry
points no-op'd so the build completes past `CaptureConsistencyError` and ALL kernels
report). Both kernels reach the validator hook; categories are tagged at
`KernelWriter._captureSubIterToBuilder` (KernelWriter.py:2546-2633).

The probe is deterministic — two consecutive runs produced byte-identical reports
(md5 verified, see §7).

### 2.1 MT128x64x64 (PGR=1, PLR=0, SIA=3, CMS=1)

`main_loop` and `main_loop_prev` are identical (CMS produces them via
`clone_loop_body`; ScheduleCapture.py:5039) and are reported as a single column.

| Body | Category | Default | CMS | Delta |
|---|---|---|---|---|
| ML / ML-prev | GRA    |  8 |  8 | 0  |
| ML / ML-prev | GRB    |  4 |  4 | 0  |
| ML / ML-prev | GRIncA |  9 |  0 | +9 |
| ML / ML-prev | GRIncB |  9 |  0 | +9 |
| ML / ML-prev | LCC    |  0 |  2 | -2 |
| ML / ML-prev | LRA0   |  8 |  8 | 0  |
| ML / ML-prev | LRA1   |  8 |  8 | 0  |
| ML / ML-prev | LRA2   |  8 |  8 | 0  |
| ML / ML-prev | LRA3   |  8 |  8 | 0  |
| ML / ML-prev | LRB0   |  4 |  4 | 0  |
| ML / ML-prev | LRB1   |  4 |  4 | 0  |
| ML / ML-prev | LRB2   |  4 |  4 | 0  |
| ML / ML-prev | LRB3   |  4 |  4 | 0  |
| ML / ML-prev | LWA    |  8 |  8 | 0  |
| ML / ML-prev | LWB    |  4 |  4 | 0  |
| ML / ML-prev | MFMA   | 32 | 32 | 0  |
| ML / ML-prev | SYNC   | 34 |  5 | +29|
| ML / ML-prev | **TOTAL** | **156** | **111** | **+45** |
| n_ll | LRA0  |  8 |  8 | 0  |
| n_ll | LRA1  |  8 |  8 | 0  |
| n_ll | LRA2  |  8 |  8 | 0  |
| n_ll | LRA3  |  8 |  0 | +8 |
| n_ll | LRB0  |  4 |  4 | 0  |
| n_ll | LRB1  |  4 |  4 | 0  |
| n_ll | LRB2  |  4 |  4 | 0  |
| n_ll | LRB3  |  4 |  0 | +4 |
| n_ll | MFMA  | 32 | 32 | 0  |
| n_ll | SYNC  | 22 |  5 | +17|
| n_ll | **TOTAL** | **102** | **73** | **+29** |

n_gl is correctly absent on both sides (PGR=1 path; `kernel_emits_n_gl(kernel)`
returns False for this kernel — see ScheduleCapture.py:4935). No row reported.

### 2.2 MT64x32x64 (PGR=2, PLR=1, SIA=3, CMS=1)

ML and ML-prev are again identical and reported as a single column.

| Body | Category | Default | CMS | Delta |
|---|---|---|---|---|
| ML / ML-prev | GRA    |  4 |  4 | 0  |
| ML / ML-prev | GRB    |  2 |  2 | 0  |
| ML / ML-prev | GRIncA |  9 |  0 | +9 |
| ML / ML-prev | GRIncB |  9 |  0 | +9 |
| ML / ML-prev | LCC    |  0 |  2 | -2 |
| ML / ML-prev | LRA0   | 32 |  2 | +30|
| ML / ML-prev | LRA1   | 32 |  2 | +30|
| ML / ML-prev | LRA2   | 32 |  2 | +30|
| ML / ML-prev | LRA3   | 32 |  2 | +30|
| ML / ML-prev | LRB0   | 16 |  1 | +15|
| ML / ML-prev | LRB1   | 16 |  1 | +15|
| ML / ML-prev | LRB2   | 16 |  1 | +15|
| ML / ML-prev | LRB3   | 16 |  1 | +15|
| ML / ML-prev | LWA    | 32 |  4 | +28|
| ML / ML-prev | LWB    | 16 |  2 | +14|
| ML / ML-prev | MFMA   |  8 |  8 | 0  |
| ML / ML-prev | SYNC   | 13 |  5 | +8 |
| ML / ML-prev | **TOTAL** | **285** | **39** | **+246** |
| n_gl | GRIncA  |   9 |  0 | +9 |
| n_gl | GRIncB  |   9 |  0 | +9 |
| n_gl | LRA0..3 | 4×32| 4×2| +120|
| n_gl | LRB0..3 | 4×16| 4×1| +60 |
| n_gl | LW (generic) | 1 | 0 | +1 |
| n_gl | MFMA    |   8 |  8 | 0   |
| n_gl | SYNC    |  15 |  5 | +10 |
| n_gl | UNKNOWN |  47 |  0 | +47 |
| n_gl | **TOTAL** | **281** | **25** | **+256** |
| n_ll | LRA0..2 | 3×32| 3×2| +90 |
| n_ll | LRB0..2 | 3×16| 3×1| +45 |
| n_ll | MFMA    |   8 |  8 | 0   |
| n_ll | SYNC    |   6 |  5 | +1  |
| n_ll | **TOTAL** | **158** | **22** | **+136** |

(LRA3/LRB3 absent from n_ll on BOTH sides for this kernel — `lastIter=3` reads are
suppressed on the default side too, because under PLR=1 with this kernel's
`numItersPLR=1`, `doReadA = u < LoopIters/numIterPerCoalescedReadA - numItersPLR`
yields `doReadA=False` for u=3; KernelWriter.py:4090. Symmetric on both sides for
this case, no delta to explain.)

## 3. Concrete diff lists (MT128x64x64 detail)

### 3.1 main_loop / main_loop_prev — only-in-default (47)

**`GRIncA` (9)** — all `ds_load`-class addressing arithmetic; never tagged on
CMS side because `_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151`'s
`optSchedule` (CustomSchedule.py:7619-7640) has no `'GRIncA'` key:
```
s_add_u32      s[sgprSrdA+0],          s[sgprSrdA+0],         s76
s_addc_u32     s[sgprSrdA+1],          s[sgprSrdA+1],         s77
s_cmp_eq_u32   s[sgprLoopCounterL],    s[sgprStaggerUIter]
s_cmp_eq_u32   s[sgprShadowLimitA+1],  0
s_cselect_b32  s76,                    s[sgprWrapUA+0],       s[sgprGlobalReadIncsA+0]
... 4 more (cselect for hi half, sub for ShadowLimit, addc, cselect for sgpr77)
```

**`GRIncB` (9)** — same structure, B-side (mirror of GRIncA list).

**`SYNC` (29)** — scheduler-injected `s_waitcnt`/`s_barrier` not enumerated by
the CMS `syncTable`:
```
s_waitcnt lgkmcnt(0)        x  6
s_waitcnt lgkmcnt(2)        x 16
s_waitcnt lgkmcnt(4)        x  4
s_waitcnt lgkmcnt(8)        x  2
s_barrier                   x  1
```
(Aggregated; full per-instruction list is in
`/tmp/lr_divergence_analysis.json` under `bodies.main_loop.only_default_by_category.SYNC`.)

### 3.2 main_loop / main_loop_prev — only-in-CMS (2)

**`LCC` (2)** — loop-counter management, present in CMS macro from `optSchedule['LCC']`
(CustomSchedule.py:7639) but never tagged on default side because the default-side
`build_idmap` call at KernelWriter.py:4444 passes `loopCounterCode=Module()`
(empty) — only customMainLoopSchedule sees the real LCC items via `closeLoop` plumbed
through `loopCounterCode` parameter at KernelWriter.py:4581:
```
s_cmp_eq_i32 s[sgprLoopCounterL], 0x1
s_sub_u32    s[sgprLoopCounterL], s[sgprLoopCounterL], 1
```

### 3.3 n_ll — only-in-default (29)

**`LRA3` (8) + `LRB3` (4)** — the actual 12-LR identity set the bead's
`CaptureConsistencyError` flagged, e.g.:
```
ds_load_b128 v[vgprValuA_X0_I0+0  : +0 +3], v[vgprLocalReadAddrA+0] offset:96
ds_load_b128 v[vgprValuA_X0_I0+12 : +12+3], v[vgprLocalReadAddrA+0] offset:4720
ds_load_b128 v[vgprValuA_X0_I0+16 : +16+3], v[vgprLocalReadAddrA+0] offset:9312
... 5 more LRA3
ds_load_b128 v[vgprValuB_X0_I0+0  : +0 +3], v[vgprLocalReadAddrB+0] offset:96
ds_load_b128 v[vgprValuB_X0_I0+12 : +12+3], v[vgprLocalReadAddrB+0] offset:4720
... 2 more LRB3
```

**`SYNC` (17)** — same character as ML SYNCs above.

n_ll has **no** only-in-CMS entries.

## 4. WHY analysis (with file:line citations)

### 4.1 GRIncA / GRIncB — un-scheduled source-module category

**Class:** *un-scheduled source-module category*.

**Why default has them:** the per-iter `_makeSubIterSchedule` flow consumes
`globalReadCode = self.codes.perIterGlobalRead[iteration]` (KernelWriter.py:875,
877), and SIA.py distributes the GR-inc items across iters' perIterGlobalRead
(documented at KernelWriter.py:1361-1364 in `build_id_to_category_per_iter`'s
docstring: "leaves are SHARED with leaves in self.codes.perIterGlobalRead[u] via
SIA.py:732"). The shadow capture path tags them as GRIncA/GRIncB via
`build_idmap` (KernelWriter.py:4438-4439, 4533-4540) and walks them into the
default-side body.

**Why CMS does not:** `customMainLoopSchedule` iterates `for k, ts in
ToSched.items()` (CustomSchedule.py:484), where `ToSched` is built from
`opt1.optSchedule` (CustomSchedule.py:425). The schedule
`_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151` (CustomSchedule.py:7619-7640)
defines no `'GRIncA'` or `'GRIncB'` key, so the GR-inc instructions never get
inserted into the macro and never get tagged via `tag_by_origin_id`. They're
absent from the CMS `FourPartCapture`. Same omission in
`_get_schedule_64x32x64_pgr2_plr1_16bit_gfx1151` (CustomSchedule.py:7366-7388).

(The CMS macro signature has `useGRInc` as a `Macro` parameter — CustomSchedule.py:389
— intended to gate where the increments would go, but the schedules don't define
the `'GRIncA'`/`'GRIncB'` streams so the macro never receives the items.)

### 4.2 LCC — CMS-only loop-counter machinery

**Class:** *CMS-only loop-counter machinery*.

**Why CMS has them:** `customMainLoopSchedule` is called with
`self.closeLoop(...)` as its `loopCounterCode` argument (KernelWriter.py:4581).
That code is populated into `optSchedule['LCC']` (e.g., CustomSchedule.py:7639:
`'LCC': [[31, 31]]` for MT128x64x64; line 7387 `'LCC': [[7, 7]]` for MT64x32x64),
emitted into the macro via the standard `scheduleInst` path (CustomSchedule.py:425).

**Why default does not:** the default-side `build_idmap` call at KernelWriter.py:4444
explicitly passes `loopCounterCode=Module()` (empty, with the comment "LCC items are
added by customMainLoopSchedule, not SIA3"). Default-side SIA3 emits the loop close
at a different point in the module that is OUTSIDE the captured per-iter loop,
so it correctly does not appear in the captured body — but CMS's choice to fold
LCC into its main-loop macro means it *does* show up in the CMS capture.

### 4.3 SYNC inflation (every body, every kernel)

**Class:** *scheduler-injected sync inflation*.

**Why default has many SYNCs:** SIA3's per-iter machinery emits multiple
`SWaitCnt`/`SBarrier` instructions per loop iteration via `waitCode`,
`waitLWCode`, `syncCode`, plus inline SWaitCnts injected into the iterCode by
the LDS-counter tracker (`localReadsVacancy`/`Wait` path in
KernelWriter.py:1086-1125). The captured items are tagged "SYNC" via the
isinstance fallback at KernelWriter.py:2596-2597 (`isinstance(item, (SWaitCnt,
SBarrier)) → category = "SYNC"`).

**Why CMS has fewer SYNCs:** CMS's sync emission is fully driven by `syncTable`
in the schedule (CustomSchedule.py:7606-7616 for 128x64x64; 7353-7363 for
64x32x64), each entry corresponding to one `SWaitCnt` or `SBarrier`. The 5
SYNCs we see on the CMS side for ML are exactly the 5 entries in `syncTable`.
The shift handling for nllvmcnt (CustomSchedule.py:427-449) deepcopies SWaitCnts
across {ML, NGL, NLL} variants — which is why n_gl/n_ll also see ~5 SYNCs on
the CMS side.

The **+29** SYNC delta for ML and **+17** for n_ll on MT128x64x64 are not a
correctness gap — they are the side-effect of SIA3's per-iter wait-tracking model
versus CMS's hand-curated minimal sync set. Whether SIA3's extras are *needed*
for correctness is a separate per-edge question, classifiable only by
`validate_edge_wait_coverage` against the dataflow graph.

### 4.4 LRA{lastIter} / LRB{lastIter} in n_ll — macro-guard suppression

**Class:** *macro-guard suppression*.

**Why CMS does not have LRA3/LRB3 in n_ll (for MT128x64x64):** the n_ll body
is built by `expand_cms_macro` with `usePLR=0` (ScheduleCapture.py:5054-5060;
specifically `useGR=0, usePLR=0, useGRInc=0, useLoop=0`). At
CustomSchedule.py:457-458, `get_macro_guard` wraps any `LRA{lastIter}` /
`LRB{lastIter}` items in `\\usePLR == 1`. With `usePLR=0` at expansion time the
guarded blocks evaluate false and those instructions never enter the n_ll
capture body.

For MT128x64x64 (LoopIters=4), `lastIter=3`, so LRA3 (8 items) and LRB3 (4
items) are guarded — exactly the 12 LR identities the original
`CaptureConsistencyError` reports.

**Why default DOES emit them:** `_noLoadLoopBodyDefault` runs the same per-iter
loop with `for uIdx in range(0, kernel["LoopIters"])` (KernelWriter.py:3142),
and the per-iter `doReadA` predicate (KernelWriter.py:4090) is
`u < LoopIters/numIterPerCoalescedReadA - numItersPLR`. For MT128x64x64,
PLR=0 means `numItersPLR=0`, so for `u=3` the predicate is `3 < 4 - 0 = True` →
default emits a local-read at u=3 (which gets tagged LRA3 via the per-iter
shadow-capture's `LRCodeAAllIters[3]` membership).

This is **intentional CMS behavior** — under PLR=0, the lastIter LRs would
prefetch reads for a non-existent next iteration in the NLL, so the macro
suppresses them. The default-side scheduler doesn't apply the same guard
because the default's `noLoadLoopBody` doesn't share CMS's macro infrastructure;
it re-runs the per-iter machinery without the lastIter-specific gating.

### 4.5 MT64x32x64 — incomplete miIndex coverage in CMS schedule

**Class:** *incomplete miIndex coverage*.

**Why default has 32 LRA0:** `LRCodeAAllIters[0]` is appended to inside the
per-iter loop at KernelWriter.py:4152: `LRCodeAAllIters[uIdx].add(localReadCodeA)`.
For MT64x32x64 (MIWT_A=2, wave32, 16-bit), one iter's `localReadDo` returns
~32 16-bit DS-loads (16 register pairs covering the per-wave A-tile slice).
All of those land in `LRCodeAAllIters[0]`, get tagged "LRA0" via the
`build_idmap` factory path (ScheduleCapture.py:1232: `idmap[f"LRA{u}"] =
LRCodeA[u]`), and the default-side capture walks them all into the body.

**Why CMS has only 2 LRA0:** the CMS schedule for MT64x32x64
(CustomSchedule.py:7372: `'LRA0': [[0, 1]]`) lists only 2 miIndex slots in the
codepath. Inside `customMainLoopSchedule`, `scheduleInst` only emits the
*indexed* instructions — `instructionList[0]` at miIndex=0 and
`instructionList[1]` at miIndex=1 (CustomSchedule.py:407-419). Once an index
is consumed, it's marked with the placeholder `ph` at CustomSchedule.py:419
so it cannot be re-emitted. The remaining 30 entries in `LRCodeAAllIters[0]`
are **never referenced by any miIndex in any optSchedule key**, so they are
never added to the macro and never appear in the CMS capture.

The same shape applies to LRA1..3 (each has only 2 indexed slots versus
default's 32), LRB0..3 (1 indexed slot versus default's 16), LWA (4 versus 32),
and LWB (2 versus 16). Concrete count ratios match the per-iter source counts
exactly: MIWT_A=2 with wave32 and 16-bit produces 32 per-iter LR-A loads while
the schedule references 2 (a 16x ratio, matching `MIInputPerThreadA × number-of-pairs`
arithmetic for the kernel).

This is a *real schedule under-specification* relative to what the kernel
generator emits. Whether it represents an intentional design decision (e.g.,
expecting later expansion or batching the rest somewhere not visible from
`optSchedule`) or a bona-fide bug in the schedule definition is **WHY-unclear
from static reading**: nothing in `customMainLoopSchedule` injects the
unreferenced instructions elsewhere, but I cannot rule out an alternate
emission site I haven't found. Worth flagging for follow-up runtime
instrumentation: dump the post-`scheduleInst` `InstStreams[k][1]` to confirm
unreferenced items are silently dropped.

CMS's structural validator (`CMSValidator.isValid`, CMSValidator.py:1313) does
not enforce "all source instructions are referenced by some miIndex in the
schedule," so the under-specification passes through without complaint.

### 4.6 n_gl UNKNOWN=47 in MT64x32x64 — capture-tagging coverage gap

**Class:** *capture-tagging coverage gap* (not a scheduler-divergence per se).

The 47 UNKNOWN items in default-side n_gl are all `ds_store_b16` /
`ds_store_b16_d16_hi` (i.e., LWA/LWB writes). The NGL/NLL capture path at
KernelWriter.py:3568-3580 builds `capture_id_to_cat` via
`build_id_to_category_per_iter` passing
`localWriteCode=self.codes.perIterLocalWrite[u][1]` — but inside
`_noLoadLoopBodyDefault` the kernel sets `self.codes.localWriteA =
self.localWriteDo(...)` at KernelWriter.py:3163,3172 to a FRESH `Module`,
whose leaves are NOT in `self.codes.perIterLocalWrite[u][1]`. Hence those
items reach `_captureSubIterToBuilder` without an id_to_category entry and
fall through the isinstance ladder to `category = "UNKNOWN"` (KernelWriter.py:2609).

This is a tagging-pipeline gap, not a scheduler difference; the CMS side
correctly tags them as LWA/LWB (via the schedule's `'LWA'`/`'LWB'` entries) so
none appear UNKNOWN there. The "+47 UNKNOWN" delta is therefore inflated by
this tagging gap rather than being a real coverage difference.

(I did not modify the tagging code — out of scope for this investigation. The
finding belongs in a separate beads-tracked ticket if it's worth fixing.)

## 5. Universality assessment

| Divergence class | MT128x64x64 (PGR=1, PLR=0) | MT64x32x64 (PGR=2, PLR=1) |
|---|---|---|
| 4.1 GRIncA/GRIncB un-scheduled | yes (+9/+9 in ML, ML-prev) | yes (+9/+9 in ML, ML-prev, n_gl) |
| 4.2 LCC CMS-only             | yes (-2 in ML, ML-prev) | yes (-2 in ML, ML-prev) |
| 4.3 SYNC inflation           | yes (+29 ML, +17 n_ll) | yes (+8 ML, +10 n_gl, +1 n_ll) |
| 4.4 macro-guard suppression  | yes (+8 LRA3, +4 LRB3 in n_ll) | n/a (PLR=1 — lastIter symmetric on both sides; see §2.2 footer) |
| 4.5 incomplete miIndex coverage | **no** — all LR/LW counts match per (LRA{u}, LRB{u}, LWA, LWB) | **yes** (the dominant divergence: +30/+30/+30/+30 LRA, +15×4 LRB, +28 LWA, +14 LWB across ML / n_gl / n_ll) |
| 4.6 UNKNOWN tagging gap | none observed | yes (+47 UNKNOWN in n_gl only) |

Classes 4.1, 4.2, 4.3 are **general** to the CMS-vs-default capture model and
will appear in any CMS=1 kernel. Class 4.4 only fires when `usePLR=0` macro
flag combinations expand the n_ll body (i.e., PLR=0 kernels). Class 4.5 is
**kernel-specific** to MT64x32x64 in this yaml; MT128x64x64's schedule
exhaustively references its source streams and shows no LR/LW-count divergence.

**CDNA 4 (gfx950) spot-check:** not performed — the sparse checkout in this
worktree does not include `library/src/.../Tensile/Logic/asm_full/gfx950/`
(only gfx1xxx variants are present). Whether Class 4.5 also appears on CDNA 4
schedules is an open question; recommend running this same probe against a
gfx950 yaml in a non-sparse checkout to settle it.

## 6. Typology of divergence classes

The five named classes, with where each applies in our data:

| # | Class name | Gist | Instances in data |
|---|---|---|---|
| 4.1 | un-scheduled source-module category | `optSchedule` simply lacks a key for a category present in `idMap`; items never enter the macro | GRIncA + GRIncB across both kernels (180 total: 9+9 × 2 kernels × 2 ML/MLp + 9+9 × MT64 n_gl) |
| 4.2 | CMS-only loop-counter machinery | `loopCounterCode` plumbed only through CMS path; default-side `build_idmap` passes empty | LCC (4 total: 2 × 2 kernels in ML/MLp; mirrors CMS-only-2 in our deltas) |
| 4.3 | scheduler-injected sync inflation | SIA3 per-iter `_wait`/dscnt-FIFO emits N waits; CMS hand-tunes a minimal `syncTable` | every CMS=1 kernel/body; biggest in MT128x64x64 ML (+29) |
| 4.4 | macro-guard suppression in n_ll | `usePLR=0` expansion drops `\\usePLR==1`-guarded LRA{lastIter}/LRB{lastIter}; default doesn't have the guard | MT128x64x64 n_ll (LRA3 +8, LRB3 +4) |
| 4.5 | incomplete miIndex coverage | CMS `optSchedule[k]` references fewer `instructionList[idx]` than the kernel emits; remainder silently dropped | MT64x32x64 only — dominates the +246 ML, +256 n_gl, +136 n_ll deltas |
| 4.6 | capture-tagging coverage gap (separate from scheduling) | NGL/NLL capture path can't tag locally-rebound `self.codes.localWriteA/B` items; they fall through as UNKNOWN | MT64x32x64 n_gl (+47 UNKNOWN) |

Class 4.6 is a side-finding about the *capture pipeline*, not the *scheduler
divergence*; included for completeness.

## 7. Probe extension and reproducibility

This investigation added one probe script:
`projects/hipblaslt/tensilelite/Tensile/Components/GFX1151_AUDIT/lr_divergence_capture_probe.py`.

It hooks `KernelWriter.kernelBody` to snapshot the `_capture_context.{default,cms}`
`FourPartCapture` per kernel, no-ops `compare_graphs`/`build_dataflow_graph`/
`validate_edge_wait_coverage` so the build runs past `CaptureConsistencyError`,
and writes `/tmp/lr_divergence_probe_report.json` (~700 KB) with each
`TaggedInstruction`'s `(category, slot, rocisa class name, asm-rendered string)`
on both sides per body.

**Determinism check:** two consecutive runs of the probe produced byte-identical
JSON reports (`md5sum` confirmed identical). Counts in §2 and diff lists in §3
are therefore reproducible from a fresh rebuild.

**Reproducer:**

```
env -C projects/hipblaslt/tensilelite \
  PYTHONPATH=projects/hipblaslt/tensilelite \
  python3 Tensile/Components/GFX1151_AUDIT/lr_divergence_capture_probe.py
```

The analysis script `/tmp/analyze_lr_divergence.py` (single-shot helper, not
committed; counted ~80 lines) consumes the probe's JSON and emits the per-
kernel/body/category tables in §2 and the only-default/only-CMS lists in §3.

## 8. Adversarial self-review

> Per bead instructions: critical-review pass before commit. Subagents cannot
> spawn an agent for review, so this is in-line.

1. **Is every WHY claim grounded in source citation, or smuggled speculation?**
   Re-read §4. Every WHY paragraph cites file:line. The single explicit
   "WHY-unclear" admission is at §4.5: I cannot rule out an alternate emission
   path for the un-referenced LR/LW source items in MT64x32x64 without runtime
   instrumentation. Marked as such, not guessed.

2. **Are the count tables actually FROM the probe data, or eyeballed?**
   §2 tables come from `/tmp/lr_divergence_analysis.json`, generated by
   `/tmp/analyze_lr_divergence.py` over `/tmp/lr_divergence_probe_report.json`.
   The probe's per-kernel `[PROBE-KERNEL]` lines (in `/tmp/lr_probe_run.log`)
   independently confirm `ml_def=156 ml_cms=111` for MT128x64x64 and
   `ml_def=285 ml_cms=39` for MT64x32x64. No eyeballing.

3. **Did I cover both kernels, or only the easier one?** Both. §2.1 + §2.2;
   §5 universality table walks each class through both. §3 detailed diff
   lists are MT128x64x64-only as the bead permits ("for at least
   MT128x64x64"); MT64x32x64 has its diff data available in the JSON for
   anyone who wants the per-instruction list.

4. **Did I produce a typology, or just a flat list of differences?** §6 names
   five classes (4.1–4.5 scheduler-divergence, 4.6 capture-tagging side
   finding) with one-row gist + instance counts. The five classes account for
   100% of the deltas observed in §2.

5. **Are diff lists deterministic (same instructions every run)?** Yes — see
   §7 md5 check. Two consecutive probe runs produced byte-identical JSON.
   The analysis script is pure (no random ordering), so the §3 diff lists are
   reproducible.

**Concerns that remain (not blockers for an investigation-only bead):**
- The "+47 UNKNOWN" in MT64x32x64 n_gl is partly a tagging gap (§4.6) versus
  a scheduler difference; I noted this but did not deduplicate against
  Class 4.5. The honest read: SOME of those 47 UNKNOWN items are also
  affected by Class 4.5 (i.e., they wouldn't appear in CMS even if tagged
  correctly), but the UNKNOWN counter conflates the two. A more refined
  decomposition would require fixing the tagging gap first.
- CDNA 4 universality unverified due to sparse checkout (§5).
- MT64x32x64 §4.5's "WHY unclear" footnote — runtime confirmation needed
  that no other code path picks up the dropped LR/LW items.

These are documented openly above and not papered over.
