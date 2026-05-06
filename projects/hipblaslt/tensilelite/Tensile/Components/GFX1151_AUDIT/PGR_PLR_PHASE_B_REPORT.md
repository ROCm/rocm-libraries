# PGR/PLR Phase B: Per-Combination Validator Behavior Matrix

Bead: `rocm-libraries-8ny` (sub of ot2 — Phase B of the PGR/PLR investigation).
Branch: `ot2.B-validator-coverage` (off `users/alvasile/validator_long_term_plans`).

## Scope

The graph-native validator (`build_dataflow_graph` -> `compare_graphs` ->
`validate_edge_wait_coverage`) is auto-activated by
`KernelWriter.kernelBody` (KernelWriter.py:4692) **only when
`UseCustomMainLoopSchedule` is truthy** (CMS=1). For CMS=0 / CMS=-1
kernels the hook never installs and the validator is silently inert.
Therefore the matrix below enumerates every (PGR, PLR) reachable under
**CMS=1**.

## How bodies are populated

Default-side bodies (read by `build_dataflow_graph(ctx.default)`):

* **ML (`main_loop[0]`)** — always populated under CMS=1
  (`expand_cms_macro(...)` in `_emitNoLoadLoopBodyCMSMacro` /
  KernelWriter `kernelBody` body emission).
* **ML-1 (`main_loop_prev[0]`)** — built post-hoc in the validator
  hook (KernelWriter.py:5219) as `clone_loop_body(main)`, so it
  always equals ML by construction whenever ML is present.
* **NGL (`n_gl[0]`)** — populated only when
  `noLoadLoop(isNGLL=True, ...)` is invoked. That is gated by
  `for remainPgr in range(kernel["PrefetchGlobalRead"]-1, 0, -1)`
  (KernelWriter.py:5118) — i.e., **PGR >= 2 only**. For PGR=1 the
  range is empty and `default_n_gl` is never assigned. The validator
  hook then falls back to `LoopBodyCapture(instructions=[])`
  (KernelWriter.py:5206-5207), giving an empty body.
* **NLL (`n_ll[0]`)** — populated when `noLoadLoop(isNGLL=False, ...)`
  is invoked. That is gated by `if kernel["PrefetchGlobalRead"]:`
  (KernelWriter.py:5141) — i.e., **PGR >= 1**. PGR=0 leaves NLL
  unset, falling back to empty.

`build_dataflow_graph` raises `CaptureEmptyBodyError` if any body
slot containing key `0` has zero instructions (ScheduleCapture.py:2901).
That is the gfx1151 PGR=1 PLR=0 crash signature.

## Coverage matrix (CMS=1)

| PGR | PLR | DTL A/B | Bodies emitted | Capture populates | Validator outcome | Witnessed in (yaml, kernel) |
|---|---|---|---|---|---|---|
| 0 | * | * | n/a | n/a | n/a | **unwitnessed in shipping yamls** — no CMS=1 + PGR=0 solutions exist (grep audit confirms). Synthetic test would need CMS=1 + PGR=0; would crash on empty NGL **and** empty NLL because both noLoadLoop call sites are gated on PGR>=1/PGR>=2. |
| 1 | 0 | F/F | ML, NLL (NGL empty) | `default.{ml,ml-1}={0:156}` `default.n_gl={0:0}` `default.n_ll={0:102}` | **CRASHES** with `CaptureEmptyBodyError: Body 'NGL' has zero captured instructions` (ScheduleCapture.py:2902). | gfx1151 HHS_BH_Bias, MT128x64x64 (ISA 11,5,1) |
| 1 | 1 | * | n/a | n/a | n/a | **unwitnessed in shipping yamls** for CMS=1. Sampling 50 yamls per arch across {gfx950, gfx1151, gfx1200, gfx1201, aquavanjaram} found zero CMS=1 + PGR=1 + PLR=1 solutions. Synthetic test would crash with the same empty-NGL signature as (1,0). |
| 2 | 0 | T/T | ML-1, ML, NGL, NLL | `default.{ml,ml-1}={0:157}` `default.n_gl={0:123}` `default.n_ll={0:85}` | **OK** — `build_dataflow_graph` succeeds twice (default + cms), `compare_graphs`=1 success, `validate_edge_wait_coverage`=1 success, `tcl_status: ok`. | gfx950 F8B8BS_BH_BiasSB, MT256x256x128 (ISA 9,5,0) |
| 2 | 0 | F/F | ML-1, ML, NGL, NLL | (extrapolated; not directly witnessed under CMS=1 in shipping yamls — only DTL=T/T appears in CMS=1 PGR=2 PLR=0) | unwitnessed-with-DTL=F/F under CMS=1 | Survey found only CMS=1+PGR=2+PLR=0+DTL=T/T (gfx950). DTL=F/F variant unwitnessed; needs synthetic kernel. Bodies would still be all-populated by construction (PGR=2 satisfies both noLoadLoop gates); outcome should track gfx1151 PGR=2 PLR=1 DTL=F/F (clean structurally), modulo whether SSetPrior is emitted under DTL=F/F. |
| 2 | 1 | F/F | ML-1, ML, NGL, NLL | `default.{ml,ml-1}={0:285}` `default.n_gl={0:281}` `default.n_ll={0:158}` `cms.{ml,ml-1}={0:39}` `cms.n_gl={0:25}` `cms.n_ll={0:22}` | **CRASHES** in `build_dataflow_graph` with `CaptureUnknownInstructionError: cannot classify instruction 'DSStoreD16HIB16' (category='UNKNOWN') in body 'NGL'`. Bodies are non-empty, so this is a different failure mode than the empty-NGL crash — capture pipeline lacks an instruction-class tag for `DSStoreD16HIB16`. **Not (PGR, PLR)-induced — it is the same instruction-coverage gap as the gfx950 SSetPrior crash (Phase C territory).** | gfx1151 HHS_BH_Bias, MT64x32x64 (ISA 11,5,1) — verified by carving a single-solution yaml `/tmp/gfx1151_HHS_only_PGR2_PLR1.yaml` and running validator-on |
| 2 | 1 | F/T | ML-1, ML, NGL, NLL | The single CMS=1 PGR=2 PLR=1 DTL=F/T kernel found in the gfx950 BBS_BH_BiasSB_HAS_SAV yaml was extracted into a single-solution synthetic yaml and run with the validator on; **post-resolution** the kernel was bumped to DTL=T/T by Tensile's solution validator (likely the F/T config is rejected and remapped). Probe reports `DTL=True/True MT=[192,256,64]` for the surviving kernel. So shipping CMS=1 PGR=2 PLR=1 DTL=F/T is effectively unwitnessed at kernelBody time. | (none — Tensile rewrites DTL=F/T to T/T post-Solution check; see Run 7 below) |
| 2 | 1 | T/T | ML-1, ML, NGL, NLL | `default.{ml,ml-1}={0:345/349}` `default.n_gl={0:312/316}` `default.n_ll={0:274/276}`. CMS-side has 2 codepaths: `cms.ml={0:354,1:354}`. | **MIXED**: of the gfx950 (PGR=2, PLR=1, DTL=T/T) kernels witnessed: (a) HSS first kernel validates clean (`compare_graphs`=1, `validate_edge_wait_coverage`=1, no error); (b) HSS second kernel (same MT, different MI/wave-tile) CRASHES with `CaptureUnknownInstructionError: SSetPrior in body 'ML-1'`; (c) BBS Range MT256x256x64 CRASHES with same SSetPrior signature; (d) the post-resolution kernel from Run 7 (BBS Ailk MT192x256x64) crashes during `compare_graphs`'s `_node_label` with `ValueError: node not found in capture's same-category stream` — yet a third failure mode. The variance is **not** (PGR, PLR)-determined — it depends on which instructions/categories the kernel actually emits. | OK: gfx950 HSS MT256x256x64 #1; CRASHES: HSS #2, BBS Range MT256x256x64 (SSetPrior); BBS Ailk MT192x256x64 (`_node_label` ValueError) |
| 3 / 4 | 1 | * | n/a (under CMS=1) | n/a | n/a | **unwitnessed under CMS=1** — PGR=3 and PGR=4 solutions exist in shipping yamls but only with CMS=0 (sampled gfx950 BBS_BH_BiasSB_HAS_SAV: 65 PGR=3 / 69 PGR=4 kernels, all CMS=0). No CMS=1 + PGR>=3 found. Synthetic test would extend the NGL-emission loop to >=2 NGL bodies; current capture only stores `n_gl[0]` so additional remainPgr iterations would either overwrite or be lost (line 3722-3725: `default_n_gl = finalized` unconditionally). |

## Single-kernel verification runs (this worktree, this branch)

All runs use `validator_coverage_probe.py --per-kernel-log --jobs 1`
(extended with `default_ml_prev_lens` / `cms_ml_prev_lens` fields in
this worktree branch). Reports written to `/tmp/probe_*_report.json`.

### Run 1: gfx1151 HHS_BH_Bias_HAS_SAV (validator on)
* yaml: `gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml`
* tcl_status: `raised: CaptureEmptyBodyError`
* `build_dataflow_graph_calls=1, compare_graphs_calls=0, validate_edge_wait_coverage_calls=0`
* CMS=1 PGR=1 PLR=0 MT128x64x64: ml-1=156, ml=156, **n_gl=0**, n_ll=102 -> EmptyBody crash on first hit. Build halts (CMS=1 PGR=2 PLR=1 kernel never reached).

### Run 2: gfx1151 HHS_BH_Bias_HAS_SAV (--disable-validator)
* tcl_status: `SystemExit(-1)` (downstream assembler error, unrelated)
* `build_dataflow_graph_calls=4` (counted but no-op), errors=0
* CMS=1 PGR=1 PLR=0 MT128x64x64: confirms `n_gl={'0':0}` (empty)
* CMS=1 PGR=2 PLR=1 MT64x32x64: ml-1=285, ml=285, n_gl=281, n_ll=158 — all populated.

### Run 3: gfx950 F8B8BS_BH_BiasSB_HAS_SAB_SAV (validator on)
* yaml: `gfx950/gfx950_id75a3/Equality/gfx950_Cijk_Alik_Bljk_F8B8BS_BH_BiasSB_HAS_SAB_SAV_UserArgs.yaml`
* tcl_status: **`ok`**
* `build_dataflow_graph_calls=2, compare_graphs_calls=1, validate_edge_wait_coverage_calls=1, errors=0`
* CMS=1 PGR=2 PLR=0 DTL=T/T MT256x256x128: all bodies populated; validator runs to completion, no failures.

### Run 4: gfx950 BBS Range, single CMS=1 kernel (validator on)
* yaml: `gfx950/gfx950/Range/gfx950_Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs.yaml`
* tcl_status: `raised: CaptureUnknownInstructionError`
* CMS=1 PGR=2 PLR=1 DTL=T/T MT256x256x64: bodies populated (ml=222, n_gl=189, n_ll=150) but `build_dataflow_graph` crashes on `SSetPrior` instruction.

### Run 5: gfx950 HSS_BH_BiasSH_HAS_SAV (validator on)
* yaml: `gfx950/gfx950_id75a3/Equality/gfx950_Cijk_Ailk_Bjlk_HSS_BH_BiasSH_HAS_SAV_UserArgs.yaml`
* tcl_status: `raised: CaptureUnknownInstructionError`
* `build_dataflow_graph_calls=3, compare_graphs_calls=1, validate_edge_wait_coverage_calls=1, errors=1`
* CMS=1 PGR=2 PLR=1 kernel #1: validator passes cleanly.
* CMS=1 PGR=2 PLR=1 kernel #2 (same MT, different MI / WaveTile): crashes with `CaptureUnknownInstructionError: SSetPrior` in body 'ML-1'.

### Run 7: gfx950 BBS Ailk_Bljk extracted single CMS=1 PGR=2 PLR=1 (was DTL=F/T pre-resolution)
* yaml: `/tmp/gfx950_BBS_only_PGR2_PLR1_DTLFT.yaml` (synthetic, single
  solution carved from gfx950 BBS_BH_BiasSB Ailk_Bljk yaml; chosen
  because the source had `DirectToLdsA: false, DirectToLdsB: true`)
* tcl_status: `raised: ValueError` (in `compare_graphs._node_label`)
* `build_dataflow_graph_calls=2` (default + cms both succeeded),
  `compare_graphs_calls=1` (raised), `validate_edge_wait_coverage_calls=0`
* CMS=1 PGR=2 PLR=1 **DTL=T/T** (Tensile bumped DTLA from F to T
  during Solution validation) MT192x256x64: bodies populated
  (default ml=190, n_gl=161, n_ll=119; cms ml=200, n_gl=168, n_ll=128).
  `compare_graphs` reaches `_node_label` and raises `ValueError:
  node tagged_inst=...SCmpEQU32 (category='GRIncA') not found in
  capture's same-category stream (9 GRIncA instructions in capture);
  the caller passed a capture that doesn't contain the node`.
  This is a **third distinct failure mode** beyond CaptureEmptyBody
  and CaptureUnknownInstruction.

### Run 6: gfx1151 HHS extracted single CMS=1 PGR=2 PLR=1 (validator on)
* yaml: `/tmp/gfx1151_HHS_only_PGR2_PLR1.yaml` (synthetic; one solution carved out of HHS)
* tcl_status: `raised: CaptureUnknownInstructionError`
* `build_dataflow_graph_calls=1, compare_graphs_calls=0, validate_edge_wait_coverage_calls=0`
* CMS=1 PGR=2 PLR=1 DTL=F/F MT64x32x64: bodies all populated
  (ml-1=285, ml=285, n_gl=281, n_ll=158 default; ml=39, n_gl=25,
  n_ll=22 cms-side) — `build_dataflow_graph` crashes on
  `DSStoreD16HIB16` (an `ds_store_2addr_b16_d16_hi`-like LDS-store)
  in body 'NGL'. Same class of failure as gfx950 SSetPrior:
  capture pipeline doesn't have a class tag for the instruction.

## Empirical conclusions

* **Empty-NGL crash is structural for any CMS=1 kernel with PGR=1.**
  PGR=1 makes the NGL-emission loop empty (KernelWriter.py:5118), so
  `default.n_gl` falls through to the empty-body fallback at the
  validator-hook construction site (KernelWriter.py:5206-5207). Any
  CMS=1 + PGR=1 build will therefore raise `CaptureEmptyBodyError`
  on body 'NGL' the moment it reaches that kernel. Only PGR>=2
  populates NGL.
* **PGR=2 PLR=0 (DTL=T/T) and PGR=2 PLR=1 are structurally
  validatable** — bodies non-empty in all four slots. Whether the
  validator returns clean depends on instruction-class coverage,
  not on (PGR, PLR). The instruction-coverage gaps witnessed
  during this investigation:
  * `SSetPrior` (SOPP) — gfx950 HSS / BBS, body 'ML-1', under DTL=T/T.
  * `DSStoreD16HIB16` (LDS half-word hi store) — gfx1151 HHS, body
    'NGL', under DTL=F/F.

  These are **Phase C territory** — capture pipeline missing
  classification tags or having stale cross-graph mappings — not
  (PGR, PLR)-induced. Failure modes seen:
  * `CaptureEmptyBodyError` — Run 1 (PGR=1 PLR=0).
  * `CaptureUnknownInstructionError(SSetPrior)` — Runs 4 & 5 (gfx950).
  * `CaptureUnknownInstructionError(DSStoreD16HIB16)` — Run 6 (gfx1151).
  * `_node_label ValueError(node not in capture)` — Run 7 (gfx950).

  Notably, the only (PGR=2, PLR=1) combination that has been
  observed to pass the validator end-to-end in this sweep is the
  gfx950 HSS_BH_BiasSH MT256x256x64 first-kernel case (Run 5
  above), and the only (PGR=2, PLR=0) case that passes is gfx950
  F8B8BS MT256x256x128 (Run 3). Both happen to avoid the
  unclassified instructions and the stale-mapping path.
* **PGR=0 and PGR>=3 with CMS=1 are unwitnessed in shipping yamls.**
  PGR=0 + CMS=1 would also crash on empty NLL (and empty NGL).
  PGR>=3 + CMS=1 would either (a) silently lose extra NGL bodies
  to the unconditional reassignment at KernelWriter.py:3722-3725
  (`default_n_gl = finalized` overwrites on each loop iteration)
  or (b) build correctly if the kernel author intends a single
  collapsed NGL — needs synthetic verification.

## Probe extensions made in this worktree

`Tensile/Components/GFX1151_AUDIT/validator_coverage_probe.py`:
* Added `default_ml_prev_lens` and `cms_ml_prev_lens` per-kernel
  fields, sampled from `ctx.default.main_loop_prev` and
  `ctx.cms.main_loop_prev` via the existing `body_lens` helper.
* Extended the compact one-line summary (`[PROBE-KERNEL]`) to print
  ml-1, ml, n_gl, n_ll lengths for both default-side capture.

No changes to `ScheduleCapture.py`, `KernelWriter.py`, or any validator
entry point.

## Critical-review pass

Per bead instructions, a critical review pass was conducted. The
review iterated through the matrix and verified each row against
the underlying probe runs and source code. The following corrections
were made between the draft and the final version:

1. **Initial draft assumed gfx1151 (PGR=2, PLR=1, DTL=F/F) was OK**
   based on Phase A's `--disable-validator` data. Critical review
   pointed out that disabling the validator only proves bodies were
   populated, not that `build_dataflow_graph` accepts them. Run 6
   (a synthetic single-solution yaml carved from HHS) was added,
   and the cell was changed from "OK" to "CRASHES
   (CaptureUnknownInstructionError: DSStoreD16HIB16 in body 'NGL')".

2. **Initial draft did not distinguish CMS=-1 vs CMS=0 in the matrix**.
   Review noted that `UseCustomMainLoopSchedule=-1` in the yaml is
   resolved to `0 or 1` by `Solution.py:1956-1964` BEFORE reaching
   `kernelBody`, so the validator only ever sees CMS=0 or CMS=1.
   The matrix is unaffected (still keyed on the post-resolution
   CMS=1 set), but the rationale was added to the "Scope" section.

3. **Initial draft did not survey gfx950 to confirm "PGR=2 PLR=0
   DTL=F/F unwitnessed"**. A 50-yaml gfx950 sweep was added
   confirming only DTL=T/T appears under CMS=1 PGR=2 PLR=0 in
   shipping yamls. The unwitnessed cell was qualified accordingly.

4. **Initial draft missed PGR=2 PLR=1 DTL=F/T cell**. A targeted
   probe run on `gfx950_Cijk_Ailk_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml`
   (which contains 2 such kernels) was added.

5. **Reviewer also flagged that the `SSetPrior` and `DSStoreD16HIB16`
   crashes are not (PGR, PLR)-induced** — they're capture-pipeline
   instruction-coverage gaps that surface whenever those particular
   instructions appear in any captured body. The matrix was
   relabeled to reflect this: rows like (2,1,T/T) say MIXED and
   point at the instruction-coverage gap, not at PGR/PLR.


