# D3ZJ — Residual SCmpEQI32 / SSubU32 Divergence Investigation

**Bead:** `rocm-libraries-d3zj`
**Branch base:** `users/alvasile/validator_long_term_plans` (tip `b288591d2b`)
**Date:** 2026-05-12
**Scope:** Deliverable 2 — investigation only. **No fix applied.**

After Deliverable 1 (the `data_flow_instructions(body)` shared helper)
landed, the d3zj real-kernel xfailed tests
(`test_real_kernel_per_render_counts_match` and
`test_real_kernel_per_ordinal_logical_instruction_matches` in
`Tests/unit/test_dataflow_graph_emission_ordinal.py`) shed all
SWaitCnt / SBarrier / SNop / SSetPrior noise. The residual divergence
is **LCC** instructions only: `SCmpEQI32` (loop-counter compare) and
`SSubU32` (loop-counter decrement). This memo investigates the source
and proposes — but does not implement — solutions.

> [!IMPORTANT]
> The user's strict per-body LCC invariant
> (memo `LCC_AUDIT.md`-style framing, 2026-05-13)
> states: "Every loop body (ML-1, ML, NLL, NGL) MUST have exactly one
> SCmpEQI32 + one SSubU32. PRELOOP and POSTLOOP bodies must have zero."
> **The empirical NGL/NLL evidence in §1 below contradicts the loop-
> body half of that invariant for the canonical TF32 4×4 TN
> kernel: BOTH the CMS-side and the default-side captures observe
> NGL=0+0 and NLL=0+0.** This is a "STOP and surface" point per the
> dispatch instructions — see §2.

---

## 1. Per-body counts (Q1)

Empirical counts taken by probing
`real_kernel_capture_pair` for the canonical TF32 4×4 TN kernel
(`CANONICAL_TF32_4X4_TN_CONFIG` at
`Tests/unit/test_dataflow_graph_emission_ordinal.py:328`). Probe
implementation: a temporary pytest test using the same fixture; deleted
after data collection.

### 1.1 SCmpEQI32 totals per body

| body                        | CMS-side | default-side |
|-----------------------------|----------|--------------|
| PRO   (loop_index = -1)     | absent   | absent       |
| ML-1  (loop_index =  0)     | **1**    | **0**        |
| ML    (loop_index =  1)     | **1**    | **0**        |
| NGL   (loop_index =  2)     | 0        | 0            |
| NLL   (loop_index =  3)     | 0        | 0            |
| POST_LOOP                   | absent   | absent       |

### 1.2 SSubU32 totals per body (any operand)

| body                        | CMS-side | default-side |
|-----------------------------|----------|--------------|
| PRO   (loop_index = -1)     | absent   | absent       |
| ML-1  (loop_index =  0)     | **3**    | **2**        |
| ML    (loop_index =  1)     | **3**    | **2**        |
| NGL   (loop_index =  2)     | 2        | 2            |
| NLL   (loop_index =  3)     | 0        | 0            |
| POST_LOOP                   | absent   | absent       |

### 1.3 SSubU32 split by destination operand

The 2-vs-3 SSubU32 delta is more useful when split by destination —
only the LCC variant (decrementing `LoopCounterL`) is the loop-counter
operation. The other two are GR-increment-side `ShadowLimitA` /
`ShadowLimitB` decrements (NOT LCC; they ride global-read pointer
arithmetic, not loop-iteration semantics).

| body  | side    | `s_sub_u32 LoopCounterL, …, 1` | `s_sub_u32 ShadowLimitA, …, s68` | `s_sub_u32 ShadowLimitB, …, s68` |
|-------|---------|--------------------------------|----------------------------------|----------------------------------|
| ML    | default | 0                              | 1                                | 1                                |
| ML    | cms     | **1**                          | 1                                | 1                                |
| ML-1  | default | 0                              | 1                                | 1                                |
| ML-1  | cms     | **1**                          | 1                                | 1                                |
| NGL   | default | 0                              | 1                                | 1                                |
| NGL   | cms     | 0                              | 1                                | 1                                |

The ShadowLimit decrements agree on both sides for all bodies. **The
divergence is exactly the LCC pair (one SCmpEQI32 + one SSubU32 on
`LoopCounterL`) appearing in CMS-side ML and ML-1 but missing on the
default side.**

---

## 2. Invariant verification (Q2)

The user's invariant: "loop bodies (ML-1, ML, NLL, NGL) = 1 each;
non-loop bodies (PRO, POST_LOOP) = 0."

| body  | invariant says | CMS-side observed | default-side observed | verdict             |
|-------|----------------|-------------------|------------------------|---------------------|
| PRO   | 0 LCC pairs    | absent (== 0)     | absent (== 0)          | holds (vacuously)   |
| ML-1  | 1 LCC pair     | **1**             | **0**                  | **default violates**|
| ML    | 1 LCC pair     | **1**             | **0**                  | **default violates**|
| NGL   | 1 LCC pair     | **0**             | **0**                  | **BOTH violate**    |
| NLL   | 1 LCC pair     | **0**             | **0**                  | **BOTH violate**    |

**Two distinct violation patterns:**

1. **ML-1 and ML — asymmetric (default-side missing).** The CMS-side
   conforms; the default-side is missing the LCC pair. This is the
   real per-body comparison defect that's surfacing in the d3zj test
   failures.
2. **NGL and NLL — symmetric (both sides missing).** Both captures
   agree (0 + 0), so this would NOT show up as a per-(body, render)
   count mismatch. But it does refute the user's invariant for these
   two body labels. **STOP and surface — the invariant as stated does
   not match the production code's behavior.**

   The mechanistic explanation, surfaced for review: NGL ("no global-
   load loop") and NLL ("no-load loop") are *unrolled tail bodies*,
   not iterating loops. They execute at most once each as the kernel
   exits the main loop. The CMS macro that builds them passes
   `useLoop = 0` explicitly (see `KernelWriter.py:3134, 3137`:
   `"Code-path 0, useGR=0, usePLR=1, useGRInc=1, useLoop = 0"` for
   NGL and an analogous comment for NLL). On the default side, the
   `noLoadLoop(...)` calls likewise do not invoke `closeLoop(...)` for
   the tail bodies. So neither pipeline emits LCC for these bodies —
   not because of a defect, but because the bodies are not iterating
   loops in either codegen path.

   If the user nonetheless wants to enforce LCC presence in NGL/NLL
   (e.g. for a uniform per-body validation invariant), the principled
   fix would be to *change the codegen* (have the tail bodies emit a
   single iteration of LCC for symmetry), not to relax the
   validator. But the more likely reading is that the invariant
   statement should be narrowed to "ML-1 and ML must each have
   exactly one LCC pair; NGL, NLL, PRO, POST_LOOP have zero."

   **Per dispatch rule:** I have NOT silently substituted that
   narrower invariant. Surfaced for the user's review.

---

## 3. Divergence source (Q3) — ML-1 / ML asymmetric

### 3.1 Where the LCC pair is emitted on the default side

`KernelWriter.closeLoop()` at
`KernelWriter.py:9131` is the single entry-point for emitting the
loop-counter decrement + compare. For the canonical TF32 kernel
(unroll loop, `PrefetchGlobalRead == 2`, `AssertSummationElementMultiple
% (DepthU * 2) == 0`), the relevant code path is
`KernelWriterAssembly.closeLoop()` at
`KernelWriterAssembly.py:6814-6892`:

* `KernelWriterAssembly.py:6850-6856` — `SSubU32(dst=loopCounter, ..., 1)`
  (the LCC decrement on `LoopCounterL`).
* `KernelWriterAssembly.py:6857-6859` — `SCmpEQI32(src0=loopCounter,
  src1=hex(endCounter), ...)` (the LCC compare).

These are gated by `decCode` / `condCode` and added to the module at
`KernelWriterAssembly.py:6881-6882`.

### 3.2 Where this enters the CMS-side capture

`KernelWriter._loopBody()` at `KernelWriter.py:4598`:
```
optSchedule, numCodePath = customMainLoopSchedule(
    self, kernel, ..., MfmaCodeAllIters,
    self.closeLoop(kernel, tensorParametersA, tensorParametersB,
                   self.states.unrollIdx, False))
module.add(optSchedule)
```

`self.closeLoop(...)` is called HERE and its returned module is passed
as the last positional argument into `customMainLoopSchedule(...)`,
which folds it into the CMS macro. When the CMS macro is later expanded
to build `ctx.cms` (the CMS-side capture, populated at
`KernelWriter.py:5348`), the LCC pair lands inside the ML / ML-1 main
bodies.

### 3.3 Where this is MISSING on the default side

The default-side capture is finalized at `KernelWriter.py:4589-4592`,
**before** the `self.closeLoop(...)` call at line 4598:

```python
builder = self._capture_context.builder
if builder is not None:
  self._capture_context.default_main = builder.finalize()
  self._capture_context.builder = None

optSchedule, numCodePath = customMainLoopSchedule(
    self, kernel, ...,
    self.closeLoop(kernel, ..., self.states.unrollIdx, False))
```

So the default-side `ctx.default_main` (and from it,
`ctx.default.main_loop` / `main_loop_prev`) is finalized at a point
in the build pipeline that has not yet seen the loop-close emission.
The CMS-side capture, by contrast, sees it because `closeLoop(...)`'s
output is folded INTO the CMS macro that the CMS-side capture later
expands.

The asymmetry is not in the underlying kernel codegen — it is in the
**capture timing**. Both kernels (a real CMS kernel and the
hypothetical `UseCustomMainLoopSchedule=0` reference) emit a closeLoop
in their final assembly. The default-side capture pipeline simply does
not record it.

### 3.4 Note on the "true non-CMS reference kernel" framing

The comment at `KernelWriter.py:4571-4592` (the rocm-libraries-71hw
"Approach A" framing) describes the default-side capture as
"a true non-CMS reference kernel (UseCustomMainLoopSchedule=0)." That
comment matches the *intent* of the architecture, but the *current
implementation* of `default_main` is still a shadow capture taken from
the CMS-mode build: the builder is filled by `_makeSubIterSchedule` in
the CMS-mode pass, then finalized at line 4591 before the CMS-mode
`closeLoop` call. There is no separate `UseCustomMainLoopSchedule=0`
build feeding `default_main`. The d3zj LCC divergence is one symptom
of that gap.

---

## 4. Proposed solutions (Q4) — list, NOT pre-committed

Two principled directions, ordered by how closely they realize the
"true non-CMS reference kernel" framing already documented in
KernelWriter.py:4577-4584. **Pick after user review of the §2
invariant question.**

### Option A — Capture closeLoop into the default builder (small fix)

Move the `builder.finalize()` call from `KernelWriter.py:4591` to
**after** the `closeLoop(...)` emission has been recorded in the
builder, and ensure `closeLoop`'s output is fed into the same
per-iter capture machinery that recorded the rest of the main-loop
body.

- Sketch: have `_loopBody` synthesize the `closeLoop` module first,
  then either (a) feed it through a one-off
  `_makeSubIterSchedule`-style pass that tags the LCC instructions
  into the appropriate body / sub-iter, OR (b) directly add the LCC
  TaggedInstructions to the builder at a synthetic terminal sub-iter.
  Then call `builder.finalize()`. Then call `customMainLoopSchedule()`
  with the same `closeLoop` module so CMS gets its copy too.
- Pros: minimal change; restores symmetry; default-side and CMS-side
  both capture LCC via the same emission. Closes the d3zj defect for
  ML-1 and ML.
- Cons: keeps the current shadow-capture pipeline (does not realize
  the "true non-CMS reference kernel" framing). Does NOT address the
  NGL/NLL invariant question (see §2) — but if the user agrees that
  invariant should be narrowed, that's an artifact, not a defect.

### Option B — Realize the "true non-CMS reference kernel" pipeline (architectural)

Implement the framing already documented at
`KernelWriter.py:4577-4584`: build the default-side capture from a
genuine separate `UseCustomMainLoopSchedule=0` invocation of
`KernelWriter._loopBody()`, in which `closeLoop(...)` is added to
the module at `KernelWriter.py:4611` (the existing non-CMS path)
and the builder records it as part of the natural mainloop emission.

- Pros: matches the documented architectural intent; the two captures
  become symmetric across the entire mainloop body — no need for a
  per-feature "did the shadow pipeline see this?" audit. LCC is one
  example; future divergences in other categories (e.g. tail-loop
  branches, expert-scheduling-mode RegSet writes) get the same
  symmetric treatment for free.
- Cons: substantially larger refactor; the second build adds
  build-time cost; needs a careful `_capture_context` lifecycle
  redesign to avoid cross-contamination between the two builds.

### Decision pre-requisite: NGL/NLL invariant

Before committing to either option, the user should resolve the §2
NGL/NLL invariant question:

- **If the invariant should be narrowed to "ML-1 and ML only":**
  Option A (or B) plus a one-line invariant doc-update suffices.
- **If the invariant should be enforced as stated (all four loop
  bodies):** the principled fix is to change the codegen so NGL and
  NLL each emit one LCC pair, then either Option A or B for the
  capture symmetry. This is a much larger surface — it would change
  the actual emitted assembly, not just validation.

---

## 5. References

- `EMISSION_ORDINAL_DESIGN.md` §2.6 — determinism invariant on the
  real-kernel capture pipeline.
- `EMISSION_ORDINAL_DESIGN.md` §6.2 — regression-test list (#1 and #2
  are the d3zj-pinned tests).
- `KernelWriter.py:4571-4598` — default-side capture finalize timing
  vs. CMS macro composition.
- `KernelWriter.py:4577-4584` — the rocm-libraries-71hw "Approach A"
  framing (the documented intent for a true non-CMS reference kernel).
- `KernelWriterAssembly.py:6814-6892` — the LCC emission site
  (SSubU32 + SCmpEQI32 inside `closeLoop`).
- `KernelWriter.py:3134, 3137` — CMS NGL/NLL macros pin
  `useLoop = 0` (the mechanistic basis for §2 NGL/NLL = 0).
- `Components/CMSValidator.py:1571-1576` —
  `_NO_DATAFLOW_IDENTITY_CATEGORIES`.
- `Components/CMSValidator.py:~1592` (post-d3zj) — the new
  `data_flow_instructions(body)` helper.
- `Components/CMSValidator.py:~1979` — Phase 1 site that consumes the
  helper.
- `Tests/unit/test_dataflow_graph_emission_ordinal.py:~410, ~470` —
  the two xfailed tests now driven by `data_flow_instructions(body)`.
- LCC_AUDIT.md was referenced by the dispatch but not present in the
  worktree (`find -iname '*lcc*'` returned only
  `Tests/unit/test_dataflow_graph_lcc.py`). If LCC_AUDIT.md exists on
  another branch, this memo should be cross-linked when this work
  lands.
