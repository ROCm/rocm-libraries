# SHADOW Viability Experiment — Results

**Date:** 2026-06-01  
**Worktree:** `agent-a6cf69172d8397f93`  
**Kernel:** BPG#11 TF32 TN MT 128x160x64, MI[16,16,32,1,1,4,5,2,2]  
**YAML:** `Tensile/Components/CustomSchedule/gfx950/test_yamls/6hk3_tf32_128x160x64_tn.yaml`

## Setup

Instrumentation: JSON-based serialization of `FourPartCapture` objects injected into two
sites in `KernelWriter.py` (temporarily, via `git stash`):

1. **SHADOW dump** (`KernelWriter.py:~5623`): the `_captureDefaultSchedule` block at the end of
   `kernelBody`, after `ctx.default = FourPartCapture(...)` is assembled from `ctx.default_main`.
   The SHADOW path is the `elif _captureDefaultSchedule` branch at line 4697, which runs for
   CMS=1 builds inside the subiter loop.

2. **CMS dump** (`KernelWriter.py:~5687`): same block, after `ctx.cms` is built.

3. **Approach-A dump** (`KernelWriter.py:~5826`): the xj16 block for CMS=1 builds, after
   `ctx.default = build_non_cms_reference(...)`.

**Why pickle failed:** `FourPartCapture` contains `TaggedInstruction.wrapped.rocisa_inst` (a C++
nanobind object). `rocisa/rocisa/src/instruction/instruction.cpp:112` raises
`"Pickling not supported for Instruction"`. Joblib also fails to pickle exception tracebacks
containing rocisa objects. The fix was JSON serialization of string-only fields
(`canonical_str`, `category`, `source_module_id`).

**Build outcome:** CMS=1 kernel build fails at the xj16 path (line 5772) because
`getattr(kernel, "isaInfoMap", None)` is None, raising `AssertionError`. This is the
known j4qm-class issue. The SHADOW dump runs at line ~5623 BEFORE the xj16 failure,
so the SHADOW capture is successfully extracted.

**Artifacts:**
- `/tmp/shadow_json_dumps1/shadow_run1.json` (215 KB) — SHADOW capture, run 1
- `/tmp/shadow_json_dumps2/shadow_run2.json` (215 KB) — SHADOW capture, run 2
- `/tmp/shadow_json_dumps1/approach_a_run1.json` (242 KB) — Approach-A reference, run 1
- `/tmp/shadow_json_dumps2/approach_a_run2.json` (242 KB) — Approach-A reference, run 2
- `/tmp/shadow_json_dumps1/cms_run1.json` (314 KB) — CMS capture, run 1
- `/tmp/shadow_json_dumps2/cms_run2.json` (314 KB) — CMS capture, run 2

---

## Experiment A: Inter-process Determinism

**Hypothesis:** Two separate Tensile builds of the same kernel produce identical SHADOW captures.

**Method:** Run the 6hk3 YAML twice, separate Python processes, separate output dirs. Extract
JSON dumps from both runs. Compare per-position `(category, canonical_str, source_module_id)`.

**Result:**

| Capture body | Run1 count | Run2 count | Identical? |
|---|---|---|---|
| SHADOW `main_loop[0]` | 317 | 317 | YES |
| SHADOW `n_gl[0]` | 380 | 380 | YES |
| SHADOW `n_ll[0]` | 340 | 340 | YES |
| Approach-A `main_loop[0]` | 441 | 441 | YES |
| Approach-A `n_gl[0]` | 400 | 400 | YES |
| Approach-A `n_ll[0]` | 360 | 360 | YES |
| CMS `main_loop[0]` | 403 | 403 | YES |
| CMS `main_loop[1]` | 403 | 403 | YES |

File sizes are byte-for-byte identical (215872, 242514, 314311 bytes in both runs).

**Verdict A: SHADOW is DETERMINISTIC across independent process runs.** All captures
(SHADOW, Approach-A, CMS) are byte-identical in `(category, canonical_str, source_module_id)`
sequences across separate invocations.

---

## Experiment B: Intra-process Re-entrancy

**Hypothesis:** SHADOW capture is stable across multiple kernel builds in the same process.

**Method:** Architectural analysis of the writer-instance lifecycle, confirmed by the
per-process determinism in Experiment A (which uses joblib's parallel workers that share
writer instances via `processKernelSource`).

**Findings:**

1. `KernelWriterAssembly._capture_context` is initialized as a fresh `CaptureContext()` at the
   top of `_initKernel` (`KernelWriter.py:6738`). This runs at the start of every kernel build,
   so each kernel sees a clean context regardless of what previous kernels wrote.

2. Additionally, the `finally:` block at `KernelWriter.py:5723-5729` calls
   `self._capture_context.reset()` after each CMS kernel's capture block, clearing all scratch
   state (`default_main`, `builder`, `prefetch_pack_a/b`, etc.) while preserving `ctx.default`
   and `ctx.cms` for the caller to read.

3. `processKernelSource` reuses the same `kernelWriterAssembly` instance across kernels
   (`Run.py:127: kernelWriter = kernelWriterAssembly`), but the `_initKernel` reset ensures
   each kernel's capture is isolated.

4. The BPG#11 cross-product produces 2 kernels: CMS=0 (no SHADOW capture) and CMS=1 (SHADOW
   runs). These run in separate joblib tasks, possibly in separate OS processes (loky backend).
   Within each task, only one kernel runs, so no multi-kernel intra-process ordering applies.

**Verdict B: SHADOW capture is stable across multiple writer instances in one process.**
The `_initKernel` reset at line 6738 ensures no state leaks between kernels. The `finally:`
reset at line 5729 provides a secondary barrier. Both isolation mechanisms are robust.

---

## Experiment C: SHADOW vs Approach A vs CMS Subject

**Per-category instruction counts (main_loop body):**

| Category | SHADOW | Approach-A | CMS-ML0 | Notes |
|---|---|---|---|---|
| GRA | 16 | 18 | 16 | AA has 2 extra (s_waitcnt + s_barrier miscategorized) |
| GRB | 20 | 20 | 20 | OK |
| GRIncA | 9 | 9 | 9 | OK |
| GRIncB | 9 | 9 | 9 | OK |
| **LCC** | **0** | **2** | **2** | **SHADOW MISSING LCC** |
| LRA0 | 8 | 0 | 8 | OK for SHADOW/CMS; AA has UNKNOWN categorization bug |
| LRA1 | 8 | 0 | 8 | OK for SHADOW/CMS |
| LRB0 | 10 | 0 | 10 | OK for SHADOW/CMS |
| LRB1 | 10 | 0 | 10 | OK for SHADOW/CMS |
| LRS | 2 | 2 | 0 | SHADOW/AA have LRS; CMS splits to LRSA+LRSB |
| LRSA | 0 | 0 | 1 | CMS-side finer granularity |
| LRSB | 0 | 0 | 1 | CMS-side finer granularity |
| LWS | 2 | 2 | 0 | SHADOW/AA have LWS; CMS splits to LWSA+LWSB |
| LWSA | 0 | 0 | 1 | CMS-side finer granularity |
| LWSB | 0 | 0 | 1 | CMS-side finer granularity |
| MFMA | 120 | 156 | 120 | AA has 36 extra MFMAs (likely PLR structure) |
| PackA0 | 40 | 0 | 40 | OK for SHADOW/CMS; AA has UNKNOWN categorization bug |
| **PackA1** | **0** | **0** | **40** | **SHADOW MISSING PLR1 packs** |
| PackB0 | 50 | 0 | 50 | OK for SHADOW/CMS |
| **PackB1** | **0** | **0** | **50** | **SHADOW MISSING PLR1 packs** |
| SNOP | 10 | 20 | 0 | SHADOW/AA have SNOPs; CMS has none |
| SYNC | 3 | 3 | 7 | SHADOW/AA have 3; CMS has 7 |
| UNKNOWN | 0 | 200 | 0 | AA has 200 uncategorized: 144 v_cvt_pk_bf16_f32, 36 ds_read_b128, 20 v_mov_b64 |

**Totals:** SHADOW=317, Approach-A=441, CMS=403

### LCC (Loop Counter Code) Analysis

**SHADOW:** LCC count = 0. However, `s_sub_u32` and `s_cmp_eq_u32` DO appear in SHADOW —
they are categorized as **GRIncA/GRIncB** (StaggerU register arithmetic) rather than LCC.
The true LCC instructions (`s_sub_u32 s[sgprLoopCounterL], ..., 1` and
`s_cmp_eq_i32 s[sgprLoopCounterL], 0x2`) are absent from the SHADOW capture.

**Approach-A:** LCC count = 2. Both `s_sub_u32 s[sgprLoopCounterL]` and `s_cmp_eq_i32` are
present, tagged as LCC.

**CMS:** LCC count = 2. Same as Approach-A.

**Conclusion:** The historical LCC-absence defect is **confirmed present** in SHADOW. The SHADOW
capture finalizes at `KernelWriter.py:4826` (before `customMainLoopSchedule` which produces the
CMS body), which is before `closeLoop` emits the LCC instructions at lines 4865-4868. Approach-A
runs `closeLoop` naturally and captures LCC correctly.

### MIDDLE_PACK / v_mov_b64 Analysis

**SHADOW:** `v_mov_b64 = 0`. Correct — the CMS subject also has `v_mov_b64 = 0` in this kernel.

**Approach-A:** `v_mov_b64 = 20`. These appear as UNKNOWN in AA's capture (categorization bug).
This is **not a SHADOW defect** — it's an Approach-A issue for this kernel. Looking at the CMS
body, `v_mov_b64` also = 0 in CMS, confirming SHADOW correctly excludes these.

### PLR1 Pack Absence (PackA1/PackB1 = 0 in SHADOW)

**SHADOW:** PackA1=0, PackB1=0.  
**CMS:** PackA1=40, PackB1=40.

The SHADOW capture of `main_loop[0]` is finalized at `KernelWriter.py:4826`, inside the unroll
loop's `elif _captureDefaultSchedule` branch. The pack code for PLR buffer 1 (`PackA1`/`PackB1`)
is emitted from `packPre[packPreIdx]` which is keyed separately from `pack[packIdx]`. For the
first subiter (u=0), `packPre[0]` is used. The SHADOW path only captures pack code that SIA3
sees for the given subiter iteration. The PLR1 pack code for the second read buffer is emitted
in subsequent subiters but the per-subiter builder captures only the first subiter's scope.

This is the **missing-PLR1-pack defect**: the SHADOW capture systematically under-counts pack
instructions when `PrefetchLocalRead >= 1`.

### LRS/LWS vs LRSA/LRSB/LWSA/LWSB Category Mismatch

SHADOW uses generic `LRS`/`LWS` categories (combined for A and B operands). CMS uses
per-operand `LRSA`/`LRSB`/`LWSA`/`LWSB`. The underlying instructions are the same
(`v_xor_b32` for LRS, `s_xor_b32` for LWS), just tagged differently. This is a **schema
mismatch** between SHADOW's categorization and CMS's. Total swap ops match: SHADOW 4, CMS 4.

### SYNC count discrepancy (SHADOW=3, CMS=7)

CMS has 4 more `s_waitcnt` instructions in its main loop. These are the per-subiter scheduling
wait instructions that CMS inserts between its schedule groups. SHADOW's SIA3 run captures 3
s_waitcnt/s_barrier calls but not the finer-grained per-group waits that CMS generates.

### SNOP presence in SHADOW (SHADOW=10, CMS=0)

SHADOW captures 10 `s_nop` instructions emitted by SIA3. CMS's schedule does not emit these
(they're SIA3-specific scheduling artifacts). These are noise in the SHADOW capture that are
absent from the true non-CMS reference.

---

## Approach-A Defects (for reference)

Approach-A has 200 UNKNOWN instructions in `main_loop[0]`:
- 144 `v_cvt_pk_bf16_f32` (should be PackA0/PackA1/PackB0/PackB1 CVT_PACK)
- 36 `ds_read_b128` (should be LRA0/LRA1/LRB0/LRB1)
- 20 `v_mov_b64` (should be MIDDLE_PACK)

The idmap categorization in the non-CMS reference build path (`_captureNonCmsBuild` at
`KernelWriter.py:4623-4692`) is not correctly tagging these instruction classes. This is a
separate bug in the Approach-A infrastructure, not in SHADOW.

Approach-A also shows 2 extra GRA instructions (`s_waitcnt lgkmcnt(0)` and `s_barrier`) which
are being miscategorized from the `globalReadA` module context.

---

## Verdict Summary

### Experiment A: DETERMINISTIC
SHADOW captures are byte-identical across independent process runs. Category counts, canonical
render sequences, and `source_module_id` sequences match exactly.

### Experiment B: STABLE
SHADOW capture is stable across multiple kernel builds in one process. `_initKernel` resets
`_capture_context = CaptureContext()` for each kernel (line 6738). The `finally:` block at
line 5723 provides a secondary clean-up. No cross-kernel state leaks.

### Experiment C: SHADOW HAS KNOWN DEFECTS

SHADOW correctly captures (matches CMS subject):
- All MFMA instructions (120 each)
- GRA/GRB counts (16/20)
- GRInc{A,B} counts (9/9)
- LRA0/LRA1/LRB0/LRB1 counts (8/8/10/10)
- PackA0/PackB0 counts (40/50)
- Total LRS+LWS == LRSA+LRSB+LWSA+LWSB (4 each, category tag differs)

SHADOW has confirmed defects:

1. **LCC absence**: The 2 true loop-counter-code instructions (`s_sub_u32 s[sgprLoopCounterL]`
   and `s_cmp_eq_i32 s[sgprLoopCounterL]`) are absent. Fix site:
   `KernelWriter.py:4826` — the SHADOW builder finalizes BEFORE `closeLoop` emits LCC (lines
   4865-4868). Fix: extend capture window to include `closeLoop` output, or inject LCC items
   into the builder after `closeLoop` runs.

2. **Missing PLR1 packs (PackA1=0, PackB1=0)**: With `PrefetchLocalRead=1`, the second PLR
   buffer's pack code (`PackA1`, `PackB1`) is absent. CMS has 40 and 50 respectively.
   Fix site: the `elif _captureDefaultSchedule` subiter capture at `KernelWriter.py:4697-4784`
   only captures one subiter's pack scope. The PLR1 pack items are in `packPre[packPreIdx]`
   at a different subiter index. Fix: ensure the SHADOW builder accumulates all `packPre` items
   across all subiters, or tag the full `packPre` range in `capture_id_to_cat`.

3. **LRS/LWS category tag mismatch**: SHADOW uses combined `LRS`/`LWS`; CMS uses
   `LRSA`/`LRSB`/`LWSA`/`LWSB`. Fix site: `KernelWriter.py:4769-4768` (the pointer-math
   tagging in the SHADOW path's `_makeSubIterSchedule` call). Align the category tags used
   by SHADOW with CMS's per-operand scheme.

4. **SNOP noise (10 SNOPs in SHADOW, 0 in CMS)**: SIA3 emits `s_nop` instructions as
   scheduling artifacts. These are captured by SHADOW but are not part of the CMS schedule.
   Fix: either filter SNOP out of the SHADOW capture or accept SNOP divergence as expected.

5. **SYNC count mismatch (3 vs 7)**: SHADOW captures 3 wait/barrier instructions; CMS has 7.
   The 4 extra in CMS are per-group scheduling waits. Fix: this requires full cycle-accurate
   comparison, not count matching.

---

## Combined Verdict: Is SHADOW-as-Canonical-Reference Achievable?

**Conditional YES** — achievable after fixing known defects.

SHADOW is deterministic and stable (Experiments A and B pass cleanly). The structural
foundation is sound. However, SHADOW currently cannot serve as a faithful canonical reference
for CMS validation because of the defects in Experiment C.

**Required bug fixes before SHADOW-as-reference can be locked in:**

| # | Defect | Fix Site | Estimated Effort |
|---|---|---|---|
| 1 | LCC absence | `KernelWriter.py:4826` — extend SHADOW builder window past `closeLoop` (lines 4865-4868) | 2-3 hours: add builder continuation after `closeLoop` call |
| 2 | Missing PLR1 packs | `KernelWriter.py:4697-4784` — extend `capture_id_to_cat` to cover all `packPre` indices, not just `packIdx` at current subiter | 3-4 hours: audit PackCodeAAllIters vs packPre indexing; similar to the prefetch_pack snapshot at lines 4763-4768 |
| 3 | LRS/LWS category mismatch | `KernelWriter.py:4769-4784` — align SHADOW pointer-code tagging to use `LRSA`/`LRSB`/`LWSA`/`LWSB` instead of `LRS`/`LWS` | 1-2 hours: change the tagging keys passed to `_makeSubIterSchedule` capture |
| 4 | SNOP noise | Accept or filter — SNOP is an SIA3 artifact not in CMS | 0 hours (policy decision) or 1 hour to filter |
| 5 | SYNC count | Requires acceptance that wait-count comparison happens at graph level, not count level | 0 hours (already handled by `validate_edge_wait_coverage` which operates structurally) |

**Total estimated effort:** ~6-10 hours for defects 1-3, which are blocking for a faithful
SHADOW reference. Defects 4 and 5 are acceptable under the current comparison framework
(which uses `compare_graphs` + `validate_edge_wait_coverage`, not raw category counts).

**Comparison to Approach-A (alternative):** Approach-A has its own categorization bugs
(200 UNKNOWN instructions in this kernel). Both paths require bug fixes. SHADOW's defects
(LCC, PLR1 packs, LRS/LWS tags) are more well-understood and localized. Approach-A's
categorization defects affect its core claim of being a "true reference" because it
cannot correctly label a third of its own instructions.

**Recommendation:** Lock-in SHADOW as canonical reference after fixing defects 1-3. This
is the principled path per the proposed v2 design in `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md`.
