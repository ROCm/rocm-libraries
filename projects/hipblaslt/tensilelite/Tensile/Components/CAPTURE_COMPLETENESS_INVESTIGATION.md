# Capture Completeness Investigation
## BPG#11 TF32 4x4 TN Canonical Config — GR / LR-Prefetch Presence + cms0 Listing

---

## SUMMARY

**GR (buffer_load) and LR-prefetch (ds_read with usePLR) instructions are present in both the cms1 and shadow capture listings.** They are NOT missing from the dumper output and NOT dropped by the capture pipeline. The listings are complete.

Decisive evidence: the raw assembly MAINLOOP macro definition (cms1_kernel.s lines 1889–2238) contains 16 ds_read and 8 buffer_load instructions per body invocation. The cms1_capture_listing.txt and shadow_capture_listing.txt both report exactly 16 ds_read and 8 buffer_load per ML body (ML_PREV and ML), which are the only bodies that include GR (buffer_load is gated by `.if useGR==1` in the macro).

**The 16-EdgeRoutedDifferentlyFailure analysis and uvrl conclusions are NOT invalidated.** The captures are operating on the complete instruction stream.

**cms0 assembly-derived listing produced:** `Tensile/Components/kernel_comparison/cms0_capture_listing.txt`

---

## §1 The Two Gaps (Restated)

**Gap 1:** No body-segmented listing for cms0 (UseCustomMainLoopSchedule=0). Only `cms0_kernel.s` existed; `cms1_capture_listing.txt` and `shadow_capture_listing.txt` have body-ordered views, but cms0 had none.

**Gap 2:** User observed that cms1 and shadow capture listings appear to be missing local-read-prefetch (PrefetchLocalRead ds_reads) and global_read (buffer_load) instructions. This required empirical verification.

---

## §2 cms0 Capture Listing — How Produced, Caveats

### Why a plain cms0 build does NOT populate a FourPartCapture

For a kernel built with `UseCustomMainLoopSchedule=0`:
- `_captureDefaultSchedule` is NOT activated (auto-activation only fires for `UseCustomMainLoopSchedule=1`; see KernelWriter.py line 512).
- `_captureNonCmsBuild` is only armed by `enable_capture_non_cms_build()`, which is called from `approach_a.py:126` (`build_non_cms_reference`). A plain `_getKernelSource` call for a cms0 config leaves both flags unset.
- Therefore `writer._last_cms_capture` and `writer._last_default_capture` are both `None` after a cms0 `_getKernelSource`. There is no FourPartCapture to dump.

### Assembly-derived listing (option a)

Produced by parsing `cms0_kernel.s` into body sections by label, extracting instruction lines (filtering labels, blank lines, comments, assembler directives, macro calls), and assigning a sequential `mfma_index` counter based on `v_mfma_f32_16x16x32_bf16` count.

**Output:** `Tensile/Components/kernel_comparison/cms0_capture_listing.txt`

**Column format** (differs from cms1/shadow — .s_line, not listing file line):
```
<.s_line> | body | body_idx | mfma_index | seq | instruction
```

**Body boundaries from cms0_kernel.s** (verified against `hxcx_artifacts/cms0_bodies.txt`):

| Body | .s lines     | Source labels                              |
|------|--------------|--------------------------------------------|
| PRO  | 1821–1871    | s_waitcnt after LR → label_openLoopL       |
| ML   | 1872–2200    | label_LoopBeginL → label_LoopEndL          |
| NGL  | 2200–2511    | label_LoopEndL → label_toPGR1              |
| NLL  | 2511–3295    | label_toPGR1 → label_TailLoopBeginL (spans OptNLL + OrdNLL) |

**CAVEAT:** Unlike cms1/shadow which are FourPartCapture objects (Python-level instruction objects), this listing is derived from the assembled text. mfma_index is .s-line-based. The NLL body spans both OptNLL and OrdNLL code paths combined (two subpaths in cms1's capture are merged into one flat view here). The body boundary conventions differ from the capture-based listings.

**CAVEAT:** cms0 does NOT use the MAINLOOP macro. Its ML body instructions are emitted inline (fully expanded). The cms1 ML body is emitted via the macro expansion at lines 1889–2238 and called at line 2239 (main loop) and 2253 (NGL), 2285 (OptNLL), 2484 (OrdNLL) with different parameter sets.

---

## §3 GR / LR-Prefetch Presence Test — Empirical Counts

### cms1 MAINLOOP macro structure (cms1_kernel.s lines 1889–2238)

The macro is defined once and called four times with different parameters:

| Call site (line) | Purpose | useGR | usePLR |
|-----------------|---------|-------|--------|
| 2239 | ML repeating body | 1 | 1 |
| 2253 | NGL (NoGlobalLoad) | 0 | 1 |
| 2285 | OptNLL | 0 | 0 |
| 2484 | OrdNLL | 0 | 0 |

Within the macro definition:
- **8 ds_read** are unconditional (LR for current iteration, not gated on any parameter).
- **8 ds_read** are gated by `.if \usePLR == 1` — these are the **LR-prefetch** reads (PrefetchLocalRead, preparing data for the next iteration).
- **8 buffer_load** are gated by `.if \useGR == 1` — these are the **global reads** (DirectToLds).

### Per-body instruction counts: raw .s vs capture listings

**cms1_kernel.s** (raw assembly, ML body = MAINLOOP macro def lines 1889–2238):

| Body | raw .s ds_read | raw .s buffer_load | Notes |
|------|---------------|-------------------|-------|
| ML (macro def) | 16 | 8 | 8 current LR + 8 prefetch LR; 8 GR |
| NGL (useGR=0, usePLR=1) | 16 | 0 | 8 current + 8 prefetch; no GR |
| OptNLL (useGR=0, usePLR=0) | 8 | 0 | 8 current only |
| OrdNLL (useGR=0, usePLR=0) | comment only | 0 | macro call, 8 current |
| Pre-loop (preamble GR + initial LR) | 9 | 16 | initial prefetch global reads |

**cms1_capture_listing.txt** (FourPartCapture, verified per-body):

| Body | listing ds_read | listing buffer_load |
|------|----------------|---------------------|
| PRO | 0 | 0 |
| ML_PREV | **16** | **8** |
| ML | **16** | **8** |
| NGL | **16** | **0** |
| NLL | **8** | **0** |

**shadow_capture_listing.txt** (FourPartCapture, same verification):

| Body | listing ds_read | listing buffer_load |
|------|----------------|---------------------|
| PRO | 0 | 0 |
| ML_PREV | **16** | **8** |
| ML | **16** | **8** |
| NGL | **16** | **0** |
| NLL | **8** | **0** |

**Verdict: raw .s counts and capture listing counts match exactly for every body.** The captures include all LR-current, LR-prefetch, and GR instructions.

### cms0_kernel.s per-body counts (assembly-derived, no MAINLOOP macro)

| Body | ds_read | buffer_load | Notes |
|------|---------|-------------|-------|
| PRO (1821–1871) | 0 | 0 | pack CVTs + pack MFMAs only |
| ML (1872–2200) | 16 | 8 | 8 current + 8 prefetch LR; 8 GR |
| NGL (2200–2511) | 16 | 0 | 8 current + 8 prefetch LR; no GR |
| NLL (2511–3295) | 16 | 8 | Covers OptNLL(8 LR) + OrdNLL(8 LR + 8 tail GR) |

The cms0 ML body matches cms1 ML body: 16 ds_read, 8 buffer_load. cms0 does not use a macro, the instructions are emitted inline, but the counts are identical.

---

## §4 Dumper Bug Assessment

No dumper bug exists. The `_dump_capture_listing` function in `Tensile/Tests/unit/_dump_hxcx_assembly.py` (lines 126–190) iterates `lbc.instructions` directly — all TaggedInstructions — with no filter. The function does not exclude any instruction category. If an instruction is in the capture, it appears in the listing.

The per-body counts in the listing (56 total ds_read, 16 total buffer_load in cms1 listing) are correctly accounted for:
- 56 ds_read = PRO(0) + ML_PREV(16) + ML(16) + NGL(16) + NLL(8) = 56
- 16 buffer_load = PRO(0) + ML_PREV(8) + ML(8) + NGL(0) + NLL(0) = 16

No regeneration of existing listings was needed. The cms0 assembly-derived listing is new.

---

## §5 Capture-Pipeline Bug Assessment

No capture-pipeline bug. The `_captureSubIterToBuilder` function (KernelWriter.py lines 2708–2905) skips only `TextBlock` instances (labels, raw text comments). All rocisa `Instruction` subclasses including `BufferLoad` (buffer_load) and `DSLoad` (ds_read) are captured.

The `_NO_DATAFLOW_CATEGORIES` filter in ScheduleCapture.py (line 1900) covers SWAIT, SBARRIER, SNOP, SSETPRIO only. It has no effect on buffer_load (VECTOR_LOAD category) or ds_read (LDS category). Both categories go through `LoopBodyCaptureBuilder.append`.

**The capture is complete.** ds_read and buffer_load instructions at all body positions are present.

---

## §6 Verdict + Whether Prior Conclusions Stand

**The listings are complete.** GR (buffer_load DirectToLds) and LR-prefetch (the 8 ds_reads gated by `.if usePLR==1`) are present in both cms1 and shadow capture listings.

The 16-EdgeRoutedDifferentlyFailure analysis and the uvrl conclusions were drawn from a complete instruction stream. The ds_read/pack ordering story is valid: the capture includes all prefetch LRs and GRs. No prior conclusions are invalidated.

The only structural difference between cms0 and cms1/shadow that is relevant: cms0 emits instructions inline (fully expanded), while cms1 uses a parameterized MAINLOOP macro. The instruction content (and counts) are equivalent for the ML body.

---

## §7 Beads Filed

No beads filed. Both gaps were resolved without finding production defects:
- Gap 1 (cms0 listing): produced as an assembly-derived artifact; no capture code bug.
- Gap 2 (GR/LR-prefetch missing): empirically confirmed present; no dumper or capture bug.
