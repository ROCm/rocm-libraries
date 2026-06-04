# NLL Iteration Structure — Investigation

## Verdict

**NLL is a single-invocation body that contains an internal subiter axis of size `kernel["LoopIters"]`. NLL has NO outer "iteration count" parameter.** The "iter dimension" the unrolled-validation plan must model lives *inside* one NLL body as a sequence of `LoopIters` subiter chunks — driven by the `for uIdx in range(0, kernel["LoopIters"])` loop in `_noLoadLoopBodyDefault` (KernelWriter.py:3508), with category-trailing-digit subiter tags (`PackA0`/`PackA1`/.../`PackA{N-1}`) labelling positions inside the rotating pack buffer (NOT outer-loop iteration indices).

There IS a small, well-bounded multiplicity of *physical* `noLoadLoop(...)` calls within `kernelBody` (`NLLnum=2` for DTV, `PrefetchGlobalRead-1` extra NGLLs, optional second tailloopInNll set) — but none of these are the missing "NoLoadLoopIter" parameter the plan assumed. They are condition-driven, code-generation-time duplications of the same NLL body (NGLL vs NLL, even/odd LDS buffer variants), and each individual `noLoadLoop(...)` call still emits a single NLL body whose internal structure is the `LoopIters`-long subiter chain.

The single most actionable correction: replace any reference to `kernel["NoLoadLoopIter"]` in the plan with `kernel["LoopIters"]` used as the *intra-NLL subiter-chunk count*, and remove any data-model layer that imagines materializing multiple `NLL` copies. The SHADOW/CMS captures already store NLL as `{0: body}` — one body — and that single body's instruction stream already contains all `LoopIters` subiter chunks in stream order. No outer NLL-iter materialization is appropriate.

## Q1 — Does NLL iterate?

### `kernel["NoLoadLoopIter"]` does not exist

Grep across `Tensile/`:

```bash
$ grep -rn "NoLoadLoopIter" --include="*.py" .
(no matches)
```

The parameter the plan assumed is fictional.

### Real NLL-related parameters (file:line, from `Tensile/Common/ValidParameters.py`)

| Parameter | Line | Values | What it controls |
|---|---|---|---|
| `SuppressNoLoadLoop` | 292 | `[False, True]` | If True, skip the NLL emission entirely (use buffer-limit suppression of GR + ignore unused DS-reads instead of generating a copy of the loop) |
| `PrefetchGlobalRead` | 251 | `[0,1,2,3..16]` | Number of prefetched global-read iterations; **controls how many NGLLs are emitted (one per `PrefetchGlobalRead - 1`)** and how many NLLs (1 normally, 2 for DTV `NLLnum=2`) |
| `OptNoLoadLoop` | 346 | `[0, 1, 2]` | Create a copy of NLL that interleaves stores with final MACs; **0 = no Opt copy, 1/2 = one Opt copy emitted in addition to the ordinary NLL** |
| `ExpandPointerSwap` | 296 | `[False, True]` | Indirectly affects NLL via the second loop-copy emission (does NOT iterate NLL itself) |

There is no parameter that says "emit NLL N times". The only multiplicities are:
- `NLLnum = 2 if NeedNLLOddEven else 1` (KernelWriter.py:6126) where `NeedNLLOddEven = isDTV` — DirectToVgpr's even/odd LDS buffer needs two NLL copies, each emitting the same logical NLL with different `NLLindex`.
- `for remainPgr in range(kernel["PrefetchGlobalRead"]-1, 0, -1)` (KernelWriter.py:6096) — emits *NGLLs*, not NLLs, one per extra prefetched iteration above 1.
- `if self.states.tailloopInNll:` (KernelWriter.py:6145) — optionally emits a second set of NLLs for the not-applicable case.

None of these are an "iter dimension". They are code-path multiplicities chosen at codegen time.

### How NLL is emitted

The entry point `noLoadLoop(...)` is at `KernelWriter.py:4078`. It calls `noLoadLoopBody(...)` at line 4240 which dispatches:
- Under `UseCustomMainLoopSchedule`: emits a single `MAINLOOP` macro invocation with flags `(useGR=0, usePLR=0, useGRInc=0, useLoop=0)` via `_emitNoLoadLoopBodyCMSMacro` (KernelWriter.py:3437).
- Otherwise: runs the full default-scheduler body `_noLoadLoopBodyDefault` (KernelWriter.py:3454).

**The default body itself contains the only "iter dimension" inside NLL:**

```python
# KernelWriter.py:3508
for uIdx in range(0, kernel["LoopIters"]):
    u = uIdx % kernel["LoopIters"]
    ...
    module.addComment1("iter %u%s" % (u, extraComment))   # line 3599
```

This loop emits `LoopIters` subiter chunks of (localReads + pack + MFMA + scheduling) into a single `Module("noLoadLoopBody")`. **One NLL invocation = one body containing `LoopIters` subiter chunks.**

For BPG#11 (DepthU=64, MIK=32): `LoopIters = DepthU / MIK = 2`. The 64 MFMAs observed in NLL come from `LoopIters * numMfmaPerSubiter` plus the inner-unroll dimension, NOT from any outer NLL-iter. (See Q2 for the arithmetic.)

## Q2 — What does NLL physically contain on BPG#11?

Per `n7og_PROBE_REPORT.md:189-192`:

```
NLL = {VCvtPkF32toBF16: 64, MFMAInstruction: 64, DSLoadB128: 8}  (136 nodes total)
```

(Note: the task prompt cites BPG#11's `MatrixInstruction` as `[16,16,32,1,1,4,4,2,2]`. The on-disk fixture `Tensile/Components/CustomSchedule/gfx950/test_yamls/6hk3_tf32_128x160x64_tn.yaml:39` has evolved to `[16,16,32,1,1,4,5,2,2]`. The 64-MFMA count in `n7og_PROBE_REPORT.md` is from the older `4,4` shape; the structural analysis below uses the report's counts as-is, since the SHADOW/CMS divergence pattern is independent of the exact MFMA count.)

### Subiter decomposition

The validator's `subiter()` method (CMSValidator.py:935-960) computes:
- For non-MFMA: parse trailing digit from category — `PackA0` → 0, `PackB3` → 3, etc.
- For MFMA: `tagged_inst.slot.mfma_index // num_mfma_per_subiter`.

`n7og_PROBE_REPORT.md:369` reports the observed subiter populations for the 192 NLL extras:
- Producer subiters observed (SHADOW side): all 0 (PackA0/PackB0 dominate).
- Producer subiters observed (CMS side): all 3 (PackA3/PackB3 — rotating pack buffer wrapped around).
- Consumer (MFMA) subiters observed (both sides): 0, 1, 2.

So `subiter ∈ {0, 1, 2, 3}` — four distinct subiter labels inside NLL. With 64 MFMAs total, that's 16 MFMAs per subiter (matching `MIWaveTile = 4×4 = 16`).

**Crucial caveat about the subiter labelling:** these "subiter" digits do NOT directly correspond to the `uIdx ∈ [0, LoopIters-1]` loop. They are *pack-buffer slot indices* in the rotating buffer, where the number of distinct labels is bounded by `numPackBuffer` and `numVgprBuffer` (`KernelWriter.py:3601-3603`: `plrIdx = (u+pflr) % numVgprBuffer; packPreIdx = (u+pflr) % numPackBuffer; packIdx = u % numPackBuffer`). For BPG#11 with `PrefetchLocalRead=1` and `LoopIters=2`, the pack-buffer slot indices reach four distinct values across iterations because of the `(u + pflr)` offset and `% numPackBuffer` modulus. The mapping `pack_slot_index → uIdx` is many-to-one across the rotating window.

### Where the 64 CVTs come from

Per `_noLoadLoopBodyDefault`, on each pass of the `for uIdx in range(LoopIters)` loop, the body emits localReadDo (which produces packCode pushed into `pack[packStoreIdx]` per category — see KernelWriter.py:3701-3762). The CVT instructions are pack-code outputs from BF16/F32 conversions for the TF32-XdlMathOp data path. For BPG#11 (MIWaveTile=4×4, `InnerUnroll=1`, `numIterPerCoalescedReadA/B=2`): 64 CVTs total partition across the four `pack[0..3]` buffers as 16 each, matching the 4-slot rotating pack buffer.

### `PrefetchGlobalRead=2` effect on NLL structure

`PGR=2` causes:
- One extra NGLL emission (KernelWriter.py:6096-6113, `for remainPgr in range(PGR-1, 0, -1)` runs once).
- The NLL's `NLLfirst=False` flag (KernelWriter.py:4110), which suppresses some duplicated wait/sync code already emitted in the NGLL.
- The ordinary NLL body emission itself is unchanged in shape — still `for uIdx in range(LoopIters)` filling one Module.

So `PGR=2` does NOT add an outer NLL iteration. It adds *NGLL* iterations (one per `remainPgr > 0`), and these NGLLs are captured in `default.n_gl` / `cms.n_gl` (not in `n_ll`). The NGL capture is `{0: body}` like NLL — only the *last* NGLL is captured (the loop runs once for PGR=2, so the single captured NGL body corresponds to that emission). For PGR>=3 the SHADOW/CMS captures would only retain the last NGL body emission too; the multiple physical NGLLs are not separately materialized in the capture.

## Q3 — How does the SHADOW capture record NLL?

### Data model (ScheduleCapture.py:646-700)

```python
@dataclass
class FourPartCapture:
    main_loop: dict       # {cp: body} per codepath
    main_loop_prev: dict  # {cp: body} per codepath
    n_gl: dict            # ALWAYS {0: body} or {}
    n_ll: dict            # ALWAYS {0: body} or {}
    ...
```

The `n_gl` / `n_ll` dicts are intentionally keyed only by `{0: body}` — see `ScheduleCapture.py:651-655` and the `_emitNoLoadLoopBodyCMSMacro` docstring at `KernelWriter.py:3406-3429`. There is no slot for "multiple NLL iterations".

### SHADOW write path

In `noLoadLoop(...)` (KernelWriter.py:4185-4207):

```python
if getattr(self.states, "_captureDefaultSchedule", False) and not isOptNLL:
    ...
    shadow_capture = LoopBodyCaptureBuilder()
    self._noLoadLoopBodyDefault(
        kernel, ..., capture=shadow_capture,
    )
    finalized = shadow_capture.finalize()
    if isNGLL:
        self._capture_context.default_n_gl = finalized
    else:
        self._capture_context.default_n_ll = finalized
```

`ctx.default_n_ll` is OVERWRITTEN on each `noLoadLoop` call that ends up with `isNGLL=False`. For BPG#11 (`NLLnum=1` since not DTV; `tailloopInNll` absent in the fixture), `noLoadLoop` is called exactly once with `isNGLL=False`, so `default_n_ll` is a single `LoopBodyCapture` containing all `LoopIters` subiter chunks of one NLL body in stream order — confirmed by the `n7og_PROBE_REPORT.md` total of 136 nodes in `SHADOW.NLL`.

The downstream `build_cms_four_part_capture` (ScheduleCapture.py:2660-2670) consumes `default.n_ll` as a presence flag and emits a single CMS `n_ll_body` via `expand_cms_macro(macro, id_value=0, useGR=0, useLoop=0, ...)` — wrapping it as `{0: n_ll_body}`.

### Probe — confirming structure without re-running the build

A pytest probe would require a full GPU codegen run (the fixture pipeline invokes `Tensile/bin/Tensile ... --build-only`). Re-running was deemed unnecessary for this question because the published `n7og_PROBE_REPORT.md` already empirically measures:

| Side | `NLL` node count | Composition |
|---|---|---|
| SHADOW | 136 | 64 MFMA + 64 CVT + 8 DSLoadB128 |
| CMS    | 136 | 64 MFMA + 64 CVT + 8 DSLoadB128 |

(`n7og_PROBE_REPORT.md:36, 189-192`). Both captures are single `LoopBodyCapture` instances per side, retrieved as `four_part_capture.n_ll[0]`. No multi-body iteration exists.

## Q4 — Where do the 192 extras live within NLL?

Per `n7og_PROBE_REPORT.md:251-264`:

| (producer, consumer, kind) | SHADOW | CMS | Δ |
|---|---:|---:|---:|
| `(PackA0, MFMA, raw_intrawave)` | 96 | 0 | +96 |
| `(PackB0, MFMA, raw_intrawave)` | 96 | 0 | +96 |
| `(PackA3, PackA3, raw_intrawave)` | 0 | 24 | −24 |
| `(PackB3, PackB3, raw_intrawave)` | 0 | 24 | −24 |
| `(PackA1, MFMA, raw_intrawave)` (per row 254 — typo in source row 253) | 96 | 0 | +96 |
| `(PackB1, PackB1, raw_intrawave)` | 24 | 0 | +24 |
| ... | | | |

The 192 net delta breaks down as `PackA0→MFMA: 96 + PackB0→MFMA: 96`.

### Subiter structure of the 192 extras

From `n7og_PROBE_REPORT.md:369`:

```
Subiter pair (REF=SHADOW, ref): (p_subiter, c_subiter):
  (0, 0): 96    # producer subiter 0, consumer subiter 0
  (0, 1): 48    # producer subiter 0, consumer subiter 1
  (0, 2): 48    # producer subiter 0, consumer subiter 2
```

On SHADOW: producers are PackA0/PackB0 (subiter 0), consumers are MFMAs at subiters {0, 1, 2}. SHADOW's default scheduler places the producers at stream_indices 7–10 (early in NLL) BEFORE the consumer MFMAs at stream_indices ≥14, so `latest_writer` is populated when MFMA reads resolve → 192 edges emitted.

On CMS: the *same physical instructions* are tagged `PackA3`/`PackB3` (the rotating pack-buffer wrapped to slot 3) and placed at stream_indices 84–88, AFTER MFMA consumers at stream_indices 0,3,7,11,... so `latest_writer` is empty when MFMA reads resolve → 0 edges emitted into those MFMAs.

### Characterising the cross-subiter pattern

The handoff is `subiter N` writes into pack buffer slot K and `subiter N+1`'s MFMA reads from slot K — a classic rotating-buffer producer/consumer chain across consecutive subiter chunks within ONE NLL invocation. There is no across-NLL-invocation handoff because there is only one NLL invocation. The "pack3 → pack3" entries (`PackA3 → PackA3: 24`) represent the wrap-around within the same NLL where PackA3 writes from a later subiter chunk feed PackA3 reads from a still-later subiter chunk (same rotating slot, two different time points inside one NLL).

This is fully consistent with the n7og report's own conclusion (`n7og_PROBE_REPORT.md:389`):

> "The handoff is WITHIN a single NLL invocation across subiter chunks — NOT across multiple NLL iterations."

## Q5 — Is there an iter axis inside NLL?

**Yes — exactly one: the subiter axis of length `kernel["LoopIters"]`.**

There is NO outer "prefetched-iter" axis inside NLL driven by `PrefetchGlobalRead`. PGR>=2 produces additional *NGLLs* (separate physical body emissions stored in `default.n_gl`), not additional NLL chunks. Each individual NLL is one body with one internal `for uIdx in range(LoopIters)` loop.

The category-trailing-digit subiter labels (`PackA0`/`PackA1`/`PackA2`/`PackA3`) carry rotating pack-buffer slot indices that span `4 = max(numVgprBuffer, numPackBuffer)` distinct values even when `LoopIters=2`, because of the `(u + pflr) % numPackBuffer` modulus arithmetic at KernelWriter.py:3601-3603. So "subiter label" ∈ [0, numPackBuffer) but "uIdx" ∈ [0, LoopIters); the two axes are coupled but not identical.

For BPG#11 (LoopIters=2, numPackBuffer=4): one NLL invocation emits 2 subiter chunks (one per uIdx), but the pack-buffer slot labels reach values {0, 1, 2, 3} because the chunks reference pack-buffer slots two cycles ahead of their own uIdx via the `(u+pflr)` offset and the rotating modulus interacts with prior-mainloop deposits still live in the buffer.

## Q6 — Corrected data-model recommendation for UnrolledCapture

### What the original plan assumed (wrong)

```
UnrolledCapture = PRO → ML[0..L-1] → NGL → NLL[0..K-1] → POST
  where L = kernel["LoopIters"]
        K = kernel["NoLoadLoopIter"]  # FICTITIOUS
```

### What is actually correct

```
UnrolledCapture = PRO → ML-1 (one body) → ML (one body) → NGL (one body, possibly absent) → NLL (one body, possibly absent) → [POST]
```

Each of `ML-1`, `ML`, `NGL`, `NLL` is a single `LoopBodyCapture` containing `LoopIters` subiter chunks already physicalized in stream order. The "unrolled-iter materialization" the plan envisioned applies to:

| Body | Multi-copy in capture? | Reason |
|---|---|---|
| `PRO` | No | Single prologue, single capture |
| `ML-1` | No (`{0: body}` in default capture; `{cp: body}` per codepath under CMS) | Verbatim clone of ML in default; codepath-keyed in CMS |
| `ML` | No (same shape as ML-1) | Already a single body containing `LoopIters` subiter chunks in stream order |
| `NGL` | No (`{0: body}` or `{}`) | Multiple physical NGLLs (PGR>=3) get collapsed; only last is captured |
| `NLL` | No (`{0: body}` or `{}`) | One physical NLL (`NLLnum=1` typical; DTV `NLLnum=2` only captures last; tailloopInNll adds a separate NLL set whose capture is currently the last one) |

The "unrolled walk" the validator plan needs is therefore:
1. Walk the existing `body.instructions` stream of `n_ll[0]` (and `n_gl[0]`, `main_loop[cp]`, `main_loop_prev[cp]`) in `SchedulePosition` order — that walk already traverses all `LoopIters` subiter chunks within each body.
2. The cross-body chaining (PRO → ML-1 → ML → NGL → NLL) is what `BODY_LABEL_TO_LOOP_INDEX` (ScheduleCapture.py:717-723) and the `(loop_index, stream_index)` SchedulePosition tuple already encode.

No new "NLL-iter materialization" is needed. What IS needed is that the cross-body chain be walked with `latest_writer` carried across body boundaries (currently `latest_writer` resets at each body, per `n7og_PROBE_REPORT.md` §0.7 / §"Stream-position trace"). That cross-body persistence is the actual fix for the 192 NLL extras — NOT iteration count modelling.

## Implications for `UNROLLED_VALIDATION_PLAN.md`

1. **§1.2** — Drop every reference to `kernel["NoLoadLoopIter"]`. The plan should describe the unrolled stream as a single concatenation of the existing five bodies' instruction streams (with `latest_writer` persisted across body boundaries), NOT a multiplication of NLL by an iteration count.
2. **§4 Commit 1** — Whatever commit introduces the unrolled walk must NOT add a body-multiplication step. It should add `latest_writer` persistence across body boundaries in `build_dataflow_graph` (CMSValidator.py around line 1955+). The single-body NLL captures already encode the subiter sequence correctly.
3. **Exemption removal (§Commit 3)** — The cross-subiter ALU-producer exemption at `CMSValidator.py:3831-3843` becomes obviously deletable once `latest_writer` persists across bodies AND across subiter boundaries within a body. The reason the exemption exists today is that `latest_writer` resets at body boundaries — when CMS places `PackA3` at stream_index 84 within NLL (after `MFMA(subiter=0)` at stream_index 0), the MFMA's producer-resolution fails because `PackA3`'s write is not in `latest_writer` yet (intra-body stream-position inversion). The fix is to recognise the producer-consumer relationship across the unrolled timeline — for which the existing single-body NLL capture suffices, because all `LoopIters * numMfmaPerSubiter` MFMAs and all rotating-pack-buffer writes are already in that one body's instruction stream.
4. **`UnrolledCapture` dataclass** — Should NOT include per-body multiplicity. Suggested shape:
   ```python
   @dataclass
   class UnrolledCapture:
       prologue: Optional[LoopBodyCapture]
       ml_prev: dict   # {cp: LoopBodyCapture}
       ml: dict        # {cp: LoopBodyCapture}
       ngl: Optional[LoopBodyCapture]   # n_gl[0] if present
       nll: Optional[LoopBodyCapture]   # n_ll[0] if present
   ```
   — essentially the existing `FourPartCapture` plus a single linearisation contract (PRO → ML_prev[cp] → ML[cp] → NGL → NLL with `latest_writer` carried across).

## Implications for `rocm-libraries-tne8` bead

The bead's premise that a `NoLoadLoopIter` field needs to be added to `FourPartCapture` is **incorrect**. The corrected scope:

- **Drop**: any new `NoLoadLoopIter` field on `FourPartCapture`. The existing `n_ll: dict = {0: body}` shape is structurally correct.
- **Replace with**: `num_mfma_per_subiter` (already present in `FourPartCapture`, line 679) plus a new contract on `build_dataflow_graph` that walks bodies in `BODY_LABEL_TO_LOOP_INDEX` order with `latest_writer` PERSISTED across body boundaries (currently reset per body — see `CMSValidator.py:1955-2199` for the per-body construction call sites).
- **Track separately**: the multiplicity of physical `noLoadLoop` calls (DTV NLLnum=2, tailloopInNll extra set, PGR>=2 extra NGLLs) — those represent "the capture only records the last emission". If validation needs to cover all physical emissions, that is a separate bead about capture-side multi-emission retention, not about modelling iteration count in the data structure.

If `rocm-libraries-tne8` is currently scoped as "add NoLoadLoopIter field", it should be rescoped to "persist `latest_writer` across body boundaries in unrolled walk" and the original framing should be marked corrected. Whether to file a new P0 depends on whether the existing bead has dependents that bake in the wrong scope.

---

**No new defect filed.** This investigation produced a clarification of existing scope (the plan and bead `tne8` had a wrong premise), not a new bug. Per the standing rules, a P0 bead is only filed for genuine new defects. The corrective work is to revise `UNROLLED_VALIDATION_PLAN.md` and re-scope `rocm-libraries-tne8`; both are existing artefacts whose scope clarification belongs in their own update cycles.

**Concrete file:line evidence index:**
- `Tensile/KernelWriter.py:3508` — `for uIdx in range(0, kernel["LoopIters"])` inside `_noLoadLoopBodyDefault` (the ONLY iter loop inside NLL).
- `Tensile/KernelWriter.py:4078-4252` — `noLoadLoop` method (the per-call NLL emission entry).
- `Tensile/KernelWriter.py:4185-4207` — SHADOW capture wiring; overwrites `ctx.default_n_ll` on each call (single retention).
- `Tensile/KernelWriter.py:6096-6163` — `kernelBody` driver: PGR loop emits NGLLs; `for NLLindex in range(0, NLLnum)` emits NLLs (`NLLnum=2` only for DTV).
- `Tensile/KernelWriter.py:3406-3429` — `_emitNoLoadLoopBodyCMSMacro` docstring: hard-codes `\\ID=0`, single body per CMS NLL.
- `Tensile/Components/ScheduleCapture.py:661-664` — `FourPartCapture.n_gl` / `n_ll` defined as dicts keyed by `{0: body}` (`# always {0: body}`).
- `Tensile/Components/ScheduleCapture.py:2628-2672` — `build_cms_four_part_capture` builds `n_ll_dict = {0: n_ll_body}` only when `default_has_n_ll` is True.
- `Tensile/Common/ValidParameters.py:251, 292, 296, 346` — full list of NLL-related parameters; `NoLoadLoopIter` is absent.
- `Tensile/Components/n7og_PROBE_REPORT.md:36, 189-192, 369, 389` — empirical NLL composition (136 nodes, single body, subiter handoffs within one NLL invocation).
