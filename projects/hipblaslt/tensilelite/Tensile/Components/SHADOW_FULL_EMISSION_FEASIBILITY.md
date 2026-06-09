# SHADOW full-kernel emission feasibility

**Status:** investigation (read-only), 2026-06-09
**Question:** can the TensileLite SHADOW (default-schedule) capture path be made to
emit a FULL kernel assembly text — the way the CMS path emits `kernel.s` — so we can
text-diff the SHADOW *capture* against the SHADOW *emitted* assembly and prove the
SHADOW capture is faithful?

**Short answer:** YES, qualified. The full default-schedule kernel is *already
assembled* today by `build_non_cms_reference` (Approach A) — it calls
`writer._getKernelSource(non_cms_solution)`, which returns the complete assembly
text, and then throws that text away, keeping only the `FourPartCapture`. Capturing
the return value is the whole mechanism for a "non-CMS default `kernel.s`". The
qualifier is comparability: the Approach-A build uses `UseCustomMainLoopSchedule=0`,
which flips `doFullPackCodePrefetch` and other code-path gates, so its text is NOT
register/codegen-identical to the CMS `kernel.s` (the j4qm divergence). It is a
ground-truth for the *default scheduler*, which is exactly what proving the SHADOW
capture faithful requires — but it is a different kernel from the CMS subject, so it
is a noisy diff against `kernel.s` and a clean diff only against itself.

---

## §1 The verification gap

The validator compares two captures of one CMS build:

- **subject** = CMS schedule (`customMainLoopSchedule`). Path emits a complete
  assembly text: `writer._getKernelSource(solution)` returns it; tooling dumps it
  as `kernel.s` (e.g. `h7lo_uvrl_artifacts/kernel.s`). Its capture
  (`cms_capture_listing.txt`) has been **directly verified faithful** by text+order
  match against the emitted macro body — e.g. NGL `ds_read kernel.s:2175 -> pack-mfma
  :2208 -> pack-cvt :2225` matches `cms_capture_listing.txt:584/604/612`
  (`H7LO_UVRL_NORMATIVE_ORDERING.md:355-358`).

- **reference / SHADOW** = the default schedule. Captured ONLY as a
  `FourPartCapture` (`writer._last_default_capture`, source `"default-sia3"`),
  populated by `_captureDefaultSchedule`. **There is no full assembly text.** Its
  faithfulness rests on a producer-side argument (it's a post-interleave walk of the
  real default emitter's output) plus internal consistency — NOT a direct text diff
  (`H7LO_UVRL_NORMATIVE_ORDERING.md:359-367`; the residual is stated explicitly at
  `h7lo_uvrl_artifacts/NGL_CAPTURE_INVESTIGATION_MEMO.md:238-245`).

This gap matters more now, not less: per `rocm-libraries-dm4p` Phase 2, the
production inline validator (`_captureNonCmsBuild` block,
`KernelWriter.py:6392-6413`) now consumes the SHADOW capture *as the canonical
reference* (`ctx.default = self._last_default_capture`) and no longer drives
`build_non_cms_reference`. The validator's reference side is the unverified-by-text
SHADOW capture.

---

## §2 How CMS emits full assembly today (the path)

`_getKernelSource(kernel)` (`KernelWriter.py:10481-10499`) is the single full-kernel
emitter:

```
fileString = ""
self._initKernel(kernel, tPA, tPB)
(error, kb) = self.kernelBody(kernel, tPA, tPB)   # kb is a Module tree
fileString += str(kb)                              # <-- the entire kernel as text
return fileString
```

`kernelBody` (`KernelWriter.py:5610`) assembles the whole kernel into one `Module`
`kb`: prologue, prefetch, main loop (`_loopBody`), noLoadLoop / NGL / NLL, epilogue.
For a CMS kernel the main-loop body's instruction stream is produced by
`customMainLoopSchedule` (called at `KernelWriter.py:5478`) interleaving the
per-source-bucket modules; `str(kb)` walks the tree and renders every leaf
instruction to assembly text. That is the only step that turns the assembled
instruction stream into text, and it is generic over what's in the tree — it is not
CMS-specific.

So the CMS `kernel.s` is just `str(kernelBody(...))` on a `UseCustomMainLoopSchedule=1`
kernel. Nothing about the text-emission step is bound to CMS.

---

## §3 How SHADOW is captured today: fragments-only, NOT
full-kernel-emitted-then-partially-captured

**Finding: the SHADOW path produces capture fragments only. It never assembles a full
default-schedule kernel.** The full default kernel is *not* emitted internally and
then partially captured; the default emitter is run with its emitted `Module`
*discarded*, leaving only the capture.

Evidence — the SHADOW main-loop site, `_loopBody`
(`KernelWriter.py:5103-5250`): when `_captureDefaultSchedule` is set, it calls
`self._makeSubIterSchedule(... capture=self._capture_context.builder, ...)`
(`:5226-5250`). The `iterCode` that call returns is **discarded** — the code comment
at `:5115-5117` states it plainly: *"iterCode is discarded by the capture branch —
never appended to module."* `_makeSubIterSchedule` (`:983`) builds the body, then at
`:2694-2704` calls `_captureSubIterToBuilder(iterCode, capture, ...)` which walks
`iterCode.flatitems()` into `TaggedInstruction`s and returns; the caller drops the
returned `iterCode`. Only the builder survives.

Evidence — the SHADOW NGL/NLL site, `noLoadLoop` (`KernelWriter.py:4185-4207`):

```python
if getattr(self.states, "_captureDefaultSchedule", False) and not isOptNLL:
    ...
    shadow_capture = LoopBodyCaptureBuilder()
    self._noLoadLoopBodyDefault(..., capture=shadow_capture)   # result NOT added to module
    finalized = shadow_capture.finalize()
    if isNGLL: self._capture_context.default_n_gl = finalized
    else:      self._capture_context.default_n_ll = finalized
```

Contrast the non-CMS path immediately below (`:4223-4232`):

```python
if nlnoncms_capture:               # _captureNonCmsBuild on a UseCustomMainLoopSchedule=0 build
    noncms_nl_capture = LoopBodyCaptureBuilder()
    module.add(self._noLoadLoopBodyDefault(..., capture=noncms_nl_capture))  # result ADDED to module
    finalized_nl = noncms_nl_capture.finalize()
```

The structural difference is the entire story: SHADOW calls
`self._noLoadLoopBodyDefault(...)` and drops the returned `Module`; the non-CMS path
does `module.add(self._noLoadLoopBodyDefault(...))`, so it lands in `kb` and renders
into `kernel.s`. **SHADOW runs the real default emitter but discards its output
Module; only the capture is kept.** The four captured bodies (PRO via `ctx.prologue`,
ML/ML-1 via `ctx.default_main`, NGL via `ctx.default_n_gl`, NLL via
`ctx.default_n_ll`) are assembled into the `FourPartCapture` at
`KernelWriter.py:6247-6258` (`source="default-sia3"`). No `str(kb)` for the default
schedule is ever produced.

(Note: SHADOW reuses the **CMS-mutated `kernel` dict** — the default emitter runs over
the same `doFullPackCodePrefetch`/`UsePLRPack`/T-reg state the CMS subject was built
with. That is by design per `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §2`: it is what
makes SHADOW the right *scheduler* reference. See §4/§7 for the comparability
consequence.)

---

## §4 Viable approaches to emit a full SHADOW assembly

### Approach A-text — capture the assembly text `build_non_cms_reference` already produces (LOWEST risk)

**Mechanism.** `build_non_cms_reference` (`approach_a.py:141-227`) already does the
hard part: it spins up a fully isolated second `KernelWriterAssembly`, prepares a
non-CMS config from `solution.pre_cms_state()` with `UseCustomMainLoopSchedule=0`
(`approach_a.py:105-138`), and calls `writer._getKernelSource(non_cms_solution)`
(`:214`) — which **returns the complete assembly text**. Today the return value is
ignored and only `writer._last_default_capture` is kept (`:216`). To emit a full
default `kernel.s`, capture that return value.

This is not hypothetical: `Tensile/Tests/unit/_dump_carveout_assembly.py:226-234`
already does exactly this — a second writer, `UseCustomMainLoopSchedule=0`,
`default_asm_text = default_writer._getKernelSource(default_solution)`, written to
`kernel_default.s`. So a full default-schedule `.s` is demonstrably producible
*today* with no source change to the emitter.

**What changes.** A small helper (or a one-off dump test mirroring
`_dump_carveout_assembly.py`) that keeps the `_getKernelSource` return value
alongside the `FourPartCapture`. `build_non_cms_reference` itself returns only the
capture; a sibling that also returns the text, or a test harness, suffices. No change
to `_captureDefaultSchedule`, `_loopBody`, `noLoadLoop`, or
`_makeSubIterSchedule` — so zero risk of perturbing what SHADOW captures on the CMS
build.

**Risk.** Minimal. No production-path code changes; the SHADOW capture machinery is
untouched. Build cost: ~one extra `_getKernelSource` per dump (acceptable for a
verification/dev tool; not on the hot validator path).

**Comparability to `kernel.s`.** *Caveat — this is the central limitation.* The
Approach-A build is `UseCustomMainLoopSchedule=0` re-derived from the pre-CMS
snapshot, so `doFullPackCodePrefetch = UsePLRPack and not UseCustomMainLoopSchedule`
flips ON, plus other gates. Per `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §1`, this
yields a *different instruction set* than the CMS subject (BPG#11: 20 extra
`v_mov_b64`; 200 UNKNOWN-classified instructions per §3 of the design doc). So a text
diff of `kernel_default.s` against the CMS `kernel.s` is **noisy** — it confounds
scheduler order with codegen-branch divergence. But a text diff of `kernel_default.s`
against the **Approach-A capture of the same build** is clean and is exactly the
SHADOW-style faithfulness proof for the *default scheduler on its own codegen state*.

### Approach SHADOW-tee — assemble the SHADOW bodies the CMS build already emits into a text artifact (HIGHER fidelity to "the SHADOW the validator uses", HIGHER risk)

**Mechanism.** The validator's reference is specifically the SHADOW capture taken
*on the CMS build over the CMS-mutated kernel dict* (`source="default-sia3"`), NOT
the Approach-A separate build. To text-diff *that* capture against an emission, the
emission must come from the same SHADOW run. Today the SHADOW emitter's output
`Module`s are discarded (§3). To tee them: stop discarding the `iterCode` from the
SHADOW `_makeSubIterSchedule` call (`:5226`) and the `Module` from the SHADOW
`_noLoadLoopBodyDefault` call (`:4196`), collect them into a side `Module`, and
`str(...)` it into a `shadow_kernel.s` artifact (bodies only, or stitched with the
shared prologue/epilogue).

**What changes.** `_loopBody` and `noLoadLoop` SHADOW branches must retain the
emitter output instead of dropping it. This is invasive precisely where the standing
rule says not to perturb: the SHADOW branches use `structural_clone` of inputs and
rely on the emitted body being thrown away (label sharing is "safe because iterCode
is discarded ... never appended to module", `:5115-5117`). Teeing it to a *separate*
text-only `Module` (never added to `kb`) preserves that invariant — the cloned body
is rendered to text in isolation, never assembled into the runnable kernel — but it
is still new code in the load-bearing SHADOW path.

**Risk.** Moderate. Must guarantee the teed `Module` is `str()`-only and never enters
`kb` (else duplicate-label / double-emission at the assembler). Must not consume items
the CMS bucket-accumulation needs (the reason `structural_clone` exists). Done
carefully — render the *already-cloned* SHADOW body to text and discard the text-Module
— it does not change what SHADOW captures; it only also renders it.

**Comparability.** This is the *highest-fidelity* artifact: it is literally the
bytes the SHADOW capture walked, so a diff of `shadow_capture_listing.txt` against it
is the §2-style direct proof the CMS side already enjoys. It is NOT directly
comparable to the CMS `kernel.s` line-for-line (different scheduler), but it doesn't
need to be — its job is to ground-truth the *capture*, not to match CMS.

### Approach reuse-dump-test — promote `_dump_carveout_assembly.py` into the verification loop (LOWEST effort, partial)

`_dump_carveout_assembly.py` already writes `kernel_cms.s`, `kernel_default.s`,
`cms.s`, `default.s` from one fixture. It is a print-only dev test, not an assertion.
A thin assertion test could diff the default capture stream (`default.s`) against
`kernel_default.s` for that fixture. This is Approach A-text scoped to one fixture,
zero new emitter code. Good for a fast regression pin; not general across the CMS
test surface.

---

## §5 Recommendation

**Possible: YES (qualified).** A full default-schedule `kernel.s` is already produced
today and discarded; capturing it is trivial. The qualifier is *which* full-SHADOW
text you want:

1. **To ground-truth the validator's actual reference** (the `"default-sia3"` capture
   on the CMS build), you need **Approach SHADOW-tee** — render the CMS-build SHADOW
   bodies to text. Approach A-text builds a *different* kernel (CMS=0 codegen) and so
   cannot ground-truth the CMS-build SHADOW capture byte-for-byte; it grounds a
   *different* (though legitimate) default scheduler reference.

2. **To get a runnable, faithful default-scheduler `.s` cheaply** for spot
   verification (e.g. resolving uvrl), use **Approach A-text** — capture the
   `_getKernelSource` return inside (or beside) `build_non_cms_reference`, or the
   ready-made `_dump_carveout_assembly.py` dump. No emitter changes; no risk to the
   SHADOW path.

**Recommended sequencing.** Start with **Approach A-text** (no risk, immediate
artifact) to answer concrete questions like uvrl, accepting the codegen-branch
caveat. If the validator's confidence specifically in the `"default-sia3"` capture
must be raised to the CMS side's text-proven level, do **Approach SHADOW-tee** as a
targeted, text-only tee (never into `kb`) — but treat it as touching the load-bearing
SHADOW branch and gate it behind the standing "do not change what SHADOW captures"
rule (it must render only the already-cloned body, never assemble it into the
runnable kernel).

**What it requires (semantic scope).**
- Approach A-text: capture the existing `_getKernelSource` return value in
  `approach_a.py` (a sibling returning `(capture, text)`), or a dump test mirroring
  `_dump_carveout_assembly.py`. No change to `KernelWriter` emission.
- Approach SHADOW-tee: in `_loopBody` (`:5226`) and `noLoadLoop` (`:4196`) SHADOW
  branches, route the already-`structural_clone`d body the emitter returns into a
  side `Module`, `str()` it, never `.add()` it to `kb`. New text-only collection in
  `CaptureContext`; finalize/dump in `kernelBody` next to the existing
  `ctx.default` assembly (`:6247`).

---

## §6 Payoff: what verification it unlocks (and uvrl B1/B2)

If we emit a full default `.s` and diff it against the corresponding capture, we run
**the same capture-vs-emission text proof for the default scheduler that we already
run for CMS** (`H7LO_UVRL_NORMATIVE_ORDERING.md:355-358`). Specifically:

- **Approach SHADOW-tee** converts the `"default-sia3"` capture's "faithful by
  construction" argument (`H7LO_UVRL_NORMATIVE_ORDERING.md:359-367`) into a direct
  text diff — closing the residual stated verbatim at
  `NGL_CAPTURE_INVESTIGATION_MEMO.md:238-245` ("SHADOW has no emitted `.s`, so its
  faithfulness rests on the producer-side argument ... What would fully close it: a
  one-off build ... that DOES emit the default NGL as `.s`, then diff that `.s`
  against `shadow_capture_listing.txt`").

- **For uvrl B1/B2 specifically:** the uvrl question is whether the SHADOW NGL
  ordering (`ds_read` LAST, after the pack chain — `shadow_capture_listing.txt:477
  pack-mfma / :490 pack-cvt / :610 ds_read`) is REAL (B1: validator-modeling fix) or
  a CAPTURE ARTIFACT (B2-adjacent: the capture mis-orders what the emitter actually
  produced). A full SHADOW `.s` answers the **artifact half decisively**: if the
  emitted default NGL body shows `ds_read` after the pack chain in the same order the
  capture records, the capture is faithful and the divergence is real (consistent
  with the doc's current B1 conclusion at `:359-370`). It does **not by itself**
  decide B1-vs-B2's normative half (is the default's `pack -> ds_read` or CMS's
  `ds_read -> pack` the *authoritative* order) — that is a scheduler-intent question
  for the `_get_schedule_128x128x32_TF32` NGL-ordering owner (uvrl bead). So:
  **emitting full SHADOW assembly resolves the "is the SHADOW capture lying?" risk
  that B2 partly rests on, but the residual normative B1-vs-B2 decision still needs
  the schedule owner.** Net: it removes one of the two live uncertainties under uvrl
  (capture fidelity), strengthening the existing B1 lean to "proven not a capture
  artifact" rather than "argued not a capture artifact."

---

## §7 Open questions / risks

1. **Comparability ceiling (A-text).** A CMS=0 build is a genuinely different kernel
   (`doFullPackCodePrefetch` flip; j4qm). Its `.s` is the right ground-truth for the
   *default scheduler*, but a line diff against the CMS `kernel.s` is noisy by
   construction. Anyone using A-text to "compare against `kernel.s`" must scope to
   the bodies/regions and expect codegen-branch deltas. The clean diff is A-text-`.s`
   vs the A-build's own capture.

2. **SHADOW-tee invariant (the load-bearing risk).** The SHADOW branches are correct
   *because* they discard the emitted body (label-sharing safety, `:5115-5117`).
   Teeing must render the already-cloned body to text in isolation and never add it
   to `kb`. If that invariant slips, the runnable kernel gets duplicate labels /
   double bodies. This is the one place the standing "do not perturb SHADOW" rule is
   genuinely at stake.

3. **Prologue/epilogue stitching.** A "full" SHADOW `.s` needs the shared
   prologue/epilogue around the four bodies to be a real kernel. The bodies are
   captured; the surrounding scaffolding on the CMS build is the CMS scaffolding. A
   true stand-alone default `.s` is exactly what A-text already gives (whole kernel);
   SHADOW-tee gives bodies-on-CMS-scaffolding (fine for capture-fidelity diffing, not
   a runnable default kernel).

4. **Approach A is scheduled for retirement** (`rocm-libraries-u89e` Phase 4,
   `rocm-libraries-75kj` Phase 0). If A-text becomes the verification mechanism, the
   `build_non_cms_reference` + `pre_cms_state` machinery it depends on must survive as
   a *verification tool* even after it's removed from the production validator path.
   Worth an explicit decision before retirement deletes it.

5. **`u6nn` pre-existing defect** (`build_non_cms_reference` called with a dict
   instead of a Solution in one test) sits on the Approach-A path; any A-text work
   should not be blocked by it but should be aware the helper's contract takes a
   `Solution`.

---

## §8 Bead filed

**`rocm-libraries-4ydd`** — "Emit full SHADOW/default-schedule assembly text to
ground-truth the SHADOW capture (close the no-emitted-`.s` residual)." Related to the
uvrl normative question (`rocm-libraries-uvrl` `related` `rocm-libraries-4ydd`);
description references Approach-A retirement (`u89e`, `75kj`) so the verification
tooling isn't deleted with the production path, and the `u6nn` dict-vs-Solution defect
on the same helper.
