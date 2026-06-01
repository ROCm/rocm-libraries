# dm4p Correctness & Soundness Report

## Verdict

**CLEAN.** The Phase 2 swap is structurally sound. Call ordering is correct, `_last_default_capture` is populated as a fully-formed `FourPartCapture` before the xj16 assertion consumes it, the lock-in test is source-level but appropriate, the broader baseline matches plan-adherence expectations exactly (3 failures: u6nn + 2 × w5xw), and BPG#11 builds end-to-end with no validator-class errors. No new bugs surfaced.

## Per-Q

### Q1 — `_last_default_capture` populated before xj16?

**YES.** Call ordering verified by reading `KernelWriter.kernelBody`:

- `kernelBody` auto-activates BOTH `_captureDefaultSchedule` and `_captureNonCmsBuild` together at `Tensile/KernelWriter.py:5617-5630` whenever `kernel["UseCustomMainLoopSchedule"]` is truthy. The two flags are coupled by construction.
- `_loopBody` is invoked at `:6046/6060/6067`; `noLoadLoop` at `:6102/6112/6141/6162`. Both populate scratch state (`default_main`, `default_n_gl`, `default_n_ll`) before the next block runs.
- The SHADOW block at `Tensile/KernelWriter.py:6170-6318` runs strictly AFTER all `_loopBody` / `noLoadLoop` emissions. It assembles the `FourPartCapture` and assigns `ctx.default = FourPartCapture(...)` at `:6240-6251`.
- The `_captureNonCmsBuild` block at `Tensile/KernelWriter.py:6361+` runs strictly AFTER the SHADOW block. `ctx.default = self._last_default_capture` at `:6399` therefore reads the SHADOW-populated capture.
- `CaptureContext.reset()` at `Tensile/Components/ScheduleCapture.py:630-642` clears scratch state only; `default` and `cms` are intentionally preserved (line 631-633 docstring + behaviour verified by reading `reset` body and by the unit test `test_capture_context_default_survives_reset`).
- The xj16 assertion at `Tensile/KernelWriter.py:6443-6444` is additionally guarded by `ctx.default is not None`, so even a pathological code path where SHADOW left `ctx.default = None` would skip silently rather than AttributeError.

### Q2 — FourPartCapture fully populated?

**YES, on CMS callsites.** Reading the SHADOW assembly at `Tensile/KernelWriter.py:6240-6251`:

- `main_loop = {0: main}` where `main = ctx.default_main` (populated by the per-iter SHADOW path that finalizes at `:5475`).
- `main_loop_prev = {0: clone_loop_body(main)}` (deep clone of main).
- `n_gl = {0: ctx.default_n_gl}` if not None (populated at `:4205` from `_noLoadLoopBodyDefault` SHADOW emission).
- `n_ll = {0: ctx.default_n_ll}` if not None (populated at `:4207` from same path).
- `prologue = ctx.prologue` (populated by oram Phase 2 prologue harvest).

The Approach-A `build_non_cms_reference` constructed a FourPartCapture with the same slot semantics, so consumers expecting all four slots see equivalent shape. `build_dataflow_graph` (used in xj16's assertion) handles missing `n_gl`/`n_ll` keys cleanly per the comment at `Tensile/KernelWriter.py:6190-6195`.

### Q3 — Lock-in test rigor?

**SOURCE-LEVEL (regex over `inspect.getsourcelines`), NOT RUNTIME.** Reading `Tensile/Tests/unit/test_dm4p_shadow_as_canonical_reference.py:97-153`:

- `test_xj16_cms_callsite_consumes_shadow_capture` greps the source text of `KernelWriter.kernelBody` for `self._last_default_capture` (present) and `build_non_cms_reference` (absent), inside the `if is_cms_callsite:` branch only, with comments stripped at `:128-130`.
- It would NOT catch an attribute rename (e.g. `_last_default_capture` → `_last_shadow_capture`); a future renamer who updates the property setter/getter at `KernelWriter.py:452-457` would have to also update the regex literal in this test, but the test would currently pass-by-mismatch (silently) if the regex still finds `self._last_default_capture` somewhere else in `kernelBody`. The risk is small because `kernelBody` is one method.
- The second test (`test_capture_context_default_survives_reset` at `:156-194`) is a real runtime test pinning the `reset()`-preserves-`default` invariant.

The test author's rationale (`:38-46`) explicitly acknowledges this is a fast deterministic source-pin rather than an end-to-end runtime check, and that the BPG#11 build is the primary signal. That trade-off is reasonable; the source-pin protects against the most likely regression vector (a merge-conflict resolver restoring the Approach-A call). Not a defect, but worth noting.

### Q4 — Baseline match (3 failures, expected u6nn + w5xw)?

**EXACT MATCH.** Re-ran from worktree (cd-first, with the mandated `--ignore` flag):

```
3 failed, 1060 passed, 4 skipped, 2 xfailed in 22.99s
```

Failures:
- `test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_render_counts_match` (w5xw)
- `test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_ordinal_logical_instruction_matches` (w5xw)
- `test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference` (u6nn)

Matches plan-adherence verifier's expectation (and counts/IDs in the report).

### Q5 — BPG#11 build outcome?

**CLEAN.** `Tensile/bin/Tensile ...6hk3_tf32_128x160x64_tn.yaml /tmp/dm4p_correctness --build-only --gpu-targets gfx950` exited 0. Log inspection:

- `# Actual Solutions: 2 / 2 after KernelWriter` — both solutions survived.
- No matches in log for `compare_graphs|validate_edge_wait|verify_correct_number|CaptureCategory|CaptureConsistency|AssertionError|Traceback|Error` (only the benign `WARNING: Logic file has invalid or unsupported chip IDs` lines).
- xj16 inline `compare_graphs` and `validate_edge_wait_coverage` are inside this build path; clean exit implies both passed against the SHADOW capture.

### Q6 — `is_cms_callsite` predicate?

**STILL LOAD-BEARING (gates SHADOW source vs. Approach-A shape-b assembly).** Reading `Tensile/KernelWriter.py:6375-6435`:

- `is_cms_callsite = bool(kernel.get("UseCustomMainLoopSchedule"))` at `:6375`.
- `if is_cms_callsite:` branch → consume SHADOW (`ctx.default = self._last_default_capture`, `:6399`).
- `else:` branch → consume the natural `_loopBody`/`noLoadLoop` non-CMS captures from `ctx.default_main`/`default_n_gl`/`default_n_ll` and assemble a FourPartCapture with `source="non-cms-reference"` (`:6403-6435`). This serves `enable_capture_non_cms_build()` callers (Approach-A's shape-b, used by tests).

The two branches are NOT redundant — shape-b is exercised by external tests that opt into `_captureNonCmsBuild` without `UseCustomMainLoopSchedule`. Phase 4 (`u89e`) will retire this scaffolding when Approach-A is fully deleted; the design doc update at `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md:139-141` explicitly notes this scoping.

If `_last_default_capture` is None on a CMS callsite (e.g. malformed kernel that skipped SHADOW), `ctx.default = None`, and the xj16 assertion's `ctx.default is not None` guard at `:6443` causes a silent skip rather than crash. Sound but worth documenting; the SHADOW gating coupling at `:5617-5630` makes this hypothetical.

### Q7 — w5xw bead properly filed?

**YES.** `br show rocm-libraries-w5xw --json`:
- Exists, P0 bug, status open.
- Labels: `approach-default-scheduler-ref`, `phase-3-blocker`, `shadow-capture`, `validator`.
- Dependents: `rocm-libraries-r62g` (status open, priority 0, dependency_type `blocks`) — correct link.
- Description covers symptom (~642 mismatches in ML body), repro command, scope (pre-existing on d606518ab21), and distinguishes it from u6nn.
- I re-ran the 2 failing tests in the current worktree; they reproduce with the documented mismatch counts.

`br show rocm-libraries-u6nn --json`: also confirmed exists, P0 bug, dependents include `rocm-libraries-u89e` with `blocks`.

### Q8 — Corner cases?

- **Non-CMS kernel (UseCustomMainLoopSchedule=0)**: the SHADOW + `_captureNonCmsBuild` auto-activation at `:5617-5630` is gated on `if kernel.get("UseCustomMainLoopSchedule"):`. Neither flag is set. The `_captureNonCmsBuild` block at `:6361+` is gated on `getattr(self.states, "_captureNonCmsBuild", False)`. Non-CMS kernels skip both. Safe.
- **External non-CMS callers via `enable_capture_non_cms_build()`** (Approach-A shape b): `is_cms_callsite=False` since `UseCustomMainLoopSchedule=0`; falls into the `else` branch which builds its own FourPartCapture from `ctx.default_main`/`default_n_gl`/`default_n_ll`. The dm4p swap does not touch this path. Safe.
- **Multi-kernel builds**: Per-kernel `reset()` at `:6318` and `:6483` clears scratch (`default_main`, `default_n_gl`, `default_n_ll`, `builder`, `prologue`, prefetch packs) but preserves `default`/`cms` (consumer-facing). The next kernel's SHADOW block will rebuild `ctx.default` (overwriting). The only theoretical hazard would be a second kernel where SHADOW fails to populate `default` AND `_captureNonCmsBuild` still reads — but the SHADOW gating couples to `UseCustomMainLoopSchedule`, so both run or both don't. The xj16 assertion is also extra-guarded by `ctx.default is not None`. Safe.
- **OptNoLoadLoop kernels (`_noLoadLoopBodyDefault` path with `isOptNLL=True`)**: SHADOW `default_n_gl`/`default_n_ll` capture at `:4185` is gated on `and not isOptNLL`. So `n_gl`/`n_ll` may be missing for optNLL builds — but `n_gl_dict`/`n_ll_dict` at `:6196-6197` use `if ... is not None` conditional, and `build_dataflow_graph` skips absent keys per the comment at `:6190-6195`. Safe.

## Bugs / new beads filed

**None.** No new defects surfaced during this verification pass. The 3 baseline failures are already covered by u6nn (P0, blocks u89e) and w5xw (P0, blocks r62g), both filed by prior agents in this loop.

## Recommendation

**CLEAN → proceed to Step 9 squash-merge + cleanup (task #306).** The Phase 2 swap is correct, minimal, and well-instrumented. Both lock-in tests pass. BPG#11 builds end-to-end. Baseline strictly improves (−16 failures, −2 errors, +20 passes vs. parent). Discoveries are filed with proper dependency links. Design doc updated with LANDED marker. No backwards-compat bloat. No tactical scaffolding beyond the explicitly-scoped `is_cms_callsite` predicate and `else` branch retained for Phase 4 retirement.
