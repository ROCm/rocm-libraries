# CMS Validator — Stage 09 Remaining Work

## Wholesale `make_mock_id_map` deletion

Phase 1 leaves `make_mock_id_map` in place. The wholesale migration
requires:

- Per-class kernel rewrite (~15 `setup_method` methods affected)
- Registered-schedule audit: every test class's tile shape must map to a
  registered CMS schedule. Some don't (e.g. there is no 128×64×32 TF32 TN
  schedule for `TestValidatePackTF32MultipleGroups`'s native MT0×MT1).
- 27 `_mock_dtype` call sites in `test_CustomSchedule.py` that swap to real
  `DataType` objects, which may change schedule predicate dispatch
- All `assert self.num_vmfma == N` test assertions audited if
  `create_base_kernel` baseline changes
- `hasCustomSchedule` mutates the kernel dict (e.g.
  `kernel["UseMFMAF32XEmulation"] = True`); `kernel_to_solution_config`
  must thread these mutations through, or validator and kernel writer
  disagree on what the kernel says

## `create_base_kernel` valid-Solution baseline

Same dependency chain as wholesale deletion. Must coordinate with that
migration.

## Delete dead positional code

Blocked on wiring `set_lr_needed_by_from_mfma_operands` into
`add_local_read_constraints` (caller must scope LR0 to same loop,
LR1/LR3 to next loop, apply hybrid wholesale fallback to
`set_lr_needed_by_for_VMFMA` when chain dies). Then the following
become safe to delete:
- `_hook_up_packs_bf16`
- `_hook_up_packs_f32` (keep `_hook_up_middle_16_pairs`)
- `_hook_up_packs_f32_mfma`
- Hardcoded `pack_dependencies` dicts
- `set_lr_needed_by_for_VMFMA` (and `lr_needed_by_mfma` tile-math)

## Wiring follow-ups for `set_lr_needed_by_from_mfma_operands`

- Author a `_REAL_TWIN_CONFIG_PLR_PACK` config (UsePLRPack=True) for
  cross-loop LR1/LR3 parity testing before wiring lands. The current
  `_REAL_TWIN_CONFIG_VW4` has PrefetchLocalRead=1 and exercises only
  LR0/LR1 (not LR3).
- Author a non-SwapPack registered config (LocalReadVectorWidth=1
  variant of an existing TF32 schedule) so the parity test can verify
  full chain coverage rather than the current "no false answers"
  weak parity (the SwapPack path silently truncates the chain for
  every LR in `_REAL_TWIN_CONFIG_VW4`).

## Dual-path comparison re-add

When the primary path was wired (commit `23d8627a22`), the dual-path
comparison block (`derive_pack_must_start_after` vs positional
`_hook_up_packs_*`) was removed from `hook_up_packs`. We now have no way to
detect subtle bugs in the new path. Re-add as a debug-only mode (e.g.
`CMS_VALIDATE_DUAL_PATH=1` env var) before deleting the positional
functions.

## `_get_lrs_for_pack` still required

The original stage-09 plan was to delete `_get_lrs_for_pack`, but it's
still needed to tell `derive_pack_must_start_after` WHICH local reads to
consider for a given pack (the function inspects pack source registers but
needs to know the relevant LR scope). Either keep it or replace with a
register-based helper that walks the timeline.

## `MIArchVgpr` test verification

`MIArchVgpr` is copied through `kernel_to_solution_config` but the test
that exercises it (`test_schedule_128x128x32_TF32_MIArchVgpr`) calls the
mock-path `_make_context`. `MIArchVgpr` affects register allocation, so
the real idMap may differ in subtle ways from the test's hardcoded
schedule. Verify if/when that test is migrated to the real path.


## Investigate if build a full register chain is a good idea
Instead of having diferent functions recreating a traversal fo the registers, build the chain onces, and then have the different functions simply walk the already build graph.

## Schedule capture follow-ups (Phases 5, 7, 9 of plans/then-let-s-work-on-jaunty-reddy.md)

The schedule capture infrastructure (Phases 1-4, 6, 8) lands the data
structures, the macro walker for CMS-side capture, the SIA3 instrumentation
and shadow driver for default-side `main_loop`, the `compare_captures` rule
wired into `isValid`, and the OptNLL+CMS rejection. The following are
explicitly deferred:

- **Phase 5: Default-side `n_gl` and `n_ll` capture.** The shadow runs need
  to live inside `KernelWriter.noLoadLoop` (after the pre-trimming logic at
  lines 3352-3379) to capture the default scheduler's tail-loop bodies.
  Currently `n_gl` and `n_ll` in `_last_default_capture` are empty
  `LoopBodyCapture` instances. `compare_captures` skips empty bodies in the
  per-body MFMA-count check.
- **Phase 7: Per-edge dataflow comparison.** `build_dataflow_graph` is a
  no-op skeleton. The cross-scheduler `compare_captures` currently checks
  only the combined GR+LW data-movement total. Real semantic equivalence
  needs per-register RAW-edge comparison. F32X-emulation kernels in
  particular have `MFMA` instructions that the default scheduler tags into
  pack-code submodules (`PackB1`/`LRA3`/etc.) while CMS tags as `MFMA`; the
  cross-scheduler `num_mfma` count therefore differs and is currently NOT
  checked by `compare_captures`.
- **Phase 9 documentation gaps**:
  - `SSetPrior` instructions have no CMS id_map equivalent and currently
    fall through to `'UNKNOWN'` category in the default-side capture.
  - DirectToLds=1 kernels share Instruction objects between
    `self.codes.globalReadA/B` and `self.codes.localWriteA/B`; the
    default-side identity-set classification uses first-tag-wins which
    typically tags them `'GR'` or `'LW'` depending on traversal order.
    `compare_captures` aggregates both into a single GR+LW total to
    sidestep the difference.
  - `_makeSubIterSchedule` shadow runs do not currently snapshot/restore
    `vgprPool` because SIA3 doesn't check out vregs (line 1027's checkout
    is in the SIA2 branch). If SIA3 is ever modified to check out vregs,
    the shadow path needs `vgprPool` cloning too.
  - OptNLL+CMS+GSU>1 kernels are explicitly rejected at the capture
    dispatch site (`KernelWriter._loopBody:~4280`) rather than producing
    a partial capture. Lifting this requires Phase 5+ first.

## NGL/NLL hard-coded to code-path 0 — `isValid` does not validate other codepaths' tail behavior

`KernelWriter.py:2858-2866` (CMS branch of `noLoadLoopBody`) emits the NGL and NLL
loop bodies as a single `MAINLOOP` macro invocation with `\ID = 0`:

```python
if isNGLL:
    module.add(MacroInstruction(name="MAINLOOP", args=[0,0,1,1,0]))  # \ID=0
else:
    module.add(MacroInstruction(name="MAINLOOP", args=[0,0,0,0,0]))  # \ID=0
```

There is no `simdSpecDispatch` wrapper for the tail loops. So when a CMS schedule
has `numCodePath > 1`, every SIMD runs **code-path 0's** schedule for NGL and NLL,
including the SIMDs that ran code-path 1+ in the main loop. Code-paths only differ
in scheduling order (slot assignment) — same instructions, same registers — so this
is functionally correct, just suboptimal.

`isValid` currently only inspects `optSchedule[key][cp]` slot lists in the abstract
and does not verify NGL/NLL behavior under each codepath. It should:
- Confirm that code-path 0 of the schedule is valid as an NGL body (after
  `\useGR=0, \usePLR=1, \useGRInc=1, \useLoop=0` macro-guard stripping and
  `nglshift` vmcnt adjustment).
- Confirm that code-path 0 is also valid as an NLL body (after
  `\useGR=0, \usePLR=0, \useGRInc=0, \useLoop=0` stripping, `nllshift`,
  and `nllZeroDscnt`).
- Optionally, flag schedules where code-path 1+ would produce a *different*
  (and possibly better) NGL/NLL body if hardware were available to dispatch
  them — these are missed-optimization opportunities, not correctness bugs.

## Phase 7 — Dataflow graph comparison & typed Failures (in progress)

**Landed (Stack 2 graph machinery, all unit-tested with TDD):**

- Typed `Failure` hierarchy with polymorphic `format(capture)` —
  12 user-facing subclasses + 10 named `Capture*Error` exceptions.
  `ScheduleCapture.py`. Pinning tests in `test_failure_formatters.py`.
- Unified 4-body `DataflowGraph` (single graph, cross-body edges
  represented natively) with `GraphNode` / `DataflowEdge` /
  `GraphPosition`. `BODY_LABEL_TO_LOOP_INDEX` maps body labels to
  `loop_index` so cross-body order is well-defined.
- `build_dataflow_graph(four_part_capture)` with FIFO drain semantics
  for `raw_intrawave` edges and SBarrier collectors for the two
  LDS-reuse patterns (`lr_to_gr_lds_reuse`, `gr_to_lr_lds_reuse`).
- `compare_graphs` + `diagnose_missing_edge` per the phased classifier:
  Phase 0 missing-node guard (raise, not assert), Phase 1
  OrderInverted (same-body only), Phase 2 mutually-exclusive Wait
  failures + role-aware MissingBarrier with double-report suppression.
- `LoopBodyCaptureBuilder.finalize()` capture-pipeline guards:
  `CaptureWiringError` / `CaptureSMEMError` / `CaptureFlatError` /
  `CaptureStoreError` (all `raise`, not `assert` — survive `python -O`).
- `assert_idmap_completeness(idmap, capture)` — pure function; checks
  per-category instruction counts; excludes SYNC/SNOP (CMS adds these
  freely).
- Test infrastructure: `dataflow_fixtures.py` synthetic builders
  (`make_lr`, `make_lw`, `make_gr`, `make_mfma`, `make_swait`,
  `make_sbarrier`, `make_capture`) bypass production
  category-resolution so unit tests exercise the units in isolation.

**Test counts:**
- `test_failure_formatters.py` — 36 tests (formatter wording pinning).
- `test_dataflow_graph_builder.py` — 18 tests (SWait FIFO semantics,
  cross-body queue persistence, structural).
- `test_dataflow_graph_barriers.py` — 9 tests (LR0→GR and GR→LR1
  patterns, strict-ordering invariant, cross-iteration DTL+LdsBuf).
- `test_dataflow_graph_comparison.py` — 11 tests (per-Failure-class
  diagnosis, identity coverage, double-report suppression).
- `test_capture_pipeline_checks.py` — 12 tests (finalize guards,
  idMap completeness).

**Remaining work (Stack 1.3 finish + Stack 2.10–13):**

- **Stack 1 — linter refactor (partial).**
  Migrated to typed `Failure` emission:
  - `SWait.validate()` -> `InvalidCounterValueFailure`
  - `verify_ascending_order()` -> `OutOfOrderSequenceFailure(kind='sequence')`
  - `verify_scc_overlap.verifyIndices()` -> `SCCConflictFailure`
  - `verify_swaitcnt_counters()` -> `SWaitCountExceedsOutstandingFailure`
  - `ValidationRule.run` / `StructuralRule.run` ABCs now accept
    `Optional[Failure]` in their docstring contract; `isValid` boundary
    helper `_failure_to_string` normalizes either return shape.

  Still on `Optional[str]`: `LocalRead.validate()`,
  `GlobalRead._validate_must_start_after()`,
  `GlobalRead._validate_needed_by()`, `Pack.validate()`,
  `MiddlePack.validate()`, the CVT0/CVT1 ordering assertion, and the
  Pack-count rocisa-wiring assertion.

  These rules emit highly contextual error strings (e.g.
  `"PackA0 @ idx=2 issued too early, must be issued after idx=3
   (because of LRA0 issued @ idx=0)."`) that ~30 test sites assert on
  exactly. Full migration requires:
    1. Adding richer fields to `OrderInvertedFailure` /
       `WaitTooLateFailure` / `MissingBarrierFailure` (constraint
       instruction reference, done_idx, name) so the canonical
       formatter can produce the same level of detail.
    2. Updating ~30 `self.validate(..., expected_message=...)` call
       sites in `test_ValidatePack.py`,
       `test_ValidateLRsCompleteBeforeVMFMA.py`,
       `test_ValidateGRsCompleteBeforeLr1s.py`,
       `test_ValidateGRsCompleteBeforeLr3s.py`,
       `test_LR_Pack_interaction.py` to use `isinstance(failure, X)` +
       field assertions instead of text-equality.
    3. Promoting `assert cvt0[-1].issue_index < cvt1[0].issue_index`
       (CMSValidator.py:1642) to emit
       `OutOfOrderSequenceFailure(kind='cvt_pair')`.
    4. Promoting Pack-count rocisa-wiring assert (CMSValidator.py:2270-2277)
       to `raise CaptureWiringError`.

  Best done in a single focused PR per rule.
- **Stack 2.10–13 — production wiring.**
  - Activate `_captureDefaultSchedule` automatically when CMS is in
    use (production-default for CMS kernels).
  - Wire validation pipeline in `customMainLoopSchedule` with
    `try/finally` cleanup; consolidate Phase-5 attributes onto a
    single `CaptureContext` dataclass.
  - Replace `compare_captures` (the data-movement-totals comparison
    at `CMSValidator.py:3434-3438`) with `compare_graphs`. Per-edge
    equality subsumes totals.
  - Integration tests on real CMS YAMLs
    (`custom_mainloop_scheduling.yaml`,
    `custom_mainloop_scheduling_tf32.yaml`).
- **Pre-existing test failures noted during Stack 2 work:**
  `TestPhase5DefaultTailCapture::test_n_gl_n_ll_state_resets_after_kernel`
  and `test_no_false_positive_on_clean_cms_kernel` — both flag
  `compare_captures` data-movement count mismatches (`default=17 vs
  cms=16`). Replacing `compare_captures` with `compare_graphs` should
  resolve or refine these as per-edge Failures.