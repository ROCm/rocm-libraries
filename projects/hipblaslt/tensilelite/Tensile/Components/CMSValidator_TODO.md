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

**Stack 1 linter refactor — COMPLETE for all enumerated rules.**

Every rule emits typed `Failure` subclasses; the boundary helper
`_failure_to_string` in `validate_timeline` and `isValid` normalizes
the return shape so existing callers (and ~30 hardcoded message-text
test sites) stay green via `_legacy_msg` per-instance overrides.

Migrated rules:
- `SWait.validate()`           -> `InvalidCounterValueFailure`
- `verify_ascending_order()`   -> `OutOfOrderSequenceFailure(kind='sequence')`
- `verify_scc_overlap.verifyIndices()` -> `SCCConflictFailure`
- `verify_swaitcnt_counters()` -> `SWaitCountExceedsOutstandingFailure`
- `LocalRead.validate()`        -> `MissingWaitFailure` (no SWait) /
                                   `WaitTooLateFailure` (late guarantee)
- `Pack.validate()`             -> `OrderInvertedFailure` (early/late) /
                                   `TimingTooCloseFailure` (quad-cycle)
- `MiddlePack.validate()`       -> `WrongInterleavingFailure`
- `GlobalRead._validate_must_start_after()` -> `OrderInvertedFailure` /
                                   `MissingBarrierFailure(role='must_start_after')`
- `GlobalRead._validate_needed_by()` -> `MissingWaitFailure` /
                                   `MissingBarrierFailure(role='needed_by')` /
                                   `WaitTooLateFailure`

ABCs (`ValidationRule.run`, `StructuralRule.run`) updated to document
the union return type. `Failure` base class refactored: subclass
formatters renamed to `_format_canonical(capture)`; the base
`format()` checks `_legacy_msg` first and falls back to
`_format_canonical()`. `with_legacy_msg(str)` setter on every Failure.

Stack 2 graph comparison (`compare_graphs` / `diagnose_missing_edge`)
constructs Failures WITHOUT `_legacy_msg`, so it gets the canonical
formatter output exercised by `test_failure_formatters.py`.

**Internal-promotion candidates left as-is (not user-actionable bugs):**
- CVT0/CVT1 ordering at `CMSValidator.py:1760, 1822` — construction-time
  asserts inside helper functions; promoting to a Failure would require
  threading the ordering check into the validator state. The asserts
  fire so loudly that promotion is low-priority.
- Pack-count rocisa-wiring at `CMSValidator.py:2385/2391` — already
  a `ValueError` raise (not bare assert); survives `python -O`.

**Stack 2.10-13 production wiring — DONE as opt-in observability.**

`KernelWriter.kernelBody` now invokes `build_dataflow_graph` +
`compare_graphs` alongside the legacy `compare_captures` call when
`_useDataflowGraphComparison` is set on the writer. Logs at
WARN (`printWarning`) on edge differences or capture-pipeline errors;
logs at debug (`print2`) on clean comparison. Does NOT affect build
pass/fail by itself — the legacy comparison still gates assertions.

Runs BEFORE the legacy assertion so it remains observable on kernels
where the legacy data-movement-totals check fires its known false
positive.

Graph builder gained a `strict_unknown_instructions` parameter
(default LENIENT for production; STRICT for unit-test fixture-mistake
catching). Real captures contain many instruction kinds beyond the
LR/LW/GR/MFMA/SWait/SBarrier set the dataflow model covers (scalar
arith for GRInc, pack ops, etc.); lenient mode skips them.

Per-instruction shape extractors (`_inst_dst`, `_inst_lds_offset`,
`_inst_buffer_srd`, `_inst_buffer_offset`, `_inst_mfma_acc/_a/_b`,
`_inst_dsstore_src`) fall back to positional `getParams()` when the
named attribute isn't exposed — bridges the synthetic-fixture and
real-rocisa instance shapes (`BufferLoad*` and `MFMAInstruction`
don't expose ctor args as named attrs; `DSLoad*` does).

Integration test
`TestPhase7DataflowGraphIntegration::test_dataflow_comparison_runs_without_raising`
exercises the wiring on a real F32X TF32 kernel (MI=16,16,32 /
MT=128x128 / DU=32) and verifies dataflow comparison runs without
crashing. Currently surfaces an `UnexplainedMissingEdgeError` on this
specific kernel — diagnosis classifier reaches a fall-through that
hints at either a register-extraction bug for some real-rocisa class
or a graph-builder gap. Tracked as a follow-up.

**F32X TF32 integration diagnostic findings:**

Running `compare_graphs` on the integration kernel (MI=16,16,32 /
MT=128x128 / DU=32, F32XdlMathOp=X, DTL=1, PLR=1, PGR=2) reveals
TWO categories of issues, one fixed and one outstanding:

**Fixed in this session:**
1. `_reg_signature` was ignoring the symbolic `regName`, collapsing all
   named registers (regIdx=-1) to ('vgpr', -1, N). Fixed by including
   `(regName.name, regName.getOffsets())` in the signature when
   regIdx == -1.
2. `_inst_mfma_acc/_a/_b` was reading `getParams()` indices 4/5/6
   assuming the full constructor arg list. Real
   `MFMAInstruction.getParams()` returns `[acc, a, b, acc_or_d, comment]`
   (5 entries). Fixed to read indices 0/1/2.
3. `compare_graphs` identity-mismatch error message was unreadable when
   16+ identities differed. Now emits a class-tag breakdown
   (`{'MFMA': 16}`) plus the first 3 full identities for investigation.

**Outstanding — Phase 5 capture mechanism issues:**

Running with the above fixes, compare_graphs surfaces 16 MFMA
identities in the CMS capture but missing from default. Their
register signatures suggest these are F32X TF32 emulation
`v_mfma_f32_4x4x4_16b_bf16` operations:

  ('MFMA', 0, ('v', -1, 4, ('ValuA_T0_I0', (0,))),
              ('v', 74, 2, ()),
              ('v', -1, 2, ('ValuA_X0_I0', (0,))))

Diagnostic counts (main_loop[0] only):
  DEFAULT MFMAInstruction class total: 56
  CMS MFMAInstruction class total:     64    (+8)
  DEFAULT VCvtPkF32toBF16 total:        32
  CMS VCvtPkF32toBF16 total:            64    (+32)

DEFAULT category distribution shows a tagging gap:
  UNKNOWN: 66  (28 VCvtPkF32toBF16, 16 DSLoadB128, scalar arith)
  Plus 4 MFMAInstruction-class entries in LRB0, 1 each in LRA3/LRB3/GR.

Root causes (require Phase-5 investigation):

1. **Symbolic-vs-numeric register naming gap**: the missing 16 MFMAs
   have one symbolic input (e.g. ValuA_T0_I0) and one NUMERIC input
   (e.g. v[74:75]). The numeric input is allocated at codegen time and
   may differ between SIA3 and CMS even for the same logical MFMA.
   For dataflow comparison to be authoritative, both paths must use
   the same register-naming convention.
2. **Tagging-coverage gap in `_buildCaptureIdentityMap`**: 28
   VCvtPkF32toBF16 and 16 DSLoadB128 land in UNKNOWN because the
   tagger's submodule-name lookups (`LocalReadDoA_I{iui}`,
   `packA_I{iui}`) miss the F32X-specific module structure. (Names
   match in source but the items don't end up where the tagger looks.)
3. **MFMA-class-in-LR-bucket on default side**: SIA3's path puts some
   MFMAInstruction-class items into LRB0/LRA3/LRB3/GR buckets,
   suggesting these MFMAs are scheduled INSIDE LR/GR submodules. The
   tagger then categorizes them as LR/GR, not MFMA — but their identity
   is still ("MFMA", ...) per `_identity_for`'s class-based dispatch.

Until these are fixed, `compare_graphs` is opt-in observability; the
legacy `compare_captures` continues to gate assertions. The wiring at
`KernelWriter.kernelBody:5050+` runs both checks.

- **Replace legacy `compare_captures` with `compare_graphs`.** Currently
  both run; legacy gates assertions. Blocked on the Phase-5 capture-
  mechanism issues above.
- **`_captureDefaultSchedule` auto-activation under CMS.** Today it's
  test-only; production kernels don't capture by default. Plan called
  for production-default; deferred until graph-comparison is reliable
  enough to be authoritative.
- **`CaptureContext` dataclass refactor.** Phase-5 capture state is
  scattered across `writer._last_default_capture`,
  `writer._last_cms_capture`, `writer._last_default_main_capture`,
  `self.states._defaultNGLCapture`, `self.states._defaultNLLCapture`.
  Plan called for consolidating onto a single dataclass with
  `try/finally` cleanup. Mechanical refactor; deferred.
- **Per-rule test migration to type+field assertions** (Stack 1
  ship-gate intent). The `_legacy_msg` shim makes the migration
  optional; test sites can move incrementally as test files are
  touched for other reasons.
- **Pre-existing test failures noted during Stack 2 work:**
  `TestPhase5DefaultTailCapture::test_n_gl_n_ll_state_resets_after_kernel`
  and `test_no_false_positive_on_clean_cms_kernel` — both flag
  `compare_captures` data-movement count mismatches (`default=17 vs
  cms=16`). Replacing `compare_captures` with `compare_graphs` should
  resolve or refine these as per-edge Failures.