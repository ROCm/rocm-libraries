# PAP implementation checkpoint

## Port to current develop (2026-10-13)

The two checkpoint commits are carried on `users/nhenders/pap-prefetch-plr`,
based on develop `562a252cbabeda26223f64ce09134c46dcb99525`.
The port follows the renamed `tensilelite/` package layout and current
DataParallel policy and register-pool APIs. PAP's CPU interpreter is retained
as the scoped `_pap_isa.py` test helper; upstream GPU control-flow tests remain.

The current base's separate LDS storage for vector epilogues is preserved.
Such epilogues retain ordinary PAP behavior and remain outside the optimized
two-group gate. Epilogues that still overlap prefetched LDS remain rejected,
and the current base's persistent `OutputAmaxD` rejection is unchanged.
DataParallel multicast uses the existing grid and padded-tile overflow checks;
unmerged StreamK spatial-multicast infrastructure is not a prerequisite.

Fresh checks on the port pass 654 focused Python tests (9 skipped), four
code-generation/assembly checks, 69 native barrier/wait-count tests, and
72 focused host argument/launch-summary tests. The full host-test executable
could not link several type-info/MLP symbols; the two relevant test sources
were linked separately against the freshly built host library. Host checks
explicitly put that library ahead of the installed ROCm copy at runtime.
Local logs and results are under `build_tmp/pap-port-checks/`.
The September 30 GPU correctness and performance results below describe the
original checkpoint. GPU validation and performance on this updated base
have not been rerun. The HalfPLR extension below remains unimplemented.
Historical `triage/pap_optimization/` artifacts remain local to the original
checkout and are not copied by cherry-picking these commits.

## Original checkpoint (2026-09-30)

Updated 2026-09-30. Two next-tile global-prefetch groups and retained first-PLR
operands are implemented for the eligible classic gfx1250 DataParallel/StaticGrid
PAP path, including persistent multicast. Final production kernels pass all 30
supported functional comparisons, four smoke cases, and the two original
long-K regression cases. All 24 paired performance runs passed; measured mean
kernel-time reductions range from 1.70% to 7.68% for the four configurations below.
This checkpoint records the integrated fixes developed after `85fae93fd19`.
Work is paused for a local WIP commit; the HalfPLR extension described below
has not been implemented.

## Implemented behavior

- Prepare next-tile addresses while preserving current tile identity. Issue
  group0 into its released LDS bank, advance the descriptor by a positive DepthU
  step, then issue group1 into the other released bank.
- Group0 uses token0; group1 and surviving current final-bank reads use token1.
  Group0 readiness allows the younger group1 transfer to remain outstanding.
  Retained first-PLR operands remain allocated through the epilogue and next-tile
  setup, with typed completion before both first-WMMA consumer routes.
- Multicast has guarded release and publication phases for each group. Producers
  complete and all local waves join before the elected wave signals the cluster;
  all waves then wait for publication. The late group1 local join names both
  producer tokens because it completes all tensor producers. Its cluster phase
  still names token1.
- Phantom peers participate in the same phases while preserving store eligibility.
  Preparation, next-tile, K, and loop-close guards prevent duplicate issues or
  unmatched cluster phases. Resource denial retains the corresponding ordinary
  address or local-read path.

The native cluster pass recognizes token-tagged cluster waits by physical
barrier ID. Its Rule3 forward scan is bounded by the current wait and control-flow
segment, fixing the reproduced out-of-segment anchor assertion. Local producer
joins remain in the explicit cluster-handshake module so Python barrier
reconstruction preserves publication order.

## Long-K correctness repair

GPU validation exposed a missing repeated main-loop tensor wait. The original
PAP1 kernel passed K=0/256/512/768 but failed K=1024/1536 and the ragged batched
case. Matching PAP0 controls passed; PAP1 also failed with only one persistent
assignment, so a next-tile overwrite was not required to trigger the error.
A diagnostic restoring only `s_wait_tensorcnt 1` at the shared LDS1 consumer
boundary passed all seven original shapes.

Physical LDS banks rotate on every trip of the one-copy loop, while its static
token names repeat. Native tensor analysis freezes after its first sweep. PAP's
valid entry drain removed the initial producer state that previously caused a
partial wait to be emitted, although later trips still needed that wait.

The source fix adds the typed `SBarrier(tensorcnt=...)` contract. Native wait
reconstruction preserves the required count, strengthens it when inferred
dependencies demand a deeper drain, and carries its token provenance forward.
The annotation survives legacy barrier lowering and prevents unsafe scheduling
motion or barrier merging. Default `tensorcnt=-1` retains inference alone.
The eligible PAP loop uses count 1 for data or count 2 for paired data plus MX.

The final Python schedule explicitly issues the younger TDM group before this
contracted boundary, after the descriptor update and prior bank's read release.
This ordering is necessary: the initial integration pinned the barrier before
the younger issue, making its partial count insufficient. The final correction
checks the younger issue count and preserves both LDS release and pointer order.

## Validation

- Final focused Python suite: **572 passed, 9 skipped** on the rebuilt production
  runtime, including the tensor-contract and producer-order regressions.
- Native wait-count, barrier, and barrier-merge suites: **46 passed** against
  rebuilt production objects. New coverage includes a fully drained loop entry,
  counts 1/2, stronger inferred count 0, unset behavior, serialization, legalization,
  scheduling motion, and both merge directions. Five new regressions fail against
  the original native implementation. The earlier cluster suite passed 22 tests.
- Installed API, deepcopy, argument validation, and actual native count-1/count-2
  lowering checks pass. The isolated parent and two loky workers load the same
  rebuilt native library; both frozen library names were refreshed.
- Final GPU functional matrix: **30/30 supported comparisons passed** across
  non-MX MC0, non-MX 2x1 multicast, paired-MX MC0, and paired-MX 2x4 multicast.
  Regular problems use `(M,N,B)=(512,32768,1)` and K=0/256/512/768/1024/1280/1536,
  plus `(520,32760,2,1536)` for ragged batched coverage. Only the two MX/K=0
  comparisons are excluded, as described below. The original K=1024/1536
  regression pair and four persistent smoke cases also pass.
- Independent final assembly and physical-bank audits confirm the younger tensor
  issue precedes the partial wait and that later loop trips read completed data.
  Runtime, code-object, input, launch, and source provenance accompany the results.

The two MX/K=0 inputs are **host-side exclusions, not GPU passes**. The existing
MX data generator rejects nonpositive dimensions or strides before launching a
kernel (`shared/mxdatagenerator/lib/include/mxDataGenerator/DataGenerator.hpp`,
lines 638-643). The non-MX K=0 cases pass. No generator behavior was changed.

## Measured performance

All timings use `(M,N,B,K)=(512,32768,1,1536)`, persistent grid 256, and three
samples per PAP setting. Each row compares matching PAP0 and PAP1 configurations;
24 runs cover four representatives, two settings, and three repetitions.
The runs use five warmups, 1,000 enqueues, one sync and benchmark, the host
timer, and a 384 MiB rotating-buffer request. PAP order alternates across the
three repetitions: (0,1), (1,0), (0,1). Full parameters and sample values are in
the report. MC0 means multicast disabled. Times are the means of the three
samples. Reduction is
`100 * (PAP0 mean - PAP1 mean) / PAP0 mean`.

| Configuration | PAP0 mean (us) | PAP1 mean (us) | Time reduction |
|---|---:|---:|---:|
| Non-MX, MC0 | 19.1121 | 18.5422 | 2.98% |
| Non-MX, multicast 2x1 | 17.4121 | 17.1153 | 1.70% |
| Paired MX, MC0 | 20.8585 | 19.6864 | 5.62% |
| Paired MX, multicast 2x4 | 19.1518 | 17.6800 | 7.68% |

These measurements establish improvements for these configurations and this
shape. They do not establish gains for other K values, problem sizes, devices,
or unsupported PAP geometries.

## Evidence and limits

- [Final GPU report](triage/pap_optimization/finish/gpu/ordered_20260930T165416Z/final_gpu_report.md)
- [Final GPU summary](triage/pap_optimization/finish/gpu/ordered_20260930T165416Z/final_gpu_summary.json)
- [Final Python test log](triage/pap_optimization/finish/gpu/ordered_20260930T165416Z/python_tests.log)
- [Installed native test results](triage/pap_optimization/finish/native/tensor_contract_candidate/installed_tests.json)
- [Native diagnosis and fixes](triage/pap_optimization/finish/native/diagnosis.md)
- [Installed API and lowering checks](triage/pap_optimization/finish/native/tensor_contract_candidate/api_result.json)
- [Native runtime provenance](triage/pap_optimization/finish/native/tensor_contract_candidate/import_provenance.json)
- [Performance run ledger](triage/pap_optimization/finish/gpu/ordered_20260930T165416Z/continuation_20260930T170410Z/performance/ledger.json)

The optimization remains restricted to the eligible full-K, two-bank, PGR2/PLR1,
NoTailLoop path and its existing address, type, epilogue, and resource guards.
It does not enable unsupported LocalSplitU, StoreRemap, LDS-writing epilogues,
or other rejected geometries. A conservative full tensor drain remains at next
entry after NLL, store, and setup overlap; group1 could potentially wait until a
later consumer. The current implementation does not claim the latest possible
wait placement or universal performance gains.

Historical snapshots, failed trials, and diagnostic assembly copies are retained
under `triage/pap_optimization/` for traceability. Final results above use newly
generated production kernels with the integrated source fixes.


## Paused HalfPLR follow-up

The requested extension is still pending. The resumed HalfPLR task performed
source/phase analysis and preserved the pre-change source snapshot, but made
no production or test edits and ran no builds, tests, or GPU work before the
user paused implementation. Existing HalfPLR PAP still carries one next-tile
PGR group and uses the shared incremental-descriptor and late-handoff work;
it does not use the classic two-group or retained-first-PLR path.

The intended first scope is full-K gfx1250 DataParallel/StaticGrid, HalfPLR1
(A only), PGR2, ScheduleIterAlg4, with paired MX or non-MX and multicast off/on.
Preserve the original FlyDSL case35 MT256x256, four-wave, DepthU256 geometry
and three-half A allocation. ASE256 truthfully specializes K1536; other
HalfPLR modes and tail contracts must retain their existing fallback.

Preliminary source analysis, not emitted or validated: the existing final-trip
lookahead appears to read next group0 into A halves
`[(loopCopy + 1) % 3, (loopCopy + 2) % 3]` for LoopIters2, plus B_X0 and MX_X0.
After final u0 reads release the current last bank, its descriptor swap and
increment appear to provide a site for next group1; the existing last-trip
issue is currently descriptor-disabled. A future implementation must prove
that release/publication order, preserve or normalize the A phase for next
entry, protect retained ranges through store/setup, and skip matching entry
reads. This is a design lead, not a completed or correctness-checked path.

Prior case35 measurements use `(M,N,B,K)=(512,65536,1,1536)` and grid256.
Each completed matrix passed four correctness runs and 12 interleaved timings:

| Tuning | Multicast | PAP0 mean (us) | PAP1 mean (us) | PAP latency change |
|---|---|---:|---:|---:|
| Original HalfPLR1, MT256x256 | Off | 27.1690 | 27.5571 | +1.43% |
| Original HalfPLR1, MT256x256 | 2x4 | 23.9315 | 24.3569 | +1.78% |
| Retuned HalfPLR0, MT128x256 | Off | 38.2308 | 35.8348 | -6.27% |
| Retuned HalfPLR0, MT128x256 | 2x4 | 35.6249 | 32.3901 | -9.08% |

The HalfPLR0 rows change tile/tuning parameters and are not an isolated HalfPLR
comparison. Literal MT256x256 four-wave HalfPLR0 variants exceeded the VGPR
limit. An eight-wave same-tile attempt compiled, but its first PAP0 dispatch
failed with HSA invalid dispatch / hip701; it provides no valid comparison
and establishes no specific hardware cause.

Local evidence remains under `triage/pap_optimization/finish/gpu/` in
`flydsl35_20260930T171647Z/` and
`flydsl35_halfplr0_20260930T173030Z/known_four_wave/`.
The pre-change snapshot and manifest are under
`triage/pap_optimization/halfplr/implementation/prechange/`.
These generated artifacts remain local and are not part of this source commit.
