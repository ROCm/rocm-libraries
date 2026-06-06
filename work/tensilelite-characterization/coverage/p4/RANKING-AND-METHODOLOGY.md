# P4 — gap ranking + measurement-methodology reconciliation

**Authored by the P4 driver, 2026-06-06.** Source of the ranking:
`coverage/head-unit-coverage.log` (the methodology-A `--show-missing` report behind the
68.85% HEAD baseline), extracted to `coverage/p4/head-term-missing-raw.txt` and ranked in
`coverage/p4/gap-by-miss.tsv`.

## Methodology reconciliation (READ FIRST — supersedes the literal WORKFLOW-SPECS P4 combine)

The WORKFLOW-SPECS P4 script gates on a **methodology-B** combine of `.coverage.seedw_* +
.coverage.kept_*` (seed-subset, 35.89%). That spec predates the methodology-A 68.85% HEAD
measurement. **`BASELINE-AND-PROGRESS.md` §4 is authoritative**: the ≥80% gate is the
**methodology-A whole `-m unit` suite** number (68.85% on HEAD), apples-to-apples with the
develop 22.47% baseline. P4/P5 therefore gate on **methodology A**, not the seed combine.

- **Prior baseline FILE (the BEFORE for round 1):** `coverage/head-unit-baseline.txt`,
  TOTAL **68.85%** (54867 stmts, 15723 miss), commit `6f1e20b1a7f`.
- **Gate command (methodology A — identical to head-unit-baseline so the delta is valid):**
  ```bash
  CON=tl-char; PROJ=/work/projects/hipblaslt/tensilelite
  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.mA -w $PROJ $CON \
    pytest -p no:cacheprovider -m unit --cov=Tensile --cov=rocisa \
    --cov-config=pyproject.toml -n 4 -q Tensile/Tests/unit
  docker exec -e COVERAGE_FILE=$PROJ/.coverage.mA -w $PROJ $CON \
    coverage report --show-missing | tee coverage/p4/master-baseline-<N>.txt | tail -1
  ```
  pytest-cov auto-combines the `-n 4` xdist worker data into `.coverage.mA`. New add-only
  char tests under `Tensile/Tests/unit/characterization/` automatically join this suite, so
  any line they newly execute that is in the term-missing report becomes a real gate gain.
- **Fast keep/drop predictor (methodology B, per new input):** a line listed as missing in
  the methodology-A term-missing report is uncovered by the ENTIRE current suite. So if a new
  isolated `--cov` run (one `docker exec pytest` for the new test, `coverage json`) executes
  that exact line, adding the test WILL raise the methodology-A gate. Cheapest-input agents
  use this to keep/drop before the expensive full gate runs once in Assemble.

## Gate

- **Target:** whole-project methodology-A TOTAL ≥ **80.00%**.
- **Now:** 68.85%. **Gap:** ~11.15 pts ≈ **~6118** covered stmt/branch units.

### Deterministic-gate refinement (decided R1, 2026-06-06)

The single-process `-n4` methodology-A gate is **run-to-run flaky**: `test_cpu_only_switch.py::
test_cpu_only_end_to_end[*]` intermittently fails with `AttributeError: 'int' object has no
attribute 'isSingle'` at `SolutionStructs/Solution.py:1443`. Root cause is a **pre-existing latent
product bug**, not the coverage tests: `SolutionStructs/Problem.py:711 problemTypeToEnum()` mutates
a ProblemType dict **in place**, converting every DataType field (incl. `F32XdlMathOp`) to its int
`.value`. When xdist co-schedules whatever test triggers that mutation on shared state with
`cpu_only_end_to_end` (which later reads `F32XdlMathOp.isSingle()`), the victim breaks. Proof it is
scheduling, not the new tests: `[R1 tests + cpu_only_switch]` under `-n4` passes (twice); serial
`[seeds→cpu]` and `[R1→cpu]` pass; the failure only appears in the full-suite `-n4` population and
flips run-to-run (gate run 1 = 0 failures, run 2 = 3). Adding R1 tests merely shifted the xdist load
buckets so the leaker and the victim co-scheduled.

**Deterministic gate = two coverage processes + combine** (add-only; no source/test edit; coverage
union identical to the single-process run, so still apples-to-apples with the 68.85% baseline):
```bash
CON=tl-char; PROJ=/work/projects/hipblaslt/tensilelite
# Part A — bulk suite, isolating the victim out
docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.mA_main -w $PROJ $CON \
  pytest -p no:cacheprovider -m unit --cov=Tensile --cov=rocisa --cov-config=pyproject.toml \
  -n4 -q Tensile/Tests/unit --ignore=Tensile/Tests/unit/test_cpu_only_switch.py
# Part B — the victim, in its own process (no leaker co-scheduled)
docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.mA_cpu -w $PROJ $CON \
  pytest -p no:cacheprovider -m unit --cov=Tensile --cov=rocisa --cov-config=pyproject.toml \
  -q Tensile/Tests/unit/test_cpu_only_switch.py
# Combine + report
docker exec -e COVERAGE_FILE=$PROJ/.coverage.mA_combined -w $PROJ $CON \
  coverage combine --keep $PROJ/.coverage.mA_main $PROJ/.coverage.mA_cpu
docker exec -e COVERAGE_FILE=$PROJ/.coverage.mA_combined -w $PROJ $CON \
  coverage report --show-missing | tee coverage/p4/master-baseline-<N>.txt | tail -1
```
Both parts must show 0 failed. The pre-existing `problemTypeToEnum` flake is logged as a campaign
finding (candidate for an upstream fix outside this add-only campaign).

## Round plan (gap-driven; one workflow run per round)

### Round 1 — cheap standalone / library-management modules (low-/zero-coverage, import+invoke)
Highest miss-per-effort; reachable CPU-only by import + small-fixture invocation. Cutoff: keep
a test only if its isolated run newly executes ≥ 10 methodology-A-missing lines of its target.

| target file | miss | cover | channel |
| --- | --- | --- | --- |
| `Tensile/GenerateSummations.py` | 107 | 0.00% | import + invoke main() w/ mock argv/logic |
| `Tensile/TensileUpdateLibrary.py` | 97 | 0.00% | import + invoke w/ tiny library fixture |
| `Tensile/TensileRetuneLibrary.py` | 93 | 25.00% | import + invoke |
| `Tensile/TensileMergeLibrary.py` | 133 | 49.55% | merge two tiny logic YAMLs |
| `Tensile/verify_stinky_comment_vs_elf_text.py` | 101 | 9.88% | import + invoke on a fixture |
| `Tensile/BenchmarkProblems.py` | 111 | 64.69% | config_harness ForkParameters breadth |
| `Tensile/LibraryLogic.py` | 535 | 39.50% | parse + analyze tiny logic set |
| `Tensile/TensileBenchmarkLibraryClient.py` | 92 | 19.12% | switch-present client driver path |

### Round 2 — codegen emit widening (THE BULK, ~ +6-8 pts)
Drive more ProblemType / DataType / `ForkParameters` combos through the full
`emit_kernels_from_config` / `emit_kernels_from_logic` path to toggle untaken codegen branches.
Targets: `KernelWriterAssembly.py` (3987), `KernelWriter.py` (1879), `SolutionStructs/Solution.py`
(1328), `Components/StreamK.py` (883), `GlobalWriteBatch.py` (787), `LocalRead.py` (526),
`WorkGroupMappingAlgos.py` (364), `Subtile/*` (313+178+247+132), `Activation.py` (302),
`ShiftVectorComponents.py` (293), `LraTileAssignment.py` (288), `GSU.py` (257), MAC_* (88+85+78),
`AsmStoreState.py`/`AsmAddressCalculation.py` (192+187). Use P1 attribution-{arch}.json to pick
the params that move these lines; cheapest = a new `ForkParameters` fork in a designed seed YAML.

### Round 3 — client / run path (switch present, haveSwitch=true)
`ClientWriter.py` (221), `TensileCreateLibrary/Run.py` (275), client driver tail. The `--cpu-only`
switch makes these CPU-reachable now.

Rounds repeat until ≥80% or a round adds ~nothing and the remainder is provably
GPU-only/unreachable → `CEILING-FINDINGS.md` with file:line evidence.
