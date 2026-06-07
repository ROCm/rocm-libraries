# TensileLite codegen-coverage campaign — recommendations & outcome

Final state of the P0–P5 campaign (P6 mutation validation follows). Companion to
`PLAN-CODEGEN-WORKFLOW.md` (plan), `BASELINE-AND-PROGRESS.md` (provenance),
`golden-governance.md` (goldens).

## Outcome — ≥80% target MET

| Milestone | Whole-project Cover (methodology A, branch-incl.) | Line-only |
| --- | --- | --- |
| develop baseline (`8f9a5fe9ad8`, `-m unit`) | 22.47% | 25.87% |
| our branch pre-P4 (`6f1e20b1a7f`, switch integrated) | 68.85% | 71.34% |
| **P4 final (`663a0390391`, R7)** | **80.70%** | **83.48%** |

**develop → final: +58.23 points. The Stage-2 (P4) expansion added +11.85 points over 7
gap-driven rounds**, all add-only, full `-m unit` 0 failed (3326 passed / 201 skipped), each
committed with a saved receipt (`coverage/p4/master-baseline-R<N>.txt`). Final receipt:
`coverage/p5/master-baseline-final.txt`.

Round deltas: R1 +0.36, R2 +3.32, R3 +2.82, R4 +1.33, R5 +1.50, R6 +0.60, R7 +1.92.

## What worked (reuse these)

1. **Methodology-A gate is the truth** (full `-m unit`, `--cov=Tensile --cov=rocisa`), apples-to-
   apples with develop. The seed-subset combine (35.89%) is only the Stage-1 fast harness.
2. **Deterministic multi-process gate** (bulk `-n4` + `cpu_only_switch` + `ClientPath` +
   `TensileCreateLibraryRun` isolated, then `coverage combine`) — works around a pre-existing
   `Problem.py:711 problemTypeToEnum()` in-place DataType→int mutation that flakily breaks
   `cpu_only_end_to_end` under xdist co-scheduling. See `coverage/p4/RANKING-AND-METHODOLOGY.md`.
3. **Driver owns the gate + commit**, workflows only author + isolated-measure + verify (a
   workflow Assemble *agent* returned prematurely on a Monitor — don't put the 12-min gate in an agent).
4. **Gap-driven, cheapest-first**: rank methodology-A term-missing → author the cheapest add-only
   input that reaches the cluster → keep only if it adds whole-project lines. Drop already-covered.
5. **Feature-family targeting beats knob-sweeping** for the codegen core: the big KWA/KW wins came
   from turning ON whole features (sparsity, multi-index summation, UseE, XCC remap, int8, fp8/MX,
   complex, StreamK fixup, MFMA pack scheduling) rather than parameter nudges.
6. **Relaxed two-run verify to pass/fail** (R7): the strict "identical coverage both runs" rule
   wrongly dropped tests whose coverage merely *jitters* under multiprocessing — those lines still
   count in the combined gate. This unlocked the final +1.92.

## What did NOT work (don't repeat)

- **Arch-breadth via rich per-arch configs** (R6): added ~11 whole-project lines for 6 configs —
  the arch-specific asm-cap arms were already covered by the existing `test_emit_<arch>` tests.
- **Isolated marginal ≠ whole-project marginal**: several tests covered hundreds of lines in
  isolation but +0 to the gate (already covered by the full suite) — e.g. `gsu3` (762 GSU lines
  isolated, +0 gate), R1 BenchmarkProblems/LibraryLogic. Always confirm against the gate.

## Remaining gap (the 9064 missed stmts, ~19.3% — honest classification)

Not pursued to ≥80% because the target is met; documented for future work. Largest residue:

| File | Miss | Cover | Nature of remainder |
| --- | --- | --- | --- |
| KernelWriterAssembly.py | 2718 | 74.9% | rare/advanced emit arms (deep sparsity, exotic dtype combos, error/assert paths) |
| KernelWriter.py | 1189 | 79.4% | rare schedule/helper arms |
| SolutionStructs/Solution.py | 823 | 72.2% | deep validity-reject + rare derivation branches |
| Components/LocalRead.py | 449 | 57.8% | DirectToLds variants partly emit-resistant CPU-only |
| Components/GlobalWriteBatch.py | 338 | 76.7% | rare fused/atomic/remap store combos |

**Provably hard / not cheaply CPU-reachable (ceiling evidence):**
- **`localSplitUGlobalWriteIndices`** (KWA ~13214-13517): LocalSplitU-store did not emit via the
  single-`BenchmarkProblems[0]` config harness (design-rejected R5).
- **Coverage measurement jitter**: codegen runs in multiprocessing workers, so some reachable lines
  are counted non-deterministically under coverage.py `concurrency=multiprocessing`. This *understates*
  per-test coverage; the combined gate mitigates it but a deterministic coverage config (or
  `parallel`-aware combine of worker data) would let future rounds attribute these reliably.

## Recommendations

1. **Maintain the gate** exactly as in `RANKING-AND-METHODOLOGY.md` (deterministic 4-process +
   combine). Re-baseline with the same command for any develop→branch delta.
2. **Golden hygiene** per `golden-governance.md` — treat stable-arch digest changes as regressions.
3. **Upstream-fix candidates (outside this add-only campaign):**
   - `Problem.py:711 problemTypeToEnum()` mutates a ProblemType dict in place (DataType→int) → the
     `cpu_only_end_to_end` xdist flake. A non-mutating copy would let the gate run single-process.
   - A deterministic coverage setup for the multiprocessing codegen path (so per-test coverage is
     reproducible and the strict two-run verify can be restored).
4. **To push beyond 80%**: target the KWA/KW/Solution residue feature-by-feature (deep sparsity
   variants, more validity-reject paths, DirectToLds emit), and consider widening the config harness
   to drive >1 `BenchmarkProblems` group (would unlock LocalSplitU-store and multi-group derivation).
5. **P6 mutation validation** (next): confirm the suite catches regressions; feed survivors back as
   new P4-style targets.
