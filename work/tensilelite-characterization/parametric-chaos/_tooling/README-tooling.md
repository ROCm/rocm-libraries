# Parametric Chaos Tooling Bundle

Surface: PublicInputSurface
Container: tl-pchaos-tools

## What each receipt proves

### pict — Pairwise Combinatorial Test Generation

Receipt: `pict/receipt.json`

The smoke ran PICT against the real domain model
(`PublicInputSurface/covering_array/model.json`) and produced `pict/cases.tsv`.

Cross-check (`pict_superset_of_stdlib_pairs`): PICT generated 19 rows covering
all 409 2-way pairs. The stdlib covering_array.py produced 17 rows covering the
same 409 pairs. PICT row count exceeds or equals stdlib, and zero stdlib pairs
are missing from the PICT output. Cross-check: PASSED.

Version note: pict does not expose a `--version` flag. The binary is confirmed
present at `/usr/local/bin/pict` and responds correctly to a model input.
Version recorded as `present`.

### codeql — Semantic Def-Use Slice

Receipt: `codeql/receipt.json`
Query: `codeql/slice.ql`
Output: `codeql/slice.json`

The smoke created a CodeQL Python database scoped to two source files
(`Tensile/Tensile.py`, `Tensile/Configuration.py`) to keep DB creation under
180 seconds (actual: 4.1s). A custom QL query computed an interprocedural
backward def-use slice starting from branch L526 (predicate:
`altFormat and len(configPaths) > 2`).

Cross-check (`codeql_superset_of_ast_defuse`): The stdlib oracle (ast-based
intra-function analysis) identified {altFormat, configPaths, len}. CodeQL
resolved all three and additionally reached {args, argParser, userArgs} via
cross-statement SSA chains. No oracle symbol was missed. Cross-check: PASSED.

Packs: `codeql/python-all` (7.1.2) and `codeql/python-queries` (1.8.4) are
baked into the image at build time, so Slice queries resolve offline. To keep
each Slice unit under the 180s watchdog, scope DB creates to small Python
subsets (the smoke used 2 files; DB create took 4.1s).

### atheris — Coverage-Guided Fuzzing

Receipt: `atheris/receipt.json`
Harness: `atheris/fuzz_helper.py`
Witness: `atheris/witness.json`

The smoke ran atheris with `-max_total_time=30 -runs=200000` against the branch
predicate `altFormat and len(configPaths) > 2`. A witness
(`alt_format=True, n_config_files=10`) triggering the True branch was found at
run #63 (wall time: 1s, 101 coverage edges observed).

Cross-check (`atheris_witness_reproduces`): the witness was reproduced directly
against the plain helper. Boundary points confirmed:
- (True, 3) -> True
- (True, 2) -> False (boundary)
- (False, 10) -> False

Fallback oracle: Hypothesis property-based testing. Atheris result agrees with
and extends Hypothesis reach. Cross-check: PASSED.

Version: `atheris.__version__` is not set; `pip show atheris` confirms 3.0.0.

## Verification status

| Tool    | Available | Smoke ran | Cross-check          | Result |
|---------|-----------|-----------|----------------------|--------|
| pict    | yes       | yes       | pict_superset_of_stdlib_pairs | PASSED |
| codeql  | yes       | yes       | codeql_superset_of_ast_defuse | PASSED |
| atheris | yes       | yes       | atheris_witness_reproduces    | PASSED |

All three tools are real-output-verified. No tools are accepted-unavailable.

The stdlib fallbacks (covering_array.py, branch_census harvester, Hypothesis)
remain the oracle and floor throughout. Real tool output must superset stdlib
output; all crosschecks confirm this. The stdlib fallbacks are never replaced.

## Rebuilding the image

```
docker build -f env/Dockerfile.tools -t tl-pchaos-tools work/tensilelite-characterization/env
```

Run from the worktree root. The Dockerfile installs pict, codeql, and atheris
into the image. After build, start the container with the worktree mounted at
`/work`:

```
docker run -d --name tl-pchaos-tools \
  -v $(pwd):/work \
  tl-pchaos-tools tail -f /dev/null
```

## File index

```
_tooling/
  preflight_tools.json        — version/availability check for all three tools
  tooling_summary.json        — assembled cross-check summary (this bundle)
  README-tooling.md           — this file
  pict/
    model.pict                — PICT model file derived from domain model
    cases.tsv                 — PICT output (19 rows, 409 pairs)
    crosscheck.py             — pair-coverage superset verifier
    receipt.json              — smoke + crosscheck result
  codeql/
    qlpack.yml                — qlpack descriptor
    codeql-pack.lock.yml      — locked pack resolution
    slice.ql                  — backward def-use slice query
    slice.json                — query output (symbol set)
    receipt.json              — smoke + crosscheck result
  atheris/
    fuzz_helper.py            — coverage-guided fuzzing harness
    witness.json              — found witness input
    receipt.json              — smoke + crosscheck result
```
