# Parametric-Chaos Run-2 Analyst Summary

## Overview

Run-2 characterized 20 branch predicates drawn from the TensileLite deep public-input surface: BenchmarkProblems.py benchmark orchestration, ClientWriter.py client-launch guards, Toolchain/Validators.py compiler/executable validation, and LibraryIO.py CU-count logic. The characterization pipeline ran 6 phases: Census -> Slice -> Domain -> Solve -> Verify -> Reify. All 20 branches have Census and Solve/Verify records; 4 have Domain records; 4 have Slice records; and 9 have pytest-passing Reify records.

---

## Branch families (clustered)

### Family A: BenchmarkProblems.py benchmark orchestration

These branches control benchmark execution flow — custom kernel matching, cache validity, solution generation, and benchmark result file existence. Most involve YAML config and filesystem state.

| branch_id | file:line | predicate | solver_status | confirmed |
|---|---|---|---|---|
| `0643ca620d99` | Tensile/BenchmarkProblems.py:302 | `solution["ProblemType"] != problemType` | SAT | yes |
| `099093bf09ab` | Tensile/BenchmarkProblems.py:557 | `not cacheValid` | SAT | yes |
| `0d3cd6b0f663` | Tensile/BenchmarkProblems.py:304 | `failOnMismatch` | SAT | yes |
| `6647a7e665fa` | Tensile/BenchmarkProblems.py:586 | `len(solutions) == 0` | SAT | yes |
| `6ff09bcdcc57` | Tensile/BenchmarkProblems.py:657 | `not os.path.exists(resultsFileName) or globalParameters["For...` | SAT | no |
| `83e4a1ea64ad` | Tensile/BenchmarkProblems.py:133 | `not os.path.isfile(cachePath)` | UNKNOWN | no |
| `8e797886ed0f` | Tensile/BenchmarkProblems.py:740 | `globalParameters["ForceRedoBenchmarkProblems"] \` | SAT | yes |

The `0643ca620d99` branch (BenchmarkProblems.py:302) is the canonical custom-kernel ProblemType mismatch guard. z3 confirmed both sides via an abstract state fingerprint. Assumptions: isinstance(problemType, ProblemType) is statically True; ActivationType normalization on line 301 prevents it from being the discriminating field. The Reify test pins 3 true-examples and 2 false-examples.

### Family B: ClientWriter.py client-launch and script-writing guards

These branches control OS-specific client script generation and executable validation.

| branch_id | file:line | predicate | solver_status | confirmed |
|---|---|---|---|---|
| `3ae422d17a07` | Tensile/ClientWriter.py:366 | `os.name != "nt"` | SAT | yes |
| `85cf4fadab76` | Tensile/ClientWriter.py:798 | `not os.path.isfile(clientExe)` | UNKNOWN | yes |
| `927cbfe5d810` | Tensile/ClientWriter.py:787 | `not os.path.exists(metaDataFilePath)` | UNKNOWN | no |
| `e278ed047bbb` | Tensile/ClientWriter.py:574 | `os.path.exists(sourceDir)` | UNKNOWN | yes |

The `3ae422d17a07` branch (ClientWriter.py:366) — `os.name != "nt"` — gates POSIX-specific clock-reset writes. SAT-confirmed; both branches reified (posix -> True; nt -> False). The `927cbfe5d810` branch (ClientWriter.py:787) depends on metadata.yaml filesystem existence — UNKNOWN, not confirmed.

### Family C: Toolchain/Validators.py compiler/executable validation

These branches validate toolchain paths (compiler, assembler, offload-bundler) and POSIX/Windows guard logic.

| branch_id | file:line | predicate | solver_status | confirmed |
|---|---|---|---|---|
| `09380ac263b6` | Tensile/Toolchain/Validators.py:237 | `_exeExists(Path(file))` | UNKNOWN | yes |
| `8fc5b4598eb9` | Tensile/Toolchain/Validators.py:226 | `not any((` | SAT | yes |
| `9a47d378ae60` | Tensile/Toolchain/Validators.py:236 | `Path(file).is_absolute()` | SAT | yes |
| `bfe92c77b1f3` | Tensile/Toolchain/Validators.py:73 | `Path(defaultPath).exists()` | UNKNOWN | no |
| `c03b6953169e` | Tensile/Toolchain/Validators.py:97 | `os.environ.get("ROCM_PATH")` | SAT | yes |
| `cc98dba04c70` | Tensile/Toolchain/Validators.py:86 | `not os.name == "nt"` | SAT | yes |
| `ffb27402fcf8` | Tensile/Toolchain/Validators.py:195 | `os.name == "nt"` | SAT | yes |

Key branches: `cc98dba04c70` (Validators.py:86, `not os.name == "nt"`) — POSIX-only guard raising ValueError for Windows; `ffb27402fcf8` (Validators.py:195, `os.name == "nt"`) — selects Windows device enumerator. Both confirmed SAT with reified tests. `09380ac263b6` (Validators.py:237, `_exeExists(Path(file))`) — filesystem exec-permission probe; UNKNOWN (filesystem state cannot be enumerated by z3) but confirmed by in-container witness.

### Family D: LibraryIO.py CU-count environment probe

| branch_id | file:line | predicate | solver_status | confirmed |
|---|---|---|---|---|
| `6869457874b8` | Tensile/LibraryIO.py:701 | `CU is None` | SAT | yes |
| `d2f6f0df95db` | Tensile/LibraryIO.py:689 | `CU is None` | SAT | yes |

Both `6869457874b8` (LibraryIO.py:701) and `d2f6f0df95db` (LibraryIO.py:689) gate on `CU is None`. The CU value is read from `os.environ.get('CU', None)`. Absent the env var: CU is None -> True branch (gpu-probe / rocminfo). Present: CU is a string -> False branch (direct use). SAT-confirmed via z3 encoding of `os.environ.get` as a free symbolic string-or-None.

---

## Prioritized hotspots

1. **BenchmarkProblems.py:302** (`0643ca620d99`) — custom-kernel ProblemType mismatch: highest coverage leverage because the predicate determines whether a user-supplied kernel is accepted or rejected. The mismatch branch silently skips the kernel in `solve` mode or raises in `failOnMismatch` mode.

2. **BenchmarkProblems.py:657** (`6ff09bcdcc57`) — `not os.path.exists(resultsFileName) or ForceRedoBenchmarkProblems`: canonical worked example (z3 truth-table over 2 booleans, 4 assignments). ForceRedoBenchmarkProblems is True by default (GlobalParameters.py:101), so the FALSE branch (skip benchmark) requires explicit `ForceRedoBenchmarkProblems: false` in YAML or CLI override. This is a blind spot in the default test path.

3. **Toolchain/Validators.py:86/195/226** (`cc98dba04c70` / `ffb27402fcf8` / `8fc5b4598eb9`) — os.name == 'nt' family: three sibling guards that diverge toolchain behavior on Windows vs POSIX. All confirmed and reified. The Windows branch is never exercised in CI (which runs Linux). Any Windows-path regression is dark.

4. **ClientWriter.py:787** (`927cbfe5d810`) — `not os.path.exists(metaDataFilePath)`: UNKNOWN, not confirmed. The metaDataFilePath is constructed from output paths determined at benchmark runtime; the branch is only reachable after a completed benchmark run produces a metadata.yaml. No CPU-only test can exercise this without fabricating the file.

---

## Canonical worked example: BenchmarkProblems.py:657 (`6ff09bcdcc57`)

Predicate: `not os.path.exists(resultsFileName) or globalParameters["ForceRedoBenchmarkProblems"]`

Public inputs:
- `OutputPath` (CLI): determines `resultsFileName` path
- `BenchmarkProblems[N][0]` (YAML): problemType string -> part of groupName -> resultsFileName
- `GlobalParameters.ForceRedoBenchmarkProblems` (YAML): bool, default True

Z3 truth table (2 free booleans: `exists`, `force`):

| exists | force | guard |
|---|---|---|
| False | False | True (run benchmark — file absent) |
| False | True  | True (run benchmark — file absent + force) |
| True  | False | False (skip — file present, not forced) |
| True  | True  | True (re-run — forced) |

The FALSE branch is the UNIQUE assignment `exists=True, force=False`. Because `force` defaults to True, reaching the FALSE branch requires explicit `GlobalParameters.ForceRedoBenchmarkProblems: false` in the config YAML. Confirmed by Verify agent (z3 + in-container truth-table check).

---

## Caveats and blind spots

1. **Filesystem probes** (`83e4a1ea64ad`, `927cbfe5d810`, `bfe92c77b1f3`): UNKNOWN solver status. os.path.exists/isfile calls cannot be enumerated by z3. Both branch outcomes are structurally reachable but no CPU-only pytest witness was produced. These branches can only be exercised with real filesystem state.

2. **Windows branches** (os.name=='nt'): All Windows-path branches are SAT-confirmed but the TRUE branch is never exercised in Linux-only CI. Regressions in Windows toolchain logic are dark.

3. **Slice coverage**: Only 4 of 20 branches have formal Slice fragments (backward def-use chains). The other 16 were characterized at the Solve/Verify level without a complete slice. Public-input counts for those branches reflect solver-phase analysis only.

4. **GPU-dependent branches** (`6869457874b8` LibraryIO.py:701): The `True` branch calls rocminfo. In the CPU-only test environment the true-branch cannot be exercised end-to-end; the Reify test pins the predicate behavior only (not the rocminfo call).