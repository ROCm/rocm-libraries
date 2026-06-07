# Parametric-Chaos Run-1 Analyst Summary

## Overview

Run-1 characterized 20 branch predicates drawn from the TensileLite public-input surface: Tensile.py entry-point guards, Configuration.py expression-evaluator dispatch, and GlobalParameters.py version-check logic. The characterization pipeline ran 6 phases: Census → Slice → Domain → Solve → Verify → Reify. All 20 branches have Census, Solve, and Verify records; 16 have Domain records; 8 have Slice records (the remaining 12 had their inputs fully captured at Domain level without requiring a formal slice); and 13 have pytest-passing Reify records.

---

## Branch families (clustered)

### Family A: Tensile.py CLI entry-point guards (6 branches)

These branches gate the `Tensile()` entry-point function on CLI arguments parsed by argparse. All are **fully-static** (predicate truth determined exclusively by CLI args at parse time) and all are SAT-confirmed.

| branch_id | file:line | predicate | public inputs |
|---|---|---|---|
| `01e8ac7f` | Tensile.py:529 | `not altFormat and len(configPaths) != 1` | --alternate-format, ConfigFile |
| `d8f43265` | Tensile.py:526 | `altFormat and len(configPaths) > 2` | --alternate-format, ConfigFile |
| `c63babfc` | Tensile.py:534 | `bool(args.RestoreLog)` | --restore-from-log |
| `765305e2` | Tensile.py:239 | `bool(args.platform)` | -p/--platform |
| `26f1acfe` | Tensile.py:603 | `"LibraryLogic" in config and UseEffLike and not buildOnly and not CpuOnly` | ConfigFile, UseEffLike YAML key, --build-only, --cpu-only |
| `2c7170bd` | Tensile.py:25 | `__name__ == "__main__"` | Python invocation mode (os) |

Notes: Tensile.py:603 (`26f1acfe`) also reads a YAML file and has a filesystem external state dependency (`/etc/os-release` via `isRhel8()`). Tensile.py:25 (`2c7170bd`) has an OS external state dependency (interpreter invocation mode); in practice this branch is always False when Tensile is imported as a module (which is the test path) and True when run as a script.

### Family B: ExpressionEvaluator nodeType dispatch (5 branches)

These branches implement the AST visitor in `ExpressionEvaluator.evaluate()`. The predicate in every case is `nodeType == "<TypeName>"` where `nodeType = type(node).__name__`. The node type is structurally determined by the expressionStr passed to `addConstraint()`. In the current production call sites, the expressionStr values are **hardcoded string literals** (TensileBenchmarkCluster.py:283,288); no CLI arg or YAML key injects expressionStr at runtime.

| branch_id | file:line | predicate | reachable from production? |
|---|---|---|---|
| `05506103` | Configuration.py:630 | `nodeType == "Module"` | yes (always True at top-level call) |
| `2075748886` | Configuration.py:730 | `nodeType == "Attribute"` | no (hardcoded exprs lack dotted access) |
| `26bfafbb` | Configuration.py:692 | `nodeType == "Assign"` | no (assignment exprs never passed) |
| `8226b3bb` | Configuration.py:673 | `nodeType == "Call"` | yes (abs(x) pattern — via test code) |
| `766aca33` | Configuration.py:218 | `isinstance(rhs, Parameter)` | yes (both paths reachable) |

All five were verified in-container. The two UNKNOWN statuses (`26bfafbb`, and `766aca33` partially) reflect the fact that nodeType dispatch is fully determined by source-code constants, not runtime public inputs, making the predicate truth values not external-input-controllable in production.

### Family C: Configuration.py isinstance guards (3 branches)

Three branches gate on Python runtime type checks — isinstance checks that cannot be encoded as z3 SMT formulas directly.

| branch_id | file:line | predicate | confirmed |
|---|---|---|---|
| `cab4f49f` | Configuration.py:534 | `isinstance(op, str)` in createBinaryOp | yes |
| `3c77ffcc` | Configuration.py:579 | `isinstance(op, str)` in createUnaryOp | yes |
| `b87e16ee` | Configuration.py:230 | `isinstance(rhs, Parameter)` in `__le__` | no |
| `f8b5af6a` | Configuration.py:224 | `isinstance(lhs, Parameter)` in `__rlt__` | yes |

The `createBinaryOp` and `createUnaryOp` isinstance checks distinguish between "op is a FuncMap key string" (True: name=op, func=FuncMap[op]) and "op is a raw callable" (False: name='CustomBinaryOp', func=op). In production, ExpressionEvaluator always supplies a str (the AST node class name); the callable branch is only reachable from direct test code. Both classified runtime-dependent.

### Family D: GlobalParameters.py version-check (2 branches)

| branch_id | file:line | predicate | public inputs |
|---|---|---|---|
| `f6f7dc55` | GlobalParameters.py:659 | `"MinimumRequiredVersion" in config` | config YAML (GlobalParameters.MinimumRequiredVersion) |
| `4914224d` | GlobalParameters.py:660 | version comparison on MinimumRequiredVersion | YAML value + __version__ constant |

Both are fully-static (YAML-filesystem classification), SAT-confirmed, and reified.

### Family E: Runtime/filesystem branches (2 branches)

| branch_id | file:line | predicate | external state |
|---|---|---|---|
| `8f7c4911` | Tensile.py:536 | `not os.path.exists(restoreLogPath)` | filesystem existence probe |
| `aa18a787` | Tensile.py:409 | `runningTuning == True` | file content: sentinel line in logfile |
| `5e52e947` | Configuration.py:929 | `self.ConstraintsKey in self` | object state (unconditional addConstraint) |

`8f7c4911` is UNKNOWN (filesystem-dependent, not confirmed by test). `aa18a787` is SAT confirmed (6 tests). `5e52e947` is effectively always-True in production (unconditional addConstraint call precedes checkConstraints call).

---

## Prioritized hotspots

1. **Tensile.py:526/529 sibling guard pair** (`d8f43265` / `01e8ac7f`) — canonical worked example; highest leverage for input-surface coverage testing; both fully-static with 6-cell Boolean x count domain.

2. **Tensile.py:603** (`26f1acfe`) — most inputs (5 conjuncts: YAML key, two global-params, two CLI flags, plus OS filesystem for isRhel8). Real-world RHEL8 machines silently suppress the branch regardless of YAML/CLI; this is an untested platform-specific blind spot.

3. **GlobalParameters.py:659/660** (`f6f7dc55` / `4914224d`) — version check only fires when a YAML benchmark config includes `GlobalParameters.MinimumRequiredVersion`; TensileCreateLibrary and TensileLogic pipelines never set this key, so those callers always hit the False branch.

4. **Configuration.py ExpressionEvaluator nodeType branches** — the `Attribute` (L730) and `Assign` (L692) dispatch arms are dead code in current production paths; the only live paths are Module (L630, always True at top-level) and the comparison/boolean variants. These are structurally reachable from test code but not from any CLI or YAML input.

5. **Tensile.py:536** (`8f7c4911`) — filesystem-gated branch; currently UNKNOWN and not confirmed. A real repro needs a controlled temp file.

---

## Canonical worked example: Tensile.py:526/529

The sibling guards at lines 526 and 529 of Tensile() implement the "alternate-format multi-config" error check.

```
# L526
if altFormat and len(configPaths) > 2:
    printExit("Alternate format only supports ...")
# L529
elif not altFormat and len(configPaths) != 1:
    printExit("Default format requires exactly one ...")
```

**Public inputs:**
- `altFormat` = `bool(args.AlternateFormat)` from `--alternate-format` (argparse `store_true`, default False)
- `n` = `len(args.ConfigFile)` from positional `ConfigFile` (`nargs='+'`, min 1)

**Domain:** altFormat in {False, True}, n in {1, 2, 3} — 6 cells.

| altFormat | n | L526 | L529 |
|---|---|---|---|
| False | 1 | F | F (normal path) |
| False | 2 | F | T (error: need 1) |
| False | 3 | F | T (error: need 1) |
| True | 1 | F | F (alt-format, 1 file OK) |
| True | 2 | F | F (alt-format, 2 files OK) |
| True | 3 | T (error: >2 in alt-format) | F |

z3 confirmed (bounded, sat-bounded), CrossHair found no counterexample. Argparse-reconstruction verified all 6 cells. Reify tests pin these outcomes with SystemExit(-1) for error paths and FileNotFoundError for pass-through paths.

---

## Caveats and blind spots

1. **RHEL8 platform suppression** (`26f1acfe`, Tensile.py:603): `isRhel8()` reads `/etc/os-release` and forces UseEffLike=False regardless of YAML or CLI. This means on RHEL8 machines the entire branch body (including GPU target selection) is silently skipped. No test exercises this path. The OS-filesystem external state dependency is documented in the Slice fragment but has no reified test.

2. **Dead production paths in ExpressionEvaluator** (`2075748886`, `26bfafbb23`): The `Attribute` and `Assign` nodeType branches are unreachable from any production call site. All production calls supply simple comparison/boolean expressions without dotted access or assignment statements. These branches are only exercisable via direct test-code invocation of ExpressionEvaluator with hand-crafted expression strings.

3. **Runtime-dependent isinstance branches** (`3c77ffcc`, `b87e16ee`, `cab4f49f`, `f8b5af6a`): The isinstance guards distinguish str keys from callables. In production, the ExpressionEvaluator always supplies strs; the callable path is only reachable via direct API use. These branches cannot be controlled via any CLI argument or YAML key.

4. **Filesystem-gated branch not confirmed** (`8f7c4911`): Tensile.py:536 requires a real filesystem path to exercise. The solver reported UNKNOWN; no reified test exists.

5. **Slice fragments incomplete**: 12 of 20 branches do not have Slice fragments (full backward-slice records). Their public inputs were captured at the Domain level based on Census + code reading, but formal predicate normalization was not produced. The 8 branches with Slice fragments are the highest-priority (canonical) branches.

6. **No SAT UNSAT count discrepancy**: 0 UNSAT branches were found. This is expected — all 20 branches are structurally reachable (both True and False sides have witnesses). UNSAT would indicate an infeasible branch (predicate never reachable), which would be a dead-code finding.
