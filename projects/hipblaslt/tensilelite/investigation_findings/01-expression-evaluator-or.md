# ExpressionEvaluator 3+ Operand `or`

## Verdict

Open.

The current target source still evaluates only the first two operands of an AST
`BoolOp`, so `a or b or c` is reduced to `a or b`. This still breaks the
`--results-only` cluster path because only the third constraint operand,
`RunResultsStep`, is true.

## Current Source References

- `Tensile/Configuration.py:649-654`: `ExpressionEvaluator.evaluate()` handles
  `BoolOp` by evaluating only `node.values[0]` and `node.values[1]`, then
  returning a single binary operation. It never iterates over `node.values[2:]`.
- `Tensile/Configuration.py:508-510`: `CallableParameter.createBinaryOp()` maps
  `And` and `Or` as binary operations.
- `Tensile/Configuration.py:921-934`: `ProjectConfig.addConstraint()` stores the
  parsed AST and `checkConstraints()` evaluates it with `ExpressionEvaluator`,
  then asserts the resulting value.
- `Tensile/TensileBenchmarkCluster.py:285-288`: `--results-only` sets
  `RunDeployStep=False`, `RunBenchmarkStep=False`, `RunResultsStep=True`, and the
  constraint is `RunDeployStep or RunBenchmarkStep or RunResultsStep`.
- `Tensile/TensileBenchmarkCluster.py:330-333`: command-line flags include
  `--results-only` and `--run-and-results-only`.
- `Tensile/Tests/unit/test_Configuration.py:818-832`: existing unit coverage only
  checks two-operand boolean expressions (`a and b`, `a or b`), so it does not
  catch the three-operand case.
- `Tensile/Tests/unit/test_TensileBenchmarkCluster.py:503-521`: existing
  `--results-only` test mocks `ProjectConfig`, bypassing the real constraint
  evaluation path that fails.
- `Tensile/Tests/unit/characterization/DECISIONS.md:165-180` and
  `Tensile/Tests/unit/characterization/TensileBenchmarkCluster/test_tensile_benchmark_cluster_char.py:68-77`
  already document and pin the bug in the current target tree as well as in the
  referenced investigation tree.

## Reproduction / Static Evidence

Static AST evidence:

```text
BoolOp(op=Or(), values=[Name(id='RunDeployStep', ctx=Load()), Name(id='RunBenchmarkStep', ctx=Load()), Name(id='RunResultsStep', ctx=Load())])
```

Minimal evaluator reproduction run from the target checkout:

```bash
python3 -c "import ast; from Tensile.Configuration import ExpressionEvaluator; expr='RunDeployStep or RunBenchmarkStep or RunResultsStep'; ctx={'RunDeployStep': False, 'RunBenchmarkStep': False, 'RunResultsStep': True}; tree=ast.parse(expr, mode='exec'); print(ExpressionEvaluator().evaluate(tree, ctx))"
```

Observed result:

```text
False
```

The expected Python truth value for that expression and context is `True`.

Cluster path reproduction run from the target checkout:

```bash
python3 -c "import sys; from Tensile.TensileBenchmarkCluster import TensileBenchmarkCluster as TBC; sys.argv=['prog','/logic','/deploy','--results-only'];\
try:\
    c=TBC(sys.argv[1:]); print('constructed', c.workflowSteps())\
except Exception as e:\
    print(type(e).__name__ + ': ' + str(e))"
```

Observed result:

```text
AssertionError: Constraint evaluation failed: RunDeployStep or RunBenchmarkStep or RunResultsStep
```

Control check:

```bash
python3 -c "import sys; from Tensile.TensileBenchmarkCluster import TensileBenchmarkCluster as TBC; sys.argv=['prog','/logic','/deploy','--run-and-results-only']; c=TBC(sys.argv[1:]); print(c.workflowSteps())"
```

Observed result:

```text
(False, True, True)
```

This passes because the second operand, `RunBenchmarkStep`, is true, so the
current two-operand-only evaluator happens to return true.

## Impact

Any `ProjectConfig` constraint using a three-or-more operand `and` or `or` can be
mis-evaluated whenever the decisive value is in `node.values[2:]`. The known
user-visible failure is `TensileBenchmarkCluster --results-only`, which cannot
construct the cluster object and therefore cannot reduce existing results on its
own. Other two-operand boolean constraints remain unaffected.

## Recommended Fix / Test

Fix `ExpressionEvaluator.evaluate()`'s `BoolOp` branch to fold all operands, for
example by evaluating `node.values[0]` and then iteratively combining each
subsequent value with `CallableParameter.createBinaryOp(result, next_value, op)`.
Preserve the existing `CallableParameter` behavior unless broader semantic work
is intended.

Add focused tests:

- `ExpressionEvaluator` unit tests for `False or False or True == True` and
  `True and True and False == False`.
- A real `TensileBenchmarkCluster` construction test for `--results-only` that
  does not mock `ProjectConfig`, asserting `workflowSteps() == (False, False,
  True)`.
