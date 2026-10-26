# ADR 0006: Accept `Configuration.py` below 95% combined coverage

Status:  Accepted
Defect:  none — accepted coverage ceiling, not a bug
Commit:  4640395 + 263910a (PR #7989) — https://github.com/ROCm/rocm-libraries/commit/4640395d117367e3c3f6bae77dc2137ce6df4178 (operators/ProjectConfig) and https://github.com/ROCm/rocm-libraries/commit/263910ae38e648a79410f6b5cb281f868f466c97 (ExpressionEvaluator deferral) — landed on develop via PR #7989's squash merge, 74e4693

## Context
`Configuration.py`'s `Parameter` operator surface, `ReadWriteTransformDict`,
and `ProjectConfig` (sections/dotted-get/defaults/constraints) are
straightforward to characterize. Two remaining pieces are not:

(a) The reflected-operator branches (`isinstance(lhs, Parameter)` inside
`__radd__`/`__rlt__`/etc.) are dead code reachable only via real operators:
Python only dispatches a reflected dunder when the *left* operand does not
implement the forward one, so inside those methods `lhs` is never a
`Parameter`. Reflected *comparison* dunders aren't auto-called by Python at
all (it uses the opposite operator instead). These branches can only be
pinned by explicit direct calls, not by exercising real operator syntax.

(b) `ExpressionEvaluator.evaluate` is a ~70-line `ast` node walker
(`CallableParameter`/`createBinaryOp` and friends). Exhaustive coverage needs
a full AST-node matrix (`BinOp`/`BoolOp`/`Compare`/`Name`/`Num`/...) — a
focused slice of work disproportionate to this module's per-module budget in
this sweep.

## Decision
Cover the `Parameter` operator surface, `ReadWriteTransformDict`, and
`ProjectConfig`. Pin the reflected-operator dead branches by explicit call
where meaningful, and otherwise document them as dead. Defer the
`ExpressionEvaluator` AST-walker matrix as its own future slice
("Configuration-slice-2"). Accept `Configuration.py` below 95% combined.

## Consequences
`Configuration.py` is a partial module. The operator/config surface is
protected; the `ExpressionEvaluator` AST-walker branches remain unpinned
until a dedicated AST-node matrix slice covers them.

**Rejected alternatives:**
- Force the dead reflected branches via `__radd__` internals — impossible
  without a `Parameter` left operand; rejected.
- Build the full `ExpressionEvaluator` AST matrix now — deferred as its own
  slice rather than folded into this module's budget.
