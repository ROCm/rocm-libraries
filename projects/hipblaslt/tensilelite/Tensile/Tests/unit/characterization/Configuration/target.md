# Characterization target — `Tensile/Configuration.py`

Part of the master-plan remaining-module sweep. Covers `Parameter` (all
comparison + arithmetic/bitwise operators param-param/param-scalar/scalar-param,
explicit reflected dunders, unary/bool, value accessors, type-preservation
raise), `ReadWriteTransformDict` (get/set/toDict/transforms), and `ProjectConfig`
(createValue/createSection, dotted get, contains, resetToDefaults,
getDefaultValue/getDescription, addConstraint + checkConstraints pass/fail via
the AST evaluator). 33 tests.

**Accepted <95% combined — see DECISIONS D9.** Residual: (a) the reflected-
operator `isinstance(lhs, Parameter)` branches are dead (Python dispatch); (b)
the `ExpressionEvaluator` AST walker + `CallableParameter`/`createBinaryOp`/
`createUnaryOp` are a deferred expression-machinery slice (an AST-node matrix).
A quirk noted: setting a section-contained value and dotted-get after
resetToDefaults raise (pinned by avoidance, not asserted).
