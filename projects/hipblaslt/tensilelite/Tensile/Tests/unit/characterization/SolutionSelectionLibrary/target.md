# SolutionSelectionLibrary.py — characterization target

Pins the pure selection-analysis helpers: getSummationKeys, makeKey,
getSolutionBaseKey, updateIfGT (all 3 branches), updateValidSolutions
(included + remainder branches), analyzeSolutionSelection (CSV-driven, incl. the
`value > valueOld` performance-map update).

Coverage: 109 stmts, 0 missed → 100% line.

Notes: solutions are represented by a tiny hashable mapping wrapper (`Sol`)
since the real code uses Solution objects as dict keys *and* indexes them.
The two Naming imports (getSolutionNameMin/getKernelNameMin) are stubbed —
they need fully-derived Solution state; we pin that updateValidSolutions calls
and stores them.
