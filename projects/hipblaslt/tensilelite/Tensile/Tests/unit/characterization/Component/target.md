# Characterization target — `Tensile/Component.py`

Part of the master-plan remaining-module sweep. **Before 39.5% → after 100.00%
line** (124 stmts, 0 miss; 98.95% blended — 2 partial branches in `find`'s
len-checks). Drives `PartialMatch` (callable/Mapping/nested/scalar + debug),
`matches`/`versions`, `findAll`/`find` (single/none/multiple-RuntimeError/
nested-abstract recursion), `componentPath`/`commentHeader`, and
`LocalRead._getLdsReadMemToken`/`_emitLdsRead`. Uses an isolated private
component hierarchy (see DECISIONS D3) to avoid polluting the global registry.
