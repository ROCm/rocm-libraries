# Characterization target — `Tensile/Common/RegisterPool.py`

Part of the master-plan remaining-module sweep. **Before 35.1% → after 100.00%
line** (57 stmts, 0 miss). Drives `allocTmpGpr` / `allocTmpGprList` over a real
rocisa `RegisterPool`: default vs explicit alignment, explicit tag, the
broadcast and matching alignment-mod `assert 0` paths, and both overflow paths.

**Characterized bug (pinned, not fixed — add-only):** `ResourceOverflowException`
is defined as a *function* (`def ResourceOverflowException(Exception): pass`),
so the "exception" built on overflow is actually `None`. Without an
`overflowListener` the code does `raise None` → `TypeError`; with a listener the
listener receives `None`. Both behaviours are pinned via `pytest.raises(TypeError)`
and `seen == [None]`.
