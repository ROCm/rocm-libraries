# Characterization target — `Tensile/CustomYamlLoader.py`

Part of the master-plan remaining-module sweep. **Before 59.0% → after 97.4%
line** (117 stmts, 3 miss). Drives the event-based parser (parse_general/
sequence/mapping/scalar over all scalar types incl. quoted-null, nested) and the
stream / sequence-item (idx + OOB + root-not-seq) / dict-item (key + missing +
root-not-map) / logic-gfx-arch (seq-string, seq-dict, map-fallback) loaders.
Residual: L9-11 = the `CSafeLoader` import-fallback (CSafeLoader is installed, so
the except arm can't run) + 2 loop-back partial branches.
