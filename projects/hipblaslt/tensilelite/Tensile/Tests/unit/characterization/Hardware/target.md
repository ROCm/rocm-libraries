# Characterization target — `Tensile/Hardware.py`

Part of the master-plan remaining-module sweep. **Before 86.8% → after 97.4%
line** (152 stmts, 4 miss). Drives parseDeviceNameToHex (valid/None/invalid),
_extractPciChipIds (None/PciChipId/Or/other), HardwarePredicate.FromISA /
FromHardware (processor-only, cuCount, single/multi/single-string chip-id,
unsupported+mixed warning branches, empty), and __lt__ (TruePred, chip-id
specificity, CU-count priority, processor compare, differing chip-id sets).
Residual: L98 (topological-rank cycle back-edge — needs a cyclic fallback graph)
+ a __lt__ rank sub-path (276-280); both defensive/data-dependent.
