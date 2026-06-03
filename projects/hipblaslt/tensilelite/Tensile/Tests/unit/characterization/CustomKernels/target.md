# Characterization target — `Tensile/CustomKernels.py`

Part of the master-plan remaining-module sweep. **Before 24.5% → after 100.00%
line** (49 stmts, 0 miss; 96.92% blended — 2 loop-back partial branches). Drives
isCustomKernelConfig / getCustomKernelFilepath / getAllCustomKernelNames /
getCustomKernelContents (ok + missing) / getCustomKernelConfigAndAssembly /
readCustomKernelConfig (ok + bad-yaml) / getCustomKernelConfig (ok + missing
InternalSupportParams + missing KernArgsVersion) over crafted .s files in
tmp dirs. 11 tests.
