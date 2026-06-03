# Characterization target — `Tensile/Common/GlobalParameters.py`

Part of the master-plan remaining-module sweep. **Before 90.0% → after 99.1%
line** (220 stmts, 2 miss). Drives `restoreDefaultGlobalParameters`,
`printCapabilitiesTable`, `assignGlobalParameters` (same/override/unspecified
merge, env ROCM_PATH/TENSILE_ROCM_PATH/CMAKE_*, compatible+incompatible
MinimumRequiredVersion, recognised/unrecognised/ignored keys, hipcc-probe
success+failure, locateExe OSError re-raise, verbose printCaps), and
`setupRestoreClocks` (handler captured + invoked, with/without rocm-smi). The
process-global dict is isolated per test; `subprocess`/`locateExe`/`atexit`
monkeypatched. Residual: L266 (module-import line) and L669 (Windows-only
HIP_DIR) — unreachable in this Linux/import context.
