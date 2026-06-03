# Characterization target — `Tensile/Toolchain/Assembly.py`

Part of the master-plan remaining-module sweep. `buildAssemblyCodeObjectFiles`
driven with **stub linker/bundler** + fake kernel dicts (no real subprocess):
compress=True (bundler.compress), compress=False (shutil.move), with/without
`codeObjectFile`, and empty kernels. `makeAssemblyToolchain` (L46-49) is covered
in the full `-m unit` run by the other suites' `assembler` fixtures. Residual:
L82 (`if len(archKernels)==0: continue`) is **dead** — the `defaultdict` only
holds non-empty lists. Suite-alone 89%; full-run ≈98% line.
