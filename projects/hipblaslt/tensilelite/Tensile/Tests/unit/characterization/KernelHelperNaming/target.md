# Characterization target — `Tensile/KernelHelperNaming.py`

Part of the master-plan remaining-module sweep. **Before 42.3% → after ~33.7%
line on the naming half** (the full-suite number was higher because other suites
incidentally touch it; this suite pins the naming contract directly). Drives
`KernelHelperEnum`, `kernelObjectNameCallables`, and the five `*Names` functions
(conversion incl. SingleBuffer / MultipleBufferSingleKernel-None / default;
activation-enum-header + activation-function incl. the not-`all` empty case;
reduction; beta-only incl. the GSU>1 case) over a real solution + flag variants.

**Accepted <95% — see DECISIONS D6.** The `init*` object-construction functions
(L110-240, ~half the module) build `KernelWriter*` instances — the GPU code-emit
classes excluded by D0; not unit-characterizable here.

NOTE: module loaded via `importlib.import_module` (package shadowing, D5).
