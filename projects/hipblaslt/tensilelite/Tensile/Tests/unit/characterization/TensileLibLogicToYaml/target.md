# TensileLibLogicToYaml.py — characterization target

Pins the library-logic -> benchmark-config YAML transformers (setGlobalParams,
formProblemTypeYamlData, formGroups, form9BitMIInst, formForkParams,
formProblemSize, formLibraryLogic), writeToTensileYamlFile, the
TensileLibLogicToYaml orchestrator (LibraryIO read/parse stubbed), parseArgs,
and main (single + multi-index suffixing).

Coverage: 199 stmts, 4 missed → 98% line (96.39% blended).

Pinned latent bug (D14): the skipMI / MI-disabled path passes the string "None"
to formGroups().items() -> AttributeError, so `--skipMI` is currently broken.
Tests drive the working MI-enabled path and pin the crash on the broken one.

Residual misses: two yaml representer callbacks and two orchestrator
RuntimeError guards.
