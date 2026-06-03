# Characterization target — `Tensile/Common/Utilities.py`

Part of the master-plan remaining-module sweep. **Before 53.4% → after ~95.2%
line standalone** (249 stmts, 12 miss). Drives fastdeepcopy, verbosity+prints
(print1/2/Warning/Exit), hasParam, isExe/locateExe/ensurePath, roundUp/log2/
ceilDivide(+neg/zero)/roundUpToNearestMultiple/choose_multiplier, versionIs
Compatible, elineno, state (all variants), state_key_ordering, hash_combine/
hash_objs, ClientExecutionLock, assignParameterWithDefault, ProgressBar,
SpinnyThing, iterate_progress (len + no-len), DataDirection, isRhel8
(monkeypatched os-release: rhel/other/missing), wmmaV3InputVgprLayout. 29 tests.
Residual: a few edge branches (ensurePath OSError, version minor== detail,
hash_combine shift kwarg). NOTE: module via importlib (D5).
