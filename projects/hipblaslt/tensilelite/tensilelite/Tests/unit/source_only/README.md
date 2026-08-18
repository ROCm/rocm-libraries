# Source-only unit tests

This directory is collected recursively by source-tree CI and excluded as one
unit from the installed test artifact. The audit for the installed-wheel split
classified these checkout dependencies here:

- release inputs and wheel construction: `test_release_metadata.py`;
- developer Invoke workflows: `test_install_task.py`;
- checkout-only developer scripts: `test_analyze_timing.py` and
  `test_precommit_affected_tests.py`;
- source/AST and cross-component inspection: `test_EnableESM2TrackValuVsrc.py`,
  `test_PlaceholderMerge.py`, `test_StinkyTofuESM2_sparse_guard.py`, and
  `test_specs_amdsmi.py`.

Production-behavior tests remain in the parent unit directory. Their
installed-artifact adaptations land with the corresponding artifact behavior
changes.
