# Source-only unit tests

This directory is collected recursively by source-tree CI and excluded as one
unit from the installed test artifact. The audit for the installed-wheel split
classified these checkout dependencies here:

- release inputs and wheel construction: `test_release_metadata.py`,
  `test_release_wheel_contents.py`, and `test_unbound_release_wheel.py`;
- CMake presets/options and developer Invoke workflows:
  `test_cmake_device_invariants.py` and `test_install_task.py`;
- checkout-only developer scripts: `test_analyze_timing.py` and
  `test_precommit_affected_tests.py`;
- source/AST and cross-component inspection: `test_EnableESM2TrackValuVsrc.py`,
  `test_PlaceholderMerge.py`, `test_StinkyTofuESM2_sparse_guard.py`, and
  `test_specs_amdsmi.py`.

Production-behavior tests remain in the parent unit directory. Tests that had
read adjacent production sources (`KnownBugs`, `ValidChipId`, and custom-kernel
resource coverage) now resolve the installed package API or resources instead.
