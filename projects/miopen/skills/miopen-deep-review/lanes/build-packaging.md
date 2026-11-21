# Lane: build and packaging (`BUILD`)

Review CMake targets, options, install rules, dependency manifests, test registration and CI configuration.

## Checks

- **Targets.** Scope of `PUBLIC`/`PRIVATE`/`INTERFACE` matches use; no global
  `include_directories`, `add_definitions` or `link_libraries`; generator expressions for
  build vs install interfaces. New sources are listed in `projects/miopen/src/CMakeLists.txt` under every backend that needs
  them (`HIP`, `HIPNOGPU`) and, for kernels, added to the kernel lists that embed or install them.
- **Options.** Every new option has a default, a doc string and a reason for the default.
  Code behind a default-OFF option is untested by default. Check the default and how TheRock sets it (`ml-libs/CMakeLists.txt`
  `CMAKE_ARGS`, `FLAGS.cmake`; `ROCK` owns the resolution). Flag-gated split code (`MIOPEN_ENABLE_HIPDNN_WRAPPER`) leaves the
  flag-off build unaffected: `component-ci-miopen.yml` checks that the forwarding parity harness is absent with
  the flag off.
- **Install.** Every new header, library, kernel, database, model and test has an `install()` rule with the intended
  `COMPONENT` (`clients` for test binaries and the dbsync runner) and a `CMAKE_INSTALL_*DIR`, `DATA_INSTALL_DIR` or `DATABASE_INSTALL_DIR` destination.
  Nothing installs from or references the build tree. Databases install per architecture through `MIOPEN_INSTALL_GPU_DATABASES`.
- **Exported config.** `rocm_export_targets` in `projects/miopen/src/CMakeLists.txt` and its `PACKAGE_DEPENDS` list new public
  requirements and match the `PUBLIC` links.
- **Tests.** New gtest sources sit in `projects/miopen/test/gtest/` and build into both the discrete binaries
  (`MIOPEN_TEST_DISCRETE=ON`) and the single `miopen_gtest` (`OFF`, which TheRock uses); their names pass `check_names.py`
  (the `miopen_gtest_check` target) and are unique across files; a pattern in `projects/miopen/test/gtest/test_categories.yaml`
  selects them for the intended tier; the installed `install_CTestTestfile.cmake` registers what the build tree registers. Read
  this from the CMake and YAML; `RUN` confirms by running.
- **Configurations.** Configures with the HIP and HIPNOGPU backends, with CK on and off (`MIOPEN_USE_COMPOSABLEKERNEL`), with
  `BUILD_TESTING` off, and as TheRock configures it (`MIOPEN_TEST_DISCRETE=OFF`, `MIOPEN_USE_MLIR=OFF`). Read this from the
  CMake; `RUN` runs the Linux configures.
- **Dependency manifests.** `projects/miopen/requirements.txt`, `dev-requirements.txt`, `rbuild.ini` and `install_deps.cmake`
  stay consistent with what CMake finds; a version pin moved in one and not the others is a finding. `DEPS` owns whether a new
  dependency is justified.
- **CI files.** Workflow changes keep required checks, do not weaken triggers or skip suites, and pin actions and refs
  (`component-ci-miopen.yml` pins its container image and the rocjitsu ref). A new `.github/skip-ci-config.toml` pattern
  matches no path that is built or installed.
- **Version.** `projects/miopen/CMakeLists.txt` `rocm_setup_version` and `CHANGELOG.md` change when the contract does; the
  public ABI baselines under `projects/miopen/test/public_abi/` change only with an intended API change (`API`).

Read the full CMake file for each change, not just the hunk: ordering and scope matter.
