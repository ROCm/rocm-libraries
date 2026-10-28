# Lane: build and packaging (`BUILD`)

Review CMake targets, options, presets, install rules and CI configuration.

## Checks

- **Targets.** Scope of `PUBLIC`/`PRIVATE`/`INTERFACE` matches use; no global
  `include_directories`, `add_definitions` or `link_libraries`; generator expressions for
  build vs install interfaces.
- **Options.** Every new option has a default, a doc string and a reason for the default.
  Code behind a default-OFF option is untested by default. Check the default and the
  registration.
- **Install.** Every new header, library, plugin, data file and test has an `install()` rule
  with the intended `COMPONENT` and a `CMAKE_INSTALL_*DIR` destination. Nothing installs from
  or references the build tree.
- **Exported config.** `*Config.cmake.in` and exported targets list new public requirements;
  `find_dependency` matches `PUBLIC` links.
- **Tests.** New tests register with `add_hipdnn_test` (or the provider equivalent), have an
  install rule, appear in the category YAML, and pass the strict name check
  (`--strict` of `projects/hipdnn/cmake/scripts/test_name_validator.py`). Read this from the
  CMake and YAML; `RUN` confirms by running.
- **Configurations.** Configures under the standalone build and the superbuild, on Linux and
  Windows, and with tests OFF. Read this from the CMake; `RUN` runs the Linux configures.
- **CI files.** Workflow changes keep required checks, do not weaken triggers or skip suites,
  and pin actions and refs. A new `.github/skip-ci-config.toml` pattern matches no path that
  is built or installed.
- **Version.** `version.json` and generated version headers change when the contract does.

Read the full CMake file for each change, not just the hunk: ordering and scope matter.
