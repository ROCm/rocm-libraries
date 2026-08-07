<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Build Implementation Plan: rocm-libraries

Status: Proposed implementation plan  
Decision record: `PythonBuildGrillingDecisions.md`  
Baseline: `users/alvasile/tensile_package` at `941b802c27`

## Summary

This plan is the remaining delta from the current packaging branch to the
accepted design. It does not restate the namespace, package CLI, resource, and
GEKO migration work already present on the branch.

The final device-generation graph is:

```text
canonical wheel -----------------+
_rocisa --------------------------+--> installed/configured build Python
tensilelite-client ---------------+                 |
                                                       +--> logic validation
                                                       +--> create-library
                                                       +--> ext-op generation
```

Current-scope exclusions are source/channel/SHA versioning, exact same-version
source identity, proper rocisa distribution packaging, and the separate Windows
reconstructed-artifact test lane.

## Public Interface Changes

| Interface | Final contract |
|---|---|
| CMake mode | Remove `HIPBLASLT_TENSILELITE_PYTHON_MODE` and its `BUILD`/`SYSTEM` branches. |
| CMake targets | Add `tensilelite-canonical-release-wheel`, `tensilelite-compatibility-release-wheel`, `tensilelite-build-release-wheels`, and `tensilelite-python-build-environment`. |
| Release outputs | Produce exact canonical and compatibility wheels under `tensilelite-release-wheels/`. |
| Client identity | `tensilelite-client --version` prints exactly `<component>+rocm<A.B.C>` without GPU initialization. |
| Client binding | Add `python -m tensilelite_configure_client --client <absolute-path>` and console script `tensilelite-configure-client`; support `--reset`. |
| Runtime resolution | A configured binding is exclusive; otherwise use only the fixed `ROCM_PATH/libexec/hipblaslt/tensilelite/` directory with `tensilelite-client` on non-Windows and `tensilelite-client.exe` on Windows. Never search alternate directories or `PATH`. |
| Binding storage | Store the bare JSON absolute-path string under `~/.tensilelite/bindings/<installation-id>/client.json`, where the ID is derived only from the exact resolved installed package directory. |
| Per-user root | Rename the existing `~/.tensile/helper_cache` default to `~/.tensilelite/helper_cache` and share `~/.tensilelite` with the keyed binding registry. |
| Production artifact | Install `tensilelite-client` only through the runtime component, not the test component. |
| Artifact testing | Package the canonical and compatibility wheels, raw rocisa, and copied installed-wheel tests for a fresh test job; exclude `unit/source_only/`. |

## Commit Plan

### 1. `build(tensilelite): centralize release identity`

- Add a checked-in `VERSION` containing `5.0.0`. Keep `GENERATOR_VERSION`
  independent because it remains a generator/logic compatibility concept.
- Make the root `release_metadata.py` the sole implementation that validates
  `VERSION`, reads the selected SDK's base `.info/version`, and composes a PEP
  440 version such as `5.0.0+rocm10.1.0`. Remove the duplicated compatibility
  implementation and all package-version literals from setup code.
- Add the build-ROCm-root resolver with this precedence: required
  `THEROCK_TOOLCHAIN_ROOT` for TheRock; CMake `ROCM_PATH`; environment
  `ROCM_PATH`; `/opt/rocm` on non-Windows. Standalone Windows without an
  explicit SDK root fails configuration. Never fall back from an invalid
  TheRock root or mutate it. Do not add CMake preflight checks for
  `.info/version` or a platform toolchain inventory: the sole metadata helper's
  actual read/parse and the real CMake/generator consumers provide authoritative
  validation and diagnostics.
- Generate a C++ version header from the composed value and handle plain
  `tensilelite-client --version` before normal option parsing or GPU setup.
  Enforce one stdout line, empty stderr, and status zero.
- Generate that header during the same CMake configuration which computes the
  exact wheel filenames. Treat it as a normal `tensilelite-client` input so
  automatic regeneration on `VERSION`, `release_metadata.py`, or selected
  `.info/version` changes rewrites the header and recompiles/relinks the client.
  Do not add a second build-time version authority or path manifest.
- Add focused metadata tests proving canonical metadata, compatibility metadata
  and exact pin, and generated client identity all use the same value. Add a
  GPU-less native version test.

### 2. `feat(tensilelite): configure installed client bindings`

- Add `_tensilelite_client_binding.py` as the shared pre-runtime policy module
  and `tensilelite_configure_client.py` as the thin installed-state adapter.
  Package both as top-level modules so configuration does not execute
  `tensilelite.__init__`.
- Derive one installation ID only from the exact resolved imported `tensilelite`
  package directory. Regular wheels in different environments and editable
  installs in different worktrees naturally resolve to different directories.
  A reinstall or upgrade into the same directory intentionally selects the same
  binding slot; this is not a collision. Locate the package directory with
  `importlib.util.find_spec` without executing `tensilelite.__init__`, because
  configuration may be establishing the client required by initialization.
- Validate an absolute regular executable, run `--version` with a five-second
  timeout, parse it with `packaging.version.Version`, and require exact equality
  with the installed distribution. Report distinct launch, loader, timeout,
  signal, nonzero-exit, stderr, malformed/extra-output, and mismatch failures.
- Store the existing bare JSON path at
  `~/.tensilelite/bindings/<installation-id>/client.json` using one atomic
  replacement. A successful configure always replaces an existing file at that
  key rather than failing. `--reset` removes that one file for the current
  installation.
  Concurrent configure/reset for the same installation is unsupported; do not
  add `.dist-info` mutation, `RECORD` rewriting, a lock, tombstones, or a
  recovery state machine. The file intentionally survives `pip uninstall`.
- Rename the helper-kernel cache default from `~/.tensile/helper_cache` to
  the single shared per-user `~/.tensilelite/helper_cache`; never nest it below
  an installation-specific binding directory. Its existing content-derived keys
  allow compatible wheels, venvs, and worktrees to reuse entries. Keep the
  explicit cache-directory override and do not automatically migrate the
  disposable old cache.
- Replace the custom PEP 517 backend with standard `setuptools.build_meta`;
  remove `tensilelite.client-path` and all wheel rewriting. Every wheel remains
  byte-for-byte unchanged by configuration.
- Route runtime resolution through the shared module. A present binding is
  exclusive even when broken; without one, resolve only the fixed `ROCM_PATH`
  location. Validate the SDK compatibility tag and client version once during
  package import, then freeze the path for that process.
- Change `invoke install` to editable-install first and invoke the configuration
  tool second.
- Replace the existing build-time binding tests with normal-wheel and editable
  installation-key isolation, multiple-worktree isolation, per-user-root
  isolation, configure/reset and atomic single-file replacement, no-fallback,
  standard-client, failure-diagnostic, process-freezing, cache-root rename, and
  original-wheel-hash tests.

### 3. `build(tensilelite): define release wheel targets`

- Build canonical and compatibility wheels as independent custom outputs with
  exact paths; do not use one shared stamp.
- Give each target its own private staging directory. Clean only that directory,
  build only that target's wheel there, and validate the staged wheel in the
  corresponding independent validator mode. Neither target cleans or builds
  directly in the shared final release-wheel directory.
- Publish by copying that one validated wheel to its exact release path and then
  touching a target-specific completion stamp. Declare the wheel as a byproduct
  and the stamp as the custom-command output so interrupted publication reruns.
  Follow TheRock's copy-then-stamp convention; do not add atomic replacement,
  clear the shared release directory, or modify the sibling target's output.
- Compose the version and exact wheel filenames during CMake configuration.
  Register `VERSION`, `release_metadata.py`, and the selected SDK's
  `.info/version` as configure dependencies so an incremental `cmake --build`
  automatically regenerates the build and install graph before using changed
  filenames. Do not add a build-time wheel-path manifest.
- Install only the configured canonical and compatibility wheel paths with
  `install(FILES ...)`; do not install the release-wheel directory. This filters
  stale build-tree wheels without adding install-prefix cleanup or changing
  TheRock's stage-to-artifact copy behavior.
- Make `tensilelite-build-release-wheels` aggregate both targets. The canonical
  target participates in every device-generation graph. Add the aggregate to
  `ALL` only when the existing
  `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` option is enabled, ensuring the
  normal build produces both install inputs. Do not build the compatibility
  wheel solely for production or make `cmake --install` invoke a nested build.
- Run both independent builds with the selected Python as
  `python -m pip wheel --no-build-isolation --no-deps`. Do not let the
  compatibility build resolve, download, or rebuild its declared canonical
  dependency.
- Rename the validator to `check_release_wheel_contents.py`. Validate both
  versions, the Python 3.10 floor, pure-wheel tags, canonical resources and
  entry points, the compatibility wheel's exact canonical pin and legacy entry
  points, and absence of bindings, tests, native objects, or source-layout
  debris.
- In the independent validator modes, assert canonical `Requires-Dist: rocisa`;
  normalized canonical and compatibility names; exact `py3-none-any` filename
  and WHEEL tags with agreement between them; presence of
  `_tensilelite_client_binding.py` and `tensilelite_configure_client.py`; and the
  `tensilelite-configure-client` console-script target.
- Include package sources/resources, `VERSION`, setup/build metadata, validator
  code, requirements metadata, and `.info/version` in the appropriate target
  dependencies so relevant edits rebuild the exact wheel.
- Keep compatibility-wheel removal isolated: deleting its target and tests
  later must not change the canonical generation graph.

### 4. `build(hipblaslt): generate device libraries from the canonical wheel`

- Remove `SYSTEM`, the private CMake venv, the synthetic staged ROCm root,
  editable installation, bootstrap `.pth`, dependency import checklist, and
  writability/import probes.
- Enforce one centralized invariant for `HIPBLASLT_ENABLE_DEVICE=ON`: build
  `_rocisa`, `tensilelite-host`, `tensilelite-client`, and the canonical wheel;
  force-reinstall that wheel into the found Python with `--no-deps`; configure
  the exact target-file client; then permit generator commands.
- Make the integrated `_rocisa` target follow `HIPBLASLT_ENABLE_DEVICE`
  directly. Remove `HIPBLASLT_BUNDLE_PYTHON_DEPS` as a gate or external-rocisa
  escape hatch for device generation; every such build owns the in-tree target.
  Preserve rocisa's independent standalone build entry point.
- Preserve the client's existing link to `roc::tensilelite-host`; device
  generation therefore enables the host as the client's transitive prerequisite.
  Do not refactor the client into a host-independent executable in this change.
- In the one centralized invariant, reject explicit
  `TENSILELITE_ENABLE_HOST=OFF` or `TENSILELITE_ENABLE_CLIENT=OFF` when
  `HIPBLASLT_ENABLE_DEVICE=ON`. Emit an actionable configure-time error instead
  of silently forcing cache values. Remove contradictory values from maintained
  presets, Invoke arguments, TheRock, and superbuild callers.
- Derive Python components from real targets: `_rocisa` or a future native
  extension requires `Interpreter + Development.Module`; client-only tooling
  requires `Interpreter`; a true host-only build requires no TensileLite
  Python.
- Preserve the existing `_rocisa` stable-ABI specialization: with
  `ROCISA_USE_STABLE_ABI=ON`, require Python 3.12+ and
  `Interpreter + Development.Module + Development.SABIModule`; otherwise keep
  Python 3.10+ and `Interpreter + Development.Module`. Do not propagate either
  extension-only requirement to client-only or host-only paths.
- Expose only `$<TARGET_FILE_DIR:_rocisa>/..` through scoped `PYTHONPATH`. Scope
  the graph-owned build root as `ROCM_PATH` only around wheel and generator
  subprocesses; do not restore an ambient root globally.
- Introduce one custom-command helper that supplies the Python executable,
  scoped environment, sanitizer settings, and dependency on
  `tensilelite-python-build-environment`. Migrate logic, create-library, and
  every ext-op Python command to it. Keep `hipblaslt_gentest.py` on
  `Python3_EXECUTABLE` directly. Run `tensilelite-configure-client` with the same
  command-local loader/sanitizer environment; plain pip installation of the
  already-built wheel does not require `ROCM_PATH`.
- Use a configuration-independent wheel-install stamp. Do not use permanent
  per-configuration binding stamps: before every generator command, invoke a
  state-aware, idempotent check that reads the current keyed per-user binding
  and configures it when it does not select that command's exact
  `$<TARGET_FILE:tensilelite-client>`. Treat this as binding-state correctness
  within the build paths already supported on `develop`, not as new
  Visual Studio-style multi-config support. Preserve the existing raw rocisa
  layout; configuration-specific raw-package staging is out of scope.
- Remove the contradictory `TENSILELITE_ENABLE_CLIENT=OFF` value from
  `hipblaslt-clients` and remove obsolete mode values from all presets and
  Invoke-generated CMake arguments.
- Apply the same canonical-wheel/client graph on Windows. The Windows build
  must produce and production-install the client and use it for canonical-wheel
  code generation. The standard resolver uses the fixed production directory
  with `tensilelite-client.exe`; reconstructed test artifacts remain Linux-only.
- Verify host-only, rocisa-only, client-only, and device configurations; the
  target dependency graph; binding-state refresh on a supported generator path;
  and one filtered device-library generation case.
- Add one parameterized raw-CMake configure test proving that device generation
  rejects explicit host-off and client-off values with their exact actionable
  diagnostics. Add the Q113 positive `hipblaslt-clients` configure test proving
  that the maintained preset selects the complete graph. Do not duplicate this
  unit-level invariant test for Invoke, TheRock, sanitizer, or superbuild entry
  points; retain their existing integration CI coverage.

### 5. `test(tensilelite): package installed-wheel artifact tests`

- Install the two validated wheels under
  `share/hipblaslt/tensilelite/wheels`, copied installed-wheel tests under
  `tensilelite/Tests` with `unit/source_only/` excluded, compatibility tests under
  `compat/tests`, and pytest/category configuration beside them.
- Restore the existing Linux raw rocisa artifact behavior: package `rocisa`,
  `_rocisa`, its required colocated native library where applicable, and
  `rocisa_tests`. Do not create or install a rocisa distribution.
- Remove the duplicate test-component installation of `tensilelite-client`;
  its only installed owner is the runtime component.
- Rewrite the eleven `tensilelite.Tests.*` imports as test-local imports so the
  separately copied tests run with pytest's default `prepend` mode while
  production imports come from the installed wheel.
- Audit every test selected for the copied artifact tree, not only those eleven
  imports, for project-root paths, adjacent production-source reads,
  build/version metadata, checkout-only scripts, and source-relative resources.
  Record each dependency for classification; do not automatically rewrite every
  discovered test or expand the installed artifact.
- Classify by test subject. Production-behavior tests must use the installed API
  or `importlib.resources`; source/build tests for `VERSION`, wheel construction,
  metadata helpers, `tasks.py`, or developer Invoke workflows remain in source
  CI and are excluded from the installed suite. Do not copy source/build inputs
  merely to satisfy relative paths or call a test source-only solely because its
  conversion is inconvenient.
- Move all genuine source/build tests into
  `tensilelite/Tests/unit/source_only/`. Keep recursive source-CI discovery, but
  exclude that directory from artifact installation with one rule and assert it
  is absent from the installed layout. Update any moved tests' source-root
  resolution as needed; do not use per-file exclusions, markers, `-k`, or
  runtime skips as the artifact boundary.
- Do not add a separate `pytest --collect-only` run. Normal quick, standard,
  comprehensive, full, and FFM category invocations collect their configured
  installed paths with the checkout absent and fail on collection errors; retain
  their category-specific options instead of duplicating collection globally.
- Register the `compat` marker and `--run-compat`, with compatibility tests
  skipped by default.
- Make both artifact-runner install phases invoke the active interpreter as
  `sys.executable -m pip install --force-reinstall --no-deps <exact-wheel>`.
  Assert both complete argument lists so the compatibility phase cannot resolve
  its canonical pin, contact an index, or replace the
  phase-1 canonical installation.
- Construct one final reconstructed environment before querying the production
  client's version: set `ROCM_PATH`, platform loader paths, `PATH`, and only the
  raw-rocisa parent on `PYTHONPATH`. Pass that same environment to the native
  `--version` subprocess, both wheel installs, and both pytest phases; native
  loader resolution occurs before the client's version branch reaches `main`.
- Make the thin runner enforce Q078 for that subprocess: five-second timeout,
  zero status, exactly one canonical stdout line, empty stderr, PEP 440 parsing,
  and equality with the selected wheel. Assert exact arguments/environment and
  distinct launch/loader, timeout, signal, nonzero, stderr,
  missing/malformed/extra-output, and mismatch diagnostics.
- Expand compatibility tests to cover every legacy console-script metadata
  entry, delegated function, exact argument order/value, return-code
  propagation, and once-per-process deprecation warning.
- Remove the isolated production pandas use from `GenerateSummations.py` and the
  compatibility wheel's pandas dependency. Parse `benchmark.csv` with
  `csv.DictReader`, then use existing NumPy arrays, `nanmax`, and `polyfit` while
  preserving stripped/quoted headers, first-seen `SizeL` order, numeric and NaN
  behavior, `Cij` selection, and kernel-column lookup. Remove the characterization
  pandas mock and add a focused CSV fixture comparing parsed vectors, maximum,
  and fitted model; compatibility forwarding tests import the real module but
  may still mock expensive execution.
- Validate the installed test layout and run the canonical suite from a fresh
  venv with the checkout absent and only the raw-rocisa parent on `PYTHONPATH`.

### 6. `ci(hipblaslt): provision TensileLite build requirements`

- For native TheRock staged builds, select the rocm-libraries source root once
  through the extended `configure_stage.py` interface. Resolve
  `projects/hipblaslt/tensilelite/requirements.txt` from that root and emit the
  matching `THEROCK_ROCM_LIBRARIES_SOURCE_DIR` CMake argument from the same
  selection. Default to TheRock's submodule; support conventional external and
  arbitrary absolute local checkouts without a second source-root argument.
- In rocm-libraries' Linux, Windows, and hipBLASLt ASAN TheRock wrapper
  workflows, install
  `projects/hipblaslt/tensilelite/requirements.txt` alongside TheRock's
  requirements before configuration.
- Add no TensileLite dependency names to TheRock's root requirements or CMake,
  and leave the existing root `joblib`/`msgpack` compatibility entries and
  `requirements-test.txt` completely untouched. Their eventual removal requires
  separate non-staged local/test provisioning migrations. Do not modify
  lint-only or coverage-only lanes that do not build hipBLASLt device libraries.
- Retain authoritative failures from `pip wheel`, `pip install`, or the first
  real generator command.
- Test native source selection for the default submodule, conventional external
  checkout, arbitrary local override, and a missing requirements file whose
  diagnostic includes the resolved path.

### 7. `docs(tensilelite): document the canonical Python build flow`

- Update only affected README, contributor, agent/reference, and packaging
  documentation.
- Document the canonical controlled-artifact wheel, `--no-deps`, raw rocisa
  `PYTHONPATH`, unbound wheels, configure/reset commands, fixed production
  client path, editable workflow, found-Python single-owner rule, Python 3.10
  requirement, and standalone Windows SDK-root requirement.
- Remove references to `BUILD|SYSTEM`, the CMake private venv/staged root, the
  pip client-path setting, and test-artifact ownership of the client.
- Mark the older packaging plan/decisions as superseded where they conflict
  with `PythonBuildGrillingDecisions.md`; leave unrelated documentation and
  coverage behavior unchanged.

## Validation and Acceptance

- Run focused Python package, binding, validator, and compatibility tests with
  their feature commits.
- Build the canonical wheel, `_rocisa`, client, and one smallest representative
  filtered device library. Confirm a deliberate client-version mismatch fails
  with the identity diagnostic.
- Run a no-clean incremental version-input test and prove that the rebuilt
  canonical wheel and `tensilelite-client --version` change to the same value.
- Verify runtime installation contains the standard client and the test
  component does not duplicate it.
- Install the test component into a fresh environment and verify the exact
  wheel/raw-rocisa/copied-test layout and absence of `unit/source_only/`.
- Exercise a local editable install followed by configure and reset, proving
  the source wheel and editable wheel archives remain unbound and only the
  current keyed per-user binding changes.
- Require existing Linux, Windows, sanitizer, and superbuild CI to pass. Do not
  add a new comprehensive CMake matrix or redesign coverage.

## Cross-Repository Landing

- Develop TheRock against this checkout through
  `THEROCK_ROCM_LIBRARIES_SOURCE_DIR`.
- Finalize or land the rocm-libraries stack before TheRock's final submodule
  update.
- Keep the TheRock gitlink update isolated so review commits survive rebases or
  replacement of the rocm-libraries SHA.
