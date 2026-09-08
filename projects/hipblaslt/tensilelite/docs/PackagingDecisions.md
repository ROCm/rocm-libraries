<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Packaging Decisions

> **Superseded for Python build integration:** `PythonBuildGrillingDecisions.md`
> is the current decision record for wheel identity, client binding, CMake
> generation, rocisa staging, and installed-artifact testing. Historical
> decisions below remain useful only where they do not conflict with that file.

Status: Accepted for implementation
Decision date: 2026-07-29
Source proposal: `Public.md`

## Purpose

This document is the durable decision log for packaging the hipBLASLt TensileLite
generator. It records the choices made after reviewing the proposal against the
current repository, including the reasons, rejected alternatives, and expected
consequences. Update this document when a decision changes; do not rely on chat
history as the source of truth.

## Current-State Evidence

- The distribution is currently named `tensile`, installs the `Tensile` import
  namespace, and publishes legacy `Tensile*` console scripts.
- The wheel built from the current branch contains 1,304 files, expands to
  approximately 102 MB, and includes 1,002 test files plus
  `Tensile/Tests/unit/build_tmp/CMakeCache.txt`.
- The wheel already contains the six required static headers, all 119 tracked
  custom kernels, `tensilelite_logic/known_bugs.yaml`, and the ductile defaults.
- The branch already contains an `importlib.resources` implementation for
  headers, custom kernels, known bugs, and ductile defaults. Its 29 focused
  resource tests pass and should be retained.
- hipBLASLt CMake still adds source/build directories to `PYTHONPATH`, invokes
  checkout-relative launchers, and passes a checkout-relative known-bugs file.
- `tensilelite-client` is already installed at the desired `libexec` location,
  but only when test-artifact installation and client building are enabled.
- `cpu-gemm-driver` is built beside the client even though it is test-only.
- `rocisa` is an independent native Python dependency with its own existing
  build and packaging workflow. Its future wheel format, Python ABI, native
  artifact ownership, and versioning are explicitly outside this migration.
- The current branch raises Python to 3.10 for both hipBLASLt and rocBLAS.
  TensileLite retains a Python 3.10 floor; rocBLAS policy is unrelated.

## Accepted Decisions

### D1. Canonical Python and command names

Decision:

- The canonical distribution and import namespace are `tensilelite`.
- The canonical commands are `tensilelite benchmark-cluster`,
  `tensilelite create-library`, `tensilelite generate-summations`,
  `tensilelite logic`, `tensilelite logic-to-yaml`,
  `tensilelite merge-library`, `tensilelite retune-library`,
  `tensilelite run`, and `tensilelite update-library`.
- `python -m tensilelite` dispatches the same command surface.
- The default wheel never installs a top-level `Tensile` package.

Reasoning:

The existing `Tensile` name collides with upstream rocBLAS Tensile and keeps
downstream users coupled to checkout layout. A single command family gives the
package one discoverable public interface without multiplying console scripts.

Consequences:

- The Python package directory is physically renamed.
- Production imports, tests, generated Python snippets, CMake, tox, coverage,
  and documentation must move together.
- C++ `include/Tensile`, generated kernel names, logic fields, and the host ABI
  are not renamed.

### D2. ROCm-only release model

Decision:

Released `tensilelite` wheels are supported only with a complete ROCm SDK on
the matching base release line. ROCm publishes the wheel through its wheel
index and owns `tensilelite-client` under the selected ROCm root. The SDK may be
a filesystem installation or TheRock's matching Python SDK with `devel`. A
suitable `rocisa` distribution is supplied independently.

Reasoning:

The generator depends on ROCm compiler tools, native components, and the
production client that TensileLite does not vendor. TheRock may supply that SDK
through pip, but the TensileLite wheel remains coupled to the matching ROCm SDK
rather than independently portable.

Rejected alternatives:

- Standalone release wheels that vendor the client and native dependencies.
- Dual standalone and ROCm-coupled release channels.
- Installing wrappers directly into arbitrary system Python site-packages.

Consequences:

- `pip install` can install a wheel without ROCm, but first import fails with an
  actionable error.
- Clean test environments install the matching complete ROCm SDK and TensileLite
  wheels from the same ROCm/local wheelhouse.

### D3. Version scheme and generator-format version

Decision:

- `tensilelite` uses versions such as `5.0.0+rocm7.2.4`.
- `tensilelite` declares `rocisa` as a dependency without choosing rocisa's
  release version or local-version policy.
- The existing generator/logic compatibility version `5.0.0` becomes a separate
  named constant and remains the value written to `MinimumRequiredVersion`.
- `tensilelite.__version__` reports the distribution version.

Reasoning:

The component versions and generated-logic compatibility version have meaning
independent of the ROCm release. The local version segment identifies the exact
binary set without changing those component versions.

Rejected alternatives:

- Giving all packages the ROCm release number as their base version.
- Keeping component-only versions and relying on users to match artifacts.
- Reusing `__version__` for both distribution and generated-logic compatibility.

Consequences:

- PEP 517 metadata must be generated from `ROCM_VERSION` or the selected
  installation's `.info/version`.
- Package builds fail clearly when no target ROCm version can be determined.

### D4. Runtime compatibility check

Decision:

Compatibility is established by:

1. Resolving the active Python's TheRock SDK root when present, otherwise an
   explicit `ROCM_PATH`, otherwise `/opt/rocm` on Unix. Windows without either
   a Python SDK or `ROCM_PATH` fails clearly.
2. Comparing the exact base value of `<resolved-root>/.info/version` with the
   wheel's `+rocmA.B.C` version segment.
3. Requiring fixed native artifact locations.
4. Successfully importing the independently installed `rocisa` package.

No artifact manifest, hashes, or component ABI handshake is added.

Reasoning:

ROCm package management is responsible for file integrity. Exact release
matching plus fixed layout and native loading is sufficient for this package
boundary and avoids hashing binaries during every process import.

Consequences:

- Errors must report the expected release, discovered release, selected root,
  root-selection mechanism, failed artifact, and remediation.
- rocisa loader and native-dependency errors are reported as dependency import
  failures without TensileLite interpreting rocisa's binary layout.

### D5. Import-time native requirements

Decision:

- `import tensilelite` requires `import rocisa` to succeed and also requires the
  ROCm-installed `tensilelite-client`.
- TensileLite does not inspect where rocisa stores its extension, how it loads
  native dependencies, or which Python ABI it uses.
- `--prebuilt-client` is removed.

Reasoning:

The wrappers and native artifacts are one release unit. Removing the override
avoids a second, weakly versioned client path and makes import behavior
deterministic.

Rejected alternatives:

- Checking the client only when `run` starts.
- Keeping a developer or sidecar-verified client override.
- Allowing an environment-variable client fallback.

Consequences:

- Even `tensilelite --help`, `logic`, and `create-library` require the client to
  be staged.
- Development and CI must create a complete temporary ROCm artifact root.

### D6. rocisa packaging and ABI are deferred

Decision:

- This migration makes no stable-ABI, wheel-tag, extension-location,
  `libstinkytofu`, or native ownership decision for rocisa.
- TensileLite assumes a properly packaged and importable rocisa is present.
- TensileLite requires Python 3.10 or newer. rocisa will establish its own
  Python support policy during its later packaging-preparation step.
- Existing rocisa source, build options, developer tasks, tests, and release
  metadata are not migrated as part of the TensileLite package cutover.

Reasoning:

The objective here is the TensileLite package boundary. Choosing rocisa's ABI
now would couple two migrations and constrain rocisa before its requirements
have been reviewed independently. The canonical TensileLite and compatibility
wheels are pure Python, so they have no Python extension ABI to stabilize; the
earlier stable-ABI question applied only to rocisa's native extension.

Consequences:

- TensileLite CI verifies only that its supported environment can import
  rocisa. rocisa's own CI owns native and cross-Python ABI coverage.
- `HIPBLASLT_BUNDLE_PYTHON_DEPS` remains a legacy rocisa build concern. It
  defaults off, the rocisa-only preset opts in explicitly, and the new
  TensileLite `BUILD|SYSTEM` environment selection neither redefines nor forces
  it.
- Existing rocisa-specific development and coverage commands may opt into the
  legacy in-tree build explicitly; that does not make rocisa a TensileLite
  package artifact.

### D7. Source-build Python environment

Decision:

Use a CMake-owned staged ROCm artifact root and private Python environment for
source builds. The stage copies `tensilelite-client`, writes `.info/version`,
installs TensileLite locally, inherits the selected environment's independently
installed rocisa, and is a prerequisite of every code-generation command.

Reasoning:

This exercises the installed-artifact contract without modifying `/opt/rocm`,
depending on a preinstalled wheel set, or injecting source directories through
`PYTHONPATH`.

The two modes cover opposite sides of that contract. `BUILD` is the producer
path for source development: it stages the artifacts built from the current
checkout without requiring them to be installed into the system ROCm tree.
`SYSTEM` is the consumer path for packaging and downstream environments: it
uses and validates the version-matched TensileLite wheel, client, and rocisa
already supplied by the selected ROCm installation. Keeping both modes lets the
same hipBLASLt build exercise source and installed-artifact workflows without
forcing source developers to preinstall a wheel or packaged builds to rebuild
TensileLite from source.

Consequences:

- Code generation waits for the client/runtime stage even when it does not
  execute the client.
- CMake offers explicit build-local and system-installed Python modes.
- Pure Python dependency provisioning remains an invoke/CI responsibility; the
  CMake staging path performs no implicit network access.

### D8. Ext-op generators

Decision:

AMax, layer-normalization, softmax, and ext-op library generators become
installed internal modules under `tensilelite._extops`. They are callable by
CMake but do not appear in the public `tensilelite` command list.

Reasoning:

Ext-ops are enabled by default, so leaving them as checkout scripts would keep
the hipBLASLt build checkout-dependent. Publishing them as public commands would
create additional interfaces with no demonstrated downstream requirement.

### D9. Compatibility scope and lifetime

Decision:

- Publish a separate `tensilelite-tensile-compat` wheel containing legacy
  `Tensile*` command wrappers only.
- It does not install a `Tensile` import namespace; `import Tensile` continues
  to fail.
- Wrappers print a once-per-process deprecation warning to stderr and delegate
  arguments and exit status unchanged.
- Support the compatibility wheel for the initial release and the next ROCm
  major. If introduced in ROCm 7.x, remove it at ROCm 9.0.

Reasoning:

Command wrappers give scripts a bounded migration window without loading the
same implementation under two Python module names, which would break class
identity and module-global state.

### D10. Runtime wheel boundary

Decision:

The `tensilelite` wheel contains production Python code and only these data
resources:

- `TensileTypes.h`
- `tensile_bfloat16.h`
- `tensile_float8_bfloat8.h`
- `KernelHeader.h`
- `ReductionTemplate.h`
- `memory_gfx.h`
- all tracked `CustomKernels/*.s`
- `tensilelite_logic/known_bugs.yaml`
- ductile defaults

Tests, source launchers, Invoke tasks, CMake helpers not used at runtime, native
artifacts, rocisa source, and all build output are excluded.

Reasoning:

An explicit allowlist makes accidental package contents a test failure rather
than a growing public contract.

### D11. Optional fallback dependencies remain available

Decision:

- A dependency used by an optional runtime path or fallback remains published
  as an installable extra rather than disappearing from package metadata.
- The supported extras are `profile` for yappi, `hip-query` for hip-python,
  and one extra per JSON backend: `orjson`, `ujson`, and `simplejson`.
- The JSON selection order remains orjson, ujson, simplejson, then Python's
  standard-library `json`. Users may install one backend or combine extras.
- Test runners, source-checkout task frameworks, and native build backends do
  not become runtime extras because they are development/build tools rather
  than optional runtime capabilities.

Reasoning:

Optional dependencies should remain discoverable and easy to request without
making every installation carry redundant JSON implementations, a profiler, or
a platform-specific GPU binding. Separate extras preserve user choice while
keeping the default runtime portable and minimal.

Consequences:

- Wheel metadata and documentation expose every supported extra explicitly.
- CI verifies the `Provides-Extra` and conditional `Requires-Dist` entries.
- Adding a new optional fallback requires adding a corresponding extra rather
  than merely deleting it from the mandatory dependency set.

## 2026-08-04 Amendment: Source-development client binding

The following decisions restore the ability to pair an installed-from-source
TensileLite Python package with a source-built `tensilelite-client`, while
preserving the deterministic release-wheel boundary established above.

These decisions supersede two narrower parts of the original record:

- D3's allowance for `ROCM_VERSION` to override the selected ROCm
  installation's release metadata is removed. Package compatibility is derived
  from the selected ROCm root only.
- D5's prohibition on every non-ROCm-layout client path is narrowed to released
  wheels and runtime overrides. A source build may freeze one custom client
  path while the Python distribution is built; that path cannot be changed at
  runtime.

The remainder of D3 and D5, including exact release coupling, import-time
validation, fixed release layout, and independently supplied rocisa, remains in
force.

### D12. Two immutable client-binding modes

Decision:

TensileLite has two client-binding modes. The mode is fixed when the Python
distribution is built and cannot be replaced by a command-line argument, YAML,
environment variable, or mutable Python global.

1. **Standard ROCm-relative binding** is the default. When no custom build
   setting is supplied, the client is resolved from the version-validated ROCm
   root at:

   ```text
   libexec/hipblaslt/tensilelite/tensilelite-client
   ```

   Windows uses the platform executable name. The relative locator is fixed;
   `ROCM_PATH` continues to select the complete ROCm installation rather than
   independently selecting a client.

2. **Custom absolute binding** is available only while building a normal or
   editable distribution from source. It freezes the exact absolute path
   supplied by the developer. The final symlink is preserved so a stable
   symlink may be retargeted, but `~` and `..` are normalized. Relative input is
   rejected.

A custom binding has strict semantics. If its file later disappears, becomes
non-executable, or points at a broken symlink, TensileLite fails and never
falls back to the standard ROCm client. Rebuilding the executable at the same
path or retargeting the preserved symlink requires no Python reinstall.

Reasoning:

Released wheels must work under `/opt/rocm`, non-standard prefixes, and
equivalent relocated ROCm installations, so they cannot embed the temporary
absolute prefix used by a release builder. Source development needs the
opposite property: Python must keep using the exact client under development
without requiring an override on every invocation. Freezing either the
ROCm-relative release contract or one explicit absolute development path gives
both workflows deterministic behavior.

### D13. PEP 517 build interface and minimal persistence

Decision:

The canonical source-build interface is the namespaced PEP 517/660 setting:

```bash
python -m pip install -e . \
  --config-settings=tensilelite.client-path=/absolute/path/to/tensilelite-client
```

The shorter `-C` spelling is equivalent. The setting works for editable and
non-editable source builds. Installing an already-built wheel cannot consume
the setting because wheel installation does not invoke a build backend; such
wheels always use the binding already present in the artifact.

A thin in-repository build backend delegates ordinary behavior to
`setuptools.build_meta`, consumes only `tensilelite.client-path`, and forwards
unrelated settings such as `editable_mode=compat`. Pip never builds the C++
client. The executable must exist before the Python build starts.

PEP 517 config settings are transient, so a custom binding is persisted as one
minimal, installation-specific file in the distribution's `.dist-info`
directory. The file contains only the absolute path as a JSON string and is
listed in wheel `RECORD`. No file is emitted for the standard binding: absence
means the built-in ROCm-relative default. Sdists never contain a machine-local
path.

The source checkout is never modified. Consequently, two Python environments
may share one editable checkout while retaining different client bindings.
Reinstalling the distribution is the only supported way to replace a frozen
path.

Reasoning:

`config_settings` is the modern installer-to-backend transport; legacy
`setup.py`, `--install-option`, and `--global-option` interfaces are not
appropriate. A generated source module would be shared by every editable
environment and let one environment overwrite another. Mutable user config
would no longer be build-frozen. Per-installation `.dist-info` is already owned
and removed by pip, works for both wheel and editable installations, and adds
only the state that the exceptional custom mode actually requires.

### D14. One private runtime client service

Decision:

Client selection is internal package state, not a public or configurable
Tensile global parameter. Remove the branch-added public and legacy plumbing:

- `tensilelite.TENSILELITE_CLIENT_PATH`;
- public `tensilelite.RUNTIME` and `RuntimeInfo`;
- `globalParameters["ClientExecutable"]`;
- `--prebuilt-client` and every other runtime client-path override.

An internal runtime service has only two responsibilities:

1. initialize during `import tensilelite`, validating rocisa, ROCm release,
   binding metadata, and the selected client; and
2. return the process-frozen client path to `ClientWriter` when it writes a run
   script or launches the client.

Retrieval rechecks that the selected file still exists and is executable, but
never chooses a different path. No production caller needs a public runtime
object containing the ROCm root, release, and client.

Reasoning:

The prior `globalParameters` entry was retained as legacy plumbing after
`PrebuiltClient` was removed. It made installation state mutable, required each
entry point to restore it after `restoreDefaultGlobalParameters()`, and could
allow YAML assignment to replace it. A small private service supplies every
entry point consistently without expanding public API or threading a new path
argument through the legacy benchmarking call graph.

### D15. ROCm compatibility has one source of truth

Decision:

Build frontends supply one base ROCm version to the release-metadata composer.
TheRock derives that value from its graph-owned `version.json` and passes it
explicitly to hipBLASLt; standalone frontends derive it from the selected root's
`.info/version`. The canonical and compatibility package metadata flows do not
accept a loose ambient `ROCM_VERSION` override.

At runtime, TensileLite resolves one root through the established Python-SDK,
explicit-root, and platform-fallback order, reads that root's `.info/version`,
and requires an exact base-release match with the distribution's `+rocm...`
local version. Full nightly, RC, and development publication identity is
deferred. Equality means release equality, not filesystem-path or inode
identity, so an equivalent ROCm installation may be relocated or selected
through a symlink.

Custom client selection overrides only the client locator. It never bypasses
the ROCm release check or the requirement that rocisa import successfully.
Non-standard ROCm installations remain environment-selected through
`ROCM_PATH`; the Python package does not attempt to persist or activate an
entire ROCm environment.

Reasoning:

`ROCM_VERSION` and its `rocm_version()` packaging helper were introduced on
this branch to let pure-Python metadata declare a target without inspecting a
ROCm installation. That permits an invalid artifact such as a wheel labeled
for ROCm 7.3 while `ROCM_PATH` and its native client come from ROCm 7.2.4. The
integrated build already reads the real root, writes the same release into its
stage, and builds against that stage, making the override both redundant and a
way to conceal mismatches.

### D16. One-command and manual source-development workflows

Decision:

The individual building blocks remain supported, including direct CMake,
`invoke rocisa`, `invoke build-client`, and manual pip installation with the
PEP 517 setting. In addition, a Linux-only `invoke install` task is the complete
TensileLite source-development bootstrap after cloning into a supported ROCm
development environment.

`invoke install`:

1. uses the active Python interpreter and never creates or silently activates a
   virtual environment;
2. installs the shared development requirements dynamically from a factored
   requirements file;
3. runs the existing rocisa task, producing the specialized editable in-tree
   rocisa build;
4. runs the existing client build and refreshes the synthetic ROCm stage;
5. installs TensileLite editably into the active environment, freezing the
   absolute path of the staged client; and
6. verifies rocisa and TensileLite imports, exact ROCm compatibility, and the
   selected client.

The task is always editable, repeatable, and incremental. Advanced or
non-editable installs use the manual interface. The developer must already have
Python, Invoke, the ROCm SDK/compiler, and required system build tools because
no Invoke task can bootstrap the interpreter that runs it.

Requirements are factored so direct pip users retain a complete interface:

```text
requirements-dev-common.txt  # shared runtime, native-build, and test tools
requirements-dev.txt         # includes common requirements plus ./rocisa
```

Thus `python -m pip install -r requirements-dev.txt` continues to install the
complete direct-pip development dependency set, while `invoke install` consumes
the common file and then owns the specialized rocisa installation. Optional
runtime extras such as profilers, hip-python, and alternate JSON backends remain
explicit rather than part of the default bootstrap.

Reasoning:

One command is the appropriate ergonomic interface for the common development
loop, but the underlying commands must remain visible and independently useful.
Invoke already owns ROCm and GPU-target discovery, compiler selection, client
flags, and cross-platform command construction. Factoring the requirements
avoids hard-coding a second dependency list, parsing pip requirement syntax, or
building rocisa twice.

### D17. Reuse the staged client and preserve release-layout validation

Decision:

The existing synthetic ROCm stage remains part of integrated CMake `BUILD`
mode. It owns `.info/version` and the copied client at the standard `libexec`
location while reusing the selected ROCm toolchain. The copy is inexpensive and
continuously tests the layout expected from a released ROCm artifact without
writing into `/opt/rocm`.

Both automated source workflows consume the same staged client:

- the CMake-owned private environment resolves it through the standard
  ROCm-relative binding; and
- `invoke install` freezes its absolute staged path into the developer's active
  editable environment.

The fully manual interface may bind any valid absolute client. After changing
client sources, rerunning `invoke build-client` or `invoke install` refreshes
the stage. A clean build deliberately leaves the frozen path invalid until the
stage is rebuilt, which strict validation reports.

Official release wheels are built with no custom config setting against a
matching staged ROCm root. They contain no custom path record, use the standard
ROCm-relative binding, and are checked for accidental custom metadata before
publication. A locally built non-editable custom distribution necessarily
passes through a wheel internally; such wheels are allowed as machine-local,
non-relocatable artifacts but must not be distributed.

Reasoning:

Binding the active environment directly to the original CMake output would add
a third artifact path for little benefit. Reusing the existing staged copy
keeps the integrated and convenience workflows aligned and preserves continual
release-layout coverage. Release wheels cannot embed the stage's absolute path
because that prefix exists only on the build worker.

### D18. Validation and acceptance boundary

Decision:

Custom-path validation is deliberately deterministic. At build time and import
time, require an absolute path to a regular file and POSIX execute permission
where applicable. Validate again when internal code retrieves the client.
Do not execute the client, inspect its linked libraries, hash it, or attempt a
GPU operation during package installation or import. The developer who
explicitly binds a source client owns source-revision compatibility; functional
tests validate actual execution.

The change is complete only when tests cover:

- normal and editable backend builds with and without the custom setting;
- invalid, relative, missing, directory, non-executable, malformed, and broken
  custom bindings;
- final-symlink preservation and correct wheel `RECORD` contents;
- strict no-fallback behavior and the standard ROCm-relative default;
- exact release and nightly matching for both binding modes;
- isolation of two editable environments sharing one checkout;
- non-editable source installation and sdist exclusion of local paths;
- absence of public runtime/client constants and mutable global/YAML overrides;
- `invoke install` ordering, repeatability, active-interpreter targeting, and
  staged-client selection;
- integrated CMake standard-layout validation; and
- release-wheel rejection of custom binding metadata.

Reasoning:

Filesystem and release checks catch deterministic configuration errors without
requiring a GPU or making packaging depend on client command-line behavior.
Testing every boundary is necessary because the feature crosses PEP 517 wheel
construction, PEP 660 editable installation, import-time validation, legacy
client launch code, CMake staging, and developer orchestration.

### Rejected alternatives for source client binding

- **Copy the custom binary into the Python wheel.** This creates a stale
  snapshot, transfers native-artifact ownership to pip, and requires reinstall
  after every client rebuild.
- **Restore a runtime flag or environment override.** This makes the selected
  binary invocation-dependent and can silently test the wrong client.
- **Write generated configuration into the source package.** Shared editable
  checkouts would make environments overwrite one another.
- **Emit metadata for the standard binding.** Absence already has one
  unambiguous meaning, so an explicit two-mode schema duplicates the default
  without solving a user problem.
- **Let pip build the C++ client.** Native CMake/toolchain failures do not belong
  in the Python backend, and the branch deliberately removed Python-side client
  autobuilding.
- **Execute the client as package validation.** GPU and dynamic-runtime
  availability are not guaranteed in build environments, and no stable client
  ABI/version handshake exists.
- **Remove the synthetic ROCm stage.** The copy is inexpensive, avoids writes to
  a system ROCm root, and continuously exercises the release filesystem
  contract.
- **Add another shell install script.** The existing Invoke tasks already own
  the developer build orchestration and are the thinner reusable layer over
  CMake and pip.

## Non-Goals

- Changing generated kernel semantics or library logic formats.
- Changing hipBLASLt's C++ host ABI.
- Renaming the C++ Tensile runtime or generated artifacts.
- Bundling the ROCm compiler/runtime in Python wheels.
- Preparing, building, publishing, or choosing an ABI/version policy for
  rocisa.
- Publishing ext-op generation as a supported user API.
- Providing legacy `Tensile` Python imports.

## Revisit Triggers

Revisit these decisions only when one of the following occurs:

- ROCm adopts a shared, cross-component Python artifact manifest.
- rocisa publishes its own packaging/ABI decision that changes how consumers
  declare or provision the dependency.
- A supported downstream requires a public ext-op interface.
- The ROCm wheel index cannot express exact local-version dependencies.
- Compatibility telemetry justifies changing the documented removal release.
