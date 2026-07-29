<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Packaging Decisions

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
  custom kernels, `TensileLogic/known_bugs.yaml`, and the ductile defaults.
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
- The canonical commands are `tensilelite create-library`, `tensilelite logic`,
  and `tensilelite run`.
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

Released `tensilelite` wheels are supported only with the exact ROCm artifact
set for which they were built. ROCm publishes the wheel through its wheel
index and owns `tensilelite-client` under `ROCM_PATH`. A suitable `rocisa`
distribution is supplied independently.

Reasoning:

The generator depends on ROCm compiler tools and native components that pip
cannot provide. Treating the wheels as independently portable would create an
untestable compatibility matrix and obscure missing system dependencies.

Rejected alternatives:

- Standalone release wheels that vendor the client and native dependencies.
- Dual standalone and ROCm-coupled release channels.
- Installing wrappers directly into arbitrary system Python site-packages.

Consequences:

- `pip install` can install a wheel without ROCm, but first import fails with an
  actionable error.
- Clean test environments install wheels from a ROCm/local wheelhouse and point
  at a staged matching ROCm root.

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

1. Resolving `ROCM_PATH`, falling back to `/opt/rocm` on Unix or the configured
   ROCm SDK root on Windows.
2. Comparing the exact full value of `$ROCM_PATH/.info/version` with the wheel's
   `+rocmA.B.C` version segment.
3. Requiring fixed native artifact locations.
4. Successfully importing the independently installed `rocisa` package.

No artifact manifest, hashes, or component ABI handshake is added.

Reasoning:

ROCm package management is responsible for file integrity. Exact release
matching plus fixed layout and native loading is sufficient for this package
boundary and avoids hashing binaries during every process import.

Consequences:

- Errors must report the expected release, discovered release, resolved root,
  failed artifact, and remediation.
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
- `TensileLogic/known_bugs.yaml`
- ductile defaults

Tests, source launchers, Invoke tasks, CMake helpers not used at runtime, native
artifacts, rocisa source, and all build output are excluded.

Reasoning:

An explicit allowlist makes accidental package contents a test failure rather
than a growing public contract.

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
