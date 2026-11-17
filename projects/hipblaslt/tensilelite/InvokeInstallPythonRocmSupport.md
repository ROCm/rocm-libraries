# Supporting Python ROCm venvs in `invoke install`

## Status (2026-11-17)

Native build-root selection is implemented in #15579. Complete native-build
support for the Python ROCm SDK remains follow-up work. The installed Python
SDK runtime already has a separate implementation in #11023/#11024, enabled
for the SDK in #11032.

The current native-build contract is:

- `--rocm-path` selects a native build installation. Otherwise use nonempty
  `ROCM_PATH` (set but empty is an error), then the existing
  `rocm-sdk path --root` discovery, then `/opt/rocm`.
- `RocmBuildEnvironment` supplies the selected root, compiler paths, CMake
  inputs and subprocess environment. The root overrides the client preset's
  `/opt/rocm`. rocisa lets scikit-build-core manage `CMAKE_PREFIX_PATH`;
  its CMake adds the selected `ROCM_PATH` to package search.
- Each public native task reports `Using ROCm at <root> (from <source>)`.
  `install` resolves and reports once, then shares that environment with its
  rocisa, client and optional StinkyTofu native build helpers.
- `build-client` runs both selected compilers with `--version`, with a
  five-second timeout, for every root source. Missing or unusable compilers
  fail before that task cleans its build directory or invokes CMake.
- Automatic client/coverage targets and StinkyTofu provisioning use the
  selected root's GPU tools and subprocess `ROCM_PATH`. They do not borrow
  tools from another root or ambient `PATH` if selected tools are absent.
  Explicit targets and forced StinkyTofu choices remain authoritative;
  standalone forced-off provisioning needs no ROCm installation.
- GPU detection preserves the existing `amdgpu-arch` enumeration plus
  `rocminfo` stepping restoration. An `offload-arch`-only replacement loses
  the `gfx1250-strict` revision on toolchains that report only the base ISA.

`install --rocm-path` does not save a runtime ROCm root. It warns once when a
nonempty exported `ROCM_PATH` resolves to a different directory; relative and
symlink aliases of the same root stay quiet. The parent environment remains
unchanged. Editable TensileLite installation and client binding receive no
ROCm environment override or identity. Binding stores the client path only.

## Source development and runtime are separate selections

`invoke install` installs development requirements in its active interpreter,
builds/editable-installs rocisa, builds the native client, editable-installs
TensileLite, and binds that exact client. Native compilation needs a coherent
development prefix. Later Python processes select their runtime independently.

| Concern | Python SDK runtime | Conventional-prefix runtime | Invoke native build |
| --- | --- | --- | --- |
| Selection | `rocm_sdk_core` in active interpreter first | Used when active SDK is absent | Flag, environment, SDK CLI root, default |
| Identity | Full `rocm_sdk_core.__version__` | Selected prefix metadata | Client records its build ROCm separately |
| Tool lookup | Interpreter and user script locations | Selected prefix tool locations | Compilers and GPU tools under selected build root |
| Development root | Not required | Conventional installation | Required for native CMake dependencies |

An active-interpreter SDK runtime wins even with a conflicting conventional
`ROCM_PATH`. Missing SDK tools or an incompatible SDK do not authorize runtime
to borrow another installation. That runtime rule does not replace Invoke's
explicit native-build selection order. Raw CMake builds receive their own build
inputs; Invoke passes its chosen native root to CMake explicitly.

The downstream client check validates the client's recorded build ROCm against
the selected process runtime. A different directory alone does not establish
incompatibility. Source-development compatibility continues to follow Q134;
no persistent root or new compatibility override is added here.

## Identity requirements superseded by Q121/Q134

[PythonBuildGrillingDecisions.md](PythonBuildGrillingDecisions.md), Q121 and
Q134, and the [packaging identity plan](docs/PackagingIdentityPlan.md) are the
current identity contract. They supersede this proposal's earlier requirements
for ROCm-stamped editable wheel/client versions, identity propagation to pip and
binding, and a new standalone `TENSILELITE_ROCM_VERSION` CMake override.

- A standalone source wheel/editable distribution is `+g<sha>[.dirty]` and
  makes no ROCm distribution-identity claim.
- Wheel/client pairing uses source revision. The client separately reports
  `--build-rocm`; its distribution version need not equal the Python package's.
- TheRock passes its explicit publication identity for canonical wheel/native
  metadata through the existing frontend. No replacement override is added.
- Editable package installation and binding use the caller's environment.
  Only the native build subprocesses receive the selected build environment.

A future SDK build adapter may use the full SDK publication identity to verify
and report its inputs. It must not reintroduce source-wheel ROCm stamping or
require Python/client distribution-version equality.

## Why a Python ROCm source build needs development payloads

The core package supplies interpreter-local compiler/tool trampolines for the
supported runtime and generator contract. Native CMake projects also need
headers, CMake/pkg-config metadata and associated libraries:

- rocisa finds HIP and links `hip::host`.
- The client links HIP, rocisa, TensileLite host libraries and, on Linux,
  `amd_smi`.
- StinkyTofu finds `amd_comgr`.

`rocm-sdk-devel` supplies the development tree. Its public
`rocm-sdk path --root` and `rocm-sdk path --cmake` discovery requires a matching
devel payload. Runtime/generator use of `rocm[libraries,device-*]` remains
valid without devel. Native source development needs a matching
`rocm[libraries,devel,device-*]` environment.

## Follow-up: complete Python SDK native-build support

The existing root-discovery fallback is not a complete interpreter-local SDK
build adapter. The remaining work is:

1. Detect `rocm_sdk_core` and resolve public SDK scripts from the active
   interpreter's normal/user script locations. Use that exact `rocm-sdk`
   script for `path --root` and `path --cmake`.
2. Check that the core package and development prefix have matching `A.B.C`
   base versions. Fail early with an actionable matching-devel diagnostic
   when the root, CMake metadata or required native payloads are unavailable.
3. Supply the SDK's development root, CMake search prefix and compiler
   trampolines through the existing build-environment seam. Set
   `ROCmCMakeBuildTools_DIR` when the corresponding package is present.
4. Check that `VIRTUAL_ENV`, when set, matches Invoke's interpreter, and keep
   pip installs in that interpreter with `sys.executable -m pip`.
5. Add actual SDK native-build integration lanes in matching environments
   with and without devel, alongside focused command-capture regressions.

The old proposal put an active SDK ahead of `--rocm-path` and rejected roots
outside its development tree. That precedence is not the implemented contract.
A future adapter must reconcile SDK selection with the explicit build-only
flag before changing precedence; #15579 retains the selected current order.
Runtime continues to select the active-interpreter SDK independently.

Selected-build GPU detection is implemented now. Any future interpreter-script
adapter must preserve stepping restoration when its enumerator reports only
`gfx1250`; it cannot substitute `offload-arch` alone. Generic runtime detection
callers retain their existing behavior.

## Validation

The #15579 owner task suite passes **49 cases**. Tests cover all root sources,
original-source reporting, compiler failures before client cleanup/CMake,
root aliases and warning behavior, shared native environment, explicit targets,
forced provisioning choices and preserved parent environment. Real executable
fixtures cover distinct selected/ambient installations, SDK CLI discovery,
selected-root GPU results and stepping restoration, and missing selected tools
without borrowing ambient tools. Fifteen new regressions fail on the original
owner branch.

The two existing source-only GPU-target tests were adapted to selected-root
detection and compiler preflight, preserving feature-stripping assertions;
the combined owner task/target suite passes **51 cases**. Final stack source,
binding and runtime validation passes **565 cases**, with one optional actual
installed-artifact-root check skipped. This source-tree run does not cover the
separate installed-artifact binding fixture issue tracked for #15580.

Relevant existing GPU/tool/stepping cases pass **19 cases**. The broad stepping
module has 919 passes and four inherited StreamK fence expectation failures;
the original owner branch reproduces the identical failures. No tests were
weakened. Required pre-commit and separate Black/flake8 checks pass. These tests
are collected by the shared coverage-unit CI lane after publication.

This batch has no actual HIP/client build, SDK native integration run or target
GPU numerical result. Fixtures prove selection, environments and generated
commands. The final stack's runtime tests exercise synthetic active-SDK and
conventional installations; they do not prove complete SDK native builds.

The future integration lanes should verify source-revision pairing and the
client's build-ROCm compatibility, rather than wheel/client version equality.
Use independently valid roots for end-to-end precedence checks. Keep native
source-build integration separate from the existing no-devel SDK runtime smoke
lane and from tests of shipped hipBLASLt GPU binaries.
