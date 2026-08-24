# Supporting Python ROCm venvs in `invoke install`

## Status

This is an implementation design for making the TensileLite source-development
workflow support an active Python ROCm environment. It describes the required
behavior and tests; it does not mean that `tasks.py` implements this behavior
yet.

## Scope

`invoke install` is intentionally different from a normal TensileLite wheel
runtime or code-generation invocation. It is the one-command source-development
workflow that:

1. installs development requirements into the active Python interpreter;
2. builds and editable-installs `rocisa`;
3. builds `tensilelite-client` and its native dependencies;
4. installs TensileLite editably; and
5. binds that exact client to the editable installation.

The normal Python ROCm adapter must remain client-free and root-free for import,
compatibility validation, and generator commands. This proposal adds the
separate native-development capability needed by `invoke install`; it does not
make `rocm[devel]` a runtime dependency of the TensileLite wheel.

## Why a Python ROCm venv needs `rocm[devel]` here

The ROCm core wheel already provides the public console-script trampolines used
by the Python runtime, including `amdclang`, `amdclang++`,
`clang-offload-bundler`, `offload-arch`, and `hipconfig`. Those tools are enough
for the supported Python SDK runtime and generator contract.

`invoke install` also compiles native C++ code. Its CMake projects need a
coherent ROCm development prefix containing headers, CMake/package-config
metadata, and the associated imported or static libraries:

- `rocisa` runs `find_package(hip REQUIRED)` and links `hip::host`.
- `tensilelite-client` links `hip::host`, `hip::device`, `rocisa`, the
  TensileLite host library, and, on Linux, `amd_smi`.
- StinkyTofu requires `amd_comgr` through `find_package(amd_comgr CONFIG)`.

`rocm-sdk-devel` provides that `ROCM_PATH`-style CMake development tree. Its
public `rocm-sdk path --root` and `rocm-sdk path --cmake` interfaces therefore
require a matching devel payload. A normal installed-wheel environment such as
`rocm[libraries,device-*]` remains valid for runtime and generator work, but a
full native source-development setup needs a matching
`rocm[libraries,devel,device-*]` environment.

## Existing contracts to preserve

The current branch already defines two mutually exclusive runtime models:

| Concern | Python SDK packages | Conventional ROCm prefix |
| --- | --- | --- |
| ROCm identity | Exact `rocm_sdk_core.__version__` | `<root>/.info/version` base version |
| Tool lookup | Active interpreter scripts, then user scripts | `<root>/bin` and `<root>/lib/llvm/bin` |
| Root discovery | None for runtime | `ROCM_PATH`, `/opt/rocm`, then `hipconfig` |
| Fallback rule | Never borrow a prefix tool or ambient `PATH` tool | Never inspect Python package payloads |

This distinction matters for nightly and RC builds: a Python package can carry
the full identity `10.1.0aYYYYMMDD`, while its development root's
`.info/version` contains only `10.1.0`. The full Python identity must be used
for both the editable TensileLite distribution and the client version. The root
base version is useful only to confirm that the development root belongs to the
same ROCm compatibility line.

The existing StinkyTofu task is the closest source-build precedent. It obtains
the materialized root with `rocm-sdk path --root` and CMake package locations
with `rocm-sdk path --cmake`. TensileLite should reuse those public discovery
interfaces, but not its `ROCM_PATH`-first or ambient-tool precedence: an active
Python SDK must remain the selected model.

## Target selection behavior

Add one private bootstrap-safe seam in `tasks.py`:

```python
_resolve_rocm_build_environment(rocm_path) -> RocmBuildEnvironment
```

`RocmBuildEnvironment` hides the two layouts behind a single interface. It
contains the full distribution identity, native CMake root, CMake prefix,
absolute C/C++ compiler paths, executable search paths, and a diagnostic
source. It should also provide the command environment and CMake arguments used
by its callers so the layout knowledge does not leak into every task.

The resolver follows this order:

1. If the active interpreter imports `rocm_sdk_core`, select the Python SDK
   build model.
2. Otherwise select the existing conventional-prefix model:
   `--rocm-path`, nonempty `ROCM_PATH`, then `/opt/rocm`.

An active Python SDK ignores an ambient `ROCM_PATH`. If `--rocm-path` is passed
while a Python SDK is active, it must either resolve to the SDK's own discovered
development root or fail clearly. It must never combine a Python SDK version or
compiler trampoline with an unrelated conventional root.

### Python SDK build model

The Python SDK implementation must:

1. Read the full identity from `rocm_sdk_core.__version__`.
2. Resolve `amdclang`, `amdclang++`, `offload-arch`, and `rocm-sdk` only from
   the active interpreter's normal and user script directories. Do not use
   `shutil.which()` for these selected tools.
3. Invoke that exact `rocm-sdk` script for `path --root` and `path --cmake`.
4. Verify that the root's `.info/version` has the same `A.B.C` base as the
   Python core package. Keep the full package identity as the wheel/client
   identity.
5. Fail before any native build if the root or CMake prefix is unavailable.
   The error should explain that `invoke install` needs matching
   `rocm[libraries,devel,device-<target>]` in the active environment.

### Conventional-prefix build model

The conventional-prefix model preserves the existing selection order:

1. Select the root from `--rocm-path`, `ROCM_PATH`, or `/opt/rocm`.
2. Read the distribution identity from `<root>/.info/version`.
3. Resolve compilers from the selected root.

The selected conventional root must also be authoritative at every native CMake
configure. Pass it explicitly as `-DROCM_PATH=<root>` and pass its resolved CMake
search prefix as `-DCMAKE_PREFIX_PATH=<prefix>`, in addition to the selected
compiler paths. Compiler-only overrides are insufficient: an inherited preset
prefix can otherwise resolve packages and build identity from `/opt/rocm` or an
unrelated ambient SDK.

## Required task changes

Refactor these task paths to accept one `RocmBuildEnvironment` rather than a
loosely related `rocm_path` string:

- `rocisa` and `_pip_install_rocisa`;
- `_build_and_install_stinkytofu`;
- `build_client`;
- `install`; and
- `build_coverage`.

`install` resolves the environment once, then calls private raw helpers so that
rocisa, the client, editable package installation, and client binding all use
the same selection.

For a Python SDK source build, every native CMake invocation must receive:

```text
-DROCM_PATH=<rocm-sdk path --root>
-DCMAKE_PREFIX_PATH=<rocm-sdk path --cmake>
-DCMAKE_C_COMPILER=<active Python scripts>/amdclang
-DCMAKE_CXX_COMPILER=<active Python scripts>/amdclang++
```

Set `ROCmCMakeBuildTools_DIR` from the resolved development root when the
corresponding CMake package is present. Pass the same environment only to build
subprocesses; do not permanently mutate the user's shell environment or make a
runtime import rely on `ROCM_PATH`.

`_build_and_install_stinkytofu` must stop preferring arbitrary `PATH` results.
Its compilers come from the resolved build environment. `_pip_install_rocisa`
must run `sys.executable -m pip`, so rocisa lands in the same interpreter whose
ROCm SDK was selected. Add a guard that rejects a system-wide `invoke` process
when `VIRTUAL_ENV` points at a different interpreter.

GPU auto-detection must also use the selected environment. Prefer its
`offload-arch` trampoline, then its `amdgpu-arch` compatibility fallback, rather
than unqualified `rocm_agent_enumerator` lookup.

## Required CMake companion change

Updating `tasks.py` alone is insufficient for nightly Python ROCm builds.

The top-level hipBLASLt CMake configuration currently derives
`HIPBLASLT_BUILD_ROCM_VERSION` from `<root>/.info/version` for standalone
builds. That value forms `TENSILELITE_DISTRIBUTION_VERSION`, which is embedded
in `tensilelite-client --version`.

Add a narrow standalone CMake cache input:

```text
TENSILELITE_ROCM_VERSION=<full Python SDK publication identity>
```

`hipblaslt_resolve_build_rocm_version()` must honor this input before falling
back to the root marker. The Python SDK task passes the full
`rocm_sdk_core.__version__` value. TheRock continues to own its identity through
`THEROCK_PACKAGE_VERSION`; the new standalone override must not weaken that
contract.

This prevents the following incorrect mixed identity:

```text
Python SDK / editable wheel:  <component>+rocm10.1.0aYYYYMMDD
CMake-built client:           <component>+rocm10.1.0
```

The correct fix is to preserve the full identity through CMake. Do not suppress
or weaken the existing client-version validation to make this mismatch pass.

## Tests

Extend `tensilelite/Tests/unit/source_only/test_install_task.py` with mocked
Python SDK and prefix environments covering:

1. Python SDK selection ahead of a conflicting ambient `ROCM_PATH`.
2. Full nightly identity propagation to editable pip, CMake, and client binding.
3. Exact venv compiler paths, SDK development root, and CMake prefix passed to
   StinkyTofu, rocisa, and client CMake commands.
4. Missing devel root/CMake prefix failing without a fallback to `/opt/rocm`.
5. A Python core/devel base-version mismatch failing clearly.
6. Unchanged conventional-prefix behavior when Python ROCm is absent.
7. An empty `ROCM_PATH` failing in conventional-prefix mode.
8. An explicit conventional-prefix `--rocm-path` winning over a conflicting
   ambient `ROCM_PATH`, with the selected root, CMake prefix, compilers, package
   identity, editable-install environment, and binding environment agreeing.

Add a command-capture test for `build_client` that proves it overrides the
inherited `/opt/rocm` CMake preset with the selected root and CMake prefix.
Extend the CMake source-only identity tests to prove that the explicit Python
SDK identity wins for standalone builds while TheRock still requires and uses
`THEROCK_PACKAGE_VERSION`.

Finally, add two integration lanes:

1. A fresh matching `rocm[libraries,devel,device-<target>]` venv with a
   deliberately conflicting `ROCM_PATH`. Run `invoke install --gpu-targets ...`
   and verify that `tensilelite-client --version` equals the installed
   `tensilelite` distribution version.
2. A fresh `rocm[libraries,device-<target>]` venv without devel. Verify that
   `invoke install` fails early with the actionable devel-payload diagnostic.

Mocked command-capture tests are the required regression gate for root
consistency. A one-root smoke test confirms only its happy path; a conventional-
prefix end-to-end test that proves root precedence needs two independently valid
ROCm development prefixes.

## Non-goals

- Do not make `rocm[devel]` a dependency of the normal TensileLite runtime,
  import path, or generator commands.
- Do not import `tensilelite._rocm` from `tasks.py` during source bootstrap.
- Do not borrow an individual tool from `/opt/rocm` or ambient `PATH` after a
  Python SDK has been selected.
- Do not downgrade a full Python ROCm publication identity to a root base
  version, and do not relax client-version validation.
