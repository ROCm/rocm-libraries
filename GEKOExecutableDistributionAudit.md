# GEKO executable and ROCm distribution audit

Date: 2026-09-24 UTC

Source snapshots examined:

- `rocm-libraries`: `4ea2fb3b6cf549c6e5e737775a760189f13041e5`
- `TheRock`: `8329d38e66410ac2c5308aca4743a8b91eb52e5e`
- TheRock's `rocm-libraries` gitlink: `983d768c95086e69aa641578a3d28cf8398e9a0f`
- Installed ROCm: `/opt/rocm -> /opt/rocm-7.2.4`, Debian package `rocm` version `7.2.4.70204-93~24.04`, `hipblaslt` version `1.2.2.70204-93~24.04`

The two source repositories are not on one exact source snapshot: the checked-out
`rocm-libraries` branch and TheRock's pinned gitlink diverge after
`2e8f62c7e1e1fd05b354720f0a2c6d7d2c758e89`. The GEKO dependency analysis below
therefore uses the requested `rocm-libraries` checkout, while the release/package
analysis uses TheRock's current packaging definitions and notes the pin where it
changes the contents.

## Bottom line

GEKO is currently a **source-checkout tool**, not a utility that can run from a
normal installed ROCm prefix. All three main CLI modes require a hipBLASLt root
that contains both `tensilelite/` and `build/release/`; GEKO then uses fixed
source/build-tree paths rather than looking for equivalent installed programs.
This contract is explicit in
`projects/hipblaslt/utilities/geko/geko/paths.py:4-14,35-53,83-119` and in the
CLI's `require_built=True` call at
`projects/hipblaslt/utilities/geko/geko/cli.py:322-332`.

The decisive distribution result is:

1. **GEKO itself is not installed by hipBLASLt CMake and is not named anywhere in
   TheRock's packaging rules.** Its only install definition is the standalone
   Python console entry point in
   `projects/hipblaslt/utilities/geko/pyproject.toml:8-20`.
2. **`hipblaslt-bench` is built and staged, but TheRock classifies it as a test
   artifact.** Standard release tarballs exclude the entire `test` component.
   It is available only in separately generated `-tests` tarballs or the
   optional `amdrocm-blas-test` native package.
3. **Some TensileLite tooling is also staged only as test material, not as a
   supported GEKO runtime.** The paths differ from the fixed paths GEKO uses,
   `TensileMergeLibrary` is not explicitly preserved as an executable, and the
   installed tree does not contain the required checkout-style
   `build/release/device-library/MatchTable.yaml`.
4. **The locally installed ROCm 7.2.4 has none of GEKO's direct helper
   executables:** no `geko`, `hipblaslt-bench`, `Tensile`,
   `TensileCreateLibrary`, `TensileMergeLibrary`, or `tensilelite-client` was
   found anywhere under `/opt/rocm-7.2.4`.

Therefore neither the ordinary ROCm installation on this host nor TheRock's
standard runtime/SDK release is sufficient to run GEKO. A built hipBLASLt source
checkout is still required.

## What GEKO directly executes

The table covers subprocesses named directly by GEKO's current source. Data files
and the deeper Tensile toolchain are separated below.

| Executable | When it is needed | How GEKO locates it | Direct source evidence |
| --- | --- | --- | --- |
| `python3` | All use of the repo launcher and all source-tree Tensile launchers | `#!/usr/bin/env python3`; GEKO requires Python 3.10+ | `projects/hipblaslt/utilities/geko/bin/geko:1-2`; `projects/hipblaslt/utilities/geko/pyproject.toml:8-20`; `projects/hipblaslt/tensilelite/Tensile/bin/Tensile:1` |
| `hipblaslt-bench` | `--bench`; baseline and candidate measurement in `--search`; initial and post-optimization measurement in `--tune` (unless a valid cached result bypasses a particular call) | Fixed path `<hipblaslt>/build/release/clients/hipblaslt-bench`; there is no `PATH` or installed-prefix fallback | `projects/hipblaslt/utilities/geko/geko/bench/bench.py:123-145`; CLI routing at `projects/hipblaslt/utilities/geko/geko/cli.py:370-420` |
| `Tensile` | The optimization phase of `--tune` | Fixed path `<hipblaslt>/tensilelite/Tensile/bin/Tensile` | `projects/hipblaslt/utilities/geko/geko/optim/optim.py:260-261,306-324` |
| `tensilelite-client` | The benchmark client used inside a fresh `--tune` optimization | Fixed build output `<client_build_dir>/tensilelite/client/tensilelite-client`, passed to `Tensile` with `--prebuilt-client` | `projects/hipblaslt/utilities/geko/geko/utils.py:104-141`; `projects/hipblaslt/utilities/geko/geko/optim/optim.py:308-317` |
| `TensileCreateLibrary` | Fresh tune/search analysis; also `--bench --custom-lib-src` | Fixed path `<hipblaslt>/tensilelite/Tensile/bin/TensileCreateLibrary` | `projects/hipblaslt/utilities/geko/geko/library/operations.py:329-367`; conditional bench use at `projects/hipblaslt/utilities/geko/geko/pipeline.py:117-145`; analysis use at `projects/hipblaslt/utilities/geko/geko/bench/bench.py:423-438` |
| `TensileMergeLibrary` | Public `geko.library.operations.merge()` helper and manual integration, but not the normal `--tune`, `--search`, or `--bench` dispatch path | Fixed path `<hipblaslt>/tensilelite/Tensile/bin/TensileMergeLibrary` | `projects/hipblaslt/utilities/geko/geko/library/operations.py:294-326`; the main tune path instead uses in-process `merge_solutions()` at `projects/hipblaslt/utilities/geko/geko/pipeline.py:569-578` |
| `invoke` | Only when `tensilelite-client` is absent or stale and must be rebuilt | Resolved by executable name from `PATH`; GEKO checks that the Python `invoke` module imports, then runs `invoke build-client ...` | `projects/hipblaslt/utilities/geko/geko/utils.py:111-141` |
| `taskkill` | Windows only, when cancelling a running tuning subprocess tree | Resolved by name from `PATH` | `projects/hipblaslt/utilities/geko/geko/concurrency/utils.py:44-56,108-115` |

### Mode-by-mode minimums

- `geko --bench`: `python3` plus `hipblaslt-bench`. Adding
  `--custom-lib-src` also needs `TensileCreateLibrary` and its compiler
  toolchain. A prebuilt `--custom-lib-dir` avoids that compilation step.
- `geko --search`: `python3`, `hipblaslt-bench`, and
  `TensileCreateLibrary` plus its compiler toolchain. Search reads the build-tree
  MatchTable when extracting solution indices
  (`projects/hipblaslt/utilities/geko/geko/pipeline.py:229-249,275-325`).
- `geko --tune`: all of `hipblaslt-bench`, `Tensile`,
  `tensilelite-client`, and `TensileCreateLibrary`, plus the client-build and
  Tensile compiler tools below. The call sequence is visible at
  `projects/hipblaslt/utilities/geko/geko/pipeline.py:418-446,557-623`.
- Configure-only is the narrow exception. With `keep_thr=0`, workload parsing
  can avoid benchmarking, and the main CLI's configure phase emits YAML without
  shell scripts. The relevant branches are
  `projects/hipblaslt/utilities/geko/geko/bench/log.py:444-477` and
  `projects/hipblaslt/utilities/geko/geko/optim/optim.py:183-190`.

## Immediate build/toolchain executables

When GEKO has to build the TensileLite client, it runs
`invoke build-client --build-dir ...`. The invoked task uses `cmake` for both
configuration and build (`projects/hipblaslt/tensilelite/tasks.py:182-267`). Its
`tensilelite` CMake preset points at `/opt/rocm/bin/amdclang` and
`/opt/rocm/bin/amdclang++`
(`projects/hipblaslt/CMakePresets.json:5-31`). With the current preset no generator
is named, so a CMake-selected native builder is also required; on this host the
available default is GNU `make`.

Because GEKO does not pass a target to `build-client`, automatic client target
detection uses `rocm_agent_enumerator`
(`projects/hipblaslt/tensilelite/tasks.py:210-214` and
`projects/hipblaslt/tensilelite/Tensile/GpuRevisionTarget.py:30-52`).

Once `Tensile` or `TensileCreateLibrary` is running, its Linux defaults are:

- `amdclang++` as C++/HIP compiler and assembler;
- `amdclang` as C compiler;
- `clang-offload-bundler` as the code-object bundler;
- `amdgpu-arch` as the normal Linux device enumerator
  (`rocm_agent_enumerator` on RHEL 8 or FFM environments);
- `hipconfig` for ROCm/toolchain version discovery.

Those defaults and search paths are defined in
`projects/hipblaslt/tensilelite/Tensile/Toolchain/Validators.py:93-123`; the
drivers validate them at
`projects/hipblaslt/tensilelite/Tensile/Tensile.py:667-687` and
`projects/hipblaslt/tensilelite/Tensile/TensileCreateLibrary/Run.py:1103-1147`.
The compiler/assembler and bundler are actual subprocesses, not merely probes
(`projects/hipblaslt/tensilelite/Tensile/Toolchain/Component.py:190-212,239-281,284-325`).

Tensile also generates and launches a `/bin/bash` client script
(`projects/hipblaslt/tensilelite/Tensile/ClientWriter.py:349-362`). Library-logic
generation can run `rocminfo | grep Compute` unless `CU` is supplied in the
environment (`projects/hipblaslt/tensilelite/Tensile/LibraryIO.py:927-945`).

`amd-smi`, `hipcc`, `ccache`, `pip`, and `rocm-sdk` are conditional rather than
baseline requirements:

- `amd-smi` is optional clock/frequency handling; missing it is explicitly
  non-fatal (`projects/hipblaslt/tensilelite/Tensile/Common/GlobalParameters.py:911-925`),
  and GEKO-generated configurations leave `PinClocks` at the Tensile default
  `False`.
- `hipcc` is a best-effort version/revision probe
  (`projects/hipblaslt/tensilelite/Tensile/Common/GlobalParameters.py:942-967` and
  `projects/hipblaslt/tensilelite/Tensile/GpuRevisionTarget.py:68-119`).
- `ccache` is used only when found; `pip` and optionally `rocm-sdk` are involved
  only in the conditional editable-rocisa rebuild path
  (`projects/hipblaslt/tensilelite/tasks.py:32-51,95-126,228-267`).

There is also a standalone configuration-generator surface that defaults to
writing runnable shell scripts. Those generated scripts need `/bin/bash`,
`tee`, `mkdir`, `cp`, `mv`, and `rm`, plus `Tensile`
(`projects/hipblaslt/utilities/geko/geko/config_generator/output_writer.py:255-307`).
The main `geko --tune` path does not use these scripts.

Finally, the normal GEKO wheel is incomplete as a fresh tuning environment:
`invoke` is listed only in the development dependency group, not in runtime
`requirements.txt` (`projects/hipblaslt/utilities/geko/pyproject.toml:22-32` and
`projects/hipblaslt/utilities/geko/requirements.txt:1-6`). A plain
`pip install .` can therefore install the `geko` console script without the
executable it needs to build a missing client.

## Non-executable artifacts that are equally mandatory

The executable inventory alone is insufficient:

- Every CLI mode currently insists on a checkout-shaped hipBLASLt root with
  `tensilelite/` and `build/release/`, not merely installed hipBLASLt libraries
  (`geko/paths.py`, cited above).
- Search and tune require
  `<hipblaslt>/build/release/device-library/MatchTable.yaml`
  (`projects/hipblaslt/utilities/geko/geko/pipeline.py:229,616-621`).
- Benchmarking needs the hipBLASLt runtime library and its device-library data.
  An installed ROCm tree normally has those runtime pieces, but that does not
  compensate for the missing build-tree path and MatchTable.
- The `Tensile`, create-library, and merge-library launchers also depend on the
  adjacent Tensile Python source/package tree; copying only the script files is
  not sufficient.

## What TheRock builds and publishes

### Artifact construction

TheRock points its hipBLASLt subproject at the pinned `rocm-libraries` checkout,
enables hipBLASLt clients by their upstream default, maps both
`TENSILELITE_BUILD_TESTING` and
`HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` to
`THEROCK_BUILD_TESTING`, and declares the hipBLASLt runtime/build dependencies
in `/home/alvasile/TheRock/math-libs/BLAS/CMakeLists.txt:130-201`.
The release artifact configure helper explicitly passes `-DBUILD_TESTING=ON`
(`/home/alvasile/TheRock/build_tools/github_actions/build_configure.py:88-112`),
so the test payload is built before later packaging selects or excludes it.

The BLAS artifact is split into `dbg`, `dev`, `doc`, `lib`, `run`, and `test`
components (`/home/alvasile/TheRock/math-libs/BLAS/CMakeLists.txt:570-583`). Its
descriptor puts these relevant paths in the **test** component:

- `bin/hipblaslt-bench*`;
- `share/hipblaslt/tensilelite/**`;
- `libexec/hipblaslt/tensilelite/**`.

See `/home/alvasile/TheRock/math-libs/BLAS/artifact-blas.toml:66-108`.

The artifact machinery flattens selected component trees into
`build/dist/rocm` (`/home/alvasile/TheRock/cmake/therock_artifacts.cmake:13-22,99-136,203-235`).

### Tarball releases

The standard TheRock tarball explicitly excludes component `test`; a second
archive with a `-tests` suffix is created only when test tarballs are enabled
(`/home/alvasile/TheRock/build_tools/packaging/archives/build_tarballs.py:20-30,77-78,399-443`).
The reusable tarball workflow defaults `include_test_tarballs` to true, so the
release workflow currently produces both variants
(`/home/alvasile/TheRock/.github/workflows/multi_arch_build_tarballs.yml:37-40,68-70,135-150`;
`/home/alvasile/TheRock/.github/workflows/multi_arch_release_linux.yml:107-120`).

Result: `hipblaslt-bench` is absent from the normal
`therock-dist-linux-...tar.gz`, but is intended to be present in the separately
named `therock-dist-linux-...-tests-...tar.gz`. GEKO is absent from both because
no stage/install rule places it in any artifact.

### Native DEB/RPM releases

The normal `amdrocm-blas` package selects only hipBLASLt `lib`, `run`, and `doc`
components (`/home/alvasile/TheRock/build_tools/packaging/linux/package.json:526-609`).
The separate, optional `amdrocm-blas-test` package selects the hipBLASLt `test`
component (`package.json:705-777`), so that is where `hipblaslt-bench` and the
TensileLite test payload are assigned.

Neither the ordinary `amdrocm` runtime metapackage nor `amdrocm-core-sdk`
depends on `amdrocm-blas-test`
(`/home/alvasile/TheRock/build_tools/packaging/linux/package.json:3487-3569,3873-3899`).
The native release workflow does build all package definitions from fetched
artifacts (`/home/alvasile/TheRock/.github/workflows/multi_arch_build_native_linux_packages.yml:130-156`),
but users must explicitly install the test package.

TheRock's current kpack-split Python SDK packaging is not an alternate delivery
path for these tools: it explicitly excludes the `test` component while
constructing the devel wheel
(`/home/alvasile/TheRock/build_tools/build_python_packages.py:421-444`).

### What the optional test payload still does not solve

The checked-out `rocm-libraries` CMake source installs `Tensile` and
`TensileCreateLibrary` into `share/hipblaslt/tensilelite/Tensile/bin` and a
prebuilt client into `libexec/hipblaslt/tensilelite`, all behind the test-artifact
option (`projects/hipblaslt/CMakeLists.txt:723-767,834-839`). It does not
explicitly preserve `TensileMergeLibrary`, and there is no GEKO install rule.

TheRock currently pins a divergent `rocm-libraries` commit (`983d768c9508`). At
that pin, the test staging code installs/extracts ROCm-versioned TensileLite
wheels and the prebuilt client rather than installing the legacy
`Tensile/bin/*` launcher tree. Its GEKO invokes the active interpreter with
`-m tensilelite`; merely extracting that package under `share/` does not create
a `PATH` entry or make it importable outside an explicitly prepared
`PYTHONPATH`. This was verified with:

```text
git -C /home/alvasile/TheRock ls-tree HEAD rocm-libraries
160000 commit 983d768c95086e69aa641578a3d28cf8398e9a0f rocm-libraries
```

In either source snapshot, the installed layout cannot satisfy current GEKO's
fixed paths:

| GEKO asks for | TheRock test payload uses |
| --- | --- |
| `<hip>/build/release/clients/hipblaslt-bench` | `<prefix>/bin/hipblaslt-bench` |
| `<hip>/tensilelite/Tensile/bin/Tensile` | Current checkout: `<prefix>/share/hipblaslt/tensilelite/Tensile/bin/Tensile`; pinned TheRock source changed GEKO to `python -m tensilelite`, while its test payload extracts the module under `<prefix>/share/hipblaslt/tensilelite` |
| `<client_build>/tensilelite/client/tensilelite-client` | `<prefix>/libexec/hipblaslt/tensilelite/tensilelite-client` |
| `<hip>/build/release/device-library/MatchTable.yaml` | No matching installed build-tree artifact |

So the optional test package/tarball is useful raw material, but it is not a
working installed GEKO distribution contract.

## What is actually installed on this host

The host's active ROCm symlink resolves to `/opt/rocm-7.2.4`. Package-manager
evidence:

```text
hipblaslt      1.2.2.70204-93~24.04   install ok installed
hipblaslt-dev  1.2.2.70204-93~24.04   install ok installed
rocm           7.2.4.70204-93~24.04   install ok installed
```

The configured apt repository offers `hipblaslt`, `hipblaslt-dev`, and their
versioned/rpath variants; it does not list a `hipblaslt-clients` or GEKO package.

An exact-name scan under `/opt/rocm-7.2.4` found zero copies of each direct GEKO
payload:

```text
geko                     0
hipblaslt-bench          0
Tensile                  0
TensileCreateLibrary     0
TensileMergeLibrary      0
tensilelite-client       0
```

The runtime installation does contain `libhipblaslt.so.1` and 3,000
`TensileLibrary*` files under `/opt/rocm-7.2.4/lib/hipblaslt/library`, so this is
not a missing hipBLASLt runtime installation. It is specifically missing the
development/tuning executables and build-tree metadata GEKO expects.

The ROCm compiler/toolchain side is present:

```text
amdclang                 /opt/rocm-7.2.4/bin/amdclang
amdclang++               /opt/rocm-7.2.4/bin/amdclang++
clang-offload-bundler    /opt/rocm-7.2.4/lib/llvm/bin/clang-offload-bundler
amdgpu-arch              /opt/rocm-7.2.4/lib/llvm/bin/amdgpu-arch
rocm_agent_enumerator    /opt/rocm-7.2.4/bin/rocm_agent_enumerator
hipconfig                /opt/rocm-7.2.4/bin/hipconfig
hipcc                    /opt/rocm-7.2.4/bin/hipcc
amd-smi                  /opt/rocm-7.2.4/bin/amd-smi
```

`cmake`, `make`, and `ninja` are installed as host tools. `invoke` exists only
in `/home/alvasile/venv/bin`, not under `/opt/rocm`; `tensilelite
5.0.0+rocm7.2.4` is also installed only in that developer virtual environment.
There is no installed `geko` command or Python package. The source checkout has
a cached `tensilelite-client` at
`projects/hipblaslt/tensilelite/build_tmp/tensilelite/client/tensilelite-client`,
but its required `projects/hipblaslt/build/release/clients/hipblaslt-bench` and
`projects/hipblaslt/build/release/device-library/MatchTable.yaml` are absent.

There is also an older developer-created TheRock Python SDK under
`/home/alvasile/sim/venv`. Its `rocm-sdk-devel` payload contains
`_rocm_sdk_devel/bin/hipblaslt-bench` and the compiler tools, but it does not
contain `tensilelite-client` or `geko`, and `hipblaslt-bench` is not exposed as
`/home/alvasile/sim/venv/bin/hipblaslt-bench`. That environment therefore does
not provide a working installed GEKO contract either. Its separately installed
legacy `Tensile` entry point comes from an editable source checkout, not from
the ROCm SDK payload.

## Practical verdict

| Question | Verdict |
| --- | --- |
| Can GEKO run from the installed ROCm 7.2.4 tree on this host? | **No.** The direct frontend/client executables and MatchTable are absent, and the installed tree is not the checkout layout GEKO accepts. |
| Does TheRock's ordinary runtime or SDK release make GEKO usable? | **No.** It does not ship GEKO, and standard tarballs/packages omit the test component containing `hipblaslt-bench`. |
| Does TheRock ship any of the needed pieces somewhere? | **Yes, partially.** `hipblaslt-bench` and the TensileLite test/client payload are assigned to optional test archives/packages; ROCm compiler tools are normal runtime/SDK content. |
| Would installing `amdrocm-blas-test` or extracting a `-tests` tarball be enough? | **No.** GEKO itself, its fixed checkout paths, and the build-tree MatchTable contract remain unresolved; `TensileMergeLibrary` is also not deliberately shipped as an executable. |
| What works today? | Build hipBLASLt with clients for the target architecture and run GEKO against that same source checkout, as the GEKO README instructs at `projects/hipblaslt/utilities/geko/README.md:226-280`. |

## Reproducible inspection commands

```bash
git -C /home/alvasile/rocm-libraries rev-parse HEAD
git -C /home/alvasile/TheRock rev-parse HEAD
git -C /home/alvasile/TheRock ls-tree HEAD rocm-libraries
readlink -f /opt/rocm
dpkg-query -W -f='${Package}\t${Version}\t${Status}\n' hipblaslt hipblaslt-dev rocm
apt-cache search '^hipblaslt|hipblaslt'
find /opt/rocm-7.2.4 -xdev \( -type f -o -type l \) \
  \( -name geko -o -name hipblaslt-bench -o -name Tensile \
     -o -name TensileCreateLibrary -o -name TensileMergeLibrary \
     -o -name tensilelite-client \) -print
```
