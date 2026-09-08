# TheRock ROCm Python Wheels: Build, Contents, and Intended Roles

## Scope and source snapshot

This guide describes the ROCm SDK packages published by TheRock—not the
separate PyTorch or JAX wheels that consume them. It was researched against
TheRock at commit
[`6339d5fc`](https://github.com/ROCm/TheRock/tree/6339d5fc2e8f924a7b3072a9c34623271c9626f7)
(checked out on 2026-08-12), current first-party release and packaging
documentation, and the linked GitHub issue/PR history. Source links to code are
pinned to that commit so the claims remain auditable if `main` changes.

The important current distinction is between **multi-architecture,
kpack-split** releases and the older **legacy per-family** releases. The current
release documentation says multi-arch builds compile all supported GPU targets
together, then split GPU-specific kernel packs from host code; the originating
tracking issue identifies build scalability, binary size, selective deployment,
and compression as the motivation. [Release guide](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/RELEASES.md#L46-L69)
[issue #3323](https://github.com/ROCm/TheRock/issues/3323)
[PR #6278](https://github.com/ROCm/TheRock/pull/6278)

The legacy model remains useful context, but should not be used to infer the
current wheel contents: it put host and device payload into target-family
`rocm-sdk-libraries-{family}` wheels, while kpack-split releases use one
architecture-neutral libraries wheel plus one device wheel per GPU ISA.
[Packaging design](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L45-L68)
[PR #4308](https://github.com/ROCm/TheRock/pull/4308)

## Mental model

TheRock first produces staged **artifacts**, then repackages selected artifact
components into Python distributions. Artifact components have distinct
meanings: `lib` is runtime library payload, `run` is runtime tools, `dev` is
headers/static or import libraries/CMake and pkg-config metadata/build tools,
and `test` is test-only content. [Artifact component definitions](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/development/artifacts.md#L182-L201)

The high-level relationship is:

```text
project builds and stage directories
              |
              v
  TheRock artifact slices: lib / run / dev / test
              |
              v
 build_python_packages.py --artifact-dir ...
              |
              +-- rocm sdist (install-time selector)
              +-- rocm-sdk-core wheel
              +-- rocm-sdk-libraries wheel
              +-- rocm-sdk-device-<gfx> wheel(s), kpack-split only
              +-- rocm-sdk-devel wheel
              +-- rocm-profiler wheel, when profiler artifacts exist
              |
              v
  index/upload, then pip resolves rocm[extra] to the matching wheels
```

This is deliberately a **cooperating wheel set**, not a collection of
independent prefixes. Runtime packages use sibling-relative RPATHs and the
development archive contains links back to the runtime packages, so all selected
packages must live in the same `site-packages` directory. `rocm-sdk test` checks
this installation model. [Packaging design](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L20-L25)
[co-installation requirement](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L70-L72)

## What `rocm[...]` means

`rocm` is an install-time selector/meta distribution, intentionally built as an
sdist rather than a runtime wheel. It provides the `rocm-sdk` command and uses
the Python namespace `rocm_sdk`; it selects exact-version dependencies when pip
installs it. [Packaging design](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L11-L19)
[selector implementation](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/setup.py#L51-L99)

The currently declared selector extras are `libraries`, `device`, `devel`, and
`profiler`; there is no `rocm[sdk]` extra. `core` is required by the selector,
whereas the other package roles become extras. [Package registry](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py#L195-L241)

| Pip request | Resolved package role | Included payload and purpose |
| --- | --- | --- |
| `rocm` | `rocm` + required `rocm-sdk-core` | The meta selector plus the OS-specific core runtime/SDK payload. It is the common foundation for the optional groups, not a complete math-library or development installation. [Registry](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py#L199-L220) |
| `rocm[libraries]` | `rocm-sdk-libraries` | Host-side ROCm library runtime: the packager selects the `lib` artifacts for BLAS, FFT, hipDNN, MIOpen, RAND, RCCL, RPP, and the listed provider artifacts, then gives them RPATHs to find core dependencies. Its public lookup surface includes libraries such as hipBLAS, hipBLASLt, hipFFT, hipRAND, hipSPARSE, hipSPARSELt, hipSOLVER, RCCL, MIOpen, RPP, and hipDNN. In the current split model it contains architecture-neutral host code, **not** the GPU kernel packs. [Library filter](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L538-L559) [public library map](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py#L276-L287) [split packaging](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L320-L353) |
| `rocm[device-gfxNNNN]` | `rocm-sdk-device-gfxNNNN` | The GPU-specific companion for a selected ISA. It contains `.kpack` archives, Tensile code objects/databases, MIOpen kernel databases, possible per-ISA shared objects, and other device data; it overlays the libraries wheel's platform directory and requires the same-version libraries wheel. Install only the target(s) required by the machine or deployment image. [Device template](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm-sdk-device/setup.py#L4-L8) [payload/dependency](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm-sdk-device/setup.py#L35-L74) |
| `rocm[device]` | The device wheel inferred for the detected target | A convenience form. At install time, `rocm` chooses the target in this order: `ROCM_SDK_TARGET_FAMILY`, `offload-arch` detection, then the distribution default. On a GPU-less installer, use an explicit `device-gfxNNNN` extra instead; the selector removes generic `device` where it would silently use an unsafe fallback. [Selection order](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L88-L102) [fallback guard](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/setup.py#L63-L83) |
| `rocm[device-all]` | Every published `rocm-sdk-device-gfx*` wheel for the platform | A multi-arch deployment convenience, useful for an image that must serve several GPU types. Per-ISA extras and the `device-all` aggregate are generated from the published target set; platform markers prevent pulling targets not published for the current OS. [Extra generation](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py#L342-L370) |
| `rocm[devel]` | `rocm-sdk-devel` | Development-only prefix material: headers, static/import libraries, CMake/pkg-config metadata, compiler resources, and tools that would make runtime wheels unnecessarily large or unable to preserve links. It is a catch-all development package, not a replacement for `libraries`; use `rocm[libraries,devel]` when building against ROCm. [Devel design](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L35-L43) [framework build guidance](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L283-L305) |
| `rocm[profiler]` | `rocm-profiler` plus required core | Optional profiling tools and their runtime components: ROCm Systems Profiler (`rocprof-sys-*`) and ROCm Compute Profiler (`rocprof-compute`). It is intentionally outside the ordinary runtime/library group, so applications do not acquire profiler payload unless requested. It is not built on Windows. [Profiler contents](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L212-L276) [Windows guard](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L259-L295) |

The `rocm-sdk-core` console scripts are wrappers, not evidence that core alone
is a general development prefix. For compiler/build tools the wrapper expands
and dispatches through `rocm-sdk-devel` when present; system-information tools
can execute directly from core without the expansion cost. [Core CLI dispatch](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm-sdk-core/src/rocm_sdk_core/_cli.py#L71-L124)

## Why the development package expands at runtime

Wheel files cannot faithfully contain the symlink and hardlink arrangement of a
normal ROCm prefix. The `rocm-sdk-devel` wheel therefore carries `_devel.tar` or
`_devel.tar.xz`; `rocm-sdk init` or `rocm-sdk path --root` expands it as a
sibling tree in `site-packages`. During expansion, file symlinks are normally
materialized as hardlinks to retain compatibility and save space. [Devel implementation](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_devel.py#L4-L20)
[expansion path](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_devel.py#L50-L86)

In kpack-split releases the devel wheel is architecture-neutral, so it cannot
contain every device payload itself. Each installed device wheel supplies a
manifest; devel initialization hardlinks the selected device files from the
libraries overlay into the expanded development tree and records those links in
the device wheel's `RECORD`, allowing `pip uninstall` to remove them. This
behavior was added specifically to close the incomplete-devel-tree gap in the
split design. [Implementation](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_devel.py#L308-L381)
[PR #5778](https://github.com/ROCm/TheRock/pull/5778)

Operational consequence: after adding or removing a device wheel from an
already initialized environment, rerun `rocm-sdk init` or `rocm-sdk test`.
Compiler wrappers deliberately do not rescan device metadata on every
invocation. [Release instructions](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/RELEASES.md#L261-L271)

## How TheRock builds the wheel set

### 1. Build and slice artifacts

At CMake level, `therock-artifacts` is the fan-in target for all artifact
slices, while `therock-dist` materializes distribution directories. Each
artifact target receives staged project outputs through its descriptor and is
added as a dependency of both its distribution and the global artifact target.
[Target definitions](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/cmake/therock_default_targets.cmake#L4-L26)
[Artifact assembly](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/cmake/therock_artifacts.cmake#L8-L23)
[Fan-in wiring](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/cmake/therock_artifacts.cmake#L331-L348)

If `KPACK_SPLIT_ARTIFACTS` is enabled, TheRock first produces unsplit artifact
slices, then invokes the kpack splitter to create generic and per-target
artifacts before flattening the split output. [Split path](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/cmake/therock_artifacts.cmake#L196-L328)

### 2. Re-layout artifacts into package staging directories

`build_tools/build_python_packages.py` reads
`therock_manifest.json` from the artifacts, detects the
`KPACK_SPLIT_ARTIFACTS` flag, validates that all advertised device targets were
actually fetched, and constructs package staging directories from templates.
It rejects partial device sets rather than publishing an installable-looking
but incomplete release. [Manifest and completeness checks](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L49-L66)
[Failure on incomplete targets](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L95-L187)
[PR #6433](https://github.com/ROCm/TheRock/pull/6433)

For Linux runtime files the packager preserves only SONAME-bearing shared
libraries, resolves or replaces symlinks, and patches/normalizes RPATHs to find
dependent sibling packages. This is why the runtime wheels are relocatable
inside one Python environment without trying to reproduce a traditional
`/opt/rocm` symlink tree. [Runtime materialization and RPATH](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/_therock_utils/py_packaging.py#L341-L372)
[RPATH patching](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/_therock_utils/py_packaging.py#L469-L558)

In kpack-split mode, the build order is conceptually:

1. Populate `rocm-sdk-core` and the generic host `rocm-sdk-libraries` wheel.
2. Populate one `rocm-sdk-device-<target>` wheel for each ISA target.
3. Build the single generic `rocm` selector sdist.
4. Build one generic `rocm-sdk-devel` wheel; it excludes the generic test
   component because those host-only test binaries cannot execute without
   device code.

The code embodies that order and the distinct host/device ownership. [Kpack-split implementation](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L317-L400)

In legacy mode, TheRock instead creates one `rocm-sdk-libraries-{family}` wheel
per target family, followed by matching family-specific `rocm` sdists and devel
wheels. [Legacy implementation](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L402-L493)

All distributions except `rocm` are built as wheels; `rocm` is built as an
sdist. The closed release-builder context is why the code calls setuptools
directly instead of a generic PEP 517 frontend. [Build invocation](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/_therock_utils/py_packaging.py#L795-L854)

### 3. CI/release packaging and publication

The multi-arch release workflow first builds artifacts, then calls the reusable
Python-package workflow with the artifact run and Linux/Windows GPU-family
metadata. That workflow fetches all staged artifacts, invokes
`build_python_packages.py`, and uploads a package index for the workflow run.
[Release dependency graph](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/.github/workflows/multi_arch_release_linux.yml#L56-L108)
[Python-wheel workflow](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/.github/workflows/build_portable_linux_python_packages.yml#L126-L175)

For a release, publication waits on the Python-package and other package jobs,
then calls the release-bucket publisher with the kpack-split result. The top
level supports manual `dev` dispatches, while scheduled nightly releases are
called from `rockrel`. [Release entry point](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/.github/workflows/multi_arch_release.yml#L4-L16)
[Publishing step](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/.github/workflows/multi_arch_release_linux.yml#L142-L175)

The release-maintenance procedure then promotes selected prerelease/nightly
wheel sets to release versions and uploads them to the production Python bucket;
the scripts support a flat multi-arch layout. [Release promotion procedure](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/how_to_do_release.md#L1-L9)
[Multi-arch promotion](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/how_to_do_release.md#L127-L176)

## Recommended installation shapes

The current release guide publishes nightly multi-arch wheels at
`https://rocm.nightlies.amd.com/whl-multi-arch/`; create a virtual environment
first. Select the exact version/index appropriate to the intended release
channel rather than mixing independently sourced wheel sets. [Release index and
venv guidance](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/RELEASES.md#L102-L123)

The checked-out guide does not yet document an equivalent stable `pip` index—it
contains a TODO for the `repo.amd.com` release channel—so the commands below use
the documented nightly endpoint only as an example of the package shape, not as
a claim about a stable-release URL. [Release-guide TODO](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/RELEASES.md#L102-L109)

```bash
# Runtime with the host libraries and device payload for one GPU target.
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[libraries,device-gfx942]"

# Build C++/HIP projects as well: add development files and tools.
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[libraries,devel,device-gfx942]"

# An image that must carry every published device payload.
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[libraries,device-all]"

# Profiling tools without the math-library/device payload.
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[profiler]"
```

These commands follow the documented multi-arch forms. A device wheel being
installable only proves that it was published; it does not guarantee that the
target is yet runtime-tested. Consult `SUPPORTED_GPUS.md` for the release's
sanity-tested status. [Documented examples and caution](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/RELEASES.md#L145-L174)

After a development installation, use the public command rather than guessing a
site-packages path:

```bash
rocm-sdk init                  # eagerly expand rocm-sdk-devel
cmake -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" ...
# or: -DROCM_HOME="$(rocm-sdk path --root)"
```

`rocm-sdk path --root`, `--bin`, and `--cmake` all require the devel package;
the framework guidance uses these paths for CMake and `PATH` setup. [CLI behavior](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/__main__.py#L15-L49)
[Framework guidance](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L283-L305)

## Relationship to framework wheels

TheRock treats ROCm Python packages as the SDK/runtime substrate for framework
wheels rather than embedding a complete ROCm prefix in every framework wheel.
For a framework build, the documented input is `rocm[libraries,devel]`; for a
framework wheel's runtime requirements, the documented pattern is an exact
`rocm[libraries]==<rocm-sdk-version>` dependency followed by
`rocm_sdk.initialize_process()` before loading ROCm-linked extensions.
[Framework dependency and initialization contract](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L307-L339)

In the release DAG, the freshly built ROCm wheel index is passed to subsequent
PyTorch and JAX wheel workflows. This makes the ROCm wheel set the pinned input
to those builds, rather than a separately rebuilt SDK in each framework job.
[PyTorch/JAX handoff](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/.github/workflows/multi_arch_release_linux.yml#L234-L279)

## Practical conclusions

- Use `libraries` for host-side ROCm library runtime, but add an explicit
  `device-gfx*` extra when the application requires TheRock's split GPU kernels.
- Use `libraries,devel` for a conventional CMake/HIP development environment;
  `devel` creates the coherent prefix but expects the runtime wheels beside it.
- Use `profiler` only for profiling tools, and prefer the selector extra over a
  direct `rocm-profiler` install because the direct package intentionally does
  not declare its own core dependency. [Profiler dependency model](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md#L268-L281)
- Treat legacy family-suffixed library wheels as historical compatibility
  layout. The multi-arch/device-extra model is the current release path, as
  documented by the merged release-workflow migration. [PR #5790](https://github.com/ROCm/TheRock/pull/5790)
- Verify a concrete installation with `rocm-sdk test`, and refresh the devel
  tree after device-wheel changes. [CLI test selection](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/__main__.py#L54-L92)

## TensileLite-client ownership review (2026-08-13)

### Conclusion

The executable's own native dependency closure does **not** require devel
headers, CMake metadata, or static archives. Its eventual existing-package
destination is therefore the production BLAS runtime:
**`blas_lib` → `rocm-sdk-libraries` → `rocm[libraries]`**. Do not embed it in
the pure-Python `tensilelite` wheel. Q117 remains the current implementation
policy: the client stays in `blas_test` until it is promoted.

Q128 removes `rocm[devel]` from the TensileLite Python runtime. The package uses
the core wheel's public metadata/root API for compatibility validation and
compiler lookup; it is not a native `tensilelite-client` dependency. A user who
only executes the client against already-provided configuration and code objects
needs the assembled runtime closure, not devel. The Python installation uses
`rocm[libraries,device-gfx<target>]`, and the executable remains at
`libexec/hipblaslt/tensilelite/tensilelite-client`.
[TensileLite installation contract](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/README.md#L30-L56)

This does **not** mean the Python wheel needs the client for every use. The
current authoritative decision record says that import, help, logic validation,
`TensileCreateLibrary`, and hipBLASLt device-library generation are client-free.
The client is required only when a benchmark or retune operation actually needs
to launch it. That separation keeps it out of the Python wheel and the
code-generation dependency graph; it does not make devel files a native-client
requirement.
[Q115](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/PythonBuildGrillingDecisions.md#L1992-L2017)

### Why it is missing today

The implementation and current accepted Q117 policy put the client only in the
non-Windows CMake `tests` component and in TheRock's `blas_test` artifact. The
kpack-split Python packager creates `rocm-sdk-libraries` from the `blas` **`lib`**
component only, and deliberately excludes the `test` component from the generic
devel wheel. Therefore the executable reaches only a reconstructed CI test root;
it cannot reach either the final `rocm[libraries]` or `rocm[devel]` install.
[Current CMake install rule](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/CMakeLists.txt#L802-L892)
[BLAS artifact rule](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/math-libs/BLAS/artifact-blas.toml#L66-L109)
[libraries filter](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L538-L559)
[devel test exclusion](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L370-L392)

Q117 explicitly chose this as a temporary maturity policy and superseded the
earlier Q098 proposal to put the client in the production `blas_lib` artifact.
Q129 now records the eventual `blas_lib` / `rocm[libraries]` destination, while
leaving Q117 as the current implementation policy. The current source wiring
therefore remains test-owned until the agreed promotion occurs.
[Q117, prior policy](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/PythonBuildGrillingDecisions.md#L2046-L2076)

### Accepted eventual ownership

Q129 records this future production-runtime ownership decision:

```text
native client artifact component: blas_lib
public SDK destination:          rocm-sdk-libraries / rocm[libraries]
runtime prerequisites:           rocm (core) + rocm[libraries]
Python runtime install:          rocm[libraries,device-gfx<target>]
Python distribution:             tensilelite wheel remains client-free
standard executable path:        <ROCm root>/libexec/hipblaslt/tensilelite/tensilelite-client
initial platform scope:          non-Windows, until Windows client support is separately ready
```

TheRock's existing `rocm-sdk-libraries` filter already selects the BLAS `lib`
component, so production `blas_lib` ownership is the minimal path into the final
regular distribution and the Python `rocm[libraries]` wheel. The devel tree
will then obtain the same runtime file through its normal composition; no second
client installation or dedicated package is needed. [Libraries filter](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L538-L559)
[devel composition](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/_therock_utils/py_packaging.py#L560-L636)

This preserves the intended user experience:

```bash
python -m pip install --index-url <rocm-wheel-index> \
  'rocm[libraries,device-gfx<target>]'
python -m pip install --index-url <rocm-wheel-index> tensilelite

# create-library remains client-free; benchmark/retune resolves the client
# from the selected SDK root only when it needs to execute it.
```

When the client is promoted, the implementation work is bounded: build and
install it for the production BLAS artifact; classify its `libexec` path in
`blas_lib` instead of `blas_test`; and add an installed-SDK test that installs
the above wheel set and executes a client-using benchmark category. Do not move
raw `rocisa`, copied Python tests, or the native client into the `tensilelite`
wheel.

Do not introduce a dedicated `rocm[tensilelite]` package. It would create a
second, artificial installation boundary around an executable that consumes the
existing SDK tree. `rocm[devel]` remains the additional SDK layer for the full
Python tuning/code-generation environment, not the executable's sole owner.

### Dependency closure: why it must live in the existing SDK tree

The executable is not a self-contained companion binary. Its direct CMake link
closure is `hip::device`, `hip::host`, `roc::tensilelite-host`,
`rocisa::rocisa-cpp`, OpenMP, and (on non-Windows) `amd_smi`; optional
MX-data support adds the hipBLASLt mxDataGenerator target, and optional
in-process profiling adds rocprofiler-sdk. The default TensileLite host build is
static, so it normally becomes part of the executable rather than a separately
shipped `libtensilelite-host.so`; the other native dependencies remain ROCm SDK
dependencies. Neither client target directly links `libhipblaslt`; the native
runner is a HIP/Tensile benchmark executable, not a wrapper around the public
hipBLASLt library. [Client link graph](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/client/CMakeLists.txt#L15-L43)
[optional profiling dependency](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/client/CMakeLists.txt#L77-L104)
[static-by-default host](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/CMakeLists.txt#L15-L31)

The host library's public closure includes Origami and HIP, while its private
closure includes the HIP device interface and zlib. TheRock builds Origami as a
shared BLAS-side dependency, and zlib/zstd are sysdeps `lib` artifacts. A
complete runtime installation must therefore retain the normal core and BLAS
runtime closures rather than pretending that a copied executable can own all of
its dependencies. [Host link closure](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/CMakeLists.txt#L86-L94)
[TheRock Origami build](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/math-libs/BLAS/CMakeLists.txt#L97-L110)
[sysdeps library ownership](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/third-party/sysdeps/linux/artifact.toml#L56-L62)

TheRock's wheel-test runner explicitly supplies the SDK root's `lib/` and
`lib/llvm/lib/` because the client needs `libomp` from the latter and otherwise
fails at load; the client link graph also requires HIP and AMD-SMI support. Those
payloads belong to the existing core SDK artifacts: the Python core-wheel filter
includes `core-hip`, `core-amdsmi`, `amd-llvm`, core runtime, and system
dependencies. [Loader setup](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/github_actions/test_executable_scripts/pytest_runner.py#L190-L203)
[core artifact selection](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L496-L535)

At execution time, the client also consumes the generated benchmark
configuration, the selected per-GFX `TensileLibrary.{dat,yaml}`, and its `.co`
code objects. Those are provided by the caller's generated library tree or, for
installed ROCm content, by the selected library/device payload; they are not
assets that a standalone client wheel can sensibly duplicate. [Client launch and
configuration](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/ClientWriter.py#L208-L268)
[per-GFX library and code-object selection](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/ClientWriter.py#L800-L832)

The Python package resolves the client only on first use and validates its
version. Under Q128 it obtains the active Python SDK's core payload and version
without expanding devel, so its wheel runtime requires the matching
`rocm[libraries,device-...]` set and never searches an arbitrary `PATH`.
[Deferred native lookup](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/_runtime.py#L37-L75)
[Python-SDK root selection](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/_rocm.py#L77-L122)

### Why `rocm-sdk-core` is not the client owner

`rocm` is only the selector/meta sdist. Its required native foundation is
`rocm-sdk-core`; `rocm[libraries]` is the optional math/ML layer. Core currently
ships AMD LLVM, base/ROCm-core metadata, HIP and low-level runtime pieces, AMD
SMI, kpack/OCL support, host BLAS/SuiteSparse, and system dependencies. The
libraries wheel separately selects the `lib` components of BLAS, FFT, hipDNN,
MIOpen, provider, RAND, RCCL, and RPP artifacts. [Package registry](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/_dist_info.py#L199-L240)
[core selector](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L496-L535)
[libraries selector](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L538-L559)

The argument **for** putting the client in core is narrow but real: its direct
native dependencies include HIP, AMD SMI, OpenMP/LLVM, and sysdeps, all of which
are core-owned. It does not directly link `libhipblaslt`.

The argument **against** core is decisive. The client is built and installed by
the hipBLASLt subproject, links the TensileLite host library, and that host has
a BLAS-side Origami dependency. TheRock's build graph deliberately has the
target-specific `blas` artifact depend on core artifacts, not the reverse.
Making core own this BLAS/Tensile executable would either make the mandatory,
target-neutral core depend on the optional BLAS layer, duplicate that closure,
or leave an under-specified runtime dependency. [Client link graph](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/client/CMakeLists.txt#L15-L43)
[host dependency](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/CMakeLists.txt#L86-L94)
[BLAS dependency direction](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/BUILD_TOPOLOGY.toml#L596-L603)

Thus, if constrained to the existing package set, `rocm-sdk-libraries` is the
right owner: it already represents the optional BLAS runtime layer that depends
on core and can carry the client at its standard `libexec` path. Core remains
the common prerequisite through the meta package; devel remains a separate
complete-SDK view.

### Why the native client does not belong in the Python wheel

The `tensilelite` wheel is intentionally a pure Python distribution. Its
release validator requires the exact `py3-none-any` wheel tag, while the client
is a platform-native executable with HIP, AMD-SMI, OpenMP/LLVM, and BLAS-side
runtime dependencies. Including it would either make every TensileLite wheel a
platform-specific native bundle or require vendoring/duplicating the same ROCm
runtime closure that TheRock already owns and version-locks. [Wheel-tag
contract](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/scripts/check_release_wheel_contents.py#L92-L113)
[client native closure](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/client/CMakeLists.txt#L15-L43)

The wheel must instead be able to *invoke* a compatible external client. Its
runtime resolves and validates the executable only on first benchmark/retune
use; import, logic validation, and device-library generation remain
client-free. The standard external location is the ROCm root's
`libexec/hipblaslt/tensilelite/tensilelite-client[.exe]`, which allows the
native executable to share the same core/libraries/device package closure as
the installed ROCm SDK. [Deferred client lookup](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/_runtime.py#L37-L68)
[standard client path](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/_tensilelite_client_binding.py#L104-L106)

### Command-flow distinction: client versus devel root

The supported command dispatcher makes the execution boundary explicit:

```text
tensilelite create-library
  -> validates compiler, assembler, offload bundler, and HIP configuration
  -> emits library metadata and code objects
  -> never resolves or launches tensilelite-client

tensilelite run
  -> validates the same compiler/bundler toolchain
  -> BenchmarkProblems -> LibraryLogic -> LibraryClient
  -> only the actual benchmark/client-execution stage resolves and launches
     tensilelite-client with a generated configuration
```

`create-library` and `run --build-only` therefore do not require the client.
The regular `run` workflow needs it only when it reaches device-bound benchmark
execution; `ClientWriter` writes the generated library and code-object paths
into the config, then invokes the client. [CLI dispatch](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/cli.py#L13-L33)
[create-library toolchain setup](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/tensilelite_create_library/run.py#L1029-L1071)
[run toolchain setup](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/tensilelite.py#L643-L700)
[client execution](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/ClientWriter.py#L224-L268)

Neither command has a devel-only native dependency in this code path. Their
compiler and bundler lookups are constrained to `<selected-root>/bin` and
`<selected-root>/lib/llvm/bin`; TheRock puts the AMD LLVM and HIP runtime/tool
payload in `rocm-sdk-core`. Q128 resolves that root through the core package's
public API, so devel is not a `tensilelite-client`, `run`, or
`create-library` payload dependency.
[tool lookup paths](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/Toolchain/Validators.py#L35-L57)
[core tool payload selection](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/build_python_packages.py#L496-L535)

The compiler payload itself is already in `rocm-sdk-core`: TheRock classifies
AMD LLVM's `lib/llvm/bin/**`, `libexec/**`, and HIP support tree as `run`, and
the core wheel includes that artifact's `lib` and `run` components. The core
wheel declares venv console scripts for `amdclang++`, `amdclang`,
`clang-offload-bundler`, `hipcc`, `hipconfig`, and `offload-arch`. Its wrapper
uses the core physical package as the fallback whenever devel is not installed;
core-wheel tests exercise those compiler commands. Thus `rocm[devel]` is not
needed merely to obtain or execute TensileLite's compiler/bundler tools.
[AMD LLVM artifact layout](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/compiler/artifact-amd-llvm.toml#L12-L48)
[core console scripts](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm-sdk-core/setup.py#L68-L106)
[core fallback](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm-sdk-core/src/rocm_sdk_core/_cli.py#L71-L105)
[compiler command tests](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm/src/rocm_sdk/tests/core_test.py#L28-L45)

### Wheel build/install boundary

`rocm_sdk path --root` is not part of wheel construction or `pip install`.
The wheel build receives the base ROCm release through
`TENSILELITE_ROCM_VERSION`; `setup.py` only composes that value with the checked
in component version. hipBLASLt's CMake build installs the already-built wheel
with `pip --force-reinstall --no-deps` and invokes generator commands with an
explicit `ROCM_PATH=${HIPBLASLT_BUILD_ROCM_ROOT}`. Neither path invokes the
Python SDK CLI. [Wheel metadata build path](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/setup.py#L12-L50)
[CMake installation and build environment](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/cmake/hipblaslt_python.cmake#L65-L95)

Before Q128, the command was reached after installation when Python imported
the installed `tensilelite` package. Q128 replaces that devel-root lookup with
the core package's public API, so release/artifact CI no longer needs devel to
exercise installed-package initialization.
[Import initialization](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/__init__.py#L4-L20)
[Python SDK root call](https://github.com/ROCm/rocm-libraries/blob/df51e32d0b4e9186333e94fa0ddec32bbee8c9e1/projects/hipblaslt/tensilelite/tensilelite/_rocm.py#L77-L105)

### Removing the devel dependency from version validation

Q128 uses the required core wheel without expanding `rocm[devel]`. The public
Python module is named `rocm_sdk_core` (not `rocm_core`) and exposes the core
payload root plus its base native ROCm version. This preserves the existing
`.info/version` base-release compatibility policy while keeping the lookup
inside the required core package. [Core public module](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/build_tools/packaging/python/templates/rocm-sdk-core/src/rocm_sdk_core/__init__.py#L1-L4)

Conceptually, the replacement is:

The later two-adapter runtime design supersedes this physical-core-root example.
For a Python SDK package installation, TensileLite validates the active
`rocm_sdk_core.__version__` directly and runs compiler/toolchain commands through
the active interpreter's `rocm-sdk-core` console-script trampolines. It does not
import a core-root helper. A conventional ROCm prefix continues to use its
root-relative tools and `.info/version` base compatibility value.

This removes the version-validation reason to call `rocm_sdk path --root` while
retaining core's `.info/version` as the native compatibility authority.

It does not by itself locate the client after it moves into
`rocm-sdk-libraries`: the current public SDK offers library lookup and a devel
root API, but no public "libraries package root" API. The matching follow-up is
to expose that narrow package-root/tool-path lookup from TheRock's Python SDK,
or otherwise make TensileLite's client resolver consume a stable public library
path API. It should not rely on the generated private `_rocm_sdk_libraries...`
module name.

## Primary-source reading list

- [TheRock Python packaging design](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/docs/packaging/python_packaging.md)
- [TheRock release and installation guide](https://github.com/ROCm/TheRock/blob/6339d5fc2e8f924a7b3072a9c34623271c9626f7/RELEASES.md)
- [Multi-architecture packaging issue #3323](https://github.com/ROCm/TheRock/issues/3323)
- [Per-ISA device-wheel PR #4308](https://github.com/ROCm/TheRock/pull/4308)
- [Multi-arch Python CI PR #4487](https://github.com/ROCm/TheRock/pull/4487)
- [Devel/device linking PR #5778](https://github.com/ROCm/TheRock/pull/5778)
- [Release-set completeness PR #6433](https://github.com/ROCm/TheRock/pull/6433)
- [Installed profiler-wheel test PR #6867](https://github.com/ROCm/TheRock/pull/6867)
