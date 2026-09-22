---
name: hipdnn-superbuild
description: Build hipDNN with providers via the repository superbuild. Faster than standalone since providers build alongside hipDNN in a single CMake invocation. On Windows, auto-runs the wheel-based ROCm setup if not already prepared.
argument-hint: "[preset] [clean] [ROCM_PATH=<path>] [CLANG_PATH=<path>] [GPU_TARGETS=<arch>] [SHA=<commit>]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Superbuild

Configures and builds hipDNN through the rocm-libraries repository superbuild. It builds
only; use `hipdnn-superbuild-test` for tests after a successful build.

## Inputs

Infer options from the user request:

- **Preset**: default `hipdnn-providers`
- **Clean rebuild**: remove the build directory before configuring only when the user asks for a clean build and the active host policy permits deletion
- **ROCm path**: optional `ROCM_PATH=<path>` override; Linux defaults to `/opt/rocm`. On Windows, when omitted it is derived from the wheel venv (`<venv>/Lib/site-packages/_rocm_sdk_devel`, venv default `D:/develop/latest_wheels`)
- **Clang path**: optional Windows `CLANG_PATH=<path>` override; default `D:/develop/dist/clang/bin`. Clang is a prerequisite and is not provisioned; install it via `projects/hipdnn/scripts/windows/windows_build_setup.ps1` (repo-relative) if missing
- **GPU targets**: optional `GPU_TARGETS=<arch>` override; Windows wheel setup defaults to `gfx1151`
- **Wheel SHA**: optional Windows `SHA=<commit>` to install pinned S3 staging wheels instead of nightlies
- **Provision mode**: Windows `--provision auto|always|never`; default `auto` provisions (creates the venv and pip-installs the ROCm SDK wheels) only when the SDK is missing, `always` forces a fresh wheel pull, `never` validates existing paths only
- **Jobs**: optional explicit parallelism only when the user requests it and active workspace instructions permit it; otherwise let Ninja auto-detect

## Presets

Read `CMakePresets.json` from the repository root if exact preset contents matter.
Common hipDNN presets:

| Preset | Components |
|--------|------------|
| `hipdnn` | hipDNN only |
| `hipdnn-integration-tests` | hipDNN plus integration tests |
| `hipdnn-providers` | hipDNN, miopen-provider, hipblaslt-provider, integration tests |
| `hipdnn-providers-all` | All providers, including unsupported providers |
| `miopen-provider` | hipDNN, miopen-provider, integration tests |
| `hipblaslt-provider` | hipDNN, hipblaslt-provider, integration tests |
| `hip-kernel-provider` | hipDNN, hip-kernel-provider, integration tests |
| `hipdnn-samples` | hipDNN, supported providers, integration tests, samples |

## Workflow

Placeholders below: `<repo-root>` (step 1), `<build-dir>` and `<log>` (step 2),
`<scripts>` (step 3), `<rocm-bin>` — the ROCm `bin` directory, on Windows the wheel
venv's `_rocm_sdk_devel/bin` — and `<PY>`, the Python interpreter the active workspace
or repository instructions mandate, otherwise the active venv's `python`. Resolve it
once and substitute the full path; never invoke a bare `python` or `python3` off `PATH`.

1. Determine the repository root:
   ```bash
   git rev-parse --show-toplevel
   ```

2. Choose the build and log locations:
   - Honor active workspace or repository instructions for artifact directories and
     build output safety first; otherwise use `BUILD_DIR=<repo-root>/build`.
   - Keep full configure/build output in a log file and show only a short tail on
     failure.

3. Locate this skill's helper directory. Skills are host-level — **default to the
   scripts bundled with the skill you were invoked from** (`<skill-directory>/scripts`),
   even inside a repo or worktree. The
   `<repo-root>/projects/hipdnn/tools/ai/skills/hipdnn-superbuild/scripts` copy can be a
   stale stub (on `develop`) or unmerged work (on a feature branch); use it only when
   developing this skill, or when the invoked skill bundles no `scripts/` directory.

4. Resolve ROCm and Clang paths (Windows also provisions the ROCm SDK wheels when
   missing):
   ```bash
   <PY> <scripts>/windows_rocm_setup.py --repo-root <repo-root> [--venv-path <path>] [--rocm-path <path>] [--clang-path <path>] [--gpu-targets <arch>] [--sha <commit>] [--provision auto|always|never]
   ```
   On Linux this echoes only provided overrides. On Windows it validates the wheel-based
   ROCm install and, when the SDK is absent (or `--provision always`), creates the venv
   and pip-installs the ROCm SDK wheels before printing `KEY=VALUE` lines on stdout.
   Progress goes to stderr, so stdout carries only the
   `ROCM_PATH=`/`CLANG_PATH=`/`GPU_TARGETS=` lines. Clang is a prerequisite and is not
   provisioned; a missing clang is reported as an error.

5. If a clean rebuild was requested, remove the selected build directory using the
   active host's normal approval/safety flow.

6. Configure from the repository root, binding the preset configure to the selected
   build directory so configure and build operate on the same tree:
   ```bash
   cmake --preset <preset> -B <build-dir> -G Ninja [extra -D options]
   ```
   Add `-DROCM_PATH=<path>` when a ROCm path is resolved or provided. On Windows also
   add `-DCMAKE_PROGRAM_PATH=<clang-path>` and `-DGPU_TARGETS=<arch>`.

   **`-G Ninja` is required.** No preset sets a `generator`, and downstream tooling
   needs a Ninja tree: `hipdnn-superbuild-test` stops unless `<build-dir>/build.ninja`
   exists, and its `discover_test_targets.py` runs `ninja -t targets`.

   **Use the preset**; it carries the `rocm-clang.cmake` toolchain and the component
   list. Hand-rolling `-DCMAKE_CXX_COMPILER=...` and `-DROCM_LIBS_ENABLE_COMPONENTS=...`
   loses the toolchain, so the host compiler falls back to system GCC and the build
   fails with `-Werror` diagnostics in untouched files, which reads as broken source
   rather than a wrong compiler.

   **Generic-kernel-ingestor / rocKE builds** need flags no preset sets:

   | Flag | Default | Needed when |
   |---|---|---|
   | `HIPDNN_ENABLE_KERNEL_INGESTOR` | OFF | Any descriptor-backed engine. Also gates `hipdnn_validate_descriptors`, which is why that binary is usually absent. |
   | `HIPDNN_ENABLE_SDPA` | OFF | Any attention graph. This is the **frontend**: with it off the SDPA API is `#ifdef`-compiled out and plans silently DECLINE. Must be ON for both the SDK and the provider. |
   | `ENABLE_ASM_SDPA_ENGINE` | ON | Controls the incumbent ASM engine; disabling it is not proof that the intended new engine serves a graph. |
   | `HIPKERNELPROVIDER_ENABLE_ROCKE` | OFF | **Required ON whenever `HIPDNN_ENABLE_KERNEL_INGESTOR` is ON.** The coupling is unconditional: the provider's top-level check inspects no source kind and no descriptor root, so it also fires for HIP-only and embedded-source bundles and when no rocKE KDP exists anywhere. Ingestor ON with this OFF is a fatal configure error, not a degraded build. |
   | `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` | the in-tree `.../kernel_ingestor_engine/descriptors` | `CACHE PATH` naming the authored tree production packaging compiles from. Packaging is wired only while that root holds at least one non-hidden `*.kdp.json`; with none it is dormant and any stale product tree is removed, neither being an error. Set but not a directory is fatal. |
   | `HIPKERNELPROVIDER_KPACK_PYTHON_DIR` | unset | Directory **containing** `rocm_kpack/`; this locates a package, not a compiler interpreter. |
   | `Python3_EXECUTABLE` | system | Explicit environment for packaging dependencies such as `msgpack` and `zstandard`; production compilation retains its selected hermetic wheel interpreter. |

   These flags only do anything on a preset that actually builds hip-kernel-provider. The
   default `hipdnn-providers` preset does **not** include it; the presets that do are
   `hipdnn-providers-all`, `hip-kernel-provider`, `hipdnn-dev-all` and
   `miopen-hipdnn-dev-all`.

   **There is no per-producer production switch.** Producer selection is per-UKD on
   `kernel_source.kind`, so one source root feeds every producer and the descriptors
   under the root decide what gets built. rocKE is resolved once for *every* root, test
   roots included, so an unresolvable comgr is fatal at configure even in a hip-only
   build; `HIPKERNELPROVIDER_ROCKE_COMGR_LIB` names an explicit `libamd_comgr` where a
   System32 copy would otherwise shadow the ROCm one.

   For an ingestor create/extend task,
   [the ingestor RUNBOOK](../hipdnn-ingestor-engine/RUNBOOK.md) owns the full sequence.
   Early device/workspace feasibility has no installation requirement; installed probing
   follows build and installation. Build production packaging as well as provider,
   validator and applicable tests; a plugin build alone does not show that current
   descriptors were packed.

   Declarations travel in UKD `provenance.specialization_contract`. Only the producing
   compiler writes `provenance.effective_spec`, distinct from authored `provenance.spec`;
   generic generation is toolchain-free. No packaging `--profile`, CMake `PROFILES` or
   external root manifest is part of this interface. Read the packaging reference at
   `dnn-providers/hip-kernel-provider/descriptor-packaging/README.md`, resolved against
   the `<repo-root>` from step 1 rather than this skill's own directory — an installed
   skill is copied without the tree above it. A build is not compiler-agreement,
   native-registration or numerical evidence by itself; the RUNBOOK requires those
   observations against the final installation.

7. Build with output redirected to a log:
   ```bash
   cmake --build <build-dir> > <log> 2>&1
   ```
   If explicit jobs are allowed and requested, pass them through to CMake/Ninja. On
   failure, report the log path and tail the last relevant lines.

8. If the build fails with a stale CMake cache error such as `does not match the source`,
   the selected build directory has to be cleaned once. Unlike step 5 this deletion is
   model-initiated: clean only when the active host policy permits deletion, go through
   the active host's normal approval/safety flow, and ask the user first because they
   did not request a clean build. Then reconfigure with the same `-B <build-dir>` command
   and retry once. Do not loop. If deletion is not permitted or the user declines, report
   the stale-cache diagnostic and the build directory path and stop.

9. On Windows, always stage the wheel's `amd_comgr.dll` app-local into `<build-dir>/bin`
   after a successful build:
   ```bash
   <PY> <scripts>/comgr_stage.py --rocm-bin <rocm-bin> --build-dir <build-dir> --verbose
   ```
   The AMD driver leaves an old `amd_comgr.dll` in `C:\Windows\System32` that outranks
   the wheel's copy on PATH, so MIOpen otherwise loads stale comgr and can fail to JIT
   kernels at runtime (commonly GCN-assembly Winograd solvers). The Win32 loader checks
   the executable's own directory before System32, so an app-local copy in
   `<build-dir>/bin` wins where PATH manipulation cannot. The helper **skips the copy
   when the staged PE version matches** (content-hash fallback when version metadata is
   absent), so it is cheap to re-run. No-op on Linux; `cmake_run.py` stages comgr too.

## Report

Summarize:

- Preset used and components expected from that preset
- Build result
- Build directory and log path
- Windows ROCm, Clang, and GPU target values when applicable
- Next step: run `hipdnn-superbuild-test` if tests are needed

## Notes

- **Build speed — comgr cache location.** Kernel packing is dominated by lowering
  through `libamd_comgr`, which caches at `~/.cache/comgr`. On a network home every
  lookup is a network round trip and packing slows by more than 10x, so a *cold* local
  cache beats a *warm* network one. Export `AMD_COMGR_CACHE_DIR` to a RAM disk or local
  disk (e.g. `/tmp/comgr-cache`) before building.
- `scripts/windows_rocm_setup.py` and `scripts/comgr_stage.py` are bundled in this skill
  so linked and copied installs work independently. `windows_rocm_setup.py`'s Windows
  wheel-provisioning logic is a Python port of
  `projects/hipdnn/scripts/windows/wheel_build_setup.ps1`, which remains in place for
  interactive users and for a setup script in a separate benchmarking checkout external
  to this repository. Keep the two in sync.
- `comgr_stage.py` only does work on Windows; it stages the wheel's `amd_comgr.dll`
  app-local and emits a diagnostic when `C:\Windows\System32\amd_comgr.dll` is present.
- Missing provider dependencies such as MIOpen or hipBLASLt still need to be installed
  or available through the selected ROCm environment.
- Product test execution is intentionally out of scope for this skill.
