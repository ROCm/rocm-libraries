---
name: hipdnn-superbuild
description: Build hipDNN with providers via the repository superbuild. Faster than standalone since providers build alongside hipDNN in a single CMake invocation. On Windows, auto-runs the wheel-based ROCm setup if not already prepared.
argument-hint: "[preset] [clean] [ROCM_PATH=<path>] [CLANG_PATH=<path>] [GPU_TARGETS=<arch>] [SHA=<commit>]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Superbuild

Use this skill when the user asks to configure or build hipDNN through the rocm-libraries repository superbuild. It builds only; use `hipdnn-superbuild-test` for tests after a successful build.

## Inputs

Infer options from the user request:

- **Preset**: default `hipdnn-providers`
- **Clean rebuild**: remove the build directory before configuring only when the user asks for a clean build and the active host policy permits deletion
- **ROCm path**: optional `ROCM_PATH=<path>` override; Linux defaults to `/opt/rocm`. On Windows, when omitted it is derived from the wheel venv (`<venv>/Lib/site-packages/_rocm_sdk_devel`, venv default `D:/develop/latest_wheels`)
- **Clang path**: optional Windows `CLANG_PATH=<path>` override; default `D:/develop/dist/clang/bin`. Clang is a prerequisite and is not provisioned; install it via `scripts/windows/windows_build_setup.ps1` if missing
- **GPU targets**: optional `GPU_TARGETS=<arch>` override; Windows wheel setup defaults to `gfx1151`
- **Wheel SHA**: optional Windows `SHA=<commit>` to install pinned S3 staging wheels instead of nightlies
- **Provision mode**: Windows `--provision auto|always|never`; default `auto` provisions (creates the venv and pip-installs the ROCm SDK wheels) only when the SDK is missing, `always` forces a fresh wheel pull, `never` validates existing paths only
- **Jobs**: optional explicit parallelism only when the user requests it and active workspace instructions permit it; otherwise let Ninja auto-detect

## Presets

Read `CMakePresets.json` from the repository root if exact preset contents matter. Common hipDNN presets:

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

1. Determine the repository root:
   ```bash
   git rev-parse --show-toplevel
   ```

2. Choose the build and log locations:
   - First honor any active workspace or repository instructions for artifact directories and build output safety.
   - If no such instructions exist, use `BUILD_DIR=<repo-root>/build`.
   - Keep full configure/build output in a log file and show only a short tail on failure.

3. Locate this skill's helper directory. Skills are host-level, not tied to a repo checkout — **default to the scripts bundled with the skill you were invoked from** (`<skill-directory>/scripts`), even when you are working inside a repo or worktree. Do NOT run the `<repo-root>/projects/hipdnn/tools/ai/skills/hipdnn-superbuild/scripts` copy just because a checkout is present: that copy can be a stale stub (on `develop`) or an unmerged in-progress version (on a feature branch). Use the source-checkout copy only when you are actively developing this skill itself and intend to exercise your in-progress edits, or when the invoked skill has no bundled `scripts/` directory.

4. Resolve ROCm and Clang paths (Windows also provisions the ROCm SDK wheels when missing):
   ```bash
   python3 <scripts>/windows_rocm_setup.py --repo-root <repo-root> [--venv-path <path>] [--rocm-path <path>] [--clang-path <path>] [--gpu-targets <arch>] [--sha <commit>] [--provision auto|always|never]
   ```
   On Linux this echoes only provided overrides. On Windows it validates the wheel-based ROCm install and, when the SDK is absent (or `--provision always`), creates the venv and pip-installs the ROCm SDK wheels before printing `KEY=VALUE` lines on stdout. Progress goes to stderr, so stdout carries only the `ROCM_PATH=`/`CLANG_PATH=`/`GPU_TARGETS=` lines. Clang is a prerequisite and is not provisioned; a missing clang is reported as an error.

5. If a clean rebuild was requested, remove the selected build directory using the active host's normal approval/safety flow.

6. Configure from the repository root. Always bind the preset configure to the selected build directory so configure and build operate on the same tree:
   ```bash
   cmake --preset <preset> -B <build-dir> [extra -D options]
   ```
   Add `-DROCM_PATH=<path>` when a ROCm path is resolved or provided. On Windows also add `-DCMAKE_PROGRAM_PATH=<clang-path>` and `-DGPU_TARGETS=<arch>`.

7. Build with output redirected to a log:
   ```bash
   cmake --build <build-dir> > <log> 2>&1
   ```
   If explicit jobs are allowed and requested, pass them through to CMake/Ninja. On failure, report the log path and tail the last relevant lines.

8. If the build fails with a stale CMake cache error such as `does not match the source`, clean the selected build directory once, reconfigure with the same `-B <build-dir>` command, and retry once. Do not loop.

9. On Windows, always stage the wheel's `amd_comgr.dll` app-local into `<build-dir>/bin` after a successful build:
   ```bash
   python3 <scripts>/comgr_stage.py --rocm-bin <rocm-bin> --build-dir <build-dir> --verbose
   ```
   The AMD driver leaves an old `amd_comgr.dll` in `C:\Windows\System32` that outranks the wheel's copy on PATH, so MIOpen otherwise loads stale comgr and can fail to JIT-build kernels at runtime (GCN-assembly Winograd solvers are the common example, but the mismatch is not limited to them). Do this on every Windows build rather than only when a specific kernel path is expected. The Win32 loader checks the executable's own directory before System32, so an app-local copy in `<build-dir>/bin` wins; PATH manipulation alone cannot. The helper compares the wheel comgr's PE version against any already-staged copy and **skips the copy when the versions match** (content-hash fallback when version metadata is absent), so it is cheap to re-run. This step is a no-op on Linux. The test runner (`cmake_run.py`) stages comgr on its own as well, so this build step is belt-and-suspenders that makes the app-local copy present immediately after build.

## Report

Summarize:

- Preset used and components expected from that preset
- Build result
- Build directory and log path
- Windows ROCm, Clang, and GPU target values when applicable
- Next step: run `hipdnn-superbuild-test` if tests are needed

## Notes

- `scripts/windows_rocm_setup.py` and `scripts/comgr_stage.py` are bundled in this skill so linked and copied installs work independently. `windows_rocm_setup.py`'s Windows wheel-provisioning logic is a Python port of `projects/hipdnn/scripts/windows/wheel_build_setup.ps1`; that PowerShell script is left in place for interactive users and `tools/dnn-benchmarking/setup.ps1`. Keep the two in sync.
- `comgr_stage.py` only does work on Windows; it stages the wheel's `amd_comgr.dll` app-local and emits a diagnostic when `C:\Windows\System32\amd_comgr.dll` is present (it shadows PATH and is why the app-local copy is needed).
- Missing provider dependencies such as MIOpen or hipBLASLt still need to be installed or available through the selected ROCm environment.
- Product test execution is intentionally out of scope for this skill.
