---
name: hipdnn-superbuild
description: Build hipDNN with providers via the repository superbuild. Faster than standalone since providers build alongside hipDNN in a single CMake invocation. On Windows, auto-runs the wheel-based ROCm setup if not already prepared.
argument-hint: "[preset] [clean] [ROCM_PATH=<path>] [CLANG_PATH=<path>] [GPU_TARGETS=<arch>] [SHA=<commit>] [VENV_PATH=<path>] [pull]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Superbuild

Use this skill when the user asks to configure or build hipDNN through the rocm-libraries repository superbuild. It builds only; use `hipdnn-superbuild-test` for tests after a successful build.

## Inputs

Infer options from the user request:

- **Preset**: default `hipdnn-providers`
- **Clean rebuild**: remove the build directory before configuring only when the user asks for a clean build and the active host policy permits deletion
- **ROCm path**: optional `ROCM_PATH=<path>` override; Linux defaults to `/opt/rocm`
- **Clang path**: optional Windows `CLANG_PATH=<path>` override; default `D:/develop/dist/clang/bin`
- **GPU targets**: optional `GPU_TARGETS=<arch>` override; Windows wheel setup defaults to `gfx1151`
- **Wheel SHA**: optional Windows `SHA=<commit>` to pin the wheel setup
- **Wheel venv**: optional `VENV_PATH=<path>` override; default is the per-worktree `<repo-root>/.rocm_wheels`
- **Pull wheels**: pass through to `wheel_setup.py --pull` only when the user asks to refresh wheels; it reinstalls the current worktree's venv
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

3. Locate this skill's helper directory:
   - Installed skill layout: `<skill-directory>/scripts`
   - Source checkout fallback: `<repo-root>/projects/hipdnn/tools/ai/skills/hipdnn-superbuild/scripts`

4. Resolve ROCm and Clang paths:
   ```bash
   python3 <scripts>/windows_rocm_setup.py --repo-root <repo-root> [--rocm-path <path>] [--clang-path <path>] [--gpu-targets <arch>] [--sha <commit>]
   ```
   On Linux this echoes only provided overrides. On Windows it detects the wheel-based ROCm install and prints `KEY=VALUE` lines for subsequent commands. When `--rocm-path` is omitted it auto-discovers the per-worktree venv at `<repo-root>/.rocm_wheels` first, then the global fallback venv.

   To **provision or refresh** the wheel-based ROCm install (instead of only detecting an existing one), use the cross-platform `wheel_setup.py`. This is the Python port of `projects/hipdnn/scripts/windows/wheel_build_setup.ps1`, so the same wheel-pull workflow runs on Windows or Linux:
   ```bash
   python3 <scripts>/wheel_setup.py --repo-root <repo-root> [--gpu-targets <arch>] [--sha <commit>] [--pull]
   ```
   With `--repo-root` it provisions a **per-worktree** venv at `<repo-root>/.rocm_wheels` (each git worktree is its own root, and the directory is gitignored). It creates or reuses the venv, installs ROCm wheels from nightlies (or S3 staging when `--sha` is given), runs `rocm-sdk init`, and prints `ROCM_PATH=`, `ROCM_BIN=`, `GPU_TARGETS=`, and (when applicable) `CLANG_PATH=`. An existing venv is reused untouched unless `--pull` is passed, which deletes and reinstalls it. Feed the emitted `ROCM_PATH`/`CLANG_PATH`/`GPU_TARGETS` into the configure step and the emitted `ROCM_BIN` into the comgr staging step below.

   Per-worktree wheels keep each build self-consistent: a `--pull` (or a re-pull of a shared venv) changes DLL sonames such as `hiprtc<ver>.dll`, which silently invalidates builds linked against the old names (load-time `0xC0000135`). Isolating wheels per worktree means a pull in one worktree never breaks another's build; the cost is ~9 GB of disk per worktree. Use `--venv-path` to point at a shared venv instead when disk is tight, accepting that a re-pull then requires rebuilding every dependent build tree.

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

9. On Windows, stage the wheel's `amd_comgr.dll` app-local into `<build-dir>/bin` after a successful build:
   ```bash
   python3 <scripts>/comgr_stage.py --rocm-bin <rocm-bin> --build-dir <build-dir> --verbose
   ```
   The AMD driver leaves an old `amd_comgr.dll` in `C:\Windows\System32` that outranks the wheel's copy on PATH, so MIOpen otherwise loads stale comgr and fails to JIT-build GCN-assembly (Winograd) kernels at runtime. The Win32 loader checks the executable's own directory before System32, so an app-local copy in `<build-dir>/bin` wins; PATH manipulation alone cannot. The helper compares the wheel comgr's PE version against any already-staged copy and **skips the copy when the versions match** (content-hash fallback when version metadata is absent), so it is cheap to re-run. This step is a no-op on Linux. The test runner stages comgr on its own as well, so this build step is belt-and-suspenders that makes the app-local copy present immediately after build.

## Report

Summarize:

- Preset used and components expected from that preset
- Build result
- Build directory and log path
- Windows ROCm, Clang, and GPU target values when applicable
- Next step: run `hipdnn-superbuild-test` if tests are needed

## Notes

- `scripts/windows_rocm_setup.py`, `scripts/wheel_setup.py`, and `scripts/comgr_stage.py` are bundled in this skill so linked and copied installs work independently.
- `wheel_setup.py` is the cross-platform replacement for the external `projects/hipdnn/scripts/windows/wheel_build_setup.ps1`; prefer it so the wheel-pull workflow is the same on Windows and Linux.
- Wheels are provisioned per-worktree at `<repo-root>/.rocm_wheels` (gitignored) so a `--pull` never invalidates another worktree's build. This trades disk (~9 GB per worktree) for isolation; pass `VENV_PATH=<path>` to share one venv when disk is constrained.
- `comgr_stage.py` only does work on Windows. It also emits a diagnostic when `C:\Windows\System32\amd_comgr.dll` is present, explaining that it shadows PATH and is the reason for the app-local copy.
- Missing provider dependencies such as MIOpen or hipBLASLt still need to be installed or available through the selected ROCm environment.
- Product test execution is intentionally out of scope for this skill.
