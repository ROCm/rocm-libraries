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
- **ROCm path**: optional `ROCM_PATH=<path>` override; Linux defaults to `/opt/rocm`
- **Clang path**: optional Windows `CLANG_PATH=<path>` override; default `D:/develop/dist/clang/bin`
- **GPU targets**: optional `GPU_TARGETS=<arch>` override; Windows wheel setup defaults to `gfx1151`
- **Wheel SHA**: optional Windows `SHA=<commit>` to pin the wheel setup
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
   On Linux this echoes only provided overrides. On Windows it detects or provisions the wheel-based ROCm install and prints `KEY=VALUE` lines for subsequent commands.

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

## Report

Summarize:

- Preset used and components expected from that preset
- Build result
- Build directory and log path
- Windows ROCm, Clang, and GPU target values when applicable
- Next step: run `hipdnn-superbuild-test` if tests are needed

## Notes

- `scripts/windows_rocm_setup.py` is bundled in this skill so linked and copied installs work independently.
- Missing provider dependencies such as MIOpen or hipBLASLt still need to be installed or available through the selected ROCm environment.
- Product test execution is intentionally out of scope for this skill.
