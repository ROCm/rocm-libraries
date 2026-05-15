---
name: hipdnn-superbuild
description: Build hipDNN with providers via the repository superbuild. Faster than standalone since providers build alongside hipDNN in a single CMake invocation. On Windows, auto-runs the wheel-based ROCm setup if not already prepared.
argument-hint: "[preset] [clean] [jobs=<N>] [ROCM_PATH=<path>] [CLANG_PATH=<path>] [GPU_TARGETS=<arch>] [SHA=<commit>]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Superbuild

Build hipDNN with one or more providers using the repository-root superbuild system. The superbuild configures all components in a single CMake invocation from the repository root, eliminating the need for separate install steps between hipDNN and providers.

> To run tests after the build, use `/hipdnn-superbuild-test`. This skill builds only.

## Arguments

- **Preset** (default: `hipdnn-providers`) — Which CMake preset to use (see table below)
- `clean` — Remove build directory before configuring
- `jobs=<N>` — Parallel build jobs (default: ninja's auto-detect)
- `ROCM_PATH=<path>` — Override the path to the ROCm SDK devel directory
  - On Linux: optional, defaults to `/opt/rocm`
  - On Windows: optional. If unset, auto-discovered or provisioned via the wheel setup script (see Step 1b)
- `CLANG_PATH=<path>` — Windows only. Override the clang toolchain `bin` directory (default: `D:/develop/dist/clang/bin`). Maps to cmake's `CMAKE_PROGRAM_PATH`.
- `GPU_TARGETS=<arch>` — GPU architecture (e.g., `gfx1151`, `gfx942`). Default on Windows wheel setup: `gfx1151`
- `SHA=<commit>` — Windows only. Specific S3 staging build SHA to install via the wheel setup; otherwise uses ROCm nightlies

## Available Presets

From the repository root `CMakePresets.json`:

| Preset | Components |
|--------|-----------|
| `hipdnn` | hipDNN only |
| `hipdnn-integration-tests` | hipDNN + integration tests |
| `hipdnn-providers` | hipDNN + miopen-provider + hipblaslt-provider + integration tests |
| `hipdnn-providers-all` | All providers including unsupported (+ hip-kernel-provider) |
| `miopen-provider` | hipDNN + miopen-provider + integration tests |
| `hipblaslt-provider` | hipDNN + hipblaslt-provider + integration tests |
| `hip-kernel-provider` | hipDNN + hip-kernel-provider + integration tests |
| `hipdnn-samples` | hipDNN + all supported providers + integration tests + samples |

## Step 0: Detect Environment

Run these commands separately (never chain with `&&`):

1. Detect the repository root:
```bash
git rev-parse --show-toplevel
```
Store the result as `REPO_ROOT`.

2. Detect the platform:
```bash
[[ -f /etc/os-release ]] && echo "Linux" || echo "Windows"
```
Store the result as `PLATFORM`.

Derive these paths:
- `BUILD_DIR` = `$REPO_ROOT/build`
- `HELPERS` = `$REPO_ROOT/projects/hipdnn/tools/ai/skills/helpers`

## Step 1: Parse Arguments

Parse `$ARGUMENTS` for:
- **Preset**: one of the preset names above (default: `hipdnn-providers`)
- **Clean**: presence of `clean` keyword
- **Jobs**: value after `jobs=`
- **ROCM_PATH**: value after `ROCM_PATH=`
- **CLANG_PATH**: value after `CLANG_PATH=`
- **GPU_TARGETS**: value after `GPU_TARGETS=`
- **SHA**: value after `SHA=`

## Step 1b: Resolve ROCm and Clang Paths

Run the shared helper. It is a no-op on Linux unless overrides were provided. On Windows it checks for an existing wheel install at the default location, runs the wheel setup script if missing, and prints the resolved paths to stdout:

```bash
python $HELPERS/windows_rocm_setup.py --repo-root $REPO_ROOT [--rocm-path $ROCM_PATH] [--clang-path $CLANG_PATH] [--gpu-targets $GPU_TARGETS] [--sha $SHA]
```

Pass through whichever optional args the user provided. Parse the `KEY=VALUE` lines from the helper's stdout to set `ROCM_PATH`, `CLANG_PATH`, and `GPU_TARGETS` for the subsequent steps. If the helper exits non-zero, surface its stderr and stop.

## Step 2: Clean (if requested)

```bash
rm -rf $BUILD_DIR
```

## Step 3: Configure

Always run cmake configuration — it is incremental and fast when nothing changed, but skipping it after a rebase/merge causes stale cache failures.

The superbuild uses `cmake/toolchains/rocm-clang.cmake`. On Windows it requires both `ROCM_PATH` (the toolchain enforces this with `FATAL_ERROR` if missing) and `CMAKE_PROGRAM_PATH` (so cmake can find `clang.exe` outside the ROCm devel tree).

**Linux (no overrides):**
```bash
cd $REPO_ROOT
cmake --preset <preset>
```

**Linux (with ROCM_PATH override):**
```bash
cd $REPO_ROOT
cmake --preset <preset> -DROCM_PATH="$ROCM_PATH"
```

If `GPU_TARGETS` was provided on Linux, append `-DGPU_TARGETS="$GPU_TARGETS"`.

**Windows (using paths from Step 1b):**
```bash
cd $REPO_ROOT
cmake --preset <preset> -DROCM_PATH="$ROCM_PATH" -DCMAKE_PROGRAM_PATH="$CLANG_PATH" -DGPU_TARGETS="$GPU_TARGETS"
```

## Step 4: Build

If a `jobs` value was specified, pass it via `-j`. Otherwise let ninja auto-detect.

Run the build with stdout/stderr captured to a log; tail only on failure and propagate the exit code so PASS/FAIL is detectable from the bash exit status. Don't pipe directly to `tail` — bash without `pipefail` would mask the failing exit code.

With explicit jobs:
```bash
LOG=$(mktemp)
cmake --build $BUILD_DIR -- -j <jobs> > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "BUILD FAILED (exit $RC). Full log: $LOG"; tail -200 "$LOG"; else rm -f "$LOG"; fi
exit $RC
```

Without explicit jobs:
```bash
LOG=$(mktemp)
cmake --build $BUILD_DIR > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "BUILD FAILED (exit $RC). Full log: $LOG"; tail -200 "$LOG"; else rm -f "$LOG"; fi
exit $RC
```

### Stale Cache Auto-Recovery

If the build fails with `does not match the source` (flatbuffers cache mismatch — common after rebasing or merging), automatically recover:

1. Detect the pattern in the failure tail (or `grep` the kept `$LOG` if the tail truncated it).
2. Remove the build directory: `rm -rf $BUILD_DIR`
3. Re-run Step 3 (configure) and Step 4 (build) once more.

If the second attempt also fails, surface the error and stop — don't loop.

## Step 5: Report

Summarize:
- Preset used and components built
- Build result (success/failure)
- Build output directory: `$BUILD_DIR/`
- For Windows: the ROCm and Clang paths used

To run tests, invoke `/hipdnn-superbuild-test`.

## Common Issues

- **Stale cache after rebase** — Step 4 auto-recovers from the most common pattern. If a different cache failure appears, use `clean` argument or `rm -rf $BUILD_DIR` and rebuild.
- **Windows ROCM_PATH** — The `rocm-clang.cmake` toolchain file enforces `ROCM_PATH` on Windows with a `FATAL_ERROR` if missing.
- **Windows CMAKE_PROGRAM_PATH** — Required separately from `ROCM_PATH` since clang lives in its own toolchain location, not under the ROCm devel tree.
- **Windows: wheel install failed** — Check the helper's stderr output for network errors when fetching ROCm wheels. Re-run with `SHA=<commit>` to pin to a known good build.
- **Missing provider dependencies** — Some providers require additional ROCm libraries (MIOpen, hipBLASLt) to be installed.

## Related Skills

- `/hipdnn-superbuild-test` — Run tests against this superbuild
- `/hipdnn-build` — Build hipDNN standalone (without providers)
- `/hipdnn-build-test` — Full standalone build + test + install workflow
- `/hipdnn-samples` — Build and run sample applications
