---
name: hipdnn-superbuild-test
description: Run tests against an existing hipDNN superbuild. Supports per-component selection (hipdnn, miopen-provider, hipblaslt-provider, hip-kernel-provider, integration-tests), unit/integration scope, and gtest filtering. Handles Windows DLL PATH automatically.
argument-hint: "[component: hipdnn|miopen|hipblaslt|hip-kernel|integration-tests|all] [scope: unit|integration|all] [jobs=<N>] [ROCM_PATH=<path>] [--filter=<gtest_pattern>] [--verbose] [--keep-going]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Superbuild Test Runner

Run tests against an existing superbuild produced by `/hipdnn-superbuild`. Each component built into the superbuild registers its own ctest targets prefixed with the component name (e.g., `hipdnn-unit-check`, `miopen-provider-unit-check`).

> Requires a prior `/hipdnn-superbuild` run. This skill does not configure or build — it only runs tests.

## Arguments

- **Component** (default: `all`):
  - `hipdnn` — only hipDNN tests
  - `miopen` — only miopen-provider tests
  - `hipblaslt` — only hipblaslt-provider tests
  - `hip-kernel` — only hip-kernel-provider tests
  - `integration-tests` — the hipdnn-integration-tests component
  - `all` — every component that has test targets in the build
- **Scope** (default: `unit`):
  - `unit` — fast tests, no GPU required
  - `integration` — GPU required
  - `all` — both unit and integration
- `jobs=<N>` — Parallel job count for test execution (passed via `-j`)
- `ROCM_PATH=<path>` — Override ROCm SDK devel path. Auto-discovered via the wheel setup helper if unset on Windows; defaults to `/opt/rocm` on Linux. Threaded into the test process env as `ROCM_PATH` so providers that JIT-compile kernels via hiprtc (e.g. `hip-kernel-provider`) can find the HIP headers.
- `--filter=<gtest_pattern>` — Run only matching tests via gtest_filter (forces direct binary execution)
- `--verbose` — Use the `-verbose` ctest target variants
- `--keep-going` — When `component: all`, continue running remaining components after one fails (rather than stopping)

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
- `BIN_DIR` = `$REPO_ROOT/build/bin`
- `HELPERS` = `$REPO_ROOT/projects/hipdnn/tools/ai/skills/helpers`

## Step 1: Parse Arguments

Parse `$ARGUMENTS` for:
- **Component**: `hipdnn`, `miopen`, `hipblaslt`, `hip-kernel`, `integration-tests`, or `all` (default: `all`)
- **Scope**: `unit`, `integration`, or `all` (default: `unit`)
- **Jobs**: value after `jobs=`
- **ROCM_PATH**: value after `ROCM_PATH=`
- **Filter**: value after `--filter=`
- **Verbose**: presence of `--verbose`
- **Keep going**: presence of `--keep-going`

## Step 2: Verify Superbuild Exists

```bash
ls $BUILD_DIR/build.ninja
```

If missing: tell the user to run `/hipdnn-superbuild [preset]` first and stop.

## Step 3: Resolve ROCm Path (Windows only)

Windows test execution needs the ROCm bin on PATH for DLL loading. Use the shared helper to locate it:

```bash
python $HELPERS/windows_rocm_setup.py --repo-root $REPO_ROOT [--rocm-path $ROCM_PATH]
```

Parse the `ROCM_PATH=...` line from stdout. Set `ROCM_BIN = $ROCM_PATH/bin`. On Linux, skip this step.

## Step 4: Discover Targets

Use the discovery helper, which knows about the hip-kernel-provider naming inconsistency and falls back to its path-qualified target automatically:

```bash
python $HELPERS/discover_test_targets.py --build-dir $BUILD_DIR --component <component> --scope <scope>
```

The helper outputs one `<component>:<target>` line per match. If `--filter` was provided, jump to Step 5b. Otherwise continue to Step 5a with this list.

## Step 5: Run Tests

### Step 5a: Run via cmake target (helper-wrapped)

For each `component:target` line from Step 4, run the shared cmake build helper. It runs `cmake --build … --target …` with `ROCM_PATH` set in the test process env, and on Windows also prepends `<build>/bin` and the ROCm bin directory to `PATH` so the test executables ctest spawns can resolve their DLLs. The env block propagates through cmake → ninja → ctest → test.exe via `CreateProcessW`, so no PowerShell wrapper is needed.

Run the helper with stdout/stderr captured to a log, tail only on failure, and propagate the exit code so PASS/FAIL is detectable from the bash exit status (don't rely on parsing tail output — a segfault or `0xc0000135` DLL-load crash may produce no "FAIL" string):

```bash
LOG=$(mktemp)
python $HELPERS/cmake_run.py --build-dir $BUILD_DIR --target <target> [--jobs <N>] [--rocm-path $ROCM_PATH] [--rocm-bin $ROCM_BIN] > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "FAILED (exit $RC). Full log: $LOG"; tail -200 "$LOG"; else rm -f "$LOG"; fi
exit $RC
```

`--rocm-path` is optional: the helper defaults to `/opt/rocm` on Linux and to `--rocm-bin`'s parent on Windows. Pass it explicitly only when the user supplied `ROCM_PATH=`. Pass `--rocm-bin $ROCM_BIN` only on Windows.

If `--verbose` was provided, append `-verbose` to the target name (e.g., `hipdnn-unit-check-verbose`). For path-qualified targets like `dnn-providers/hip-kernel-provider/src/unit-check`, use `dnn-providers/hip-kernel-provider/src/unit-check-verbose`.

Run targets sequentially in separate Bash invocations (do not chain with `&&`). Track pass/fail per component.

If a component fails:
- If `--keep-going` was provided, record the failure and continue with the remaining components.
- Otherwise stop and report.

### Step 5b: Run via direct binary with --filter

`--filter` requires running the test binary directly. List available binaries:

```bash
ls $BIN_DIR/*tests*.exe 2>/dev/null
```

(On Linux, omit `.exe`.)

Component → binary mapping:

| Component | Unit Binaries | Integration Binaries |
|-----------|---------------|---------------------|
| `hipdnn` | `hipdnn_backend_tests`, `hipdnn_frontend_tests`, `hipdnn_data_sdk_tests`, `hipdnn_flatbuffers_sdk_tests`, `hipdnn_plugin_sdk_tests`, `hipdnn_test_sdk_tests` | `hipdnn_public_backend_tests`, `hipdnn_public_frontend_tests`, `hipdnn_backend_logging_shutdown_tests` |
| `miopen` | `miopen_plugin_tests` | `miopen_plugin_integration_tests` |
| `hipblaslt` | `hipblaslt_plugin_tests` | `hipblaslt_plugin_integration_tests` |
| `hip-kernel` | `hip_kernel_provider_tests` | `hip_kernel_provider_integration_tests` |
| `integration-tests` | `hipdnn_integration_tests_unit_tests` | `hipdnn_integration_tests`, `hipdnn_gpu_ref_tests` |

Use the same helper in `--binary` mode — it sets `PATH`/`ROCM_PATH` the same way and avoids any shell-quoting concerns with paths containing spaces or special characters:

```bash
LOG=$(mktemp)
python $HELPERS/cmake_run.py --build-dir $BUILD_DIR --binary $BIN_DIR/<binary>[.exe] --gtest-filter "<filter>" [--rocm-path $ROCM_PATH] [--rocm-bin $ROCM_BIN] > "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo "FAILED (exit $RC). Full log: $LOG"; tail -200 "$LOG"; else rm -f "$LOG"; fi
exit $RC
```

## Step 6: Report

Summarize per-component test results. If `--keep-going` was used and some failed, list each failure separately. Example:

```
hipdnn:
  hipdnn-unit-check:                 PASS  6/6 binaries, 7150 tests, 12.45s
miopen-provider:
  miopen-provider-unit-check:        PASS  1/1 binaries, 2875 tests, 29.94s
hipblaslt-provider:
  hipblaslt-provider-unit-check:     PASS  1/1 binaries, 0.94s
hip-kernel-provider:
  dnn-providers/hip-kernel-provider/src/unit-check:  FAIL  2 tests:
    - TestRMSnormPlanBuilder.IsApplicableReturnsTrueForValidInferenceGraph
    - TestRMSnormValidator.Valid
```

If a component was requested but its targets are not in the build, say so explicitly (the discovery helper will return zero matches for that component).

## Notes

- **Helpers** under `$REPO_ROOT/projects/hipdnn/tools/ai/skills/helpers/`:
  - `windows_rocm_setup.py` — detects/installs the wheel ROCm and prints paths
  - `discover_test_targets.py` — finds available test targets, with hip-kernel fallback
  - `cmake_run.py` — runs `cmake --build <target>` or a test binary directly with `PATH`/`ROCM_PATH` set in the subprocess env
- **Windows DLL/PATH** — Handled by `cmake_run.py`: it prepends `<build>/bin` and the ROCm bin to `PATH` on Python's subprocess env, which `CreateProcessW` propagates to grandchildren. Do **not** try to set `PATH` from bash — MSYS2 doesn't translate it into the Win32 form ninja's `cmd.exe /C` invocations need.
- **Integration tests require an AMD GPU.** They will fail on machines without GPU access. For unit-only verification, use `scope: unit` (the default).
- **hip-kernel-provider target naming** — Auto-handled by the discovery helper. The provider currently uses unprefixed `unit-check` rather than `hip-kernel-provider-unit-check`; the helper finds it under `dnn-providers/hip-kernel-provider/src/unit-check`.
- **Stale build** — If the discovery helper returns no targets for a component, that component was not in the superbuild's preset. Re-run `/hipdnn-superbuild <preset>` with a preset that includes it.

## Related Skills

- `/hipdnn-superbuild` — Build hipDNN with providers
- `/hipdnn-test` — Test the standalone hipDNN build
