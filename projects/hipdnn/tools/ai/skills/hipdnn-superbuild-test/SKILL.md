---
name: hipdnn-superbuild-test
description: Run tests against an existing hipDNN superbuild. Supports per-component selection (hipdnn, miopen-provider, hipblaslt-provider, hip-kernel-provider, integration-tests), unit/integration/external-integration scope, and gtest filtering. Reproduces the cross-provider external-integration-check suite. Handles Windows DLL PATH automatically.
argument-hint: "[component: hipdnn|miopen|hipblaslt|hip-kernel|integration-tests|all] [scope: unit|integration|external-integration|all] [ROCM_PATH=<path>] [--filter=<gtest_pattern>] [--verbose] [--keep-going]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Superbuild Test Runner

Use this skill when the user asks to test an existing hipDNN superbuild. It does not configure or build the project. If no superbuild exists, tell the user to build first with `hipdnn-superbuild`.

## Inputs

Infer options from the user request:

- **Component**: `hipdnn`, `miopen`, `hipblaslt`, `hip-kernel`, `integration-tests`, or `all`; default `all`
- **Scope**: `unit`, `integration`, `external-integration`, or `all`; default `unit`. `external-integration` covers the cross-provider `hipdnn_integration_tests` suite (the `<provider>-external-integration-check` targets)
- **Filter**: optional gtest filter; when present, run test binaries directly
- **Verbose**: use verbose test targets when requested
- **Keep going**: continue after failures only when requested
- **ROCm path**: optional `ROCM_PATH=<path>` override; Linux defaults to `/opt/rocm`
- **Jobs**: optional explicit parallelism only when the user requests it and active workspace instructions permit it

## Workflow

1. Determine the repository root:
   ```bash
   git rev-parse --show-toplevel
   ```

2. Resolve paths:
   - Build directory: honor active workspace instructions first; otherwise use `<repo-root>/build`.
   - Binary directory: `<build-dir>/bin`.
   - Helper scripts: skills are host-level, not tied to a repo checkout — **default to the scripts bundled with the skill you were invoked from** (`<skill-directory>/scripts`), even when working inside a repo or worktree. Do NOT run the `<repo-root>/projects/hipdnn/tools/ai/skills/hipdnn-superbuild-test/scripts` copy just because a checkout is present: it can be a stale stub (on `develop`) or an unmerged in-progress version (on a feature branch). Use the source-checkout copy only when actively developing this skill itself to exercise your in-progress edits, or when the invoked skill has no bundled `scripts/` directory.

3. Verify the superbuild exists:
   ```bash
   ls <build-dir>/build.ninja
   ```
   Stop if it is missing.

4. Resolve ROCm path on Windows:
   ```bash
   python3 <scripts>/windows_rocm_setup.py --repo-root <repo-root> [--rocm-path <path>]
   ```
   Parse `ROCM_PATH=...` from stdout and set `ROCM_BIN=<rocm-path>/bin`. Skip this step on Linux unless the user supplied an override. On Windows, always pass the resolved `ROCM_BIN` to `cmake_run.py` (steps 6-8) via `--rocm-bin`: it is required both for the runtime PATH and for staging the wheel's `amd_comgr.dll` app-local (see Notes).

5. Discover CMake test targets:
   ```bash
   python3 <scripts>/discover_test_targets.py --build-dir <build-dir> --component <component> --scope <scope>
   ```
   The helper prints `<component>:<target>` lines. It also handles the hip-kernel-provider path-qualified target naming. With `--scope external-integration` (or `all`) it also emits a `<component>:command:<cmdline>` line — the resolved cross-provider `hipdnn_integration_tests` invocation (with `--test-article`/`--test-engine`/`--test-config`) read from the generated `CTestTestfile.cmake`, with any baked-in `--gtest_filter` stripped so you can supply your own.
   If the helper reports that Ninja target discovery failed, treat that as an invalid or stale build directory and stop with the helper's diagnostic. If discovery succeeds but no targets match, report that the requested component or scope is not present in the existing superbuild.

6. Run tests through `cmake_run.py` when no gtest filter is requested:
   ```bash
   python3 <scripts>/cmake_run.py --build-dir <build-dir> --target <target> [--rocm-path <path>] [--rocm-bin <path>] > <log> 2>&1
   ```
   Add `--jobs <N>` only when explicit jobs are both requested and permitted. For verbose mode, append `-verbose` to the target name.

7. Run direct binaries when a gtest filter is requested:
   ```bash
   python3 <scripts>/cmake_run.py --build-dir <build-dir> --binary <binary-path> --gtest-filter "<filter>" [--extra-arg=<flag> ...] [-- <passthrough args>] [--rocm-path <path>] [--rocm-bin <path>] > <log> 2>&1
   ```
   Use the component-to-binary mapping below to choose binaries. `cmake_run.py` accepts arbitrary passthrough flags for the binary: simple values via repeatable `--extra-arg` (use `--extra-arg=--flag` for flag-like values), or an entire flag list after a literal `--`. Passing multiple tokens inside `--binary` is rejected with a clear error.

8. Reproduce the cross-provider external-integration suite (`--scope external-integration`):
   - To run the whole suite exactly as CI does, build the custom target:
     ```bash
     python3 <scripts>/cmake_run.py --build-dir <build-dir> --target <provider>-external-integration-check [--rocm-path <path>] [--rocm-bin <path>] > <log> 2>&1
     ```
   - To run with a custom gtest filter, take the `<component>:command:<cmdline>` line from step 5, run the first token as `--binary` and the rest after `--`, adding your own `--gtest-filter`:
     ```bash
     python3 <scripts>/cmake_run.py --build-dir <build-dir> --binary <hipdnn_integration_tests> -- <--test-article ... --test-engine ... --test-config ...> --gtest_filter=<filter> > <log> 2>&1
     ```

9. For every command, keep full output in a log and show only a short tail on failure. Track pass/fail per component. Stop at the first failure unless keep-going was requested.

## Direct Binary Mapping

| Component | Unit Binaries | Integration Binaries | External Integration (cross-provider) |
|-----------|---------------|----------------------|----------------------------------------|
| `hipdnn` | `hipdnn_backend_tests`, `hipdnn_frontend_tests`, `hipdnn_data_sdk_tests`, `hipdnn_flatbuffers_sdk_tests`, `hipdnn_plugin_sdk_tests`, `hipdnn_test_sdk_tests` | `hipdnn_public_backend_tests`, `hipdnn_public_frontend_tests`, `hipdnn_backend_logging_shutdown_tests` | — |
| `miopen` | `miopen_plugin_tests` | `miopen_plugin_integration_tests` | `miopen-provider-external-integration-check` (`hipdnn_integration_tests` + `miopen_plugin`, engine `MIOPEN_ENGINE`) |
| `hipblaslt` | `hipblaslt_plugin_tests` | `hipblaslt_plugin_integration_tests` | `hipblaslt-provider-external-integration-check` (`hipdnn_integration_tests` + `hipblaslt_plugin`, engine `HIPBLASLT_ENGINE`) |
| `hip-kernel` | `hip_kernel_provider_tests` | `hip_kernel_provider_integration_tests` | `hip-kernel-provider-external-integration-check` when present |
| `integration-tests` | `hipdnn_integration_tests_unit_tests` | `hipdnn_integration_tests`, `hipdnn_gpu_ref_tests` | — |

The exact article/engine/config for the external suite is resolved at build time; get the ready-to-run command from `discover_test_targets.py --scope external-integration` (the `command:` line) rather than hardcoding paths.

## Report

Summarize per-component results:

```text
hipdnn:
  hipdnn-unit-check: PASS
miopen-provider:
  miopen-provider-unit-check: FAIL (see <log>)
```

If a requested component has no matching target, say that it was not present in the existing superbuild and name the preset or component likely needed.

## Notes

- `scripts/cmake_run.py`, `scripts/discover_test_targets.py`, `scripts/windows_rocm_setup.py`, and `scripts/comgr_stage.py` are bundled in this skill so linked and copied installs work independently.
- Windows DLL loading is handled by `cmake_run.py`, which sets PATH in Python's subprocess environment before launching CMake or test binaries.
- Windows comgr staging: before launching any target or binary on Windows, `cmake_run.py` stages the wheel's `amd_comgr.dll` into `<build-dir>/bin` (via `comgr_stage.py`) so MIOpen's runtime JIT does not load the driver's stale `System32` comgr. This happens on every Windows run, not just for a specific kernel path; GCN-assembly Winograd solvers are the common failure (`[BuildAsm] comgr status = ERROR` / `unknown emulation: no-xnack`), but the version mismatch is not limited to them. This needs `--rocm-bin` to be passed. The copy is skipped when the staged comgr already matches the wheel's PE version, so it adds no cost on repeat runs. Disable with `--no-stage-comgr` if ever needed. To confirm which comgr loaded, run a test with `MIOPEN_LOG_LEVEL=7 MIOPEN_ENABLE_LOGGING=1` and grep for `COMgr v.` (a low version indicates the stale System32 copy; the wheel's is newer).
- Integration tests require an AMD GPU. Unit scope is the default for CPU-only validation.
- Prefer running test binaries through `cmake_run.py` (it wires PATH/ROCM_PATH for the loader); pass extra binary flags via `--extra-arg`/`-- <args>` rather than folding them into `--binary`.
