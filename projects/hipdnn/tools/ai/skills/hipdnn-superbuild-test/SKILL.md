---
name: hipdnn-superbuild-test
description: Run tests against an existing hipDNN superbuild. Supports per-component selection (hipdnn, miopen-provider, hipblaslt-provider, hip-kernel-provider, integration-tests), unit/integration scope, and gtest filtering. Handles Windows DLL PATH automatically.
argument-hint: "[component: hipdnn|miopen|hipblaslt|hip-kernel|integration-tests|all] [scope: unit|integration|all] [ROCM_PATH=<path>] [--filter=<gtest_pattern>] [--verbose] [--keep-going]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Superbuild Test Runner

Use this skill when the user asks to test an existing hipDNN superbuild. It does not configure or build the project. If no superbuild exists, tell the user to build first with `hipdnn-superbuild`.

## Inputs

Infer options from the user request:

- **Component**: `hipdnn`, `miopen`, `hipblaslt`, `hip-kernel`, `integration-tests`, or `all`; default `all`
- **Scope**: `unit`, `integration`, or `all`; default `unit`
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
   - Helper scripts: installed skill `<skill-directory>/scripts`, or source fallback `<repo-root>/projects/hipdnn/tools/ai/skills/hipdnn-superbuild-test/scripts`.

3. Verify the superbuild exists:
   ```bash
   ls <build-dir>/build.ninja
   ```
   Stop if it is missing.

4. Resolve ROCm path on Windows:
   ```bash
   python3 <scripts>/windows_rocm_setup.py --repo-root <repo-root> [--rocm-path <path>]
   ```
   Parse `ROCM_PATH=...` from stdout and set `ROCM_BIN=<rocm-path>/bin`. Skip this step on Linux unless the user supplied an override.

5. Discover CMake test targets:
   ```bash
   python3 <scripts>/discover_test_targets.py --build-dir <build-dir> --component <component> --scope <scope>
   ```
   The helper prints `<component>:<target>` lines. It also handles the hip-kernel-provider path-qualified target naming.
   If the helper reports that Ninja target discovery failed, treat that as an invalid or stale build directory and stop with the helper's diagnostic. If discovery succeeds but no targets match, report that the requested component or scope is not present in the existing superbuild.

6. Run tests through `cmake_run.py` when no gtest filter is requested:
   ```bash
   python3 <scripts>/cmake_run.py --build-dir <build-dir> --target <target> [--rocm-path <path>] [--rocm-bin <path>] > <log> 2>&1
   ```
   Add `--jobs <N>` only when explicit jobs are both requested and permitted. For verbose mode, append `-verbose` to the target name.

7. Run direct binaries when a gtest filter is requested:
   ```bash
   python3 <scripts>/cmake_run.py --build-dir <build-dir> --binary <binary-path> --gtest-filter "<filter>" [--rocm-path <path>] [--rocm-bin <path>] > <log> 2>&1
   ```
   Use the component-to-binary mapping below to choose binaries.

8. For every command, keep full output in a log and show only a short tail on failure. Track pass/fail per component. Stop at the first failure unless keep-going was requested.

## Direct Binary Mapping

| Component | Unit Binaries | Integration Binaries |
|-----------|---------------|----------------------|
| `hipdnn` | `hipdnn_backend_tests`, `hipdnn_frontend_tests`, `hipdnn_data_sdk_tests`, `hipdnn_flatbuffers_sdk_tests`, `hipdnn_plugin_sdk_tests`, `hipdnn_test_sdk_tests` | `hipdnn_public_backend_tests`, `hipdnn_public_frontend_tests`, `hipdnn_backend_logging_shutdown_tests` |
| `miopen` | `miopen_plugin_tests` | `miopen_plugin_integration_tests` |
| `hipblaslt` | `hipblaslt_plugin_tests` | `hipblaslt_plugin_integration_tests` |
| `hip-kernel` | `hip_kernel_provider_tests` | `hip_kernel_provider_integration_tests` |
| `integration-tests` | `hipdnn_integration_tests_unit_tests` | `hipdnn_integration_tests`, `hipdnn_gpu_ref_tests` |

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

- `scripts/cmake_run.py`, `scripts/discover_test_targets.py`, and `scripts/windows_rocm_setup.py` are bundled in this skill so linked and copied installs work independently.
- Windows DLL loading is handled by `cmake_run.py`, which sets PATH in Python's subprocess environment before launching CMake or test binaries.
- Integration tests require an AMD GPU. Unit scope is the default for CPU-only validation.
