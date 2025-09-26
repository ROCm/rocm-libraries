# hipDNN Test Run Template

This document provides a template for recording hipDNN test results. Use this template to document test execution details and results for tracking purposes.

---

## Test Run Template

```
Name: 
Operating System: 
Has GPU: 
ASIC: 
MIOpen hash: 
ROCm version: 
Run environment: 
Test results: 
```

---

## Test Run Details

### Field Descriptions

- **Name**: Runner's name or identifier
- **Operating System**: OS details (Windows/Linux, and distro or version of Windows)
- **Has GPU**: Yes/No
- **ASIC**: Name of device after running `rocminfo` (or "N/A" if no GPU)
- **MIOpen hash**: Hash of installed MIOpen, can be found by running:
  ```bash
  cat /opt/rocm/include/miopen/version.h
  ```
  Or "N/A" if MIOpen is not installed
- **ROCm version**: AMD-SMI ROCm version info (or "N/A" if not applicable)
- **Run environment**: One of:
  - `ninja check_ctest`
  - `make check_ctest`
  - Running installed test artifacts
  - Running built executables directly
- **Test results**: Summary of tests ran, skipped, and passed/failed

---

## Example Test Run

### Successful Run Example

```
Name: John Doe
Operating System: Linux Ubuntu 22.04
Has GPU: Yes
ASIC: gfx90a
MIOpen hash: 7ae16d1b5
ROCm version: AMD-SMI 26.0.0+842b9680 amdgpu version: 6.8.5 ROCm version: 7.1.0
Run environment: ninja check_ctest

Test results:  
    (3 / 4) Test Cases Passed
    (1 / 4) Test Cases Failed
    (0 / 4) Test Cases did not run

Section: Prerequisites:
    Test Case 1: Passed
    Test Case 2: Failed
        There is a spelling mistake in the change log

Section: Regular Tests:
    Test Case 1: Passed
        Regular test results 519/519 tests passed, 0 failed out of 519, none skipped

Section: ASAN Enabled Tests:
    Test Case 1: Passed
        ASAN test results 100% passed.
```

### CPU-Only Environment Example

```
Name: Jane Smith
Operating System: Windows 11
Has GPU: No
ASIC: N/A
MIOpen hash: N/A
ROCm version: N/A
Run environment: ninja check_ctest

Test results:
    (4 / 4) Test Cases Passed
    (0 / 4) Test Cases Failed
    (0 / 4) Test Cases did not run

Section: Prerequisites:
    Test Case 1: Passed
    Test Case 2: Passed

Section: Regular Tests:
    Test Case 1: Passed
        Regular test results 100/519 tests passed, 0 failed, 419 skipped (GPU tests)

Section: ASAN Enabled Tests:
    Test Case 1: Passed
        ASAN test results 100/519 tests passed, 0 failed, 419 skipped (GPU tests)
```

### Skipped Tests Example

When tests are skipped (e.g., due to ASAN being enabled), document them as follows:

```
Skipped:
The following tests did not run:
43 - MigratableMemory.NotInitialized (Skipped)
44 - MigratableMemory.InitializeWithSize (Skipped)
45 - MigratableMemory.MoveConstructor (Skipped)
46 - MigratableMemory.MoveAssignment (Skipped)
47 - MigratableMemory.Resize (Skipped)
48 - MigratableMemory.MigrateToDevice (Skipped)
49 - MigratableMemory.MigrateToHost (Skipped)
50 - MigratableMemory.Clear (Skipped)
68 - TestTensor.BasicRowMajorUsage (Skipped)
69 - TestTensor.FillWithValuesUsage (Skipped)
70 - TestTensor.FillWithRandomValuesUsage (Skipped)
177 - GPU_EnginePluginTest.LoadPluginsAndExecuteOpGraph (Skipped)
178 - GPU_EnginePluginResourceManagerTest.LoadPluginsAndExecuteOpGraph (Skipped)
194 - GPU_HandleTests.SetAndGetStream (Skipped)
195 - GPU_HandleTests.SetStreamToNull (Skipped)
403 - GPU_hipDNNHandleAPITests.GetStreamPointer (Skipped)
404 - IntegrationTests/Frontend_e2e_integration_test.IntegrationTest/Default_plugin_with_manual_UIDs # GetParam() = BatchnormTestCase{plugin_path: ../test_plugins/libtest_good_plugin.so, description: Default plugin with manual UIDs, graph_name: DefaultPluginBatchnormTest, expected_failure: NONE, use_manual_uids: true} (Skipped)
405 - IntegrationTests/Frontend_e2e_integration_test.IntegrationTest/Default_plugin_with_auto_UIDs # GetParam() = BatchnormTestCase{plugin_path: ../test_plugins/libtest_good_plugin.so, description: Default plugin with auto UIDs, graph_name: DefaultPluginBatchnormTestAutoUID, expected_failure: NONE, use_manual_uids: false} (Skipped)
406 - IntegrationTests/Frontend_e2e_integration_test.IntegrationTest/Execute_fails_plugin # GetParam() = BatchnormTestCase{plugin_path: ../test_plugins/libtest_execute_fails_plugin.so, description: Execute fails plugin, graph_name: ExecuteFailsPluginBatchnormTest, expected_failure: EXECUTE, use_manual_uids: true} (Skipped)
407 - IntegrationTests/Frontend_e2e_integration_test.IntegrationTest/No_applicable_engines_plugin # GetParam() = BatchnormTestCase{plugin_path: ../test_plugins/libtest_no_applicable_engines_plugin.so, description: No applicable engines plugin, graph_name: NoEnginesPluginBatchnormTest, expected_failure: CREATE_EXECUTION_PLAN, use_manual_uids: true} (Skipped)
436 - MiopenLegacyEnginePluginApiTest.EnginePluginCreateAlsoCreatesMIOpenHandleOnSuccess (Skipped)
437 - MiopenLegacyEnginePluginApiTest.EnginePluginCreateTwiceGivesTheSameContainerHandle (Skipped)
438 - MiopenLegacyEnginePluginApiTest.EnginePluginCreateNonNullHandlePointer (Skipped)
441 - MiopenLegacyEnginePluginApiTest.EnginePluginSetStreamNullStream (Skipped)
442 - MiopenLegacyEnginePluginApiTest.EnginePluginSetStreamValidStream (Skipped)
444 - MiopenLegacyEnginePluginApiTest.EnginePluginGetApplicableEngineIdsValid (Skipped)
446 - MiopenLegacyEnginePluginApiTest.EnginePluginGetEngineDetailsValid (Skipped)
449 - MiopenLegacyEnginePluginApiTest.EnginePluginGetWorkspaceSizeValid (Skipped)
451 - MiopenLegacyEnginePluginApiTest.EnginePluginCreateExecutionContextValid (Skipped)
453 - Batchnorm_execute_graph_test.RunBfloat16FwdbatchnormGraph (Skipped)
454 - Batchnorm_execute_graph_test.RunHalfFwdbatchnormGraph (Skipped)
464 - MiopenHandleFactoryTest.CreatesAndDestroysHandle (Skipped)
484 - RunFwdbatchnormGraphWithParams/Batchnorm_execute_graph_test.RunFloatFwdbatchnormGraph/(n:1 c:3 h:14 w:14) (Skipped)
485 - RunFwdbatchnormGraphWithParams/Batchnorm_execute_graph_test.RunFloatFwdbatchnormGraph/(n:2 c:3 h:14 w:14) (Skipped)
487 - RunFloatFwdBatchnormGraph/Batchnorm_forward_inference_integration_test.RunFloatFwdBatchnormGraph/(n:1 c:3 h:14 w:14) (Skipped)
488 - RunFloatFwdBatchnormGraph/Batchnorm_forward_inference_integration_test.RunFloatFwdBatchnormGraph/(n:2 c:3 h:14 w:14) (Skipped)
...
```

---

## Best Practices

1. **Be Specific**: Document exact error messages when tests fail.
2. **Include Context**: Note any environmental factors that might affect test results.
3. **Track Patterns**: If certain tests consistently skip or fail, document why.
4. **Version Information**: Always include accurate version information for reproducibility.
5. **Save logs**: Saving and attaching logs lends critical insight into your run. See [Environment.md](../Environment.md#environment-variables) for details on enabling logging.