# Plan: Replace JSON Config with Command-Line Arguments

## Overview

Replace the `HIPDNN_TEST_CONFIG_PATH` environment variable + JSON config file in `dnn-providers/integration-tests` with two command-line arguments:

- `--test-article <path>`: Full path to the hipdnn engine plugin `.so` to test
- `--test-engine <name>`: Engine name string to test against (e.g., `MIOPEN_ENGINE`)

## New Invocation

```bash
hipdnn_integration_tests \
    --test-article /path/to/libmiopen_provider_plugin.so \
    --test-engine MIOPEN_ENGINE \
    --gtest_filter="*ConvFwd*"   # optional, still works
```

## Current State

| Concern | Current mechanism |
|---------|-------------------|
| Which plugin to load | External (`HIPDNN_PLUGIN_PATH` env var or system paths) |
| Which engines to test | All engines that claim graph support, discovered at static init via `BuildEngineTestMatrix` |
| Expected failures | `expected_failures` array in JSON config |
| Tolerance mode | `engines.<name>.tolerance` in JSON config (only `Default` exists) |
| Plugin verification | `plugins.<name>.name` in JSON config, compared against loaded plugin names in `main()` |

**Static init flow**: `INSTANTIATE_TEST_SUITE_P` → `BuildEngineTestMatrix` → `getSharedHandle()` → `hipdnnCreate()` — all before `main()`.

## Design

### Key Insight

Instead of generating per-engine test cases at static init (which requires the handle/plugins to be available before `main()`), parameterize tests **only by test case data** (convolution params, batchnorm params, etc.). At test runtime, build the graph, check if `--test-engine` supports it, and `GTEST_SKIP()` if not.

This eliminates the static-init handle creation and makes CLI arg parsing in `main()` straightforward.

### What goes away

- `BuildEngineTestMatrix` and the `EngineTestCase<T>` wrapper struct
- `EngineTestNameGenerator` (engine name no longer in test params)
- `EngineTestMatrix.hpp` (can be deleted)
- JSON config file parsing, `nlohmann_json` dependency
- `HIPDNN_TEST_CONFIG_PATH` environment variable
- `_expectedFailures` / `isExpectedFailure()` (use `--gtest_filter` negative patterns instead)
- `_expectedPluginNames` / `getExpectedPluginNames()` and the plugin verification block in `main()`
- Static-init handle creation via `getSharedHandle()` in `BuildEngineTestMatrix`

### GTest Filter Compatibility

Fully compatible:

1. `::testing::InitGoogleTest(&argc, argv)` is called first, consumes `--gtest_*` flags
2. Our custom args (`--test-article`, `--test-engine`) are parsed from the remaining `argv`
3. `INSTANTIATE_TEST_SUITE_P` now uses only static test case data — no handle, no engine query
4. At test runtime, unsupported graphs get `GTEST_SKIP()`
5. `--gtest_filter` narrows which tests run, `GTEST_SKIP` further skips unsupported ones
6. `--gtest_list_tests`, `--gtest_repeat`, `--gtest_shuffle` all work normally

---

## Implementation Phases

### Phase 1: Refactor `TestConfig.hpp`

Replace JSON config singleton with CLI-arg-based configuration.

**File: `src/harness/TestConfig.hpp`**

- Remove: `#include <nlohmann/json.hpp>`, `#include <fstream>`, `#include <cstdlib>`, `#include <cstring>`
- Remove: `_expectedFailures`, `_expectedPluginNames`, `isExpectedFailure()`, `getExpectedPluginNames()`
- Remove: JSON parsing constructor (env var, file open, JSON parse)
- Add: `_articlePath` (`std::filesystem::path`) and `_engineName` (`std::string`) members
- Add: `static void initialize(std::filesystem::path articlePath, std::string engineName)` — called from `main()`, stores values in the singleton. Throws if called more than once or if the singleton was already accessed uninitialized.
- Add: `getArticlePath()` and `getEngineName()` accessors
- Add: `getEngineId()` — returns `hipdnn_data_sdk::utilities::engineNameToId(_engineName)`
- Keep: `getToleranceMode()` — simplify to always return `ToleranceMode::Default` (the only mode that exists). Accept `std::string_view engineName` instead of `int64_t engineId` to avoid the ID-to-name lookup.
- Constructor becomes private default constructor; actual init via `initialize()`

### Phase 2: Refactor `main.cpp`

**File: `src/main.cpp`**

**Arg parsing** (before `InitGoogleTest`):

Scan `argv` for `--test-article` and `--test-engine`, extract their values, then remove them from `argv`/`argc` before calling `InitGoogleTest`. This prevents gtest from warning about unknown flags.

```
// Pseudocode
articlePath = extractArg(argc, argv, "--test-article");  // returns value, removes from argv
engineName  = extractArg(argc, argv, "--test-engine");   // returns value, removes from argv
if (!articlePath || !engineName) {
    printUsage();
    return 1;
}
```

**Plugin setup** (after arg parsing, before `InitGoogleTest`):

```
// Set plugin path to the directory containing the article
hipdnnSetEnginePluginPaths_ext(1, &articleDir, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
TestConfig::initialize(articlePath, engineName);
```

**Handle creation** (after `InitGoogleTest`):

The shared handle is now created explicitly in `main()` rather than lazily during static init. `getSharedHandle()` remains lazy but is first triggered here.

**Engine validation** (after handle creation):

Verify the target engine is among the loaded engines using `hipdnnGetEngineInfo_ext`. Print a clear error if not found. This replaces the old plugin-name verification.

**Remove**: `getLoadedPluginNames()`, `formatPluginSet()`, the plugin mismatch check.

### Phase 3: Simplify test parameterization in test files

**Files: `src/IntegrationGpuConvForward.cpp`, `src/IntegrationGpuBatchnormBackwardActivation.cpp`**

Change fixture base from `TestWithParam<EngineTestCase<TestCase>>` to `TestWithParam<TestCase>`:

- `GetParam()` now returns the test case directly (e.g., `ConvFwdTestCase`), not wrapped in `EngineTestCase`
- `param.testCase` → `this->GetParam()`
- `param.engineId` → `TestConfig::get().getEngineId()`

Change `INSTANTIATE_TEST_SUITE_P` to use the test case data directly instead of `BuildEngineTestMatrix`:

```cpp
// Before:
INSTANTIATE_TEST_SUITE_P(
    Smoke, IntegrationGpuConvFwd2dFp32,
    testing::ValuesIn(BuildEngineTestMatrix<IntegrationGpuConvFwd2dFp32, ConvFwdTestCase>(
        testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                         testing::ValuesIn(test_conv_common::getConvTestCases4D())))),
    EngineTestNameGenerator<ConvFwdTestCase>);

// After:
INSTANTIATE_TEST_SUITE_P(
    Smoke, IntegrationGpuConvFwd2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));
```

No custom name generator needed — gtest's default parameterized test naming is sufficient, or we can add a simple one based on test case data.

### Phase 4: Add engine support check to test harness

**File: `src/harness/IntegrationGraphVerificationHarness.hpp`**

In `runGraphTest()` (or a new helper called from each test's `runGraphTest()`), after building the graph and operation graph, check if the target engine supports it:

```cpp
// After build_operation_graph:
std::vector<int64_t> engineIds;
auto status = graph.get_ranked_engine_ids(engineIds);
int64_t targetEngineId = TestConfig::get().getEngineId();

if (status.is_bad() ||
    std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end()) {
    GTEST_SKIP() << "Engine " << TestConfig::get().getEngineName()
                 << " does not support this graph";
}

graph.set_preferred_engine_id_ext(targetEngineId);
```

Remove the `isExpectedFailure` check in `SetUp()`.

The `getTolerance()` method simplifies — it no longer needs an `engineId` parameter since there's only one engine under test, and the tolerance mode is always `Default`.

### Phase 5: Delete `EngineTestMatrix.hpp`

**File: `src/harness/EngineTestMatrix.hpp`** — delete entirely.

- `BuildEngineTestMatrix` is no longer used
- `EngineTestCase<T>` is no longer used
- `EngineTestNameGenerator` is no longer used

### Phase 6: Update `CMakeLists.txt`

**File: `CMakeLists.txt`**

- Remove `find_package(nlohmann_json REQUIRED)`
- Remove `nlohmann_json::nlohmann_json` from `target_link_libraries`
- Update the installed CTest file to pass required args (or document that args are required)

---

## File Change Summary

| File | Action |
|------|--------|
| `src/harness/TestConfig.hpp` | Rewrite: CLI args instead of JSON |
| `src/main.cpp` | Rewrite: arg parsing, plugin setup, remove plugin verification |
| `src/IntegrationGpuConvForward.cpp` | Modify: simplify parameterization, remove `EngineTestCase` wrapping |
| `src/IntegrationGpuBatchnormBackwardActivation.cpp` | Modify: simplify parameterization, remove `EngineTestCase` wrapping |
| `src/harness/IntegrationGraphVerificationHarness.hpp` | Modify: add engine support skip, remove expected failure check, simplify tolerance |
| `src/harness/EngineTestMatrix.hpp` | Delete |
| `CMakeLists.txt` | Modify: remove nlohmann_json |
| `src/harness/SharedHandle.hpp` | No change (lazy init still works, first call now in `main()`) |
| `src/common/*.hpp` | No change |

## Testing

1. Build integration tests with `/hipdnn-build`
2. Run with valid args: `--test-article /path/to/plugin.so --test-engine MIOPEN_ENGINE`
3. Verify only supported graphs run, unsupported ones show `GTEST_SKIP`
4. Run with `--gtest_filter="*ConvFwd*"` — verify filtering works
5. Run with `--gtest_list_tests` — verify test list is correct
6. Run without args — verify clear usage message
7. Run with invalid article path — verify clear error
8. Run with wrong engine name — verify clear error
