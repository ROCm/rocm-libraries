# RFC 0006: Plugin-Agnostic Integration Tests

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Design Goals](#design-goals)
4. [Proposed Solution](#proposed-solution)
5. [Implementation Details](#implementation-details)
   - [IntegrationGraphVerificationHarness](#integrationgraphverificationharness-base-class)
   - [Test Fixture Convention](#test-fixture-convention)
   - [BuildEngineTestMatrix Function](#buildenginetestmatrix-function)
   - [GTest Test Discovery](#aside-gtest-test-discovery)
   - [Configuration System](#configuration-system)
6. [TheRock CI Integration](#therock-ci-integration)
   - [Test Configurations](#test-configurations)
   - [Managing CI Time Budgets](#managing-ci-time-budgets)
7. [Adding New Tests](#adding-new-tests)

## Executive Summary

This RFC documents a standalone integration test project for hipDNN enabling plugin-agnostic testing with runtime capability discovery.
The test infrastructure automatically defines a cartesian product of (engine, graph) pairs for numerical integration tests, giving each plugin an automatic level of test coverage "out of the box."

### Key Benefits
- **Reduced Duplication**: Eliminates near-identical test suites in each plugin
- **Plugin Independence**: Tests have no compile-time dependency on any specific plugin
- **Forward-Deployed Support**: Plugins built outside TheRock (e.g., fusilli/IREE) work seamlessly

## Problem Statement

Each plugin (MIOpen, fusilli, etc.) needs numerical integration tests for all graph variants it claims to support - preferably in a way that gives strong signal when debugging. Today, nearly identical tests exist in multiple locations: [miopen conv test](https://github.com/ROCm/rocm-libraries/blob/9924d667c218d067403608d77f432dd13585a848/dnn-providers/miopen-provider/integration_tests/IntegrationGpuConvForward.cpp) vs [fusilli conv test](https://github.com/iree-org/fusilli/blob/9328342731374ef7113f1f95c18a986c1df3739d/plugins/hipdnn-plugin/test/integration/convolution/conv_fprop_parameterized_full.cpp).

## Design Goals

1. **Plugin Independence**: No compile-time dependencies on specific plugins, the tests build only on core hipDNN; tests discover plugin capabilities at runtime
2. **Forward deploy / ThePebble Compatibility**: Tests can run standalone with forward-deployed plugins
    - Fusilli plugin (and presumably future forward deployed plugins) builds outside of `TheRock`. Currently Fusilli plugin maintains a hacky simulacrum of `TheRock` build ([`ThePebble.py`](https://github.com/iree-org/fusilli/blob/9328342731374ef7113f1f95c18a986c1df3739d/plugins/hipdnn-plugin/build_tools/ThePebble.py)) that pulls down some pre-built artifacts and sets up a local build roughly as it would be if fusilli-plugin was building as part of `TheRock`. Ideally, fusilli should be able to use the test suite by simply downloading pre-built artifacts from `TheRock` distribution.
3. **Build configurations for tests**: Build configurations for tests may not build with every available plugin, the integration suite should gracefully handle running a subset of total possible tests. For example: a change in the MIOpen plugin should not require building fusilli plugin or running fusilli plugin tests.
4. **Maximize signal within time budget**: As the suite grows, not all tests can run in every CI stage. Tests must be categorized by execution time so that CI can run the highest-signal tests within time budgets available in each stage (pre-submit, post-submit, nightly).

## Proposed Solution

### Overview

Most of the testing infrastructure can be taken directly from existing MIOpen integration test suite:

**IntegrationGraphVerificationHarness**: Base class providing GTest support and CPU reference execution for validation.

The plugin-agnostic integration test suite requires a few additional pieces; specifically:

1. **Test Fixture Convention**: Static `buildGraph()` method on fixtures enables capability queries without test execution
2. **BuildEngineTestMatrix**: Template function that expands test cases to (engine, testCase) pairs based on runtime capability queries
3. **Configuration System**: TOML file specifies expected plugins, expected failures, and per-engine tolerances

TheRock CI integration requires two additional considerations:
1. **Filtering tests to match build configurations**: not all builds include all plugins
2. **Managing test execution time across CI stages**

## Implementation Details

### IntegrationGraphVerificationHarness Base Class

This is relatively unchanged from existing MIOpen plugin infrastructure, it provides a numerical validation framework:

```cpp
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;

    void SetUp() override
    {
        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);

        // Verify loaded plugins match expected configuration
        verifyExpectedPlugins();
    }

    void verifyGraph(hipdnn_frontend::graph::Graph& graph, unsigned int seed)
    {
        // 1. Build graph for execution
        auto result = graph.build(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK);

        // 2. Generate test data bundles (CPU and GPU)
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle, cpuBundle;
        generateBundles(graph, cpuBundle, gpuBundle, outputTensorIds);
        initializeBundle(graph, gpuBundle, seed);
        initializeBundle(graph, cpuBundle, seed);

        // 3. Execute on GPU and CPU reference
        executeGpuGraph(_handle, graph, gpuBundle);
        executeCpuGraph(graph, cpuBundle);

        // 4. Validate outputs
        for(const auto& tensorId : outputTensorIds)
        {
            bool valid = _tensorIdToValidatorMap.at(tensorId)->allClose(
                *cpuBundle.tensors.at(tensorId),
                *gpuBundle.tensors.at(tensorId));
            ASSERT_TRUE(valid);
        }
    }

    virtual void runGraphTest() = 0;
};
```

### Test Fixture Convention

Fixtures must provide a static `buildGraph()` method that constructs and validates the graph without executing it:

```cpp
class ConvForward : public IntegrationGraphVerificationHarness<DataType, EngineTestCase<ConvFwdTestCase>>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    // Required: static method for capability queries
    static std::pair<graph::Graph, GraphOutputs> buildGraph(
        hipdnnHandle_t handle, const ConvFwdTestCase& tc)
    {
        // 1. Construct graph
        hipdnn_frontend::graph::Graph graphObj;
        // ... add tensors and operations ...

        // 2. Validate graph structure
        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        // 3. Build operation graph (required for capability query)
        auto buildResult = graphObj.build_operation_graph(handle);
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: " + buildResult.get_message());
        }

        return std::make_pair(std::move(graphObj), GraphOutputs{yAttr});
    }
};
```

### BuildEngineTestMatrix Function

The core filtering logic queries engine capabilities at test instantiation time:

```cpp
template <typename TestCase>
struct EngineTestCase
{
    int64_t engineId;
    TestCase testCase;
};
```

```cpp
template <typename FixtureClass, typename TestCase>
std::vector<EngineTestCase<TestCase>> BuildEngineTestMatrix(
    testing::internal::ParamGenerator<TestCase> testCaseGen) {

    std::vector<EngineTestCase<TestCase>> result;

    // Create handle for capability queries
    // Plugin loading is cached, so this is cheap after first call
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    for (const auto& testCase : testCaseGen) {
        auto [graph, outputs] = FixtureClass::buildGraph(handle, testCase);

        // Query which engines support this graph
        std::vector<int64_t> engineIds;
        auto status = graph.get_ranked_engine_ids(engineIds);
        if(status.is_bad())
        {
            // No loaded engine supports the graph - skip this test case
            continue;
        }

        for (int64_t engineId : engineIds) {
            result.push_back(EngineTestCase<TestCase>{engineId, testCase});
        }
    }

    hipdnnDestroy(handle);
    return result;
}
```

### ASIDE: GTest Test Discovery

GTest parameterized test discovery happens in two phases:

1. **Static initialization**: `INSTANTIATE_TEST_SUITE_P` registers parameter generator functions with GTest's internal registry. The generators are not called yet.

```cpp
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwd2dFp32,
    testing::ValuesIn(BuildEngineTestMatrix<IntegrationGpuConvFwd2dFp32, ConvFwdTestCase>(
        testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                         testing::ValuesIn(test_conv_common::getConvTestCases4D())))),
    EngineTestNameGenerator<ConvFwdTestCase>);
// At this point, GTest stores a pointer to a function that will call BuildEngineTestMatrix
```

2. **InitGoogleTest()**: GTest invokes all registered generator functions to create the actual test instances.

```cpp
// main.cpp
int main(int argc, char** argv)
{
    // Phase 2: Generator functions are invoked here, including BuildEngineTestMatrix()
    ::testing::InitGoogleTest(&argc, argv);

    auto result = RUN_ALL_TESTS();
    return result;
}
```

`BuildEngineTestMatrix` runs during `InitGoogleTest()`, not static init. At this point:
- Plugins are loaded (triggered by first `hipdnnCreate()` call)
- Engine capabilities can be queried
- Test cases are filtered based on actual runtime support

### Configuration System

While the test suite has no compile-time dependency on plugins, runtime configuration is still needed for:

- **expected failures**: mark specific (engine, test) combinations as expected to fail, so known issues don't block CI
- **expected plugins**: given the runtime test discovery, we need a way to verify that expected plugins are actually loaded - otherwise missing plugins generate no tests therefore no test _failures_.
  - NOTE: The TOML config lists ALL available plugins. At test time the expected plugins may be filtered down based on test configuration - see [Test Configurations](#test-configurations) for details.
- **numerical tolerance configuration**: allow for engine specific tolerance configuration.

Following the standard pattern in `TheRock`, a Python script is responsible for running the tests (see [`build_tools/github_actions/test_executable_scripts/`](https://github.com/ROCm/TheRock/tree/main/build_tools/github_actions/test_executable_scripts)). The test runner converts user-edited TOML configuration to JSON and passes it to the C++ test harness via environment variable.

#### Configuration File Format (TOML)

```toml
[plugins.miopen]
path = "libmiopen_plugin.so"
engines = ["MIOPEN_PLUGIN"]

[plugins.fusilli]
path = "libfusilli_plugin.so"
engines = ["FUSILLI"]

[engines.FUSILLI]
tolerance = "dynamic"  # maps to pre-defined methods of determining acceptable tolerance in C++
expected_failures = [
    "IntegrationGpuConvFwd3dFp32/Smoke.Correctness/NCDHW_1x1x4x4x4_1x1x3x3x3",
    "IntegrationGpuConvFwd3dFp32/Smoke.Correctness/NCDHW_1x1x8x8x8_1x1x3x3x3",
]

[engines.MIOPEN_PLUGIN]
tolerance = "gh_12678_tolerance_workaround"
```

Tolerance configuration is kept general: the name maps to a function that takes a graph as input and outputs a tolerance. Most plugins should use standard methods, but highly specific workaround methods can be added.

##### Why TOML -> JSON?

- **TOML**: Human-friendly - comments! A TOML parser is part of the Python standard library (as of 3.10)
- **JSON**: Easy to parse in C++ - `nlohmann/json` is already a hipDNN dependency

Users only interact with the TOML file.

## TheRock CI Integration

### Build configurations for tests

[Test configurations](https://github.com/AaronStGeorge/rocm-libraries/blob/b9b531a79ec16689aa839600953be238c6d10d94/.github/scripts/therock_matrix.py#L70-L78) may not include every plugin (e.g., MIOpen-only builds when changes only affect MIOpen). The integration test infrastructure must:

1. **Run only tests for plugins that are physically present** - don't fail because fusilli isn't installed in an MIOpen-only build
2. **Verify expected plugins are loaded** - if a build is supposed to include MIOpen but doesn't load it there should be a error

#### Filtering plugin-loaded checks based on enabled plugins

The test harness needs to know which plugins are expected to load. The TOML config lists all available plugins, but a given build may only include a subset. So that the test runner knows what _should_ be loaded, `TheRock` build (will) install an enabled plugin manifest as part of the super-build.

NOTE: the `THEROCK_ENABLE_*_PLUGIN` variables are set by the build topology infrastructure not necessarily by the user; so, for example, `THEROCK_ENABLE_FUSILLI_PLUGIN` is set even when the user only specified `THEROCK_ENABLE_IREE_LIBS`.

```cmake
# ml-libs/CMakeLists.txt - inside if(THEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS)
set(_plugins_json_array "")
if(THEROCK_ENABLE_MIOPEN_PLUGIN)
  list(APPEND _plugins_json_array "\"miopen\"")
endif()
if(THEROCK_ENABLE_FUSILLI_PLUGIN)
  list(APPEND _plugins_json_array "\"fusilli\"")
endif()
string(REPLACE ";" ", " _plugins_json_str "${_plugins_json_array}")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build_enabled_hipdnn_plugins_manifest.json"
  "{\n  \"plugins\": [${_plugins_json_str}]\n}\n")
```

The Python test runner reads the JSON manifest and filters the TOML config:

```python
enabled_plugins_path = Path(THEROCK_BIN_DIR) / "build_enabled_hipdnn_plugins_manifest.json"
with open(enabled_plugins_path) as f:
    enabled_plugins = set(json.load(f).get("plugins", []))

config["plugins"] = {
    name: cfg for name, cfg in config["plugins"].items()
    if name in enabled_plugins
}
```

This filtered config is passed to the C++ test harness, which verifies hipDNN loads exactly the expected plugins.

### Managing CI Time Budgets

As the test suite grows tests will need to be categorized so CI can prioritize the highest-signal tests within the time limits of each stage (pre-submit, post-submit, nightly, weekly). The integration tests will adopt the approach and infrastructure in [rocm-libraries' standardized test filtering infrastructure](https://github.com/ROCm/rocm-libraries/pull/3513) to solve this problem.

The infrastructure for test filtering and individual test timeouts is defined in [PR #3513](https://github.com/ROCm/rocm-libraries/pull/3513); this RFC only needs to establish a test naming convention to leverage it.

#### Filtering Configuration from [PR #3513](https://github.com/ROCm/rocm-libraries/pull/3513)

[PR #3513](https://github.com/ROCm/rocm-libraries/pull/3513) introduces `test_categories.yaml` to map test name patterns to CTest labels and define per-test timeouts. CI invokes tests via labels: `ctest -L Smoke` for pre-submit, `ctest -L Full` for post-submit.

##### Test Categories

NOTE: the number and names of categories can be changed later, hopefully `TheRock` standardizes names over time.

| Category | Target Time | CI Stage |
|----------|-------------|----------|
| Smoke | 5 min | Pre-submit |
| Full | 30 min | Post-submit |
| Nightly | 2 hrs | Nightly |

```yaml
test_categories:
  Smoke:
    description: "Pre-submit sanity checks"
    test_patterns:
      - "*/Smoke.*"
    labels: ["Smoke"]

  Full:
    description: "Post-submit validation"
    test_patterns:
      - "*/Smoke.*"
      - "*/Full.*"
    labels: ["Full"]

  Nightly:
    description: "Nightly full coverage"
    test_patterns:
      - "*"
    labels: ["Nightly"]

execution_settings:
  default_timeout: 60
  category_timeouts:
    Smoke: 60      # Individual Smoke tests should be fast
    Full: 300      # 5 min per test
    Nightly: 600   # 10 min per test
```

##### Naming Convention

To match the patterns above, each `INSTANTIATE_TEST_SUITE_P` uses a category prefix (`Smoke`, `Full`, or `Nightly`):

```cpp
// Pre-submit tests - small, fast cases
INSTANTIATE_TEST_SUITE_P(Smoke, IntegrationGpuConvFwd2dFp32,
    testing::ValuesIn(BuildEngineTestMatrix<IntegrationGpuConvFwd2dFp32, ConvFwdTestCase>(
        testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                         testing::ValuesIn(test_conv_common::getQuickConvTestCases4D())))),
    EngineTestNameGenerator<ConvFwdTestCase>);

// Post-submit tests - more comprehensive
INSTANTIATE_TEST_SUITE_P(Full, IntegrationGpuConvFwd2dFp32,
    testing::ValuesIn(BuildEngineTestMatrix<IntegrationGpuConvFwd2dFp32, ConvFwdTestCase>(
        testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                         testing::ValuesIn(test_conv_common::getStandardConvTestCases4D())))),
    EngineTestNameGenerator<ConvFwdTestCase>);

// Nightly tests - exhaustive coverage
INSTANTIATE_TEST_SUITE_P(Nightly, IntegrationGpuConvFwd2dFp32,
    testing::ValuesIn(BuildEngineTestMatrix<IntegrationGpuConvFwd2dFp32, ConvFwdTestCase>(
        testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                         testing::ValuesIn(test_conv_common::getComprehensiveConvTestCases4D())))),
    EngineTestNameGenerator<ConvFwdTestCase>);
```

This produces test names like:
```
IntegrationGpuConvFwd2dFp32/Smoke.Correctness/Fusilli_NCHW_1x16x16x16_1x16x3x3
IntegrationGpuConvFwd2dFp32/Smoke.Correctness/MIOpen_NCHW_1x16x16x16_1x16x3x3
IntegrationGpuConvFwd2dFp32/Full.Correctness/Fusilli_NCHW_8x32x64x64_32x32x3x3
IntegrationGpuConvFwd2dFp32/Full.Correctness/MIOpen_NCHW_8x32x64x64_32x32x3x3
```

#### Test suite (vs individual test) time budget

The test count scales combinatorially with plugins and graphs (each test runs per capable engine); configurations with all plugins enabled will eventually stretch CI time budgets. GTest and the category system provide the flexibility to address this.

Two strategies are available:
1. **Shard**: GTest supports sharding out of the box via environment variables
2. **Recategorize**: Move slower tests to a less frequent category (Smoke → Full → Nightly)

This RFC suggests setting CI stage timeouts at the maximum acceptable budget, then sharding or recategorizing as headroom runs out.

## Adding New Tests

### Step 1: Define the Test Case Type

```cpp
using ConvFwdTestCase = std::tuple<TensorLayout, ConvTestParams>;
```

### Step 2: Create the Test Fixture

```cpp
template <typename DataType>
class ConvForward : public IntegrationGraphVerificationHarness<DataType, EngineTestCase<ConvFwdTestCase>>
{
public:
    struct GraphOutputs { /* output tensor attributes */ };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(
        hipdnnHandle_t handle, const ConvFwdTestCase& tc);

protected:
    void runGraphTest() override;
};
```

### Step 3: Implement buildGraph (Static)

This method must:
1. Construct the graph from test case parameters
2. Call `validate()` and `build_operation_graph(handle)`
3. Return graph and output tensor attributes

### Step 4: Implement runGraphTest

```cpp
void runGraphTest() override
{
    const auto& param = this->GetParam();
    auto [graphObj, outputs] = buildGraph(this->_handle, param.testCase);

    // Look up tolerance based on engine config and graph properties
    auto tolerance = getTolerance(param.engineId, graphObj);

    // Register validators for outputs
    this->registerValidator(outputs.y, tolerance);

    // Force execution on the specific engine
    graphObj.set_preferred_engine_id_ext(param.engineId);

    this->verifyGraph(graphObj, seed);
}
```

### Step 5: Instantiate with BuildEngineTestMatrix

```cpp
using MyFixtureFp32 = MyFixture<float>;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(MyFixtureFp32);
TEST_P(MyFixtureFp32, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    MyFixtureFp32,
    testing::ValuesIn(BuildEngineTestMatrix<MyFixtureFp32, MyTestCase>(
        testing::Combine(
            testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
            testing::ValuesIn(getTestCases())))),
    EngineTestNameGenerator<MyTestCase>);
```

NOTE: `GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST` prevents spurious failures when no engines support a given test configuration.
