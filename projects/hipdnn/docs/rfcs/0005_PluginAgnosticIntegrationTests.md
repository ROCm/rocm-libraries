# RFC 0005: Plugin-Agnostic Integration Tests

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
6. [Configuration System](#configuration-system)
7. [Adding New Tests](#adding-new-tests)

## Executive Summary

This RFC documents a standalone integration test project for hipDNN enabling plugin-agnostic testing with runtime capability discovery.
The test infrastructure automatically defines a cartesian product of (engine, graph) paris for numerical integration tests, giving each plugin an automatic level of test coverage "out of the box."

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

## Proposed Solution

### Overview

Most of the testing infrastructure can be taken directly from existing MIOpen integration test suite:

**IntegrationGraphVerificationHarness**: Base class providing GTest support and CPU reference execution for validation.

The plugin-agnostic integration test suite requires a few additional pieces; specifically:

1. **Test Fixture Convention**: Static `buildGraph()` method on fixtures enables capability queries without test execution
2. **BuildEngineTestMatrix**: Template function that expands test cases to (engine, testCase) pairs based on runtime capability queries
3. **Configuration System**: TOML file specifies expected plugins, expected failures, and per-engine tolerances

## Implementation Details

### IntegrationGraphVerificationHarness Base Class

This is completely unchanged from existing MIOpen plugin infrastructure, it provides a validation framework:

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

    virtual void runGraphTest(DataType tolerance) = 0;
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

## Configuration System

While the test suite has no compile-time dependency on plugins, runtime configuration is still needed for:
- **expected plugins**: given the runtime test discovery, we need a way to verify that expected plugins are actually loaded - otherwise missing plugins generate no tests therefore no _failures_.
- **expected failures**: mark specific (engine, test) combinations as expected to fail, so known issues don't block CI
- **numerical tolerance configuration**: allow for engine specific tolerance configuration.

Following the standard pattern in `TheRock`, a Python script is responsible for running the tests (see [`build_tools/github_actions/test_executable_scripts/`](https://github.com/ROCm/TheRock/tree/main/build_tools/github_actions/test_executable_scripts)). The test runner converts user-edited TOML configuration to JSON and passes it to the C++ test harness via environment variable.

### Configuration File Format (TOML)

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

#### Why TOML -> JSON?

- **TOML**: Human-friendly - comments! A TOML parser is part of the Python standard library (as of 3.10)
- **JSON**: Easy to parse in C++ - `nlohmann/json` is already a hipDNN dependency

Users only interact with the TOML file.

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
    void runGraphTest(DataType tolerance) override;
};
```

### Step 3: Implement buildGraph (Static)

This method must:
1. Construct the graph from test case parameters
2. Call `validate()` and `build_operation_graph(handle)`
3. Return graph and output tensor attributes

### Step 4: Implement runGraphTest

```cpp
void runGraphTest(DataType tolerance) override
{
    const auto& param = this->GetParam();
    auto [graphObj, outputs] = buildGraph(this->_handle, param.testCase);

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
    runGraphTest(getTolerance<float>());
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
