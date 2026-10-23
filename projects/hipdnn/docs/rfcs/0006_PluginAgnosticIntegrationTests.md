# RFC 0006: Plugin-Agnostic Integration Tests

- **Status**: Implemented
- **Status notes**: The suite shipped as `dnn-providers/integration-tests` (#4087). Each provider runs the suite against its own plugin and engine (`--test-article`, `--test-engine`, `--test-config`). Graph-verification coverage is mostly data bundles (RFC 0011); C++ graph tests build only with `-DBUILD_CPP_GRAPH_TESTS=ON`.
- **Last verified**: 2026-10-15 against `develop` (2501d0b28cc)


## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Design Goals](#design-goals)
4. [Proposed Solution](#proposed-solution)
5. [Implementation Details](#implementation-details)
   - [IntegrationGraphVerificationHarness](#integrationgraphverificationharness-base-class)
   - [Test Fixture Convention](#test-fixture-convention)
   - [Engine Selection](#engine-selection)
   - [Configuration System](#configuration-system)
6. [TheRock CI Integration](#therock-ci-integration)
   - [Build configurations for tests](#build-configurations-for-tests)
   - [Managing CI Time Budgets](#managing-ci-time-budgets)
7. [Adding New Tests](#adding-new-tests)

## Executive Summary

This RFC documents a standalone integration test project for hipDNN enabling plugin-agnostic testing with runtime capability discovery.
The same tests run against each plugin. At run time, each test asks hipDNN whether the selected plugin and engine support its graph, and skips the test if not. This gives each plugin an automatic level of test coverage "out of the box."

### Key Benefits
- **Reduced Duplication**: Eliminates near-identical test suites in each plugin
- **Plugin Independence**: Tests have no compile-time dependency on any specific plugin
- **Forward-Deployed Support**: Plugins built outside TheRock (e.g., fusilli/IREE) work seamlessly

## Problem Statement

Each plugin (MIOpen, fusilli, etc.) needs numerical integration tests for all graph variants it claims to support - preferably in a way that gives strong signal when debugging. Without a shared suite, nearly identical tests exist in multiple locations: [miopen conv test](https://github.com/ROCm/rocm-libraries/blob/9924d667c218d067403608d77f432dd13585a848/dnn-providers/miopen-provider/integration_tests/IntegrationGpuConvForward.cpp) vs [fusilli conv test](https://github.com/iree-org/fusilli/blob/9328342731374ef7113f1f95c18a986c1df3739d/plugins/hipdnn-plugin/test/integration/convolution/conv_fprop_parameterized_full.cpp).

## Design Goals

1. **Plugin Independence**: No compile-time dependencies on specific plugins, the tests build only on core hipDNN; tests discover plugin capabilities at runtime
2. **Forward deploy / ThePebble Compatibility**: Tests can run standalone with forward-deployed plugins
    - Fusilli plugin (and presumably future forward deployed plugins) builds outside of `TheRock`. Currently Fusilli plugin maintains a hacky simulacrum of `TheRock` build ([`ThePebble.py`](https://github.com/iree-org/fusilli/blob/9328342731374ef7113f1f95c18a986c1df3739d/plugins/hipdnn-plugin/build_tools/ThePebble.py)) that pulls down some pre-built artifacts and sets up a local build roughly as it would be if fusilli-plugin was building as part of `TheRock`. Ideally, fusilli should be able to use the test suite by simply downloading pre-built artifacts from `TheRock` distribution.
3. **Build configurations for tests**: Build configurations for tests may not build with every available plugin, the integration suite should gracefully handle running a subset of total possible tests. For example: a change in the MIOpen plugin should not require building fusilli plugin or running fusilli plugin tests.
4. **Maximize signal within time budget**: As the suite grows, not all tests can run in every CI stage. Tests must be categorized by execution time so that CI can run the highest-signal tests within time budgets available in each stage (pre-submit, post-submit, nightly).

## Proposed Solution

### Overview

The suite is in `dnn-providers/integration-tests/` and builds one binary, `hipdnn_integration_tests`. See its [README](../../../../dnn-providers/integration-tests/README.md).

Bundles and sweeps under `integration-test-bundles/` ([RFC 0011](./0011_BundleTestFramework.md)) are the default way to test that a graph runs on an engine. C++ graph tests are kept for special cases and build only with `-DBUILD_CPP_GRAPH_TESTS=ON`.

Most of the testing infrastructure can be taken directly from existing MIOpen integration test suite:

**IntegrationGraphVerificationHarness**: Base class providing GTest support and reference execution for validation.

The plugin-agnostic integration test suite requires a few additional pieces; specifically:

1. **Test Fixture Convention**: Static `buildGraph()` method on fixtures builds the graph for a test case
2. **Engine Selection**: Command-line options select the plugin and engine under test. Each test checks engine support at run time and skips if the graph is not supported.
3. **Configuration System**: A per-engine TOML file specifies test skips and tolerance overrides

TheRock CI integration requires two additional considerations:
1. **Filtering tests to match build configurations**: not all builds include all plugins
2. **Managing test execution time across CI stages**

## Implementation Details

### IntegrationGraphVerificationHarness Base Class

The class is in `src/harness/IntegrationGraphVerificationHarness.hpp`. It provides a numerical validation framework. This is a simplified view:

```cpp
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        // Apply [[test_skips]] from the TOML config
        if(auto reason = checkTomlSkip(currentTestName()))
        {
            GTEST_SKIP() << *reason;
        }
    }

    void verifyGraph(hipdnn_frontend::graph::Graph& graph)
    {
        // 1. Skip if the selected engine does not support the graph
        ASSERT_NO_FATAL_FAILURE(ensureEngineSupport(graph));
        if(testing::Test::IsSkipped())
            return;

        // 2. Build execution plans
        ASSERT_NO_FATAL_FAILURE(buildExecutionPlans(graph));

        // 3. Generate and initialize test data bundles
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle, refBundle;
        generateBundles(graph, refBundle, gpuBundle);
        initializeBundle(graph, gpuBundle);
        initializeBundle(graph, refBundle);

        // 4. Execute on GPU and on the reference executor
        ASSERT_NO_FATAL_FAILURE(executeGpuGraph(getSharedHandle(), graph, gpuBundle));
        ASSERT_NO_FATAL_FAILURE(executeReferenceGraph(graph, refBundle));

        // 5. Validate outputs
        ASSERT_NO_FATAL_FAILURE(validateOutputs(gpuBundle, refBundle));
    }

    virtual void runGraphTest() = 0;
};
```

The reference executor is a CPU or GPU executor, selected with `--reference-executor`. All tests share one hipDNN handle from `getSharedHandle()`.

### Test Fixture Convention

Fixtures provide a static `buildGraph()` method that constructs and validates the graph without executing it:

```cpp
class ConvForward : public IntegrationGraphVerificationHarness<DataType, ConvFwdTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

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

### Engine Selection

Each run tests one plugin, given with `--test-article`/`--ta`, and, optionally, one engine, given with `--test-engine`/`--te`. Without `--test-engine`, hipDNN picks the engine.

`INSTANTIATE_TEST_SUITE_P` uses plain GTest generators. Engine support is checked inside each test by `checkEngineSupportOrSkip()`, which `verifyGraph()` calls:

1. It calls `get_ranked_engine_ids()` on the graph.
2. If an engine is named and that engine does not support the graph, it skips the test.
3. If no engine is named and no engine supports the graph, it skips the test.
4. With `--fail-on-unsupported`, it fails the test instead of skipping it.
5. If an engine is named and supports the graph, it calls `set_preferred_engine_id_ext()` with that engine.

### Configuration System

While the test suite has no compile-time dependency on plugins, runtime configuration is still needed for:

- **test skips**: skip specific tests on specific architectures, so known issues don't block CI
- **numerical tolerance configuration**: allow for engine specific tolerance configuration.

Each provider keeps a per-engine TOML file, for example `dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml`. The binary reads it with `--test-config`/`--tc` and parses it in C++ with toml++.

#### Configuration File Format (TOML)

The file has these tables:

- `[meta]`: file format version.
- `[[tolerance_overrides]]`: `filters` (GTest-style globs on the full test name), `atol`, and `rtol`.
- `[[test_skips]]`: `archs`, `filters`, and `reason`.

```toml
[meta]
version = 1

[[tolerance_overrides]]
filters = ["Full/*BatchnormBackwardCalcStats3d*.Correctness/0"]
atol = 0.002
rtol = 0.012
```

When more than one filter matches, later entries take precedence. Tests with no override use the default tolerances from `src/harness/tolerance/ToleranceResolver.hpp`.

## TheRock CI Integration

### Build configurations for tests

[Test configurations](https://github.com/AaronStGeorge/rocm-libraries/blob/b9b531a79ec16689aa839600953be238c6d10d94/.github/scripts/therock_matrix.py#L70-L78) may not include every plugin (e.g., MIOpen-only builds when changes only affect MIOpen). The integration test infrastructure must run only tests for plugins that are physically present - don't fail because fusilli isn't installed in an MIOpen-only build.

Each provider registers the suite against its own plugin with `add_external_integration_test_target()` from `dnn-providers/integration-tests/cmake/HipdnnIntegrationTestHelpers.cmake`. The provider passes its plugin target, engine name, TOML config, and test categories file. So the suite does not need to know which plugins a build enables.

### Managing CI Time Budgets

As the test suite grows tests will need to be categorized so CI can prioritize the highest-signal tests within the time limits of each stage (pre-submit, post-submit, nightly, weekly). The integration tests use [rocm-libraries' standardized test filtering infrastructure](https://github.com/ROCm/rocm-libraries/pull/3513) to solve this problem.

The infrastructure for test filtering and individual test timeouts is defined in [PR #3513](https://github.com/ROCm/rocm-libraries/pull/3513); this RFC only needs to establish a test naming convention to leverage it.

#### Filtering Configuration from [PR #3513](https://github.com/ROCm/rocm-libraries/pull/3513)

[PR #3513](https://github.com/ROCm/rocm-libraries/pull/3513) introduces test categories YAML files to map test name patterns to CTest labels and define per-test timeouts. CI invokes tests via labels, for example `ctest -L quick`.

##### Test Categories

The categories are `quick`, `standard`, `comprehensive`, and `full`. Each provider defines them for this suite in `test_categories_integration.yaml`, for example `dnn-providers/miopen-provider/test_categories_integration.yaml`. Higher categories include the patterns of lower ones, so `ctest -L standard` also runs the quick tests.

##### Naming Convention

Bundle tests are named after their tier directory (`quick_*`, `standard_*`, and so on). C++ suites use a category prefix on `INSTANTIATE_TEST_SUITE_P`: `Smoke`, `Quick`, `Standard`, `Comprehensive`, or `Full`.

```cpp
// Pre-submit tests - small, fast cases
INSTANTIATE_TEST_SUITE_P(Smoke, IntegrationGpuConvFwd2dFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getQuickConvTestCases4D())));
```

#### Test suite (vs individual test) time budget

The test count scales with plugins and graphs; configurations with all plugins enabled will eventually stretch CI time budgets. GTest and the category system provide the flexibility to address this.

Two strategies are available:
1. **Shard**: GTest supports sharding out of the box via environment variables
2. **Recategorize**: Move slower tests to a less frequent category (quick → standard → comprehensive → full)

This RFC suggests setting CI stage timeouts at the maximum acceptable budget, then sharding or recategorizing as headroom runs out.

## Adding New Tests

New graph-verification coverage must be a bundle. `src/integration-tests/CMakeLists.txt` enforces this. Use a C++ test only for behavior other than "a graph runs and matches a reference", for example error paths or serialization round-trips. See the integration-tests [README](../../../../dnn-providers/integration-tests/README.md) and [adding-tests.md](../../../../dnn-providers/integration-tests/docs/adding-tests.md).

The steps below apply to C++ tests.

### Step 1: Define the Test Case Type

```cpp
using ConvFwdTestCase = std::tuple<TensorLayout, ConvTestParams>;
```

### Step 2: Create the Test Fixture

```cpp
template <typename DataType>
class ConvForward : public IntegrationGraphVerificationHarness<DataType, ConvFwdTestCase>
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
    const auto& testCase = this->GetParam();
    auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

    // Register validators for outputs, with the resolved tolerance
    this->registerValidator(outputs.y, this->getTolerance(graphObj, outputs.y));

    // Checks engine support, then executes and validates
    this->verifyGraph(graphObj);
}
```

### Step 5: Instantiate the Test Suite

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
    testing::Combine(
        testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
        testing::ValuesIn(getTestCases())));
```
