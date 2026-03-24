// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration tests for the example plugin.
//
// These tests exercise the full hipDNN stack: frontend graph API, backend
// engine selection, plugin loading, and graph execution.  They verify that
// the example plugin can be loaded, its engines discovered, and GPU-based
// ReLU forward and ConvFwd passes executed correctly through the standard
// 6-step graph build/execute sequence.
//
// All operations run on GPU device memory via HIPRTC-compiled kernels.
// Tests skip gracefully if no GPU is detected at runtime.
//
// Plugin auto-discovery: the test fixture checks for the plugin shared
// library in the same directory as the test executable.  If found and
// HIPDNN_PLUGIN_DIR is not already set, the fixture sets it automatically
// via ScopedEnvironmentVariableSetter so that hipDNN discovers the plugin
// without any manual environment configuration.
//
// Prerequisites:
//   - hipDNN installed at /opt/rocm (frontend + backend)
//   - ROCm with HIPRTC and a compatible GPU
//   - The example_plugin shared library built in the same CMake project

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

#ifdef _WIN32
static constexpr const char* PLUGIN_FILENAME = "example_plugin.dll";
#else
static constexpr const char* PLUGIN_FILENAME = "libexample_plugin.so";
#endif

// ============================================================================
// Helper: build a pointwise ReLU forward graph
// ============================================================================
std::shared_ptr<Graph> createPointwiseReluGraph(const std::string& graphName,
                                                const std::vector<int64_t>& dims)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(graphName)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Input tensor X (UID = 1)
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim(dims)
        .set_stride({dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes attrs;
    attrs.set_name("relu_fwd");
    attrs.set_mode(PointwiseMode::RELU_FWD);

    // Output tensor Y (UID = 2)
    auto y = graph->pointwise(x, attrs);
    y->set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

    return graph;
}

// ============================================================================
// Helper: build a ConvFwd graph
// ============================================================================
std::shared_ptr<Graph> createConvFwdGraph(const std::string& graphName,
                                          int64_t N,
                                          int64_t C,
                                          int64_t H,
                                          int64_t W,
                                          int64_t K,
                                          int64_t R,
                                          int64_t S,
                                          const std::vector<int64_t>& padding,
                                          const std::vector<int64_t>& stride,
                                          const std::vector<int64_t>& dilation)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(graphName)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Input tensor X (UID = 1): NCHW layout
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({N, C, H, W})
        .set_stride({C * H * W, H * W, W, 1})
        .set_data_type(DataType::FLOAT);

    // Weight tensor W (UID = 2): KCRS layout
    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim({K, C, R, S})
        .set_stride({C * R * S, R * S, S, 1})
        .set_data_type(DataType::FLOAT);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding(padding).set_stride(stride).set_dilation(dilation);

    // Output tensor Y (UID = 3)
    auto y = graph->conv_fprop(x, w, convAttrs);
    y->set_uid(3).set_data_type(DataType::FLOAT).set_output(true);

    return graph;
}

// ============================================================================
// Helper: query loaded plugin paths via the frontend API
// ============================================================================
std::vector<std::filesystem::path> getLoadedPluginPaths(hipdnnHandle_t handle)
{
    std::vector<std::filesystem::path> paths;
    auto err = hipdnn_frontend::getLoadedEnginePluginPaths(handle, paths);
    if(err.is_bad())
    {
        return {};
    }
    return paths;
}

// ============================================================================
// Test fixture
// ============================================================================
class PluginIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check GPU availability; skip all tests if no GPU is detected
        int deviceCount = 0;
        auto hipErr = hipGetDeviceCount(&deviceCount);
        if(hipErr != hipSuccess || deviceCount == 0)
        {
            GTEST_SKIP() << "No GPU detected -- skipping integration test";
        }

        // Auto-discover the plugin from the test executable's directory.
        // The plugin shared library is expected to be co-located with the
        // test executable in the build output directory.
        auto exeDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();
        auto pluginPath = exeDir / PLUGIN_FILENAME;
        ASSERT_TRUE(std::filesystem::exists(pluginPath))
            << "Plugin not found alongside test executable: " << pluginPath;

        // If HIPDNN_PLUGIN_DIR is already set, use it; otherwise default
        // to the executable's directory so hipDNN discovers the plugin.
        std::string pluginDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_PLUGIN_DIR");
        if(pluginDir.empty())
        {
            pluginDir = std::filesystem::absolute(exeDir).string();
        }
        _envSetter.emplace("HIPDNN_PLUGIN_DIR", pluginDir);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS)
            << "Failed to create hipDNN handle";
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        _envSetter.reset();
    }

    hipdnnHandle_t _handle = nullptr;
    std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _envSetter;
};

// ============================================================================
// Tests
// ============================================================================

// Verify that the plugin was loaded and appears in the loaded-plugin list.
TEST_F(PluginIntegrationTest, PluginIsLoaded)
{
    auto paths = getLoadedPluginPaths(_handle);
    ASSERT_FALSE(paths.empty()) << "No plugins were loaded";

    bool found = false;
    for(const auto& path : paths)
    {
        if(path.string().find("example_plugin") != std::string::npos)
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "example_plugin not found among loaded plugins";
}

// End-to-end: select the example plugin's GPU ReLU engine by name and execute
// a ReLU forward pass on GPU device memory, verifying the output is correct.
TEST_F(PluginIntegrationTest, ReluForwardWithEngineSelection)
{
    std::vector<int64_t> dims = {1, 1, 1, 8};
    auto graph = createPointwiseReluGraph("ReluFwdTest", dims);

    // Select the example plugin's GPU ReLU engine by name
    graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_RELU_ENGINE");

    // 6-step build/execute sequence
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Prepare device-side input/output buffers
    const size_t numElements = 8;
    const size_t bufferSize = numElements * sizeof(float);
    std::vector<float> inputData = {-3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f, -0.1f, 5.0f};
    std::vector<float> outputData(numElements, -999.0f);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    ASSERT_EQ(hipMalloc(&dInput, bufferSize), hipSuccess);
    ASSERT_EQ(hipMalloc(&dOutput, bufferSize), hipSuccess);

    ASSERT_EQ(hipMemcpy(dInput, inputData.data(), bufferSize, hipMemcpyHostToDevice), hipSuccess);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dInput;
    variantPack[2] = dOutput;

    result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    ASSERT_EQ(hipMemcpy(outputData.data(), dOutput, bufferSize, hipMemcpyDeviceToHost), hipSuccess);

    // Verify ReLU output: max(0, x)
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 2.0f, 0.0f, 5.0f};
    for(size_t i = 0; i < numElements; ++i)
    {
        EXPECT_FLOAT_EQ(outputData[i], expected[i])
            << "Mismatch at index " << i << ": input=" << inputData[i];
    }

    static_cast<void>(hipFree(dInput));
    static_cast<void>(hipFree(dOutput));
}

// Verify the full build() convenience method also works with engine selection.
TEST_F(PluginIntegrationTest, BuildConvenienceMethod)
{
    std::vector<int64_t> dims = {1, 1, 1, 4};
    auto graph = createPointwiseReluGraph("BuildConvenience", dims);
    graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_RELU_ENGINE");

    auto result = graph->build(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const size_t numElements = 4;
    const size_t bufferSize = numElements * sizeof(float);
    std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, -2.0f};
    std::vector<float> outputData(numElements, -999.0f);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    ASSERT_EQ(hipMalloc(&dInput, bufferSize), hipSuccess);
    ASSERT_EQ(hipMalloc(&dOutput, bufferSize), hipSuccess);
    ASSERT_EQ(hipMemcpy(dInput, inputData.data(), bufferSize, hipMemcpyHostToDevice), hipSuccess);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dInput;
    variantPack[2] = dOutput;

    result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    ASSERT_EQ(hipMemcpy(outputData.data(), dOutput, bufferSize, hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_FLOAT_EQ(outputData[0], 0.0f);
    EXPECT_FLOAT_EQ(outputData[1], 0.0f);
    EXPECT_FLOAT_EQ(outputData[2], 1.0f);
    EXPECT_FLOAT_EQ(outputData[3], 0.0f);

    static_cast<void>(hipFree(dInput));
    static_cast<void>(hipFree(dOutput));
}

// ConvFwd: select the example plugin's GPU ConvFwd engine and execute a
// forward convolution, verifying the output against a CPU reference.
TEST_F(PluginIntegrationTest, ConvForwardWithEngineSelection)
{
    // Small test dimensions: 1x1x4x4 input, 1x1x3x3 filter, no padding,
    // stride=1, dilation=1 => output 1x1x2x2
    const int64_t N = 1, C = 1, H = 4, W = 4;
    const int64_t K = 1, R = 3, S = 3;
    const int64_t padH = 0, padW = 0;
    const int64_t strideH = 1, strideW = 1;
    const int64_t outH = H - R + 1; // 2
    const int64_t outW = W - S + 1; // 2

    auto graph = createConvFwdGraph(
        "ConvFwdTest", N, C, H, W, K, R, S, {padH, padW}, {strideH, strideW}, {1, 1});
    graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_CONV_FWD_ENGINE");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Input: 4x4 matrix (row-major, values 1..16)
    // clang-format off
    std::vector<float> inputData = {
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    // clang-format on

    // Weight: 3x3 filter (all ones for easy verification)
    std::vector<float> weightData(R * S, 1.0f);

    // Expected output via SDK CPU reference convolution
    using TensorF = hipdnn_data_sdk::utilities::Tensor<float>;
    TensorF xTensor({N, C, H, W});
    xTensor.fillWithData(inputData.data(), inputData.size() * sizeof(float));

    TensorF wTensor({K, C, R, S});
    wTensor.fillWithData(weightData.data(), weightData.size() * sizeof(float));

    TensorF yTensor({N, K, outH, outW});
    yTensor.fillTensorWithValue(0.0f);

    hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
        xTensor, wTensor, yTensor, {strideH, strideW}, {int64_t{1}, int64_t{1}}, {padH, padW});

    // Copy reference output to a vector for comparison
    auto expectedOutputCount = static_cast<size_t>(N * K * outH * outW);
    std::vector<float> expectedOutput(expectedOutputCount);
    std::memcpy(expectedOutput.data(), yTensor.rawHostData(), expectedOutputCount * sizeof(float));

    const size_t inputSize = inputData.size() * sizeof(float);
    const size_t weightSize = weightData.size() * sizeof(float);
    const size_t outputSize = static_cast<size_t>(N * K * outH * outW) * sizeof(float);

    float* dInput = nullptr;
    float* dWeight = nullptr;
    float* dOutput = nullptr;
    ASSERT_EQ(hipMalloc(&dInput, inputSize), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWeight, weightSize), hipSuccess);
    ASSERT_EQ(hipMalloc(&dOutput, outputSize), hipSuccess);

    ASSERT_EQ(hipMemcpy(dInput, inputData.data(), inputSize, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dWeight, weightData.data(), weightSize, hipMemcpyHostToDevice), hipSuccess);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dInput; // X tensor
    variantPack[2] = dWeight; // W tensor
    variantPack[3] = dOutput; // Y tensor

    result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<float> outputData(static_cast<size_t>(N * K * outH * outW));
    ASSERT_EQ(hipMemcpy(outputData.data(), dOutput, outputSize, hipMemcpyDeviceToHost), hipSuccess);

    // Verify GPU output against CPU reference
    for(size_t i = 0; i < expectedOutput.size(); ++i)
    {
        EXPECT_NEAR(outputData[i], expectedOutput[i], 1e-5f) << "Mismatch at index " << i;
    }

    static_cast<void>(hipFree(dInput));
    static_cast<void>(hipFree(dWeight));
    static_cast<void>(hipFree(dOutput));
}

// Verify that loading with ABSOLUTE mode works: only the example plugin is
// active and the ReLU engine is available.
TEST_F(PluginIntegrationTest, AbsoluteLoadingMode)
{
    std::string pluginDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_PLUGIN_DIR");
    ASSERT_FALSE(pluginDir.empty());

    // Destroy the fixture handle before changing plugin paths -- the backend
    // rejects setEnginePluginPaths while any handle is active.
    ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
    _handle = nullptr;

    std::vector<std::string> pluginPaths = {pluginDir};
    auto err = hipdnn_frontend::setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE);
    ASSERT_FALSE(err.is_bad()) << err.err_msg;

    // Create a new handle after setting plugin paths
    hipdnnHandle_t handle2 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle2), HIPDNN_STATUS_SUCCESS);

    std::vector<int64_t> dims = {1, 1, 1, 4};
    auto graph = createPointwiseReluGraph("AbsoluteMode", dims);
    graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_RELU_ENGINE");

    auto result = graph->build(handle2);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const size_t numElements = 4;
    const size_t bufferSize = numElements * sizeof(float);
    std::vector<float> inputData = {-5.0f, 3.0f, 0.0f, -1.0f};
    std::vector<float> outputData(numElements, -999.0f);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    ASSERT_EQ(hipMalloc(&dInput, bufferSize), hipSuccess);
    ASSERT_EQ(hipMalloc(&dOutput, bufferSize), hipSuccess);
    ASSERT_EQ(hipMemcpy(dInput, inputData.data(), bufferSize, hipMemcpyHostToDevice), hipSuccess);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dInput;
    variantPack[2] = dOutput;

    result = graph->execute(handle2, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    ASSERT_EQ(hipMemcpy(outputData.data(), dOutput, bufferSize, hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_FLOAT_EQ(outputData[0], 0.0f);
    EXPECT_FLOAT_EQ(outputData[1], 3.0f);
    EXPECT_FLOAT_EQ(outputData[2], 0.0f);
    EXPECT_FLOAT_EQ(outputData[3], 0.0f);

    static_cast<void>(hipFree(dInput));
    static_cast<void>(hipFree(dOutput));
    EXPECT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

// Verify that loading with ADDITIVE mode works: the example plugin's engines
// are added alongside any system-installed plugins.
TEST_F(PluginIntegrationTest, AdditiveLoadingMode)
{
    std::string pluginDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_PLUGIN_DIR");
    ASSERT_FALSE(pluginDir.empty());

    // Destroy the fixture handle before changing plugin paths -- the backend
    // rejects setEnginePluginPaths while any handle is active.
    ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
    _handle = nullptr;

    std::vector<std::string> pluginPaths = {pluginDir};
    auto err = hipdnn_frontend::setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ADDITIVE);
    ASSERT_FALSE(err.is_bad()) << err.err_msg;

    hipdnnHandle_t handle2 = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle2), HIPDNN_STATUS_SUCCESS);

    // The example plugin should still be discoverable
    auto loadedPaths = getLoadedPluginPaths(handle2);
    bool found = false;
    for(const auto& p : loadedPaths)
    {
        if(p.string().find("example_plugin") != std::string::npos)
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "example_plugin not found after ADDITIVE loading";

    EXPECT_EQ(hipdnnDestroy(handle2), HIPDNN_STATUS_SUCCESS);
}

} // namespace
