// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Sample application demonstrating how to load a hipDNN engine plugin and
// execute graphs using the plugin's GPU engines.
//
// This program shows five scenarios:
//   1. Loading via HIPDNN_PLUGIN_DIR environment variable (ReLU on GPU)
//   2. Loading with ADDITIVE mode (alongside system plugins)
//   3. Loading with ABSOLUTE mode (only specified plugins)
//   4. Selecting a specific engine by name (ReLU on GPU)
//   5. Convolution forward on GPU
//
// All operations execute on GPU device memory via HIPRTC-compiled kernels.
//
// Plugin directory resolution:
//   1. If a command-line argument is provided, that path is used.
//   2. Otherwise, if HIPDNN_PLUGIN_DIR is set in the environment, that is used.
//   3. Otherwise, the directory containing this executable is used as a last resort.
//
// Prerequisites:
//   - hipDNN installed at /opt/rocm
//   - ROCm with HIPRTC and a compatible GPU
//   - The example_plugin shared library built in the same CMake project

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// ============================================================================
// HIP error checking helper
// ============================================================================
static bool checkHip(hipError_t err, const char* msg)
{
    if(err != hipSuccess)
    {
        std::cerr << "  HIP ERROR: " << msg << ": " << hipGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

// ============================================================================
// Helpers
// ============================================================================

// Build a pointwise ReLU forward graph with the given tensor dimensions.
static std::shared_ptr<Graph> createReluGraph(const std::string& name,
                                              const std::vector<int64_t>& dims)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(name)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim(dims)
        .set_stride({dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1})
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes attrs;
    attrs.set_name("relu_fwd");
    attrs.set_mode(PointwiseMode::RELU_FWD);

    auto y = graph->pointwise(x, attrs);
    y->set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

    return graph;
}

// Run the full build/execute sequence for a ReLU graph using GPU device memory.
// Returns true on success.
static bool runReluGraph(hipdnnHandle_t handle,
                         const std::string& graphName,
                         const std::vector<float>& input,
                         std::vector<float>& output)
{
    auto numElements = static_cast<int64_t>(input.size());
    std::vector<int64_t> dims = {1, 1, 1, numElements};

    auto graph = createReluGraph(graphName, dims);
    graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_RELU_ENGINE");

    auto result = graph->build(handle);
    if(result.code != ErrorCode::OK)
    {
        std::cerr << "  ERROR: build failed: " << result.err_msg << "\n";
        return false;
    }

    output.resize(input.size());
    const size_t bufferSize = input.size() * sizeof(float);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    if(!checkHip(hipMalloc(&dInput, bufferSize), "hipMalloc dInput"))
        return false;
    if(!checkHip(hipMalloc(&dOutput, bufferSize), "hipMalloc dOutput"))
    {
        static_cast<void>(hipFree(dInput));
        return false;
    }
    if(!checkHip(hipMemcpy(dInput, input.data(), bufferSize, hipMemcpyHostToDevice),
                 "hipMemcpy H2D"))
    {
        static_cast<void>(hipFree(dInput));
        static_cast<void>(hipFree(dOutput));
        return false;
    }

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dInput;
    variantPack[2] = dOutput;

    result = graph->execute(handle, variantPack, nullptr);
    if(result.code != ErrorCode::OK)
    {
        std::cerr << "  ERROR: execute failed: " << result.err_msg << "\n";
        static_cast<void>(hipFree(dInput));
        static_cast<void>(hipFree(dOutput));
        return false;
    }

    if(!checkHip(hipMemcpy(output.data(), dOutput, bufferSize, hipMemcpyDeviceToHost),
                 "hipMemcpy D2H"))
    {
        static_cast<void>(hipFree(dInput));
        static_cast<void>(hipFree(dOutput));
        return false;
    }

    static_cast<void>(hipFree(dInput));
    static_cast<void>(hipFree(dOutput));
    return true;
}

// Print a float vector.
static void printVector(const std::string& label, const std::vector<float>& v)
{
    std::cout << "  " << label << ": [";
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i > 0)
        {
            std::cout << ", ";
        }
        std::cout << v[i];
    }
    std::cout << "]\n";
}

// Print a 2D matrix stored row-major.
static void printMatrix(const std::string& label, const std::vector<float>& m, int rows, int cols)
{
    std::cout << "  " << label << " (" << rows << "x" << cols << "):\n";
    for(int r = 0; r < rows; ++r)
    {
        std::cout << "    [";
        for(int c = 0; c < cols; ++c)
        {
            if(c > 0)
            {
                std::cout << ", ";
            }
            std::cout << std::setw(6) << std::fixed << std::setprecision(1) << m[r * cols + c];
        }
        std::cout << "]\n";
    }
}

// Query and print loaded plugin paths via the frontend API.
static void printLoadedPlugins(hipdnnHandle_t handle)
{
    std::vector<std::filesystem::path> paths;
    auto err = hipdnn_frontend::getLoadedEnginePluginPaths(handle, paths);
    if(err.is_bad() || paths.empty())
    {
        std::cout << "  (no plugins loaded)\n";
        return;
    }

    for(size_t i = 0; i < paths.size(); ++i)
    {
        std::cout << "  [" << i << "] " << paths[i] << "\n";
    }
}

// ============================================================================
// Scenario 1: Load via HIPDNN_PLUGIN_DIR environment variable
// ============================================================================
static bool scenario1_EnvVariable()
{
    std::cout << "\n=== Scenario 1: Load via HIPDNN_PLUGIN_DIR ===\n";
    std::cout << "The HIPDNN_PLUGIN_DIR environment variable tells hipDNN where\n"
              << "to look for plugin shared libraries at handle creation time.\n"
              << "The plugin's GPU ReLU engine is used via HIPRTC.\n\n";

    std::string pluginDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_PLUGIN_DIR");
    if(pluginDir.empty())
    {
        std::cout << "  NOTE: Scenario 1 skipped. To run this scenario, set "
                     "HIPDNN_PLUGIN_DIR=</absolute/path/to/plugin/dir> "
                     "(the example_plugin library is assumed to be in that folder).\n";
        return true; // Not a failure, just skipped
    }

    std::cout << "  HIPDNN_PLUGIN_DIR = " << pluginDir << "\n";

    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "  ERROR: hipdnnCreate failed\n";
        return false;
    }

    std::cout << "  Loaded plugins:\n";
    printLoadedPlugins(handle);

    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> output;

    if(!runReluGraph(handle, "Scenario1", input, output))
    {
        hipdnnDestroy(handle);
        return false;
    }

    printVector("Input ", input);
    printVector("Output", output);

    hipdnnDestroy(handle);
    std::cout << "  Scenario 1 completed successfully.\n";
    return true;
}

// ============================================================================
// Scenario 2: Load with ADDITIVE mode
// ============================================================================
static bool scenario2_AdditiveMode(const std::string& pluginDir)
{
    std::cout << "\n=== Scenario 2: ADDITIVE Loading Mode ===\n";
    std::cout << "ADDITIVE mode loads the specified plugin directories alongside\n"
              << "any system-installed plugins.  This is the default mode.\n\n";

    std::vector<std::string> pluginPaths = {pluginDir};
    auto err = hipdnn_frontend::setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ADDITIVE);
    if(err.is_bad())
    {
        std::cerr << "  ERROR: setEnginePluginPaths (ADDITIVE) failed: " << err.err_msg << "\n";
        return false;
    }

    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "  ERROR: hipdnnCreate failed\n";
        return false;
    }

    std::cout << "  Loaded plugins (ADDITIVE):\n";
    printLoadedPlugins(handle);

    std::vector<float> input = {-3.0f, 0.0f, 3.0f, -0.5f};
    std::vector<float> output;

    if(!runReluGraph(handle, "Scenario2", input, output))
    {
        hipdnnDestroy(handle);
        return false;
    }

    printVector("Input ", input);
    printVector("Output", output);

    hipdnnDestroy(handle);
    std::cout << "  Scenario 2 completed successfully.\n";
    return true;
}

// ============================================================================
// Scenario 3: Load with ABSOLUTE mode
// ============================================================================
static bool scenario3_AbsoluteMode(const std::string& pluginDir)
{
    std::cout << "\n=== Scenario 3: ABSOLUTE Loading Mode ===\n";
    std::cout << "ABSOLUTE mode replaces all plugin search paths with only the\n"
              << "specified directories.  System-installed plugins are ignored.\n\n";

    std::vector<std::string> pluginPaths = {pluginDir};
    auto err = hipdnn_frontend::setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE);
    if(err.is_bad())
    {
        std::cerr << "  ERROR: setEnginePluginPaths (ABSOLUTE) failed: " << err.err_msg << "\n";
        return false;
    }

    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "  ERROR: hipdnnCreate failed\n";
        return false;
    }

    std::cout << "  Loaded plugins (ABSOLUTE -- only our plugin):\n";
    printLoadedPlugins(handle);

    std::vector<float> input = {-10.0f, 0.0f, 10.0f, -0.001f};
    std::vector<float> output;

    if(!runReluGraph(handle, "Scenario3", input, output))
    {
        hipdnnDestroy(handle);
        return false;
    }

    printVector("Input ", input);
    printVector("Output", output);

    hipdnnDestroy(handle);
    std::cout << "  Scenario 3 completed successfully.\n";
    return true;
}

// ============================================================================
// Scenario 4: Engine selection by name (ReLU on GPU)
// ============================================================================
static bool scenario4_EngineSelection(const std::string& pluginDir)
{
    std::cout << "\n=== Scenario 4: Engine Selection by Name (GPU ReLU) ===\n";
    std::cout << "set_preferred_engine_id_ext() selects a specific engine by name.\n"
              << "This bypasses the heuristic engine ranking and forces hipDNN to\n"
              << "use the named engine if it supports the requested operation.\n\n";

    // Use ABSOLUTE mode so we know exactly which plugins are loaded
    std::vector<std::string> pluginPaths = {pluginDir};
    hipdnn_frontend::setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE);

    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "  ERROR: hipdnnCreate failed\n";
        return false;
    }

    // Build a ReLU graph and explicitly select the GPU engine
    std::vector<int64_t> dims = {1, 1, 1, 6};
    auto graph = createReluGraph("Scenario4", dims);

    std::cout << "  Selecting engine: EXAMPLE_PLUGIN_RELU_ENGINE\n";
    graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_RELU_ENGINE");

    auto result = graph->build(handle);
    if(result.code != ErrorCode::OK)
    {
        std::cerr << "  ERROR: build failed: " << result.err_msg << "\n";
        hipdnnDestroy(handle);
        return false;
    }

    std::vector<float> input = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
    std::vector<float> output(6, -999.0f);

    const size_t bufferSize = input.size() * sizeof(float);
    float* dInput = nullptr;
    float* dOutput = nullptr;
    if(!checkHip(hipMalloc(&dInput, bufferSize), "hipMalloc")
       || !checkHip(hipMalloc(&dOutput, bufferSize), "hipMalloc")
       || !checkHip(hipMemcpy(dInput, input.data(), bufferSize, hipMemcpyHostToDevice),
                    "hipMemcpy H2D"))
    {
        static_cast<void>(hipFree(dInput));
        static_cast<void>(hipFree(dOutput));
        hipdnnDestroy(handle);
        return false;
    }

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dInput;
    variantPack[2] = dOutput;

    result = graph->execute(handle, variantPack, nullptr);
    if(result.code != ErrorCode::OK)
    {
        std::cerr << "  ERROR: execute failed: " << result.err_msg << "\n";
        static_cast<void>(hipFree(dInput));
        static_cast<void>(hipFree(dOutput));
        hipdnnDestroy(handle);
        return false;
    }

    checkHip(hipMemcpy(output.data(), dOutput, bufferSize, hipMemcpyDeviceToHost), "hipMemcpy D2H");

    printVector("Input ", input);
    printVector("Output", output);

    // Verify correctness
    bool correct = true;
    for(size_t i = 0; i < input.size(); ++i)
    {
        float expected = std::max(0.0f, input[i]);
        if(output[i] != expected)
        {
            std::cerr << "  MISMATCH at [" << i << "]: expected " << expected << ", got "
                      << output[i] << "\n";
            correct = false;
        }
    }
    if(correct)
    {
        std::cout << "  All outputs match expected ReLU(x) = max(0, x)\n";
    }

    static_cast<void>(hipFree(dInput));
    static_cast<void>(hipFree(dOutput));
    hipdnnDestroy(handle);
    std::cout << "  Scenario 4 completed successfully.\n";
    return true;
}

// ============================================================================
// Scenario 5: Convolution Forward on GPU
// ============================================================================
static bool scenario5_ConvForward(const std::string& pluginDir)
{
    std::cout << "\n=== Scenario 5: Convolution Forward on GPU ===\n";
    std::cout << "This demonstrates the GPU ConvFwd engine using HIPRTC.\n"
              << "A 5x5 input is convolved with a 3x3 filter (no padding,\n"
              << "stride=1) to produce a 3x3 output.\n\n";

    // Use ABSOLUTE mode
    std::vector<std::string> pluginPaths = {pluginDir};
    hipdnn_frontend::setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE);

    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "  ERROR: hipdnnCreate failed\n";
        return false;
    }

    // Dimensions: N=1, C=1, H=5, W=5, K=1, R=3, S=3
    const int64_t N = 1, C = 1, H = 5, W = 5;
    const int64_t K = 1, R = 3, S = 3;
    const int64_t outH = H - R + 1; // 3
    const int64_t outW = W - S + 1; // 3

    auto graph = std::make_shared<Graph>();
    graph->set_name("Scenario5_ConvFwd")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim({N, C, H, W})
        .set_stride({C * H * W, H * W, W, 1})
        .set_data_type(DataType::FLOAT);

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(2)
        .set_name("W")
        .set_dim({K, C, R, S})
        .set_stride({C * R * S, R * S, S, 1})
        .set_data_type(DataType::FLOAT);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});

    auto y = graph->conv_fprop(x, w, convAttrs);
    y->set_uid(3).set_data_type(DataType::FLOAT).set_output(true);

    std::cout << "  Selecting engine: EXAMPLE_PLUGIN_CONV_FWD_ENGINE\n";
    graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_CONV_FWD_ENGINE");

    auto result = graph->build(handle);
    if(result.code != ErrorCode::OK)
    {
        std::cerr << "  ERROR: build failed: " << result.err_msg << "\n";
        hipdnnDestroy(handle);
        return false;
    }

    // 5x5 input matrix (values 1..25)
    // clang-format off
    std::vector<float> inputData = {
         1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
         6.0f,  7.0f,  8.0f,  9.0f, 10.0f,
        11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f, 25.0f
    };
    // clang-format on

    // 3x3 edge-detection-style filter
    // clang-format off
    std::vector<float> weightData = {
         1.0f,  0.0f, -1.0f,
         2.0f,  0.0f, -2.0f,
         1.0f,  0.0f, -1.0f
    };
    // clang-format on

    const size_t inputSize = inputData.size() * sizeof(float);
    const size_t weightSize = weightData.size() * sizeof(float);
    const size_t outputSize = static_cast<size_t>(N * K * outH * outW) * sizeof(float);

    float* dInput = nullptr;
    float* dWeight = nullptr;
    float* dOutput = nullptr;
    if(!checkHip(hipMalloc(&dInput, inputSize), "hipMalloc input")
       || !checkHip(hipMalloc(&dWeight, weightSize), "hipMalloc weight")
       || !checkHip(hipMalloc(&dOutput, outputSize), "hipMalloc output"))
    {
        static_cast<void>(hipFree(dInput));
        static_cast<void>(hipFree(dWeight));
        static_cast<void>(hipFree(dOutput));
        hipdnnDestroy(handle);
        return false;
    }

    checkHip(hipMemcpy(dInput, inputData.data(), inputSize, hipMemcpyHostToDevice), "H2D input");
    checkHip(hipMemcpy(dWeight, weightData.data(), weightSize, hipMemcpyHostToDevice),
             "H2D weight");

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dInput;
    variantPack[2] = dWeight;
    variantPack[3] = dOutput;

    result = graph->execute(handle, variantPack, nullptr);
    if(result.code != ErrorCode::OK)
    {
        std::cerr << "  ERROR: execute failed: " << result.err_msg << "\n";
        static_cast<void>(hipFree(dInput));
        static_cast<void>(hipFree(dWeight));
        static_cast<void>(hipFree(dOutput));
        hipdnnDestroy(handle);
        return false;
    }

    std::vector<float> outputData(static_cast<size_t>(N * K * outH * outW));
    checkHip(hipMemcpy(outputData.data(), dOutput, outputSize, hipMemcpyDeviceToHost),
             "D2H output");

    printMatrix("Input", inputData, static_cast<int>(H), static_cast<int>(W));
    std::cout << "\n";
    printMatrix("Filter", weightData, static_cast<int>(R), static_cast<int>(S));
    std::cout << "\n";
    printMatrix("Output", outputData, static_cast<int>(outH), static_cast<int>(outW));

    static_cast<void>(hipFree(dInput));
    static_cast<void>(hipFree(dWeight));
    static_cast<void>(hipFree(dOutput));
    hipdnnDestroy(handle);
    std::cout << "\n  Scenario 5 completed successfully.\n";
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    std::cout << "hipDNN Example Plugin Sample Application\n";
    std::cout << "=========================================\n";
    std::cout << "All operations execute on GPU via HIPRTC-compiled kernels.\n";

    // Check GPU availability
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
    {
        std::cerr << "ERROR: No GPU detected. This sample requires a GPU with ROCm support.\n";
        return 1;
    }

    hipDeviceProp_t props;
    static_cast<void>(hipGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << " (" << props.gcnArchName << ")\n";

    // Determine plugin directory: CLI argument > HIPDNN_PLUGIN_DIR > executable directory
    std::string pluginDir;
    if(argc > 1)
    {
        pluginDir = argv[1];
        std::cout << "Plugin directory (from argument): " << pluginDir << "\n";
    }
    else
    {
        pluginDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_PLUGIN_DIR");
        if(!pluginDir.empty())
        {
            std::cout << "Plugin directory (from HIPDNN_PLUGIN_DIR): " << pluginDir << "\n";
        }
        else
        {
            auto exeDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();
            pluginDir = std::filesystem::absolute(exeDir).string();
            std::cout << "Plugin directory (from executable location): " << pluginDir << "\n";
        }
    }

    bool allPassed = true;

    allPassed = scenario1_EnvVariable() && allPassed;
    allPassed = scenario2_AdditiveMode(pluginDir) && allPassed;
    allPassed = scenario3_AbsoluteMode(pluginDir) && allPassed;
    allPassed = scenario4_EngineSelection(pluginDir) && allPassed;
    allPassed = scenario5_ConvForward(pluginDir) && allPassed;

    std::cout << "\n=========================================\n";
    if(allPassed)
    {
        std::cout << "All scenarios completed successfully.\n";
        return 0;
    }

    std::cerr << "Some scenarios failed.\n";
    return 1;
}
