// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE ADAPTATION: Adapt this sample to exercise your plugin's operations. Scenario 1 (engine
// selection + knob modification) and Scenario 2 (convenience build) demonstrate the two API
// styles. Keep the structure and replace the graph construction and verification logic. Scenario 3
// (plugin loading modes) can be kept as-is with your plugin name updated, customized for your
// specific environment, or discarded if the plugin is installed directly to the ROCm hipDNN plugin
// folder where it will be loaded automatically with all hipDNN plugins.
//
// Sample application demonstrating how to load a hipDNN engine plugin and
// execute graphs using the plugin's GPU engines. This program serves as both
// a demonstration and an end-to-end acceptance test.
//
// Three scenarios are demonstrated:
//   1. ReLU forward with engine selection and knob modification (explicit 6-step
//      build, leaky ReLU via negative_slope knob, verification)
//   2. Convolution forward with engine selection (build() convenience, Tensor
//      class for GPU memory management, verification with hardcoded values)
//   3. Plugin loading modes (ADDITIVE, ABSOLUTE, presence verification)
//
// When no GPU is detected, only non-GPU portions run (plugin
// loading, presence verification, graph construction).
//
// Plugin path resolution:
//   1. If a command-line argument is provided, that path is used. This can be
//      a directory containing the plugin or the full path to the plugin file.
//   2. Otherwise, if HIPDNN_PLUGIN_DIR is set in the environment, that is used.
//   3. Otherwise, the directory containing this executable is used as a last resort.
//
// Prerequisites:
//   - hipDNN installed
//   - ROCm with HIPRTC and a compatible GPU (for GPU execution portions)
//   - The example_provider_plugin shared library (built in the same CMake project)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Utilities.hpp>

using namespace hipdnn_frontend;

// ============================================================================
// Helper Functions
// ============================================================================

// HIP error checking helper
static bool checkHip(hipError_t err, const char* msg)
{
    if(err != hipSuccess)
    {
        std::cerr << "  HIP ERROR: " << msg << ": " << hipGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

// Check a hipDNN graph operation result and print an error message on failure.
// Returns true if the result is OK, false otherwise. The caller decides how
// to handle the failure (cleanup, early return, etc.).
static bool checkGraphResult(const hipdnn_frontend::error_t& result, const char* stepName)
{
    if(result.code != ErrorCode::OK)
    {
        std::cerr << "  ERROR: " << stepName << " failed: " << result.err_msg << "\n";
        return false;
    }
    return true;
}

// Result of createReluGraph: the graph plus input/output tensor attributes.
struct ReluGraph
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> x;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> y;
};

// Build a pointwise ReLU forward graph with the given tensor dimensions.
static ReluGraph createReluGraph(const std::string& name, const std::vector<int64_t>& dims)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name(name)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim(dims)
        .set_stride({dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1})
        .set_data_type(DataType::FLOAT);

    hipdnn_frontend::graph::PointwiseAttributes attrs;
    attrs.set_name("relu_fwd");
    attrs.set_mode(PointwiseMode::RELU_FWD);

    auto y = graph->pointwise(x, attrs);
    y->set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

    return {graph, x, y};
}

// Run the full build/execute sequence for a ReLU graph using GPU device memory.
// Returns true on success.
static bool runReluGraph(hipdnnHandle_t handle,
                         const std::string& graphName,
                         const std::vector<float>& input,
                         std::vector<float>& output)
{
    auto numElements = static_cast<int64_t>(input.size());
    const std::vector<int64_t> dims = {1, 1, 1, numElements};

    auto [graph, x, y] = createReluGraph(graphName, dims);
    graph->set_preferred_engine_id_ext("EXAMPLE_PROVIDER_RELU_ENGINE");

    auto result = graph->build(handle);
    if(!checkGraphResult(result, "build"))
    {
        return false;
    }

    // Use Tensor class for GPU memory management (host+device allocation,
    // automatic H2D/D2H transfers)
    hipdnn_data_sdk::utilities::Tensor<float> xTensor(dims);
    xTensor.fillWithData(input.data(), input.size() * sizeof(float));

    hipdnn_data_sdk::utilities::Tensor<float> yTensor(dims);
    yTensor.fillWithValue(0.0f);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();

    result = graph->execute(handle, variantPack, nullptr);
    if(!checkGraphResult(result, "execute"))
    {
        return false;
    }

    // Transfer output from device to host via Tensor's managed memory
    yTensor.memory().markDeviceModified();
    auto yHostPtr = yTensor.memory().hostData();
    output.assign(yHostPtr, yHostPtr + input.size());

    return true;
}

// Print knob value
static std::string knobValueToString(const KnobValueVariant& value)
{
    if(std::holds_alternative<int64_t>(value))
    {
        return std::to_string(std::get<int64_t>(value));
    }
    if(std::holds_alternative<double>(value))
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << std::get<double>(value);
        return oss.str();
    }
    if(std::holds_alternative<std::string>(value))
    {
        return "\"" + std::get<std::string>(value) + "\"";
    }
    return "(unknown)";
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
static void printMatrix(const std::string& label, const float* m, int rows, int cols)
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
    std::cout << "\n";
}

// Query and print loaded plugin paths via the frontend API.
static void printLoadedPlugins(hipdnnHandle_t handle)
{
    std::vector<std::filesystem::path> paths;
    auto err = getLoadedEnginePluginPaths(handle, paths);
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

// Check whether the example_provider_plugin is present among loaded plugins.
// Returns true if found, false otherwise.
static bool verifyPluginPresence(hipdnnHandle_t handle, const std::string& modeLabel)
{
    std::vector<std::filesystem::path> paths;
    auto err = getLoadedEnginePluginPaths(handle, paths);
    if(err.is_bad() || paths.empty())
    {
        std::cerr << "  ERROR: No plugins loaded after " << modeLabel << "\n";
        return false;
    }

    bool found = false;
    for(const auto& path : paths)
    {
        if(path.string().find("example_provider_plugin") != std::string::npos)
        {
            found = true;
            break;
        }
    }

    if(!found)
    {
        std::cerr << "  ERROR: example_provider_plugin not found among loaded plugins after "
                  << modeLabel << "\n";
        return false;
    }

    std::cout << "  Plugin verified present after " << modeLabel << "\n";
    return true;
}

// ============================================================================
// Scenario 1: ReLU Forward with Engine Selection and Knob Modification
// ============================================================================
static bool scenario1ReluForward(const std::vector<std::string>& pluginPaths, bool hasGpu)
{
    std::cout << "\n=== Scenario 1: ReLU Forward with Engine Selection and Knob Modification ===\n";
    std::cout
        << "Demonstrates the explicit 6-step graph build sequence, engine selection by name,\n"
        << "and knob modification for leaky ReLU.\n\n";

    // Use ABSOLUTE mode so the example plugin is loaded regardless of
    // system-installed plugins
    setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE);

    auto [handle, handleErr] = hipdnn_frontend::createHipdnnHandle();
    if(handleErr.is_bad())
    {
        std::cerr << "  ERROR: createHipdnnHandle failed\n";
        return false;
    }

    std::vector<int64_t> dims = {1, 1, 1, 6};

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("Scenario1_ReLU")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim(dims)
        .set_stride({dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1})
        .set_data_type(DataType::FLOAT);

    hipdnn_frontend::graph::PointwiseAttributes attrs;
    attrs.set_name("relu_fwd");
    attrs.set_mode(PointwiseMode::RELU_FWD);

    auto y = graph->pointwise(x, attrs);
    y->set_uid(2).set_data_type(DataType::FLOAT).set_output(true);

    // Select the example plugin's GPU ReLU engine by name
    std::cout << "  Selecting engine: EXAMPLE_PROVIDER_RELU_ENGINE\n\n";
    graph->set_preferred_engine_id_ext("EXAMPLE_PROVIDER_RELU_ENGINE");

    auto result = graph->validate();
    if(!checkGraphResult(result, "validate"))
    {
        return false;
    }

    result = graph->build_operation_graph(*handle);
    if(!checkGraphResult(result, "build_operation_graph"))
    {
        return false;
    }

    // Query available engines and their knobs
    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    if(!checkGraphResult(result, "get_ranked_engine_ids"))
    {
        return false;
    }

    if(rankedEngineIds.empty())
    {
        std::cerr << "  ERROR: No engines available\n";
        return false;
    }

    const int64_t engineId = rankedEngineIds[0];

    std::vector<Knob> knobs;
    result = graph->get_knobs_for_engine(engineId, knobs);
    if(!checkGraphResult(result, "get_knobs_for_engine"))
    {
        return false;
    }

    std::cout << "  Engine has " << knobs.size() << " knob(s):\n";
    for(const auto& knob : knobs)
    {
        std::cout << "    " << knob.knobId()
                  << " (default=" << knobValueToString(knob.defaultValue()) << ")\n";
    }

    // Set negative_slope = 0.1
    std::vector<KnobSetting> settings;
    settings.emplace_back("example.relu.negative_slope", 0.1);
    std::cout << "  Setting example.relu.negative_slope = 0.1\n\n";

    result = graph->create_execution_plan_ext(engineId, settings);
    if(!checkGraphResult(result, "create_execution_plan_ext"))
    {
        return false;
    }

    result = graph->check_support();
    if(!checkGraphResult(result, "check_support"))
    {
        return false;
    }

    result = graph->build_plans();
    if(!checkGraphResult(result, "build_plans"))
    {
        return false;
    }

    std::cout << "  Graph built successfully.\n";

    if(!hasGpu)
    {
        std::cout << "  GPU execution skipped (no GPU detected).\n";
        return true;
    }

    // GPU execution and verification
    std::vector<float> input = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
    std::vector<float> output(input.size(), -999.0f);

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
        return false;
    }

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = dInput;
    variantPack[2] = dOutput;

    result = graph->execute(*handle, variantPack, nullptr);
    if(!checkGraphResult(result, "execute"))
    {
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

    printVector("Input ", input);
    printVector("Output", output);

    // Verify correctness: ReLU(x) = x >= 0 ? x : 0.1 * x
    bool correct = true;
    for(size_t i = 0; i < input.size(); ++i)
    {
        const float expected = input[i] >= 0.0f ? input[i] : 0.1f * input[i];
        if(std::abs(output[i] - expected) > 1e-6f)
        {
            std::cerr << "  MISMATCH at [" << i << "]: expected " << expected << ", got "
                      << output[i] << "\n";
            correct = false;
        }
    }
    if(correct)
    {
        std::cout << "  All outputs match expected ReLU (slope=0.1)\n";
    }

    static_cast<void>(hipFree(dInput));
    static_cast<void>(hipFree(dOutput));

    if(!correct)
    {
        return false;
    }

    std::cout << "\n  Scenario 1 completed successfully.\n";
    return true;
}

// ============================================================================
// Scenario 2: Convolution Forward with Engine Selection
// ============================================================================
static bool scenario2ConvForward(const std::vector<std::string>& pluginPaths, bool hasGpu)
{
    std::cout << "\n=== Scenario 2: Convolution Forward with Engine Selection ===\n";
    std::cout
        << "Demonstrates ConvFwd graph construction with NCHW/KCRS tensor layouts, the build()\n"
        << "convenience method, Tensor class for GPU memory management, and verification\n"
        << "against hardcoded expected values.\n\n";

    // Use ABSOLUTE mode so the example plugin is loaded regardless of
    // system-installed plugins
    setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE);

    auto [handle, handleErr] = hipdnn_frontend::createHipdnnHandle();
    if(handleErr.is_bad())
    {
        std::cerr << "  ERROR: createHipdnnHandle failed\n";
        return false;
    }

    // Dimensions: N=1, C=1, H=4, W=4, K=1, R=3, S=3
    // No padding, stride=1, dilation=1 => output 1x1x2x2
    // NOLINTBEGIN(readability-identifier-naming)
    const int64_t N = 1;
    const int64_t C = 1;
    const int64_t H = 4;
    const int64_t W = 4;
    const int64_t K = 1;
    const int64_t R = 3;
    const int64_t S = 3;
    // NOLINTEND(readability-identifier-naming)
    const int64_t outH = H - R + 1; // 2
    const int64_t outW = W - S + 1; // 2

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("Scenario2_ConvFwd")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Input tensor X: NCHW layout (using makeTensorAttributes + generateStrides)
    auto xAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        hipdnn_frontend::graph::makeTensorAttributes(
            "X",
            DataType::FLOAT,
            {N, C, H, W},
            hipdnn_data_sdk::utilities::generateStrides({N, C, H, W})));

    // Weight tensor W: KCRS layout (using makeTensorAttributes + generateStrides)
    auto wAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        hipdnn_frontend::graph::makeTensorAttributes(
            "W",
            DataType::FLOAT,
            {K, C, R, S},
            hipdnn_data_sdk::utilities::generateStrides({K, C, R, S})));

    hipdnn_frontend::graph::ConvFpropAttributes convAttrs;
    convAttrs.set_padding({0, 0}).set_stride({1, 1}).set_dilation({1, 1});

    // Output tensor Y (attributes derived from conv_fprop)
    auto yOut = graph->conv_fprop(xAttr, wAttr, convAttrs);
    yOut->set_data_type(DataType::FLOAT).set_output(true);

    // Select the example plugin's ConvFwd engine by name
    std::cout << "  Selecting engine: EXAMPLE_PROVIDER_CONV_FWD_ENGINE\n\n";
    graph->set_preferred_engine_id_ext("EXAMPLE_PROVIDER_CONV_FWD_ENGINE");

    // Use the build() convenience method (contrasts with Scenario 1's explicit
    // 6-step sequence, showing both API styles)
    auto result = graph->build(*handle);
    if(!checkGraphResult(result, "build"))
    {
        return false;
    }

    std::cout << "  Graph built successfully.\n";

    if(!hasGpu)
    {
        std::cout << "  GPU execution skipped (no GPU detected).\n";
        return true;
    }

    // 4x4 input matrix (values 1..16), managed by Tensor class
    // clang-format off
    std::vector<float> inputData = {
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    // clang-format on

    // 3x3 all-ones filter for easy verification
    std::vector<float> weightData(static_cast<size_t>(R * S), 1.0f);

    // Use Tensor class for GPU memory management (host+device allocation,
    // automatic H2D/D2H transfers) instead of raw hipMalloc/hipMemcpy/hipFree
    hipdnn_data_sdk::utilities::Tensor<float> xTensor({N, C, H, W});
    xTensor.fillWithData(inputData.data(), inputData.size() * sizeof(float));

    hipdnn_data_sdk::utilities::Tensor<float> wTensor({K, C, R, S});
    wTensor.fillWithData(weightData.data(), weightData.size() * sizeof(float));

    hipdnn_data_sdk::utilities::Tensor<float> yTensor({N, K, outH, outW});
    yTensor.fillWithValue(0.0f);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
    variantPack[yOut->get_uid()] = yTensor.memory().deviceData();

    result = graph->execute(*handle, variantPack, nullptr);
    if(!checkGraphResult(result, "execute"))
    {
        return false;
    }

    // Transfer output from device to host via Tensor's managed memory
    yTensor.memory().markDeviceModified();
    auto yHostPtr = yTensor.memory().hostData();

    printMatrix("Input", inputData.data(), static_cast<int>(H), static_cast<int>(W));
    printMatrix("Filter (all ones)", weightData.data(), static_cast<int>(R), static_cast<int>(S));
    printMatrix("Output", yHostPtr, static_cast<int>(outH), static_cast<int>(outW));

    // With a 4x4 input (values 1-16) and a 3x3 all-ones filter (no padding,
    // stride 1), each output element is the sum of a 3x3 window:
    //   output[0,0] = 1+2+3+5+6+7+9+10+11       = 54
    //   output[0,1] = 2+3+4+6+7+8+10+11+12       = 63
    //   output[1,0] = 5+6+7+9+10+11+13+14+15     = 90
    //   output[1,1] = 6+7+8+10+11+12+14+15+16    = 99
    std::vector<float> expected = {54.0f, 63.0f, 90.0f, 99.0f};

    bool correct = true;
    for(size_t i = 0; i < expected.size(); ++i)
    {
        if(std::abs(yHostPtr[i] - expected[i]) > 1e-5f)
        {
            std::cerr << "  MISMATCH at [" << i << "]: expected " << expected[i] << ", got "
                      << yHostPtr[i] << "\n";
            correct = false;
        }
    }
    if(correct)
    {
        std::cout << "  All outputs match expected values {54, 63, 90, 99}\n";
    }

    if(!correct)
    {
        return false;
    }

    std::cout << "\n  Scenario 2 completed successfully.\n";
    return true;
}

// ============================================================================
// Scenario 3: Plugin Loading Modes
// ============================================================================
static bool scenario3PluginLoadingModes(const std::vector<std::string>& pluginPaths, bool hasGpu)
{
    std::cout << "\n=== Scenario 3: Plugin Loading Modes ===\n";
    std::cout << "Demonstrates ADDITIVE and ABSOLUTE loading modes with presence verification "
                 "after each mode.\n\n";

    // ---- ADDITIVE mode ----
    std::cout << "  --- ADDITIVE mode ---\n";
    std::cout << "  ADDITIVE mode loads the specified plugin directories alongside\n"
              << "  any system-installed plugins. This is the default mode.\n\n";

    auto err = setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ADDITIVE);
    if(err.is_bad())
    {
        std::cerr << "  ERROR: setEnginePluginPaths (ADDITIVE) failed: " << err.err_msg << "\n";
        return false;
    }

    {
        auto [handle, handleErr] = hipdnn_frontend::createHipdnnHandle();
        if(handleErr.is_bad())
        {
            std::cerr << "  ERROR: createHipdnnHandle failed (ADDITIVE)\n";
            return false;
        }

        std::cout << "  Loaded plugins (ADDITIVE):\n";
        printLoadedPlugins(*handle);

        if(!verifyPluginPresence(*handle, "ADDITIVE loading"))
        {
            return false;
        }
    }

    // ---- ABSOLUTE mode ----
    std::cout << "\n  --- ABSOLUTE mode ---\n";
    std::cout << "  ABSOLUTE mode replaces all plugin search paths with only the\n"
              << "  specified directories. System-installed plugins are ignored.\n\n";

    err = setEnginePluginPaths(pluginPaths, PluginLoadingMode::MODE_ABSOLUTE);
    if(err.is_bad())
    {
        std::cerr << "  ERROR: setEnginePluginPaths (ABSOLUTE) failed: " << err.err_msg << "\n";
        return false;
    }

    {
        auto [handle, handleErr] = hipdnn_frontend::createHipdnnHandle();
        if(handleErr.is_bad())
        {
            std::cerr << "  ERROR: createHipdnnHandle failed (ABSOLUTE)\n";
            return false;
        }

        std::cout << "  Loaded plugins (ABSOLUTE, only our plugin):\n";
        printLoadedPlugins(*handle);

        if(!verifyPluginPresence(*handle, "ABSOLUTE loading"))
        {
            return false;
        }

        // Run a quick ReLU execution to confirm the loaded plugin is functional
        if(hasGpu)
        {
            std::cout << "\n  Running quick ReLU to confirm plugin functionality...\n";
            std::vector<float> input = {-2.0f, 0.0f, 3.0f};
            std::vector<float> output;
            if(!runReluGraph(*handle, "Scenario3_QuickReLU", input, output))
            {
                return false;
            }

            // Verify the output
            bool correct = true;
            for(size_t i = 0; i < input.size(); ++i)
            {
                const float expected = std::max(0.0f, input[i]);
                if(output[i] != expected)
                {
                    std::cerr << "  MISMATCH at [" << i << "]: expected " << expected << ", got "
                              << output[i] << "\n";
                    correct = false;
                }
            }
            if(correct)
            {
                std::cout << "  Quick ReLU verification passed.\n";
            }
            else
            {
                return false;
            }
        }
        else
        {
            std::cout << "\n  GPU execution skipped (no GPU detected).\n";
        }
    }

    // ---- Optional: HIPDNN_PLUGIN_DIR environment variable ----
    const std::string envPluginDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_PLUGIN_DIR");
    if(!envPluginDir.empty())
    {
        std::cout << "\n  --- HIPDNN_PLUGIN_DIR environment variable ---\n";
        std::cout << "  HIPDNN_PLUGIN_DIR = " << envPluginDir << "\n";
        std::cout << "  hipDNN uses this environment variable at handle creation time\n"
                  << "  to discover plugin shared libraries.\n";

        auto [handle, handleErr] = hipdnn_frontend::createHipdnnHandle();
        if(handleErr.is_bad())
        {
            std::cerr << "  ERROR: createHipdnnHandle failed (env var)\n";
            return false;
        }

        std::cout << "  Loaded plugins (via env var):\n";
        printLoadedPlugins(*handle);
    }
    else
    {
        std::cout << "\n  NOTE: HIPDNN_PLUGIN_DIR is not set.\n"
                  << "  Set it to demonstrate environment-variable-based plugin loading.\n";
    }

    std::cout << "\n  Scenario 3 completed successfully.\n";
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    try
    {
        std::cout << "hipDNN Example Plugin Sample Application\n";
        std::cout << "=========================================\n";

        // Check GPU availability
        bool hasGpu = false;
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0)
        {
            hasGpu = true;
            hipDeviceProp_t props;
            static_cast<void>(hipGetDeviceProperties(&props, 0));
            std::cout << "GPU: " << props.name << " (" << props.gcnArchName << ")\n";
        }
        else
        {
            std::cout << "No GPU detected. Skipping GPU execution scenarios.\n"
                      << "Running non-GPU portions only.\n";
        }

        // Determine plugin path: CLI argument > HIPDNN_PLUGIN_DIR > executable directory.
#ifdef _WIN32
        const std::string pluginFilename = "example_provider_plugin.dll";
#else
        const std::string pluginFilename = "libexample_provider_plugin.so";
#endif

        std::string pluginPath;
        if(argc > 1)
        {
            pluginPath = argv[1];
            std::cout << "Plugin path (from argument): " << pluginPath << "\n";
        }
        else
        {
            pluginPath = hipdnn_data_sdk::utilities::getEnv("HIPDNN_PLUGIN_DIR");
            if(!pluginPath.empty())
            {
                std::cout << "Plugin path (from HIPDNN_PLUGIN_DIR): " << pluginPath << "\n";
            }
            else
            {
                auto exeDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();
                pluginPath = std::filesystem::absolute(exeDir).string();
                std::cout << "Plugin path (from executable location): " << pluginPath << "\n";
            }
        }

        // To assist in debugging issues when running this sample, determine whether
        // the plugin exists before running the scenarios.
        std::vector<std::string> pluginPaths;
        auto fsPath = std::filesystem::path(pluginPath);
        if(std::filesystem::is_regular_file(fsPath))
        {
            if(fsPath.filename().string().find("example_provider_plugin") != std::string::npos)
            {
                pluginPaths.push_back(pluginPath);
            }
        }
        else if(std::filesystem::is_directory(fsPath))
        {
            if(std::filesystem::exists(fsPath / pluginFilename))
            {
                pluginPaths.push_back(pluginPath);
            }
        }

        std::cout << "Plugin paths:\n";
        for(size_t i = 0; i < pluginPaths.size(); ++i)
        {
            std::cout << "  [" << i << "] " << pluginPaths[i] << "\n";
        }

        if(pluginPaths.empty())
        {
            std::cerr << "\nERROR: Example plugin library (" << pluginFilename
                      << ") not found.\n\n";
            std::cerr << "Path provided: " << pluginPath << "\n\n";
            std::cerr << "Usage: " << argv[0] << " [plugin_path]\n\n"
                      << "The plugin path can be:\n"
                      << "  - A directory containing " << pluginFilename << "\n"
                      << "  - The full path to the plugin file itself\n\n"
                      << "Plugin path resolution (when no argument is provided):\n"
                      << "  1. If HIPDNN_PLUGIN_DIR is set, that is used.\n"
                      << "  2. Otherwise, the directory containing this executable is used.\n";
            return 1;
        }

        bool allPassed = true;

        allPassed = scenario1ReluForward(pluginPaths, hasGpu) && allPassed;
        allPassed = scenario2ConvForward(pluginPaths, hasGpu) && allPassed;
        allPassed = scenario3PluginLoadingModes(pluginPaths, hasGpu) && allPassed;

        std::cout << "\n=========================================\n";
        if(allPassed)
        {
            std::cout << "All scenarios completed successfully.\n";
            return 0;
        }

        std::cerr << "Some scenarios failed.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }
    catch(...)
    {
        std::fprintf(stderr, "Fatal error: unknown exception\n");
        return 1;
    }
}
