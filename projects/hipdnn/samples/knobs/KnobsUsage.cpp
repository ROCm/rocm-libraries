// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

// Helper function to print knob value based on its type
std::string knobValueToString(const KnobValueVariant& value)
{
    if(std::holds_alternative<int64_t>(value))
    {
        return std::to_string(std::get<int64_t>(value));
    }
    else if(std::holds_alternative<double>(value))
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << std::get<double>(value);
        return oss.str();
    }
    else if(std::holds_alternative<std::string>(value))
    {
        return "\"" + std::get<std::string>(value) + "\"";
    }
    return "<unknown>";
}

// Helper function to print knob value type
std::string knobValueTypeToString(KnobValueType type)
{
    switch(type)
    {
    case KnobValueType::INT64:
        return "int64";
    case KnobValueType::FLOAT64:
        return "float64";
    case KnobValueType::STRING:
        return "string";
    default:
        return "unknown";
    }
}

void printSeparator(const std::string& title = "")
{
    std::cout << "\n" << std::string(80, '=') << "\n";
    if(!title.empty())
    {
        std::cout << "  " << title << "\n";
        std::cout << std::string(80, '=') << "\n";
    }
}

void demonstrateKnobQuery(hipdnnHandle_t handle, int64_t engineId, Graph& graph)
{
    printSeparator("1. Querying Available Knobs");

    std::cout << "Querying knobs for engine ID: " << engineId << "\n\n";

    // Method 1: Get knobs as a vector
    std::vector<Knob> knobs;
    auto error = graph.get_knobs_for_engine(engineId, knobs);

    if(!error.is_good())
    {
        std::cerr << "Error getting knobs: " << error.get_message() << "\n";
        return;
    }

    std::cout << "Found " << knobs.size() << " knobs for this engine:\n\n";

    for(const auto& knob : knobs)
    {
        std::cout << "Knob: " << knob.knobId() << "\n";
        std::cout << "  Description: " << knob.description() << "\n";
        std::cout << "  Type: " << knobValueTypeToString(knob.valueType()) << "\n";
        std::cout << "  Default Value: " << knobValueToString(knob.defaultValue()) << "\n";
        std::cout << "  Deprecated: " << (knob.isDeprecated() ? "yes" : "no") << "\n";

        if(const auto* constraint = knob.constraint())
        {
            std::cout << "  Constraints: " << constraint->toString() << "\n";
        }
        else
        {
            std::cout << "  Constraints: none\n";
        }

        std::cout << "\n";
    }

    // Method 2: Get knobs as a map for easier lookup
    printSeparator("2. Using Knob Lookup Map");

    std::unordered_map<std::string, Knob> knobMap;
    error = graph.get_knob_lookup_for_engine(engineId, knobMap);

    if(error.is_good())
    {
        std::cout << "Knob map contains " << knobMap.size() << " entries\n\n";

        // Example: Look up a specific knob
        auto it = knobMap.find("global.benchmarking");
        if(it != knobMap.end())
        {
            const auto& benchmarkingKnob = it->second;
            std::cout << "Found 'global.benchmarking' knob:\n";
            std::cout << "  Description: " << benchmarkingKnob.description() << "\n";
            std::cout << "  Default: " << knobValueToString(benchmarkingKnob.defaultValue())
                      << "\n\n";
        }
        else
        {
            std::cout << "'global.benchmarking' knob not found for this engine\n\n";
        }
    }
}

void demonstrateDefaultKnobs(hipdnnHandle_t handle, int64_t engineId, Graph& graph)
{
    printSeparator("3. Using Default Knob Values");

    std::cout << "Creating execution plan with default knob values...\n\n";

    // Create execution plan without specifying any knob settings
    // All knobs will use their default values
    std::vector<KnobSetting> settings; // Empty settings = use all defaults
    auto error = graph.create_execution_plan_ext(engineId, settings);

    if(error.is_good())
    {
        std::cout << "Success! Execution plan created with default knob settings\n";
        std::cout << "This is the simplest way to use knobs - just use the defaults\n\n";
    }
    else
    {
        std::cerr << "Error: " << error.get_message() << "\n\n";
    }
}

void demonstrateSettingKnobs(hipdnnHandle_t handle, int64_t engineId, Graph& graph)
{
    printSeparator("4. Setting Custom Knob Values");

    std::cout << "Creating execution plan with custom knob settings...\n\n";

    // Create custom knob settings
    std::vector<KnobSetting> settings;

    // Example 1: Enable benchmarking (integer knob)
    std::cout << "Setting 'global.benchmarking' = 1 (enabled)\n";
    settings.emplace_back("global.benchmarking", static_cast<int64_t>(1));

    // Example 2: Set workspace limit if available (integer knob)
    // Note: This knob is only available for convolution operations
    std::cout << "Setting 'global.workspace_size_limit' = 64MB\n";
    settings.emplace_back("global.workspace_size_limit", static_cast<int64_t>(64 * 1024 * 1024));

    std::cout << "\nCreating execution plan with " << settings.size()
              << " custom knob settings...\n";

    auto error = graph.create_execution_plan_ext(engineId, settings);

    if(error.is_good())
    {
        std::cout << "Success! Execution plan created with custom knob settings\n";
        std::cout << "Note: Unknown knobs are ignored with a warning (not an error)\n\n";
    }
    else
    {
        std::cerr << "Error: " << error.get_message() << "\n";
        std::cout << "This might happen if:\n";
        std::cout << "  - A knob value is outside its valid range\n";
        std::cout << "  - Required constraints are violated\n\n";
    }
}

void demonstrateKnobValidation(hipdnnHandle_t handle, int64_t engineId, Graph& graph)
{
    printSeparator("5. Knob Validation");

    std::cout << "Demonstrating knob validation...\n\n";

    // Get knobs to understand constraints
    std::vector<Knob> knobs;
    auto error = graph.get_knobs_for_engine(engineId, knobs);

    if(!error.is_good())
    {
        std::cerr << "Error getting knobs: " << error.get_message() << "\n";
        return;
    }

    // Try to validate a knob setting against its constraints
    for(const auto& knob : knobs)
    {
        if(knob.knobId() == "global.benchmarking")
        {
            std::cout << "Validating 'global.benchmarking' knob settings:\n\n";

            // Valid setting
            KnobSetting validSetting("global.benchmarking", static_cast<int64_t>(1));
            auto validationError = knob.validate(validSetting);

            if(validationError.is_good())
            {
                std::cout << "  Setting value = 1: VALID ✓\n";
            }
            else
            {
                std::cout << "  Setting value = 1: INVALID - " << validationError.get_message()
                          << "\n";
            }

            // Invalid setting (outside valid range 0-1)
            KnobSetting invalidSetting("global.benchmarking", static_cast<int64_t>(5));
            validationError = knob.validate(invalidSetting);

            if(validationError.is_good())
            {
                std::cout << "  Setting value = 5: VALID ✓\n";
            }
            else
            {
                std::cout << "  Setting value = 5: INVALID - " << validationError.get_message()
                          << " ✗\n";
            }

            std::cout << "\n";
            break;
        }
    }
}

void demonstrateKnobTypes(hipdnnHandle_t handle)
{
    printSeparator("6. Different Knob Value Types");

    std::cout << "Knobs support three value types:\n\n";

    // Integer knob
    std::cout << "1. Integer (int64_t) knobs:\n";
    KnobSetting intKnob("example.int_knob", static_cast<int64_t>(42));
    std::cout << "   KnobSetting(\"example.int_knob\", 42);\n";
    std::cout << "   Value: " << knobValueToString(intKnob.value()) << "\n\n";

    // Float knob
    std::cout << "2. Float (double) knobs:\n";
    KnobSetting floatKnob("example.float_knob", 3.14159);
    std::cout << "   KnobSetting(\"example.float_knob\", 3.14159);\n";
    std::cout << "   Value: " << knobValueToString(floatKnob.value()) << "\n\n";

    // String knob
    std::cout << "3. String knobs:\n";
    KnobSetting stringKnob("example.string_knob", std::string("algorithm_choice"));
    std::cout << "   KnobSetting(\"example.string_knob\", \"algorithm_choice\");\n";
    std::cout << "   Value: " << knobValueToString(stringKnob.value()) << "\n\n";

    std::cout << "Note: The type of knob is determined by the plugin that exposes it.\n";
    std::cout << "      Always query knobs first to understand their types and constraints.\n\n";
}

void runBatchnormWithKnobs(hipdnnHandle_t handle,
                           const std::vector<KnobSetting>& knobSettings,
                           const std::string& description)
{
    printSeparator(description);

    std::cout << "Creating a batchnorm inference graph...\n\n";

    // Setup graph
    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    // Create tensors
    int64_t n = 16, c = 16, h = 16, w = 16;
    auto x = std::make_shared<graph::Tensor_attributes>();
    x->set_dim({n, c, h, w})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c * h * w, h * w, w, 1});

    auto scale = std::make_shared<graph::Tensor_attributes>();
    scale->set_dim({1, c, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c, 1, 1, 1});

    auto bias = std::make_shared<graph::Tensor_attributes>();
    bias->set_dim({1, c, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c, 1, 1, 1});

    auto mean = std::make_shared<graph::Tensor_attributes>();
    mean->set_dim({1, c, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c, 1, 1, 1});

    auto invVariance = std::make_shared<graph::Tensor_attributes>();
    invVariance->set_dim({1, c, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c, 1, 1, 1});

    // Create batchnorm node
    auto bnAttributes = graph::BatchnormInferenceAttributes();
    bnAttributes.set_name("bn_inference_node");

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);
    y->set_output(true);

    // Build graph
    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));
    std::cout << "Graph built successfully\n\n";

    // Get ranked engines
    std::vector<int64_t> rankedEngineIds;
    HIPDNN_FE_CHECK(graph->get_ranked_engine_ids(rankedEngineIds));

    if(rankedEngineIds.empty())
    {
        std::cout << "No engines available for this graph\n\n";
        return;
    }

    int64_t engineId = rankedEngineIds[0];
    std::cout << "Using engine ID: " << engineId << "\n\n";

    // Create execution plan with specified knob settings
    std::cout << "Creating execution plan with " << knobSettings.size() << " knob setting(s)...\n";
    auto error = graph->create_execution_plan_ext(engineId, knobSettings);

    if(error.is_good())
    {
        std::cout << "Success! Execution plan created\n";

        // Allocate and initialize tensors
        utilities::Tensor<float> xTensor(x->get_dim(), utilities::TensorLayout::NCHW);
        utilities::Tensor<float> scaleTensor(scale->get_dim());
        utilities::Tensor<float> biasTensor(bias->get_dim());
        utilities::Tensor<float> meanTensor(mean->get_dim());
        utilities::Tensor<float> invVarianceTensor(invVariance->get_dim());
        utilities::Tensor<float> yTensor(y->get_dim(), utilities::TensorLayout::NCHW);

        xTensor.fillWithRandomValues(0.0f, 1.0f);
        scaleTensor.fillWithRandomValues(0.0f, 1.0f);
        biasTensor.fillWithRandomValues(0.0f, 1.0f);
        meanTensor.fillWithRandomValues(0.0f, 1.0f);
        invVarianceTensor.fillWithRandomValues(0.1f, 1.0f);

        // Setup variant pack
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[x->get_uid()] = xTensor.memory().deviceData();
        variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
        variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
        variantPack[mean->get_uid()] = meanTensor.memory().deviceData();
        variantPack[invVariance->get_uid()] = invVarianceTensor.memory().deviceData();
        variantPack[y->get_uid()] = yTensor.memory().deviceData();

        // Execute
        std::cout << "Executing graph...\n";
        HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));

        yTensor.memory().markDeviceModified();
        auto yHostPtr = yTensor.memory().hostData();

        std::cout << "Execution successful!\n";
        std::cout << "First 5 output values: ";
        for(int i = 0; i < 5; ++i)
        {
            std::cout << yHostPtr[i] << " ";
        }
        std::cout << "\n\n";
    }
    else
    {
        std::cerr << "Error creating execution plan: " << error.get_message() << "\n\n";
    }
}

int main(int argc, char* argv[])
{
    // Parse command line arguments
    bool skipExecution = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if(arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "Options:\n"
                      << "  --skip-execution    Skip actual graph execution examples\n"
                      << "  --help, -h          Show this help message\n"
                      << std::endl;
            return 0;
        }
        else if(arg == "--skip-execution")
        {
            skipExecution = true;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            return 1;
        }
    }

    // Initialize
    std::cout << "\n";
    std::cout << "=========================================\n";
    std::cout << "  hipDNN Knobs Usage Sample\n";
    std::cout << "=========================================\n";
    std::cout << "\n";
    std::cout << "This sample demonstrates how to:\n";
    std::cout << "  1. Query available knobs for an engine\n";
    std::cout << "  2. Understand knob metadata and constraints\n";
    std::cout << "  3. Set custom knob values\n";
    std::cout << "  4. Validate knob settings\n";
    std::cout << "  5. Use different knob value types\n";
    std::cout << "  6. Apply knobs in real graph execution\n";
    std::cout << "\n";

    initializeFrontendLogging();

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    // Create a simple batchnorm graph to demonstrate knobs
    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    int64_t n = 16, c = 16, h = 16, w = 16;
    auto x = std::make_shared<graph::Tensor_attributes>();
    x->set_dim({n, c, h, w})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c * h * w, h * w, w, 1});

    auto scale = std::make_shared<graph::Tensor_attributes>();
    scale->set_dim({1, c, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c, 1, 1, 1});

    auto bias = std::make_shared<graph::Tensor_attributes>();
    bias->set_dim({1, c, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c, 1, 1, 1});

    auto mean = std::make_shared<graph::Tensor_attributes>();
    mean->set_dim({1, c, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c, 1, 1, 1});

    auto invVariance = std::make_shared<graph::Tensor_attributes>();
    invVariance->set_dim({1, c, 1, 1})
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({c, 1, 1, 1});

    auto bnAttributes = graph::BatchnormInferenceAttributes();
    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);
    y->set_output(true);

    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));

    // Get an engine ID
    std::vector<int64_t> rankedEngineIds;
    HIPDNN_FE_CHECK(graph->get_ranked_engine_ids(rankedEngineIds));

    if(rankedEngineIds.empty())
    {
        std::cerr << "No engines available for this graph\n";
        HIPDNN_CHECK(hipdnnDestroy(handle));
        return 1;
    }

    int64_t engineId = rankedEngineIds[0];

    // Demonstrate knob functionality
    demonstrateKnobQuery(handle, engineId, *graph);
    demonstrateDefaultKnobs(handle, engineId, *graph);
    demonstrateSettingKnobs(handle, engineId, *graph);
    demonstrateKnobValidation(handle, engineId, *graph);
    demonstrateKnobTypes(handle);

    if(!skipExecution)
    {
        // Example 7: Run with default knobs
        std::vector<KnobSetting> defaultSettings;
        runBatchnormWithKnobs(handle, defaultSettings, "7. Execution with Default Knobs");

        // Example 8: Run with benchmarking enabled
        std::vector<KnobSetting> benchmarkingSettings;
        benchmarkingSettings.emplace_back("global.benchmarking", static_cast<int64_t>(1));
        runBatchnormWithKnobs(
            handle, benchmarkingSettings, "8. Execution with Benchmarking Enabled");
    }

    // Cleanup
    HIPDNN_CHECK(hipdnnDestroy(handle));

    printSeparator("Summary");
    std::cout << "This sample demonstrated:\n";
    std::cout << "  ✓ Querying available knobs for an engine\n";
    std::cout << "  ✓ Understanding knob metadata (type, constraints, defaults)\n";
    std::cout << "  ✓ Creating execution plans with default knob values\n";
    std::cout << "  ✓ Creating execution plans with custom knob values\n";
    std::cout << "  ✓ Validating knob settings against constraints\n";
    std::cout << "  ✓ Using different knob value types (int, float, string)\n";

    if(!skipExecution)
    {
        std::cout << "  ✓ Executing graphs with different knob configurations\n";
    }

    std::cout << "\n";
    std::cout << "For more information, see:\n";
    std::cout << "  - docs/Knobs.md - Comprehensive knobs documentation\n";
    std::cout << "  - docs/HowTo.md - Quick start guide\n";
    std::cout << "  - dnn-providers/miopen-provider/docs/Knobs.md - MIOpen knobs\n";
    std::cout << "\n";

    return 0;
}
