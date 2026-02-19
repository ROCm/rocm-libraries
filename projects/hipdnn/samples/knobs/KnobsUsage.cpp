// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

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

void printHeader(const std::string& title)
{
    std::cout << "\n--- " << title << " ---\n";
}

void demonstrateKnobQuery(hipdnnHandle_t handle, int64_t engineId, graph::Graph& graph)
{
    printHeader("1. Query Knobs");

    std::vector<Knob> knobs;
    auto error = graph.get_knobs_for_engine(engineId, knobs);

    if(!error.is_good())
    {
        std::cerr << "Error: " << error.get_message() << "\n";
        return;
    }

    std::cout << "Engine " << engineId << " has " << knobs.size() << " knob(s):\n";

    for(const auto& knob : knobs)
    {
        std::cout << "  " << knob.knobId() << " (" << knobValueTypeToString(knob.valueType())
                  << ", default=" << knobValueToString(knob.defaultValue()) << ")";
        if(const auto* constraint = knob.constraint())
        {
            std::cout << " [" << constraint->toString() << "]";
        }
        std::cout << "\n";
    }

    printHeader("2. Knob Lookup Map");

    std::unordered_map<std::string, Knob> knobMap;
    error = graph.get_knob_lookup_for_engine(engineId, knobMap);

    if(error.is_good())
    {
        auto it = knobMap.find("global.benchmarking");
        if(it != knobMap.end())
        {
            std::cout << "Found 'global.benchmarking': " << it->second.description() << "\n";
        }
        else
        {
            std::cout << "'global.benchmarking' not found\n";
        }
    }
}

void demonstrateDefaultKnobs(hipdnnHandle_t handle, int64_t engineId, graph::Graph& graph)
{
    printHeader("3. Default Knob Values");

    std::vector<KnobSetting> settings;
    auto error = graph.create_execution_plan_ext(engineId, settings);

    std::cout << "Execution plan with defaults: " << (error.is_good() ? "OK" : error.get_message())
              << "\n";
}

void demonstrateSettingKnobs(hipdnnHandle_t handle, int64_t engineId, graph::Graph& graph)
{
    printHeader("4. Custom Knob Values");

    std::vector<KnobSetting> settings;
    settings.emplace_back("global.benchmarking", static_cast<int64_t>(1));
    settings.emplace_back("global.workspace_size_limit", static_cast<int64_t>(64 * 1024 * 1024));

    std::cout << "Settings: global.benchmarking=1, global.workspace_size_limit=64MB\n";

    auto error = graph.create_execution_plan_ext(engineId, settings);
    std::cout << "Execution plan: " << (error.is_good() ? "OK" : error.get_message()) << "\n";
}

void demonstrateKnobValidation(hipdnnHandle_t handle, int64_t engineId, graph::Graph& graph)
{
    printHeader("5. Knob Validation");

    std::vector<Knob> knobs;
    auto error = graph.get_knobs_for_engine(engineId, knobs);

    if(!error.is_good())
    {
        std::cerr << "Error: " << error.get_message() << "\n";
        return;
    }

    for(const auto& knob : knobs)
    {
        if(knob.knobId() == "global.benchmarking")
        {
            KnobSetting validSetting("global.benchmarking", static_cast<int64_t>(1));
            auto validationError = knob.validate(validSetting);
            std::cout << "global.benchmarking=1: "
                      << (validationError.is_good() ? "VALID" : "INVALID") << "\n";

            KnobSetting invalidSetting("global.benchmarking", static_cast<int64_t>(5));
            validationError = knob.validate(invalidSetting);
            std::cout << "global.benchmarking=5: "
                      << (validationError.is_good()
                              ? "VALID"
                              : "INVALID (" + validationError.get_message() + ")")
                      << "\n";
            break;
        }
    }
}

void demonstrateKnobTypes(hipdnnHandle_t handle)
{
    printHeader("6. Knob Value Types");

    KnobSetting intKnob("example.int_knob", static_cast<int64_t>(42));
    KnobSetting floatKnob("example.float_knob", 3.14159);
    KnobSetting stringKnob("example.string_knob", std::string("algorithm_choice"));

    std::cout << "int64:  " << knobValueToString(intKnob.value()) << "\n";
    std::cout << "double: " << knobValueToString(floatKnob.value()) << "\n";
    std::cout << "string: " << knobValueToString(stringKnob.value()) << "\n";
}

void runBatchnormWithKnobs(hipdnnHandle_t handle,
                           const std::vector<KnobSetting>& knobSettings,
                           const std::string& description)
{
    printHeader(description);

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    int64_t n = 16; // BATCH SIZE
    int64_t c = 16; // CHANNELS (FEATURES)
    int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
    int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

    auto inputType = hipdnn_frontend::DataType::FLOAT;
    auto intermediateType = hipdnn_frontend::DataType::FLOAT;
    auto layout = utilities::TensorLayout::NCHW;
    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto bias = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto mean = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto invVariance = createTensor({1, c, 1, 1}, intermediateType, layout);

    auto bnAttributes = graph::BatchnormInferenceAttributes();
    bnAttributes.set_name("bn_inference_node");

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);
    y->set_output(true);

    HIPDNN_FE_CHECK(graph->validate());

    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));

    std::vector<int64_t> rankedEngineIds;
    HIPDNN_FE_CHECK(graph->get_ranked_engine_ids(rankedEngineIds));

    if(rankedEngineIds.empty())
    {
        std::cout << "No engines available\n";
        return;
    }

    int64_t engineId = rankedEngineIds[0];

    HIPDNN_FE_CHECK(graph->create_execution_plan_ext(engineId, knobSettings));
    HIPDNN_FE_CHECK(graph->check_support());
    HIPDNN_FE_CHECK(graph->build_plans());

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

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[x->get_uid()] = xTensor.memory().deviceData();
    variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
    variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
    variantPack[mean->get_uid()] = meanTensor.memory().deviceData();
    variantPack[invVariance->get_uid()] = invVarianceTensor.memory().deviceData();
    variantPack[y->get_uid()] = yTensor.memory().deviceData();

    HIPDNN_FE_CHECK(graph->execute(handle, variantPack, nullptr));

    yTensor.memory().markDeviceModified();
    auto yHostPtr = yTensor.memory().hostData();

    std::cout << "Execution OK, output[0..4]: ";
    for(int i = 0; i < 5; ++i)
    {
        std::cout << yHostPtr[i] << " ";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[])
{
    bool skipExecution = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if(arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "  --skip-execution  Skip graph execution demos\n"
                      << "  --help, -h        Show this help\n";
            return 0;
        }
        else if(arg == "--skip-execution")
        {
            skipExecution = true;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << " (use --help)\n";
            return 1;
        }
    }

    std::cout << "hipDNN Knobs Usage Sample\n";

    initializeFrontendLogging();

    hipdnnHandle_t handle;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    int64_t n = 16; // BATCH SIZE
    int64_t c = 16; // CHANNELS (FEATURES)
    int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
    int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

    auto inputType = hipdnn_frontend::DataType::FLOAT;
    auto intermediateType = hipdnn_frontend::DataType::FLOAT;
    auto layout = utilities::TensorLayout::NCHW;
    auto x = createTensor({n, c, h, w}, inputType, layout);
    auto scale = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto bias = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto mean = createTensor({1, c, 1, 1}, intermediateType, layout);
    auto invVariance = createTensor({1, c, 1, 1}, intermediateType, layout);

    auto bnAttributes = graph::BatchnormInferenceAttributes();
    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttributes);
    y->set_output(true);

    HIPDNN_FE_CHECK(graph->validate());

    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));

    std::vector<int64_t> rankedEngineIds;
    HIPDNN_FE_CHECK(graph->get_ranked_engine_ids(rankedEngineIds));

    if(rankedEngineIds.empty())
    {
        std::cerr << "No engines available\n";
        HIPDNN_CHECK(hipdnnDestroy(handle));
        return 1;
    }

    int64_t engineId = rankedEngineIds[0];

    demonstrateKnobQuery(handle, engineId, *graph);
    demonstrateDefaultKnobs(handle, engineId, *graph);
    demonstrateSettingKnobs(handle, engineId, *graph);
    demonstrateKnobValidation(handle, engineId, *graph);
    demonstrateKnobTypes(handle);

    if(!skipExecution)
    {
        std::vector<KnobSetting> defaultSettings;
        runBatchnormWithKnobs(handle, defaultSettings, "7. Execute (default knobs)");

        std::vector<KnobSetting> benchmarkingSettings;
        benchmarkingSettings.emplace_back("global.benchmarking", static_cast<int64_t>(1));
        runBatchnormWithKnobs(handle, benchmarkingSettings, "8. Execute (benchmarking=1)");
    }

    HIPDNN_CHECK(hipdnnDestroy(handle));

    std::cout << "\nDone. See docs/Knobs.md for details.\n";

    return 0;
}
