// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
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

namespace
{

std::string knobValueToString(const KnobValueVariant& value)
{
    std::string ret;

    if(std::holds_alternative<int64_t>(value))
    {
        ret = std::to_string(std::get<int64_t>(value));
    }
    else if(std::holds_alternative<double>(value))
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << std::get<double>(value);
        ret = oss.str();
    }
    else if(std::holds_alternative<std::string>(value))
    {
        ret = "\"" + std::get<std::string>(value) + "\"";
    }

    return ret;
}

void demonstrateKnobQuery(int64_t engineId, graph::Graph& graph)
{
    std::cout << "Querying Knobs (vector):\n";

    std::vector<Knob> knobs;
    auto error = graph.get_knobs_for_engine(engineId, knobs);

    if(!error.is_good())
    {
        std::cerr << "    Error: " << error.get_message() << "\n";
        return;
    }

    std::cout << "    Engine " << hipdnn_data_sdk::utilities::getEngineNameFromId(engineId)
              << " has " << knobs.size() << " knob(s):\n";

    for(const auto& knob : knobs)
    {
        std::cout << "    " << knob.knobId() << " (" << hipdnn_frontend::to_string(knob.valueType())
                  << ", default=" << knobValueToString(knob.defaultValue()) << ")";
        if(const auto* constraint = knob.constraint())
        {
            std::cout << " [" << constraint->toString() << "]";
        }
        std::cout << "\n";
    }

    std::cout << "Querying Knobs (map):\n";

    std::unordered_map<std::string, Knob> knobMap;
    error = graph.get_knob_lookup_for_engine(engineId, knobMap);

    if(error.is_good())
    {
        auto it = knobMap.find("global.benchmarking");
        if(it != knobMap.end())
        {
            std::cout << "    Found 'global.benchmarking': " << it->second.description() << "\n";
        }
        else
        {
            std::cout << "    'global.benchmarking' not found\n";
        }
    }
}

void demonstrateSettingKnobs(int64_t engineId, graph::Graph& graph)
{
    std::cout << "Setting KnobSettings:\n";

    std::vector<KnobSetting> settings;
    settings.emplace_back("global.benchmarking", static_cast<int64_t>(1));
    settings.emplace_back("global.workspace_size_limit", static_cast<int64_t>(64 * 1024 * 1024));

    std::cout << "    Settings: global.benchmarking=1, global.workspace_size_limit=64MB\n";

    auto error = graph.create_execution_plan_ext(engineId, settings);
    std::cout << "    Execution plan: " << (error.is_good() ? "OK" : error.get_message()) << "\n";
}

void demonstrateKnobValidation(int64_t engineId, graph::Graph& graph)
{
    std::cout << "Validating KnobSettings:\n";

    std::vector<Knob> knobs;
    auto error = graph.get_knobs_for_engine(engineId, knobs);

    if(!error.is_good())
    {
        std::cerr << "    Error: " << error.get_message() << "\n";
        return;
    }

    for(const auto& knob : knobs)
    {
        if(knob.knobId() == "global.benchmarking")
        {
            const KnobSetting validSetting("global.benchmarking", static_cast<int64_t>(1));
            auto validationError = knob.validate(validSetting);
            std::cout << "    global.benchmarking=1: "
                      << (validationError.is_good() ? "VALID" : "INVALID") << "\n";

            const KnobSetting invalidSetting("global.benchmarking", static_cast<int64_t>(5));
            validationError = knob.validate(invalidSetting);
            std::cout << "    global.benchmarking=5: "
                      << (validationError.is_good()
                              ? "VALID"
                              : "INVALID (" + validationError.get_message() + ")")
                      << "\n";
            break;
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        bool useBenchmarking = false;
        for(int i = 1; i < argc; ++i)
        {
            const std::string arg(argv[i]);
            if(arg == "--help" || arg == "-h")
            {
                std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                          << "  --benchmark  Use the benchmarking knob\n"
                          << "  --help, -h        Show this help\n";
                return 0;
            }

            if(arg == "--benchmark")
            {
                useBenchmarking = true;
            }
            else
            {
                std::cerr << "Unknown argument: " << arg << " (use --help)\n";
                return 1;
            }
        }

        std::cout << "hipDNN Knobs Usage Sample\n";

        initializeFrontendLogging();

        hipdnnHandle_t handle = nullptr;
        HIPDNN_CHECK(hipdnnCreate(&handle));

        auto graph = std::make_shared<graph::Graph>();
        graph->set_io_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

        const int64_t n = 16; // BATCH SIZE
        const int64_t c = 16; // CHANNELS (FEATURES)
        const int64_t h = 16; // HEIGHT (SPATIAL DIMENSION)
        const int64_t w = 16; // WIDTH (SPATIAL DIMENSION)

        const auto inputType = hipdnn_frontend::DataType::FLOAT;
        const auto intermediateType = hipdnn_frontend::DataType::FLOAT;
        const auto layout = utilities::TensorLayout::NCHW;
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

        const int64_t engineId = rankedEngineIds[0];

        demonstrateKnobQuery(engineId, *graph);
        demonstrateSettingKnobs(engineId, *graph);
        demonstrateKnobValidation(engineId, *graph);

        const std::vector<KnobSetting> defaultSettings;

        std::vector<KnobSetting> benchmarkingSettings;
        benchmarkingSettings.emplace_back("global.benchmarking", static_cast<int64_t>(1));

        HIPDNN_FE_CHECK(graph->create_execution_plan_ext(
            engineId, useBenchmarking ? benchmarkingSettings : defaultSettings));
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

        HIPDNN_CHECK(hipdnnDestroy(handle));

        std::cout << "\nDone. See docs/Knobs.md for details.\n";

        return 0;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
