// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenContainer.hpp"
#include "engines/MiopenEngine.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.hpp"
#include "engines/plans/MiopenBatchnormPlanBuilder.hpp"
#include "engines/plans/MiopenBinaryPointwisePlanBuilder.hpp"
#include "engines/plans/MiopenConvFwdBiasActivPlanBuilder.hpp"
#include "engines/plans/MiopenConvPlanBuilder.hpp"
#include "engines/plans/MiopenUnaryActivationPlanBuilder.hpp"

#include <map>
#include <string>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace miopen_plugin
{

// ============================================================================
// Engine Registration
// ============================================================================
// For plugins that are not yet globally registered (by adding a call to
// HIPDNN_REGISTER_ENGINE() in "hipdnn_data_sdk/utilities/EngineNames.hpp"),
// use HIPDNN_REGISTER_ENGINE to register the engine names here. This will:
// 1. Create _NAME and _ID constants for the engine
// 2. Detect hash collisions with other formally-registered engines
//
// Example for new engines:
// HIPDNN_REGISTER_ENGINE(MY_CUSTOM_ENGINE)
// HIPDNN_REGISTER_ENGINE(MY_OTHER_ENGINE)
//
// Note: MIOPEN_ENGINE is already registered in EngineNames.hpp via
// HIPDNN_REGISTER_ENGINE(MIOPEN_ENGINE), so we can use
// the MIOPEN_ENGINE_NAME and MIOPEN_ENGINE_ID constants directly from there.
// ============================================================================

// L1 engine models (RFC 0019 Open Question 7): MIOpen ships no UED, so each engine names
// its `predict_engine` UHDs by UUID here, per ranking metric, bound under `default`.
// Once a model ships under an id, changing that id orphans the model. An unresolved id
// just means no estimate, so ids can ship ahead of their models.
const std::map<std::string, std::string> MIOPEN_ENGINE_L1_MODELS{
    {"tflops", "c47e1b3a-8f60-4a92-b5d4-1e08c9a27f63"},
    {"time", "30284ebe-6e15-4f8e-968d-09f92d8a9480"}};
const std::map<std::string, std::string> MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS{
    {"tflops", "2d95f8e7-16c4-4b03-a8f1-7be25390c4da"},
    {"time", "8ae8a348-8d47-4ad8-992f-2114d722f7cb"}};

const std::vector<MiopenContainer::EngineDefinition>& MiopenContainer::getEngineDefinitions()
{
    using namespace hipdnn_data_sdk::utilities;

    static const std::vector<EngineDefinition> s_engineDefinitions = {
        // MIOPEN_ENGINE (non-deterministic, default)
        {MIOPEN_ENGINE_ID,
         []() -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<HipdnnMiopenHandle,
                                                            HipdnnMiopenSettings,
                                                            HipdnnMiopenContext>> {
             auto engine = std::make_unique<MiopenEngine>(
                 MIOPEN_ENGINE_ID, MIOPEN_ENGINE_NAME, MIOPEN_ENGINE_L1_MODELS);

             engine->addPlanBuilder(std::make_unique<MiopenBatchnormPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<MiopenBatchnormFwdTrainingPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<MiopenConvPlanBuilder>(false));
             engine->addPlanBuilder(std::make_unique<MiopenConvFwdBiasActivPlanBuilder>(false));
             engine->addPlanBuilder(std::make_unique<MiopenUnaryActivationPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<MiopenBinaryPointwisePlanBuilder>());

             return engine;
         }},

        // MIOPEN_ENGINE_DETERMINISTIC
        {MIOPEN_ENGINE_DETERMINISTIC_ID,
         []() -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<HipdnnMiopenHandle,
                                                            HipdnnMiopenSettings,
                                                            HipdnnMiopenContext>> {
             auto engine = std::make_unique<MiopenEngine>(MIOPEN_ENGINE_DETERMINISTIC_ID,
                                                          MIOPEN_ENGINE_DETERMINISTIC_NAME,
                                                          MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS);

             // Batchnorm doesn't support deterministic mode.
             engine->addPlanBuilder(std::make_unique<MiopenConvPlanBuilder>(true));
             engine->addPlanBuilder(std::make_unique<MiopenConvFwdBiasActivPlanBuilder>(true));
             engine->addPlanBuilder(std::make_unique<MiopenUnaryActivationPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<MiopenBinaryPointwisePlanBuilder>());
             return engine;
         }}

        // ====================================================================
        // Additional engines would be added here
        // ====================================================================
    };

    return s_engineDefinitions;
}

uint32_t
    MiopenContainer::copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines)
{
    const auto& engineDefinitions = getEngineDefinitions();
    auto totalEngines = static_cast<uint32_t>(engineDefinitions.size());

    if(maxEngines == 0)
    {
        // When maxEngines is 0, set numEngines to total count
        numEngines = totalEngines;
        return totalEngines;
    }

    auto enginesToCopy = std::min(maxEngines, totalEngines);
    for(uint32_t i = 0; i < enginesToCopy; ++i)
    {
        engineIds[i] = engineDefinitions[i].id;
    }

    numEngines = enginesToCopy;

    return totalEngines;
}

MiopenContainer::MiopenContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating MiopenContainer");

    _engineManager = std::make_unique<hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle,
                                                                       HipdnnMiopenSettings,
                                                                       HipdnnMiopenContext>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine());
    }
}

MiopenContainer::~MiopenContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying MiopenContainer");
}

hipdnn_plugin_sdk::EngineManager<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>&
    MiopenContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace miopen_plugin
