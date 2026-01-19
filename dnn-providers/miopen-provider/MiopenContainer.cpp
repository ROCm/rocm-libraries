// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenContainer.hpp"
#include "EngineManager.hpp"
#include "engines/MiopenEngine.hpp"
#include "engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.hpp"
#include "engines/plans/MiopenBatchnormPlanBuilder.hpp"
#include "engines/plans/MiopenConvFwdBiasActivPlanBuilder.hpp"
#include "engines/plans/MiopenConvPlanBuilder.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

namespace miopen_legacy_plugin
{

const std::vector<MiopenContainer::EngineDefinition>& MiopenContainer::getEngineDefinitions()
{
    using namespace hipdnn_data_sdk::utilities;

    static const std::vector<EngineDefinition> s_engineDefinitions = {
        // MIOPEN_ENGINE
        {MIOPEN_ENGINE_ID,
         []() -> std::unique_ptr<IEngine> {
             auto engine = std::make_unique<MiopenEngine>(MIOPEN_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<MiopenBatchnormPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<MiopenBatchnormFwdTrainingPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<MiopenConvPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<MiopenConvFwdBiasActivPlanBuilder>());
             return engine;
         }}

        // ====================================================================
        // Additional engines would be added here
        // ====================================================================
        // Example:
        // ,{MY_CUSTOM_ENGINE_ID, []() -> std::unique_ptr<IEngine> {
        //     auto engine = std::make_unique<MyCustomEngine>(MY_CUSTOM_ENGINE_ID);
        //     engine->addPlanBuilder(std::make_unique<CustomPlanBuilder>());
        //     // ... configure plan builders for this engine
        //     return engine;
        // }}
        // ,{MY_OTHER_ENGINE_ID, []() -> std::unique_ptr<IEngine> {
        //     auto engine = std::make_unique<MyOtherEngine>(MY_OTHER_ENGINE_ID);
        //     engine->addPlanBuilder(std::make_unique<OtherPlanBuilder>());
        //     // ... configure plan builders for this engine
        //     return engine;
        // }}
        // ====================================================================
    };

    return s_engineDefinitions;
}

void MiopenContainer::copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    const auto& engineDefinitions = getEngineDefinitions();
    auto totalEngines = static_cast<uint32_t>(engineDefinitions.size());

    if(numEngines != nullptr)
    {
        *numEngines = totalEngines;
    }

    // Copy up to maxEngines IDs
    uint32_t enginesCopied = 0;
    for(const auto& engineDefinition : engineDefinitions)
    {
        if(enginesCopied >= maxEngines)
        {
            break;
        }
        engineIds[enginesCopied] = engineDefinition.id;
        enginesCopied++;
    }
}

MiopenContainer::MiopenContainer()
{
    HIPDNN_LOG_INFO("Creating MiopenContainer");

    _engineManager = std::make_unique<EngineManager>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine());
    }
}

MiopenContainer::~MiopenContainer()
{
    HIPDNN_LOG_INFO("Destroying MiopenContainer");
}

EngineManager& MiopenContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace miopen_legacy_plugin
