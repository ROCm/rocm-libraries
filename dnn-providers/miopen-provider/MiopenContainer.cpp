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
// its `predict_engine` UHDs by UUID here, in two tables:
// - `*_L1_MODELS`, per ranking metric, bound under `default`: a model an operator deploys
//   through HIPDNN_DESCRIPTOR_DIR, answering on any architecture without a model of its own.
// - `*_ARCH_L1_MODELS`, per architecture: the tflops model shipped for it under descriptors/
//   (written by `uhd_gen promote`), staged to arch_content/miopen-provider/.
// Once a model ships under an id, changing that id orphans the model. An unresolved id
// just means no estimate, so ids can ship ahead of their models.
const std::map<std::string, std::string> MIOPEN_ENGINE_L1_MODELS{
    {"tflops", "c47e1b3a-8f60-4a92-b5d4-1e08c9a27f63"},
    {"time", "30284ebe-6e15-4f8e-968d-09f92d8a9480"}};
const std::map<std::string, std::string> MIOPEN_ENGINE_ARCH_L1_MODELS{
    {"gfx90a", "df2d3f54-dc4f-495a-890b-7f3a1a952036"},
    {"gfx942", "ac3cd7bd-24b4-43ad-9e60-a467df10ff07"},
    {"gfx950", "cc6eb5cd-7514-46d0-a267-e73b32a61c6e"},
    {"gfx1030", "5c3f554a-35e8-4585-98e7-4fd729eccb83"},
    {"gfx1100", "09c70474-3f9d-44c2-9b13-87eb7373ce17"},
    {"gfx1101", "93179874-1adc-4a08-9000-ddf2daa47af2"},
    {"gfx1102", "87680169-d235-4cc4-b16f-b5c64eade127"},
    {"gfx1150", "4433424e-bedb-47fc-a9fe-c6e5f5bc1ba9"},
    {"gfx1151", "3bc76e29-8d28-4b0f-a3b6-cde608c4d888"},
    {"gfx1201", "67ee5363-adfa-4ed2-b799-161665b36af3"}};
const std::map<std::string, std::string> MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS{
    {"tflops", "2d95f8e7-16c4-4b03-a8f1-7be25390c4da"},
    {"time", "8ae8a348-8d47-4ad8-992f-2114d722f7cb"}};
const std::map<std::string, std::string> MIOPEN_ENGINE_DETERMINISTIC_ARCH_L1_MODELS{
    {"gfx90a", "4a0cb8ab-d690-4e8e-ad15-40b105dfe133"},
    {"gfx942", "3a4629df-391e-420e-b3e9-7798ae08bb4f"},
    {"gfx950", "17e05b12-cc0c-4100-8e69-8e14c2758ffe"},
    {"gfx1030", "0a8afdac-9317-41f4-83e3-69944c4923dc"},
    {"gfx1100", "79153f8d-5cf4-4f0c-b3b1-c1ed6372a552"},
    {"gfx1101", "fab5bd50-574e-4cc0-a360-882f493460b3"},
    {"gfx1102", "96751fcc-7c94-47f5-9f99-2dab4f2578d1"},
    {"gfx1150", "5209c94e-e028-4d32-9096-cf448a80ed0a"},
    {"gfx1151", "cea5aef1-0468-4e88-9a9b-e974facaabff"},
    {"gfx1201", "e45cafb6-3b9b-4511-8d69-7e8c63a54df2"}};

L1ModelIds declaredL1Models(const std::map<std::string, std::string>& byMetric,
                            const std::map<std::string, std::string>& byArch)
{
    L1ModelIds declared{{"default", byMetric}};
    for(const auto& [arch, id] : byArch)
    {
        declared[arch] = {{"tflops", id}};
    }
    return declared;
}

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
                 MIOPEN_ENGINE_ID,
                 MIOPEN_ENGINE_NAME,
                 declaredL1Models(MIOPEN_ENGINE_L1_MODELS, MIOPEN_ENGINE_ARCH_L1_MODELS));

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
             auto engine = std::make_unique<MiopenEngine>(
                 MIOPEN_ENGINE_DETERMINISTIC_ID,
                 MIOPEN_ENGINE_DETERMINISTIC_NAME,
                 declaredL1Models(MIOPEN_ENGINE_DETERMINISTIC_L1_MODELS,
                                  MIOPEN_ENGINE_DETERMINISTIC_ARCH_L1_MODELS));

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
