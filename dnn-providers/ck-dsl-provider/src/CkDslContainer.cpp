// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/CkDslAttentionEngine.hpp"
#include "engines/CkDslConvEngine.hpp"
#include "engines/CkDslGemmEngine.hpp"
#include "engines/plans/CkDslAttnPlanBuilder.hpp"
#include "engines/plans/CkDslConvPlanBuilder.hpp"
#include "engines/plans/CkDslGemmPlanBuilder.hpp"

using namespace hipdnn_data_sdk::utilities;

namespace ck_dsl_plugin {

HIPDNN_REGISTER_ENGINE(CK_DSL_GEMM_ENGINE, "CK_DSL_GEMM_ENGINE")
HIPDNN_REGISTER_ENGINE(CK_DSL_ATTENTION_ENGINE, "CK_DSL_ATTENTION_ENGINE")
HIPDNN_REGISTER_ENGINE(CK_DSL_CONV_ENGINE, "CK_DSL_CONV_ENGINE")

const std::vector<CkDslContainer::EngineDefinition>& CkDslContainer::getEngineDefinitions() {
    static const std::vector<EngineDefinition> s_defs = {
        {CK_DSL_GEMM_ENGINE_ID,
         []() -> std::unique_ptr<
                  hipdnn_plugin_sdk::IEngine<CkDslHandle, CkDslSettings, CkDslContext>> {
             auto engine = std::make_unique<CkDslGemmEngine>(CK_DSL_GEMM_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<CkDslGemmPlanBuilder>());
             return engine;
         }},
        {CK_DSL_ATTENTION_ENGINE_ID,
         []() -> std::unique_ptr<
                  hipdnn_plugin_sdk::IEngine<CkDslHandle, CkDslSettings, CkDslContext>> {
             auto engine = std::make_unique<CkDslAttentionEngine>(CK_DSL_ATTENTION_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<CkDslAttnPlanBuilder>());
             return engine;
         }},
        {CK_DSL_CONV_ENGINE_ID,
         []() -> std::unique_ptr<
                  hipdnn_plugin_sdk::IEngine<CkDslHandle, CkDslSettings, CkDslContext>> {
             auto engine = std::make_unique<CkDslConvEngine>(CK_DSL_CONV_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<CkDslConvPlanBuilder>());
             return engine;
         }}};
    return s_defs;
}

uint32_t CkDslContainer::copyEngineIds(int64_t* engineIds, uint32_t maxEngines,
                                       uint32_t& numEngines) {
    const auto& defs = getEngineDefinitions();
    auto total = static_cast<uint32_t>(defs.size());
    if (maxEngines == 0) {
        numEngines = total;
        return total;
    }
    auto count = std::min(maxEngines, total);
    for (uint32_t i = 0; i < count; ++i) engineIds[i] = defs[i].id;
    numEngines = count;
    return total;
}

CkDslContainer::CkDslContainer() {
    HIPDNN_PLUGIN_LOG_INFO("Creating CkDslContainer");
    engine_manager_ = std::make_unique<
        hipdnn_plugin_sdk::EngineManager<CkDslHandle, CkDslSettings, CkDslContext>>();
    for (const auto& def : getEngineDefinitions()) engine_manager_->addEngine(def.createEngine());
}

CkDslContainer::~CkDslContainer() {
    HIPDNN_PLUGIN_LOG_INFO("Destroying CkDslContainer");
}

hipdnn_plugin_sdk::EngineManager<CkDslHandle, CkDslSettings, CkDslContext>&
CkDslContainer::getEngineManager() {
    return *engine_manager_;
}

}  // namespace ck_dsl_plugin
