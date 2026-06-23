// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "CkDslHandle.hpp"

namespace ck_dsl_plugin {

class CkDslContainer {
   public:
    CkDslContainer();
    ~CkDslContainer();

    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    hipdnn_plugin_sdk::EngineManager<CkDslHandle, CkDslSettings, CkDslContext>& getEngineManager();

   private:
    struct EngineDefinition {
        int64_t id;
        std::function<
            std::unique_ptr<hipdnn_plugin_sdk::IEngine<CkDslHandle, CkDslSettings, CkDslContext>>()>
            createEngine;
    };
    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<CkDslHandle, CkDslSettings, CkDslContext>>
        engine_manager_;
};

}  // namespace ck_dsl_plugin
