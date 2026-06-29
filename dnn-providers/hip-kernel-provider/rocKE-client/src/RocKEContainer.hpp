// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <hipdnn_plugin_sdk/EngineManager.hpp>

#include "RocKEHandle.hpp"

namespace rocke_client
{

class RocKEContainer
{
public:
    RocKEContainer();
    ~RocKEContainer();

    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    hipdnn_plugin_sdk::EngineManager<RocKEHandle, RocKESettings, RocKEContext>& getEngineManager();

private:
    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<RocKEHandle, RocKESettings, RocKEContext>>
        _engineManager;
};

} // namespace rocke_client
