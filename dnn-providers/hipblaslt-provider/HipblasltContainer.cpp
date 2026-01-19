// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include "EngineManager.hpp"
#include "HipblasltContainer.hpp"
#include "engines/HipblasltEngine.hpp"

namespace hipblaslt_plugin
{

HipblasltContainer::HipblasltContainer()
{
    HIPDNN_LOG_INFO("Creating HipblasltContainer");

    int64_t engineId = 1;
    auto hipblasltEngine = std::make_unique<HipblasltEngine>(engineId++);

    _engineManager = std::make_unique<EngineManager>();
    _engineManager->addEngine(std::move(hipblasltEngine));
}

HipblasltContainer::~HipblasltContainer()
{
    HIPDNN_LOG_INFO("Destroying HipblasltContainer");
}

EngineManager& HipblasltContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace hipblaslt_plugin
