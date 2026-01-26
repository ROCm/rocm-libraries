// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "EngineManager.hpp"
#include "HipblasltContainer.hpp"
#include "engines/HipblasltEngine.hpp"

namespace hipblaslt_plugin
{

HipblasltContainer::HipblasltContainer()
{
    HIPDNN_LOG_INFO("Creating HipblasltContainer");

    auto hipblasltEngine
        = std::make_unique<HipblasltEngine>(hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);

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
