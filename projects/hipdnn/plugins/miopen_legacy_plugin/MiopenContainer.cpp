/*
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
*/

#include <hipdnn_sdk/logging/Logger.hpp>

#include "EngineManager.hpp"
#include "MiopenContainer.hpp"
#include "engines/MiopenEngine.hpp"
#include "engines/plans/MiopenBatchnormPlanBuilder.hpp"
#include "engines/plans/MiopenConvPlanBuilder.hpp"

namespace miopen_legacy_plugin
{

MiopenContainer::MiopenContainer()
{
    HIPDNN_LOG_INFO("Creating MiopenContainer");

    int64_t engineId = 1;
    auto miopenEngine = std::make_unique<MiopenEngine>(engineId++);

    auto batchnormPlanBuilder = std::make_unique<MiopenBatchnormPlanBuilder>();
    miopenEngine->addPlanBuilder(std::move(batchnormPlanBuilder));

    auto convPlanBuilder = std::make_unique<MiopenConvPlanBuilder>();
    miopenEngine->addPlanBuilder(std::move(convPlanBuilder));

    _engineManager = std::make_unique<EngineManager>();
    _engineManager->addEngine(std::move(miopenEngine));
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
