// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "EngineManager.hpp"
#include "HipblasltContainer.hpp"
#include "engines/HipblasltEngine.hpp"
#include "engines/plans/HipblasltMatmulPlanBuilder.hpp"
#ifdef HIPDNN_HIPBLASLT_PROVIDER_ENABLE_MX_GEMM
#include "engines/plans/HipblasltMxMatmulPlanBuilder.hpp"
#endif

namespace hipblaslt_plugin
{

HipblasltContainer::HipblasltContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating HipblasltContainer");

    auto hipblasltEngine
        = std::make_unique<HipblasltEngine>(hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);

    auto matmulPlanBuilder = std::make_unique<HipblasltMatmulPlanBuilder>();
    hipblasltEngine->addPlanBuilder(std::move(matmulPlanBuilder));

#ifdef HIPDNN_HIPBLASLT_PROVIDER_ENABLE_MX_GEMM
    auto mxMatmulPlanBuilder = std::make_unique<HipblasltMxMatmulPlanBuilder>();
    hipblasltEngine->addPlanBuilder(std::move(mxMatmulPlanBuilder));
#endif

    _engineManager = std::make_unique<EngineManager>();
    _engineManager->addEngine(std::move(hipblasltEngine));
}

HipblasltContainer::~HipblasltContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying HipblasltContainer");
}

EngineManager& HipblasltContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace hipblaslt_plugin
