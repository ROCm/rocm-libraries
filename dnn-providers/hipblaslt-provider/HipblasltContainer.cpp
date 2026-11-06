// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "EngineManager.hpp"
#include "HipblasltContainer.hpp"
#include "engines/PlanBuilderEngine.hpp"
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
#include "engines/plans/HipblasltMatmulPlanBuilder.hpp"
#ifdef HIPBLASLTPROVIDER_ENABLE_MX_GEMM
#include "engines/plans/HipblasltMxMatmulPlanBuilder.hpp"
#endif
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
#include "engines/rocblas/RocblasMatmulPlanBuilder.hpp"
#endif

namespace hipblaslt_plugin
{

HipblasltContainer::HipblasltContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating HipblasltContainer");

    _engineManager = std::make_unique<EngineManager>();

#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    auto hipblasltEngine
        = std::make_unique<PlanBuilderEngine>(hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID);

    auto matmulPlanBuilder = std::make_unique<HipblasltMatmulPlanBuilder>();
    hipblasltEngine->addPlanBuilder(std::move(matmulPlanBuilder));

#ifdef HIPBLASLTPROVIDER_ENABLE_MX_GEMM
    auto mxMatmulPlanBuilder = std::make_unique<HipblasltMxMatmulPlanBuilder>();
    hipblasltEngine->addPlanBuilder(std::move(mxMatmulPlanBuilder));
#endif

    _engineManager->addEngine(std::move(hipblasltEngine));
#endif

#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    auto rocblasMatmulEngine
        = std::make_unique<PlanBuilderEngine>(hipdnn_data_sdk::utilities::ROCBLAS_MATMUL_ENGINE_ID);
    rocblasMatmulEngine->addPlanBuilder(std::make_unique<RocblasMatmulPlanBuilder>());
    _engineManager->addEngine(std::move(rocblasMatmulEngine));
#endif
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
