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
#include "engines/rocblas/RocblasGemmPlan.hpp"
#include "engines/rocblas/RocblasMatmulPlanBuilder.hpp"
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_CONV
#include "engines/conv/ConvGemmPlanBuilder.hpp"
#include "engines/conv/ConvReducedPrecision.hpp"
#include "engines/conv/ConvTf32.hpp"
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
#include "engines/hipblaslt/HipblasltGemmPlan.hpp"
#endif
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

#ifdef HIPBLASLTPROVIDER_ENABLE_CONV
    // The fp32 explicit-GEMM convolution engines, and their TF32 twins behind the temporary switch.
    const bool tf32Engines = convTf32EnginesEnabled();
    if(tf32Engines)
    {
        HIPDNN_PLUGIN_LOG_INFO(CONV_TF32_ENV_NAME << " set: registering the TF32 explicit-GEMM "
                                                     "convolution engines");
    }
    // The reduced-precision-reduction engines likewise, behind their own temporary switch.
    const bool rprEngines = convReducedPrecisionEnginesEnabled();
    if(rprEngines)
    {
        HIPDNN_PLUGIN_LOG_INFO(CONV_RPR_ENV_NAME << " set: registering the reduced-precision-"
                                                    "reduction explicit-GEMM convolution engines");
    }
    const auto addConvEngine = [this](int64_t engineId,
                                      ConvGemmPlanBuilder::PlanFactory factory,
                                      bool tf32,
                                      bool reducedPrecisionReduction = false) {
        auto engine = std::make_unique<PlanBuilderEngine>(engineId);
        engine->addPlanBuilder(std::make_unique<ConvGemmPlanBuilder>(
            std::move(factory), MAX_REPACK_WORKSPACE_BYTES, tf32, reducedPrecisionReduction));
        _engineManager->addEngine(std::move(engine));
    };
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    const auto hipblasltGemm = [](const HipdnnEnginePluginHandle& handle,
                                  std::vector<GemmCall> calls) -> std::unique_ptr<IPlan> {
        return std::make_unique<HipblasltGemmPlan>(handle, std::move(calls));
    };
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_ENGINE_ID, hipblasltGemm, false);
    if(tf32Engines)
    {
        addConvEngine(
            hipdnn_data_sdk::utilities::HIPBLASLT_CONV_TF32_ENGINE_ID, hipblasltGemm, true);
    }
    if(rprEngines)
    {
        addConvEngine(
            hipdnn_data_sdk::utilities::HIPBLASLT_CONV_RPR_ENGINE_ID, hipblasltGemm, false, true);
    }
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    const auto rocblasGemm = [](const HipdnnEnginePluginHandle& /*handle*/,
                                std::vector<GemmCall> calls) -> std::unique_ptr<IPlan> {
        return std::make_unique<RocblasGemmPlan>(std::move(calls));
    };
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_ENGINE_ID, rocblasGemm, false);
    if(tf32Engines)
    {
        addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_TF32_ENGINE_ID, rocblasGemm, true);
    }
    if(rprEngines)
    {
        addConvEngine(
            hipdnn_data_sdk::utilities::ROCBLAS_CONV_RPR_ENGINE_ID, rocblasGemm, false, true);
    }
#endif
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
