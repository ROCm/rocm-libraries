// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/NumericalNote.h>
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
    // The hipBLASLt matmul plans are capture-safe; the engine reports graph capture support.
    auto hipblasltEngine = std::make_unique<PlanBuilderEngine>(
        hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID, true);

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
    // The explicit-GEMM convolution engines, with their TF32 and reduced-precision-reduction
    // siblings, all offered by default. Each reports its RFC 0025 numerical notes, by which a
    // caller excludes it (deselect_numeric_notes):
    // - TF32: TENSOR_CORE, the note cuDNN reports for its TF32 convolution engines. PyTorch's
    //   cuDNN path drops an fp32 engine with CUDNN_NUMERICAL_NOTE_TENSOR_CORE when allow_tf32 is
    //   off, and every engine with DOWN_CONVERT_INPUTS always (aten/src/ATen/native/cudnn/
    //   Conv_v8.cpp, the engine-config filter), so TENSOR_CORE is the TF32 opt-out.
    // - Reduced-precision reduction: REDUCED_PRECISION_REDUCTION (fp16 partial sums).
    // - Determinism: the rocBLAS engines run every GEMM with rocBLAS atomics not allowed
    //   (RocblasGemmPlan::requireBitwiseReproducible) and repack by gathers, so they claim
    //   bitwise-reproducible results (TestGpuConvGemm reruns each plan). hipBLASLt has no such
    //   control (its solutions may split a reduction across workgroups), so its engines report
    //   NONDETERMINISTIC.
    using Notes = std::vector<int32_t>;
    [[maybe_unused]] const Notes tf32Notes{HIPDNN_NUMERICAL_NOTE_TENSOR_CORE};
    [[maybe_unused]] const Notes rprNotes{HIPDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION};
    [[maybe_unused]] const auto nondeterministic = [](Notes notes) {
        notes.push_back(HIPDNN_NUMERICAL_NOTE_NONDETERMINISTIC);
        return notes;
    };
    const auto addConvEngine = [this](int64_t engineId,
                                      ConvGemmPlanBuilder::PlanFactory factory,
                                      bool tf32,
                                      ConvGemmPlanBuilder::NarrowPartials narrowPartials,
                                      std::optional<Notes> numericalNotes,
                                      bool reducedPrecisionReduction = false) {
        auto engine
            = std::make_unique<PlanBuilderEngine>(engineId, false, std::move(numericalNotes));
        engine->addPlanBuilder(std::make_unique<ConvGemmPlanBuilder>(std::move(factory),
                                                                     MAX_REPACK_WORKSPACE_BYTES,
                                                                     tf32,
                                                                     reducedPrecisionReduction,
                                                                     false,
                                                                     ColumnLayout::FOLLOW_Y,
                                                                     narrowPartials));
        _engineManager->addEngine(std::move(engine));
    };
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    const auto hipblasltGemm = [](const HipdnnEnginePluginHandle& handle,
                                  const std::vector<GemmCall>& calls) -> std::unique_ptr<IPlan> {
        return std::make_unique<HipblasltGemmPlan>(handle, calls);
    };
    // hipBLASLt on gfx11 may round the parts of a long reduction to a 16-bit output.
    constexpr auto HIPBLASLT_NARROW = ConvGemmPlanBuilder::NarrowPartials::ON_GFX11;
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_ENGINE_ID,
                  hipblasltGemm,
                  false,
                  HIPBLASLT_NARROW,
                  nondeterministic({}));
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_TF32_ENGINE_ID,
                  hipblasltGemm,
                  true,
                  HIPBLASLT_NARROW,
                  nondeterministic(tf32Notes));
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_RPR_ENGINE_ID,
                  hipblasltGemm,
                  false,
                  HIPBLASLT_NARROW,
                  nondeterministic(rprNotes),
                  true);
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    const auto rocblasGemm = [](const HipdnnEnginePluginHandle& /*handle*/,
                                const std::vector<GemmCall>& calls) -> std::unique_ptr<IPlan> {
        auto plan = std::make_unique<RocblasGemmPlan>(calls);
        plan->requireBitwiseReproducible();
        return plan;
    };
    constexpr auto ROCBLAS_NARROW = ConvGemmPlanBuilder::NarrowPartials::NEVER;
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_ENGINE_ID,
                  rocblasGemm,
                  false,
                  ROCBLAS_NARROW,
                  Notes{});
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_TF32_ENGINE_ID,
                  rocblasGemm,
                  true,
                  ROCBLAS_NARROW,
                  tf32Notes);
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_RPR_ENGINE_ID,
                  rocblasGemm,
                  false,
                  ROCBLAS_NARROW,
                  rprNotes,
                  true);
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
