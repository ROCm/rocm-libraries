// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
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
#include "engines/conv/ConvLibraryActivations.hpp"
#include "engines/conv/ConvRepackWorkspaceCap.hpp"
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
    // - Determinism, cuDNN's model (a per-engine note a framework deselects): the base and TF32
    //   engines run the libraries' defaults (rocBLAS atomics allowed, hipBLASLt offered its full
    //   workspace) and report NONDETERMINISTIC; each has a *_DETERMINISTIC sibling that runs
    //   rocBLAS with atomics not allowed, or offers hipBLASLt no workspace (as MIOpen's
    //   deterministic GEMMs do), and claims bitwise-reproducible results
    //   (TestGpuConvGemmDeterministic reruns each plan). The RPR engines have one form.
    const auto capOverride = repackWorkspaceCapOverride();
    if(capOverride.has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(CONV_REPACK_WORKSPACE_CAP_ENV_NAME
                               << " set: explicit-GEMM convolution repack workspace cap "
                               << *capOverride << " bytes in every direction");
    }
    const size_t repackCap = capOverride.value_or(MAX_REPACK_WORKSPACE_BYTES);
    const size_t wrwRepackCap = capOverride.value_or(MAX_BACKWARD_WEIGHTS_REPACK_WORKSPACE_BYTES);
    using Notes = std::vector<int32_t>;
    [[maybe_unused]] constexpr int32_t NONDETERMINISTIC = HIPDNN_NUMERICAL_NOTE_NONDETERMINISTIC;
    [[maybe_unused]] constexpr int32_t TF32 = HIPDNN_NUMERICAL_NOTE_TENSOR_CORE;
    [[maybe_unused]] constexpr int32_t RPR = HIPDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION;
    const auto addConvEngine =
        [this, repackCap, wrwRepackCap](int64_t engineId,
                                        ConvGemmPlanBuilder::PlanFactory factory,
                                        bool tf32,
                                        ConvGemmPlanBuilder::NarrowPartials narrowPartials,
                                        Notes numericalNotes,
                                        bool reducedPrecisionReduction = false,
                                        GemmEpilogueSupport gemmEpilogue = {}) {
            auto engine
                = std::make_unique<PlanBuilderEngine>(engineId, false, std::move(numericalNotes));
            engine->addPlanBuilder(std::make_unique<ConvGemmPlanBuilder>(std::move(factory),
                                                                         repackCap,
                                                                         tf32,
                                                                         reducedPrecisionReduction,
                                                                         false,
                                                                         ColumnLayout::FOLLOW_Y,
                                                                         narrowPartials,
                                                                         wrwRepackCap,
                                                                         gemmEpilogue));
            _engineManager->addEngine(std::move(engine));
        };
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    const auto hipblasltGemm = [](bool deterministic) -> ConvGemmPlanBuilder::PlanFactory {
        return [deterministic](const HipdnnEnginePluginHandle& handle,
                               const std::vector<GemmCall>& calls) -> std::unique_ptr<IPlan> {
            return std::make_unique<HipblasltGemmPlan>(handle, calls, deterministic);
        };
    };
    // hipBLASLt on gfx11 may round the parts of a long reduction to a 16-bit output.
    constexpr auto HIPBLASLT_NARROW = ConvGemmPlanBuilder::NarrowPartials::ON_GFX11;
    // hipBLASLt's bias and the activations it computes as the chain does, in its GEMM.
    constexpr GemmEpilogueSupport HIPBLASLT_GEMM_EPILOGUE{true, false};
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_ENGINE_ID,
                  hipblasltGemm(false),
                  false,
                  HIPBLASLT_NARROW,
                  Notes{NONDETERMINISTIC},
                  false,
                  HIPBLASLT_GEMM_EPILOGUE);
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_ENGINE_DETERMINISTIC_ID,
                  hipblasltGemm(true),
                  false,
                  HIPBLASLT_NARROW,
                  Notes{},
                  false,
                  HIPBLASLT_GEMM_EPILOGUE);
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_TF32_ENGINE_ID,
                  hipblasltGemm(false),
                  true,
                  HIPBLASLT_NARROW,
                  Notes{TF32, NONDETERMINISTIC},
                  false,
                  HIPBLASLT_GEMM_EPILOGUE);
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_TF32_ENGINE_DETERMINISTIC_ID,
                  hipblasltGemm(true),
                  true,
                  HIPBLASLT_NARROW,
                  Notes{TF32},
                  false,
                  HIPBLASLT_GEMM_EPILOGUE);
    addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_RPR_ENGINE_ID,
                  hipblasltGemm(false),
                  false,
                  HIPBLASLT_NARROW,
                  Notes{RPR, NONDETERMINISTIC},
                  true);
    // Opt-in: the fused activations hipBLASLt computes with other special values. Deterministic
    // (hipBLASLt offered no workspace, as the *_DETERMINISTIC engines).
    if(convLibraryActivationsEnabled())
    {
        HIPDNN_PLUGIN_LOG_INFO(CONV_LIBRARY_ACTIVATIONS_ENV_NAME
                               << " set: offering HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE");
        addConvEngine(hipdnn_data_sdk::utilities::HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE_ID,
                      hipblasltGemm(true),
                      false,
                      HIPBLASLT_NARROW,
                      Notes{},
                      false,
                      GemmEpilogueSupport{true, true});
    }
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    const auto rocblasGemm = [](bool deterministic) -> ConvGemmPlanBuilder::PlanFactory {
        return [deterministic](const HipdnnEnginePluginHandle& /*handle*/,
                               const std::vector<GemmCall>& calls) -> std::unique_ptr<IPlan> {
            // A GEMM rocBLAS has only untuned kernels for runs 25-180x behind: declined.
            requireTunedRocblasKernels(calls);
            auto plan = std::make_unique<RocblasGemmPlan>(calls);
            if(deterministic)
            {
                plan->requireBitwiseReproducible();
            }
            return plan;
        };
    };
    constexpr auto ROCBLAS_NARROW = ConvGemmPlanBuilder::NarrowPartials::NEVER;
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_ENGINE_ID,
                  rocblasGemm(false),
                  false,
                  ROCBLAS_NARROW,
                  Notes{NONDETERMINISTIC});
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_ENGINE_DETERMINISTIC_ID,
                  rocblasGemm(true),
                  false,
                  ROCBLAS_NARROW,
                  Notes{});
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_TF32_ENGINE_ID,
                  rocblasGemm(false),
                  true,
                  ROCBLAS_NARROW,
                  Notes{TF32, NONDETERMINISTIC});
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_TF32_ENGINE_DETERMINISTIC_ID,
                  rocblasGemm(true),
                  true,
                  ROCBLAS_NARROW,
                  Notes{TF32});
    addConvEngine(hipdnn_data_sdk::utilities::ROCBLAS_CONV_RPR_ENGINE_ID,
                  rocblasGemm(false),
                  false,
                  ROCBLAS_NARROW,
                  Notes{RPR, NONDETERMINISTIC},
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
