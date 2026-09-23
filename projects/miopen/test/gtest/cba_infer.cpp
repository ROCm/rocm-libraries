// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cctype>
#include <string>

#include <gtest/gtest.h>
#include <half/half.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/miopen.h>

#include "cba.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "tensor_util.hpp"

namespace {

using float16 = half_float::half;

struct CbaParamNameGenerator
{
    template <typename ParamType>
    std::string operator()(const testing::TestParamInfo<ParamType>& info) const
    {
        std::string name = testing::PrintToString(info.param);
        std::transform(name.begin(), name.end(), name.begin(), [](const char c) {
            return std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
        });
        if(name.empty())
            name = "param";
        return "case_" + std::to_string(info.index) + "_" + name;
    }
};

template <typename T, typename TestCaseType = ConvTestCaseBase>
struct CBAInferBase : ConvBiasActivInferTest<T, TestCaseType>
{
    void RunSolver(const miopen::solver::fusion::FusionSolverBase& solv)
    {
        auto& handle              = get_handle();
        const auto fusion_problem = miopen::FusionDescription{&this->fusePlanDesc};
        auto fusion_ctx           = miopen::FusionContext{handle};
        if(!solv.IsApplicable(fusion_ctx, fusion_problem))
        {
            this->test_skipped = true;
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << this->conv_config;
        }
        ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_problem));
        auto sol = solv.GetSolution(fusion_ctx, fusion_problem);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);

        const auto plan_params =
            std::make_unique<miopen::fusion::FusionInvokeParams>(this->params,
                                                                 this->input.desc,
                                                                 this->in_dev.get(),
                                                                 this->output.desc,
                                                                 this->out_dev.get(),
                                                                 false);

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, *(plan_params.get()));
        handle.Finish();
    }

    std::unique_ptr<miopen::fusion::FusionInvokeParams> createFusionInvokeParams(
        const miopen::FusionDescription& fusion_desc,
        const miopen::FusionContext& fusion_ctx,
        const miopen::solver::SolverInterfaceTunable<miopen::FusionContext,
                                                     miopen::FusionDescription>& solv,
        bool useWorkspace = false)
    {
        if(useWorkspace)
        {
            this->wspace.resize(solv.GetWorkspaceSize(fusion_ctx, fusion_desc));

            return std::make_unique<miopen::fusion::FusionInvokeParams>(this->params,
                                                                        this->input.desc,
                                                                        this->in_dev.get(),
                                                                        this->output.desc,
                                                                        this->out_dev.get(),
                                                                        false,
                                                                        this->wspace.ptr(),
                                                                        this->wspace.size());
        }
        else
        {
            return std::make_unique<miopen::fusion::FusionInvokeParams>(this->params,
                                                                        this->input.desc,
                                                                        this->in_dev.get(),
                                                                        this->output.desc,
                                                                        this->out_dev.get(),
                                                                        false);
        }
    }

    // Have to keep it a template besause of GetDefaultPerformanceConfig() call
    template <typename Solver>
    void RunTunableSolver()
    {
        auto& handle = get_handle();
        Solver solv{};
        const auto fusion_problem = miopen::FusionDescription{&this->fusePlanDesc};
        auto fusion_ctx           = miopen::FusionContext{handle};
        if(!solv.IsApplicable(fusion_ctx, fusion_problem))
        {
            this->test_skipped = true;
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << this->conv_config;
        }
        ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_problem));
        auto sol = solv.GetSolution(fusion_ctx,
                                    fusion_problem,
                                    solv.GetDefaultPerformanceConfig(fusion_ctx, fusion_problem));
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);

        auto plan_params =
            createFusionInvokeParams(fusion_problem, fusion_ctx, solv, solv.MayNeedWorkspace());

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, *(plan_params.get()));
        handle.Finish();
    }
};

using GPU_ConvBiasActivInfer_FP32                  = CBAInferBase<float>;
using GPU_ConvBiasActivInferFusionCompileStep_FP32 = CBAInferBase<float>;
using GPU_ConvBiasActivInfer_FP16                  = CBAInferBase<half_float::half>;

using GPU_ConvGrpBiasActivInfer_BFP16 = CBAInferBase<bfloat16, GroupConvTestConfig<2u>>;
using GPU_ConvGrpBiasActivInfer_FP16  = CBAInferBase<float16, GroupConvTestConfig<2u>>;
using GPU_ConvGrpBiasActivInfer_FP32  = CBAInferBase<float, GroupConvTestConfig<2u>>;

using GPU_ConvGrpBiasActivInfer3D_BFP16 = CBAInferBase<bfloat16, GroupConvTestConfig<3u>>;
using GPU_ConvGrpBiasActivInfer3D_FP16  = CBAInferBase<float16, GroupConvTestConfig<3u>>;
using GPU_ConvGrpBiasActivInfer3D_FP32  = CBAInferBase<float, GroupConvTestConfig<3u>>;

// RELU and CLIPPEDRELU run in every grouped tier; the tiers differ only in the
// config set they pass. (CLAMP is intentionally not exercised here: it was never
// run before -- its only home was the excluded Full tier -- so enabling it is
// left out of scope.)
template <typename Configs, typename TensorTypes>
inline auto gcbaInferParamGen(Configs configs, TensorTypes tensorTypes)
{
    return ::testing::Combine(testing::Values(miopenActivationRELU, miopenActivationCLIPPEDRELU),
                              testing::ValuesIn(configs),
                              tensorTypes,
                              testing::Values(0.5f),
                              testing::Values(1.0f),
                              testing::Values(0.5f));
}

// Grouped config tiers, each with its own config set. Smoke and Standard use the
// small synthetic GetSmokeConfigs() (Smoke runs a single layout for the
// pre-commit gate, Standard runs both); Full uses the resnet-like GetConfigs().
// Full is excluded from every category in the yaml (built but not run), since
// those configs were not previously running.
template <unsigned NDim>
std::vector<GroupConvTestConfig<NDim>> GroupedSmokeConfigs()
{
    return GroupConvTestConfig<NDim>::template GetSmokeConfigs<Direction::Forward>();
}

template <unsigned NDim>
std::vector<GroupConvTestConfig<NDim>> GroupedFullConfigs()
{
    return GroupConvTestConfig<NDim>::template GetConfigs<Direction::Forward>();
}

} // namespace

TEST_P(GPU_ConvBiasActivInfer_FP32, ConvBiasActivAsm1x1UFloat)
{
    RunTunableSolver<miopen::solver::fusion::ConvBiasActivAsm1x1U>();
}
TEST_P(GPU_ConvBiasActivInfer_FP32, ConvHipDirectFwdFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvHipDirectFwdFused>();
}
TEST_P(GPU_ConvBiasActivInfer_FP32, ConvBinWinogradRxSFused)
{
    RunSolver(miopen::solver::fusion::ConvBinWinogradRxSFused{});
}
TEST_P(GPU_ConvBiasActivInfer_FP32, ConvBinWinogradRxSf2x3g1Fused)
{
    RunSolver(miopen::solver::fusion::ConvBinWinogradRxSf2x3g1Fused{});
}
TEST_P(GPU_ConvBiasActivInfer_FP16, ConvWinoFuryRxSf2x3Fused)
{
    RunSolver(miopen::solver::fusion::ConvWinoFuryRxSFused<2, 3>{});
}
TEST_P(GPU_ConvBiasActivInfer_FP16, ConvWinoRageRxSf2x3Fused)
{
    RunSolver(miopen::solver::fusion::ConvWinoRageRxSFused<2, 3>{});
}

TEST_P(GPU_ConvBiasActivInfer_FP16, ConvCKIgemmFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused>();
}

TEST_P(GPU_ConvGrpBiasActivInfer_BFP16, ConvCKIgemmGrpFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused>();
}
TEST_P(GPU_ConvGrpBiasActivInfer3D_BFP16, ConvCKIgemmGrpFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused>();
}
TEST_P(GPU_ConvGrpBiasActivInfer_FP16, ConvCKIgemmGrpFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused>();
}
TEST_P(GPU_ConvGrpBiasActivInfer3D_FP16, ConvCKIgemmGrpFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused>();
}
TEST_P(GPU_ConvGrpBiasActivInfer_FP32, ConvCKIgemmGrpFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused>();
}
TEST_P(GPU_ConvGrpBiasActivInfer3D_FP32, ConvCKIgemmGrpFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused>();
}

#if MIOPEN_BACKEND_HIP

TEST_P(GPU_ConvBiasActivInferFusionCompileStep_FP32, ConvHipDirectFwdFused_testCompile)
{
    ScopedEnvironment<std::string> find_enforce_env(MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE");
    ScopedEnvironment<int> find_enforce_tuning_iter_env(wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5);

    fusePlanDesc.Compile(get_handle());
    RunTunableSolver<miopen::solver::fusion::ConvHipDirectFwdFused>();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvBiasActivInferFusionCompileStep_FP32,
    testing::Combine(testing::Values(miopenActivationRELU),
                     testing::ValuesIn(GetNetworkForFusionCompileStepTest<ConvTestCaseBase>()),
                     testing::Values(miopenTensorNCHW),
                     testing::Values(0.25f),
                     testing::Values(0.75f),
                     testing::Values(0.5f)),
    CbaParamNameGenerator{});

#endif

// Evenly-spaced subset of GetNetwork1 configs for the Smoke (pre-commit) and
// Standard (per-CI) tiers. Each instantiation is further multiplied by the
// fixture's TEST_P count and the layout count.
static std::vector<ConvTestCaseBase> CbaNetworkSubset(std::size_t count)
{
    const auto all      = GetNetwork1<ConvTestCaseBase>();
    const std::size_t n = std::min(count, all.size());
    std::vector<ConvTestCaseBase> out;
    out.reserve(n);
    for(std::size_t i = 0; i < n; ++i)
        out.push_back(all[i * all.size() / n]);
    return out;
}

// The GetNetwork1 configs NOT in CbaNetworkSubset(count). The Full tier uses
// this so it does not repeat the Standard tier's configs: comprehensive/full run
// Standard and Full together, and CbaNetworkSubset(count) + CbaNetworkBeyond(count)
// is the complete GetNetwork1 list, so no coverage is lost.
static std::vector<ConvTestCaseBase> CbaNetworkBeyond(std::size_t count)
{
    const auto all      = GetNetwork1<ConvTestCaseBase>();
    const std::size_t n = std::min(count, all.size());
    std::vector<bool> in_subset(all.size(), false);
    for(std::size_t i = 0; i < n; ++i)
        in_subset[i * all.size() / n] = true;
    std::vector<ConvTestCaseBase> out;
    for(std::size_t i = 0; i < all.size(); ++i)
        if(!in_subset[i])
            out.push_back(all[i]);
    return out;
}

// FP32: tiered (4 TEST_P fixtures x configs x 1 layout).
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvBiasActivInfer_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(CbaNetworkSubset(3)),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvBiasActivInfer_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(CbaNetworkSubset(10)),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvBiasActivInfer_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(CbaNetworkBeyond(10)),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)),
                         CbaParamNameGenerator{});

// FP16: tiered (3 TEST_P fixtures x configs x layouts; Smoke uses one layout).
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvBiasActivInfer_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(CbaNetworkSubset(3)),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvBiasActivInfer_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(CbaNetworkSubset(10)),
                                          testing::Values(miopenTensorNCHW, miopenTensorNHWC),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvBiasActivInfer_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(CbaNetworkBeyond(10)),
                                          testing::Values(miopenTensorNCHW, miopenTensorNHWC),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)),
                         CbaParamNameGenerator{});

// Grouped fixtures: tiered Smoke/Standard/Full with nested configs (Smoke subset
// of Standard subset of Full) and all three activations in each tier. 2D fixtures
// run NHWC+NCHW; 3D fixtures run NDHWC+NCDHW.
#define GCBA_2D_LAYOUTS testing::Values(miopenTensorNHWC, miopenTensorNCHW)
#define GCBA_3D_LAYOUTS testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)
// Smoke (pre-commit) runs a single layout to stay small; the other layout is
// still covered on every PR by the Standard tier (and nightly by Full).
#define GCBA_2D_SMOKE_LAYOUT testing::Values(miopenTensorNHWC)
// Standard runs the layout(s) Smoke does not, so Smoke+Standard cover both with
// no repeat (the standard category runs both Smoke and Standard).
#define GCBA_2D_STD_LAYOUT testing::Values(miopenTensorNCHW)
#define GCBA_3D_SMOKE_LAYOUT testing::Values(miopenTensorNDHWC)
#define GCBA_3D_STD_LAYOUT testing::Values(miopenTensorNCDHW)

// BFP16 tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvGrpBiasActivInfer_BFP16,
                         gcbaInferParamGen(GroupedSmokeConfigs<2u>(), GCBA_2D_SMOKE_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvGrpBiasActivInfer_BFP16,
                         gcbaInferParamGen(GroupedSmokeConfigs<2u>(), GCBA_2D_STD_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvGrpBiasActivInfer_BFP16,
                         gcbaInferParamGen(GroupedFullConfigs<2u>(), GCBA_2D_LAYOUTS),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvGrpBiasActivInfer3D_BFP16,
                         gcbaInferParamGen(GroupedSmokeConfigs<3u>(), GCBA_3D_SMOKE_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvGrpBiasActivInfer3D_BFP16,
                         gcbaInferParamGen(GroupedSmokeConfigs<3u>(), GCBA_3D_STD_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvGrpBiasActivInfer3D_BFP16,
                         gcbaInferParamGen(GroupedFullConfigs<3u>(), GCBA_3D_LAYOUTS),
                         CbaParamNameGenerator{});

// FP16 tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvGrpBiasActivInfer_FP16,
                         gcbaInferParamGen(GroupedSmokeConfigs<2u>(), GCBA_2D_SMOKE_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvGrpBiasActivInfer_FP16,
                         gcbaInferParamGen(GroupedSmokeConfigs<2u>(), GCBA_2D_STD_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvGrpBiasActivInfer_FP16,
                         gcbaInferParamGen(GroupedFullConfigs<2u>(), GCBA_2D_LAYOUTS),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvGrpBiasActivInfer3D_FP16,
                         gcbaInferParamGen(GroupedSmokeConfigs<3u>(), GCBA_3D_SMOKE_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvGrpBiasActivInfer3D_FP16,
                         gcbaInferParamGen(GroupedSmokeConfigs<3u>(), GCBA_3D_STD_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvGrpBiasActivInfer3D_FP16,
                         gcbaInferParamGen(GroupedFullConfigs<3u>(), GCBA_3D_LAYOUTS),
                         CbaParamNameGenerator{});

// FP32 tests
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvGrpBiasActivInfer_FP32,
                         gcbaInferParamGen(GroupedSmokeConfigs<2u>(), GCBA_2D_SMOKE_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvGrpBiasActivInfer_FP32,
                         gcbaInferParamGen(GroupedSmokeConfigs<2u>(), GCBA_2D_STD_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvGrpBiasActivInfer_FP32,
                         gcbaInferParamGen(GroupedFullConfigs<2u>(), GCBA_2D_LAYOUTS),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvGrpBiasActivInfer3D_FP32,
                         gcbaInferParamGen(GroupedSmokeConfigs<3u>(), GCBA_3D_SMOKE_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_ConvGrpBiasActivInfer3D_FP32,
                         gcbaInferParamGen(GroupedSmokeConfigs<3u>(), GCBA_3D_STD_LAYOUT),
                         CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvGrpBiasActivInfer3D_FP32,
                         gcbaInferParamGen(GroupedFullConfigs<3u>(), GCBA_3D_LAYOUTS),
                         CbaParamNameGenerator{});

#undef GCBA_2D_LAYOUTS
#undef GCBA_2D_SMOKE_LAYOUT
#undef GCBA_2D_STD_LAYOUT
#undef GCBA_3D_LAYOUTS
#undef GCBA_3D_SMOKE_LAYOUT
#undef GCBA_3D_STD_LAYOUT
