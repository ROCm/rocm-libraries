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

template <typename Configs, typename TensorTypes>
inline auto gcbaInferParamGenSmoke(Configs configs, TensorTypes tensorTypes)
{
    return ::testing::Combine(testing::Values(miopenActivationRELU, miopenActivationCLIPPEDRELU),
                              testing::ValuesIn(configs),
                              tensorTypes,
                              testing::Values(0.5f),
                              testing::Values(1.0f),
                              testing::Values(0.5f));
}

template <typename Configs, typename TensorTypes>
inline auto gcbaInferParamGenFull(Configs configs, TensorTypes tensorTypes)
{
    return ::testing::Combine(testing::Values(miopenActivationCLAMP),
                              testing::ValuesIn(configs),
                              tensorTypes,
                              testing::Values(0.5f),
                              testing::Values(1.0f),
                              testing::Values(0.5f));
}

} // namespace

TEST_P(GPU_ConvBiasActivInfer_FP32, ConvBiasActivAsm1x1UFloat)
{
    RunTunableSolver<miopen::solver::fusion::ConvBiasActivAsm1x1U>();
}
TEST_P(GPU_ConvBiasActivInfer_FP32, ConvOclDirectFwdFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvOclDirectFwdFused>();
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

TEST_P(GPU_ConvBiasActivInferFusionCompileStep_FP32, ConvBiasActivAsm1x1UFloat_testCompile)
{
    ScopedEnvironment<std::string> find_enforce_env(MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE");
    ScopedEnvironment<int> find_enforce_tuning_iter_env(wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5);

    fusePlanDesc.Compile(get_handle());
    RunTunableSolver<miopen::solver::fusion::ConvOclDirectFwdFused>();
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

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvBiasActivInfer_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)),
                         CbaParamNameGenerator{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvBiasActivInfer_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNCHW, miopenTensorNHWC),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)),
                         CbaParamNameGenerator{});

// BFP16 tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpBiasActivInfer_BFP16,
    gcbaInferParamGenSmoke(GroupConvTestConfig<2u>::GetSmokeConfigs<Direction::Forward>(),
                           testing::Values(miopenTensorNHWC, miopenTensorNCHW)),
    CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpBiasActivInfer3D_BFP16,
    gcbaInferParamGenSmoke(GroupConvTestConfig<3u>::GetSmokeConfigs<Direction::Forward>(),
                           testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)),
    CbaParamNameGenerator{});

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpBiasActivInfer_BFP16,
    gcbaInferParamGenFull(GroupConvTestConfig<2u>::GetConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNHWC, miopenTensorNCHW)),
    CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpBiasActivInfer3D_BFP16,
    gcbaInferParamGenFull(GroupConvTestConfig<3u>::GetConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)),
    CbaParamNameGenerator{});

// FP16 tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpBiasActivInfer_FP16,
    gcbaInferParamGenSmoke(GroupConvTestConfig<2u>::GetSmokeConfigs<Direction::Forward>(),
                           testing::Values(miopenTensorNHWC, miopenTensorNCHW)),
    CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpBiasActivInfer3D_FP16,
    gcbaInferParamGenSmoke(GroupConvTestConfig<3u>::GetSmokeConfigs<Direction::Forward>(),
                           testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)),
    CbaParamNameGenerator{});

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpBiasActivInfer_FP16,
    gcbaInferParamGenFull(GroupConvTestConfig<2u>::GetConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNHWC, miopenTensorNCHW)),
    CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpBiasActivInfer3D_FP16,
    gcbaInferParamGenFull(GroupConvTestConfig<3u>::GetConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)),
    CbaParamNameGenerator{});

// FP32 tests
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpBiasActivInfer_FP32,
    gcbaInferParamGenSmoke(GroupConvTestConfig<2u>::GetSmokeConfigs<Direction::Forward>(),
                           testing::Values(miopenTensorNHWC, miopenTensorNCHW)),
    CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpBiasActivInfer3D_FP32,
    gcbaInferParamGenSmoke(GroupConvTestConfig<3u>::GetSmokeConfigs<Direction::Forward>(),
                           testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)),
    CbaParamNameGenerator{});

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpBiasActivInfer_FP32,
    gcbaInferParamGenFull(GroupConvTestConfig<2u>::GetConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNHWC, miopenTensorNCHW)),
    CbaParamNameGenerator{});
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpBiasActivInfer3D_FP32,
    gcbaInferParamGenFull(GroupConvTestConfig<3u>::GetConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)),
    CbaParamNameGenerator{});
