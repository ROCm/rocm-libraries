// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/env.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

void PerformanceConfigConvDepthwiseFwd2D::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";

#if MIOPEN_BACKEND_HIP
    const auto data_type = problem.GetInDataType();
    if(data_type != miopenHalf)
        return;

    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return;

    valid_kernels = loader.FillValidKernels(CKSolverType::DepthwiseFwd, problem, data_type, false);

    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[0];
    }
#endif
}

bool PerformanceConfigConvDepthwiseFwd2D::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit({}, problem);
        return true;
    }
    if(index + 1 < valid_kernels.size())
    {
        index++;
        kernel_id = valid_kernels[index];
    }
    else
    {
        return false;
    }
    return true;
}

bool PerformanceConfigConvDepthwiseFwd2D::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvDepthwiseFwd2D::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP
    if(kernel_id.empty())
        return false;

    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;

    const auto data_type = problem.GetInDataType();
    return loader.IsArgsSupported(CKSolverType::DepthwiseFwd, problem, kernel_id, data_type, false);
#else
    return IsValidValue();
#endif
}

bool PerformanceConfigConvDepthwiseFwd2D::operator==(
    const PerformanceConfigConvDepthwiseFwd2D& other) const
{
    return kernel_id == other.kernel_id;
}

bool ConvDepthwiseFwd2D::IsApplicable(const ExecutionContext& ctx,
                                      const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // Kernel requires a wavefront size of 64
    if(64 != ctx.GetStream().GetWavefrontWidth())
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    if(!problem.IsFp16())
        return false;

    if(!problem.IsDirectionForward())
        return false;

    // Only depthwise convolution is supported
    if((problem.GetGroupCount() != problem.GetOutChannels()) ||
       (problem.GetGroupCount() != problem.GetInChannels()))
        return false;

    const std::string arch = ctx.GetStream().GetDeviceName();
    const auto& loader     = CkImplLibLoader::Get(arch);
    if(!loader.IsLoaded())
        return false;

    const auto data_type = problem.GetInDataType();
    return loader.IsApplicable(CKSolverType::DepthwiseFwd, problem, data_type, false);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

PerformanceConfigConvDepthwiseFwd2D ConvDepthwiseFwd2D::GetDefaultPerformanceConfig(
    [[maybe_unused]] const ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigConvDepthwiseFwd2D pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvDepthwiseFwd2D::IsValidPerformanceConfig(
    const ExecutionContext&,
    const miopen::conv::ProblemDescription& problem,
    const PerformanceConfigConvDepthwiseFwd2D& config) const
{
    return config.IsValid((problem));
}

PerformanceConfigConvDepthwiseFwd2D
ConvDepthwiseFwd2D::Search(const ExecutionContext& ctx,
                           const miopen::conv::ProblemDescription& problem,
                           const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution
ConvDepthwiseFwd2D::GetSolution(const ExecutionContext& ctx,
                                const miopen::conv::ProblemDescription& problem,
                                const PerformanceConfigConvDepthwiseFwd2D& config) const
{
#if MIOPEN_BACKEND_HIP
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(CKSolverType::DepthwiseFwd, ctx, problem, config.kernel_id, false);
#else
    std::ignore = ctx;
    std::ignore = problem;
    std::ignore = config;
    return ConvSolution{miopenStatusInternalError};
#endif
}
} // namespace conv
} // namespace solver
} // namespace miopen
