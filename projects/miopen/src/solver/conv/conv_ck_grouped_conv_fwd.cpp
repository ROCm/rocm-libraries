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
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_BWD_DATA_2D)

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
    if(data_type != miopenHalf && data_type != miopenBFloat16)
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

    if(!problem.IsLayoutDefault())
        return false;

    if(!problem.IsFp16() && !problem.IsBfp16())
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

    // Guard: only request a solution for a kernel_id that this build's CK instance factory
    // actually contains. A SystemDB entry can carry a depthwise kernel_id tuned against a
    // different CK kernel set (e.g. another commit); without this check the unrecognized id
    // reaches ck_impl_depthwise_fwd_get_solution, which throws
    // "No matching kernel found for kernel_id" (CK_IMPL_STATUS_INVALID_VALUE) and aborts the
    // process instead of failing gracefully. IsArgsSupported performs the same factory lookup
    // gracefully, matching IsValidPerformanceConfig() / PerformanceConfig::IsValid().
    const auto data_type = problem.GetInDataType();
    if(config.kernel_id.empty() ||
       !loader.IsArgsSupported(
           CKSolverType::DepthwiseFwd, problem, config.kernel_id, data_type, false))
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(CKSolverType::DepthwiseFwd, ctx, problem, config.kernel_id, false);
#else
    std::ignore = ctx;
    std::ignore = problem;
    std::ignore = config;
    return ConvSolution{miopenStatusInternalError};
#endif
}

// ===========================================================================
// ConvDepthwiseBwdData2D -- input-gradient (dgrad) via the forward CK depthwise
// kernel with 180-degree-rotated filter (valid for stride-1 depthwise).
// ===========================================================================

void PerformanceConfigConvDepthwiseBwdData2D::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";
#if MIOPEN_BACKEND_HIP
    const auto data_type = problem.GetInDataType();
    if(data_type != miopenHalf && data_type != miopenBFloat16)
        return;
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return;
    valid_kernels =
        loader.FillValidKernels(CKSolverType::DepthwiseBwdData, problem, data_type, false);
    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[0];
    }
#endif
}

bool PerformanceConfigConvDepthwiseBwdData2D::SetNextValue(const ProblemDescription& problem)
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

bool PerformanceConfigConvDepthwiseBwdData2D::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvDepthwiseBwdData2D::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP
    if(kernel_id.empty())
        return false;
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;
    const auto data_type = problem.GetInDataType();
    return loader.IsArgsSupported(
        CKSolverType::DepthwiseBwdData, problem, kernel_id, data_type, false);
#else
    return IsValidValue();
#endif
}

bool PerformanceConfigConvDepthwiseBwdData2D::operator==(
    const PerformanceConfigConvDepthwiseBwdData2D& other) const
{
    return kernel_id == other.kernel_id;
}

bool ConvDepthwiseBwdData2D::IsApplicable(const ExecutionContext& ctx,
                                          const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_BWD_DATA_2D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(64 != ctx.GetStream().GetWavefrontWidth())
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(!problem.IsFp16() && !problem.IsBfp16())
        return false;
    if(!problem.IsDirectionBackwardData())
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
    return loader.IsApplicable(CKSolverType::DepthwiseBwdData, problem, data_type, false);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

PerformanceConfigConvDepthwiseBwdData2D ConvDepthwiseBwdData2D::GetDefaultPerformanceConfig(
    [[maybe_unused]] const ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigConvDepthwiseBwdData2D pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvDepthwiseBwdData2D::IsValidPerformanceConfig(
    const ExecutionContext&,
    const miopen::conv::ProblemDescription& problem,
    const PerformanceConfigConvDepthwiseBwdData2D& config) const
{
    return config.IsValid((problem));
}

PerformanceConfigConvDepthwiseBwdData2D
ConvDepthwiseBwdData2D::Search(const ExecutionContext& ctx,
                               const miopen::conv::ProblemDescription& problem,
                               const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution
ConvDepthwiseBwdData2D::GetSolution(const ExecutionContext& ctx,
                                    const miopen::conv::ProblemDescription& problem,
                                    const PerformanceConfigConvDepthwiseBwdData2D& config) const
{
#if MIOPEN_BACKEND_HIP
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    // Guard (same rationale as ConvDepthwiseFwd2D::GetSolution): only request a solution for a
    // kernel_id this build's CK dgrad factory actually contains. A stale SystemDB entry (tuned
    // against a different CK kernel set) would otherwise reach
    // ck_impl_depthwise_bwd_data_get_solution, which throws "No matching kernel found for
    // kernel_id" and aborts the process instead of failing gracefully. IsArgsSupported performs the
    // same factory lookup gracefully.
    const auto data_type = problem.GetInDataType();
    if(config.kernel_id.empty() ||
       !loader.IsArgsSupported(
           CKSolverType::DepthwiseBwdData, problem, config.kernel_id, data_type, false))
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(
        CKSolverType::DepthwiseBwdData, ctx, problem, config.kernel_id, false);
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
