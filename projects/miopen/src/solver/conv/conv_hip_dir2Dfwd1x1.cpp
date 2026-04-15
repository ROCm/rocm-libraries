// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT


#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_HIP_FWD1X1)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvHipDirectFwd1x1::IsApplicable(const ExecutionContext& ctx,
                                       const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_HIP_FWD1X1))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.IsDirectionForward() || problem.IsDirectionBackwardData()))
        return false;
    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16()))
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(problem.GetWeightsHeight() != 1 || problem.GetWeightsWidth() != 1)
        return false;
    if(problem.GetGroupCount() != 1)
        return false;
    if(problem.GetDilationH() != 1 || problem.GetDilationW() != 1)
        return false;
    if(problem.GetPadH() != 0 || problem.GetPadW() != 0)
        return false;
    return true;
}

ConvSolution ConvHipDirectFwd1x1::GetSolution(const ExecutionContext& ctx,
                                              const ProblemDescription& problem) const
{
    if(problem.IsDirectionForward())
        return conv_internal::GetConv2DFWDSolution(ctx, problem);
    return conv_internal::GetConv2DBWDSolution(ctx, problem);
}

} // namespace conv
} // namespace solver
} // namespace miopen
