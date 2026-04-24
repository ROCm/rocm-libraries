/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>
#include <cstdint>

#include <miopen/fusion/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_IGEMM_GRP_FWD_ACTIV)

namespace miopen {
namespace solver {
namespace fusion {

void PerformanceConfigConvCKIgemmGrpFwdActivFused::HeuristicInit(
    const FusionDescription& fdesc_problem)
{
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return;

    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto data_type    = conv_problem.GetInDataType();

    valid_kernels =
        loader.FillValidKernels(CKSolverType::FusedGrpActiv, conv_problem, data_type, false);

    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[0];
    }
}

bool PerformanceConfigConvCKIgemmGrpFwdActivFused::SetNextValue(
    const FusionDescription& fdesc_problem)
{
    if(valid_kernels.empty())
    {
        this->HeuristicInit(fdesc_problem);
        if(valid_kernels.empty())
            return false;
        return true;
    }
    if(static_cast<size_t>(index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
        return false;
}

bool PerformanceConfigConvCKIgemmGrpFwdActivFused::IsValidValue() const
{
    return static_cast<size_t>(index) < valid_kernels.size();
}

bool PerformanceConfigConvCKIgemmGrpFwdActivFused::IsValid(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;

    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    const auto data_type    = conv_problem.GetInDataType();
    return loader.IsArgsSupported(
        CKSolverType::FusedGrpActiv, conv_problem, kernel_id, data_type, false);
}

bool PerformanceConfigConvCKIgemmGrpFwdActivFused::operator==(
    const PerformanceConfigConvCKIgemmGrpFwdActivFused& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigConvCKIgemmGrpFwdActivFused
ConvCKIgemmGrpFwdActivFused::GetDefaultPerformanceConfig(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
    PerformanceConfigConvCKIgemmGrpFwdActivFused pp;
    pp.HeuristicInit(fdesc_problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvCKIgemmGrpFwdActivFused::IsValidPerformanceConfig(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigConvCKIgemmGrpFwdActivFused& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

size_t ConvCKIgemmGrpFwdActivFused::GetWorkspaceSize(const FusionContext&,
                                                     const FusionDescription& fdesc_problem) const
{
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    return GetWorkspaceSizeLayoutTransformConv(conv_problem);
}

PerformanceConfigConvCKIgemmGrpFwdActivFused
ConvCKIgemmGrpFwdActivFused::Search(const FusionContext& ctx,
                                    const FusionDescription& fdesc_problem,
                                    const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool ConvCKIgemmGrpFwdActivFused::IsApplicable(const FusionContext& ctx,
                                               const FusionDescription& fdesc_problem) const
{
    const auto& desc = *fdesc_problem.fusion_plan_desc;
    if(desc.op_map.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError, "desc.op_map.empty()");
    }
    if(desc.op_map.size() != 2)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map[1]->kind() != miopenFusionOpActivForward)
        return false;
    const auto& activationType =
        dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[1]).activMode;
    if(activationType != miopenActivationRELU && activationType != miopenActivationCLIPPEDRELU &&
       activationType != miopenActivationCLAMP)
        return false;
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    if(env::disabled(MIOPEN_DEBUG_CONV_CK_IGEMM_GRP_FWD_ACTIV))
        return false;
    if(!conv_problem.IsBfp16() && !conv_problem.IsFp16() && !conv_problem.IsFp32())
        return false;
    if(conv_problem.IsTensorsCasted())
        return false;
    if(conv_problem.GetConv().attribute.deterministic)
        return false;
    if(conv_problem.HasNonPackedTensors())
        return false;
    if(!conv_problem.AllTensorsDimsFitIntoInt())
        return false;
    if(conv_problem.HasMixedDataTypes())
        return false;
    if(!(conv_problem.Is2d() || conv_problem.Is3d()))
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream().GetDeviceName()))
        return false;
    if(!conv_problem.IsLayoutNHWC() && !conv_problem.IsLayoutDefault())
        return false;

    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return false;

    return loader.IsApplicable(
        CKSolverType::FusedGrpActiv, conv_problem, conv_problem.GetInDataType(), false);
}

ConvSolution ConvCKIgemmGrpFwdActivFused::GetSolution(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigConvCKIgemmGrpFwdActivFused& config) const
{
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    return loader.GetSolution(
        CKSolverType::FusedGrpActiv, ctx, conv_problem, config.kernel_id, false);
}

} // namespace fusion
} // namespace solver
} // namespace miopen
