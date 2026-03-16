/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/library/tensor_operation_instance/gpu/convolution_forward.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#endif
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_DEFAULT_KERNELS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType>
using DeviceOp =
    ck::tensor_operation::device::DeviceConvFwd<2,
                                                ck::tensor_layout::convolution::NHWC,
                                                ck::tensor_layout::convolution::KYXC,
                                                ck::tensor_layout::convolution::NHWK,
                                                DataType,
                                                DataType,
                                                DataType,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

template <typename DataType>
using DeviceOpPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOp<DataType>>;

namespace {
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        N        = ProblemInterpreter::GetBatchN(problem);
        K        = ProblemInterpreter::GetOutputChannelK(problem);
        C        = ProblemInterpreter::GetInputChannelC(problem);
        input    = {ProblemInterpreter::GetInputHeightHi(problem),
                    ProblemInterpreter::GetInputWidthWi(problem)};
        output   = {ProblemInterpreter::GetOutputHeightHo(problem),
                    ProblemInterpreter::GetOutputWidthWo(problem)};
        filter   = {ProblemInterpreter::GetFilterHeightY(problem),
                    ProblemInterpreter::GetFilterWidthX(problem)};
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
    CKArgs& operator=(const CKArgs&) = default;
    ~CKArgs()                        = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    ConstData_t w,
                    Data_t out,
                    float alpha,
                    float beta) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             out,
                                             N,
                                             K,
                                             C,
                                             input,
                                             filter,
                                             output,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {});
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const ConvDataTensors& tensors,
                    float alpha,
                    float beta) const
    {
        return MakeArgPtr(conv_ptr, tensors.in, tensors.w, tensors.out, alpha, beta);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int N;
    int K;
    int C;
    std::vector<int> input;
    std::vector<int> output;
    std::vector<int> filter;
    std::vector<int> strides;
    std::vector<int> dilation;
    std::vector<int> lPadding;
    std::vector<int> rPadding;
};
} // namespace

template <typename DataType>
void PerformanceConfigHipImplicitGemmFwdXdlops::Init(const ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpPtrs<DataType>, CKArgs>(problem);
    index         = 0;
    kernel_id     = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmFwdXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpPtrs<DataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType>
bool ConvHipImplicitGemmFwdXdlops::CheckCKApplicability(const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpPtrs<DataType>, CKArgs>(problem);
}
#endif

// clang-format off
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables, cert-err58-cpp)
static const std::vector<std::string> ranked_gemm_fwd = {
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 64, 128, 4, Filter1x1Pad0, 4, 32, 32, 1, 2, 4, 4, 4, 4>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 64, 128, 4, Default, 4, 32, 32, 1, 2, 4, 4, 4, 4>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<64, 32, 64, 4, Filter1x1Stride1Pad0, 4, 32, 32, 1, 2, 4, 4, 4, 4>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 128, 64, 4, Filter1x1Pad0, 4, 32, 32, 2, 1, 4, 4, 4, 4>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<64, 64, 32, 4, Filter1x1Stride1Pad0, 4, 32, 32, 2, 1, 4, 4, 4, 4>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 128, 64, 4, Default, 4, 32, 32, 2, 1, 4, 4, 4, 4>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<64, 64, 32, 4, Default, 4, 32, 32, 2, 1, 4, 4, 4, 4>",
"DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 128, 256, 4, Default, 8, 2, 4, 8, 8, 8, 8, 1, 1, 8>",
"DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 64, 128, 4, Filter1x1Stride1Pad0, 8, 1, 2, 8, 8, 8, 8, 1, 1, 8>",
"DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<128, 64, 128, 4, Filter1x1Pad0, 8, 2, 2, 8, 8, 8, 8, 1, 1, 8>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 64, 128, 4, Filter1x1Stride1Pad0, 8, 32, 32, 1, 2, 8, 8, 8, 8>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 64, 128, 4, Filter1x1Pad0, 8, 32, 32, 1, 2, 8, 8, 8, 8>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<128, 64, 128, 4, Default, 8, 32, 32, 2, 2, 8, 8, 8, 8>",
"DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 64, 128, 4, OddC, 8, 1, 2, 1, 1, 1, 1, 1, 1, 8>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 128, 64, 4, Filter1x1Pad0, 8, 32, 32, 2, 1, 8, 8, 8, 8>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<128, 128, 32, 4, Filter1x1Stride1Pad0, 8, 32, 32, 2, 1, 8, 8, 8, 8>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 128, 64, 4, Default, 8, 32, 32, 2, 1, 8, 8, 8, 8>",
"DeviceConv2dFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<128, 128, 32, 4, Default, 8, 32, 32, 2, 1, 8, 8, 8, 8>",
"DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<256, 128, 64, 4, OddC, 8, 2, 1, 1, 1, 1, 1, 1, 1, 8>",
"DeviceConv2dFwdXdl_C_Shuffle_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K<64, 64, 32, 4, OddC, 8, 2, 1, 1, 1, 1, 1, 1, 1, 8>"
};

//no results for navi
static const std::vector<std::string> ranked_gemm_fwd_navi = {};
// clang-format on

void PerformanceConfigHipImplicitGemmFwdXdlops::DefaultKernelFromList(const ExecutionContext& ctx)
{
    const auto dev_name = ctx.GetStream().GetDeviceName();
    const bool is_gfx11 = StartsWith(dev_name, "gfx11");
    const bool is_gfx12 = StartsWith(dev_name, "gfx12");

    auto* ranked_p = &ranked_gemm_fwd;
    if(is_gfx11 || is_gfx12)
        ranked_p = &ranked_gemm_fwd_navi;

    const auto ranked_1st_applicable = *ranked_p;

    for(const auto& kernel_str : ranked_1st_applicable)
    {
        auto it = std::find(valid_kernels.begin(), valid_kernels.end(), kernel_str);
        if(it != valid_kernels.end())
        {
            index     = it - valid_kernels.begin();
            kernel_id = valid_kernels[index];
            return;
        }
    }
}

void PerformanceConfigHipImplicitGemmFwdXdlops::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenInt8: Init<int8_t>(problem); break;
    case miopenHalf: Init<ck::half_t>(problem); break;
    case miopenFloat: Init<float>(problem); break;
    case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt64:
    case miopenInt32:
    case miopenDouble: break;
    }

    if(!env::disabled(MIOPEN_DEBUG_CK_DEFAULT_KERNELS))
        DefaultKernelFromList(ctx);
#endif
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::SetNextValue(const ProblemDescription& problem)
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(valid_kernels.empty())
    {
        switch(problem.GetInDataType())
        {
        case miopenInt8: Init<int8_t>(problem); break;
        case miopenHalf: Init<ck::half_t>(problem); break;
        case miopenFloat: Init<float>(problem); break;
        case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        case miopenInt64:
        case miopenInt32:
        case miopenDouble: break;
        }

        assert(!valid_kernels.empty());
        return true;
    }

    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
#endif
        return false;
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenInt8: return CheckIsSupportCKArgs<int8_t>(problem);
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(problem);
    case miopenFloat: return CheckIsSupportCKArgs<float>(problem);
    case miopenBFloat16: return CheckIsSupportCKArgs<ck::bhalf_t>(problem);
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt64:
    case miopenInt32:
    case miopenDouble: break;
    }
#endif
    return false;
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmFwdXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmFwdXdlops
ConvHipImplicitGemmFwdXdlops::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                                          const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmFwdXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemmFwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmFwdXdlops& config) const
{
    return config.IsValid(problem);
}

PerformanceConfigHipImplicitGemmFwdXdlops
ConvHipImplicitGemmFwdXdlops::Search(const ExecutionContext& ctx,
                                     const ProblemDescription& problem,
                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmFwdXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.Is2d())
        return false;
    if(!IsXdlopsSupport(ctx))
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream()))
        return false;
    const std::string& arch = ctx.GetStream().GetDeviceName();
    if(arch == "gfx90a" && problem.IsGfx90aFp16altRequired())
        return false;
    if(!IsIndexRangeLargeEnough(problem))
        return false;
    if(!problem.IsLayoutNHWC())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(problem.GetGroupCount() > 1)
        return false;
    switch(problem.GetInDataType())
    {
    case miopenInt8: return CheckCKApplicability<int8_t>(problem);
    case miopenHalf: return CheckCKApplicability<ck::half_t>(problem);
    case miopenFloat: return CheckCKApplicability<float>(problem);
    case miopenBFloat16: return CheckCKApplicability<ck::bhalf_t>(problem);
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt64:
    case miopenInt32:
    case miopenDouble: break;
    }
#endif
    return false;
}

ConvSolution ConvHipImplicitGemmFwdXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmFwdXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenInt8:
        return InitInvokerFactoryNHWC<true,
                                      DeviceOpPtrs<int8_t>,
                                      CKArgs,
                                      miopen::conv::DataInvokeParams>(
            ctx, problem, config.kernel_id);
    case miopenHalf:
        return InitInvokerFactoryNHWC<true,
                                      DeviceOpPtrs<ck::half_t>,
                                      CKArgs,
                                      miopen::conv::DataInvokeParams>(
            ctx, problem, config.kernel_id);
    case miopenFloat:
        return InitInvokerFactoryNHWC<true,
                                      DeviceOpPtrs<float>,
                                      CKArgs,
                                      miopen::conv::DataInvokeParams>(
            ctx, problem, config.kernel_id);
    case miopenBFloat16:
        return InitInvokerFactoryNHWC<true,
                                      DeviceOpPtrs<ck::bhalf_t>,
                                      CKArgs,
                                      miopen::conv::DataInvokeParams>(
            ctx, problem, config.kernel_id);
    case miopenInt64:
    case miopenInt32:
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "ConvHipImplicitGemmFwdXdlops operation not implemented for this data type");
    }
#endif
    return {};
}

} // namespace conv
} // namespace solver
} // namespace miopen
