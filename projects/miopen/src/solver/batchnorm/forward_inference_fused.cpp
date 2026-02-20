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

#include <miopen/batchnorm/solvers.hpp>

#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/env.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/fusion/solvers.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_BN_FWDINFER_ACTIV_FUSED)

namespace miopen {

namespace solver {

namespace fusion {

bool BnFwdInferActivationFused::IsApplicable(const FusionContext& /*context*/,
                                             const FusionDescription& problem) const
{
    const auto& desc = *problem.fusion_plan_desc;
    if(desc.op_map.empty())
        MIOPEN_THROW("");
    if(env::disabled(MIOPEN_DEBUG_BN_FWDINFER_ACTIV_FUSED))
        return false;
    if(desc.op_map.size() != 2)
        return false;
    if(desc.op_map.at(0)->kind() != miopenFusionOpBatchNormInference)
        return false;
    if(desc.op_map.at(1)->kind() != miopenFusionOpActivForward)
        return false;
    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBFp16()))
        return false;
    if(!(problem.Is2D()))
        return false;
    if(!(problem.IsLayoutNCHW() || problem.IsLayoutNHWC()))
        return false;
    const auto bn_problem = problem.GetBnProblem(0, miopen::batchnorm::Direction::ForwardInference);
    if(!IsOCLInferTypeValid(bn_problem))
        return false;
    return true;
}

ConvSolution BnFwdInferActivationFused::GetSolution(const FusionContext&,
                                                    const FusionDescription& problem) const
{
    const auto bn_problem = problem.GetBnProblem(0, miopen::batchnorm::Direction::ForwardInference);

    auto result = ConvSolution{miopenStatusSuccess};
    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenBatchNormActivInfer.cpp";
    kernel.kernel_name = "MIOpenBatchNormActivInfer";
    const auto mode    = bn_problem.GetMode();
    if(mode == miopenBNSpatial)
    { // SPATIAL kernels
        kernel.kernel_name += "SpatialEst";
    }
    else
    { // PER ACTIVATION
        kernel.kernel_name += "PerActEst";
    }

    int n, c, h, w;
    const auto& input_desc = bn_problem.GetXDesc();
    std::tie(n, c, h, w)   = tien<4>(input_desc.GetLengths());

    bool is_layout_NHWC  = (input_desc.GetLayout_t() == miopenTensorNHWC);
    size_t read_len      = (mode == miopenBNSpatial) ? (is_layout_NHWC ? c : h * w) : c * h * w;
    size_t read_unit     = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    size_t max_localsize = 256;
    size_t xlocalsize    = std::min(size_t{read_len / read_unit}, max_localsize);
    size_t xgridsize     = AlignUp(size_t{read_len / read_unit}, xlocalsize);
    size_t ylocalsize    = 1;
    size_t ygridsize     = (mode == miopenBNSpatial) ? size_t{is_layout_NHWC ? h * w : c} : 1;
    size_t zlocalsize    = 1;
    size_t zgridsize     = 1;

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    const auto& activ_op =
        dynamic_cast<ActivFwdFusionOpDescriptor&>(*problem.fusion_plan_desc->op_map[1]);
    const auto build_params = KernelBuildParameters{
        {"MIO_BN_CHW", static_cast<int>(c * h * w)},
        {"MIO_BN_HW", static_cast<int>(h * w)},
        {"MIO_BN_N", static_cast<int>(n)},
        {"MIO_BN_C", static_cast<int>(c)},
        {"MIO_BN_GRP0", static_cast<int>(xlocalsize)},
        {"MIO_BN_GRP1", static_cast<int>(ylocalsize)},
        {"MIO_BN_GRP2", static_cast<int>(zlocalsize)},
        {"MIO_LAYOUT_NHWC", static_cast<int>(is_layout_NHWC)},
        {"MIOPEN_READ_UNIT", static_cast<int>(read_unit)},
        {"MIOPEN_SBN_BOUNDS", static_cast<unsigned int>(read_len / read_unit)},
        {"MIOPEN_NRN_OP_ID", static_cast<int>(activ_op.activMode)},
        {"MIOPEN_USE_BFPMIX", static_cast<int>(input_desc.GetType() == miopenBFloat16)},
        {"MIOPEN_USE_FPMIX", static_cast<int>(input_desc.GetType() == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(input_desc.GetType() == miopenFloat)}};
    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});
    if(bn_problem.GetMode() == miopenBNSpatial)
        kernel.comp_options += " -DSPATIAL_BN";
    else
        kernel.comp_options += " -DPERACT_BN";

    result.construction_params.push_back(kernel);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) run_kernel = handle_.Run(kernels.front());
            const auto& invoke_ctx    = raw_params.CastTo<miopen::fusion::FusionInvokeParams>();
            const auto& bot_ocl_buf   = invoke_ctx.in;
            const auto& top_ocl_buf   = invoke_ctx.out;
            assert(invoke_ctx.op_args.params[0] != nullptr);
            const auto& bn_invoke = dynamic_cast<miopen::fusion::BatchNormInferenceOpInvokeParam&>(
                *invoke_ctx.op_args.params[0]);
            const auto& activ_invoker = dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(
                *invoke_ctx.op_args.params[1]);
            const auto activ_alpha = activ_invoker.activAlpha;
            const auto activ_beta  = activ_invoker.activBeta;
            const auto activ_gamma = activ_invoker.activGamma;
            std::vector<OpKernelArg> kern_args;
            const auto input_type = input_desc.GetType();
            if(input_type == miopenFloat)
            {
                kern_args.push_back({static_cast<float>(activ_alpha)});
                kern_args.push_back({static_cast<float>(activ_beta)});
                kern_args.push_back({static_cast<float>(activ_gamma)});
            }
            else if(input_type == miopenHalf)
            {
                kern_args.push_back({static_cast<half_float::half>(activ_alpha)});
                kern_args.push_back({static_cast<half_float::half>(activ_beta)});
                kern_args.push_back({static_cast<half_float::half>(activ_gamma)});
            }
            else if(input_type == miopenBFloat16)
            {
                kern_args.push_back({static_cast<bfloat16>(activ_alpha)});
                kern_args.push_back({static_cast<bfloat16>(activ_beta)});
                kern_args.push_back({static_cast<bfloat16>(activ_gamma)});
            }
            else
                MIOPEN_THROW("Unsupported Precision");
            kern_args.push_back(bn_invoke.epsilon);
            kern_args.push_back(bot_ocl_buf);
            kern_args.push_back(top_ocl_buf);
            kern_args.push_back(bn_invoke.bnBias);
            kern_args.push_back(bn_invoke.bnScale);
            kern_args.push_back(bn_invoke.estimatedMean);
            kern_args.push_back(bn_invoke.estimatedVariance);
            run_kernel(kern_args);
        };
    };

    return result;
}

} // namespace fusion

} // namespace solver

} // namespace miopen
