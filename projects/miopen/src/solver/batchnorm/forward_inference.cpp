/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2025 Advanced Micro Devices, Inc.
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
#include <miopen/stringutils.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>

namespace miopen {

namespace solver {

namespace batchnorm {

bool BnFwdInference::IsApplicable(const ExecutionContext&,
                                  const miopen::batchnorm::ProblemDescription& bn_problem) const
{
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::ForwardInference)
        return false;
    if(!(bn_problem.IsFp32() or bn_problem.IsFp16() or bn_problem.IsBFp16()))
        return false;
    if(!bn_problem.Is2D())
        return false;
    if(!IsOCLInferTypeValid(bn_problem))
        return false;

    int activ_mode = bn_problem.GetActivationDesc().GetMode();
    if(activ_mode != miopenActivationPASTHRU && activ_mode != miopenActivationRELU &&
       activ_mode != miopenActivationCLIPPEDRELU && activ_mode != miopenActivationCLAMP)
        return false;

    return true;
}

ConvSolution BnFwdInference::GetSolution(const ExecutionContext& context,
                                         const miopen::batchnorm::ProblemDescription& problem) const
{
    const auto& handle = context.GetStream();

    bool bfpmixparm   = false;
    bool bbfpmixparam = false;
    bool bfp32parm    = true;
    if(problem.GetXDesc().GetType() == miopenHalf && problem.GetBnScale().GetType() == miopenFloat)
    {
        bfpmixparm = true;
        bfp32parm  = false;
    }
    else if(problem.GetXDesc().GetType() == miopenBFloat16 &&
            problem.GetBnScale().GetType() == miopenFloat)
    {
        bbfpmixparam = true;
        bfp32parm    = false;
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;

    auto result = ConvSolution{miopenStatusSuccess};

    {
        size_t xlocalsize, xgridsize, ylocalsize, ygridsize, zlocalsize, zgridsize;
        size_t max_localsize = 256;
        size_t vectorsize    = problem.IsLayoutNHWC()
                                   ? (c % 4 == 0 ? 4 : (c % 2 == 0 ? 2 : 1))
                                   : (in_cstride % 4 == 0 ? 4 : (in_cstride % 2 == 0 ? 2 : 1));

        if(problem.GetXDesc().GetLayoutEnum() == miopenTensorNHWC)
        {
            xlocalsize = std::min(size_t{c / vectorsize}, max_localsize);
            xgridsize  = AlignUp(size_t{c / vectorsize}, xlocalsize);

            ylocalsize = max_localsize / xlocalsize;
            ygridsize  = AlignUp(size_t{in_cstride}, ylocalsize);
        }
        else
        {
            xlocalsize = 1;
            xgridsize  = AlignUp(size_t{c}, xlocalsize);

            ylocalsize = max_localsize;
            ygridsize  = AlignUp(size_t{in_cstride / vectorsize}, ylocalsize);
        }

        // HIP runtime does not support non-uniform blocks
        // Adjust the global worker sizes accordingly
        xgridsize = AlignUp(xgridsize, xlocalsize);
        ygridsize = AlignUp(ygridsize, ylocalsize);

        zlocalsize                = 1;
        size_t active_threads_xy  = xgridsize * ygridsize;
        size_t max_active_threads = handle.GetMaxComputeUnits() * 32 *
                                    handle.GetWavefrontWidth(); // considering max 32 waves per CU
        if(active_threads_xy < max_active_threads)
        {
            zgridsize = std::min(size_t{(max_active_threads / active_threads_xy)}, size_t{n});
        }
        else
        {
            zgridsize = 1;
        }

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenBatchNormFwdInfer"; // build this up
        kernel.kernel_name = "MIOpenBatchNormFwdInfer";
        if(problem.GetMode() == miopenBNSpatial)
        { // SPATIAL kernels
            kernel.kernel_file += "Spatial.cpp";
            kernel.kernel_name += "SpatialEst";
            if(problem.isInverseVariance())
            {
                kernel.kernel_name += "InvVar";
            }
        }
        else
        { // PER ACTIVATION
            kernel.kernel_file += "PerAct.cpp";
            kernel.kernel_name += "PerActivationEst";
            if(problem.isInverseVariance())
            {
                kernel.kernel_name += "InvVar";
            }
        }

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(bfpmixparm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_BFP16", static_cast<int>(bbfpmixparam)},
            {"MIOPEN_USE_BFPMIX", static_cast<int>(bbfpmixparam)},
            {"MIO_BN_GRP0", xlocalsize},
            {"MIO_BN_GRP1", ylocalsize},
            {"MIO_BN_GRP2", zlocalsize},
            {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
            {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
            {"MIO_BN_GFX115X", (StartsWith(handle.GetDeviceName(), "gfx115") ? "1" : "0")},
            {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
            {"MIO_BN_GFX125X", (StartsWith(handle.GetDeviceName(), "gfx125") ? "1" : "0")},
            {"MIO_LAYOUT_NHWC", static_cast<int>(problem.IsLayoutNHWC())},
            {"MIO_BN_VECTORIZE", static_cast<int>(vectorsize > 1)},
            {"MIO_BN_VEC_SIZE", vectorsize},
            {"MIOPEN_NRN_OP_ID", problem.GetActivationDesc().GetMode()}};

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::InfInvokeParams>();

            int n_, c_, h_, w_;
            std::tie(n_, c_, h_, w_) = tien<4>(params.xDesc->GetLengths());

            unsigned int in_nstride_ = c_ * h_ * w_;

            float alpha_activ = problem.GetActivationDesc().GetAlpha();
            float beta_activ  = problem.GetActivationDesc().GetBeta();

            if(params.xDesc->GetLayoutEnum() == miopenTensorNHWC)
            {
                if(problem.isInverseVariance())
                {
                    kernel(params.x,
                           params.y,
                           params.estimatedMean,
                           params.estimatedVariance,
                           params.bnScale,
                           params.bnBias,
                           c_,
                           h_ * w_,
                           n_,
                           1,           // cStride
                           c_,          // hwStride
                           in_nstride_, // batchStride
                           alpha_activ,
                           beta_activ);
                }
                else
                {
                    kernel(params.x,
                           params.y,
                           params.estimatedMean,
                           params.estimatedVariance,
                           params.bnScale,
                           params.bnBias,
                           params.epsilon,
                           c_,
                           h_ * w_,
                           n_,
                           1,           // cStride
                           c_,          // hwStride
                           in_nstride_, // batchStride
                           alpha_activ,
                           beta_activ);
                }
            }
            else
            {
                if(problem.isInverseVariance())
                {
                    kernel(params.x,
                           params.y,
                           params.estimatedMean,
                           params.estimatedVariance,
                           params.bnScale,
                           params.bnBias,
                           c_,
                           h_ * w_,
                           n_,
                           h_ * w_,     // cStride
                           1,           // hwStride
                           in_nstride_, // batchStride
                           alpha_activ,
                           beta_activ);
                }
                else
                {
                    kernel(params.x,
                           params.y,
                           params.estimatedMean,
                           params.estimatedVariance,
                           params.bnScale,
                           params.bnBias,
                           params.epsilon,
                           c_,
                           h_ * w_,
                           n_,
                           h_ * w_,     // cStride
                           1,           // hwStride
                           in_nstride_, // batchStride
                           alpha_activ,
                           beta_activ);
                }
            }
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
