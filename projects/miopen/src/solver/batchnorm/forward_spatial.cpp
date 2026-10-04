/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/batchnorm/common_spatial.hpp>
#include <miopen/batchnorm/solvers.hpp>

#include <miopen/generic_search.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/kernel_build_params.hpp>

namespace miopen {

namespace solver {

namespace batchnorm {

// Spatial multiple needs space for 2 fp32 elements
// per each x thread (including the last workgroup)
// to stash intermediate mean and variance
const unsigned int stash_values_fwd = 2;

namespace {
bool IsFineSpatialProblem(const miopen::batchnorm::ProblemDescription& problem)
{
    if(!problem.Is2D() || !problem.IsLayoutNCHW() || !problem.IsBFp16() ||
       !problem.IsScaleFp32() || !problem.GetXDesc().IsPacked() ||
       !problem.GetYDesc().IsPacked())
        return false;
    const auto& lengths = problem.GetXDesc().GetLengths();
    return lengths[0] >= 16 && lengths[0] <= 64 && lengths[1] <= 64 &&
           lengths[2] * lengths[3] >= 4096 && lengths[2] * lengths[3] <= 8192;
}

bool IsFineSpatialConfig(const miopen::batchnorm::ProblemDescription& problem,
                         size_t vectorsize,
                         size_t xlocalsize,
                         size_t ylocalsize,
                         size_t zlocalsize,
                         size_t nelements)
{
    if(!IsFineSpatialProblem(problem) || vectorsize != 4 || xlocalsize != 1 ||
       (ylocalsize != 64 && ylocalsize != 128) || zlocalsize != 2 || nelements <= 1)
        return false;
    const auto& lengths = problem.GetXDesc().GetLengths();
    const size_t n = lengths[0], hw = lengths[2] * lengths[3];
    return hw % vectorsize == 0 && n % nelements == 0 &&
           (hw - 1) % (ylocalsize * vectorsize) + 1 >= 2 * stash_values_fwd &&
           IsSpatialMultipleApplicable(
               problem, vectorsize, stash_values_fwd, ylocalsize, zlocalsize, nelements);
}
} // namespace

bool PerformanceConfigBnFwdTraining::IsValid(
    const ExecutionContext& context, const miopen::batchnorm::ProblemDescription& problem) const
{
    if(this->kernel_id.empty())
    {
        return false;
    }

    // Device and shape restrictions for the specialized spatial implementations.
    size_t vectorsize = 1, xlocalsize = 1, ylocalsize = 1, zlocalsize = 1, nelements = 1;
    int variant = -1;
    GetVariantFromKernelId(
        this->kernel_id, variant, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements);
    if(variant == 5 || variant == 7 || variant == 8)
        return false;
    if(variant == 6)
    {
        const auto& handle = context.GetStream();
        return handle.GetDeviceName() == "gfx1250" && handle.GetWavefrontWidth() == 32 &&
               ylocalsize == 1 && zlocalsize == 1 && nelements == 1 &&
               IsSpatialStreamingApplicable(problem, vectorsize, xlocalsize);
    }
    if(variant == 4)
    {
        const auto& handle = context.GetStream();
        return handle.GetDeviceName() == "gfx1250" && handle.GetWavefrontWidth() == 32 &&
               ylocalsize == 1 && zlocalsize == 1 && nelements == 1 &&
               IsSpatialBufferedApplicable(problem, vectorsize, xlocalsize);
    }
    if(variant == 2)
    {
        if(IsFineSpatialProblem(problem) && ylocalsize < 256)
        {
            const auto& handle = context.GetStream();
            if(handle.GetDeviceName() != "gfx1250" || handle.GetWavefrontWidth() != 32 ||
               !IsFineSpatialConfig(
                   problem, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements))
                return false;
        }
        if(problem.IsLayoutNCHW() && vectorsize > 4 && problem.IsBFp16())
        {
            const auto& handle = context.GetStream();
            if(vectorsize != 8 || !StartsWith(handle.GetDeviceName(), "gfx125") ||
               handle.GetWavefrontWidth() != 32 ||
               !problem.IsScaleFp32() || !problem.GetXDesc().IsPacked() ||
               !problem.GetYDesc().IsPacked() || nelements == 0 ||
               problem.GetXDesc().GetLengths()[0] % nelements != 0)
                return false;
        }
        return IsSpatialMultipleApplicable(
            problem, vectorsize, stash_values_fwd, ylocalsize, zlocalsize, nelements);
    }
    return true;
}

void PerformanceConfigBnFwdTraining::HeuristicInit(
    const miopen::batchnorm::ProblemDescription& problem)
{
    // Define default configuration based on heuristics and
    // add all other valid configurations for the given problem
    if(UseMultiple(problem))
    {
        DefaultConfigSpatialMultiple(problem, stash_values_fwd, this->valid_kernels);
        // if more than 2 instances are present, it means that variant 1 will be slower
        if((this->valid_kernels.size() < 2 && problem.IsLayoutNHWC()) || !problem.IsLayoutNHWC())
        {
            DefaultConfigSpatialSingle(problem, this->valid_kernels);
        }
    }
    else
    {
        DefaultConfigSpatialSingle(problem, this->valid_kernels);
        // if valid_kernels is 2, it means that variant 0 or variant 3 were added and in
        // this case it doesn't make sense to add instances for variant 2 because it is
        // very unlikely that they will be faster than those variants
        if(this->valid_kernels.size() < 2)
        {
            DefaultConfigSpatialMultiple(problem, stash_values_fwd, this->valid_kernels);
        }
    }
    AddSpatialBufferedConfigs(problem, this->valid_kernels);
    AddSpatialStreamingConfigs(problem, this->valid_kernels);
    if(IsFineSpatialProblem(problem))
    {
        const size_t n = problem.GetXDesc().GetLengths()[0];
        size_t nelements = std::min(size_t{8}, n / 2);
        while(n % nelements != 0)
            --nelements;
        for(size_t ylocalsize : {size_t{64}, size_t{128}})
            if(IsFineSpatialConfig(problem, 4, 1, ylocalsize, 2, nelements))
                valid_kernels.push_back(GetKernelIdFromVariant(
                    2, 4, 1, ylocalsize, 2, nelements));
    }

    // Set index and kernel_id to default value
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

bool PerformanceConfigBnFwdTraining::SetNextValue(
    const miopen::batchnorm::ProblemDescription& problem_desc)
{
    // In case the valid_kernel list is empty, we fill it with
    // default value as first one and all other valid ones will follow
    if(this->valid_kernels.empty())
    {
        this->HeuristicInit(problem_desc);
        return true;
    }
    // Get next valid configuration
    if((this->index + 1) < valid_kernels.size())
    {
        ++this->index;
        this->kernel_id = this->valid_kernels[index];
        return true;
    }
    else
    {
        return false;
    }
}

bool PerformanceConfigBnFwdTraining::operator==(const PerformanceConfigBnFwdTraining& other) const
{
    return this->kernel_id == other.kernel_id;
}

bool PerformanceConfigBnFwdTraining::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool BnFwdTrainingSpatial::IsApplicable(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& bn_problem) const
{
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::ForwardTraining ||
       bn_problem.GetMode() != miopenBNSpatial)
        return false;

    if(!bn_problem.Is2D())
        return false;

    if(!IsFwdTrainTypeValid(bn_problem))
        return false;

    int activ_mode = bn_problem.GetActivationDesc().GetMode();
    if(activ_mode != miopenActivationPASTHRU && activ_mode != miopenActivationRELU &&
       activ_mode != miopenActivationCLIPPEDRELU && activ_mode != miopenActivationCLAMP)
    {
        return false;
    }

    return true;
}

PerformanceConfigBnFwdTraining BnFwdTrainingSpatial::GetDefaultPerformanceConfig(
    const ExecutionContext& context, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
    PerformanceConfigBnFwdTraining pp;
    pp.HeuristicInit(problem_desc);
    const auto buffered = PerformanceConfigBnFwdTraining{
        0, GetKernelIdFromVariant(4, 8, 512, 1, 1, 1)};
    if(buffered.IsValid(context, problem_desc))
    {
        size_t n, c, h, w;
        std::tie(n, c, h, w) = tien<4>(problem_desc.GetXDesc().GetLengths());
        // Bounded packed retention avoids the register-pressure crossover seen
        // in larger NHW; enough channels are needed to occupy the device.
        if(n * h * w <= 65536 && h * w >= 512 && c >= 128)
        {
            pp.kernel_id = buffered.kernel_id;
            pp.index = std::distance(
                pp.valid_kernels.begin(),
                std::find(pp.valid_kernels.begin(), pp.valid_kernels.end(), pp.kernel_id));
        }
    }
    const auto streaming = PerformanceConfigBnFwdTraining{
        0, GetKernelIdFromVariant(6, 8, 1024, 1, 1, 1)};
    if(streaming.IsValid(context, problem_desc))
    {
        size_t n, c, h, w;
        std::tie(n, c, h, w) = tien<4>(problem_desc.GetXDesc().GetLengths());
        const size_t hw = h * w, nhw = n * hw;
        // Streaming avoids the retained-input register-pressure crossover while
        // keeping the buffered small-NHW default and low-channel heuristics intact.
        if(c >= 128 && c <= 256 && nhw > 65536 && nhw <= 262144 &&
           hw >= 4096 && hw <= 8192)
        {
            pp.kernel_id = streaming.kernel_id;
            pp.index = std::distance(
                pp.valid_kernels.begin(),
                std::find(pp.valid_kernels.begin(), pp.valid_kernels.end(), pp.kernel_id));
        }
    }
    if(IsFineSpatialProblem(problem_desc))
    {
        const size_t n = problem_desc.GetXDesc().GetLengths()[0];
        size_t nelements = std::min(size_t{8}, n / 2);
        while(n % nelements != 0)
            --nelements;
        const auto fine = PerformanceConfigBnFwdTraining{
            0, GetKernelIdFromVariant(2, 4, 1, 64, 2, nelements)};
        if(fine.IsValid(context, problem_desc))
        {
            pp.kernel_id = fine.kernel_id;
            pp.index = std::distance(
                pp.valid_kernels.begin(),
                std::find(pp.valid_kernels.begin(), pp.valid_kernels.end(), pp.kernel_id));
        }
    }
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool BnFwdTrainingSpatial::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const miopen::batchnorm::ProblemDescription& problem_desc,
    const PerformanceConfigBnFwdTraining& config) const
{
    bool valid = config.IsValid(ctx, problem_desc);
    return valid;
}

PerformanceConfigBnFwdTraining
BnFwdTrainingSpatial::Search(const ExecutionContext& ctx,
                             const miopen::batchnorm::ProblemDescription& problem,
                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution BnFwdTrainingSpatial::GetSolution(const ExecutionContext& context,
                                               const miopen::batchnorm::ProblemDescription& problem,
                                               const PerformanceConfigBnFwdTraining& config) const
{
    const auto& handle = context.GetStream();
    // Only one can be true
    bool bfpmixparm   = false;
    bool bbfpmixparam = false;
    bool bfp16parm    = false;
    bool bfp32parm    = true;

    if(problem.GetXDesc().GetType() == miopenHalf && problem.GetBnScale().GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }
    else if(problem.GetXDesc().GetType() == miopenHalf &&
            problem.GetBnScale().GetType() == miopenFloat)
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
    unsigned int in_nstride = c * in_cstride;
    unsigned int in_nhw     = n * in_cstride;
    unsigned int in_nchw    = n * in_nstride;
    auto inhw               = float(1.0 / in_nhw);

    int variant       = -1;
    size_t vectorsize = 1;
    size_t xlocalsize = 1, xgridsize = 1;
    size_t ylocalsize = 1, ygridsize = 1;
    size_t zlocalsize = 1, zgridsize = 1;
    unsigned int ldsgcn = 0, ldsnogcn = 0;
    int stash_method = 0;
    size_t nelements = 1;

    auto const waveSize = handle.GetWavefrontWidth();

    GetVariantFromKernelId(
        config.kernel_id, variant, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements);

    // GetSolution is also called directly by forced-config consumers.
    if(variant == 2 && problem.IsLayoutNCHW() && problem.IsBFp16() && vectorsize > 4 &&
       !config.IsValid(context, problem))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported BF16 NCHW vector8 batchnorm configuration");
    if(variant == 4 && !config.IsValid(context, problem))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported buffered FP16/BF16 batchnorm configuration");
    if(variant == 5 || variant == 7 || variant == 8)
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported specialized forward batchnorm configuration");
    if(variant == 6 && !config.IsValid(context, problem))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported streaming BF16 batchnorm configuration");

    size_t xlocalsize_final = xlocalsize, ylocalsize_final = ylocalsize,
           zlocalsize_final = zlocalsize;
    if(variant != 2)
    {
        if(variant != 4 && variant != 6)
        {
            xlocalsize = 1024;
            if(((in_cstride < 256) && (n < 256)) || ((in_cstride < 100) && (n <= 256)))
                xlocalsize = 256;
        }
        xgridsize = c * xlocalsize;
        ldsgcn    = xlocalsize / waveSize;
        ldsnogcn  = xlocalsize;
    }
    else
    {
        // Compute grid size
        if(problem.IsLayoutNHWC())
        {
            xgridsize = xlocalsize * ((c / vectorsize + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((c + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((in_cstride / vectorsize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize * ((n / nelements + zlocalsize - 1) / zlocalsize);

        // Get the stash method based on problem size and WG size
        stash_method = GetStashMethod(problem.IsLayoutNHWC(),
                                      problem.GetXDesc().GetType(),
                                      stash_values_fwd,
                                      c,
                                      n,
                                      in_cstride,
                                      ylocalsize,
                                      zlocalsize,
                                      nelements);

        // WG size for Final kernels (NHWC)
        if(problem.IsLayoutNHWC() && c % 2 == 0 && xlocalsize % 2 == 0)
        {
            // increase number of blocks (xgridsize does not change for final kernels)
            // 2 is the lower bound because of stashing
            xlocalsize_final = 2;
            // increase the number of threads in the y and z direction to decrease the number of
            // loads/stores for each thread
            zlocalsize_final = zgridsize / zlocalsize * zlocalsize;
            ylocalsize_final =
                (xlocalsize * ylocalsize * zlocalsize) / xlocalsize_final / zlocalsize_final;
        }
        ldsnogcn = xlocalsize * ylocalsize * zlocalsize;
        ldsgcn   = xlocalsize * ylocalsize * zlocalsize / waveSize;
    }

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_RNE_BFLOAT16", MIOPEN_USE_RNE_BFLOAT16},
            {"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
            {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
            {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
            {"MIOPEN_USE_BFPMIX", static_cast<int>(bbfpmixparam)},
            {"MIO_SAVE_MEAN_VARIANCE", static_cast<int>(problem.GetResultSave())},
            {"MIO_RUNNING_RESULT",
             context.is_for_generic_search ? static_cast<int>(0)
                                           : static_cast<int>(problem.GetResultRunning())},
            {"MIO_BN_VARIANT", variant},
            {"MIO_BN_LDS_SIZE", ldsnogcn},
            {"MIO_BN_LDSGCN_SIZE", std::to_string(ldsgcn)},
            {"MIO_BN_N", n},
            {"MIO_BN_NGRPS", ygridsize / ylocalsize},
            {"MIO_BN_NGRPS2", zgridsize / zlocalsize},
            {"MIO_BN_N_ELEMENTS", nelements},
            {"MIO_BN_GRP0", xlocalsize},
            {"MIO_BN_GRP1", ylocalsize},
            {"MIO_BN_GRP2", zlocalsize},
            {"MIO_BN_GRP0_FINAL", xlocalsize_final},
            {"MIO_BN_GRP1_FINAL", ylocalsize_final},
            {"MIO_BN_GRP2_FINAL", zlocalsize_final},
            {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
            {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
            {"MIO_BN_GFX115X", (StartsWith(handle.GetDeviceName(), "gfx115") ? "1" : "0")},
            {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
            {"MIO_BN_GFX125X", (StartsWith(handle.GetDeviceName(), "gfx125") ? "1" : "0")},
            {"MIO_LAYOUT_NHWC", static_cast<int>(problem.IsLayoutNHWC())},
            {"MIO_BN_VECTORIZE", static_cast<int>(vectorsize > 1)},
            {"MIO_BN_VEC_SIZE", vectorsize},
            {"MIO_BN_STASH_METHOD", stash_method},
            {"MIOPEN_NRN_OP_ID", problem.GetActivationDesc().GetMode()}};

        build_params.Define("MIO_BN_C", c);
        build_params.Define("MIO_BN_HW", in_cstride);
        build_params.Define("MIO_BN_NHW", in_nhw);
        build_params.Define("MIO_BN_CHW", in_nstride);
        build_params.Define("MIO_BN_NCHW", in_nchw);

        build_params.Define("HIP_ENABLE_EXTRA_WARP_SYNC_TYPES");

        kernel.kernel_file      = "MIOpenBatchNormFwdTrainSpatial.cpp";
        std::string kernel_name = "MIOpenBatchNormFwdTrainSpatial";
        kernel.comp_options     = build_params.GenerateFor(kbp::HIP{});

        kernel.kernel_name = kernel_name;

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        if(variant != 2)
        {
            result.construction_params.push_back(kernel);
        }
        else
        {
            auto single_yzgroup_kernel    = kernel;
            single_yzgroup_kernel.l_wk[0] = xlocalsize_final;
            single_yzgroup_kernel.l_wk[1] = ylocalsize_final;
            single_yzgroup_kernel.l_wk[2] = zlocalsize_final;
            single_yzgroup_kernel.g_wk[1] = ylocalsize_final;
            single_yzgroup_kernel.g_wk[2] = zlocalsize_final;

            kernel.kernel_name = kernel_name + "MeanVariance";
            result.construction_params.push_back(kernel);

            single_yzgroup_kernel.kernel_name = kernel_name + "FinalMeanVariance";
            result.construction_params.push_back(single_yzgroup_kernel);

            kernel.kernel_name = kernel_name + "Norm";
            result.construction_params.push_back(kernel);
        }
    }

    const auto dtype = problem.GetBnScale().GetType();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::FwdTrainInvokeParams>();
            const auto resultsave =
                params.resultSaveMean != nullptr && params.resultSaveInvVariance != nullptr;
            const auto resultrunning = params.prevResultRunningMean != nullptr &&
                                       params.prevResultRunningVariance != nullptr &&
                                       params.nextResultRunningMean != nullptr &&
                                       params.nextResultRunningVariance != nullptr &&
                                       !context.is_for_generic_search;

            float alpha_activ = problem.GetActivationDesc().GetAlpha();
            float beta_activ  = problem.GetActivationDesc().GetBeta();

            float ctime = 0.;
            visit_float(dtype, [&](auto as_float) {
                if(variant != 2)
                {
                    decltype(auto) kernel = handle_.Run(kernels.front());
                    if(resultsave && resultrunning)
                    {
                        kernel(params.x,
                               params.y,
                               params.bnScale,
                               params.bnBias,
                               as_float(inhw),
                               params.expAvgFactor,
                               params.prevResultRunningMean,
                               params.prevResultRunningVariance,
                               params.nextResultRunningMean,
                               params.nextResultRunningVariance,
                               params.epsilon,
                               params.resultSaveMean,
                               params.resultSaveInvVariance,
                               alpha_activ,
                               beta_activ);
                    }
                    else if(resultsave)
                    {
                        kernel(params.x,
                               params.y,
                               params.bnScale,
                               params.bnBias,
                               as_float(inhw),
                               params.epsilon,
                               params.resultSaveMean,
                               params.resultSaveInvVariance,
                               alpha_activ,
                               beta_activ);
                    }
                    else if(resultrunning)
                    {
                        kernel(params.x,
                               params.y,
                               params.bnScale,
                               params.bnBias,
                               as_float(inhw),
                               params.expAvgFactor,
                               params.prevResultRunningMean,
                               params.prevResultRunningVariance,
                               params.nextResultRunningMean,
                               params.nextResultRunningVariance,
                               params.epsilon,
                               alpha_activ,
                               beta_activ);
                    }
                    else
                    {
                        kernel(params.x,
                               params.y,
                               params.bnScale,
                               params.bnBias,
                               as_float(inhw),
                               params.epsilon,
                               alpha_activ,
                               beta_activ);
                    }
                }
                else
                {
                    handle_.Run(kernels[0])(params.x, params.y);
                    profileSequence(handle_, 0, &ctime);

                    if(resultsave && resultrunning)
                    {
                        handle_.Run(kernels[1])(params.y,
                                                as_float(inhw),
                                                params.expAvgFactor,
                                                params.prevResultRunningMean,
                                                params.prevResultRunningVariance,
                                                params.nextResultRunningMean,
                                                params.nextResultRunningVariance,
                                                params.epsilon,
                                                params.resultSaveMean,
                                                params.resultSaveInvVariance);
                    }
                    else if(resultsave)
                    {
                        handle_.Run(kernels[1])(params.y,
                                                as_float(inhw),
                                                params.epsilon,
                                                params.resultSaveMean,
                                                params.resultSaveInvVariance);
                    }
                    else if(resultrunning)
                    {
                        handle_.Run(kernels[1])(params.y,
                                                as_float(inhw),
                                                params.expAvgFactor,
                                                params.prevResultRunningMean,
                                                params.prevResultRunningVariance,
                                                params.nextResultRunningMean,
                                                params.nextResultRunningVariance,
                                                params.epsilon);
                    }
                    else
                    {
                        handle_.Run(kernels[1])(params.y, as_float(inhw), params.epsilon);
                    }

                    profileSequence(handle_, 1, &ctime);

                    handle_.Run(kernels[2])(
                        params.x, params.y, params.bnScale, params.bnBias, alpha_activ, beta_activ);
                    profileSequence(handle_, 2, &ctime);
                }
            });
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
