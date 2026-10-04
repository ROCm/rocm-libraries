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

#include <limits>

namespace miopen {

namespace solver {

namespace batchnorm {

// Spatial multiple needs space for 4 fp32 elements
// per each x thread (including the last workgroup)
// to stash intermediate mean and variance
const unsigned int stash_values_bwd = 4;

namespace {
bool IsSplitBatchProblem(const miopen::batchnorm::ProblemDescription& problem)
{
    const auto& lengths = problem.GetXDesc().GetLengths();
    return problem.GetDirection() == miopen::batchnorm::Direction::Backward &&
           problem.GetMode() == miopenBNSpatial && problem.Is2D() &&
           problem.IsLayoutNCHW() && IsBwdTypeValid(problem) &&
           problem.GetXDesc().IsPacked() && problem.GetDYDesc().IsPacked() &&
           problem.GetDXDesc().IsPacked() &&
           lengths == problem.GetDYDesc().GetLengths() &&
           lengths == problem.GetDXDesc().GetLengths() &&
           lengths[0] >= 8 && lengths[0] <= 64;
}

bool IsSplitBatchConfig(const miopen::batchnorm::ProblemDescription& problem,
                        size_t vectorsize,
                        size_t xlocalsize,
                        size_t ylocalsize,
                        size_t zlocalsize,
                        size_t nelements)
{
    if(!IsSplitBatchProblem(problem) || xlocalsize != 1 ||
       (vectorsize != 1 && vectorsize != 2 && vectorsize != 4 && vectorsize != 8) ||
       (ylocalsize != 256 && ylocalsize != 512) ||
       (zlocalsize != 1 && zlocalsize != 2) || nelements <= 1)
        return false;
    const auto& lengths = problem.GetXDesc().GetLengths();
    const size_t n = lengths[0], c = lengths[1], hw = lengths[2] * lengths[3];
    if(hw == 0 || hw % vectorsize != 0 || n % nelements != 0 || nelements >= n)
        return false;
    const unsigned int stash_values = problem.UseSaved() ? stash_values_bwd / 2 : stash_values_bwd;
    if(!IsSpatialMultipleApplicable(
           problem, vectorsize, stash_values, ylocalsize, zlocalsize, nelements))
        return false;
    const size_t ytail = (hw - 1) % (ylocalsize * vectorsize) + 1;
    const size_t ztail = (n - 1) % (zlocalsize * nelements) + 1;
    const auto method = GetStashMethod(
        false, problem.GetXDesc().GetType(), stash_values, c, n, hw,
        ylocalsize, zlocalsize, nelements);
    const bool packed_half = problem.GetXDesc().GetType() != miopenFloat;
    const size_t stash_slots = stash_values * (packed_half ? 2 : 1);
    return method == 0 ? ytail >= stash_slots + (packed_half ? hw % 2 : 0)
                       : (method == 1 && ztail >= stash_slots && (!packed_half || hw % 2 == 0));
}

bool IsBufferedBackwardConfig(const miopen::batchnorm::ProblemDescription& problem,
                              size_t vectorsize,
                              size_t xlocalsize,
                              size_t ylocalsize,
                              size_t zlocalsize,
                              size_t nelements,
                              bool streaming = false)
{
    if(problem.GetDirection() != miopen::batchnorm::Direction::Backward ||
       problem.GetMode() != miopenBNSpatial || !problem.Is2D() || !problem.IsLayoutNCHW() ||
       !problem.IsBFp16() || !problem.IsScaleFp32() || !problem.UseSaved() ||
       problem.GetDYDesc().GetType() != miopenBFloat16 ||
       problem.GetDXDesc().GetType() != miopenBFloat16 ||
       problem.GetBnBias().GetType() != miopenFloat ||
       problem.GetBnSMean().GetType() != miopenFloat ||
       problem.GetBnSVar().GetType() != miopenFloat || !problem.GetXDesc().IsPacked() ||
       !problem.GetDYDesc().IsPacked() || !problem.GetDXDesc().IsPacked() ||
       !problem.GetBnScale().IsPacked() || !problem.GetBnBias().IsPacked() ||
       !problem.GetBnSMean().IsPacked() || !problem.GetBnSVar().IsPacked() ||
       problem.GetXDesc().GetLengths() != problem.GetDYDesc().GetLengths() ||
       problem.GetXDesc().GetLengths() != problem.GetDXDesc().GetLengths() ||
       (vectorsize != 4 && vectorsize != 8) ||
       (xlocalsize != 256 && xlocalsize != 512 && xlocalsize != 1024) ||
       ylocalsize != 1 || zlocalsize != 1 || nelements != 1)
        return false;
    const auto& lengths = problem.GetXDesc().GetLengths();
    const size_t n = lengths[0], c = lengths[1], h = lengths[2], w = lengths[3];
    const size_t max_hw = streaming ? 32768 : 4800;
    if(n == 0 || n > 64 || c == 0 || h == 0 || h > max_hw || w == 0 || w > max_hw / h ||
       problem.GetBnScale().GetElementSize() != c ||
       problem.GetBnBias().GetElementSize() != c ||
       problem.GetBnSMean().GetElementSize() != c ||
       problem.GetBnSVar().GetElementSize() != c)
        return false;
    const size_t hw = h * w, nhw = n * hw;
    const size_t vector_count = nhw / vectorsize;
    const size_t buffer_size = (vector_count + xlocalsize - 1) / xlocalsize;
    return hw % vectorsize == 0 && (streaming || 2 * buffer_size * vectorsize <= 128) &&
           c <= std::numeric_limits<unsigned int>::max() / nhw;
}
} // namespace

bool PerformanceConfigBnBwdBackward::IsValid(
    const ExecutionContext& context, const miopen::batchnorm::ProblemDescription& problem) const
{
    if(this->kernel_id.empty())
    {
        return false;
    }

    // if default config is variant 2, check if it can be applied
    // (based on variant 2 restrictions)
    size_t vectorsize = 1, xlocalsize = 1, ylocalsize = 1, zlocalsize = 1, nelements = 1;
    int variant = -1;
    GetVariantFromKernelId(
        this->kernel_id, variant, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements);
    if(variant == 4 || variant == 6 || variant == 7)
        return false; // These spatial configurations are forward-only.
    if(variant == 5 || variant == 8)
    {
        const auto& handle = context.GetStream();
        return handle.GetDeviceName() == "gfx1250" && handle.GetWavefrontWidth() == 32 &&
               IsBufferedBackwardConfig(
                   problem, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements, variant == 8);
    }
    if(variant == 2)
    {
        if(vectorsize > 4)
        {
            const auto& handle = context.GetStream();
            if(vectorsize != 8 || handle.GetDeviceName() != "gfx1250" ||
               handle.GetWavefrontWidth() != 32 ||
               problem.GetDirection() != miopen::batchnorm::Direction::Backward ||
               problem.GetMode() != miopenBNSpatial || !IsBwdTypeValid(problem) ||
               !problem.GetXDesc().IsPacked() || !problem.GetDYDesc().IsPacked() ||
               !problem.GetDXDesc().IsPacked() || nelements == 0 ||
               problem.GetXDesc().GetLengths()[0] % nelements != 0 ||
               problem.GetXDesc().GetLengths() != problem.GetDYDesc().GetLengths() ||
               problem.GetXDesc().GetLengths() != problem.GetDXDesc().GetLengths())
                return false;
            if(problem.IsLayoutNCHW())
            {
                if(!IsSplitBatchProblem(problem) || xlocalsize != 1 ||
                   (ylocalsize != 256 && ylocalsize != 512 && ylocalsize != 1024) ||
                   (zlocalsize != 1 && zlocalsize != 2))
                    return false;
            }
            else if(!problem.IsLayoutNHWC() || xlocalsize == 0 || ylocalsize == 0 ||
                    zlocalsize == 0 || xlocalsize > 64 || ylocalsize > 1024 / xlocalsize ||
                    zlocalsize > 1024 / xlocalsize / ylocalsize ||
                    (xlocalsize & (xlocalsize - 1)) != 0)
                return false;
            if(problem.IsLayoutNHWC() && !problem.IsFp32() &&
               problem.GetXDesc().GetLengths()[1] != vectorsize &&
               (problem.GetXDesc().GetLengths()[1] % (2 * vectorsize) != 0 ||
                xlocalsize % 2 != 0))
                return false; // Mixed-precision stash pairs must stay in one vector-channel row.
        }
        const auto& handle = context.GetStream();
        const bool validate_split =
            problem.IsBFp16() ||
            (handle.GetDeviceName() == "gfx1250" && handle.GetWavefrontWidth() == 32 &&
             IsSplitBatchProblem(problem));
        if(problem.IsLayoutNCHW() && validate_split &&
           nelements < problem.GetXDesc().GetLengths()[0])
        {
            if(handle.GetDeviceName() != "gfx1250" || handle.GetWavefrontWidth() != 32 ||
               !IsSplitBatchConfig(
                   problem, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements))
                return false;
        }
        unsigned int stash_values = !problem.UseSaved() ? stash_values_bwd : stash_values_bwd / 2;
        return IsSpatialMultipleApplicable(
            problem, vectorsize, stash_values, ylocalsize, zlocalsize, nelements);
    }
    return true;
}

void PerformanceConfigBnBwdBackward::HeuristicInit(
    const miopen::batchnorm::ProblemDescription& problem)
{
    unsigned int stash_values = !problem.UseSaved() ? stash_values_bwd : stash_values_bwd / 2;
    // Define default configuration based on heuristics and
    // add all other valid configurations for the given problem
    if(UseMultiple(problem))
    {
        DefaultConfigSpatialMultiple(problem, stash_values, this->valid_kernels);
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
            DefaultConfigSpatialMultiple(problem, stash_values, this->valid_kernels);
        }
    }

    if(IsSplitBatchProblem(problem))
    {
        const size_t n = problem.GetXDesc().GetLengths()[0];
        std::vector<size_t> batch_elements;
        if(problem.IsBFp16())
        {
            for(size_t target : {size_t{32}, size_t{16}, size_t{8}})
            {
                size_t nelements = std::min(target, n / 2);
                while(n % nelements != 0)
                    --nelements;
                if(nelements > 1 &&
                   std::find(batch_elements.begin(), batch_elements.end(), nelements) ==
                       batch_elements.end())
                    batch_elements.push_back(nelements);
            }
        }
        else
        {
            // Exact divisors keep every batch load in bounds, including the final z group.
            // Append these candidates after the established heuristics; defaults stay unchanged.
            for(size_t nelements = n / 2; nelements > 1; --nelements)
                if(n % nelements == 0)
                    batch_elements.push_back(nelements);
        }
        for(size_t vectorsize : {size_t{4}, size_t{2}, size_t{1}})
            for(size_t ylocalsize : {size_t{256}, size_t{512}})
                for(size_t zlocalsize : {size_t{1}, size_t{2}})
                    for(size_t nelements : batch_elements)
                    {
                        if(zlocalsize == 2 && (ylocalsize != 256 || nelements > 16))
                            continue;
                        if(IsSplitBatchConfig(
                               problem, vectorsize, 1, ylocalsize, zlocalsize, nelements))
                            valid_kernels.push_back(GetKernelIdFromVariant(
                                2, vectorsize, 1, ylocalsize, zlocalsize, nelements));
                    }
        const auto& lengths = problem.GetXDesc().GetLengths();
        const size_t c = lengths[1], hw = lengths[2] * lengths[3];
        if(c >= 128 || (c <= 64 && hw >= 8192))
        {
            for(size_t ylocalsize : {size_t{256}, size_t{512}, size_t{1024}})
                if(IsSpatialMultipleApplicable(problem, 8, stash_values, ylocalsize, 1, n))
                    valid_kernels.push_back(GetKernelIdFromVariant(2, 8, 1, ylocalsize, 1, n));
            for(size_t nelements : batch_elements)
                for(size_t ylocalsize : {size_t{256}, size_t{512}})
                    for(size_t zlocalsize : {size_t{1}, size_t{2}})
                        if(IsSplitBatchConfig(problem, 8, 1, ylocalsize, zlocalsize, nelements))
                            valid_kernels.push_back(
                                GetKernelIdFromVariant(2, 8, 1, ylocalsize, zlocalsize, nelements));
        }
    }

    for(size_t vectorsize : {size_t{4}, size_t{8}})
        for(size_t xlocalsize : {size_t{256}, size_t{512}, size_t{1024}})
            if(IsBufferedBackwardConfig(problem, vectorsize, xlocalsize, 1, 1, 1))
                valid_kernels.push_back(
                    GetKernelIdFromVariant(5, vectorsize, xlocalsize, 1, 1, 1));

    if(problem.Is2D())
    {
        const auto& lengths = problem.GetXDesc().GetLengths();
        const size_t c = lengths[1], hw = lengths[2] * lengths[3];
        if(c >= 128 && hw >= 4096 && hw <= 8192)
            for(size_t vectorsize : {size_t{4}, size_t{8}})
                for(size_t xlocalsize : {size_t{256}, size_t{512}, size_t{1024}})
                    if(IsBufferedBackwardConfig(problem, vectorsize, xlocalsize, 1, 1, 1, true))
                        valid_kernels.push_back(
                            GetKernelIdFromVariant(8, vectorsize, xlocalsize, 1, 1, 1));
    }

    // Set index and kernel_id to default value
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

bool PerformanceConfigBnBwdBackward::SetNextValue(
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

bool PerformanceConfigBnBwdBackward::operator==(const PerformanceConfigBnBwdBackward& other) const
{
    return this->kernel_id == other.kernel_id;
}

bool PerformanceConfigBnBwdBackward::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool BnBwdTrainingSpatial::IsApplicable(
    const ExecutionContext&, const miopen::batchnorm::ProblemDescription& bn_problem) const
{
    if(bn_problem.GetDirection() != miopen::batchnorm::Direction::Backward ||
       bn_problem.GetMode() != miopenBNSpatial)
        return false;

    if(!bn_problem.Is2D())
        return false;

#if WORKAROUND_ISSUE_1549_FP16_BUILD_ERROR
    if(bn_problem.GetXDesc().GetType() == miopenHalf &&
       bn_problem.GetBnScale().GetType() == miopenHalf)
    {
        // bfp16parm = true;
        // Unsupported kernel mode, error in kernel code
        // MIOpenBatchNormBwdSpatial.cl:526 issue#1549
        return false;
    }
#endif
    if(!IsBwdTypeValid(bn_problem))
        return false;

    int activ_mode = bn_problem.GetActivationDesc().GetMode();
    if(activ_mode != miopenActivationPASTHRU && activ_mode != miopenActivationRELU &&
       activ_mode != miopenActivationCLIPPEDRELU && activ_mode != miopenActivationCLAMP)
    {
        return false;
    }

    return true;
}

PerformanceConfigBnBwdBackward BnBwdTrainingSpatial::GetDefaultPerformanceConfig(
    const ExecutionContext& context, const miopen::batchnorm::ProblemDescription& problem_desc) const
{
    PerformanceConfigBnBwdBackward pp;
    pp.HeuristicInit(problem_desc);
    const auto& handle = context.GetStream();
    if(handle.GetDeviceName() == "gfx1250" && handle.GetWavefrontWidth() == 32 &&
       problem_desc.IsBFp16() && IsSplitBatchProblem(problem_desc) && problem_desc.UseSaved())
    {
        const auto& lengths = problem_desc.GetXDesc().GetLengths();
        const size_t n = lengths[0], c = lengths[1], hw = lengths[2] * lengths[3];
        // Low-channel spatial problems need more independent batch workgroups.
        // Restrict the default to the measured crossover; other candidates remain tunable.
        if(n >= 16 && c <= 64 && hw >= 4096 && hw <= 8192)
        {
            size_t nelements = std::min(size_t{8}, n / 2);
            while(n % nelements != 0)
                --nelements;
            const auto split = PerformanceConfigBnBwdBackward{
                0, GetKernelIdFromVariant(2, 4, 1, 256, 2, nelements)};
            if(split.IsValid(context, problem_desc))
            {
                pp.kernel_id = split.kernel_id;
                pp.index = std::distance(
                    pp.valid_kernels.begin(),
                    std::find(pp.valid_kernels.begin(), pp.valid_kernels.end(), pp.kernel_id));
            }
        }
    }
    const auto buffered = PerformanceConfigBnBwdBackward{
        0, GetKernelIdFromVariant(5, 8, 1024, 1, 1, 1)};
    if(buffered.IsValid(context, problem_desc))
    {
        const auto& lengths = problem_desc.GetXDesc().GetLengths();
        const size_t c = lengths[1], hw = lengths[2] * lengths[3];
        // Retaining both inputs wins for the measured high-channel, small-spatial family.
        // Leave smaller channels and spatial sizes on their established defaults.
        if(c >= 128 && hw >= 512)
        {
            pp.kernel_id = buffered.kernel_id;
            pp.index = std::distance(
                pp.valid_kernels.begin(),
                std::find(pp.valid_kernels.begin(), pp.valid_kernels.end(), pp.kernel_id));
        }
    }
    const auto streaming = PerformanceConfigBnBwdBackward{
        0, GetKernelIdFromVariant(8, 8, 1024, 1, 1, 1)};
    if(streaming.IsValid(context, problem_desc))
    {
        const auto& lengths = problem_desc.GetXDesc().GetLengths();
        const size_t c = lengths[1], hw = lengths[2] * lengths[3], nhw = lengths[0] * hw;
        // The streaming crossover is limited to the measured medium-spatial family.
        if(c >= 128 && c <= 256 && hw >= 4096 && hw <= 8192 &&
           nhw >= 65537 && nhw <= 262144)
        {
            pp.kernel_id = streaming.kernel_id;
            pp.index = std::distance(
                pp.valid_kernels.begin(),
                std::find(pp.valid_kernels.begin(), pp.valid_kernels.end(), pp.kernel_id));
        }
    }
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool BnBwdTrainingSpatial::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const miopen::batchnorm::ProblemDescription& problem_desc,
    const PerformanceConfigBnBwdBackward& config) const
{
    return config.IsValid(ctx, problem_desc);
}

PerformanceConfigBnBwdBackward
BnBwdTrainingSpatial::Search(const ExecutionContext& ctx,
                             const miopen::batchnorm::ProblemDescription& problem,
                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution BnBwdTrainingSpatial::GetSolution(const ExecutionContext& context,
                                               const miopen::batchnorm::ProblemDescription& problem,
                                               const PerformanceConfigBnBwdBackward& config) const
{
    const auto& handle      = context.GetStream();
    const unsigned wavesize = handle.GetWavefrontWidth();
    int variant       = -1;
    size_t vectorsize = 1;
    size_t xlocalsize = 1, xgridsize = 1;
    size_t ylocalsize = 1, ygridsize = 1;
    size_t zlocalsize = 1, zgridsize = 1;
    unsigned int ldsgcn = 0, ldsnogcn = 0;
    int stash_method = 0;
    size_t nelements = 1;

    GetVariantFromKernelId(
        config.kernel_id, variant, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements);
    if(variant == 4 || variant == 6 || variant == 7)
        MIOPEN_THROW(miopenStatusBadParm, "Batchnorm configuration is forward-only");
    if((variant == 5 || variant == 8) && !config.IsValid(context, problem))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported buffered or streaming backward configuration");

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

    auto inhw = float(1.0 / in_nhw);

    if(variant == 2 &&
       (vectorsize > 4 ||
        (problem.IsLayoutNCHW() && nelements < static_cast<size_t>(n) &&
         (problem.IsBFp16() ||
          (handle.GetDeviceName() == "gfx1250" && wavesize == 32 &&
           IsSplitBatchProblem(problem))))) &&
       !config.IsValid(context, problem))
        MIOPEN_THROW(miopenStatusBadParm, "Unsupported vectorized or split-batch backward configuration");

    size_t xlocalsize_final = xlocalsize, ylocalsize_final = ylocalsize,
           zlocalsize_final = zlocalsize;
    if(variant != 2)
    {
        if(variant != 5 && variant != 8)
            xlocalsize = 1024;
        xgridsize  = xlocalsize * c;
        ldsgcn     = xlocalsize / wavesize;
        ldsnogcn   = xlocalsize;
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

        unsigned int stash_values = !problem.UseSaved() ? stash_values_bwd : stash_values_bwd / 2;
        // Get the stash method based on problem size and WG size
        stash_method = GetStashMethod(problem.IsLayoutNHWC(),
                                      problem.GetXDesc().GetType(),
                                      stash_values,
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
        ldsgcn   = xlocalsize * ylocalsize * zlocalsize / wavesize;
    }

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto kernel = KernelInfo{};

        auto build_params =
            KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(bfp16parm)},
                                  {"MIOPEN_USE_RNE_BFLOAT16", MIOPEN_USE_RNE_BFLOAT16},
                                  {"MIOPEN_USE_FP32", static_cast<int>(bfp32parm)},
                                  {"MIOPEN_USE_FPMIX", static_cast<int>(bfpmixparm)},
                                  {"MIOPEN_USE_BFPMIX", static_cast<int>(bbfpmixparam)},
                                  {"MIO_BN_USESAVED", static_cast<int>(problem.UseSaved())},
                                  {"MIO_BN_N", static_cast<int>(n)},
                                  {"MIO_BN_C", static_cast<int>(c)},
                                  {"MIO_BN_HW", static_cast<int>(in_cstride)},
                                  {"MIO_BN_NHW", static_cast<int>(in_nhw)},
                                  {"MIO_BN_CHW", in_nstride},
                                  {"MIO_BN_NCHW", in_nchw},
                                  {"MIO_BN_NGRPS", ygridsize / ylocalsize},
                                  {"MIO_BN_NGRPS2", zgridsize / zlocalsize},
                                  {"MIO_BN_N_ELEMENTS", nelements},
                                  {"MIO_BN_LDS_SIZE", ldsnogcn},
                                  {"MIO_BN_LDSGCN_SIZE", ldsgcn},
                                  {"MIO_BN_VARIANT", variant},
                                  {"MIO_WAVESIZE", wavesize},
                                  {"MIO_BN_GRP0", xlocalsize},
                                  {"MIO_BN_GRP1", ylocalsize},
                                  {"MIO_BN_GRP2", zlocalsize},
                                  {"MIO_BN_GRP0_FINAL", xlocalsize_final},
                                  {"MIO_BN_GRP1_FINAL", ylocalsize_final},
                                  {"MIO_BN_GRP2_FINAL", zlocalsize_final},
                                  {"MIO_BN_GFX125X", StartsWith(handle.GetDeviceName(), "gfx125") ? 1 : 0},
                                  {"MIO_LAYOUT_NHWC", static_cast<int>(problem.IsLayoutNHWC())},
                                  {"MIO_BN_VECTORIZE", static_cast<int>(vectorsize > 1)},
                                  {"MIO_BN_VEC_SIZE", vectorsize},
                                  {"MIO_BN_BUFFERED_GFX1250",
                                   handle.GetDeviceName() == "gfx1250" && wavesize == 32 ? 1 : 0},
                                  {"MIO_BN_STASH_METHOD", stash_method},
                                  {"MIOPEN_NRN_OP_ID", problem.GetActivationDesc().GetMode()}};

        {
            kernel.kernel_file      = "MIOpenBatchNormBwdSpatial.cpp";
            std::string kernel_name = "MIOpenBatchNormBwdSpatial";

            build_params << KernelBuildParameters{
                {"MIO_BN_GFX103X", (StartsWith(handle.GetDeviceName(), "gfx103") ? "1" : "0")},
                {"MIO_BN_GFX110X", (StartsWith(handle.GetDeviceName(), "gfx110") ? "1" : "0")},
                {"MIO_BN_GFX115X", (StartsWith(handle.GetDeviceName(), "gfx115") ? "1" : "0")},
                {"MIO_BN_GFX120X", (StartsWith(handle.GetDeviceName(), "gfx120") ? "1" : "0")},
                {"MIO_BN_GFX125X", (StartsWith(handle.GetDeviceName(), "gfx125") ? "1" : "0")},
            };

            build_params.Define("HIP_ENABLE_EXTRA_WARP_SYNC_TYPES");

            kernel.comp_options = build_params.GenerateFor(kbp::HIP());

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            if(variant != 2)
            {
                kernel.kernel_name = kernel_name;
                result.construction_params.push_back(kernel);
            }
            else
            {
                auto single_yzgroup_kernel = kernel;

                single_yzgroup_kernel.l_wk[0] = xlocalsize_final;
                single_yzgroup_kernel.l_wk[1] = ylocalsize_final;
                single_yzgroup_kernel.l_wk[2] = zlocalsize_final;
                single_yzgroup_kernel.g_wk[1] = ylocalsize_final;
                single_yzgroup_kernel.g_wk[2] = zlocalsize_final;

                if(!problem.UseSaved())
                {
                    kernel.kernel_name = kernel_name + "MeanVariance";
                    result.construction_params.push_back(kernel);

                    single_yzgroup_kernel.kernel_name = kernel_name + "FinalMeanVariance";
                    result.construction_params.push_back(single_yzgroup_kernel);
                }

                kernel.kernel_name = kernel_name + "DScaleDBias";
                result.construction_params.push_back(kernel);

                single_yzgroup_kernel.kernel_name = kernel_name + "FinalDScaleDBias";
                result.construction_params.push_back(single_yzgroup_kernel);

                kernel.kernel_name = kernel_name + "DX";
                result.construction_params.push_back(kernel);
            }
        }
    }

    const auto dtype    = problem.GetBnScale().GetType();
    const auto useSaved = problem.UseSaved();

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::batchnorm::BwdInvokeParams>();

            float alpha_activ = problem.GetActivationDesc().GetAlpha();
            float beta_activ  = problem.GetActivationDesc().GetBeta();
            float ctime       = 0.;
            visit_float(dtype, [&](auto as_float) {
                if(variant != 2)
                {
                    decltype(auto) kernel = handle_.Run(kernels.front());
                    if(useSaved)
                    {
                        kernel(params.x,
                               params.dy,
                               params.dx,
                               params.bnScale,
                               params.bnBias,
                               params.resultBnScaleDiff,
                               params.resultBnBiasDiff,
                               params.savedMean,
                               params.savedInvVariance,
                               as_float(inhw),
                               alpha_activ,
                               beta_activ);
                    }
                    else
                    {
                        kernel(params.x,
                               params.dy,
                               params.dx,
                               params.bnScale,
                               params.bnBias,
                               params.resultBnScaleDiff,
                               params.resultBnBiasDiff,
                               params.epsilon,
                               inhw,
                               alpha_activ,
                               beta_activ);
                    }
                }
                else
                {
                    if(useSaved)
                    {
                        handle_.Run(kernels[0])(params.x,
                                                params.dy,
                                                params.dx,
                                                params.bnScale,
                                                params.bnBias,
                                                params.savedMean,
                                                params.savedInvVariance,
                                                alpha_activ,
                                                beta_activ);
                        profileSequence(handle_, 0, &ctime);

                        handle_.Run(kernels[1])(
                            params.dx, params.resultBnScaleDiff, params.resultBnBiasDiff);
                        profileSequence(handle_, 1, &ctime);

                        handle_.Run(kernels[2])(params.x,
                                                params.dy,
                                                params.dx,
                                                params.bnScale,
                                                params.bnBias,
                                                params.resultBnScaleDiff,
                                                params.resultBnBiasDiff,
                                                params.savedMean,
                                                params.savedInvVariance,
                                                as_float(inhw),
                                                alpha_activ,
                                                beta_activ);
                        profileSequence(handle_, 2, &ctime);
                    }
                    else
                    {
                        handle_.Run(kernels[0])(params.x, params.dx); // mean variance
                        profileSequence(handle_, 0, &ctime);

                        handle_.Run(kernels[1])(
                            params.dx, as_float(inhw), params.epsilon); // final mean variance
                        profileSequence(handle_, 1, &ctime);

                        handle_.Run(kernels[2])(params.x,
                                                params.dy,
                                                params.dx, // dscale dbias
                                                params.bnScale,
                                                params.bnBias,
                                                alpha_activ,
                                                beta_activ);
                        profileSequence(handle_, 1, &ctime);

                        handle_.Run(kernels[3])(params.dx,
                                                params.resultBnScaleDiff,
                                                params.resultBnBiasDiff); // final dscale dbias
                        profileSequence(handle_, 1, &ctime);

                        handle_.Run(kernels[4])(params.x,
                                                params.dy,
                                                params.dx,
                                                params.bnScale,
                                                params.bnBias,
                                                params.resultBnScaleDiff,
                                                params.resultBnBiasDiff,
                                                as_float(inhw),
                                                alpha_activ,
                                                beta_activ);
                        profileSequence(handle_, 2, &ctime);
                    }
                }
            });
        };
    };

    return result;
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
