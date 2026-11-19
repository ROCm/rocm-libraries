/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/algorithm.hpp>
#include <miopen/conv/asm_implicit_gemm.hpp>
#include <miopen/env.hpp>
#include <miopen/solver/gemm_common.hpp>

// Set to 0 to disable the small-batch NHWC backward GEMM exclusion.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM_NHWC_BACKWARD_SMALL_BATCH)

namespace miopen {
namespace solver {
namespace conv {
namespace gemm {

bool IsAnyBufferBf16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenBFloat16    //
           || yDesc.GetType() == miopenBFloat16 //
           || wDesc.GetType() == miopenBFloat16;
}

bool IsAnyBufferFp16(const TensorDescriptor& xDesc,
                     const TensorDescriptor& yDesc,
                     const TensorDescriptor& wDesc)
{
    return xDesc.GetType() == miopenHalf    //
           || yDesc.GetType() == miopenHalf //
           || wDesc.GetType() == miopenHalf;
}

double SlowdownFactor(const int n_oper, const double oper_factor, const double multiple_oper_factor)
{
    if(n_oper > 0)
    {
        auto rv = oper_factor;
        if(n_oper > 1)
            rv *= multiple_oper_factor;
        return rv;
    }
    else
        return 1.0;
}

bool IsSmallBatchNhwcBackwardExcluded(const std::string& device_name,
                                      const miopen::conv::ProblemDescription& problem)
{
    if(env::disabled(MIOPEN_DEBUG_CONV_GEMM_NHWC_BACKWARD_SMALL_BATCH))
        return false;
    constexpr std::size_t small_batch_cutoff = 4; // batch sizes 1..3
    const auto& conv                         = problem.GetConv();
    if(!(problem.IsDirectionBackwardWrW() || problem.IsDirectionBackwardData()) ||
       !problem.IsLayoutNHWC() || conv.GetSpatialDimension() != 2 || conv.group_count != 1 ||
       device_name != "gfx942" || problem.GetBatchSize() >= small_batch_cutoff)
        return false;
    // Validated regime: 1x1 filter, unit stride, zero pad only. GemmBwdRest / GemmWrwUniversal
    // also match other strides (e.g. the 1x1 stride-2 downsample), which is out of scope here.
    const auto& wei = problem.GetWeights().GetLengths(); // [K, C, fy, fx]
    if(wei[2] != 1 || wei[3] != 1)
        return false;
    if(!miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }) ||
       !miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }))
        return false;
    // Only exclude GEMM where the ASM Xdlops fallback can address the tensors, so selection
    // reverts to a fast implicit-GEMM solver. Reuses ASM's own size gate, which is direction-aware:
    // wrw drops out above the 32-bit byte-offset limit (keep GEMM), while bwd-data stays
    // addressable via batch split.
    return !igemm_tensor_size_exceeds_asm_gtc_addressing(problem);
}

} // namespace gemm
} // namespace conv
} // namespace solver
} // namespace miopen
