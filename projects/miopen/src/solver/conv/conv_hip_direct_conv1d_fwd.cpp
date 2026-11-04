/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include <miopen/config.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/stringutils.hpp>

#include <string>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_HIP_CONV1D_FWD)

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Must match MIOpenConv1dFwdHip.cpp.
constexpr int kTileK          = 64;
constexpr int kTileW          = 128;
constexpr int kTileC          = 8;
constexpr int kBlockSize      = 256;
constexpr std::size_t kLdsMax = std::size_t{64} * 1024;
// Below this many output channels per group most of the 64-wide channel tile idles, and Find
// does not pick the solver; skip it rather than compile and benchmark it.
constexpr std::size_t kMinOutChannelsPerGroup = 16;
// Per-dimension HSA grid limits, in work-items.
constexpr std::size_t kMaxGridX  = 0x7fffffff;
constexpr std::size_t kMaxGridYZ = 0xffff;

std::size_t LdsBytes(int r, int stride, int dil)
{
    const auto r_sz = static_cast<std::size_t>(r);
    const auto span = static_cast<std::size_t>(kTileW - 1) * static_cast<std::size_t>(stride) +
                      (r_sz - 1) * static_cast<std::size_t>(dil) + 1;
    return sizeof(float) * kTileC * (span + r_sz * kTileK);
}

} // namespace

bool ConvHipDirectConv1dFwd::IsApplicable(const ExecutionContext& ctx,
                                          const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_HIP_CONV1D_FWD))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // RDNA only. There the CK grouped convolution kernels do not take fp32 inputs, so dilated
    // fp32 1D convolutions otherwise fall back to the naive kernel or GemmFwdRest. Family
    // prefixes rather than a bare "gfx12", which would also match gfx125X.
    const std::string dev_name = ctx.GetStream().GetDeviceName();
    if(!StartsWith(dev_name, "gfx101") && !StartsWith(dev_name, "gfx103") &&
       !StartsWith(dev_name, "gfx110") && !StartsWith(dev_name, "gfx115") &&
       !StartsWith(dev_name, "gfx117") && !StartsWith(dev_name, "gfx120"))
        return false;

    if(!problem.Is2d())
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(problem.GetConv().mode != miopenConvolution)
        return false;
    if(problem.GetConv().paddingMode != miopenPaddingDefault)
        return false;
    if(problem.GetAlphaBetaCase() != DEFAULT)
        return false;
    if(!problem.IsFp32())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;

    // 1D convolution expressed as NCHW with H == 1.
    if(problem.GetInHeight() != 1 || problem.GetWeightsHeight() != 1 || problem.GetPadH() != 0 ||
       problem.GetKernelStrideH() != 1 || problem.GetDilationH() != 1)
        return false;

    const int r      = static_cast<int>(problem.GetWeightsWidth());
    const int stride = problem.GetKernelStrideW();
    const int dil    = problem.GetDilationW();
    if(r < 1 || stride < 1 || dil < 1)
        return false;
    if(LdsBytes(r, stride, dil) > kLdsMax)
        return false;

    const std::size_t groups  = problem.GetGroupCount();
    const std::size_t k_per_g = problem.GetOutChannels() / groups;
    if(k_per_g < kMinOutChannelsPerGroup)
        return false;

    // Grid: x = width tiles * block, y = groups * channel tiles, z = batch.
    const std::size_t wo = problem.GetOutWidth();
    return ((wo + kTileW - 1) / kTileW) * kBlockSize <= kMaxGridX &&
           groups * ((k_per_g + kTileK - 1) / kTileK) <= kMaxGridYZ &&
           problem.GetBatchSize() <= kMaxGridYZ;
}

ConvSolution
ConvHipDirectConv1dFwd::GetSolution(const ExecutionContext&,
                                    const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;
    kernel.kernel_file = "MIOpenConv1dFwdHip.cpp";
    kernel.kernel_name = "MIOpenConv1dFwdHipF32";

    const int r      = static_cast<int>(problem.GetWeightsWidth());
    const int stride = problem.GetKernelStrideW();
    const int dil    = problem.GetDilationW();
    const int pad    = problem.GetPadW();
    const int groups = problem.GetGroupCount();
    const int c      = static_cast<int>(problem.GetInChannels());
    const int wi     = static_cast<int>(problem.GetInWidth());
    const int k      = static_cast<int>(problem.GetOutChannels());
    const int wo_arg = static_cast<int>(problem.GetOutWidth());

    const std::size_t n       = problem.GetBatchSize();
    const std::size_t k_per_g = problem.GetOutChannels() / groups;
    const std::size_t wo      = problem.GetOutWidth();

    kernel.l_wk = {kBlockSize, 1, 1};
    kernel.g_wk = {
        ((wo + kTileW - 1) / kTileW) * kBlockSize, groups * ((k_per_g + kTileK - 1) / kTileK), n};

    kernel.comp_options = " -DKR=" + std::to_string(r) + " -DSTRIDE=" + std::to_string(stride) +
                          " -DDIL=" + std::to_string(dil);

    // The grid above and these arguments both come from the problem; the binary itself only
    // depends on the compile options (filter width, stride, dilation).
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& tensors =
                primitive_parameters.CastTo<miopen::conv::DataInvokeParams>().tensors;
            handle.Run(kern)(tensors.in, tensors.w, tensors.out, c, wi, k, wo_arg, pad, groups);
        };
    };
    result.construction_params.push_back(kernel);
    result.solver_id = SolverDbId();
    return result;
}

#else

bool ConvHipDirectConv1dFwd::IsApplicable(const ExecutionContext&,
                                          const miopen::conv::ProblemDescription&) const
{
    return false;
}

ConvSolution ConvHipDirectConv1dFwd::GetSolution(const ExecutionContext&,
                                                 const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
