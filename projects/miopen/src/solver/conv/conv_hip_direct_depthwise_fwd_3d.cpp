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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_HIP_DEPTHWISE_3D_FWD)

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Must match MIOpenConvDepthwise3dFwdHip.cpp.
constexpr std::size_t kTileW     = 4;
constexpr std::size_t kBlockSize = 256;
// HSA grid limit in work-items (1D grid).
constexpr std::size_t kMaxGridX = 0x7fffffff;

std::size_t WorkItems(const miopen::conv::ProblemDescription& problem)
{
    return problem.GetBatchSize() * problem.GetOutChannels() * problem.GetOutDepth() *
           problem.GetOutHeight() * ((problem.GetOutWidth() + kTileW - 1) / kTileW);
}

} // namespace

bool ConvHipDirectDepthwiseFwd3d::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_HIP_DEPTHWISE_3D_FWD))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // RDNA only. There no other solver is built for depthwise 3D convolutions:
    // ConvDepthwiseFwd3D covers gfx942 / gfx950 only, and the CK grouped kernels handle one
    // channel per group poorly or, on gfx101X / gfx103X, not at all. Family prefixes rather than
    // a bare "gfx12", which would also match gfx125X.
    const std::string dev_name = ctx.GetStream().GetDeviceName();
    if(!StartsWith(dev_name, "gfx101") && !StartsWith(dev_name, "gfx103") &&
       !StartsWith(dev_name, "gfx110") && !StartsWith(dev_name, "gfx115") &&
       !StartsWith(dev_name, "gfx117") && !StartsWith(dev_name, "gfx120"))
        return false;

    if(!problem.Is3d())
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(problem.GetConv().mode != miopenConvolution)
        return false;
    if(problem.GetConv().paddingMode != miopenPaddingDefault)
        return false;
    if(problem.GetAlphaBetaCase() != DEFAULT)
        return false;
    if(!problem.IsFp32() && !problem.IsFp16() && !problem.IsBfp16())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsLayoutDefault() && !problem.IsLayoutNHWC())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;

    // Depthwise: one input channel per group, output channels a multiple of the input channels.
    const std::size_t groups = problem.GetGroupCount();
    const std::size_t c      = problem.GetInChannels();
    const std::size_t k      = problem.GetOutChannels();
    if(groups < 2 || groups != c || k % c != 0)
        return false;

    return ((WorkItems(problem) + kBlockSize - 1) / kBlockSize) * kBlockSize <= kMaxGridX;
}

ConvSolution
ConvHipDirectDepthwiseFwd3d::GetSolution(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;
    kernel.kernel_file = "MIOpenConvDepthwise3dFwdHip.cpp";
    kernel.kernel_name = "MIOpenConvDepthwise3dFwdHip";

    const int dtype = problem.IsFp32() ? 0 : (problem.IsFp16() ? 1 : 2);
    const int ndhwc = problem.IsLayoutNHWC() ? 1 : 0;

    kernel.l_wk = {kBlockSize, 1, 1};
    kernel.g_wk = {((WorkItems(problem) + kBlockSize - 1) / kBlockSize) * kBlockSize, 1, 1};

    kernel.comp_options =
        " -DDW_DTYPE=" + std::to_string(dtype) + " -DDW_NDHWC=" + std::to_string(ndhwc);
    result.construction_params.push_back(kernel);

    const int n   = static_cast<int>(problem.GetBatchSize());
    const int c   = static_cast<int>(problem.GetInChannels());
    const int di  = static_cast<int>(problem.GetInDepth());
    const int hi  = static_cast<int>(problem.GetInHeight());
    const int wi  = static_cast<int>(problem.GetInWidth());
    const int k   = static_cast<int>(problem.GetOutChannels());
    const int z   = static_cast<int>(problem.GetWeightsDepth());
    const int y   = static_cast<int>(problem.GetWeightsHeight());
    const int x   = static_cast<int>(problem.GetWeightsWidth());
    const int d_o = static_cast<int>(problem.GetOutDepth());
    const int ho  = static_cast<int>(problem.GetOutHeight());
    const int wo  = static_cast<int>(problem.GetOutWidth());
    const int sd  = problem.GetKernelStrideD();
    const int sh  = problem.GetKernelStrideH();
    const int sw  = problem.GetKernelStrideW();
    const int pd  = problem.GetPadD();
    const int ph  = problem.GetPadH();
    const int pw  = problem.GetPadW();
    const int dd  = problem.GetDilationD();
    const int dh  = problem.GetDilationH();
    const int dw  = problem.GetDilationW();

    // The grid above and these arguments both come from the problem; the binary itself only
    // depends on the compile options (data type, layout).
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& tensors =
                primitive_parameters.CastTo<miopen::conv::DataInvokeParams>().tensors;
            handle.Run(kern)(tensors.in,
                             tensors.w,
                             tensors.out,
                             n,
                             c,
                             di,
                             hi,
                             wi,
                             k,
                             z,
                             y,
                             x,
                             d_o,
                             ho,
                             wo,
                             sd,
                             sh,
                             sw,
                             pd,
                             ph,
                             pw,
                             dd,
                             dh,
                             dw);
        };
    };
    result.solver_id = SolverDbId();
    return result;
}

#else

bool ConvHipDirectDepthwiseFwd3d::IsApplicable(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&) const
{
    return false;
}

ConvSolution ConvHipDirectDepthwiseFwd3d::GetSolution(const ExecutionContext&,
                                                      const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
