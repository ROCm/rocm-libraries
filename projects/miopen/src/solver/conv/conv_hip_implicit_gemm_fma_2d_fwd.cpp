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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_2D_FWD)

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Runs the kernel of ConvHipImplicitGemmFma3dFwd (MIOpenConv3dFwdFmaHip.cpp) with depth 1. NCHW
// weights are read in place (CONV3D_WEI_KCZYX), so unlike the 3D solver this one never needs a
// workspace; that keeps it in Find when the caller allocates only what GetWorkSpaceSize returned
// for a solver without workspace.

// Must match MIOpenConv3dFwdFmaHip.cpp.
constexpr std::size_t kTileM     = 128;
constexpr std::size_t kBlockSize = 256;
// Same limit as the 3D solver: fewer output channels per group leave most of the (at least
// 32-wide) channel tile idle. Depthwise convolutions are left to the depthwise solvers.
constexpr std::size_t kMinOutChannelsPerGroup = 16;
// Per-dimension HSA grid limits, in work-items.
constexpr std::size_t kMaxGridX  = 0x7fffffff;
constexpr std::size_t kMaxGridYZ = 0xffff;

std::size_t GemmM(const miopen::conv::ProblemDescription& problem)
{
    return problem.GetBatchSize() * problem.GetOutHeight() * problem.GetOutWidth();
}

// Output channels per workgroup, chosen as in the 3D solver.
std::size_t TileN(const miopen::conv::ProblemDescription& problem)
{
    const std::size_t k_per_g = problem.GetOutChannels() / problem.GetGroupCount();
    return k_per_g <= 32 ? 32 : (k_per_g <= 64 ? 64 : 128);
}

std::size_t GemmNBlocks(const miopen::conv::ProblemDescription& problem)
{
    const std::size_t groups  = problem.GetGroupCount();
    const std::size_t k_per_g = problem.GetOutChannels() / groups;
    const std::size_t tile_n  = TileN(problem);
    return groups * ((k_per_g + tile_n - 1) / tile_n);
}

} // namespace

bool ConvHipImplicitGemmFma2dFwd::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_2D_FWD))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // RDNA only, as the 3D solver. Family prefixes rather than a bare "gfx12", which would also
    // match gfx125X.
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

    const std::size_t groups = problem.GetGroupCount();
    if(groups > 1 && problem.GetOutChannels() / groups < kMinOutChannelsPerGroup)
        return false;

    // Grid: x = M tiles * block, y = groups * N tiles.
    return ((GemmM(problem) + kTileM - 1) / kTileM) * kBlockSize <= kMaxGridX &&
           GemmNBlocks(problem) <= kMaxGridYZ;
}

ConvSolution
ConvHipImplicitGemmFma2dFwd::GetSolution(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;
    kernel.kernel_file = "MIOpenConv3dFwdFmaHip.cpp";
    kernel.kernel_name = "MIOpenConv3dFwdFmaHip";

    const int dtype = problem.IsFp32() ? 0 : (problem.IsFp16() ? 1 : 2);
    const int nhwc  = problem.IsLayoutNHWC() ? 1 : 0;

    kernel.l_wk = {kBlockSize, 1, 1};
    kernel.g_wk = {((GemmM(problem) + kTileM - 1) / kTileM) * kBlockSize, GemmNBlocks(problem), 1};

    // NHWC weights are K x Y x X x (C / groups) already; NCHW ones are read in place.
    kernel.comp_options = " -DCONV3D_DTYPE=" + std::to_string(dtype) +
                          " -DCONV3D_NDHWC=" + std::to_string(nhwc) +
                          " -DCONV3D_WEI_KCZYX=" + std::to_string(1 - nhwc) +
                          " -DTILE_N=" + std::to_string(TileN(problem));

    const int n      = static_cast<int>(problem.GetBatchSize());
    const int c      = static_cast<int>(problem.GetInChannels());
    const int hi     = static_cast<int>(problem.GetInHeight());
    const int wi     = static_cast<int>(problem.GetInWidth());
    const int k      = static_cast<int>(problem.GetOutChannels());
    const int y      = static_cast<int>(problem.GetWeightsHeight());
    const int x      = static_cast<int>(problem.GetWeightsWidth());
    const int ho     = static_cast<int>(problem.GetOutHeight());
    const int wo     = static_cast<int>(problem.GetOutWidth());
    const int sh     = problem.GetKernelStrideH();
    const int sw     = problem.GetKernelStrideW();
    const int ph     = problem.GetPadH();
    const int pw     = problem.GetPadW();
    const int dh     = problem.GetDilationH();
    const int dw     = problem.GetDilationW();
    const int groups = problem.GetGroupCount();

    result.construction_params.push_back(kernel);

    // The grid above and these arguments all come from the problem; the binaries only depend on
    // the compile options (data type, layout, channel tile width).
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& tensors =
                primitive_parameters.CastTo<miopen::conv::DataInvokeParams>().tensors;
            // Depth 1: Di = Z = Do = 1, depth stride and dilation 1, depth padding 0.
            handle.Run(kernels[0])(tensors.in,
                                   tensors.w,
                                   tensors.out,
                                   n,
                                   c,
                                   1,
                                   hi,
                                   wi,
                                   k,
                                   1,
                                   y,
                                   x,
                                   1,
                                   ho,
                                   wo,
                                   1,
                                   sh,
                                   sw,
                                   0,
                                   ph,
                                   pw,
                                   1,
                                   dh,
                                   dw,
                                   groups);
        };
    };
    result.solver_id = SolverDbId();
    return result;
}

#else

bool ConvHipImplicitGemmFma2dFwd::IsApplicable(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&) const
{
    return false;
}

ConvSolution ConvHipImplicitGemmFma2dFwd::GetSolution(const ExecutionContext&,
                                                      const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
