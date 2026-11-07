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

#include <algorithm>
#include <string>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_3D_FWD)

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Must match MIOpenConv3dFwdFmaHip.cpp.
constexpr std::size_t kTileM     = 128;
constexpr std::size_t kBlockSize = 256;
// Grouped convolutions with fewer output channels per group leave most of the (at least 32-wide)
// channel tile idle; on gfx1151 a depthwise 3x3x3 case ran 30x slower than ConvDirectNaiveConvFwd.
// Ungrouped convolutions are kept at any channel count.
constexpr std::size_t kMinOutChannelsPerGroup = 16;
// Workgroups for the weight transpose; it loops over the remaining elements.
constexpr std::size_t kMaxTransposeGroups = 1024;
// Per-dimension HSA grid limits, in work-items.
constexpr std::size_t kMaxGridX  = 0x7fffffff;
constexpr std::size_t kMaxGridYZ = 0xffff;

std::size_t GemmM(const miopen::conv::ProblemDescription& problem)
{
    return problem.GetBatchSize() * problem.GetOutDepth() * problem.GetOutHeight() *
           problem.GetOutWidth();
}

// Output channels per workgroup. Narrower tiles for few output channels: on gfx1151 the 128-wide
// tile was 1.5x slower than the 64-wide one at K = 64, and the 32-wide one was 1.3x faster than
// the 64-wide one at K = 32.
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

std::size_t FilterTaps(const miopen::conv::ProblemDescription& problem)
{
    return problem.GetWeightsDepth() * problem.GetWeightsHeight() * problem.GetWeightsWidth();
}

// The kernel reads the weights as K x taps x (C / groups). NCDHW weights are K x (C / groups) x
// taps, which is the same thing only for a 1x1x1 filter; otherwise they are transposed into the
// workspace first.
bool NeedsWeightTranspose(const miopen::conv::ProblemDescription& problem)
{
    return !problem.IsLayoutNHWC() && FilterTaps(problem) > 1;
}

} // namespace

bool ConvHipImplicitGemmFma3dFwd::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_3D_FWD))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // RDNA only. gfx101X and gfx103X have no matrix instructions, so the CK grouped 3D kernels
    // do not exist there; on gfx110X and later the CK kernels do not take fp32 inputs. Family
    // prefixes rather than a bare "gfx12", which would also match gfx125X.
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

    const std::size_t groups = problem.GetGroupCount();
    if(groups > 1 && problem.GetOutChannels() / groups < kMinOutChannelsPerGroup)
        return false;

    // Grid: x = M tiles * block, y = groups * N tiles.
    return ((GemmM(problem) + kTileM - 1) / kTileM) * kBlockSize <= kMaxGridX &&
           GemmNBlocks(problem) <= kMaxGridYZ;
}

ConvSolution
ConvHipImplicitGemmFma3dFwd::GetSolution(const ExecutionContext& ctx,
                                         const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;
    kernel.kernel_file = "MIOpenConv3dFwdFmaHip.cpp";
    kernel.kernel_name = "MIOpenConv3dFwdFmaHip";

    const int dtype = problem.IsFp32() ? 0 : (problem.IsFp16() ? 1 : 2);
    const int ndhwc = problem.IsLayoutNHWC() ? 1 : 0;

    kernel.l_wk = {kBlockSize, 1, 1};
    kernel.g_wk = {((GemmM(problem) + kTileM - 1) / kTileM) * kBlockSize, GemmNBlocks(problem), 1};

    kernel.comp_options = " -DCONV3D_DTYPE=" + std::to_string(dtype) +
                          " -DCONV3D_NDHWC=" + std::to_string(ndhwc) +
                          " -DTILE_N=" + std::to_string(TileN(problem));

    const int n       = static_cast<int>(problem.GetBatchSize());
    const int c       = static_cast<int>(problem.GetInChannels());
    const int di      = static_cast<int>(problem.GetInDepth());
    const int hi      = static_cast<int>(problem.GetInHeight());
    const int wi      = static_cast<int>(problem.GetInWidth());
    const int k       = static_cast<int>(problem.GetOutChannels());
    const int z       = static_cast<int>(problem.GetWeightsDepth());
    const int y       = static_cast<int>(problem.GetWeightsHeight());
    const int x       = static_cast<int>(problem.GetWeightsWidth());
    const int d_o     = static_cast<int>(problem.GetOutDepth());
    const int ho      = static_cast<int>(problem.GetOutHeight());
    const int wo      = static_cast<int>(problem.GetOutWidth());
    const int sd      = problem.GetKernelStrideD();
    const int sh      = problem.GetKernelStrideH();
    const int sw      = problem.GetKernelStrideW();
    const int pd      = problem.GetPadD();
    const int ph      = problem.GetPadH();
    const int pw      = problem.GetPadW();
    const int dd      = problem.GetDilationD();
    const int dh      = problem.GetDilationH();
    const int dw      = problem.GetDilationW();
    const int groups  = problem.GetGroupCount();
    const int c_per_g = c / groups;
    const int taps    = z * y * x;

    result.construction_params.push_back(kernel);

    const bool transpose = NeedsWeightTranspose(problem);
    const auto ws_size   = GetWorkspaceSize(ctx, problem);
    if(transpose)
    {
        KernelInfo wei_kernel   = kernel;
        wei_kernel.kernel_name  = "MIOpenConv3dFwdFmaHipWeiToKzyxc";
        const std::size_t total = problem.GetWeightsSize() / GetTypeSize(problem.GetInDataType());
        const std::size_t wgs =
            std::min((total + kBlockSize - 1) / kBlockSize, kMaxTransposeGroups);
        wei_kernel.g_wk = {wgs * kBlockSize, 1, 1};
        result.construction_params.push_back(wei_kernel);
    }

    // The grids above and these arguments all come from the problem; the binaries only depend on
    // the compile options (data type, layout, channel tile width).
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& params  = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors = params.tensors;
            auto wei            = tensors.w;
            float elapsed       = 0.f;
            if(transpose)
            {
                if(params.workSpace == nullptr || params.workSpaceSize < ws_size)
                    MIOPEN_THROW("Not enough workspace has been provided for "
                                 "ConvHipImplicitGemmFma3dFwd.");
                handle.Run(kernels[1])(tensors.w, params.workSpace, k, c_per_g, taps);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
                wei = params.workSpace;
            }
            handle.Run(kernels[0])(tensors.in,
                                   wei,
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
                                   dw,
                                   groups);
            if(transpose && handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    result.workspace_sz = ws_size;
    result.solver_id    = SolverDbId();
    return result;
}

std::size_t
ConvHipImplicitGemmFma3dFwd::GetWorkspaceSize(const ExecutionContext&,
                                              const miopen::conv::ProblemDescription& problem) const
{
    return NeedsWeightTranspose(problem) ? problem.GetWeightsSize() : 0;
}

#else

bool ConvHipImplicitGemmFma3dFwd::IsApplicable(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&) const
{
    return false;
}

ConvSolution ConvHipImplicitGemmFma3dFwd::GetSolution(const ExecutionContext&,
                                                      const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

std::size_t
ConvHipImplicitGemmFma3dFwd::GetWorkspaceSize(const ExecutionContext&,
                                              const miopen::conv::ProblemDescription&) const
{
    return 0;
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
