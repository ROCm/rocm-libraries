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
constexpr std::size_t kTileH = 4;
// The filter, H / W strides and dilations are compile-time constants and the input windows live
// in registers; these bounds keep the per-thread register footprint (output tile, filter rows,
// input window) small enough not to spill. A 7x7 filter with stride 2 (21-wide window) compiles
// without spills for gfx1010 through gfx1201; a 32-wide window (7x7, dilation 4) spills.
constexpr std::size_t kMaxFilterTaps2d = 49;
constexpr std::size_t kMaxWindow       = 21;
// HSA grid limits in work-items.
constexpr std::size_t kMaxGridX  = 0x7fffffff;
constexpr std::size_t kMaxGridYZ = 0xffff;

// NCDHW launches one (n, k) output volume per blockIdx.y, which makes the filter wave-uniform so
// it is read through the scalar unit and does not occupy VGPRs; on gfx1151 that was 1.1x to 1.25x
// faster for 5x5x5. Blocks are then 64 threads, so small volumes leave few threads idle. NDHWC
// (consecutive threads are consecutive channels), N x K beyond the grid y limit and filter slices
// above 5x5 (whose weights spill SGPRs) use a 1D grid.
constexpr std::size_t kMaxFilterTaps2dUniformK = 25;

bool UniformK(const miopen::conv::ProblemDescription& problem)
{
    return !problem.IsLayoutNHWC() &&
           problem.GetBatchSize() * problem.GetOutChannels() <= kMaxGridYZ &&
           problem.GetWeightsHeight() * problem.GetWeightsWidth() <= kMaxFilterTaps2dUniformK;
}

std::size_t BlockSize(const miopen::conv::ProblemDescription& problem)
{
    return UniformK(problem) ? 64 : 256;
}

// Output depths per thread. Two depths share their input planes, which cuts DRAM reads; on gfx1151
// it was about 1.1x faster for NCDHW 5x5x5 and neutral for 3x3x3, but slower for bf16 NDHWC. It
// needs depth stride and dilation 1, and holds two filter slices, so it is limited to 5x5 slices.
constexpr std::size_t kMaxFilterTaps2dTd2 = 25;

std::size_t TileD(const miopen::conv::ProblemDescription& problem)
{
    return !problem.IsLayoutNHWC() && problem.GetKernelStrideD() == 1 &&
                   problem.GetDilationD() == 1 &&
                   problem.GetWeightsHeight() * problem.GetWeightsWidth() <= kMaxFilterTaps2dTd2
               ? 2
               : 1;
}

// Output columns per thread: a wider tile reuses more of each input row. On gfx1151 it was
// 1.05x to 1.26x faster for 5-wide filters and 1.1x to 1.3x slower for 3-wide ones, except with
// two output depths in fp32, where the narrower tile was faster.
std::size_t TileW(const miopen::conv::ProblemDescription& problem)
{
    if(TileD(problem) == 2 && problem.IsFp32())
        return 4;
    return problem.GetWeightsWidth() >= 5 ? 8 : 4;
}

std::size_t WindowW(const miopen::conv::ProblemDescription& problem)
{
    return (TileW(problem) - 1) * problem.GetKernelStrideW() +
           (problem.GetWeightsWidth() - 1) * problem.GetDilationW() + 1;
}

std::size_t WindowH(const miopen::conv::ProblemDescription& problem)
{
    return (kTileH - 1) * problem.GetKernelStrideH() +
           (problem.GetWeightsHeight() - 1) * problem.GetDilationH() + 1;
}

// Output tiles of one (n, k) output volume.
std::size_t TilesPerVolume(const miopen::conv::ProblemDescription& problem)
{
    const std::size_t tile_w = TileW(problem);
    const std::size_t tile_d = TileD(problem);
    return ((problem.GetOutDepth() + tile_d - 1) / tile_d) *
           ((problem.GetOutHeight() + kTileH - 1) / kTileH) *
           ((problem.GetOutWidth() + tile_w - 1) / tile_w);
}

std::size_t GridX(const miopen::conv::ProblemDescription& problem)
{
    const std::size_t block = BlockSize(problem);
    const std::size_t items =
        UniformK(problem)
            ? TilesPerVolume(problem)
            : problem.GetBatchSize() * problem.GetOutChannels() * TilesPerVolume(problem);
    return ((items + block - 1) / block) * block;
}

std::size_t GridY(const miopen::conv::ProblemDescription& problem)
{
    return UniformK(problem) ? problem.GetBatchSize() * problem.GetOutChannels() : 1;
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

    if(problem.GetWeightsHeight() * problem.GetWeightsWidth() > kMaxFilterTaps2d)
        return false;
    if(WindowW(problem) > kMaxWindow || WindowH(problem) > kMaxWindow)
        return false;

    return GridX(problem) <= kMaxGridX;
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

    kernel.l_wk = {BlockSize(problem), 1, 1};
    kernel.g_wk = {GridX(problem), GridY(problem), 1};

    kernel.comp_options =
        " -DDW_DTYPE=" + std::to_string(dtype) + " -DDW_NDHWC=" + std::to_string(ndhwc) +
        " -DDW_KZ=" + std::to_string(problem.GetWeightsDepth()) +
        " -DDW_KY=" + std::to_string(problem.GetWeightsHeight()) +
        " -DDW_KX=" + std::to_string(problem.GetWeightsWidth()) +
        " -DDW_SH=" + std::to_string(problem.GetKernelStrideH()) +
        " -DDW_SW=" + std::to_string(problem.GetKernelStrideW()) +
        " -DDW_DH=" + std::to_string(problem.GetDilationH()) +
        " -DDW_DW=" + std::to_string(problem.GetDilationW()) +
        " -DDW_TD=" + std::to_string(TileD(problem)) + " -DDW_TH=" + std::to_string(kTileH) +
        " -DDW_TW=" + std::to_string(TileW(problem)) +
        " -DDW_UK=" + std::to_string(UniformK(problem) ? 1 : 0);
    result.construction_params.push_back(kernel);

    const int n   = static_cast<int>(problem.GetBatchSize());
    const int c   = static_cast<int>(problem.GetInChannels());
    const int di  = static_cast<int>(problem.GetInDepth());
    const int hi  = static_cast<int>(problem.GetInHeight());
    const int wi  = static_cast<int>(problem.GetInWidth());
    const int k   = static_cast<int>(problem.GetOutChannels());
    const int d_o = static_cast<int>(problem.GetOutDepth());
    const int ho  = static_cast<int>(problem.GetOutHeight());
    const int wo  = static_cast<int>(problem.GetOutWidth());
    const int sd  = problem.GetKernelStrideD();
    const int pd  = problem.GetPadD();
    const int ph  = problem.GetPadH();
    const int pw  = problem.GetPadW();
    const int dd  = problem.GetDilationD();

    // The grid above and these arguments both come from the problem; the binary itself only
    // depends on the compile options (data type, layout, filter, H / W stride and dilation,
    // output tile).
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
                             d_o,
                             ho,
                             wo,
                             sd,
                             pd,
                             ph,
                             pw,
                             dd);
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
