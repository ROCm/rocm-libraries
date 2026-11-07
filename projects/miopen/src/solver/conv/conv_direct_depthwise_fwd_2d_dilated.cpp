// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_info.hpp>

#include <algorithm>
#include <numeric>
#include <string>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_DEPTHWISE_FWD_2D_DILATED)

namespace miopen {
namespace solver {
namespace conv {

// Dilated depthwise forward on the ConvDirectDepthwiseFwd2D kernel, built with
// LANE_ON_DILATION_GRID: a lane walks outputs a dilation step apart (at stride
// 1), so the inputs they read coincide and each further output row loads only
// the rows it gains. ConvDirectDepthwiseFwd2D walks adjacent outputs, which at
// stride 1 share no input row of a dilated filter. This variant has not been
// shown faster on every shape and target, so both stay applicable and Find
// picks between them.

namespace {

constexpr int kWorkgroupSize = 256;
constexpr int kRowChunk      = 8;

// Same channel floor and filter extent as ConvDirectDepthwiseFwd2D.
constexpr std::size_t kMinChannels = 32;
constexpr int kMaxFilterExtent     = 7;

// Float slots the kernel keeps live per lane, as in ConvDirectDepthwiseFwd2D:
// the window plus the taps.
constexpr int kMaxWindowRegs = 128;

// Outputs per lane. Four amortize the 16-bit to fp32 conversion of the window
// over more FMAs; in fp32 four were slower than two on gfx1151 on most of the
// shapes measured.
int OutPerLane(const miopen::conv::ProblemDescription& problem) { return problem.IsFp32() ? 2 : 4; }

// 1xK filters have no window rows to share between output rows; along W, with
// the lane's outputs a dilation step apart they were slower on gfx1151 than
// with adjacent outputs, so they keep adjacent outputs (PAIR_COLUMNS).
bool PairColumns(const miopen::conv::ProblemDescription& problem)
{
    return problem.GetWeightsHeight() == 1;
}

// Outputs of a lane are this many rows (columns) apart: kRowStep / kColStep in
// the kernel.
int LaneStep(int stride, int dilation) { return dilation / std::gcd(stride, dilation); }

// Mirror of the kernel's FILTER_H x kColSlots window plus its taps[][].
int WindowRegs(const miopen::conv::ProblemDescription& problem)
{
    const int filter_h   = static_cast<int>(problem.GetWeightsHeight());
    const int filter_w   = static_cast<int>(problem.GetWeightsWidth());
    const int stride_w   = static_cast<int>(problem.GetKernelStrideW());
    const int dilation_w = static_cast<int>(problem.GetDilationW());
    const int out        = OutPerLane(problem);
    const int col_slots =
        PairColumns(problem)
            ? out * filter_w
            : (out - 1) * (LaneStep(stride_w, dilation_w) * stride_w / dilation_w) + filter_w;
    return filter_h * col_slots + filter_h * filter_w;
}

} // namespace

bool ConvDirectDepthwiseFwd2DDilated::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_DEPTHWISE_FWD_2D_DILATED))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // Same targets as ConvDirectDepthwiseFwd2D: RDNA3, RDNA3.5 and RDNA4.
    const std::string dev_name = ctx.GetStream().GetDeviceName();
    if(!StartsWith(dev_name, "gfx110") && !StartsWith(dev_name, "gfx115") &&
       !StartsWith(dev_name, "gfx120"))
        return false;

    if(!problem.Is2d())
        return false;

    if(!problem.IsDirectionForward())
        return false;

    if(!problem.IsFp16() && !problem.IsBfp16() && !problem.IsFp32())
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    if(problem.IsTensorsCasted() || problem.HasMixedDataTypes())
        return false;

    if(problem.HasNonPackedTensors())
        return false;

    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;

    // Depthwise: exactly one channel per group.
    const auto g = static_cast<std::size_t>(problem.GetGroupCount());
    if(g < kMinChannels || problem.GetInChannels() != g || problem.GetOutChannels() != g)
        return false;

    const auto filter_h = static_cast<int>(problem.GetWeightsHeight());
    const auto filter_w = static_cast<int>(problem.GetWeightsWidth());
    if(filter_h > kMaxFilterExtent || filter_w > kMaxFilterExtent)
        return false;

    // Dilated along an axis the filter spans; otherwise this is the undilated
    // case ConvDirectDepthwiseFwd2D covers with the same code.
    const bool dilated = (filter_h > 1 && problem.GetDilationH() > 1) ||
                         (filter_w > 1 && problem.GetDilationW() > 1);
    if(!dilated)
        return false;

    return WindowRegs(problem) <= kMaxWindowRegs;
}

ConvSolution
ConvDirectDepthwiseFwd2DDilated::GetSolution(const ExecutionContext&,
                                             const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;
    kernel.kernel_file = "miopen_conv2d_depthwise_fwd.cpp";
    kernel.kernel_name = "miopen_conv2d_depthwise_fwd";

    const auto n     = static_cast<int>(problem.GetBatchSize());
    const auto c     = static_cast<int>(problem.GetInChannels());
    const auto out_h = static_cast<int>(problem.GetOutHeight());
    const auto out_w = static_cast<int>(problem.GetOutWidth());
    const int out    = OutPerLane(problem);
    const bool pairs = PairColumns(problem);

    // Same tiling as the kernel: kOutPerLane outputs kColStep apart by kRowChunk
    // rows kRowStep apart.
    const int col_step  = pairs ? 1 : LaneStep(problem.GetKernelStrideW(), problem.GetDilationW());
    const int row_step  = LaneStep(problem.GetKernelStrideH(), problem.GetDilationH());
    const int tiles_w   = (out_w + out * col_step - 1) / (out * col_step) * col_step;
    const int row_tiles = (out_h + kRowChunk * row_step - 1) / (kRowChunk * row_step) * row_step;

    const std::size_t tiles = static_cast<std::size_t>(tiles_w) * row_tiles * n * c;
    const std::size_t groups =
        std::min<std::size_t>((tiles + kWorkgroupSize - 1) / kWorkgroupSize, 2048);

    kernel.l_wk.push_back(kWorkgroupSize);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);
    kernel.g_wk.push_back(groups * kWorkgroupSize);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    std::string dtype = "float";
    if(problem.IsFp16())
        dtype = "__half";
    else if(problem.IsBfp16())
        dtype = "__hip_bfloat16";

    kernel.comp_options =
        std::string(" -DIO_DTYPE=") + dtype +
        " -DFILTER_H=" + std::to_string(problem.GetWeightsHeight()) +
        " -DFILTER_W=" + std::to_string(problem.GetWeightsWidth()) +
        " -DPAD_H=" + std::to_string(problem.GetPadH()) +
        " -DPAD_W=" + std::to_string(problem.GetPadW()) +
        " -DSTRIDE_H=" + std::to_string(problem.GetKernelStrideH()) +
        " -DSTRIDE_W=" + std::to_string(problem.GetKernelStrideW()) +
        " -DDILATION_H=" + std::to_string(problem.GetDilationH()) +
        " -DDILATION_W=" + std::to_string(problem.GetDilationW()) +
        " -DOUT_PER_LANE=" + std::to_string(out) + " -DROW_CHUNK=" + std::to_string(kRowChunk) +
        " -DWORKGROUP_SIZE=" + std::to_string(kWorkgroupSize) + " -DLANE_ON_DILATION_GRID=1" +
        " -DPAIR_COLUMNS=" + std::to_string(pairs ? 1 : 0);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [kern](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& data_ctx = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;
            const auto& in_len   = tensors.inDesc.GetLengths();
            const auto& out_len  = tensors.outDesc.GetLengths();
            const int nb         = static_cast<int>(in_len[0]);
            const int ic         = static_cast<int>(in_len[1]);
            const int ih         = static_cast<int>(in_len[2]);
            const int iw         = static_cast<int>(in_len[3]);
            const int oh         = static_cast<int>(out_len[2]);
            const int ow         = static_cast<int>(out_len[3]);
            handle.Run(kern)(tensors.in, tensors.w, tensors.out, nb, ic, ih, iw, oh, ow);
        };
    };
    result.construction_params.push_back(kernel);
    result.solver_id = SolverDbId();
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
