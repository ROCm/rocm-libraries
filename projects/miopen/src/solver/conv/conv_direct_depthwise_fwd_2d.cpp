// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_info.hpp>

#include <algorithm>
#include <string>

// Not MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D: that name is already owned by
// ConvDepthwiseFwd2D in conv_ck_grouped_conv_fwd.cpp, and the declaration
// macro yields one shared instance across translation units, so reusing it
// would make a single flag gate two unrelated solvers.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_DEPTHWISE_FWD_2D)

namespace miopen {
namespace solver {
namespace conv {

namespace {

constexpr int kWorkgroupSize = 256;
constexpr int kOutPerLane    = 2;

// Output rows one lane walks. Fixed rather than derived from the output height:
// the kernel already exits a short final block early, and keeping it constant
// keeps the output height out of the kernel's compile options, which is what
// IsDynamic() promises.
constexpr int kRowChunk = 8;

// Filters wider than this hold too many taps and window columns in registers to
// stay ahead of the grouped implicit-GEMM path.
constexpr int kMaxFilterExtent = 7;

// The grouped implicit-GEMM tile only becomes degenerate once the group count is
// large; below that the existing path still wins.
constexpr std::size_t kMinChannels = 32;

// Float slots the kernel holds live per lane: the sliding window plus the taps.
// The kernel keeps all of it in registers under __launch_bounds__, so the cap
// has to leave room within the VGPR file or the window spills to scratch and
// the solver loses the very margin it is claiming.
constexpr int kMaxWindowRegs = 128;

// Mirror of the kernel's kWinSlots * kInSpanW window plus its taps[][]. Filter
// extent alone does not bound this: both dilation and the horizontal stride
// stretch the window independently.
constexpr int WindowRegs(int filter_h, int filter_w, int stride_w, int dilation_h, int dilation_w)
{
    const int win_slots = (filter_h - 1) * dilation_h + 1;
    const int in_span_w = (kOutPerLane - 1) * stride_w + (filter_w - 1) * dilation_w + 1;
    return win_slots * in_span_w + filter_h * filter_w;
}

} // namespace

bool ConvDirectDepthwiseFwd2D::IsApplicable(const ExecutionContext& ctx,
                                            const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_DEPTHWISE_FWD_2D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // RDNA3, RDNA3.5 and RDNA4 only. The CDNA parts, gfx125X among them, reach
    // this shape class through the XDLOPS grouped path, which is tuned for them.
    // Hence the family prefixes rather than a bare "gfx12", which would also
    // match gfx125X.
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

    // Dilation and horizontal stride widen the per-lane register window without
    // touching the filter extent, so bound the window itself.
    if(WindowRegs(filter_h,
                  filter_w,
                  static_cast<int>(problem.GetKernelStrideW()),
                  static_cast<int>(problem.GetDilationH()),
                  static_cast<int>(problem.GetDilationW())) > kMaxWindowRegs)
        return false;

    // No constraint tying stride to dilation: when the vertical stride is not a
    // whole number of dilation steps no row survives the slide, and the kernel
    // simply reloads the whole window for each output row.

    return true;
}

float ConvDirectDepthwiseFwd2D::GetWti(const ExecutionContext& ctx,
                                       const miopen::conv::ProblemDescription& problem) const
{
    const std::string dev_name = ctx.GetStream().GetDeviceName();
    const bool measured_family =
        StartsWith(dev_name, "gfx1151") && problem.IsFp16() && problem.GetWeightsHeight() == 3 &&
        problem.GetWeightsWidth() == 3 && problem.GetPadH() == 1 && problem.GetPadW() == 1 &&
        problem.GetKernelStrideH() == 1 && problem.GetKernelStrideW() == 1 &&
        problem.GetDilationH() == 1 && problem.GetDilationW() == 1 &&
        problem.GetOutHeight() >= 16 && problem.GetOutWidth() >= 16;

    if(!measured_family)
        return wti_approximate_worst;

    // Twelve sampled gfx1151 shapes spanning N=1..8, C=32..384, and
    // H/W=16..80 (including 17x33) were 2.69x to 35.07x faster than
    // GemmFwdRest. This narrowly clears its 0.342 estimate for N=1.
    return 0.35f;
}

ConvSolution
ConvDirectDepthwiseFwd2D::GetSolution(const ExecutionContext&,
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

    const int tiles_w   = (out_w + kOutPerLane - 1) / kOutPerLane;
    const int row_tiles = (out_h + kRowChunk - 1) / kRowChunk;

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

    kernel.comp_options = std::string(" -DIO_DTYPE=") + dtype +
                          " -DFILTER_H=" + std::to_string(problem.GetWeightsHeight()) +
                          " -DFILTER_W=" + std::to_string(problem.GetWeightsWidth()) +
                          " -DPAD_H=" + std::to_string(problem.GetPadH()) +
                          " -DPAD_W=" + std::to_string(problem.GetPadW()) +
                          " -DSTRIDE_H=" + std::to_string(problem.GetKernelStrideH()) +
                          " -DSTRIDE_W=" + std::to_string(problem.GetKernelStrideW()) +
                          " -DDILATION_H=" + std::to_string(problem.GetDilationH()) +
                          " -DDILATION_W=" + std::to_string(problem.GetDilationW()) +
                          " -DOUT_PER_LANE=" + std::to_string(kOutPerLane) +
                          " -DROW_CHUNK=" + std::to_string(kRowChunk) +
                          " -DWORKGROUP_SIZE=" + std::to_string(kWorkgroupSize);

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
