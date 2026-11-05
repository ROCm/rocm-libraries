// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/stringutils.hpp>

#include <algorithm>
#include <string>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WMMA_BWD_DATA_1D)

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Must match MIOpenConvTranspose1dWmmaHip.cpp.
constexpr std::size_t kBlockSize = 256;
constexpr std::size_t kCoTile    = 64;
constexpr std::size_t kQTile     = 64;
constexpr std::size_t kMaxLds    = 65536;

// Input channels per stage (CT_CI_T). 32 halves the stage count and was 1.03x to 1.7x faster than
// 16 on gfx1151 where its tiles fit in LDS; filters with more taps per phase use 16.
constexpr std::size_t kCiTileLarge = 32;
constexpr std::size_t kCiTileSmall = 16;

// Filter width and stride are bounded by LDS (LdsBytes below), which leaves stride <= 7 and
// stride x taps per phase <= 20. Compiled for gfx1100 to gfx1103 and gfx1150 to gfx1153 (gfx11
// layout) and for gfx1200 and gfx1201 (gfx12 layout), every such filter needs at most 255 VGPRs
// and no scratch in fp16 and bf16, with either stage size.
constexpr int kMaxFilterWidth = 64;
constexpr int kMaxStride      = 8;

constexpr std::size_t kMaxGridYZ = 0xffff;
constexpr std::size_t kMaxGridX  = 0x7fffffff;

std::size_t RoundUp(std::size_t v, std::size_t m) { return (v + m - 1) / m * m; }

// gfx12 WMMA operand layout (8 elements per lane, no replication) instead of the gfx11 one. Chosen
// from the device name rather than from the __GFX11__ / __GFX12__ macros in the kernel.
bool WmmaGfx12Layout(const ExecutionContext& ctx)
{
    return StartsWith(ctx.GetStream().GetDeviceName(), "gfx120");
}

std::size_t TapsPerPhase(const miopen::conv::ProblemDescription& problem)
{
    const std::size_t k = problem.GetWeightsWidth();
    const std::size_t s = problem.GetKernelStrideW();
    return (k + s - 1) / s;
}

// dy window [q][ci] plus the larger of the weight tile [r][m][co][ci] and the dx tile [co][S q];
// LDS rows hold ci_tile channels plus 8 of padding.
std::size_t LdsBytes(const miopen::conv::ProblemDescription& problem, std::size_t ci_tile)
{
    const std::size_t row = ci_tile + 8;
    const std::size_t s   = problem.GetKernelStrideW();
    const std::size_t t   = TapsPerPhase(problem);
    const std::size_t xs  = (kQTile + t - 1) * row;
    const std::size_t ws  = s * t * kCoTile * row;
    const std::size_t out = kCoTile * s * kQTile;
    return 2 * (xs + std::max(ws, out));
}

std::size_t CiTile(const miopen::conv::ProblemDescription& problem)
{
    return LdsBytes(problem, kCiTileLarge) <= kMaxLds ? kCiTileLarge : kCiTileSmall;
}

// Backward data: In is dy (ConvTranspose1d input), Out is dx (its output).
std::size_t CiPad(const miopen::conv::ProblemDescription& problem)
{
    return RoundUp(problem.GetInChannels(), CiTile(problem));
}

std::size_t CoPad(const miopen::conv::ProblemDescription& problem)
{
    return RoundUp(problem.GetOutChannels(), kCoTile);
}

std::size_t WeightsWorkspaceElems(const miopen::conv::ProblemDescription& problem)
{
    return problem.GetKernelStrideW() * TapsPerPhase(problem) * CoPad(problem) * CiPad(problem);
}

std::size_t GridX(const miopen::conv::ProblemDescription& problem)
{
    const std::size_t s = problem.GetKernelStrideW();
    const std::size_t q = (problem.GetOutWidth() + problem.GetPadW() + s - 1) / s;
    return ((q + kQTile - 1) / kQTile) * kBlockSize;
}

} // namespace

bool ConvHipImplicitGemmWmmaBwdData1d::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WMMA_BWD_DATA_1D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // WMMA targets whose operand layout the kernel implements: gfx110X / gfx115X (gfx11 layout)
    // and gfx120X (gfx12 layout). gfx101X / gfx103X have no WMMA.
    const std::string dev_name = ctx.GetStream().GetDeviceName();
    if(!StartsWith(dev_name, "gfx110") && !StartsWith(dev_name, "gfx115") && !WmmaGfx12Layout(ctx))
        return false;

    if(!problem.Is2d())
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    if(problem.GetConv().paddingMode != miopenPaddingDefault)
        return false;
    if(problem.GetAlphaBetaCase() != DEFAULT)
        return false;
    // WMMA takes 16-bit inputs; fp32 has no WMMA path.
    if(!problem.IsFp16() && !problem.IsBfp16())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;

    // 1D: H == 1 everywhere, a 1xK filter, no H padding.
    if(problem.GetInHeight() != 1 || problem.GetOutHeight() != 1 ||
       problem.GetWeightsHeight() != 1 || problem.GetPadH() != 0)
        return false;
    if(problem.GetGroupCount() != 1)
        return false;
    if(problem.GetDilationW() != 1)
        return false;

    const int filter_w = static_cast<int>(problem.GetWeightsWidth());
    const int stride_w = problem.GetKernelStrideW();
    if(filter_w > kMaxFilterWidth || stride_w > kMaxStride || problem.GetPadW() < 0)
        return false;
    if(LdsBytes(problem, kCiTileSmall) > kMaxLds)
        return false;

    return CoPad(problem) / kCoTile <= kMaxGridYZ && problem.GetBatchSize() <= kMaxGridYZ &&
           GridX(problem) <= kMaxGridX;
}

std::size_t ConvHipImplicitGemmWmmaBwdData1d::GetWorkspaceSize(
    const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
{
    return WeightsWorkspaceElems(problem) * 2;
}

ConvSolution
ConvHipImplicitGemmWmmaBwdData1d::GetSolution(const ExecutionContext& ctx,
                                              const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;

    const std::string options = " -DCT_DTYPE=" + std::to_string(problem.IsFp16() ? 1 : 2) +
                                " -DCT_K=" + std::to_string(problem.GetWeightsWidth()) +
                                " -DCT_S=" + std::to_string(problem.GetKernelStrideW()) +
                                " -DCT_CI_T=" + std::to_string(CiTile(problem)) +
                                " -DCT_WMMA_GFX12=" + std::to_string(WmmaGfx12Layout(ctx) ? 1 : 0);

    const std::size_t wt_elems = WeightsWorkspaceElems(problem);

    KernelInfo weights;
    weights.kernel_file  = "MIOpenConvTranspose1dWmmaHip.cpp";
    weights.kernel_name  = "MIOpenConvTranspose1dWmmaHipWeights";
    weights.comp_options = options;
    weights.l_wk         = {kBlockSize, 1, 1};
    weights.g_wk         = {
        std::min<std::size_t>((wt_elems + kBlockSize - 1) / kBlockSize, 4096) * kBlockSize, 1, 1};
    result.construction_params.push_back(weights);

    KernelInfo main;
    main.kernel_file  = "MIOpenConvTranspose1dWmmaHip.cpp";
    main.kernel_name  = "MIOpenConvTranspose1dWmmaHip";
    main.comp_options = options;
    main.l_wk         = {kBlockSize, 1, 1};
    main.g_wk         = {GridX(problem), CoPad(problem) / kCoTile, problem.GetBatchSize()};
    result.construction_params.push_back(main);

    const int ci               = static_cast<int>(problem.GetInChannels());
    const int co               = static_cast<int>(problem.GetOutChannels());
    const int ci_pad           = static_cast<int>(CiPad(problem));
    const int co_pad           = static_cast<int>(CoPad(problem));
    const int len_dy           = static_cast<int>(problem.GetInWidth());
    const int len_dx           = static_cast<int>(problem.GetOutWidth());
    const int pad              = problem.GetPadW();
    const std::size_t ws_bytes = WeightsWorkspaceElems(problem) * 2;

    // Backward data: tensors.in is dy, tensors.out is dx; the workspace holds the arranged weights.
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto weights_kern = kernels[0];
        const auto main_kern    = kernels[1];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& params  = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors = params.tensors;
            if(params.workSpace == nullptr || params.workSpaceSize < ws_bytes)
                MIOPEN_THROW(miopenStatusBadParm, "Not enough workspace");
            // Find times the invoker through the handle, which keeps the last kernel's time only:
            // report the sum of both kernels.
            float elapsed = 0.0f;
            handle.Run(weights_kern)(tensors.w, params.workSpace, ci, co, ci_pad, co_pad);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();
            handle.Run(main_kern)(tensors.in,
                                  params.workSpace,
                                  tensors.out,
                                  ci,
                                  co,
                                  ci_pad,
                                  co_pad,
                                  len_dy,
                                  len_dx,
                                  pad);
            if(handle.IsProfilingEnabled())
            {
                elapsed += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };
    result.workspace_sz = ws_bytes;
    result.solver_id    = SolverDbId();
    return result;
}

#else

bool ConvHipImplicitGemmWmmaBwdData1d::IsApplicable(const ExecutionContext&,
                                                    const miopen::conv::ProblemDescription&) const
{
    return false;
}

std::size_t
ConvHipImplicitGemmWmmaBwdData1d::GetWorkspaceSize(const ExecutionContext&,
                                                   const miopen::conv::ProblemDescription&) const
{
    return 0;
}

ConvSolution
ConvHipImplicitGemmWmmaBwdData1d::GetSolution(const ExecutionContext&,
                                              const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
