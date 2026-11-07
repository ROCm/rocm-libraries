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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_BWD_DATA_1D)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_BWD_DATA_1D_NO_WS)

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Must match MIOpenConvTranspose1dFmaHip.cpp.
constexpr std::size_t kBlockSize = 256;
constexpr std::size_t kWaves     = 8;
constexpr std::size_t kLanes     = 32;

// The weights of a wave's channels are read with scalar loads, one channel (K values) at a time.
// Compiled for every RDNA target, filter widths up to 64 at strides 1 to 8 need at most 126 VGPRs
// and no scratch in fp32, fp16 and bf16, and at most 158 VGPRs and no scratch with the 16-bit
// weights read in place (the tiles below).
constexpr int kMaxFilterWidth = 64;
constexpr int kMaxStride      = 8;

constexpr std::size_t kMaxGridYZ = 0xffff;
constexpr std::size_t kMaxGridX  = 0x7fffffff;

struct Tiles
{
    std::size_t q;    // q per lane (CT_QR)
    std::size_t co_w; // output channels per wave (CT_CO_W)
};

// Two q per lane was the fastest or within a few percent of it in a sweep of 1, 2 and 4 over audio
// decoder layers on gfx1151, with 8 channels per wave up to stride 4 and 4 above (at most 64
// accumulators per lane). With 16-bit weights read in place and an odd filter width, each weight is
// unpacked from two words, and 4 q x 2 channels, which uses each weight twice as often, was the
// fastest. A wave loads the 32-bit weight words of all its channels at once, so with 16-bit weights
// in place channels x words per channel stays at or below 128 (8 channels at K = 64 spill to
// scratch in bf16). Fewer channels when the layer has few output channels.
Tiles GetTiles(const miopen::conv::ProblemDescription& problem, bool in_place_16bit)
{
    Tiles tiles{2, problem.GetKernelStrideW() <= 4 ? std::size_t{8} : std::size_t{4}};
    if(in_place_16bit)
    {
        const std::size_t k = problem.GetWeightsWidth();
        if(k % 2 != 0)
            tiles = {4, 2};
        while(tiles.co_w > 1 && tiles.co_w * ((k + 1) / 2) > 128)
            tiles.co_w /= 2;
    }
    while(tiles.co_w > 1 && kWaves * (tiles.co_w / 2) >= problem.GetOutChannels())
        tiles.co_w /= 2;
    return tiles;
}

// Backward data: In is dy (ConvTranspose1d input), Out is dx (its output).
std::size_t GridX(const miopen::conv::ProblemDescription& problem, const Tiles& tiles)
{
    const std::size_t s   = problem.GetKernelStrideW();
    const std::size_t q   = (problem.GetOutWidth() + problem.GetPadW() + s - 1) / s;
    const std::size_t q_t = kLanes * tiles.q;
    return ((q + q_t - 1) / q_t) * kBlockSize;
}

std::size_t GridY(const miopen::conv::ProblemDescription& problem, const Tiles& tiles)
{
    const std::size_t co_t = kWaves * tiles.co_w;
    return (problem.GetOutChannels() + co_t - 1) / co_t;
}

std::size_t WeightsElems(const miopen::conv::ProblemDescription& problem)
{
    return problem.GetInChannels() * problem.GetOutChannels() * problem.GetWeightsWidth();
}

// Everything but the data type and the solver's own env switch.
bool IsApplicableCommon(const ExecutionContext& ctx,
                        const miopen::conv::ProblemDescription& problem,
                        bool in_place_16bit)
{
    if(!ctx.use_hip_kernels)
        return false;

    // RDNA. Family prefixes rather than a bare "gfx12", which would also match gfx125X.
    const std::string dev_name = ctx.GetStream().GetDeviceName();
    if(!StartsWith(dev_name, "gfx101") && !StartsWith(dev_name, "gfx103") &&
       !StartsWith(dev_name, "gfx110") && !StartsWith(dev_name, "gfx115") &&
       !StartsWith(dev_name, "gfx117") && !StartsWith(dev_name, "gfx120"))
        return false;

    if(!problem.Is2d())
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    if(problem.GetConv().paddingMode != miopenPaddingDefault)
        return false;
    if(problem.GetAlphaBetaCase() != DEFAULT)
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

    const auto tiles = GetTiles(problem, in_place_16bit);
    return GridY(problem, tiles) <= kMaxGridYZ && problem.GetBatchSize() <= kMaxGridYZ &&
           GridX(problem, tiles) <= kMaxGridX;
}

// in_place_16bit: fp16 / bf16 weights read in place (no workspace) instead of converted to fp32
// into the workspace first. fp32 weights are always read in place.
ConvSolution MakeSolution(const miopen::conv::ProblemDescription& problem,
                          bool in_place_16bit,
                          const std::string& solver_id)
{
    ConvSolution result;

    const int dtype           = problem.IsFp32() ? 0 : (problem.IsFp16() ? 1 : 2);
    const auto tiles          = GetTiles(problem, in_place_16bit);
    const std::string options = " -DCT_DTYPE=" + std::to_string(dtype) +
                                " -DCT_K=" + std::to_string(problem.GetWeightsWidth()) +
                                " -DCT_S=" + std::to_string(problem.GetKernelStrideW()) +
                                " -DCT_QR=" + std::to_string(tiles.q) +
                                " -DCT_CO_W=" + std::to_string(tiles.co_w) +
                                " -DCT_WS=" + std::to_string(in_place_16bit ? 0 : 1);

    const bool convert_weights = !problem.IsFp32() && !in_place_16bit;
    const int weights_n        = static_cast<int>(WeightsElems(problem));
    if(convert_weights)
    {
        KernelInfo weights;
        weights.kernel_file  = "MIOpenConvTranspose1dFmaHip.cpp";
        weights.kernel_name  = "MIOpenConvTranspose1dFmaHipWeights";
        weights.comp_options = options;
        weights.l_wk         = {kBlockSize, 1, 1};
        weights.g_wk = {std::min<std::size_t>((weights_n + kBlockSize - 1) / kBlockSize, 4096) *
                            kBlockSize,
                        1,
                        1};
        result.construction_params.push_back(weights);
    }

    KernelInfo kernel;
    kernel.kernel_file  = "MIOpenConvTranspose1dFmaHip.cpp";
    kernel.kernel_name  = "MIOpenConvTranspose1dFmaHip";
    kernel.comp_options = options;
    kernel.l_wk         = {kBlockSize, 1, 1};
    kernel.g_wk         = {GridX(problem, tiles), GridY(problem, tiles), problem.GetBatchSize()};
    result.construction_params.push_back(kernel);

    const int ci               = static_cast<int>(problem.GetInChannels());
    const int co               = static_cast<int>(problem.GetOutChannels());
    const int len_dy           = static_cast<int>(problem.GetInWidth());
    const int len_dx           = static_cast<int>(problem.GetOutWidth());
    const int pad              = problem.GetPadW();
    const std::size_t ws_bytes = convert_weights ? WeightsElems(problem) * sizeof(float) : 0;

    // Backward data: tensors.in is dy, tensors.out is dx.
    if(!convert_weights)
    {
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            const auto kern = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& tensors =
                    primitive_parameters.CastTo<miopen::conv::DataInvokeParams>().tensors;
                handle.Run(kern)(tensors.in, tensors.w, tensors.out, ci, co, len_dy, len_dx, pad);
            };
        };
    }
    else
    {
        // fp16 / bf16: the weights are converted to fp32 into the workspace first.
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            const auto weights_kern = kernels[0];
            const auto main_kern    = kernels[1];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& params  = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
                const auto& tensors = params.tensors;
                if(params.workSpace == nullptr || params.workSpaceSize < ws_bytes)
                    MIOPEN_THROW(miopenStatusBadParm, "Not enough workspace");
                // Find times the invoker through the handle, which keeps the last kernel's time
                // only: report the sum of both kernels.
                float elapsed = 0.0f;
                handle.Run(weights_kern)(tensors.w, params.workSpace, weights_n);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();
                handle.Run(main_kern)(
                    tensors.in, params.workSpace, tensors.out, ci, co, len_dy, len_dx, pad);
                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    }
    result.workspace_sz = ws_bytes;
    result.solver_id    = solver_id;
    return result;
}

} // namespace

bool ConvHipImplicitGemmFmaBwdData1d::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_BWD_DATA_1D))
        return false;
    // Accumulation is fp32 for every type; fp16 / bf16 weights go through the workspace as fp32.
    if(!problem.IsFp32() && !problem.IsFp16() && !problem.IsBfp16())
        return false;
    return IsApplicableCommon(ctx, problem, false);
}

std::size_t ConvHipImplicitGemmFmaBwdData1d::GetWorkspaceSize(
    const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
{
    return problem.IsFp32() ? 0 : WeightsElems(problem) * sizeof(float);
}

ConvSolution
ConvHipImplicitGemmFmaBwdData1d::GetSolution(const ExecutionContext&,
                                             const miopen::conv::ProblemDescription& problem) const
{
    return MakeSolution(problem, false, SolverDbId());
}

bool ConvHipImplicitGemmFmaBwdData1dNoWorkspace::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_BWD_DATA_1D_NO_WS))
        return false;
    // fp32 needs no workspace in ConvHipImplicitGemmFmaBwdData1d already.
    if(!problem.IsFp16() && !problem.IsBfp16())
        return false;
    return IsApplicableCommon(ctx, problem, true);
}

ConvSolution ConvHipImplicitGemmFmaBwdData1dNoWorkspace::GetSolution(
    const ExecutionContext&, const miopen::conv::ProblemDescription& problem) const
{
    return MakeSolution(problem, true, SolverDbId());
}

#else

bool ConvHipImplicitGemmFmaBwdData1d::IsApplicable(const ExecutionContext&,
                                                   const miopen::conv::ProblemDescription&) const
{
    return false;
}

std::size_t
ConvHipImplicitGemmFmaBwdData1d::GetWorkspaceSize(const ExecutionContext&,
                                                  const miopen::conv::ProblemDescription&) const
{
    return 0;
}

ConvSolution
ConvHipImplicitGemmFmaBwdData1d::GetSolution(const ExecutionContext&,
                                             const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

bool ConvHipImplicitGemmFmaBwdData1dNoWorkspace::IsApplicable(
    const ExecutionContext&, const miopen::conv::ProblemDescription&) const
{
    return false;
}

ConvSolution ConvHipImplicitGemmFmaBwdData1dNoWorkspace::GetSolution(
    const ExecutionContext&, const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
