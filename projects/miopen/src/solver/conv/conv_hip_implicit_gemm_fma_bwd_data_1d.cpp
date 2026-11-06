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

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Must match MIOpenConvTranspose1dFmaHip.cpp.
constexpr std::size_t kBlockSize = 256;
constexpr std::size_t kWaves     = 8;
constexpr std::size_t kLanes     = 32;

// q per lane (CT_QR). Two was the fastest or within a few percent of it in a sweep of 1, 2 and 4
// over audio decoder layers on gfx1151.
constexpr std::size_t kTileQ = 2;

// The weights of a wave's channels are read with scalar loads, one channel (K values) at a time.
// Compiled for every RDNA target, filter widths up to 64 at strides 1 to 8 need at most 124 VGPRs
// and no scratch.
constexpr int kMaxFilterWidth = 64;
constexpr int kMaxStride      = 8;

constexpr std::size_t kMaxGridYZ = 0xffff;
constexpr std::size_t kMaxGridX  = 0x7fffffff;

// Output channels per wave (CT_CO_W): 8 up to stride 4 and 4 above, so a lane holds at most 64
// accumulators (stride x channels x q); fewer when the layer has few output channels.
std::size_t TileCo(const miopen::conv::ProblemDescription& problem)
{
    std::size_t co_w = problem.GetKernelStrideW() <= 4 ? 8 : 4;
    while(co_w > 1 && kWaves * (co_w / 2) >= problem.GetOutChannels())
        co_w /= 2;
    return co_w;
}

// Backward data: In is dy (ConvTranspose1d input), Out is dx (its output).
std::size_t GridX(const miopen::conv::ProblemDescription& problem)
{
    const std::size_t s   = problem.GetKernelStrideW();
    const std::size_t q   = (problem.GetOutWidth() + problem.GetPadW() + s - 1) / s;
    const std::size_t q_t = kLanes * kTileQ;
    return ((q + q_t - 1) / q_t) * kBlockSize;
}

std::size_t GridY(const miopen::conv::ProblemDescription& problem)
{
    const std::size_t co_t = kWaves * TileCo(problem);
    return (problem.GetOutChannels() + co_t - 1) / co_t;
}

} // namespace

bool ConvHipImplicitGemmFmaBwdData1d::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FMA_BWD_DATA_1D))
        return false;
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

    return GridY(problem) <= kMaxGridYZ && problem.GetBatchSize() <= kMaxGridYZ &&
           GridX(problem) <= kMaxGridX;
}

ConvSolution
ConvHipImplicitGemmFmaBwdData1d::GetSolution(const ExecutionContext&,
                                             const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;

    KernelInfo kernel;
    kernel.kernel_file  = "MIOpenConvTranspose1dFmaHip.cpp";
    kernel.kernel_name  = "MIOpenConvTranspose1dFmaHip";
    kernel.comp_options = " -DCT_K=" + std::to_string(problem.GetWeightsWidth()) +
                          " -DCT_S=" + std::to_string(problem.GetKernelStrideW()) +
                          " -DCT_QR=" + std::to_string(kTileQ) +
                          " -DCT_CO_W=" + std::to_string(TileCo(problem));
    kernel.l_wk = {kBlockSize, 1, 1};
    kernel.g_wk = {GridX(problem), GridY(problem), problem.GetBatchSize()};
    result.construction_params.push_back(kernel);

    const int ci     = static_cast<int>(problem.GetInChannels());
    const int co     = static_cast<int>(problem.GetOutChannels());
    const int len_dy = static_cast<int>(problem.GetInWidth());
    const int len_dx = static_cast<int>(problem.GetOutWidth());
    const int pad    = problem.GetPadW();

    // Backward data: tensors.in is dy, tensors.out is dx.
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& tensors =
                primitive_parameters.CastTo<miopen::conv::DataInvokeParams>().tensors;
            handle.Run(kern)(tensors.in, tensors.w, tensors.out, ci, co, len_dy, len_dx, pad);
        };
    };
    result.solver_id = SolverDbId();
    return result;
}

#else

bool ConvHipImplicitGemmFmaBwdData1d::IsApplicable(const ExecutionContext&,
                                                   const miopen::conv::ProblemDescription&) const
{
    return false;
}

ConvSolution
ConvHipImplicitGemmFmaBwdData1d::GetSolution(const ExecutionContext&,
                                             const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
