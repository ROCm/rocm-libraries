// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/config.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/stringutils.hpp>

#include <string>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_HIP_DEPTHWISE_1D_BWD)

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Must match MIOpenConvDepthwiseBwd1dHip.cpp.
constexpr std::size_t kBlockSize = 256;

// The filter is held in registers (scalar registers, being wave-uniform). Compiled for every RDNA
// target (gfx1010 to gfx1013, gfx1030 to gfx1036, gfx1100 to gfx1103, gfx1150 to gfx1154, gfx1170
// to gfx1172, gfx1200, gfx1201), filter widths up to 64 at strides 1 to 8 need at most 115 VGPRs
// and no scratch in fp32, fp16 and bf16.
constexpr int kMaxFilterWidth = 64;
constexpr int kMaxStride      = 8;

// HSA grid limits in work-items.
constexpr std::size_t kMaxGridX  = 0x7fffffff;
constexpr std::size_t kMaxGridYZ = 0xffff;

// Output phases (q values) per lane. On gfx1151, one was fastest in fp32; in fp16 / bf16 two,
// with the lane's outputs written by one vector store, was up to 1.3x faster.
int TileQ(const miopen::conv::ProblemDescription& problem) { return problem.IsFp32() ? 1 : 2; }

std::size_t GridX(const miopen::conv::ProblemDescription& problem)
{
    // dx = Out, dy = In for backward data.
    const std::size_t stride = problem.GetKernelStrideW();
    const std::size_t q      = (problem.GetOutWidth() + problem.GetPadW() + stride - 1) / stride;
    const std::size_t lanes  = (q + TileQ(problem) - 1) / TileQ(problem);
    return ((lanes + kBlockSize - 1) / kBlockSize) * kBlockSize;
}

} // namespace

bool ConvHipDirectDepthwiseBwd1d::IsApplicable(
    const ExecutionContext& ctx, const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_HIP_DEPTHWISE_1D_BWD))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    // RDNA only. There, depthwise backward data has no other direct solver: the CK
    // ConvDepthwiseBwdData2D needs stride 1, fp16 / bf16 and a CK library, which gfx101X /
    // gfx103X lack, so stride-2 upsampling (depthwise ConvTranspose1d) falls to GemmBwdRest.
    // Family prefixes rather than a bare "gfx12", which would also match gfx125X.
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
    if(!problem.IsFp32() && !problem.IsFp16() && !problem.IsBfp16())
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

    // Depthwise with one channel per group on both sides.
    const std::size_t groups = problem.GetGroupCount();
    if(groups < 2 || problem.GetInChannels() != groups || problem.GetOutChannels() != groups)
        return false;

    if(problem.GetDilationW() != 1)
        return false;
    const int filter_w = static_cast<int>(problem.GetWeightsWidth());
    const int stride_w = problem.GetKernelStrideW();
    if(filter_w > kMaxFilterWidth || stride_w > kMaxStride || problem.GetPadW() < 0)
        return false;

    return groups <= kMaxGridYZ && problem.GetBatchSize() <= kMaxGridYZ &&
           GridX(problem) <= kMaxGridX;
}

ConvSolution
ConvHipDirectDepthwiseBwd1d::GetSolution(const ExecutionContext&,
                                         const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;
    kernel.kernel_file = "MIOpenConvDepthwiseBwd1dHip.cpp";
    kernel.kernel_name = "MIOpenConvDepthwiseBwd1dHip";

    const int dtype = problem.IsFp32() ? 0 : (problem.IsFp16() ? 1 : 2);

    kernel.l_wk = {kBlockSize, 1, 1};
    kernel.g_wk = {GridX(problem), problem.GetGroupCount(), problem.GetBatchSize()};

    kernel.comp_options = " -DDW_DTYPE=" + std::to_string(dtype) +
                          " -DDW_K=" + std::to_string(problem.GetWeightsWidth()) +
                          " -DDW_S=" + std::to_string(problem.GetKernelStrideW()) +
                          " -DDW_TW=" + std::to_string(TileQ(problem));
    result.construction_params.push_back(kernel);

    const int c      = static_cast<int>(problem.GetGroupCount());
    const int len_dy = static_cast<int>(problem.GetInWidth());
    const int len_dx = static_cast<int>(problem.GetOutWidth());
    const int pad    = problem.GetPadW();

    // Backward data: tensors.in is dy, tensors.out is dx.
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& tensors =
                primitive_parameters.CastTo<miopen::conv::DataInvokeParams>().tensors;
            handle.Run(kern)(tensors.in, tensors.w, tensors.out, c, len_dy, len_dx, pad);
        };
    };
    result.solver_id = SolverDbId();
    return result;
}

#else

bool ConvHipDirectDepthwiseBwd1d::IsApplicable(const ExecutionContext&,
                                               const miopen::conv::ProblemDescription&) const
{
    return false;
}

ConvSolution ConvHipDirectDepthwiseBwd1d::GetSolution(const ExecutionContext&,
                                                      const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
