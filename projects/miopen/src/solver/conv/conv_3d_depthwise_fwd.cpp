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

#include <string>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_3D)

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP

namespace {

// Fused shape for one case only
//    B,C_in,C_out,D,H,W = 1,512,512,61,45,80; kernel (3,5,5); pad (0,2,2);
//     groups = C_out
constexpr std::size_t kCase3Batch    = 1;
constexpr std::size_t kCase3Channels = 512;
constexpr std::size_t kCase3InD      = 61;
constexpr std::size_t kCase3InH      = 45;
constexpr std::size_t kCase3InW      = 80;
constexpr std::size_t kCase3OutD     = 59;
constexpr std::size_t kCase3OutH     = 45;
constexpr std::size_t kCase3OutW     = 80;
constexpr int kCase3Kd               = 3;
constexpr int kCase3Kh               = 5;
constexpr int kCase3Kw               = 5;
constexpr int kCase3PadD             = 0;
constexpr int kCase3PadH             = 2;
constexpr int kCase3PadW             = 2;
constexpr int kCase3Stride           = 1;
constexpr int kCase3Dilation         = 1;
constexpr std::size_t kCase3Group    = 512;

} // namespace

bool ConvDepthwiseFwd3D::IsApplicable(const ExecutionContext& ctx,
                                      const miopen::conv::ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_3D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    const std::string dev_name = ctx.GetStream().GetDeviceName();
    if(dev_name != "gfx942" && dev_name != "gfx950")
        return false;

    if(!problem.Is3d())
        return false;

    if(!problem.IsDirectionForward())
        return false;

    if(!problem.IsBfp16() && !problem.IsFp16())
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    if(problem.IsTensorsCasted())
        return false;

    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;

    const auto g = static_cast<std::size_t>(problem.GetGroupCount());
    if(g == 0 || problem.GetInChannels() != g || problem.GetOutChannels() != g)
        return false;

    return problem.GetBatchSize() == kCase3Batch && problem.GetInChannels() == kCase3Channels &&
           problem.GetOutChannels() == kCase3Channels && problem.GetInDepth() == kCase3InD &&
           problem.GetInHeight() == kCase3InH && problem.GetInWidth() == kCase3InW &&
           problem.GetOutDepth() == kCase3OutD && problem.GetOutHeight() == kCase3OutH &&
           problem.GetOutWidth() == kCase3OutW &&
           static_cast<int>(problem.GetWeightsDepth()) == kCase3Kd &&
           static_cast<int>(problem.GetWeightsHeight()) == kCase3Kh &&
           static_cast<int>(problem.GetWeightsWidth()) == kCase3Kw &&
           problem.GetPadD() == kCase3PadD && problem.GetPadH() == kCase3PadH &&
           problem.GetPadW() == kCase3PadW && problem.GetKernelStrideD() == kCase3Stride &&
           problem.GetKernelStrideH() == kCase3Stride &&
           problem.GetKernelStrideW() == kCase3Stride && problem.GetDilationD() == kCase3Dilation &&
           problem.GetDilationH() == kCase3Dilation && problem.GetDilationW() == kCase3Dilation &&
           static_cast<std::size_t>(problem.GetGroupCount()) == kCase3Group; // depthwise
}

ConvSolution ConvDepthwiseFwd3D::GetSolution(const ExecutionContext&,
                                             const miopen::conv::ProblemDescription& problem) const
{
    ConvSolution result;
    KernelInfo kernel;
    kernel.kernel_file = "miopen_conv3d_depthwise_fwd.cpp";
    kernel.kernel_name = "miopen_conv3d_depthwise_fwd";

    kernel.l_wk.push_back(256);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);
    kernel.g_wk.push_back(256 * kCase3Batch);
    kernel.g_wk.push_back(kCase3Channels);
    kernel.g_wk.push_back(kCase3OutD);

    if(problem.IsFp16())
        kernel.comp_options = std::string(" -DIO_DTYPE=__half");
    else
        kernel.comp_options = std::string(" -DIO_DTYPE=__hip_bfloat16");

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [kern](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& data_ctx = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;
            const auto& in_len   = tensors.inDesc.GetLengths();
            const auto& out_len  = tensors.outDesc.GetLengths();
            const int iC         = static_cast<int>(in_len[1]);
            const int iD         = static_cast<int>(in_len[2]);
            const int iH         = static_cast<int>(in_len[3]);
            const int iW         = static_cast<int>(in_len[4]);
            const int oC         = static_cast<int>(out_len[1]);
            const int oD         = static_cast<int>(out_len[2]);
            const int oH         = static_cast<int>(out_len[3]);
            const int oW         = static_cast<int>(out_len[4]);
            handle.Run(kern)(
                tensors.in, tensors.out, tensors.w, nullptr, iC, iD, iH, iW, oC, oD, oH, oW);
        };
    };
    result.construction_params.push_back(kernel);
    result.solver_id = SolverDbId();
    return result;
}

#else

bool ConvDepthwiseFwd3D::IsApplicable(const ExecutionContext&,
                                      const miopen::conv::ProblemDescription&) const
{
    return false;
}

ConvSolution ConvDepthwiseFwd3D::GetSolution(const ExecutionContext&,
                                             const miopen::conv::ProblemDescription&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
