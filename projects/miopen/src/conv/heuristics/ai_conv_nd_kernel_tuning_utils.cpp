/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
 *******************************************************************************
 * general AI-related code for kernel tuning and heuristics. To be called in the
 * solver-specific code.
 *******************************************************************************/
#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/logger.hpp>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace miopen {
namespace solver {
namespace conv {
using ProblemInterpreter = miopen::solver::ProblemInterpreter;
using ProblemDescription = miopen::conv::ProblemDescription;

int LayoutStringToCode(const std::string& layout, const bool& is3d)
{
    if(is3d)
    {
        if(layout == "NCDHW")
            return 0;
        if(layout == "NDHWC")
            return 1;
    }
    else
    {
        if(layout == "NCHW")
            return 0;
        if(layout == "NHWC")
            return 1;
    }
    // Add more as needed
    return -1; // Unknown
}

// Helper: Extract 3D convolution features
std::map<std::string, float>
GetFeaturesND(const ProblemDescription& problem, int /*max_cu*/, const std::string& /*arch*/)
{
    std::map<std::string, float> features;

    const bool is3d = problem.Is3d();
    // 1: spatial_dim
    features["spatial_dim"] = is3d ? 3.0f : 2.0f;

    // 2–5: in_channels, in_d, in_h, in_w
    features["in_channels"] = static_cast<float>(ProblemInterpreter::GetInputChannelC(problem));
    if(is3d)
    {
        features["in_d"] = static_cast<float>(ProblemInterpreter::GetInputDepthDi(problem));
    }
    features["in_h"] = static_cast<float>(ProblemInterpreter::GetInputHeightHi(problem));
    features["in_w"] = static_cast<float>(ProblemInterpreter::GetInputWidthWi(problem));

    // 6–9: out_channels, out_d, out_h, out_w
    features["out_channels"] = static_cast<float>(ProblemInterpreter::GetOutputChannelK(problem));
    if(is3d)
    {
        features["out_d"] = static_cast<float>(ProblemInterpreter::GetOutputDepthDo(problem));
    }
    features["out_h"] = static_cast<float>(ProblemInterpreter::GetOutputHeightHo(problem));
    features["out_w"] = static_cast<float>(ProblemInterpreter::GetOutputWidthWo(problem));

    // 10–12: fil_d, fil_h, fil_w
    if(is3d)
    {
        features["fil_d"] = static_cast<float>(ProblemInterpreter::GetFilterDepthZ(problem));
    }
    features["fil_h"] = static_cast<float>(ProblemInterpreter::GetFilterHeightY(problem));
    features["fil_w"] = static_cast<float>(ProblemInterpreter::GetFilterWidthX(problem));

    // 13–15: pad_d, pad_h, pad_w
    if(is3d)
    {
        features["pad_d"] = static_cast<float>(ProblemInterpreter::GetInputLeftPadD(problem));
    }
    features["pad_h"] = static_cast<float>(ProblemInterpreter::GetInputLeftPadH(problem));
    features["pad_w"] = static_cast<float>(ProblemInterpreter::GetInputLeftPadW(problem));

    // 16–18: conv_stride_d, conv_stride_h, conv_stride_w
    if(is3d)
    {
        features["conv_stride_d"] =
            static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideD(problem));
    }
    features["conv_stride_h"] =
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideH(problem));
    features["conv_stride_w"] =
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideW(problem));

    // 19–21: dilation_d, dilation_h, dilation_w
    if(is3d)
    {
        features["dilation_d"] = static_cast<float>(problem.GetDilationD());
    }
    features["dilation_h"] = static_cast<float>(problem.GetDilationH());
    features["dilation_w"] = static_cast<float>(problem.GetDilationW());

    // 22: batchsize
    features["batchsize"] = static_cast<float>(ProblemInterpreter::GetBatchN(problem));

    // 23: bias
    features["bias"] = static_cast<float>(problem.GetBias());

    // 24–26: in_layout, fil_layout, out_layout (as codes)
    features["in_layout"] =
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetInputLayout(problem), is3d));
    features["fil_layout"] =
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetFilterLayout(problem), is3d));
    features["out_layout"] =
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetOutputLayout(problem), is3d));

    // 27: precision
    features["precision"] = static_cast<float>(problem.GetInDataType());

    // 28: direction
    features["direction"] = static_cast<float>(
        problem.GetDirection() == miopen::conv::Direction::Forward           ? 0.0f
        : problem.GetDirection() == miopen::conv::Direction::BackwardData    ? 1.0f
        : problem.GetDirection() == miopen::conv::Direction::BackwardWeights ? 2.0f
                                                                             : -1.0f);

    // 29: group_count
    features["group_count"] = static_cast<float>(problem.GetGroupCount());

    return features;
}

// helper function to get a dummy execution context for when we do not have a real context
const miopen::ExecutionContext& GetDummyCtx()
{
    static miopen::ExecutionContext ctx;
    return ctx;
}

} // namespace conv
} // namespace solver
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
