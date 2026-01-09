/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/env.hpp>
#include <miopen/softmax/solvers.hpp>

#include <miopen/softmax/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/softmax.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/float_equal.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace {
constexpr int nextPow2(int v)
{
    if(v == 1)
    {
        return (v << 1);
    }
    else
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
}
} // namespace

namespace solver {

namespace softmax {

bool Softmax::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    [[maybe_unused]] const miopen::softmax::ProblemDescription& problem) const
{
    if(!(problem.GetYDesc().GetType() == miopenFloat || problem.GetYDesc().GetType() == miopenHalf))
    {
        return false;
    }
    if(problem.IsForward())
    {
        if(problem.GetXDesc().GetType() != problem.GetYDesc().GetType())
        {
            return false;
        }
        if(problem.GetXDesc().GetVectorLength() != problem.GetYDesc().GetVectorLength())
        {
            return false;
        }
        if(problem.GetXDesc().GetLayoutEnum() != problem.GetYDesc().GetLayoutEnum())
        {
            return false;
        }
    }
    if(!problem.IsForward())
    {
        if(problem.GetdYDesc().GetType() != problem.GetYDesc().GetType())
        {
            return false;
        }
        if(problem.GetdXDesc().GetType() != problem.GetYDesc().GetType())
        {
            return false;
        }
        if(problem.GetYDesc().GetVectorLength() != problem.GetdYDesc().GetVectorLength() ||
           problem.GetYDesc().GetVectorLength() != problem.GetdXDesc().GetVectorLength())
        {
            return false;
        }
        if(problem.GetYDesc().GetLayoutEnum() != problem.GetdYDesc().GetLayoutEnum() ||
           problem.GetYDesc().GetLayoutEnum() != problem.GetdXDesc().GetLayoutEnum())
        {
            return false;
        }
    }
    return true;
}

ConvSolution Softmax::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                  const miopen::softmax::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto lengths      = problem.GetXDesc().GetLengths();
    auto strides      = problem.GetXDesc().GetStrides();
    auto dtype        = problem.GetXDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());
    auto algorithm    = problem.GetAlgorithm();
    auto mode         = problem.GetMode();

    auto grid_size =
        mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? lengths[0] : lengths[0] * lengths[2] * lengths[3];
    auto spatial_dim = mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? 1 : lengths[2] * lengths[3];
    auto vector_size =
        mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? lengths[1] * lengths[2] * lengths[3] : lengths[1];
    auto num_batch    = vector_size < LOCAL_SIZE ? nextPow2(LOCAL_SIZE / vector_size) : 1;
    auto workgroups   = num_batch == 1               ? grid_size
                        : grid_size % num_batch == 0 ? grid_size / num_batch
                                                     : grid_size / num_batch + 1;
    auto batch_size   = LOCAL_SIZE / num_batch;
    auto u_batch_size = vector_size > batch_size ? nextPow2(vector_size / batch_size) : 1;

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = workgroups * xlocalsize;
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenSoftmax.cpp";
    kernel.kernel_name = problem.IsForward() ? "SoftmaxFwd" : "SoftmaxBwd";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"INPUT_TYPE", input_dtype},
                              {"OUTPUT_TYPE", output_dtype},
                              {"USE_SOFTMAX_FAST", algorithm == MIOPEN_SOFTMAX_FAST},
                              {"USE_SOFTMAX_ACCURATE", algorithm == MIOPEN_SOFTMAX_ACCURATE},
                              {"USE_SOFTMAX_LOG", algorithm == MIOPEN_SOFTMAX_LOG},
                              {"USE_SOFTMAX_MODE_INSTANCE", mode == MIOPEN_SOFTMAX_MODE_INSTANCE},
                              {"USE_SOFTMAX_MODE_CHANNEL", mode == MIOPEN_SOFTMAX_MODE_CHANNEL},
                              {"HEIGHT", lengths[2]},
                              {"WIDTH", lengths[3]},
                              {"N_STRIDE", strides[0]},
                              {"C_STRIDE", strides[1]},
                              {"H_STRIDE", strides[2]},
                              {"W_STRIDE", strides[3]},
                              {"LOCAL_SIZE", LOCAL_SIZE},
                              {"WORKGROUPS", workgroups},
                              {"GRID_SIZE", grid_size},
                              {"SPATIAL_DIM", spatial_dim},
                              {"VECTOR_SIZE", vector_size},
                              {"NUM_BATCH", num_batch},
                              {"BATCH_SIZE", batch_size},
                              {"U_BATCH_SIZE", u_batch_size},
                              {"IS_INPUT_PACKED", problem.GetXDesc().IsPacked()},
                              {"IS_OUTPUT_PACKED", problem.GetYDesc().IsPacked()},
                              {"IS_DINPUT_PACKED", problem.GetdXDesc().IsPacked()},
                              {"IS_DOUTPUT_PACKED", problem.GetdYDesc().IsPacked()}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    if(problem.IsForward())
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::softmax::InvokeParams>();

                kernel(params.x,
                       params.forward_y,
                       params.xdx_offset,
                       params.y_offset,
                       params.alpha,
                       params.beta);
            };
        };
    }
    else
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::softmax::InvokeParams>();

                kernel(params.backward_y,
                       params.dy,
                       params.dx,
                       params.y_offset,
                       params.dy_offset,
                       params.xdx_offset,
                       params.alpha,
                       params.beta);
            };
        };
    }

    return result;
}

} // namespace softmax

} // namespace solver

} // namespace miopen
