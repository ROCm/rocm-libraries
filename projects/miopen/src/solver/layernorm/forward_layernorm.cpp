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

#include "miopen/layernorm/problem_description.hpp"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include "miopen/mlo_internal.hpp"
#include <miopen/layernorm.hpp>
#include <miopen/layernorm/solvers.hpp>
#include <miopen/layernorm/invoke_params.hpp>
#include <miopen/layernorm/utils.hpp>
#include <miopen/target_properties.hpp>

namespace miopen {

namespace solver {

namespace layernorm {

ConvSolution LayernormForward::GetSolution(const ExecutionContext& context,
                                           const miopen::layernorm::ProblemDescription& problem,
                                           const PerformanceConfigLayernorm& config) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto dtype        = problem.GetXDesc().GetType();
        auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
        auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());

        size_t xlocalsize = config.local_size;
        size_t xgridsize  = problem.outer_size * problem.stride * xlocalsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenLayerNorm.cpp";
        kernel.kernel_name = "LayernormFwd";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"OUTER_SIZE", problem.outer_size},
            {"INNER_SIZE", problem.inner_size},
            {"STRIDE", problem.stride},
            {"PARALLEL_SIZE", 1},
            {"LOCAL_SIZE", config.local_size},
            {"MIOPEN_ELEMENTWISE_AFFINE", 0},
            {"MIOPEN_WEIGHT_BIAS", 1},
            {"MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD", 2},
            {"MIOPEN_WEIGHT_BIAS_FUSED_ADD", 3},
            {"MIOPEN_ELEMENTWISE_AFFINE_T5", 4},
            {"MIOPEN_WEIGHT_BIAS_T5", 5},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::layernorm::InvokeParams>();

            kernel(params.x,
                   params.weight,
                   params.bias,
                   params.y,
                   params.mean,
                   params.rstd,
                   params.epsilon,
                   static_cast<int32_t>(params.mode));
        };
    };

    return result;
}

} // namespace layernorm

} // namespace solver

} // namespace miopen
