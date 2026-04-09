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
 *******************************************************************************/

#include "miopen/layernorm/problem_description.hpp"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/layernorm/solvers.hpp>
#include <miopen/layernorm/invoke_params.hpp>
#include <miopen/layernorm/utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/layernorm.hpp>
#include <miopen/target_properties.hpp>

namespace miopen {

namespace solver {

namespace layernorm {

ConvSolution LayernormBackward::GetSolution(const ExecutionContext& context,
                                            const miopen::layernorm::ProblemDescription& problem,
                                            const PerformanceConfigLayernorm& config) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetDYDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetDYDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetDXDesc().GetType());

    auto reqd_work_item_cnt =
        get_reqd_work_item_cnt(context, PerformanceConfigLayernorm::max_parallel_local_size);

    {
        size_t xlocalsize = config.local_size;
        size_t xgridsize  = problem.outer_size * problem.stride * xlocalsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenLayerNorm.cpp";
        kernel.kernel_name = "LayernormBwd";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"OUTER_SIZE", problem.outer_size},
            {"INNER_SIZE", problem.inner_size},
            {"STRIDE", problem.stride},
            {"LOCAL_SIZE", config.local_size},
            {"PARALLEL_SIZE", 1},
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

    if(is_parallelism(reqd_work_item_cnt, problem.inner_size, problem.outer_size))
    {
        auto parallelism_size =
            get_parallelism_size(reqd_work_item_cnt, problem.inner_size, problem.outer_size);

        {
            size_t xlocalsize = config.local_size;
            size_t xgridsize  = AlignUp(parallelism_size * problem.inner_size, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel = KernelInfo{};

            kernel.kernel_file = "MIOpenLayerNorm.cpp";
            kernel.kernel_name = "LayernormBwdWeightBiasParallel";

            const auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                {"OUTER_SIZE", problem.outer_size},
                {"INNER_SIZE", problem.inner_size},
                {"STRIDE", problem.stride},
                {"PARALLEL_SIZE", parallelism_size},
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

        {
            size_t xlocalsize = config.local_size;
            size_t xgridsize  = AlignUp(problem.inner_size, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel = KernelInfo{};

            kernel.kernel_file = "MIOpenLayerNorm.cpp";
            kernel.kernel_name = "LayernormBwdReduceSum";

            const auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                {"OUTER_SIZE", problem.outer_size},
                {"INNER_SIZE", problem.inner_size},
                {"STRIDE", problem.stride},
                {"PARALLEL_SIZE", parallelism_size},
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
    }
    else
    {
        size_t xlocalsize = config.local_size;
        size_t xgridsize  = problem.inner_size;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenLayerNorm.cpp";
        kernel.kernel_name = "LayernormBwdWeightBias";

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

    if(is_parallelism(reqd_work_item_cnt, problem.inner_size, problem.outer_size))
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel                      = handle_.Run(kernels[0]);
                decltype(auto) weight_bias_parallel_kernel = handle_.Run(kernels[1]);
                decltype(auto) weight_bias_kernel          = handle_.Run(kernels[2]);
                decltype(auto) params = raw_params.CastTo<miopen::layernorm::BwdInvokeParams>();

                auto reqd_work_item_count = get_reqd_work_item_cnt(
                    handle_, PerformanceConfigLayernorm::max_parallel_local_size);
                auto parallelism_size = get_parallelism_size(
                    reqd_work_item_count, params.inner_size, params.outer_size);

                auto elapsed = 0.f;
                HipEventPtr start;
                HipEventPtr stop;

                if(handle_.IsProfilingEnabled())
                {
                    start = miopen::make_hip_event();
                    stop  = miopen::make_hip_event();
                    (void)hipEventRecord(start.get(), handle_.GetStream());
                }

                kernel(params.dy,
                       params.x,
                       params.weight,
                       params.mean,
                       params.rstd,
                       params.dx,
                       static_cast<int32_t>(params.mode));

                weight_bias_parallel_kernel(params.dy,
                                            params.x,
                                            params.mean,
                                            params.rstd,
                                            params.workspace,
                                            parallelism_size);

                weight_bias_kernel(params.workspace, params.dw, params.db, parallelism_size);

                if(handle_.IsProfilingEnabled())
                {
                    (void)hipEventRecord(stop.get(), handle_.GetStream());
                    (void)hipEventSynchronize(stop.get());
                    (void)hipEventElapsedTime(&elapsed, start.get(), stop.get());
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);
                };
            };
        };
    }
    else
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel             = handle_.Run(kernels[0]);
                decltype(auto) weight_bias_kernel = handle_.Run(kernels[1]);
                decltype(auto) params = raw_params.CastTo<miopen::layernorm::BwdInvokeParams>();

                auto elapsed = 0.f;
                HipEventPtr start;
                HipEventPtr stop;

                if(handle_.IsProfilingEnabled())
                {
                    start = miopen::make_hip_event();
                    stop  = miopen::make_hip_event();
                    (void)hipEventRecord(start.get(), handle_.GetStream());
                }

                kernel(params.dy,
                       params.x,
                       params.weight,
                       params.mean,
                       params.rstd,
                       params.dx,
                       static_cast<int32_t>(params.mode));

                weight_bias_kernel(
                    params.dy, params.x, params.mean, params.rstd, params.dw, params.db);

                if(handle_.IsProfilingEnabled())
                {
                    (void)hipEventRecord(stop.get(), handle_.GetStream());
                    (void)hipEventSynchronize(stop.get());
                    (void)hipEventElapsedTime(&elapsed, start.get(), stop.get());
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);
                };
            };
        };
    }

    return result;
}

std::size_t
LayernormBackward::GetWorkspaceSize(const ExecutionContext& context,
                                    const miopen::layernorm::ProblemDescription& problem) const
{
    auto reqd_work_item_cnt =
        get_reqd_work_item_cnt(context, PerformanceConfigLayernorm::max_parallel_local_size);

    if(is_parallelism(reqd_work_item_cnt, problem.inner_size, problem.outer_size))
    {
        auto parallelism_size =
            get_parallelism_size(reqd_work_item_cnt, problem.inner_size, problem.outer_size);

        return 2 * parallelism_size * problem.inner_size *
               get_data_size(problem.GetXDesc().GetType());
    }

    return 0;
}

} // namespace layernorm

} // namespace solver

} // namespace miopen
