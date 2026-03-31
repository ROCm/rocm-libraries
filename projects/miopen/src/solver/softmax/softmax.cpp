// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "miopen/execution_context.hpp"
#include "miopen/miopen.h"
#include "miopen/softmax/problem_description.hpp"
#include <miopen/env.hpp>
#include <miopen/softmax/solvers.hpp>

#include <miopen/softmax/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/softmax.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/float_equal.hpp>

namespace miopen {

namespace {
constexpr uint64_t nextPow2(uint64_t v)
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
        v |= v >> 32;
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
    if(!(problem.GetYDesc().GetType() == miopenFloat ||
         problem.GetYDesc().GetType() == miopenHalf ||
         problem.GetYDesc().GetType() == miopenBFloat16))
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

PerformanceConfigSoftmax
Softmax::GetDefaultPerformanceConfig(const ExecutionContext&,
                                     const miopen::softmax::ProblemDescription& problem) const
{
    PerformanceConfigSoftmax config;
    config.HeuristicInit(problem);
    config.local_size      = PerformanceConfigSoftmax::default_local_size;
    config.vectorized      = PerformanceConfigSoftmax::default_vectorized(problem);
    config.separate_stride = PerformanceConfigSoftmax::default_separate_stride;
    MIOPEN_LOG_I(config.ToString());
    return config;
}

bool Softmax::IsValidPerformanceConfig(const ExecutionContext& context,
                                       const miopen::softmax::ProblemDescription& problem,
                                       const PerformanceConfigSoftmax& config) const
{
    return config.IsValid(context, problem);
}

ConvSolution Softmax::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                  const miopen::softmax::ProblemDescription& problem,
                                  const PerformanceConfigSoftmax& config) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype      = problem.GetXDesc().GetType();
    auto data_dtype = miopen::GetDataType(dtype);
    auto algorithm  = problem.GetAlgorithm();
    auto mode       = problem.GetMode();

    size_t grid_size, xlocalsize, ygridsize;
    if(config.separate_stride)
    {
        grid_size  = problem.outer_size;
        xlocalsize = config.local_size;
        ygridsize  = problem.stride;
    }
    else
    {
        grid_size  = problem.outer_size * problem.stride;
        xlocalsize = config.local_size;
        ygridsize  = 1;
    }
    auto num_batch =
        problem.inner_size < xlocalsize ? nextPow2(xlocalsize / problem.inner_size) : 1;
    auto batch_size       = xlocalsize / num_batch;
    auto vectorized_count = dtype == miopenFloat ? 4 : 8;
    if(config.vectorized && num_batch > 1 && batch_size >= vectorized_count)
    {
        num_batch *= vectorized_count;
        batch_size /= vectorized_count;
    }
    auto u_batch_size =
        batch_size < problem.inner_size ? nextPow2(problem.inner_size / batch_size) : 1;
    auto workgroups   = (grid_size + num_batch - 1) / num_batch;
    size_t xgridsize  = workgroups * xlocalsize;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenSoftmax.cpp";
    kernel.kernel_name = problem.IsForward() ? "SoftmaxFwd" : "SoftmaxBwd";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"DATA_TYPE", data_dtype == "bfloat16" ? "ushort" : data_dtype},
                              {"USE_SOFTMAX_FAST", algorithm == MIOPEN_SOFTMAX_FAST},
                              {"USE_SOFTMAX_ACCURATE", algorithm == MIOPEN_SOFTMAX_ACCURATE},
                              {"USE_SOFTMAX_LOG", algorithm == MIOPEN_SOFTMAX_LOG},
                              {"USE_SOFTMAX_MODE_INSTANCE", mode == MIOPEN_SOFTMAX_MODE_INSTANCE},
                              {"USE_SOFTMAX_MODE_CHANNEL", mode == MIOPEN_SOFTMAX_MODE_CHANNEL},
                              {"X_OFFSET", problem.GetXOffset()},
                              {"Y_OFFSET", problem.GetYOffset()},
                              {"DX_OFFSET", problem.GetdXOffset()},
                              {"DY_OFFSET", problem.GetdYOffset()},
                              {"OUTER_SIZE", problem.outer_size},
                              {"INNER_SIZE", problem.inner_size},
                              {"STRIDE", problem.stride},
                              {"LOCAL_SIZE", xlocalsize},
                              {"NUM_BATCH", num_batch},
                              {"BATCH_SIZE", batch_size},
                              {"U_BATCH_SIZE", u_batch_size},
                              {"VECTORIZED", config.vectorized},
                              {"SEPARATE_STRIDE", config.separate_stride}};

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
                decltype(auto) kernel_ = handle_.Run(kernels.front());
                decltype(auto) params  = raw_params.CastTo<miopen::softmax::InvokeParams>();

                kernel_(params.x, params.forward_y, params.alpha, params.beta);
            };
        };
    }
    else
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel_ = handle_.Run(kernels.front());
                decltype(auto) params  = raw_params.CastTo<miopen::softmax::InvokeParams>();

                kernel_(params.backward_y, params.dy, params.dx, params.alpha, params.beta);
            };
        };
    }

    return result;
}

void PerformanceConfigSoftmax::HeuristicInit(const miopen::softmax::ProblemDescription& problem)
{
#if !MIOPEN_BACKEND_HIP
    std::ignore = problem;
#else
    switch(problem.GetYDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16:
        local_size = PerformanceConfigSoftmax::start_local_size;
        vectorized = PerformanceConfigSoftmax::start_vectorized;
        break;
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64: MIOPEN_THROW("Unsupported datatype");
    }
#endif
    initialized = true;
}

bool PerformanceConfigSoftmax::SetNextValue(const miopen::softmax::ProblemDescription& problem)
{
#if !MIOPEN_BACKEND_HIP
    std::ignore = problem;
    return false;
#else
    if(!initialized)
    {
        HeuristicInit(problem);
    }
    if(local_size < start_local_size)
    {
        MIOPEN_THROW(miopenStatusInvalidValue, "Local size below valid value");
    }
    local_size *= 2;
    if(vectorized == start_vectorized && local_size > max_local_size)
    {
        local_size = start_local_size;
        vectorized = !start_vectorized;
    }
    if(separate_stride == start_separate_stride && vectorized != start_vectorized &&
       local_size > max_local_size)
    {
        local_size      = start_local_size;
        vectorized      = start_vectorized;
        separate_stride = !start_separate_stride;
    }
    return local_size <= max_local_size;
#endif
}

bool PerformanceConfigSoftmax::IsValidValue() const
{
    return local_size > 0 && local_size <= max_local_size;
}

bool PerformanceConfigSoftmax::IsValid(const ExecutionContext&,
                                       const miopen::softmax::ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP
    std::ignore = problem;
    return false;
#else
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16: return !(separate_stride && problem.stride == 1) && IsValidValue();
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64: MIOPEN_THROW("Unsupported datatype");
    }
#endif
}

bool PerformanceConfigSoftmax::operator==(const PerformanceConfigSoftmax& other) const
{
    return local_size == other.local_size && vectorized == other.vectorized &&
           separate_stride == other.separate_stride;
}

} // namespace softmax

} // namespace solver

} // namespace miopen
