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
    config.local_size = PerformanceConfigSoftmax::default_local_size;
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

    auto lengths    = problem.GetXDesc().GetLengths();
    auto strides    = problem.GetXDesc().GetStrides();
    auto dtype      = problem.GetXDesc().GetType();
    auto data_dtype = miopen::GetDataType(problem.GetXDesc().GetType());
    auto algorithm  = problem.GetAlgorithm();
    auto mode       = problem.GetMode();

    auto grid_size =
        mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? lengths[0] : lengths[0] * lengths[2] * lengths[3];
    auto spatial_dim = mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? 1 : lengths[2] * lengths[3];
    auto vector_size =
        mode == MIOPEN_SOFTMAX_MODE_INSTANCE ? lengths[1] * lengths[2] * lengths[3] : lengths[1];
    auto num_batch =
        vector_size < config.local_size ? nextPow2(config.local_size / vector_size) : 1;
    auto workgroups   = num_batch == 1 ? grid_size : (grid_size + num_batch - 1) / num_batch;
    auto batch_size   = config.local_size / num_batch;
    auto u_batch_size = vector_size > batch_size ? nextPow2(vector_size / batch_size) : 1;

    size_t xlocalsize = config.local_size;
    size_t xgridsize  = workgroups * xlocalsize;
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenSoftmax.cpp";
    kernel.kernel_name = problem.IsForward() ? "SoftmaxFwd" : "SoftmaxBwd";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"DATA_TYPE", data_dtype == "bfloat16" ? "ushort" : data_dtype},
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
        {"LOCAL_SIZE", config.local_size},
        {"WORKGROUPS", workgroups},
        {"GRID_SIZE", grid_size},
        {"SPATIAL_DIM", spatial_dim},
        {"VECTOR_SIZE", vector_size},
        {"NUM_BATCH", num_batch},
        {"BATCH_SIZE", batch_size},
        {"U_BATCH_SIZE", u_batch_size},
        {"IS_INPUT_CONTIGUOUS", problem.IsForward() && problem.GetXDesc().IsContiguous()},
        {"IS_OUTPUT_CONTIGUOUS", problem.GetYDesc().IsContiguous()},
        {"IS_DINPUT_CONTIGUOUS", !problem.IsForward() && problem.GetdXDesc().IsContiguous()},
        {"IS_DOUTPUT_CONTIGUOUS", !problem.IsForward() && problem.GetdYDesc().IsContiguous()}};

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

                kernel_(params.x,
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
                decltype(auto) kernel_ = handle_.Run(kernels.front());
                decltype(auto) params  = raw_params.CastTo<miopen::softmax::InvokeParams>();

                kernel_(params.backward_y,
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

void PerformanceConfigSoftmax::HeuristicInit(const miopen::softmax::ProblemDescription& problem)
{
#if !MIOPEN_BACKEND_HIP
    std::ignore = problem;
#else
    switch(problem.GetYDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16: local_size = PerformanceConfigSoftmax::start_local_size; break;
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
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
    if(local_size <= 0)
    {
        MIOPEN_THROW(miopenStatusInvalidValue, "Local size zero or negative");
    }
    local_size *= 2;
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
    case miopenBFloat16: return true;
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }
#endif
}

bool PerformanceConfigSoftmax::operator==(const PerformanceConfigSoftmax& other) const
{
    return local_size == other.local_size;
}

} // namespace softmax

} // namespace solver

} // namespace miopen
