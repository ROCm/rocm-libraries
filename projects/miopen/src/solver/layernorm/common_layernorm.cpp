// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "miopen/execution_context.hpp"
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

PerformanceConfigLayernorm LayernormBase::GetDefaultPerformanceConfig(
    const ExecutionContext&, const miopen::layernorm::ProblemDescription& problem) const
{
    PerformanceConfigLayernorm config;
    config.HeuristicInit(problem);
    config.local_size           = PerformanceConfigLayernorm::default_local_size(problem);
    config.vectorized           = PerformanceConfigLayernorm::default_vectorized(problem);
    config.separate_stride      = PerformanceConfigLayernorm::default_separate_stride(problem);
    config.stride_in_local_size = PerformanceConfigLayernorm::default_stride_in_local_size(problem);
    MIOPEN_LOG_I(config.ToString());
    return config;
}

bool LayernormBase::IsValidPerformanceConfig(const ExecutionContext& context,
                                             const miopen::layernorm::ProblemDescription& problem,
                                             const PerformanceConfigLayernorm& config) const
{
    return config.IsValid(context, problem);
}

bool LayernormBase::IsApplicable(const ExecutionContext& context,
                                 const miopen::layernorm::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsSameLength())
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!problem.IsRightNormDim())
        return false;
    if(!(sizeof_local_memory(
             problem,
             is_parallelism(get_reqd_work_item_cnt(
                                context, PerformanceConfigLayernorm::max_parallel_local_size),
                            problem.inner_size,
                            problem.outer_size)
                 ? PerformanceConfigLayernorm::max_parallel_local_size
                 : PerformanceConfigLayernorm::max_local_size) <=
         TargetProperties::GetMaxLocalMemorySize()))
        return false;
    return true;
}

void PerformanceConfigLayernorm::HeuristicInit(const miopen::layernorm::ProblemDescription& problem)
{
#if !MIOPEN_BACKEND_HIP
    std::ignore = problem;
#else
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16:
        local_size           = start_local_size;
        vectorized           = start_vectorized;
        separate_stride      = start_separate_stride;
        stride_in_local_size = start_stride_in_local_size;
        break;
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

bool PerformanceConfigLayernorm::SetNextValue(const miopen::layernorm::ProblemDescription& problem)
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
    if(stride_in_local_size == start_stride_in_local_size &&
       separate_stride != start_separate_stride && vectorized != start_vectorized &&
       local_size > max_local_size)
    {
        local_size           = start_local_size;
        vectorized           = start_vectorized;
        separate_stride      = start_separate_stride;
        stride_in_local_size = !start_stride_in_local_size;
    }
    return local_size <= max_local_size;
#endif
}

bool PerformanceConfigLayernorm::IsValidValue() const
{
    return local_size >= start_local_size && local_size <= max_local_size &&
           !(!separate_stride && stride_in_local_size);
}

bool PerformanceConfigLayernorm::CheckParallelKernelBounds(
    const ExecutionContext& context, const miopen::layernorm::ProblemDescription& problem) const
{
    // Max local size of the parallel backwards kernel is less than the max local size of the normal
    // backwards kernel
    return !(problem.GetDirection() == miopen::layernorm::Direction::Backward &&
             local_size > PerformanceConfigLayernorm::max_parallel_local_size &&
             is_parallelism(get_reqd_work_item_cnt(
                                context, PerformanceConfigLayernorm::max_parallel_local_size),
                            problem.inner_size,
                            problem.outer_size));
}

bool PerformanceConfigLayernorm::IsValid(const ExecutionContext& context,
                                         const miopen::layernorm::ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP
    std::ignore = problem;
    return false;
#else
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16:
        return CheckParallelKernelBounds(context, problem) &&
               !(stride_in_local_size && problem.stride > local_size) &&
               !((separate_stride || stride_in_local_size) && problem.stride == 1) &&
               IsValidValue();
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

bool PerformanceConfigLayernorm::operator==(const PerformanceConfigLayernorm& other) const
{
    return local_size == other.local_size && vectorized == other.vectorized &&
           separate_stride == other.separate_stride &&
           stride_in_local_size == other.stride_in_local_size;
}

} // namespace layernorm

} // namespace solver

} // namespace miopen
