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
    config.local_size = PerformanceConfigLayernorm::default_local_size(problem);
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
    case miopenBFloat16: local_size = 1; break;
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
    if(local_size <= 0)
    {
        MIOPEN_THROW(miopenStatusInvalidValue, "Local size zero or negative");
    }
    if(local_size * 2 <= max_local_size)
    {
        local_size *= 2;
        return true;
    }
    return false;
#endif
}

bool PerformanceConfigLayernorm::IsValidValue() const
{
    return local_size > 0 && local_size <= max_local_size;
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
    case miopenBFloat16: return CheckParallelKernelBounds(context, problem);
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

bool PerformanceConfigLayernorm::operator==(const PerformanceConfigLayernorm& other) const
{
    return local_size == other.local_size;
}

} // namespace layernorm

} // namespace solver

} // namespace miopen
