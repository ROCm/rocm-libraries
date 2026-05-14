// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/pooling/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

namespace miopen {

namespace solver {

namespace pooling {

template <OperationType OpType>
void PerformanceConfigPoolingNd<OpType>::Init(const miopen::pooling::ProblemDescription&)
{
    // initialize with minimum values
    pix_w_per_work = min_pix_per_work;
    pix_h_per_work = min_pix_per_work;
    pix_d_per_work = min_pix_per_work;
    local_size     = 1;
}

template <OperationType OpType>
void PerformanceConfigPoolingNd<OpType>::HeuristicInit(
    [[maybe_unused]] const miopen::pooling::ProblemDescription& problem)
{
#if MIOPEN_BACKEND_HIP
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16: Init(problem); break;
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64: MIOPEN_THROW("Unsupported datatype");
    }
#endif
}

template <OperationType OpType>
bool PerformanceConfigPoolingNd<OpType>::IsValid(
    const ExecutionContext&,
    [[maybe_unused]] const miopen::pooling::ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP
    return false;
#else
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat:
    case miopenBFloat16: return IsValidValue();
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

template <OperationType OpType>
bool PerformanceConfigPoolingNd<OpType>::operator==(
    const PerformanceConfigPoolingNd<OpType>& other) const
{
    return pix_w_per_work == other.pix_w_per_work && pix_h_per_work == other.pix_h_per_work &&
           pix_d_per_work == other.pix_d_per_work && local_size == other.local_size;
}

// explict template instantiation
template struct PerformanceConfigPoolingNd<OperationType::Forward>;
template struct PerformanceConfigPoolingNd<OperationType::Backward>;

} // namespace pooling

} // namespace solver

} // namespace miopen
