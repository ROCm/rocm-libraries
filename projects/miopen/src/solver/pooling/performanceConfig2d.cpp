// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/pooling/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

namespace miopen {

namespace solver {

namespace pooling {

template <OperationType OpType>
void PerformanceConfigPooling2d<OpType>::Init(const miopen::pooling::ProblemDescription&)
{
    // initialize with minimum values
    out_pix_tile0 = min_out_pix_tile0;
    out_pix_tile1 = min_out_pix_tile1;
    local_size0   = min_local_size0;
    local_size1   = min_local_size1;
}

template <OperationType OpType>
void PerformanceConfigPooling2d<OpType>::HeuristicInit(
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
bool PerformanceConfigPooling2d<OpType>::SetNextValue(const miopen::pooling::ProblemDescription&)
{
#if !MIOPEN_BACKEND_HIP
    return false;
#else
    if constexpr(OpType == OperationType::Backward)
    {
        // tune out_pix_tile0 only for the backward solver
        if(!NextTwoPower<min_out_pix_tile0, max_out_pix_tile0>(out_pix_tile0))
            return true;
    }
    if(!NextTwoPower<min_out_pix_tile1, max_out_pix_tile1>(out_pix_tile1))
        return true;
    if(!NextTwoPower<min_local_size0, max_local_size0>(local_size0))
        return true;
    if(!NextTwoPower<min_local_size1, max_local_size1>(local_size1))
        return true;
    return false;
#endif
}

template <OperationType OpType>
bool PerformanceConfigPooling2d<OpType>::IsValid(
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
    case miopenBFloat16: return IsValidValue(problem);
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
bool PerformanceConfigPooling2d<OpType>::operator==(
    const PerformanceConfigPooling2d<OpType>& other) const
{
    return out_pix_tile0 == other.out_pix_tile0 && out_pix_tile1 == other.out_pix_tile1 &&
           local_size0 == other.local_size0 && local_size1 == other.local_size1;
}

// explicit template instantiations
template struct PerformanceConfigPooling2d<OperationType::Forward>;
template struct PerformanceConfigPooling2d<OperationType::Backward>;

} // namespace pooling

} // namespace solver

} // namespace miopen
