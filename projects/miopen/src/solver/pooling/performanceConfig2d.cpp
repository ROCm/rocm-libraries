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
    const miopen::pooling::ProblemDescription& problem)
{
#if !MIOPEN_BACKEND_HIP
    std::ignore = problem;
#else
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat: Init(problem); break;
    case miopenBFloat16:
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

template <OperationType OpType>
bool PerformanceConfigPooling2d<OpType>::SetNextValue(
    const miopen::pooling::ProblemDescription&)
{
#if !MIOPEN_BACKEND_HIP
    return false;
#else
    do
    {
        if constexpr(OpType == OperationType::Backward)
        {
            // tune out_pix_tile0 only for the backward solver
            if(!NextTwoPower<min_out_pix_tile0, max_out_pix_tile0>(out_pix_tile0))
                break;
        }
        if(!NextTwoPower<min_out_pix_tile1, max_out_pix_tile1>(out_pix_tile1))
            break;
        if(!NextTwoPower<min_local_size0, max_local_size0>(local_size0))
            break;
        if(!NextTwoPower<min_local_size1, max_local_size1>(local_size1))
            break;
        return false;
    } while(false);
    return true;
#endif
}

template <OperationType OpType>
bool PerformanceConfigPooling2d<OpType>::IsValidValue() const
{
    if constexpr(OpType == OperationType::Backward)
    {
        // check out_pix_tile0 only for the backward solver
        if(!IsTwoPower<min_out_pix_tile0, max_out_pix_tile0>(out_pix_tile0))
            return false;
    }
    if(!IsTwoPower<min_out_pix_tile1, max_out_pix_tile1>(out_pix_tile1))
        return false;
    if(!IsTwoPower<min_local_size0, max_local_size0>(local_size0))
        return false;
    if(!IsTwoPower<min_local_size1, max_local_size1>(local_size1))
        return false;
    if constexpr(OpType == OperationType::Forward)
    {
        // this constraint is enforced to avoid grp_tile1 becoming zero in GetSolutionImpl in the
        // PoolingForward2d solver
        if(local_size1 / out_pix_tile1 < 1)
            return false;
    }
    return true;
}

template <OperationType OpType>
bool PerformanceConfigPooling2d<OpType>::IsValid(
    const ExecutionContext&, const miopen::pooling::ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP
    std::ignore = problem;
    return false;
#else
    switch(problem.GetXDesc().GetType())
    {
    case miopenHalf:
    case miopenFloat:
        return IsValidValue(); // perform further checks for problem & parameter set compatibility?
    case miopenBFloat16:
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
