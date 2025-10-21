// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/test_utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_sdk/utilities/TensorView.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class T>
class CpuFpReferenceValidation : public IReferenceValidation
{
public:
    CpuFpReferenceValidation(T absoluteTolerance = std::numeric_limits<T>::epsilon(),
                             T relativeTolerance = std::numeric_limits<T>::epsilon())
        : _absoluteTolerance(absoluteTolerance)
        , _relativeTolerance(relativeTolerance)
    {
        if(absoluteTolerance < static_cast<T>(0.0f) || relativeTolerance < static_cast<T>(0.0f))
        {
            throw std::invalid_argument("Tolerances must be non-negative");
        }
    }

    ~CpuFpReferenceValidation() override = default;

    bool allClose(ITensor& reference, ITensor& implementation) const override
    {
        if(reference.elementCount() != implementation.elementCount()
           || reference.dims() != implementation.dims())
        {
            return false;
        }
        bool result = true;

        TensorView<T> refView(reference);
        TensorView<T> implView(implementation);

        auto refItr = refView.begin();
        auto implItr = implView.begin();

        while(refItr != refView.end() && implItr != implView.end())
        {
            T refValue = *refItr++;
            T implValue = *implItr++;

            T absDiff = std::fabs(implValue - refValue);
            T threshold = _absoluteTolerance + _relativeTolerance * std::fabs(refValue);

            if(absDiff > threshold)
            {
                HIPDNN_LOG_ERROR("Validation failed: reference value = {}, "
                                 "implementation value = {}, "
                                 "absolute difference = {}, threshold = {} (atol={}, rtol={})",
                                 refValue,
                                 implValue,
                                 absDiff,
                                 threshold,
                                 _absoluteTolerance,
                                 _relativeTolerance);
                result = false;
                break;
            }
        }
        return result;
    }

    bool allClose(MigratableMemoryBase<T>& reference, MigratableMemoryBase<T>& implementation) const
    {
        if(reference.count() != implementation.count())
        {
            return false;
        }

        size_t elementCount = reference.count();

        const T* refData = reference.hostData();
        const T* implData = implementation.hostData();

        for(size_t i = 0; i < elementCount; ++i)
        {
            T refValue = refData[i];
            T implValue = implData[i];

            T absDiff = std::fabs(implValue - refValue);
            T threshold = _absoluteTolerance + _relativeTolerance * std::fabs(refValue);

            if(absDiff > threshold)
            {
                HIPDNN_LOG_ERROR("Validation failed at index {}: reference value = {}, "
                                 "implementation value = {}, "
                                 "absolute difference = {}, threshold = {} (atol={}, rtol={})",
                                 i,
                                 refValue,
                                 implValue,
                                 absDiff,
                                 threshold,
                                 _absoluteTolerance,
                                 _relativeTolerance);
                return false;
            }
        }

        return true;
    }

private:
    // Tolerances for comparison
    T _absoluteTolerance;
    T _relativeTolerance;
};

} // namespace test_utilities
} // namespace hipdnn_sdk
