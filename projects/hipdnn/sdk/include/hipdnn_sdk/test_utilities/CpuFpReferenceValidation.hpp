// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class T>
class CpuFpReferenceValidation : public IReferenceValidation<T>
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

    bool allClose(IMigratableMemory<T>& reference, IMigratableMemory<T>& implementation) override
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
