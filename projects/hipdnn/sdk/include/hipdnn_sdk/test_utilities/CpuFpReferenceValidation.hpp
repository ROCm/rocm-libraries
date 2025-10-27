// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/test_utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_sdk/test_utilities/VectorLoggingUtils.hpp>
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

        TensorView<T> refView(reference);
        TensorView<T> implView(implementation);

        std::atomic<bool> result(true);

        auto validateFunc = [&](const std::vector<int64_t>& indices) {
            T refValue = refView.getHostValue(indices);
            T implValue = implView.getHostValue(indices);

            T absDiff = std::fabs(implValue - refValue);
            T threshold = _absoluteTolerance + _relativeTolerance * std::fabs(refValue);

            if(absDiff > threshold)
            {
                // Log error and mark as failed
                HIPDNN_LOG_ERROR("Validation failed at indices {}: reference value = {}, "
                                 "implementation value = {}, "
                                 "absolute difference = {}, threshold = {} (atol={}, rtol={})",
                                 indices,
                                 refValue,
                                 implValue,
                                 absDiff,
                                 threshold,
                                 _absoluteTolerance,
                                 _relativeTolerance);
                result.store(false, std::memory_order_relaxed);
            }
            return result.load(std::memory_order_relaxed);
        };

        // Create and execute parallel functor
        auto parallelFunc = makeParallelTensorFunctor(validateFunc, reference.dims());
        parallelFunc(std::thread::hardware_concurrency());

        return result.load();
    }

private:
    // Tolerances for comparison
    T _absoluteTolerance;
    T _relativeTolerance;
};

} // namespace test_utilities
} // namespace hipdnn_sdk
