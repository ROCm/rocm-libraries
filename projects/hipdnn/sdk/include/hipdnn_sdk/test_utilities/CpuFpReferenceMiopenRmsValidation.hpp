// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__HIP_PLATFORM_AMD__)
// Need these for the half and bfloat16 types
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#endif

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/test_utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_sdk/utilities/TensorView.hpp>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

// CPU validator that uses MIOpen RMS calculation for comparing tensor likes.
// Can be used to replicate MIOpen's RMS tolerance checks in unit tests.
// Note that this class does not use the absolute tolerance value, as MIOpen's
// RMS check is only relative tolerance based. We recommend using cpu_fp_reference_validation
// instead, but this class can be used to compare with MIOpen tolerance checks.
template <class T>
class CpuFpReferenceMiopenRmsValidation : public IReferenceValidation
{
public:
    CpuFpReferenceMiopenRmsValidation(T relativeTolerance = std::numeric_limits<T>::epsilon())
        : _relativeTolerance(static_cast<double>(relativeTolerance))
    {
        if(relativeTolerance < T{0.0})
        {
            throw std::invalid_argument("Tolerances must be non-negative");
        }
    }

    ~CpuFpReferenceMiopenRmsValidation() override = default;

    bool allClose(ITensor& reference, ITensor& implementation) const override
    {
        if(reference.elementCount() != implementation.elementCount()
           || reference.dims() != implementation.dims())
        {
            return false;
        }

        if(reference.elementCount() == 0)
        {
            return true;
        }

        std::atomic<double> squareDifference(0.0);
        std::atomic<double> maxRefMagnitude(0.0);
        std::atomic<double> maxImplMagnitude(0.0);

        TensorView<T> refView(reference);
        TensorView<T> implView(implementation);

        auto validateFunc = [&](const std::vector<int64_t>& indices) {
            T refValueT = refView.getHostValue(indices);
            T implValueT = implView.getHostValue(indices);

            auto refValue = static_cast<double>(refValueT);
            auto implValue = static_cast<double>(implValueT);

            auto diff = refValue - implValue;
            double diffSquared = diff * diff;
            double currentSum = squareDifference.load(std::memory_order_relaxed);
            while(!squareDifference.compare_exchange_weak(
                currentSum, currentSum + diffSquared, std::memory_order_relaxed))
            {
            }

            // Track maximum magnitudes
            double currentMaxRef = maxRefMagnitude.load(std::memory_order_relaxed);
            double absRefValue = std::fabs(refValue);
            while(absRefValue > currentMaxRef
                  && !maxRefMagnitude.compare_exchange_weak(
                      currentMaxRef, absRefValue, std::memory_order_relaxed))
            {
            }

            double currentMaxImpl = maxImplMagnitude.load(std::memory_order_relaxed);
            double absImplValue = std::fabs(implValue);
            while(absImplValue > currentMaxImpl
                  && !maxImplMagnitude.compare_exchange_weak(
                      currentMaxImpl, absImplValue, std::memory_order_relaxed))
            {
            }
        };
        auto parallelFunc = makeParallelTensorFunctor(validateFunc, reference.dims());
        parallelFunc(std::thread::hardware_concurrency());

        return checkRmsError(
            squareDifference, maxRefMagnitude, maxImplMagnitude, reference.elementCount());
    }

private:
    bool checkRmsError(double squareDifference,
                       double maxRefMagnitude,
                       double maxImplMagnitude,
                       size_t elementCount) const
    {
        // Find the maximum magnitude between reference and implementation
        double maxMagnitude
            = std::max({maxRefMagnitude, maxImplMagnitude, std::numeric_limits<double>::min()});

        double relativeRmsError = std::sqrt(squareDifference)
                                  / (std::sqrt(static_cast<double>(elementCount)) * maxMagnitude);

        if(relativeRmsError > _relativeTolerance)
        {
            HIPDNN_LOG_ERROR("Validation failed: relative rms error = {}, relative tolerance = {}",
                             relativeRmsError,
                             _relativeTolerance);
        }

        return relativeRmsError <= _relativeTolerance;
    }

    // Tolerance for comparison
    double _relativeTolerance;
};

} // namespace test_utilities
} // namespace hipdnn_sdk
