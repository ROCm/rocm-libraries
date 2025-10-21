// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__HIP_PLATFORM_AMD__)
// Need these for the half and bfloat16 types
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#endif

#include <hipdnn_sdk/logging/Logger.hpp>
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

        double squareDifference = 0.0;
        double maxRefMagnitude = 0.0;
        double maxImplMagnitude = 0.0;

        TensorView<T> refView(reference);
        TensorView<T> implView(implementation);

        auto refItr = refView.begin();
        auto implItr = implView.begin();

        while(refItr != refView.end() && implItr != implView.end())
        {
            T refValueT = *refItr++;
            T implValueT = *implItr++;

            auto refValue = static_cast<double>(refValueT);
            auto implValue = static_cast<double>(implValueT);

            auto diff = refValue - implValue;
            squareDifference += diff * diff;

            // Track maximum magnitudes
            maxRefMagnitude = std::max(maxRefMagnitude, std::fabs(refValue));
            maxImplMagnitude = std::max(maxImplMagnitude, std::fabs(implValue));
        }

        return checkRmsError(
            squareDifference, maxRefMagnitude, maxImplMagnitude, reference.elementCount());
    }

    bool allClose(MigratableMemoryBase<T>& reference, MigratableMemoryBase<T>& implementation) const
    {
        if(reference.count() != implementation.count())
        {
            return false;
        }

        size_t elementCount = reference.count();

        if(elementCount == 0)
        {
            return true;
        }

        const T* refData = reference.hostData();
        const T* implData = implementation.hostData();

        double squareDifference = 0.0;
        double maxRefMagnitude = 0.0;
        double maxImplMagnitude = 0.0;

        // Iterate through all elements to calculate square differences and find max magnitudes
        for(size_t i = 0; i < elementCount; ++i)
        {
            auto refValue = static_cast<double>(refData[i]);
            auto implValue = static_cast<double>(implData[i]);

            // Accumulate square differences
            auto diff = refValue - implValue;
            squareDifference += diff * diff;

            // Track maximum magnitudes
            maxRefMagnitude = std::max(maxRefMagnitude, std::fabs(refValue));
            maxImplMagnitude = std::max(maxImplMagnitude, std::fabs(implValue));
        }

        return checkRmsError(squareDifference, maxRefMagnitude, maxImplMagnitude, elementCount);
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
