// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

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

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
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

        hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
        hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

        auto validateFunc = [&](const std::vector<int64_t>& indices) {
            using hipdnn_data_sdk::types::fabs;
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
            double absRefValue = fabs(refValue);
            while(absRefValue > currentMaxRef
                  && !maxRefMagnitude.compare_exchange_weak(
                      currentMaxRef, absRefValue, std::memory_order_relaxed))
            {
            }

            double currentMaxImpl = maxImplMagnitude.load(std::memory_order_relaxed);
            double absImplValue = fabs(implValue);
            while(absImplValue > currentMaxImpl
                  && !maxImplMagnitude.compare_exchange_weak(
                      currentMaxImpl, absImplValue, std::memory_order_relaxed))
            {
            }
        };
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(validateFunc, reference.dims());
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
        using hipdnn_data_sdk::types::max;
        using hipdnn_data_sdk::types::sqrt;
        // Find the maximum magnitude between reference and implementation
        double maxMagnitude
            = max(max(maxRefMagnitude, maxImplMagnitude), std::numeric_limits<double>::min());

        double relativeRmsError
            = sqrt(squareDifference) / (sqrt(static_cast<double>(elementCount)) * maxMagnitude);

        if(relativeRmsError > _relativeTolerance)
        {
            HIPDNN_SDK_LOG_ERROR("Validation failed: relative rms error = "
                                 << relativeRmsError
                                 << ", relative tolerance = " << _relativeTolerance);
        }

        return relativeRmsError <= _relativeTolerance;
    }

    // Tolerance for comparison
    double _relativeTolerance;
};

inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createRmsValidator(hipdnn_data_sdk::data_objects::DataType dataType, float relativeTolerance)
{
    switch(dataType)
    {
    case hipdnn_data_sdk::data_objects::DataType::FLOAT:
        return std::make_unique<CpuFpReferenceMiopenRmsValidation<float>>(relativeTolerance);
    case hipdnn_data_sdk::data_objects::DataType::HALF:
        return std::make_unique<CpuFpReferenceMiopenRmsValidation<hipdnn_data_sdk::types::half>>(
            hipdnn_data_sdk::types::half(relativeTolerance));
    case hipdnn_data_sdk::data_objects::DataType::BFLOAT16:
        return std::make_unique<
            CpuFpReferenceMiopenRmsValidation<hipdnn_data_sdk::types::bfloat16>>(
            hipdnn_data_sdk::types::bfloat16(relativeTolerance));
    case hipdnn_data_sdk::data_objects::DataType::DOUBLE:
        return std::make_unique<CpuFpReferenceMiopenRmsValidation<double>>(
            static_cast<double>(relativeTolerance));
    default:
        throw std::runtime_error("Unsupported data type for RMS validator");
    }
}

} // namespace hipdnn_test_sdk::utilities
