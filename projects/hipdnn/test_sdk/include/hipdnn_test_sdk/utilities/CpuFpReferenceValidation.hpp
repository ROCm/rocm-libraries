// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

template <class T>
class CpuFpReferenceValidation : public IReferenceValidation
{
public:
    // NOLINTNEXTLINE(readability-redundant-casting) - cast needed for non-float T types
    CpuFpReferenceValidation(float absoluteTolerance = float(std::numeric_limits<T>::epsilon()),
                             // NOLINTNEXTLINE(readability-redundant-casting)
                             float relativeTolerance = float(std::numeric_limits<T>::epsilon()))
        : _absoluteTolerance(absoluteTolerance)
        , _relativeTolerance(relativeTolerance)
    {
        if(absoluteTolerance < 0.0f || relativeTolerance < 0.0f || std::isnan(absoluteTolerance)
           || std::isnan(relativeTolerance) || std::isinf(absoluteTolerance)
           || std::isinf(relativeTolerance))
        {
            throw std::invalid_argument("Tolerances must be finite and non-negative");
        }
    }

    ~CpuFpReferenceValidation() override = default;

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
    {
        if(reference.elementCount() != implementation.elementCount()
           || reference.dims() != implementation.dims())
        {
            return false;
        }

        hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
        hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

        std::atomic<bool> result(true);

        auto validateFunc = [&](const std::vector<int64_t>& indices) {
            using hipdnn_data_sdk::types::fabs;
            using hipdnn_data_sdk::types::isnan;
            using hipdnn_data_sdk::types::isinf;
            T refValue = refView.getHostValue(indices);
            T implValue = implView.getHostValue(indices);

            if(isnan(refValue) || isinf(refValue) || isnan(implValue) || isinf(implValue))
            {
                HIPDNN_SDK_LOG_ERROR(
                    "NaN or Inf detected at indices "
                    << StreamVec(indices) << ": reference value = " << refValue
                    << ", implementation value = " << implValue
                    << ". This may indicate an output element was not written by the operation.");
                result.store(false, std::memory_order_relaxed);
                return result.load(std::memory_order_relaxed);
            }

            auto absDiff = fabs(static_cast<float>(implValue) - static_cast<float>(refValue));
            auto threshold
                = _absoluteTolerance + _relativeTolerance * fabs(static_cast<float>(refValue));

            if(absDiff > threshold)
            {
                // Log error and mark as failed
                HIPDNN_SDK_LOG_ERROR(
                    "Validation failed at indices "
                    << StreamVec(indices) << ": reference value = " << refValue
                    << ", implementation value = " << implValue
                    << ", absolute difference = " << absDiff << ", threshold = " << threshold
                    << ", difference - threshold = " << (absDiff - threshold)
                    << ", (atol=" << _absoluteTolerance << ", rtol=" << _relativeTolerance << ")");
                result.store(false, std::memory_order_relaxed);
            }
            return result.load(std::memory_order_relaxed);
        };

        // Create and execute parallel functor
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(validateFunc, reference.dims());
        parallelFunc(std::thread::hardware_concurrency());

        return result.load();
    }

private:
    float _absoluteTolerance;
    float _relativeTolerance;
};

template <class T>
class CpuIntReferenceValidation : public IReferenceValidation
{
public:
    CpuIntReferenceValidation() = default;
    ~CpuIntReferenceValidation() override = default;

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
    {
        if(reference.elementCount() != implementation.elementCount()
           || reference.dims() != implementation.dims())
        {
            return false;
        }

        hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
        hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

        std::atomic<bool> result(true);

        auto validateFunc = [&](const std::vector<int64_t>& indices) {
            T refValue = refView.getHostValue(indices);
            T implValue = implView.getHostValue(indices);

            if(refValue == std::numeric_limits<T>::max()
               || implValue == std::numeric_limits<T>::max())
            {
                HIPDNN_SDK_LOG_ERROR(
                    "Sentinel value detected at indices "
                    << StreamVec(indices) << ": reference value = " << refValue
                    << ", implementation value = " << implValue
                    << ". This may indicate an output element was not written by the operation.");
                result.store(false, std::memory_order_relaxed);
                return result.load(std::memory_order_relaxed);
            }

            T absDiff = static_cast<T>(hipdnn_data_sdk::types::abs(implValue - refValue));

            // Integer values must be equal
            if(absDiff > 0)
            {
                // Log error and mark as failed
                HIPDNN_SDK_LOG_ERROR("Validation failed for integer values at indices "
                                     << StreamVec(indices) << ": reference value = " << refValue
                                     << ", implementation value = " << implValue
                                     << ", absolute difference = " << absDiff);
                result.store(false, std::memory_order_relaxed);
            }
            return result.load(std::memory_order_relaxed);
        };

        // Create and execute parallel functor
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(validateFunc, reference.dims());
        parallelFunc(std::thread::hardware_concurrency());

        return result.load();
    }
};

inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createAllCloseValidator(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                            float absoluteTolerance = std::numeric_limits<float>::epsilon(),
                            float relativeTolerance = std::numeric_limits<float>::epsilon())
{
    switch(dataType)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return std::make_unique<CpuFpReferenceValidation<float>>(absoluteTolerance,
                                                                 relativeTolerance);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
        return std::make_unique<CpuFpReferenceValidation<hipdnn_data_sdk::types::half>>(
            absoluteTolerance, relativeTolerance);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return std::make_unique<CpuFpReferenceValidation<hipdnn_data_sdk::types::bfloat16>>(
            absoluteTolerance, relativeTolerance);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE:
        return std::make_unique<CpuFpReferenceValidation<double>>(absoluteTolerance,
                                                                  relativeTolerance);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT8:
        return std::make_unique<CpuIntReferenceValidation<int8_t>>();
    case hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8:
        return std::make_unique<CpuIntReferenceValidation<uint8_t>>();
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT32:
        return std::make_unique<CpuIntReferenceValidation<int32_t>>();
    default:
        throw std::runtime_error("Unsupported data type for allClose validator");
    }
}

template <typename T>
inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createAllCloseValidator(float absoluteTolerance = float(std::numeric_limits<T>::epsilon()),
                            float relativeTolerance = float(std::numeric_limits<T>::epsilon()))
{
    if constexpr(std::is_integral_v<T>)
    {
        return std::make_unique<CpuIntReferenceValidation<T>>();
    }
    else
    {
        return std::make_unique<CpuFpReferenceValidation<T>>(absoluteTolerance, relativeTolerance);
    }
}

} // namespace hipdnn_test_sdk::utilities
