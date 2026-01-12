// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>

namespace hipdnn_test_sdk::utilities
{

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

            T absDiff = static_cast<T>(std::abs(implValue - refValue));

            // Integer values ​​must be equal
            if(absDiff > 0)
            {
                // Log error and mark as failed
                HIPDNN_LOG_ERROR("Validation failed for integer values at indices {}: ",
                                 "reference value = {}, "
                                 "implementation value = {}, "
                                 "absolute difference = {}",
                                 indices,
                                 refValue,
                                 implValue,
                                 absDiff);
                result.store(false, std::memory_order_relaxed);
            }
            return result.load(std::memory_order_relaxed);
        };

        // Create and execute parallel functor
        auto parallelFunc = makeParallelTensorFunctor(validateFunc, reference.dims());
        parallelFunc(std::thread::hardware_concurrency());

        return result.load();
    }
};

inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createAllCloseValidator(hipdnn_data_sdk::data_objects::DataType dataType,
                            float absoluteTolerance = std::numeric_limits<float>::epsilon(),
                            float relativeTolerance = std::numeric_limits<float>::epsilon())
{
    switch(dataType)
    {
    case hipdnn_data_sdk::data_objects::DataType::FLOAT:
        return std::make_unique<CpuFpReferenceValidation<float>>(absoluteTolerance,
                                                                 relativeTolerance);
    case hipdnn_data_sdk::data_objects::DataType::HALF:
        return std::make_unique<CpuFpReferenceValidation<half>>(
            static_cast<half>(absoluteTolerance), static_cast<half>(relativeTolerance));
    case hipdnn_data_sdk::data_objects::DataType::BFLOAT16:
        return std::make_unique<CpuFpReferenceValidation<hip_bfloat16>>(
            static_cast<hip_bfloat16>(absoluteTolerance),
            static_cast<hip_bfloat16>(relativeTolerance));
    case hipdnn_data_sdk::data_objects::DataType::DOUBLE:
        return std::make_unique<CpuFpReferenceValidation<double>>(
            static_cast<double>(absoluteTolerance), static_cast<double>(relativeTolerance));
    case hipdnn_data_sdk::data_objects::DataType::INT8:
        return std::make_unique<CpuIntReferenceValidation<int8_t>>();
    case hipdnn_data_sdk::data_objects::DataType::UINT8:
        return std::make_unique<CpuIntReferenceValidation<uint8_t>>();
    case hipdnn_data_sdk::data_objects::DataType::INT32:
        return std::make_unique<CpuIntReferenceValidation<int32_t>>();
    default:
        throw std::runtime_error("Unsupported data type for allClose validator");
    }
}

template <typename T>
inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createAllCloseValidator(T absoluteTolerance = std::numeric_limits<T>::epsilon(),
                            T relativeTolerance = std::numeric_limits<T>::epsilon())
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
