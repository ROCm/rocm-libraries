// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_test_sdk/utilities/NonFiniteMask.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
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
    // `nonFiniteMask` is off by default, which keeps the behavior of rejecting every NaN and Inf.
    // With it on, matched non-finite positions are excluded from the RMS; see NonFiniteMask and
    // createRmsMatchingNonFiniteValidator for the denominator.
    CpuFpReferenceMiopenRmsValidation(T relativeTolerance = std::numeric_limits<T>::epsilon(),
                                      NonFiniteMask nonFiniteMask = NonFiniteMask{})
        : _relativeTolerance(static_cast<double>(relativeTolerance))
        , _nonFiniteMask(nonFiniteMask)
    {
        using hipdnn_data_sdk::types::isinf;
        using hipdnn_data_sdk::types::isnan;
        if(relativeTolerance < T{0.0} || isnan(relativeTolerance) || isinf(relativeTolerance))
        {
            throw std::invalid_argument("Tolerance must be finite and non-negative");
        }
        non_finite_detail::requireSentinelFits<T>(_nonFiniteMask);
    }

    ~CpuFpReferenceMiopenRmsValidation() override = default;

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
    {
        const auto raggedLayout = reference.raggedIterationInfo();
        if(reference.elementCount() != implementation.elementCount()
           || reference.dims() != implementation.dims()
           || raggedLayout != implementation.raggedIterationInfo())
        {
            return false;
        }

        if(reference.elementCount() == 0)
        {
            return true;
        }
        const bool isRagged = raggedLayout.has_value();

        std::atomic<double> squareDifference(0.0);
        std::atomic<double> maxRefMagnitude(0.0);
        std::atomic<double> maxImplMagnitude(0.0);
        std::atomic<bool> hasNanOrInf(false);
        std::atomic<size_t> comparedCount(0);
        std::atomic<size_t> finiteCount(0);

        hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
        hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

        auto validateFunc = [&](const std::vector<int64_t>& indices) {
            using hipdnn_data_sdk::types::fabs;
            using hipdnn_data_sdk::types::isnan;
            using hipdnn_data_sdk::types::isinf;
            using hipdnn_data_sdk::types::signbit;
            if(isRagged && raggedLayout->isOutOfBlock(indices))
            {
                return;
            }

            T refValueT = refView.getHostValue(indices);
            T implValueT = implView.getHostValue(indices);

            // A ragged reference marks the positions it leaves undefined with NaN (RFC 0014).
            if(isRagged)
            {
                if(isnan(refValueT))
                {
                    return;
                }
                comparedCount.fetch_add(1, std::memory_order_relaxed);
            }

            if(_nonFiniteMask.enabled())
            {
                const auto& sentinelBits = _nonFiniteMask.unwrittenSentinelBits();
                if(sentinelBits.has_value()
                   && non_finite_detail::elementBits(implValueT) == *sentinelBits)
                {
                    HIPDNN_SDK_LOG_ERROR(
                        "Unwritten-output sentinel detected at indices "
                        << StreamVec(indices)
                        << ": the implementation element still holds the pre-execute fill.");
                    hasNanOrInf.store(true, std::memory_order_relaxed);
                    return;
                }

                const bool refIsNan = isnan(refValueT);
                const bool implIsNan = isnan(implValueT);
                const bool refIsInf = isinf(refValueT);
                const bool implIsInf = isinf(implValueT);
                if(refIsNan || implIsNan || refIsInf || implIsInf)
                {
                    const bool matched
                        = (refIsNan && implIsNan)
                          || (refIsInf && implIsInf && signbit(refValueT) == signbit(implValueT));
                    if(!matched)
                    {
                        HIPDNN_SDK_LOG_ERROR(
                            "Non-finite value without a matching non-finite value at indices "
                            << StreamVec(indices) << ": reference value = " << refValueT
                            << ", implementation value = " << implValueT);
                        hasNanOrInf.store(true, std::memory_order_relaxed);
                    }
                    return;
                }
                finiteCount.fetch_add(1, std::memory_order_relaxed);
            }

            if(isnan(refValueT) || isinf(refValueT) || isnan(implValueT) || isinf(implValueT))
            {
                HIPDNN_SDK_LOG_ERROR(
                    "NaN or Inf detected at indices "
                    << StreamVec(indices) << ": reference value = " << refValueT
                    << ", implementation value = " << implValueT
                    << ". This may indicate an output element was not written by the operation.");
                hasNanOrInf.store(true, std::memory_order_relaxed);
                return;
            }

            auto refValue = static_cast<double>(refValueT);
            auto implValue = static_cast<double>(implValueT);

            auto diff = refValue - implValue;
            const double diffSquared = diff * diff;
            double currentSum = squareDifference.load(std::memory_order_relaxed);
            while(!squareDifference.compare_exchange_weak(
                currentSum, currentSum + diffSquared, std::memory_order_relaxed))
            {
            }

            // Track maximum magnitudes
            double currentMaxRef = maxRefMagnitude.load(std::memory_order_relaxed);
            const double absRefValue = fabs(refValue);
            while(absRefValue > currentMaxRef
                  && !maxRefMagnitude.compare_exchange_weak(
                      currentMaxRef, absRefValue, std::memory_order_relaxed))
            {
            }

            double currentMaxImpl = maxImplMagnitude.load(std::memory_order_relaxed);
            const double absImplValue = fabs(implValue);
            while(absImplValue > currentMaxImpl
                  && !maxImplMagnitude.compare_exchange_weak(
                      currentMaxImpl, absImplValue, std::memory_order_relaxed))
            {
            }
        };
        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(validateFunc, reference.dims());
        parallelFunc(std::thread::hardware_concurrency());

        if(hasNanOrInf.load())
        {
            return false;
        }

        if(isRagged && comparedCount.load() == 0)
        {
            HIPDNN_SDK_LOG_ERROR("No element of the ragged reference was compared: every "
                                 "in-block reference value is NaN.");
            return false;
        }

        // Elements the relative RMS error is averaged over.
        size_t gradedCount = isRagged ? comparedCount.load() : reference.elementCount();
        if(_nonFiniteMask.enabled())
        {
            // Matched NaN and Inf positions are not graded: they are neither counted here nor
            // allowed into the magnitude normalizer above.
            gradedCount = finiteCount.load();
            if(gradedCount == 0)
            {
                HIPDNN_SDK_LOG_ERROR("No finite element in the reference: every compared element "
                                     "is NaN or Inf, so nothing was graded.");
                return false;
            }
        }

        return checkRmsError(squareDifference, maxRefMagnitude, maxImplMagnitude, gradedCount);
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
        const double maxMagnitude
            = max(max(maxRefMagnitude, maxImplMagnitude), std::numeric_limits<double>::min());

        const double relativeRmsError
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

    // Non-finite handling; off unless a caller opts in.
    NonFiniteMask _nonFiniteMask;
};

namespace validation_detail
{

/// Shared body of createRmsValidator and createRmsMatchingNonFiniteValidator.
inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    makeRmsValidator(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                     float relativeTolerance,
                     const NonFiniteMask& nonFiniteMask)
{
    switch(dataType)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return std::make_unique<CpuFpReferenceMiopenRmsValidation<float>>(relativeTolerance,
                                                                          nonFiniteMask);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
        return std::make_unique<CpuFpReferenceMiopenRmsValidation<hipdnn_data_sdk::types::half>>(
            hipdnn_data_sdk::types::half(relativeTolerance), nonFiniteMask);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return std::make_unique<
            CpuFpReferenceMiopenRmsValidation<hipdnn_data_sdk::types::bfloat16>>(
            hipdnn_data_sdk::types::bfloat16(relativeTolerance), nonFiniteMask);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE:
        return std::make_unique<CpuFpReferenceMiopenRmsValidation<double>>(
            static_cast<double>(relativeTolerance), nonFiniteMask);
    default:
        throw std::runtime_error("Unsupported data type for RMS validator");
    }
}

} // namespace validation_detail

inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createRmsValidator(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                       float relativeTolerance)
{
    return validation_detail::makeRmsValidator(dataType, relativeTolerance, NonFiniteMask{});
}

/// Builds an RMS validator with the NonFiniteMask on. NaN positions that match NaN and
/// same-signed infinity positions that match are excluded from the RMS; every other
/// non-finite disagreement fails; a reference with no finite element fails. See NonFiniteMask
/// for the rules and for `unwrittenSentinelBits` (std::nullopt when the data type has no
/// payload NaN).
///
/// The relative RMS error is computed over the finite positions only:
///   sqrt(sum of squared differences) / (sqrt(F) * M)
/// where F is the number of positions at which both values are finite and M is the largest
/// magnitude among those finite values (reference and implementation). Matched non-finite
/// positions add nothing to the numerator, to F or to M, so they neither dilute the error
/// nor inflate the normalizer.
///
/// Throws for data types the RMS validator does not support (as createRmsValidator) and for
/// a sentinel wider than the data type.
inline std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>
    createRmsMatchingNonFiniteValidator(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                        float relativeTolerance,
                                        std::optional<uint64_t> unwrittenSentinelBits
                                        = std::nullopt)
{
    return validation_detail::makeRmsValidator(
        dataType, relativeTolerance, NonFiniteMask::matching(unwrittenSentinelBits));
}

} // namespace hipdnn_test_sdk::utilities
