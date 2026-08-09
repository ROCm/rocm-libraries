// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt comparison program. This header translates
// hipBLASLt descriptors and acceptance policy into the product-independent
// roc::host-validation comparison API. It intentionally contains no GTest
// assertions or output formatting.

#include <array>
#include <cstddef>
#include <cstdint>
#include <hipblaslt/hipblaslt.h>
#include <optional>
#include <roc/host_validation/adapters/hipblaslt/Comparison.hpp>
#include <span>
#include <stdexcept>

namespace roc::host_validation::hipblaslt_adapter
{
    enum class HostPointwiseComparison
    {
        Disabled,
        Unit,
        Near,
    };

    struct HostComparisonRequest
    {
        int64_t     rows             = 0;
        int64_t     columns          = 0;
        int64_t     leadingDimension = 0;
        int64_t     batchStride      = 0;
        int64_t     batchCount       = 0;
        const void* expected         = nullptr;
        const void* observed         = nullptr;
        hipDataType type             = HIPBLASLT_DATATYPE_INVALID;

        HostPointwiseComparison pointwise                      = HostPointwiseComparison::Disabled;
        double                  absoluteTolerance              = 0.0;
        bool                    requireSpecialValueConsistency = false;
        bool                    computeRelativeFrobeniusError  = false;
        bool                    findAllCloseTolerance          = false;
        bool                    computeUnitsInLastPlace        = false;
    };

    struct HostComparisonReport
    {
        ComparisonResult                   comparison;
        ComparisonResult                   unitsInLastPlaceComparison;
        double                             relativeFrobeniusError = 0.0;
        std::optional<ComparisonTolerance> allCloseTolerance;
    };

    namespace detail
    {
        inline ComparisonOptions unitComparisonOptions(hipDataType type)
        {
            const ScalarType  scalar = scalarType(type);
            ComparisonOptions options;
            options.equalNaNs                  = true;
            options.computePointwiseStatistics = false;
            options.computeFrobenius           = false;
            options.maxReportedMismatches      = 10;

            if(scalar == ScalarType::Float32 || scalar == ScalarType::Float64
               || scalar == ScalarType::ComplexFloat32 || scalar == ScalarType::ComplexFloat64)
            {
                options.pointwise           = false;
                options.computeUlp          = true;
                options.ulpType             = scalar;
                options.maximumUlpTolerance = 4.0;
            }
            return options;
        }
    } // namespace detail

    inline HostComparisonReport compareHost(const HostComparisonRequest& request)
    {
        if(request.rows < 0 || request.columns < 0 || request.leadingDimension < 0
           || request.batchStride < 0 || request.batchCount < 0)
            throw std::invalid_argument("hipBLASLt comparison dimensions must be non-negative.");

        HostComparisonReport report;
        const bool           hasElements
            = request.rows != 0 && request.columns != 0 && request.batchCount != 0;
        if(!hasElements)
        {
            if(request.pointwise != HostPointwiseComparison::Disabled
               || request.requireSpecialValueConsistency)
                (void)scalarType(request.type);
            return report;
        }

        ComparisonOptions options;
        switch(request.pointwise)
        {
        case HostPointwiseComparison::Unit:
            options = detail::unitComparisonOptions(request.type);
            break;
        case HostPointwiseComparison::Near:
            options                            = nearComparisonOptions(request.absoluteTolerance);
            options.computePointwiseStatistics = false;
            options.computeFrobenius           = false;
            options.maxReportedMismatches      = 10;
            break;
        case HostPointwiseComparison::Disabled:
            options.pointwise                  = false;
            options.computePointwiseStatistics = false;
            options.computeFrobenius           = false;
            options.maxReportedMismatches      = 0;
            break;
        }

        if(request.requireSpecialValueConsistency)
        {
            options.equalNaNs                  = true;
            options.computePointwiseStatistics = true;
        }
        const bool runPrimaryComparison = request.pointwise != HostPointwiseComparison::Disabled
                                          || request.requireSpecialValueConsistency;
        if(runPrimaryComparison)
        {
            report.comparison = compareBuffers(request.rows,
                                               request.columns,
                                               request.leadingDimension,
                                               request.batchStride,
                                               request.expected,
                                               request.observed,
                                               request.batchCount,
                                               request.type,
                                               options);
        }

        if(request.computeUnitsInLastPlace)
        {
            ComparisonOptions unitsInLastPlaceOptions;
            unitsInLastPlaceOptions.pointwise                  = false;
            unitsInLastPlaceOptions.computePointwiseStatistics = false;
            unitsInLastPlaceOptions.computeFrobenius           = false;
            unitsInLastPlaceOptions.computeUlp                 = true;
            unitsInLastPlaceOptions.ulpType                    = scalarType(request.type);
            unitsInLastPlaceOptions.maxReportedMismatches      = 0;
            report.unitsInLastPlaceComparison                  = compareBuffers(request.rows,
                                                               request.columns,
                                                               request.leadingDimension,
                                                               request.batchStride,
                                                               request.expected,
                                                               request.observed,
                                                               request.batchCount,
                                                               request.type,
                                                               unitsInLastPlaceOptions);
        }

        if(request.computeRelativeFrobeniusError)
        {
            const ScalarType scalar      = scalarType(request.type);
            const size_t     storageBits = scalarTypeInfo(scalar).storageBits;
            if(storageBits % 8 != 0)
                throw std::invalid_argument(
                    "hipBLASLt norm comparison requires byte-addressable output storage.");
            const size_t elementBytes = storageBits / 8;

            ComparisonOptions frobeniusOptions;
            frobeniusOptions.pointwise                  = false;
            frobeniusOptions.equalNaNs                  = true;
            frobeniusOptions.computePointwiseStatistics = false;
            frobeniusOptions.computeFrobenius           = true;
            frobeniusOptions.maxReportedMismatches      = 0;

            for(int64_t batch = 0; batch < request.batchCount; ++batch)
            {
                const size_t byteOffset
                    = static_cast<size_t>(batch * request.batchStride) * elementBytes;
                const auto* expected = static_cast<const std::byte*>(request.expected) + byteOffset;
                const auto* observed = static_cast<const std::byte*>(request.observed) + byteOffset;
                const ComparisonResult batchReport = compareBuffers(request.rows,
                                                                    request.columns,
                                                                    request.leadingDimension,
                                                                    0,
                                                                    expected,
                                                                    observed,
                                                                    1,
                                                                    request.type,
                                                                    frobeniusOptions);
                report.relativeFrobeniusError += batchReport.relativeFrobeniusError;
            }
        }

        if(request.findAllCloseTolerance)
        {
            const Layout      layout = comparisonLayout(request.rows,
                                                   request.columns,
                                                   request.leadingDimension,
                                                   request.batchStride,
                                                   request.batchCount);
            ComparisonOptions allCloseOptions;
            allCloseOptions.computePointwiseStatistics = false;
            allCloseOptions.computeFrobenius           = false;
            allCloseOptions.maxReportedMismatches      = 0;
            allCloseOptions.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;

            constexpr std::array<double, 6> candidates{
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
            };
            report.allCloseTolerance
                = findAllCloseTolerance(comparisonView(request.observed, request.type, layout),
                                        comparisonView(request.expected, request.type, layout),
                                        std::span<const double>(candidates),
                                        std::span<const double>(candidates),
                                        allCloseOptions);
        }

        return report;
    }
} // namespace roc::host_validation::hipblaslt_adapter
