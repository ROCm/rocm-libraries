// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt comparison adapter. This header translates
// hipBLASLt descriptor geometry and acceptance policy into the product-independent
// roc::host-numerics comparison API; presentation of the result is caller-owned.

#include <array>
#include <cstddef>
#include <cstdint>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/host_numerics/Types.hpp>
#include <optional>
#include <roc/host_numerics/comparison.hpp>
#include <span>
#include <stdexcept>

namespace hipblaslt::host_numerics
{
    using ::roc::host_numerics::allCloseComparisonOptions;
    using ::roc::host_numerics::compare;
    using ::roc::host_numerics::ComparisonOptions;
    using ::roc::host_numerics::ComparisonReport;
    using ::roc::host_numerics::ComparisonTolerance;
    using ::roc::host_numerics::findAllCloseTolerance;
    using ::roc::host_numerics::IndexOrder;
    using ::roc::host_numerics::Layout;
    using ::roc::host_numerics::nearComparisonOptions;
    using ::roc::host_numerics::ScalarType;
    using ::roc::host_numerics::scalarTypeInfo;
    using ::roc::host_numerics::Shape;

    enum class HostAllCloseMode
    {
        /// Skip finite-value all-close acceptance.
        Disabled,

        /// Apply the hipBLASLt unit-check policy: exact comparison for narrow and integer
        /// storage, or at most four encoded ULPs for float32/float64 real or complex storage.
        Unit,

        /// Apply an absolute all-close tolerance supplied by HostComparisonRequest.
        Near,

        /// Apply the scale-aware symmetric tolerance
        ///   |observed - expected| < tolerance * (|observed| + |expected| + 1).
        /// The +1 term supplies an absolute floor for cancellation near zero.
        SymmetricRelative,
    };

    struct HostComparisonRequest
    {
        /// Logical row count of each column-major matrix.
        int64_t rows = 0;

        /// Logical column count of each column-major matrix.
        int64_t columns = 0;

        /// Element stride between consecutive columns.
        int64_t leadingDimension = 0;

        /// Element stride between consecutive batches.
        int64_t batchStride = 0;

        /// Number of matrices in each buffer.
        int64_t batchCount = 0;

        /// Buffer containing the reference values.
        const void* expected = nullptr;

        /// Buffer containing the values under test.
        const void* observed = nullptr;

        /// hipBLASLt storage type shared by expected and observed.
        hipDataType type = HIPBLASLT_DATATYPE_INVALID;

        /// Finite-value all-close acceptance policy.
        HostAllCloseMode allCloseMode = HostAllCloseMode::Disabled;

        /// Absolute tolerance used only when allCloseMode is Near.
        double absoluteTolerance = 0.0;

        /// Symmetric relative coefficient used only when allCloseMode is SymmetricRelative.
        double symmetricRelativeTolerance = 0.0;

        /// Collect NaN/infinity agreement statistics in comparison, independently of finite
        /// all-close acceptance.
        bool requireSpecialValueConsistency = false;

        /// Sum the independently computed relative Frobenius error for each batch.
        bool computeRelativeFrobeniusError = false;

        /// Search the built-in absolute/relative tolerance candidates for the first passing pair.
        bool findAllCloseTolerance = false;

        /// Compute ULP summary statistics independently of the primary all-close policy.
        bool computeUnitsInLastPlace = false;
    };

    struct HostComparisonReport
    {
        /// Primary all-close result and/or special-value statistics requested by comparison.
        ComparisonReport comparison;

        /// ULP metrics requested by computeUnitsInLastPlace.
        ComparisonReport unitsInLastPlaceComparison;

        /// Sum of per-batch relative Frobenius errors.
        double relativeFrobeniusError = 0.0;

        /// First passing all-close candidate, or no value if the search was disabled or failed.
        std::optional<ComparisonTolerance> allCloseTolerance;
    };

    namespace detail
    {
        inline Layout comparisonLayout(int64_t rows,
                                       int64_t columns,
                                       int64_t leadingDimension,
                                       int64_t batchStride,
                                       int64_t batchCount)
        {
            if(rows < 0 || columns < 0 || leadingDimension < 0 || batchStride < 0
               || batchCount < 0)
                throw std::invalid_argument(
                    "hipBLASLt comparison dimensions must be non-negative.");
            return Layout(
                Shape{static_cast<size_t>(rows),
                      static_cast<size_t>(columns),
                      static_cast<size_t>(batchCount)},
                {1, static_cast<ptrdiff_t>(leadingDimension), static_cast<ptrdiff_t>(batchStride)});
        }

        inline ComparisonOptions unitComparisonOptions(hipDataType type)
        {
            const ScalarType  scalar = scalarType(type);
            ComparisonOptions options;
            options.equalNaNs                  = true;
            options.computeElementwiseStatistics = false;
            options.computeFrobenius           = false;
            options.maxReportedMismatches      = 10;

            if(scalar == ScalarType::Float32 || scalar == ScalarType::Float64
               || scalar == ScalarType::ComplexFloat32 || scalar == ScalarType::ComplexFloat64)
            {
                options.allClose            = false;
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
            if(request.allCloseMode != HostAllCloseMode::Disabled
               || request.requireSpecialValueConsistency)
                (void)scalarType(request.type);
            return report;
        }

        const bool runPrimaryComparison = request.allCloseMode != HostAllCloseMode::Disabled
                                          || request.requireSpecialValueConsistency;
        if(!runPrimaryComparison && !request.computeUnitsInLastPlace
           && !request.computeRelativeFrobeniusError && !request.findAllCloseTolerance)
            return report;

        const Layout                       layout = detail::comparisonLayout(request.rows,
                                                       request.columns,
                                                       request.leadingDimension,
                                                       request.batchStride,
                                                       request.batchCount);
        const ::roc::host_numerics::Tensor expected
            = copyTensorFromEncodedStorage(request.expected, scalarType(request.type), layout);
        const ::roc::host_numerics::Tensor observed
            = copyTensorFromEncodedStorage(request.observed, scalarType(request.type), layout);

        ComparisonOptions options;
        switch(request.allCloseMode)
        {
        case HostAllCloseMode::Unit:
            options = detail::unitComparisonOptions(request.type);
            break;
        case HostAllCloseMode::Near:
            options                            = nearComparisonOptions(request.absoluteTolerance);
            options.computeElementwiseStatistics = false;
            options.computeFrobenius           = false;
            options.maxReportedMismatches      = 10;
            break;
        case HostAllCloseMode::SymmetricRelative:
            options.symmetricRelativeTolerance = request.symmetricRelativeTolerance;
            options.strictTolerance            = true;
            options.equalNaNs                  = true;
            options.computeElementwiseStatistics = false;
            options.computeFrobenius           = false;
            options.maxReportedMismatches      = 10;
            break;
        case HostAllCloseMode::Disabled:
            options.allClose                     = false;
            options.computeElementwiseStatistics = false;
            options.computeFrobenius           = false;
            options.maxReportedMismatches      = 0;
            break;
        }
        options.selection
            = ::roc::host_numerics::OutputSelection::all(IndexOrder::FirstDimensionFastest);

        if(request.requireSpecialValueConsistency)
        {
            options.equalNaNs                  = true;
            options.computeElementwiseStatistics = true;
        }
        if(runPrimaryComparison)
            report.comparison = compare(observed, expected, options);

        if(request.computeUnitsInLastPlace)
        {
            ComparisonOptions unitsInLastPlaceOptions;
            unitsInLastPlaceOptions.allClose                     = false;
            unitsInLastPlaceOptions.computeElementwiseStatistics = false;
            unitsInLastPlaceOptions.computeFrobenius           = false;
            unitsInLastPlaceOptions.computeUlp                 = true;
            unitsInLastPlaceOptions.ulpType                    = scalarType(request.type);
            unitsInLastPlaceOptions.maxReportedMismatches      = 0;
            unitsInLastPlaceOptions.selection
                = ::roc::host_numerics::OutputSelection::all(IndexOrder::FirstDimensionFastest);
            report.unitsInLastPlaceComparison
                = compare(observed, expected, unitsInLastPlaceOptions);
        }

        if(request.computeRelativeFrobeniusError)
        {
            if(scalarTypeInfo(scalarType(request.type)).storageBits % 8 != 0)
                throw std::invalid_argument(
                    "hipBLASLt norm comparison requires byte-addressable output storage.");

            ComparisonOptions frobeniusOptions;
            frobeniusOptions.allClose                     = false;
            frobeniusOptions.equalNaNs                  = true;
            frobeniusOptions.computeElementwiseStatistics = false;
            frobeniusOptions.computeFrobenius           = true;
            frobeniusOptions.maxReportedMismatches      = 0;
            frobeniusOptions.selection
                = ::roc::host_numerics::OutputSelection::all(IndexOrder::FirstDimensionFastest);

            for(int64_t batch = 0; batch < request.batchCount; ++batch)
            {
                const std::array<size_t, 3> batchCoordinates{
                    0,
                    0,
                    static_cast<size_t>(batch),
                };
                const ptrdiff_t batchOffset = layout.elementOffset(batchCoordinates);
                const Layout    batchLayout(
                    Shape{static_cast<size_t>(request.rows), static_cast<size_t>(request.columns)},
                    {1, static_cast<ptrdiff_t>(request.leadingDimension)},
                    batchOffset);
                const ComparisonReport batchReport
                    = compare(observed.shareStorageWithLayout(batchLayout),
                              expected.shareStorageWithLayout(batchLayout),
                              frobeniusOptions);
                report.relativeFrobeniusError += batchReport.relativeFrobeniusError;
            }
        }

        if(request.findAllCloseTolerance)
        {
            ComparisonOptions allCloseOptions          = allCloseComparisonOptions();
            allCloseOptions.computeElementwiseStatistics = false;
            allCloseOptions.computeFrobenius           = false;
            allCloseOptions.maxReportedMismatches      = 0;
            allCloseOptions.selection
                = ::roc::host_numerics::OutputSelection::all(IndexOrder::FirstDimensionFastest);

            constexpr std::array<double, 6> candidates{
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
            };
            report.allCloseTolerance = findAllCloseTolerance(observed,
                                                             expected,
                                                             std::span<const double>(candidates),
                                                             std::span<const double>(candidates),
                                                             allCloseOptions);
        }

        return report;
    }
} // namespace hipblaslt::host_numerics
