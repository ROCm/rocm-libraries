// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipBuffer.hpp"
#include "hipblaslt_bench_options.hpp"
#include "hipblaslt_test.hpp"
#include "norm.hpp"
#include "utility.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hipblaslt/host_validation/HostComparison.hpp>
#include <optional>
#include <vector>

namespace hipblaslt::host_validation
{
    struct MatmulValidationMetrics
    {
        double& relativeFrobeniusError;
        double& absoluteTolerance;
        double& relativeTolerance;
        double& maximumUlp;
        double& averageUlp;
    };

    struct MatmulValidationRequest
    {
        hipStream_t                   stream;
        const Arguments&              arguments;
        int32_t                       gemmCount;
        const std::vector<int64_t>&   rows;
        const std::vector<int64_t>&   columns;
        const std::vector<int64_t>&   outputLeadingDimensions;
        const std::vector<int64_t>&   auxiliaryLeadingDimensions;
        const std::vector<int64_t>&   outputBatchStrides;
        const std::vector<int64_t>&   auxiliaryBatchStrides;
        const std::vector<int>&       batchCounts;
        const std::vector<size_t>&    biasSizes;
        std::vector<HipHostBuffer>&   expectedOutput;
        std::vector<HipHostBuffer>&   observedOutput;
        std::vector<HipHostBuffer>&   expectedMaximum;
        std::vector<HipHostBuffer>&   observedMaximum;
        std::vector<HipDeviceBuffer>& deviceMaximum;
        std::vector<HipHostBuffer>&   expectedAuxiliary;
        std::vector<HipHostBuffer>&   observedAuxiliary;
        std::vector<HipDeviceBuffer>& deviceAuxiliary;
        std::vector<HipHostBuffer>&   expectedBias;
        std::vector<HipHostBuffer>&   observedBias;
        std::vector<HipDeviceBuffer>& deviceBias;
        const std::vector<double>&    absoluteTolerances;
        const std::vector<double>&    symmetricRelativeTolerances;
        MatmulValidationMetrics       metrics;
        hipDataType                   outputType;
        hipDataType                   biasType;
        hipDataType                   auxiliaryType;
        hipDataType                   computeType;
        hipblasLtBatchMode_t          batchMode = HIPBLASLT_BATCH_MODE_STRIDED;
    };

    void validateMatmulOutputs(const MatmulValidationRequest& request)
    {
        const hipStream_t          stream               = request.stream;
        const Arguments&           arg                  = request.arguments;
        const int32_t              gemm_count           = request.gemmCount;
        const auto&                M                    = request.rows;
        const auto&                N                    = request.columns;
        const auto&                ldd                  = request.outputLeadingDimensions;
        const auto&                lde                  = request.auxiliaryLeadingDimensions;
        const auto&                stride_d             = request.outputBatchStrides;
        const auto&                stride_e             = request.auxiliaryBatchStrides;
        const auto&                num_batches          = request.batchCounts;
        const auto&                size_bias            = request.biasSizes;
        auto&                      hD_gold              = request.expectedOutput;
        auto&                      hD_1                 = request.observedOutput;
        auto&                      hAmaxD_gold          = request.expectedMaximum;
        auto&                      hAmaxD               = request.observedMaximum;
        auto&                      dAmaxD               = request.deviceMaximum;
        auto&                      hE_gold              = request.expectedAuxiliary;
        auto&                      hE                   = request.observedAuxiliary;
        auto&                      dE                   = request.deviceAuxiliary;
        auto&                      hBias_gold           = request.expectedBias;
        auto&                      hBias                = request.observedBias;
        auto&                      dBias                = request.deviceBias;
        const auto&                tol                  = request.absoluteTolerances;
        const auto&                symmetricRelativeTol = request.symmetricRelativeTolerances;
        double&                    hipblaslt_error      = request.metrics.relativeFrobeniusError;
        double&                    hipblaslt_atol       = request.metrics.absoluteTolerance;
        double&                    hipblaslt_rtol       = request.metrics.relativeTolerance;
        double&                    hipblaslt_max_ulp    = request.metrics.maximumUlp;
        double&                    hipblaslt_avg_ulp    = request.metrics.averageUlp;
        const hipDataType          To                   = request.outputType;
        const hipDataType          Tbias                = request.biasType;
        const hipDataType          Taux                 = request.auxiliaryType;
        const hipDataType          Tc                   = request.computeType;
        const hipblasLtBatchMode_t batchMode            = request.batchMode;

        // fetch GPU
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        // ULP error accumulators (sum/count are used to derive the average below)
        double ulp_sum_total   = 0.0;
        size_t ulp_count_total = 0;

        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
            {
                if(!arg.gradient && arg.use_e)
                {
                    CHECK_HIP_ERROR(
                        synchronize(hE[gemmIdx], dE[gemmIdx], 0, 0, 0, 0, 1, false, stream));
                }

                if(arg.amaxD)
                {
                    CHECK_HIP_ERROR(synchronize(
                        hAmaxD[gemmIdx], dAmaxD[gemmIdx], 0, 0, 0, 0, 1, false, stream));
                }
                if(arg.gradient && arg.bias_vector)
                {
                    CHECK_HIP_ERROR(
                        synchronize(hBias[gemmIdx], dBias[gemmIdx], 0, 0, 0, 0, 1, false, stream));
                }
            }

            const auto compareBuffer = [&](int64_t     rows,
                                           int64_t     columns,
                                           int64_t     leadingDimension,
                                           int64_t     batchStride,
                                           const void* expected,
                                           const void* observed,
                                           int64_t     batchCount,
                                           hipDataType type,
                                           bool        requireSpecialValueConsistency,
                                           bool        comparePointwise,
                                           bool        computeRelativeFrobeniusError,
                                           bool        findAllCloseTolerance,
                                           bool        computeUnitsInLastPlace) {
                HostComparisonRequest request;
                request.rows             = rows;
                request.columns          = columns;
                request.leadingDimension = leadingDimension;
                request.batchStride      = batchStride;
                request.batchCount       = batchCount;
                request.expected         = expected;
                request.observed         = observed;
                request.type             = type;
#ifdef GOOGLE_TEST
                request.requireSpecialValueConsistency = requireSpecialValueConsistency;
                if(comparePointwise)
                {
                    if(symmetricRelativeTol[gemmIdx] != 0)
                    {
                        request.pointwise = HostPointwiseComparison::SymmetricRelative;
                        request.symmetricRelativeTolerance = symmetricRelativeTol[gemmIdx];
                    }
                    else
                    {
                        request.pointwise = tol[gemmIdx] != 0 ? HostPointwiseComparison::Near
                                                              : HostPointwiseComparison::Unit;
                        request.absoluteTolerance = tol[gemmIdx];
                    }
                }
#else
                (void)requireSpecialValueConsistency;
                (void)comparePointwise;
                (void)symmetricRelativeTol;
#endif
                request.computeRelativeFrobeniusError = computeRelativeFrobeniusError;
                request.findAllCloseTolerance         = findAllCloseTolerance;
                request.computeUnitsInLastPlace       = computeUnitsInLastPlace;
                return compareHost(request);
            };

#ifdef GOOGLE_TEST
            const bool requireDSpecialValueConsistency = arg.unit_check || arg.norm_check;
            const bool compareDPointwise               = arg.unit_check;
#else
            constexpr bool requireDSpecialValueConsistency = false;
            constexpr bool compareDPointwise               = false;
#endif
            const bool compareD = requireDSpecialValueConsistency || compareDPointwise
                                  || arg.norm_check || arg.allclose_check || arg.ulp_check;

            std::optional<HostComparisonReport> dComparison;
            std::vector<HostComparisonReport>   dBatchComparisons;
            if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY && compareD)
            {
                dComparison = compareBuffer(M[gemmIdx],
                                            N[gemmIdx],
                                            ldd[gemmIdx],
                                            stride_d[gemmIdx],
                                            hD_gold[gemmIdx].buf(),
                                            hD_1[gemmIdx].buf(),
                                            num_batches[gemmIdx],
                                            To,
                                            requireDSpecialValueConsistency,
                                            compareDPointwise,
                                            arg.norm_check,
                                            arg.allclose_check,
                                            arg.ulp_check);
            }
            else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY && compareD)
            {
                dBatchComparisons.reserve(num_batches[gemmIdx]);
                for(int batch = 0; batch < num_batches[gemmIdx]; ++batch)
                {
                    dBatchComparisons.push_back(compareBuffer(M[gemmIdx],
                                                              N[gemmIdx],
                                                              ldd[gemmIdx],
                                                              0,
                                                              hD_gold[batch].buf(),
                                                              hD_1[batch].buf(),
                                                              1,
                                                              To,
                                                              requireDSpecialValueConsistency,
                                                              compareDPointwise,
                                                              arg.norm_check,
                                                              arg.allclose_check,
                                                              arg.ulp_check));
                }
            }

#ifdef GOOGLE_TEST
            const auto assertSpecialValueConsistency = [](const auto& report) {
                ASSERT_EQ(report.nonFiniteMismatches, 0)
                    << "CPU and GPU disagree on NaN/infinity classification.";
            };
#endif

            // Check Inf/NaN consistency first so "Inf turned into NaN" bugs fail with a clear message.
            // Mirror the unit/norm-check buffer branching: pointer-array mode uses per-batch buffers,
            // so a strided read over num_batches would compare the wrong buffers / go out of bounds.
            if(arg.unit_check || arg.norm_check)
            {
                if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
#ifdef GOOGLE_TEST
                    assertSpecialValueConsistency(dComparison->comparison);
#endif
                }
                else
                {
                    for(int batch = 0; batch < num_batches[gemmIdx]; batch++)
                    {
#ifdef GOOGLE_TEST
                        assertSpecialValueConsistency(dBatchComparisons[batch].comparison);
#endif
                    }
                }
            }
#ifdef GOOGLE_TEST
            const auto assertPointwiseComparison = [&](const auto& report) {
                if(tol[gemmIdx] != 0 || symmetricRelativeTol[gemmIdx] != 0)
                {
                    ASSERT_TRUE(report.passed())
                        << "tolerant comparison found " << report.mismatches << " mismatches in "
                        << report.compared << " values; max absolute difference "
                        << report.maxAbsoluteDifference << ", "
                        << (symmetricRelativeTol[gemmIdx] != 0
                                ? "symmetric relative tolerance coefficient "
                                : "absolute tolerance ")
                        << (symmetricRelativeTol[gemmIdx] != 0 ? symmetricRelativeTol[gemmIdx]
                                                               : tol[gemmIdx]);
                }
                else
                {
                    ASSERT_TRUE(report.passed())
                        << "unit comparison failed for " << report.mismatches << " of "
                        << report.compared << " values; non-finite class mismatches "
                        << report.nonFiniteMismatches << ", max absolute difference "
                        << report.maxAbsoluteDifference << ", max ULP " << report.maximumUlp;
                }
            };
#endif
            if(arg.unit_check)
            {
                if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
#ifdef GOOGLE_TEST
                    assertPointwiseComparison(dComparison->comparison);
#endif
                }
                else if(batchMode == HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    for(int batch = 0; batch < num_batches[gemmIdx]; batch++)
                    {
#ifdef GOOGLE_TEST
                        assertPointwiseComparison(dBatchComparisons[batch].comparison);
#endif
                    }
                }
                if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    if(arg.amaxD)
                    {
#ifdef GOOGLE_TEST
                        const HostComparisonReport amaxPointwiseComparison
                            = compareBuffer(1,
                                            1,
                                            1,
                                            1,
                                            hAmaxD_gold[gemmIdx].buf(),
                                            hAmaxD[gemmIdx].buf(),
                                            num_batches[gemmIdx],
                                            Tc,
                                            false,
                                            true,
                                            false,
                                            false,
                                            false);
                        assertPointwiseComparison(amaxPointwiseComparison.comparison);
#endif
                    }
                    if(!arg.gradient && arg.use_e)
                    {
#ifdef GOOGLE_TEST
                        const HostComparisonReport auxiliaryPointwiseComparison
                            = compareBuffer(M[gemmIdx],
                                            N[gemmIdx],
                                            lde[gemmIdx],
                                            stride_e[gemmIdx],
                                            hE_gold[gemmIdx].buf(),
                                            hE[gemmIdx].buf(),
                                            num_batches[gemmIdx],
                                            Taux,
                                            false,
                                            true,
                                            false,
                                            false,
                                            false);
                        assertPointwiseComparison(auxiliaryPointwiseComparison.comparison);
#endif
                    }
                    if(arg.gradient && arg.bias_vector)
                    {
#ifdef GOOGLE_TEST
                        const HostComparisonReport biasPointwiseComparison
                            = compareBuffer(size_bias[gemmIdx],
                                            1,
                                            size_bias[gemmIdx],
                                            size_bias[gemmIdx],
                                            hBias_gold[gemmIdx].buf(),
                                            hBias[gemmIdx].buf(),
                                            num_batches[gemmIdx],
                                            Tbias,
                                            false,
                                            true,
                                            false,
                                            false,
                                            false);
                        assertPointwiseComparison(biasPointwiseComparison.comparison);
#endif
                    }
                }
            }

            if(arg.norm_check)
            {
                double norm_error = 0.0;
                if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    norm_error = std::abs(dComparison->relativeFrobeniusError);
                }
                else
                {
                    for(int batch = 0; batch < num_batches[gemmIdx]; batch++)
                    {
                        norm_error = std::abs(dBatchComparisons[batch].relativeFrobeniusError);
                        hipblaslt_error += norm_error;
                    }
                }
                hipblaslt_error += norm_error;
                if(arg.norm_check_assert)
                {
                    CHECK_SUCCESS(
                        norm_check(norm_error, To, arg.compute_type, arg.a_type, arg.b_type));
                }
                if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    if(arg.amaxD)
                    {
                        const HostComparisonReport amaxFrobeniusComparison
                            = compareBuffer(1,
                                            1,
                                            1,
                                            1,
                                            hAmaxD_gold[gemmIdx].buf(),
                                            hAmaxD[gemmIdx].buf(),
                                            num_batches[gemmIdx],
                                            Tc,
                                            false,
                                            false,
                                            true,
                                            false,
                                            false);
                        double norm_error
                            = std::abs(amaxFrobeniusComparison.relativeFrobeniusError);
                        hipblaslt_error += norm_error;
                        if(arg.norm_check_assert)
                            CHECK_SUCCESS(norm_check(norm_error, Tc));
                    }
                    if(!arg.gradient && arg.use_e)
                    {
                        const HostComparisonReport auxiliaryFrobeniusComparison
                            = compareBuffer(M[gemmIdx],
                                            N[gemmIdx],
                                            lde[gemmIdx],
                                            stride_e[gemmIdx],
                                            hE_gold[gemmIdx].buf(),
                                            hE[gemmIdx].buf(),
                                            num_batches[gemmIdx],
                                            Taux,
                                            false,
                                            false,
                                            true,
                                            false,
                                            false);
                        double norm_error
                            = std::abs(auxiliaryFrobeniusComparison.relativeFrobeniusError);
                        hipblaslt_error += norm_error;
                        if(arg.norm_check_assert)
                        {
                            CHECK_SUCCESS(norm_check(
                                norm_error, Taux, arg.compute_type, arg.a_type, arg.b_type));
                        }
                    }
                    if(arg.gradient && arg.bias_vector)
                    {
                        const HostComparisonReport biasFrobeniusComparison
                            = compareBuffer(M[gemmIdx],
                                            1,
                                            M[gemmIdx],
                                            M[gemmIdx],
                                            hBias_gold[gemmIdx].buf(),
                                            hBias[gemmIdx].buf(),
                                            num_batches[gemmIdx],
                                            Tbias,
                                            false,
                                            false,
                                            true,
                                            false,
                                            false);
                        double norm_error
                            = std::abs(biasFrobeniusComparison.relativeFrobeniusError);
                        hipblaslt_error += norm_error;
                        if(arg.norm_check_assert)
                        {
                            CHECK_SUCCESS(norm_check(norm_error, Tbias));
                        }
                    }
                }
            }

            if(arg.allclose_check)
            {
                if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    if(M[gemmIdx] != 0 && N[gemmIdx] != 0 && num_batches[gemmIdx] != 0
                       && dComparison->allCloseTolerance)
                    {
                        hipblaslt_atol = dComparison->allCloseTolerance->absolute;
                        hipblaslt_rtol = dComparison->allCloseTolerance->relative;
                    }
                    else if(M[gemmIdx] != 0 && N[gemmIdx] != 0 && num_batches[gemmIdx] != 0)
                    {
                        hipblaslt_atol = 1.0;
                        hipblaslt_rtol = 1.0;
                    }
                }
                else
                {
                    for(int batch = 0; batch < num_batches[gemmIdx]; batch++)
                    {
                        if(M[gemmIdx] != 0 && N[gemmIdx] != 0
                           && dBatchComparisons[batch].allCloseTolerance)
                        {
                            hipblaslt_atol = dBatchComparisons[batch].allCloseTolerance->absolute;
                            hipblaslt_rtol = dBatchComparisons[batch].allCloseTolerance->relative;
                        }
                        else if(M[gemmIdx] != 0 && N[gemmIdx] != 0)
                        {
                            hipblaslt_atol = 1.0;
                            hipblaslt_rtol = 1.0;
                        }
                    }
                }
            }

            if(arg.ulp_check)
            {
                if(batchMode != HIPBLASLT_BATCH_MODE_POINTER_ARRAY)
                {
                    hipblaslt_max_ulp = std::max(
                        hipblaslt_max_ulp, dComparison->unitsInLastPlaceComparison.maximumUlp);
                    ulp_sum_total += dComparison->unitsInLastPlaceComparison.sumUlp;
                    ulp_count_total += dComparison->unitsInLastPlaceComparison.ulpCompared;
                }
                else
                {
                    for(int batch = 0; batch < num_batches[gemmIdx]; batch++)
                    {
                        hipblaslt_max_ulp = std::max(
                            hipblaslt_max_ulp,
                            dBatchComparisons[batch].unitsInLastPlaceComparison.maximumUlp);
                        ulp_sum_total += dBatchComparisons[batch].unitsInLastPlaceComparison.sumUlp;
                        ulp_count_total
                            += dBatchComparisons[batch].unitsInLastPlaceComparison.ulpCompared;
                    }
                }
            }
        }

        if(arg.ulp_check && ulp_count_total > 0)
            hipblaslt_avg_ulp = ulp_sum_total / ulp_count_total;
    }
} // namespace hipblaslt::host_validation
