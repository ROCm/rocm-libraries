// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_test_sdk/utilities/ElementRule.hpp>
#include <hipdnn_test_sdk/utilities/NonFiniteMask.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

/// One listed mismatch. `verdict` is GRADED for a finite pair outside tolerance, and
/// NON_FINITE_REJECT or UNWRITTEN_REJECT for an element the validator fails by the
/// non-finite rules (ElementRule.hpp). `absDiff` is |impl - ref|; a reject whose difference is
/// not finite (a NaN or Inf operand) records 0, so it is never NaN and never ranks an entry.
struct TensorDiffEntry
{
    std::vector<int64_t> indices;
    float refValue;
    float implValue;
    float absDiff;
    ElementVerdict verdict = ElementVerdict::GRADED;
};

struct TensorDiffSummary
{
    size_t totalElements;
    /// Elements the validator fails: finite pairs outside tolerance plus every
    /// NON_FINITE_REJECT and UNWRITTEN_REJECT element.
    size_t mismatchCount;
    /// Of mismatchCount, the NON_FINITE_REJECT and UNWRITTEN_REJECT elements.
    size_t nonFiniteMismatchCount;
    /// Largest finite |impl - ref| among the mismatches; elements whose difference is not
    /// finite are left out, as they are from meanAbsDiff.
    float maxAbsDiff;
    /// Mean of the finite |impl - ref| among the mismatches.
    float meanAbsDiff;
    std::vector<int64_t> maxDiffIndices;
    /// At most maxMismatches entries, listing every non-finite or unwritten reject before
    /// any graded mismatch; graded mismatches follow by descending absDiff. Ties break by
    /// index, so the listing does not depend on thread scheduling.
    std::vector<TensorDiffEntry> worstMismatches;
};

namespace tensor_diff_detail
{

/// Whether `a` is listed before `b`: rejects first (by index), then graded mismatches by
/// descending absDiff (by index on ties). A strict weak order that never compares an absDiff
/// of a reject, so a NaN or Inf operand cannot break it.
inline bool listedBefore(const TensorDiffEntry& a, const TensorDiffEntry& b)
{
    const bool aIsReject = a.verdict != ElementVerdict::GRADED;
    const bool bIsReject = b.verdict != ElementVerdict::GRADED;
    if(aIsReject != bIsReject)
    {
        return aIsReject;
    }
    if(!aIsReject && a.absDiff != b.absDiff)
    {
        return a.absDiff > b.absDiff;
    }
    return a.indices < b.indices;
}

} // namespace tensor_diff_detail

/// Diffs `implementation` against `reference` with the element rule the validators use
/// (classifyElement): a finite pair counts when |impl - ref| > atol + rtol * |ref|, and an
/// element the validator rejects as non-finite or unwritten always counts. `nonFiniteMask`
/// and `matchingInfinities` must be the settings of the validator whose verdict is being
/// explained; the defaults describe a validator with the mask off that rejects every NaN and
/// Inf.
template <class T>
TensorDiffSummary computeTensorDiff(hipdnn_data_sdk::utilities::ITensor& reference,
                                    hipdnn_data_sdk::utilities::ITensor& implementation,
                                    float absoluteTolerance,
                                    float relativeTolerance,
                                    size_t maxMismatches = 5,
                                    NonFiniteMask nonFiniteMask = NonFiniteMask{},
                                    MatchingInfinities matchingInfinities
                                    = MatchingInfinities::REJECTED)
{
    using hipdnn_data_sdk::types::fabs;

    TensorDiffSummary summary{};
    summary.totalElements = 0;
    summary.mismatchCount = 0;
    summary.nonFiniteMismatchCount = 0;
    summary.maxAbsDiff = 0.0f;
    summary.meanAbsDiff = 0.0f;

    const auto raggedLayout = reference.raggedIterationInfo();
    if(reference.elementCount() != implementation.elementCount()
       || reference.dims() != implementation.dims()
       || raggedLayout != implementation.raggedIterationInfo())
    {
        return summary;
    }
    const bool isRagged = raggedLayout.has_value();

    hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
    hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

    double sumFiniteAbsDiff = 0.0;
    size_t finiteDiffCount = 0;
    std::mutex mtx;
    std::atomic<size_t> comparedCount(0);

    auto diffFunc = [&](const std::vector<int64_t>& indices) {
        if(isRagged && raggedLayout->isOutOfBlock(indices))
        {
            return;
        }

        T refValue = refView.getHostValue(indices);
        T implValue = implView.getHostValue(indices);

        // A ragged reference marks the positions it leaves undefined with NaN (RFC 0014).
        if(isRagged)
        {
            if(std::isnan(static_cast<float>(refValue)))
            {
                return;
            }
            comparedCount.fetch_add(1, std::memory_order_relaxed);
        }

        const auto verdict
            = classifyElement(refValue, implValue, nonFiniteMask, matchingInfinities);
        if(verdict == ElementVerdict::MATCH)
        {
            return;
        }

        float absDiff = 0.0f;
        if(verdict == ElementVerdict::GRADED)
        {
            absDiff = static_cast<float>(fabs(implValue - refValue));
            const auto threshold
                = absoluteTolerance + relativeTolerance * fabs(static_cast<float>(refValue));
            if(!(absDiff > threshold))
            {
                return;
            }
        }
        // A rejected element's difference is only informative when it is finite, as it is for
        // an unwritten-output sentinel that is a finite pattern; otherwise it is recorded as 0.
        const bool hasFiniteDiff = verdict == ElementVerdict::GRADED
                                       ? std::isfinite(absDiff)
                                       : std::isfinite(static_cast<float>(implValue))
                                             && std::isfinite(static_cast<float>(refValue));
        if(verdict != ElementVerdict::GRADED && hasFiniteDiff)
        {
            absDiff = std::fabs(static_cast<float>(implValue) - static_cast<float>(refValue));
        }

        const TensorDiffEntry entry{
            indices, static_cast<float>(refValue), static_cast<float>(implValue), absDiff, verdict};

        const std::lock_guard<std::mutex> lock(mtx);
        ++summary.mismatchCount;
        if(verdict != ElementVerdict::GRADED)
        {
            ++summary.nonFiniteMismatchCount;
        }
        if(hasFiniteDiff)
        {
            sumFiniteAbsDiff += static_cast<double>(absDiff);
            ++finiteDiffCount;
            if(absDiff > summary.maxAbsDiff)
            {
                summary.maxAbsDiff = absDiff;
                summary.maxDiffIndices = indices;
            }
        }

        if(maxMismatches > 0)
        {
            if(summary.worstMismatches.size() < maxMismatches)
            {
                summary.worstMismatches.push_back(entry);
            }
            else
            {
                // The entry listed last is the one to displace.
                auto lastIt = std::max_element(summary.worstMismatches.begin(),
                                               summary.worstMismatches.end(),
                                               tensor_diff_detail::listedBefore);
                if(tensor_diff_detail::listedBefore(entry, *lastIt))
                {
                    *lastIt = entry;
                }
            }
        }
    };

    auto parallelFunc
        = hipdnn_test_sdk::detail::makeParallelTensorFunctor(diffFunc, reference.dims());
    parallelFunc(std::thread::hardware_concurrency());

    summary.totalElements = isRagged ? comparedCount.load() : reference.elementCount();

    if(finiteDiffCount > 0)
    {
        summary.meanAbsDiff
            = static_cast<float>(sumFiniteAbsDiff / static_cast<double>(finiteDiffCount));
    }

    std::sort(summary.worstMismatches.begin(),
              summary.worstMismatches.end(),
              tensor_diff_detail::listedBefore);

    return summary;
}

namespace tensor_diff_detail
{

/// A float as the diff prints it: nan, inf and -inf spelled out, anything else as the stream
/// would print it.
inline void printDiffValue(std::ostream& os, float value)
{
    if(std::isnan(value))
    {
        os << "nan";
    }
    else if(std::isinf(value))
    {
        os << (value < 0.0f ? "-inf" : "inf");
    }
    else
    {
        os << value;
    }
}

} // namespace tensor_diff_detail

inline void printTensorDiffSummary(std::ostream& os,
                                   const std::string& tensorName,
                                   const TensorDiffSummary& summary)
{
    os << "  Tensor diff for \"" << tensorName << "\":\n";
    os << "    Total elements: " << summary.totalElements << "\n";
    double mismatchPct = 0.0;
    if(summary.totalElements > 0)
    {
        mismatchPct = 100.0 * static_cast<double>(summary.mismatchCount)
                      / static_cast<double>(summary.totalElements);
    }
    os << "    Mismatched:     " << summary.mismatchCount << " (" << std::fixed
       << std::setprecision(2) << mismatchPct << "%)\n";
    if(summary.nonFiniteMismatchCount > 0)
    {
        os << "    Non-finite:     " << summary.nonFiniteMismatchCount
           << " of them NaN/Inf or unwritten-output elements the validator rejects\n";
    }
    os << "    Max abs diff:   " << std::scientific << std::setprecision(6) << summary.maxAbsDiff;
    if(!summary.maxDiffIndices.empty())
    {
        os << " at " << StreamVec(summary.maxDiffIndices);
    }
    os << "\n";
    os << "    Mean abs diff:  " << summary.meanAbsDiff << "\n";

    if(!summary.worstMismatches.empty())
    {
        os << "    Worst mismatches:\n";
        for(const auto& entry : summary.worstMismatches)
        {
            os << "      " << StreamVec(entry.indices) << ": ref=" << std::setprecision(6);
            tensor_diff_detail::printDiffValue(os, entry.refValue);
            os << ", impl=";
            tensor_diff_detail::printDiffValue(os, entry.implValue);
            if(entry.verdict == ElementVerdict::GRADED)
            {
                os << ", diff=" << entry.absDiff;
            }
            else if(entry.verdict == ElementVerdict::UNWRITTEN_REJECT)
            {
                os << " (unwritten-output sentinel)";
            }
            os << "\n";
        }
    }

    os << std::defaultfloat;
}

inline void printRaggedLayoutMismatch(std::ostream& os,
                                      const std::string& tensorName,
                                      const hipdnn_data_sdk::utilities::ITensor& reference,
                                      const hipdnn_data_sdk::utilities::ITensor& implementation)
{
    auto describeLayout = [](const hipdnn_data_sdk::utilities::ITensor& tensor) {
        const auto layout = tensor.raggedIterationInfo();
        if(!layout.has_value())
        {
            return std::string("dense");
        }
        std::ostringstream description;
        description << "ragged origins " << StreamVec(layout->origins) << " lengths "
                    << StreamVec(layout->lengths) << " seq axis " << layout->seqAxis
                    << " seq stride " << layout->seqStride;
        return description.str();
    };

    os << "  Tensor diff for \"" << tensorName << "\": ragged layout mismatch - reference "
       << describeLayout(reference) << " vs implementation " << describeLayout(implementation)
       << "\n";
}

template <class T>
void printTensorDiff(std::ostream& os,
                     const std::string& tensorName,
                     hipdnn_data_sdk::utilities::ITensor& reference,
                     hipdnn_data_sdk::utilities::ITensor& implementation,
                     float absoluteTolerance,
                     float relativeTolerance,
                     size_t maxMismatches = 5,
                     NonFiniteMask nonFiniteMask = NonFiniteMask{},
                     MatchingInfinities matchingInfinities = MatchingInfinities::REJECTED)
{
    auto summary = computeTensorDiff<T>(reference,
                                        implementation,
                                        absoluteTolerance,
                                        relativeTolerance,
                                        maxMismatches,
                                        nonFiniteMask,
                                        matchingInfinities);
    printTensorDiffSummary(os, tensorName, summary);
}

/// `nonFiniteMask` and `matchingInfinities` must be the settings `validator` was built with, so
/// the diff lists the non-finite elements that validator rejected. The defaults describe a
/// validator with the mask off that rejects every NaN and Inf.
template <class T>
bool validateAndReport(std::ostream& os,
                       const std::string& tensorName,
                       const IReferenceValidation& validator,
                       hipdnn_data_sdk::utilities::ITensor& reference,
                       hipdnn_data_sdk::utilities::ITensor& implementation,
                       float absoluteTolerance,
                       float relativeTolerance,
                       NonFiniteMask nonFiniteMask = NonFiniteMask{},
                       MatchingInfinities matchingInfinities = MatchingInfinities::REJECTED)
{
    const bool valid = validator.allClose(reference, implementation);
    os << "  " << tensorName << ": " << (valid ? "successful" : "failed") << "\n";
    if(!valid)
    {
        if(reference.raggedIterationInfo() != implementation.raggedIterationInfo())
        {
            printRaggedLayoutMismatch(os, tensorName, reference, implementation);
        }
        else if(reference.elementCount() != implementation.elementCount()
                || reference.dims() != implementation.dims())
        {
            os << "  Tensor diff for \"" << tensorName << "\": shape mismatch - reference "
               << StreamVec(reference.dims()) << " vs implementation "
               << StreamVec(implementation.dims()) << "\n";
        }
        else
        {
            printTensorDiff<T>(os,
                               tensorName,
                               reference,
                               implementation,
                               absoluteTolerance,
                               relativeTolerance,
                               5,
                               nonFiniteMask,
                               matchingInfinities);
        }
    }
    return valid;
}

} // namespace hipdnn_test_sdk::utilities
