// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

struct TensorDiffEntry
{
    std::vector<int64_t> indices;
    float refValue;
    float implValue;
    float absDiff;
};

struct TensorDiffSummary
{
    size_t totalElements;
    size_t mismatchCount;
    float maxAbsDiff;
    float meanAbsDiff;
    std::vector<int64_t> maxDiffIndices;
    std::vector<TensorDiffEntry> worstMismatches;
};

template <class T>
TensorDiffSummary computeTensorDiff(hipdnn_data_sdk::utilities::ITensor& reference,
                                    hipdnn_data_sdk::utilities::ITensor& implementation,
                                    float absoluteTolerance,
                                    float relativeTolerance,
                                    size_t maxMismatches = 5)
{
    using hipdnn_data_sdk::types::fabs;

    TensorDiffSummary summary{};
    summary.totalElements = 0;
    summary.mismatchCount = 0;
    summary.maxAbsDiff = 0.0f;
    summary.meanAbsDiff = 0.0f;

    if(reference.elementCount() != implementation.elementCount()
       || reference.dims() != implementation.dims())
    {
        return summary;
    }

    summary.totalElements = reference.elementCount();

    hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
    hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

    double sumAbsDiff = 0.0;
    std::mutex mtx;

    auto diffFunc = [&](const std::vector<int64_t>& indices) {
        T refValue = refView.getHostValue(indices);
        T implValue = implView.getHostValue(indices);

        auto absDiff = static_cast<float>(fabs(implValue - refValue));
        auto threshold = absoluteTolerance + relativeTolerance * fabs(static_cast<float>(refValue));

        if(absDiff > threshold)
        {
            const std::lock_guard<std::mutex> lock(mtx);
            ++summary.mismatchCount;
            sumAbsDiff += static_cast<double>(absDiff);

            if(absDiff > summary.maxAbsDiff)
            {
                summary.maxAbsDiff = absDiff;
                summary.maxDiffIndices = indices;
            }

            if(maxMismatches > 0)
            {
                if(summary.worstMismatches.size() < maxMismatches)
                {
                    summary.worstMismatches.push_back({indices,
                                                       static_cast<float>(refValue),
                                                       static_cast<float>(implValue),
                                                       absDiff});
                }
                else
                {
                    auto minIt
                        = std::min_element(summary.worstMismatches.begin(),
                                           summary.worstMismatches.end(),
                                           [](const TensorDiffEntry& a, const TensorDiffEntry& b) {
                                               return a.absDiff < b.absDiff;
                                           });
                    if(absDiff > minIt->absDiff)
                    {
                        *minIt = {indices,
                                  static_cast<float>(refValue),
                                  static_cast<float>(implValue),
                                  absDiff};
                    }
                }
            }
        }
    };

    auto parallelFunc
        = hipdnn_test_sdk::detail::makeParallelTensorFunctor(diffFunc, reference.dims());
    parallelFunc(std::thread::hardware_concurrency());

    if(summary.mismatchCount > 0)
    {
        summary.meanAbsDiff
            = static_cast<float>(sumAbsDiff / static_cast<double>(summary.mismatchCount));

        std::sort(summary.worstMismatches.begin(),
                  summary.worstMismatches.end(),
                  [](const TensorDiffEntry& a, const TensorDiffEntry& b) {
                      return a.absDiff > b.absDiff;
                  });
    }

    return summary;
}

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
            os << "      " << StreamVec(entry.indices) << ": ref=" << std::setprecision(6)
               << entry.refValue << ", impl=" << entry.implValue << ", diff=" << entry.absDiff
               << "\n";
        }
    }

    os << std::defaultfloat;
}

template <class T>
void printTensorDiff(std::ostream& os,
                     const std::string& tensorName,
                     hipdnn_data_sdk::utilities::ITensor& reference,
                     hipdnn_data_sdk::utilities::ITensor& implementation,
                     float absoluteTolerance,
                     float relativeTolerance,
                     size_t maxMismatches = 5)
{
    auto summary = computeTensorDiff<T>(
        reference, implementation, absoluteTolerance, relativeTolerance, maxMismatches);
    printTensorDiffSummary(os, tensorName, summary);
}

template <class T>
bool validateAndReport(std::ostream& os,
                       const std::string& tensorName,
                       const IReferenceValidation& validator,
                       hipdnn_data_sdk::utilities::ITensor& reference,
                       hipdnn_data_sdk::utilities::ITensor& implementation,
                       float absoluteTolerance,
                       float relativeTolerance)
{
    const bool valid = validator.allClose(reference, implementation);
    os << "  " << tensorName << ": " << (valid ? "successful" : "failed") << "\n";
    if(!valid)
    {
        if(reference.elementCount() != implementation.elementCount()
           || reference.dims() != implementation.dims())
        {
            os << "  Tensor diff for \"" << tensorName << "\": shape mismatch - reference "
               << StreamVec(reference.dims()) << " vs implementation "
               << StreamVec(implementation.dims()) << "\n";
        }
        else
        {
            printTensorDiff<T>(
                os, tensorName, reference, implementation, absoluteTolerance, relativeTolerance);
        }
    }
    return valid;
}

} // namespace hipdnn_test_sdk::utilities
