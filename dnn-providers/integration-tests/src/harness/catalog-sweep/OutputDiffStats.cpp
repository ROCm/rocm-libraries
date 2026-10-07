// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/catalog-sweep/OutputDiffStats.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

namespace hipdnn_integration_tests::catalog_sweep
{

namespace
{

template <class T>
OutputDiffStats measure(hipdnn_data_sdk::utilities::ITensor& expected,
                        hipdnn_data_sdk::utilities::ITensor& actual,
                        float atol,
                        float rtol)
{
    OutputDiffStats stats;
    stats.totalElements = expected.elementCount();

    hipdnn_data_sdk::utilities::TensorView<T> expectedView(expected);
    hipdnn_data_sdk::utilities::TensorView<T> actualView(actual);

    std::atomic<std::size_t> mismatched{0};
    std::atomic<std::size_t> nans{0};
    std::atomic<double> maxAbsErr{0.0};
    std::mutex maxMutex;

    auto measureElement = [&](const std::vector<int64_t>& indices) {
        const auto expectedValue = static_cast<double>(expectedView.getHostValue(indices));
        const auto actualValue = static_cast<double>(actualView.getHostValue(indices));

        if(std::isnan(actualValue))
        {
            nans.fetch_add(1);
            mismatched.fetch_add(1);
            return;
        }
        if(std::isnan(expectedValue))
        {
            // Nothing to measure against; the validator decides what a NaN reference means.
            mismatched.fetch_add(1);
            return;
        }
        if(actualValue == expectedValue)
        {
            return; // equal, including infinities of one sign
        }

        const double absDiff = std::fabs(actualValue - expectedValue);
        if(!std::isfinite(absDiff))
        {
            // An infinity against a finite value or the opposite infinity: no tolerance
            // covers it, and an infinite difference is not a maximum worth reporting.
            mismatched.fetch_add(1);
            return;
        }

        const double threshold
            = static_cast<double>(atol) + static_cast<double>(rtol) * std::fabs(expectedValue);
        if(absDiff > threshold)
        {
            mismatched.fetch_add(1);
        }
        if(absDiff > maxAbsErr.load())
        {
            const std::lock_guard<std::mutex> lock(maxMutex);
            if(absDiff > maxAbsErr.load())
            {
                maxAbsErr.store(absDiff);
            }
        }
    };

    auto parallel
        = hipdnn_test_sdk::detail::makeParallelTensorFunctor(measureElement, expected.dims());
    parallel(std::thread::hardware_concurrency());

    stats.mismatchedElements = mismatched.load();
    stats.nanElements = nans.load();
    stats.maxAbsErr = maxAbsErr.load();
    return stats;
}

} // namespace

std::optional<OutputDiffStats>
    computeOutputDiffStats(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                           hipdnn_data_sdk::utilities::ITensor& expected,
                           hipdnn_data_sdk::utilities::ITensor& actual,
                           float atol,
                           float rtol)
{
    if(expected.elementCount() != actual.elementCount() || expected.dims() != actual.dims())
    {
        return std::nullopt;
    }

    using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(dataType)
    {
    case DT::FLOAT:
        return measure<float>(expected, actual, atol, rtol);
    case DT::HALF:
        return measure<hipdnn_data_sdk::types::half>(expected, actual, atol, rtol);
    case DT::BFLOAT16:
        return measure<hipdnn_data_sdk::types::bfloat16>(expected, actual, atol, rtol);
    case DT::DOUBLE:
        return measure<double>(expected, actual, atol, rtol);
    default:
        return std::nullopt;
    }
}

} // namespace hipdnn_integration_tests::catalog_sweep
