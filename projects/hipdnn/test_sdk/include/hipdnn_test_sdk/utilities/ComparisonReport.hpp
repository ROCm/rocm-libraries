// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <ostream>
#include <sstream>
#include <string>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>

namespace hipdnn_test_sdk::utilities
{

struct ComparisonContext
{
    std::string contextLine;
    std::string tensorLabel;
    std::string dtypeName;
    float atol = 0.0f;
    float rtol = 0.0f;
    /// Set when the verdict came from an aggregate relative-RMS check rather than a
    /// per-element one. atol/rtol did not decide it and are not printed; this threshold
    /// is. Unset means an ordinary allclose comparison.
    std::optional<float> rmsThreshold;
};

inline std::string formatComparisonHeader(const ComparisonContext& ctx,
                                          const hipdnn_data_sdk::utilities::ITensor& tensor)
{
    std::ostringstream os;
    os << "\nComparison FAILED\n"
       << "  " << ctx.contextLine << "\n"
       << "  Tensor: " << ctx.tensorLabel << "\n"
       << "  Shape:  " << StreamVec(tensor.dims()) << "  " << ctx.dtypeName << "\n";
    if(ctx.rmsThreshold.has_value())
    {
        // The element listing below counts every element that differs at all, because an
        // aggregate check has no per-element budget to fail against. Say so, or the
        // counts read as failures.
        os << "  Tolerance: relative RMS <= " << *ctx.rmsThreshold
           << "  (aggregate check — the element counts below are elements that differ at "
              "all, not elements that failed)\n";
    }
    else
    {
        os << "  Tolerance: atol=" << ctx.atol << " rtol=" << ctx.rtol << "\n";
    }
    return os.str();
}

template <typename T>
void appendComparisonDiff(std::ostream& os,
                          const std::string& tensorLabel,
                          hipdnn_data_sdk::utilities::ITensor& expected,
                          hipdnn_data_sdk::utilities::ITensor& actual,
                          float atol,
                          float rtol)
{
    const auto summary = computeTensorDiff<T>(expected, actual, atol, rtol);
    printTensorDiffSummary(os, tensorLabel, summary);
}

inline void appendComparisonDiffByDataType(std::ostream& os,
                                           hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                           const std::string& tensorLabel,
                                           hipdnn_data_sdk::utilities::ITensor& expected,
                                           hipdnn_data_sdk::utilities::ITensor& actual,
                                           float atol,
                                           float rtol)
{
    using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::types::half;

    switch(dataType)
    {
    case DT::FLOAT:
        appendComparisonDiff<float>(os, tensorLabel, expected, actual, atol, rtol);
        return;
    case DT::HALF:
        appendComparisonDiff<half>(os, tensorLabel, expected, actual, atol, rtol);
        return;
    case DT::BFLOAT16:
        appendComparisonDiff<bfloat16>(os, tensorLabel, expected, actual, atol, rtol);
        return;
    case DT::DOUBLE:
        appendComparisonDiff<double>(os, tensorLabel, expected, actual, atol, rtol);
        return;
    default:
        os << "  (no element-wise diff available for data type: "
           << hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType) << ")\n";
    }
}

} // namespace hipdnn_test_sdk::utilities
