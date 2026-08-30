// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/OutputComparison.hpp"

#include <sstream>

#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

namespace hipdnn_integration_tests::bundle
{

std::string tensorLabel(int64_t uid,
                        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs)
{
    const auto* name = attrs.name();
    if(name != nullptr && !name->empty())
    {
        return name->str();
    }
    return "uid=" + std::to_string(uid);
}

std::optional<TensorMismatch>
    compareTensor(int64_t uid,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                  hipdnn_data_sdk::utilities::ITensor& expected,
                  hipdnn_data_sdk::utilities::ITensor& actual,
                  ComparisonTolerance tolerance,
                  const std::string& contextLine)
{
    const auto dataType = attrs.data_type();

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        dataType, tolerance.atol, tolerance.rtol);
    if(validator->allClose(expected, actual))
    {
        return std::nullopt;
    }

    const auto label = tensorLabel(uid, attrs);

    hipdnn_test_sdk::utilities::ComparisonContext ctx;
    ctx.contextLine = contextLine;
    ctx.tensorLabel = label + " (UID " + std::to_string(uid) + ", output)";
    ctx.dtypeName = hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType);
    ctx.atol = tolerance.atol;
    ctx.rtol = tolerance.rtol;

    std::ostringstream report;
    report << hipdnn_test_sdk::utilities::formatComparisonHeader(ctx, expected);
    hipdnn_test_sdk::utilities::appendComparisonDiffByDataType(
        report, dataType, label, expected, actual, tolerance.atol, tolerance.rtol);

    return TensorMismatch{uid, label, report.str()};
}

std::vector<TensorMismatch> compareOutputs(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
    const std::vector<int64_t>& outputUids,
    OutputTensors& actual,
    const ExpectedTensorLookup& expectedFor,
    const std::function<ComparisonTolerance(hipdnn_flatbuffers_sdk::data_objects::DataType)>&
        toleranceFor,
    const std::string& contextLine)
{
    const auto& tensorAttrMap = wrapper.getTensorMap();

    std::vector<TensorMismatch> mismatches;
    for(const int64_t uid : outputUids)
    {
        const auto* attrs = tensorAttrMap.at(uid);
        auto mismatch = compareTensor(uid,
                                      *attrs,
                                      expectedFor(uid),
                                      *actual.at(uid),
                                      toleranceFor(attrs->data_type()),
                                      contextLine);
        if(mismatch.has_value())
        {
            mismatches.push_back(*std::move(mismatch));
        }
    }
    return mismatches;
}

} // namespace hipdnn_integration_tests::bundle
