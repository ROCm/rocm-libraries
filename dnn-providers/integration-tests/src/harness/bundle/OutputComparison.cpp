// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/OutputComparison.hpp"

#include <exception>
#include <sstream>
#include <utility>

#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

namespace hipdnn_integration_tests::bundle
{

std::string tensorLabel(int64_t uid, const std::string& name)
{
    if(!name.empty())
    {
        return name;
    }
    return "uid=" + std::to_string(uid);
}

std::string tensorLabel(int64_t uid,
                        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs)
{
    const auto* name = attrs.name();
    return tensorLabel(uid, name != nullptr ? name->str() : std::string{});
}

ValidatorSelection makeValidator(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                 const std::string& label,
                                 const ComparisonTolerance& tolerance)
{
    if(tolerance.kind != ValidatorKind::RMS)
    {
        return {hipdnn_test_sdk::utilities::createAllCloseValidator(
                    dataType, tolerance.atol, tolerance.rtol),
                {}};
    }

    try
    {
        return {hipdnn_test_sdk::utilities::createRmsValidator(dataType, tolerance.rmsThreshold),
                {}};
    }
    catch(const std::exception& e)
    {
        // An over-broad [[validator_overrides]] 'tensors' glob, not a numerical failure:
        // say which tensor it caught and what to narrow, rather than unwinding the test.
        std::ostringstream error;
        error << "\nValidator override NOT APPLICABLE\n"
              << "  Tensor: " << label << "\n"
              << "  Data type: " << hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType)
              << "\n"
              << "  A [[validator_overrides]] entry in this engine's TOML config selected the\n"
                 "  rms validator for this tensor, but it does not support this data type ("
              << e.what()
              << ").\n"
                 "  Narrow that entry's 'tensors' glob so it no longer matches this tensor.\n";
        return {nullptr, error.str()};
    }
}

std::string formatMismatchReport(int64_t uid,
                                 const std::string& label,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                 hipdnn_data_sdk::utilities::ITensor& expected,
                                 hipdnn_data_sdk::utilities::ITensor& actual,
                                 const ComparisonTolerance& tolerance,
                                 const std::string& contextLine)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    const bool useRms = tolerance.kind == ValidatorKind::RMS;

    hipdnn_test_sdk::utilities::ComparisonContext ctx;
    ctx.contextLine = contextLine;
    ctx.tensorLabel = label + " (UID " + std::to_string(uid) + ", output)";
    ctx.dtypeName = dataType != DataType::UNSET
                        ? hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType)
                        : "unknown";
    ctx.atol = tolerance.atol;
    ctx.rtol = tolerance.rtol;
    if(useRms)
    {
        // atol/rtol did not decide this failure, so do not print them as if they had.
        std::ostringstream summary;
        summary << "relative RMS <= " << tolerance.rmsThreshold
                << "  (aggregate check — the element counts below are elements that "
                   "differ at all, not elements that failed)";
        ctx.toleranceSummary = summary.str();
    }

    std::ostringstream report;
    report << hipdnn_test_sdk::utilities::formatComparisonHeader(ctx, expected);
    if(dataType != DataType::UNSET)
    {
        // Zero tolerances under RMS: the per-element budget is not what was checked, and a
        // full drift profile (max/mean abs diff, worst elements) is the useful diagnostic.
        hipdnn_test_sdk::utilities::appendComparisonDiffByDataType(report,
                                                                   dataType,
                                                                   label,
                                                                   expected,
                                                                   actual,
                                                                   useRms ? 0.0f : tolerance.atol,
                                                                   useRms ? 0.0f : tolerance.rtol);
    }
    return report.str();
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
    const auto label = tensorLabel(uid, attrs);

    auto selection = makeValidator(dataType, label, tolerance);
    if(selection.validator == nullptr)
    {
        return TensorMismatch{uid, label, std::move(selection.error)};
    }
    if(selection.validator->allClose(expected, actual))
    {
        return std::nullopt;
    }

    return TensorMismatch{
        uid,
        label,
        formatMismatchReport(uid, label, dataType, expected, actual, tolerance, contextLine)};
}

std::vector<TensorMismatch>
    compareOutputs(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper& wrapper,
                   const std::vector<int64_t>& outputUids,
                   OutputTensors& actual,
                   const ExpectedTensorLookup& expectedFor,
                   const ToleranceLookup& toleranceFor,
                   const std::string& contextLine)
{
    const auto& tensorAttrMap = wrapper.getTensorMap();

    std::vector<TensorMismatch> mismatches;
    for(const int64_t uid : outputUids)
    {
        const auto* attrs = tensorAttrMap.at(uid);
        auto mismatch
            = compareTensor(uid,
                            *attrs,
                            expectedFor(uid),
                            *actual.at(uid),
                            toleranceFor(uid, tensorLabel(uid, *attrs), attrs->data_type()),
                            contextLine);
        if(mismatch.has_value())
        {
            mismatches.push_back(*std::move(mismatch));
        }
    }
    return mismatches;
}

} // namespace hipdnn_integration_tests::bundle
