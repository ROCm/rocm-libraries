// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/OutputComparison.hpp"

#include <exception>
#include <sstream>
#include <stdexcept>

#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

namespace hipdnn_integration_tests::bundle
{

namespace
{

/// The report a glob-selected validator produces when it cannot grade the tensor the
/// glob caught. Shared by every such validator so an operator is told the same three
/// things — which tensor, which config section, and what to change — whichever one
/// over-matched.
///
/// `validatorName` is the TOML spelling, because the config line is what the reader
/// has to edit.
std::string validatorNotApplicable(const std::string& label,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                   const char* validatorName,
                                   const char* reason)
{
    std::ostringstream error;
    error << "\nValidator override NOT APPLICABLE\n"
          << "  Tensor: " << label << "\n"
          << "  Data type: " << hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType)
          << "\n"
          << "  A [[validator_overrides]] entry in this engine's TOML config selected the\n  "
          << validatorName << " validator for this tensor, but it does not support this data type ("
          << reason
          << ").\n"
             "  Narrow that entry's 'tensors' glob so it no longer matches this tensor.\n";
    return error.str();
}

} // namespace

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
    switch(tolerance.kind)
    {
    case ValidatorKind::ALLCLOSE:
        return {hipdnn_test_sdk::utilities::createAllCloseValidator(
                    dataType, tolerance.atol, tolerance.rtol),
                {}};

    case ValidatorKind::RMS:
        // Only the glob-selectable kinds are caught. They are the kinds a
        // [[validator_overrides]] entry can pick, so an unsupported data type here is an
        // operator's config mistake and deserves a legible answer. allclose is the
        // default that nothing selects, so there is no glob to blame and its own throw
        // stays a throw.
        try
        {
            return {
                hipdnn_test_sdk::utilities::createRmsValidator(dataType, tolerance.rmsThreshold),
                {}};
        }
        catch(const std::exception& e)
        {
            return {nullptr, validatorNotApplicable(label, dataType, "rms", e.what())};
        }

    case ValidatorKind::ALLCLOSE_MATCHING_INFINITIES:
        try
        {
            return {hipdnn_test_sdk::utilities::createAllCloseMatchingInfinitiesValidator(
                        dataType, tolerance.atol, tolerance.rtol),
                    {}};
        }
        catch(const std::exception& e)
        {
            return {
                nullptr,
                validatorNotApplicable(label, dataType, "allclose_matching_infinities", e.what())};
        }

    default:
        // A new ValidatorKind whose case above was never written.
        // Grading it as allclose by omission is exactly the silent miscompare this whole
        // mechanism exists to prevent, so refuse instead.
        throw std::invalid_argument("makeValidator: unhandled ValidatorKind");
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
        // Carry the threshold, not a rendered sentence: the report module owns how a
        // tolerance is worded, and atol/rtol did not decide this failure.
        ctx.rmsThreshold = tolerance.rmsThreshold;
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
        auto mismatch = compareTensor(uid,
                                      *attrs,
                                      expectedFor(uid),
                                      *actual.at(uid),
                                      toleranceFor(tensorLabel(uid, *attrs), attrs->data_type()),
                                      contextLine);
        if(mismatch.has_value())
        {
            mismatches.push_back(*std::move(mismatch));
        }
    }
    return mismatches;
}

} // namespace hipdnn_integration_tests::bundle
