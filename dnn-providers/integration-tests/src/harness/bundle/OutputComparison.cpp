// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/OutputComparison.hpp"

#include <exception>
#include <sstream>
#include <stdexcept>

#include <hipdnn-gpu-ref/GpuReferenceValidationFactory.hpp>
#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

#include "harness/bundle/LiveRows.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

/// The report a glob-selected validator produces when it cannot grade the tensor the
/// glob caught. Shared by every such validator so an operator is told the same thing
/// whichever one over-matched.
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

/// The report a glob-selected validator produces when it has no implementation at the
/// site the comparison runs on. Distinct from validatorNotApplicable because the data
/// type is fine here and naming it would send the reader to the wrong config field.
///
/// makeValidator() has no device implementation of the kind, so it returns this report
/// for a device-site request.
std::string validatorNotApplicableAtSite(const std::string& label,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                         const char* validatorName)
{
    std::ostringstream error;
    error << "\nValidator override NOT APPLICABLE ON DEVICE\n"
          << "  Tensor: " << label << "\n"
          << "  Data type: " << hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType)
          << "\n"
          << "  A [[validator_overrides]] entry in this engine's TOML config selected the\n  "
          << validatorName
          << " validator for this tensor, but it exists only as a host\n"
             "  validator and this comparison runs on the device, where the reference left\n"
             "  its output.\n"
             "  Either narrow that entry's 'tensors' glob so it no longer matches this\n"
             "  tensor, or force host validation for the run with --validator cpu.\n";
    return error.str();
}

/// The report for a ragged output whose RMS comparison resolved to the device. The device
/// RMS validator does not understand ragged layouts.
std::string raggedRmsNotSupportedOnDevice(const std::string& label)
{
    std::ostringstream error;
    error << "\nRagged output NOT COMPARABLE ON DEVICE\n"
          << "  Tensor: " << label << "\n"
          << "  Ragged outputs under the rms validator are compared on the host only, and\n"
             "  this run asks for the comparison on the device.\n"
             "  Rerun with --validator cpu or --validator auto to compare it on the host.\n";
    return error.str();
}

// The report for an output with live rows whose comparison was asked to run on the
// device. The live-row view reads host memory, and no device validator skips rows.
std::string liveRowsNotSupportedOnDevice(const std::string& label)
{
    std::ostringstream error;
    error << "\nOutput with live rows NOT COMPARABLE ON DEVICE\n"
          << "  Tensor: " << label << "\n"
          << "  The metadata live_rows of this output are applied on the host only, and this\n"
             "  run asks for the comparison on the device.\n"
             "  Rerun with --validator cpu or --validator auto to compare it on the host.\n";
    return error.str();
}

// The floating-point data types, whose host allclose validator can match infinities.
bool isFloatingPoint(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(dataType)
    {
    case DataType::FLOAT:
    case DataType::HALF:
    case DataType::BFLOAT16:
    case DataType::DOUBLE:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return true;
    default:
        return false;
    }
}

// The floating-point data types that have a device allclose validator.
bool hasDeviceAllClose(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    return dataType == DataType::FLOAT || dataType == DataType::HALF
           || dataType == DataType::BFLOAT16 || dataType == DataType::DOUBLE;
}

// Grades a floating-point allclose tensor at the device site so that same-signed
// infinities match, as on the host. The device validator fails every infinity, so a
// tensor that fails on the device is graded again on the host.
bool deviceAllCloseMatchingInfinities(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                                      const std::string& label,
                                      hipdnn_data_sdk::utilities::ITensor& expected,
                                      hipdnn_data_sdk::utilities::ITensor& actual,
                                      ComparisonTolerance tolerance)
{
    tolerance.kind = ValidatorKind::ALLCLOSE;
    if(makeValidator(dataType, label, tolerance, ValidationSite::DEVICE)
           .validator->allClose(expected, actual))
    {
        return true;
    }

    expected.rawHostData();
    actual.rawHostData();
    tolerance.kind = ValidatorKind::ALLCLOSE_MATCHING_INFINITIES;
    return makeValidator(dataType, label, tolerance, ValidationSite::HOST)
        .validator->allClose(expected, actual);
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
                                 const ComparisonTolerance& tolerance,
                                 ValidationSite site)
{
    switch(tolerance.kind)
    {
    case ValidatorKind::ALLCLOSE:
        if(site == ValidationSite::DEVICE)
        {
            return {hipdnn_gpu_ref::createGpuAllCloseValidator(
                        hipdnn_test_sdk::utilities::sdkToFrontendDataType(dataType),
                        tolerance.atol,
                        tolerance.rtol),
                    {}};
        }
        return {hipdnn_test_sdk::utilities::createAllCloseValidator(
                    dataType, tolerance.atol, tolerance.rtol),
                {}};

    case ValidatorKind::RMS:
        // Only the glob-selectable kinds are caught: an unsupported data type here is an
        // operator's config mistake and deserves a legible answer. allclose has no glob to
        // blame, so its own throw stays a throw.
        try
        {
            if(site == ValidationSite::DEVICE)
            {
                return {hipdnn_gpu_ref::createGpuRmsValidator(
                            hipdnn_test_sdk::utilities::sdkToFrontendDataType(dataType),
                            tolerance.rmsThreshold),
                        {}};
            }
            return {
                hipdnn_test_sdk::utilities::createRmsValidator(dataType, tolerance.rmsThreshold),
                {}};
        }
        catch(const std::exception& e)
        {
            return {nullptr, validatorNotApplicable(label, dataType, "rms", e.what())};
        }

    case ValidatorKind::ALLCLOSE_MATCHING_INFINITIES:
        // There is no device implementation of this kind, so a DEVICE-site request is
        // refused rather than served by the host validator.
        if(site == ValidationSite::DEVICE)
        {
            return {nullptr,
                    validatorNotApplicableAtSite(label, dataType, "allclose_matching_infinities")};
        }
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
                  ValidationSite site,
                  const std::string& contextLine)
{
    const auto dataType = attrs.data_type();
    const auto label = tensorLabel(uid, attrs);

    if(site == ValidationSite::DEVICE && tolerance.kind == ValidatorKind::RMS
       && (expected.raggedIterationInfo().has_value() || actual.raggedIterationInfo().has_value()))
    {
        return TensorMismatch{uid, label, raggedRmsNotSupportedOnDevice(label)};
    }

    if(site == ValidationSite::HOST)
    {
        // Copy device-resident data to the host here, on one thread. The host validator
        // reads the elements from many threads.
        expected.rawHostData();
        actual.rawHostData();
        if(tolerance.kind == ValidatorKind::ALLCLOSE && isFloatingPoint(dataType))
        {
            tolerance.kind = ValidatorKind::ALLCLOSE_MATCHING_INFINITIES;
        }
    }
    else if(hasDeviceAllClose(dataType)
            && (tolerance.kind == ValidatorKind::ALLCLOSE
                || tolerance.kind == ValidatorKind::ALLCLOSE_MATCHING_INFINITIES))
    {
        if(deviceAllCloseMatchingInfinities(dataType, label, expected, actual, tolerance))
        {
            return std::nullopt;
        }
        return TensorMismatch{
            uid,
            label,
            formatMismatchReport(uid, label, dataType, expected, actual, tolerance, contextLine)};
    }

    auto selection = makeValidator(dataType, label, tolerance, site);
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
                   ValidationSite site,
                   const std::string& contextLine,
                   const LiveRowMap& liveRows,
                   std::optional<ValidationSite> hostOnlySite)
{
    const auto& tensorAttrMap = wrapper.getTensorMap();

    std::vector<TensorMismatch> mismatches;
    for(const int64_t uid : outputUids)
    {
        const auto* attrs = tensorAttrMap.at(uid);
        const auto tolerance = toleranceFor(tensorLabel(uid, *attrs), attrs->data_type());
        std::optional<TensorMismatch> mismatch;
        const auto live = liveRows.find(uid);
        if(live == liveRows.end())
        {
            const auto tensorSite = attrs->ragged_offset_tensor_uid().has_value()
                                        ? hostOnlySite.value_or(site)
                                        : site;
            mismatch = compareTensor(
                uid, *attrs, expectedFor(uid), *actual.at(uid), tolerance, tensorSite, contextLine);
        }
        else if(hostOnlySite.value_or(site) == ValidationSite::DEVICE)
        {
            const auto label = tensorLabel(uid, *attrs);
            mismatch = TensorMismatch{uid, label, liveRowsNotSupportedOnDevice(label)};
        }
        else
        {
            const auto expectedView
                = detail::makeLiveRowView(expectedFor(uid), attrs->data_type(), live->second);
            const auto actualView
                = detail::makeLiveRowView(*actual.at(uid), attrs->data_type(), live->second);
            mismatch = compareTensor(uid,
                                     *attrs,
                                     *expectedView,
                                     *actualView,
                                     tolerance,
                                     ValidationSite::HOST,
                                     contextLine);
        }
        if(mismatch.has_value())
        {
            mismatches.push_back(*std::move(mismatch));
        }
    }
    return mismatches;
}

} // namespace hipdnn_integration_tests::bundle
