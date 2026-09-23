// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceRmsValidation.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefValidatorHelpers.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace hipdnn_gpu_ref
{

namespace
{

double magnitudeFromBits(unsigned long long bits)
{
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

template <class T>
GpuFpReferenceRmsValidation<T>::GpuFpReferenceRmsValidation(float relativeTolerance)
    // Rounded through T first, exactly as createRmsValidator() hands the CPU validator
    // its tolerance: otherwise a bf16 threshold could differ by a few tenths of a
    // percent between the two sites and flip a verdict right at the edge.
    : _relativeTolerance(static_cast<double>(static_cast<T>(relativeTolerance)))
{
    if(!std::isfinite(_relativeTolerance) || _relativeTolerance < 0.0)
    {
        throw std::invalid_argument("Tolerance must be finite and non-negative");
    }
}

template <class T>
bool GpuFpReferenceRmsValidation<T>::allClose(
    hipdnn_data_sdk::utilities::ITensor& reference,
    hipdnn_data_sdk::utilities::ITensor& implementation) const
{
    if(reference.elementCount() != implementation.elementCount()
       || reference.dims() != implementation.dims())
    {
        return false;
    }

    const auto totalElements = static_cast<int64_t>(reference.elementCount());
    if(totalElements == 0)
    {
        return true;
    }

    hipdnn_data_sdk::utilities::MigratableMemory<detail::RmsAccumulators> accumulators(1);
    *accumulators.hostData() = detail::RmsAccumulators{};

    detail::RmsValidatorArgs args{};
    args.reference = reference.rawDeviceData();
    args.implementation = implementation.rawDeviceData();
    args.accumulators = static_cast<detail::RmsAccumulators*>(accumulators.deviceData());
    args.totalElements = totalElements;
    detail::setStridedLayout(args, reference, implementation);

    const auto defines = detail::buildValidatorDefines(detail::HipRtcTypeName<T>::VALUE, "double");
    auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefValidator.cpp", defines, "validateRms");

    detail::launchValidatorKernel(kernel.function(), totalElements, args);

    accumulators.markDeviceModified();
    const auto& totals = *accumulators.hostData();

    if(totals.nanOrInf != 0)
    {
        // The kernel keeps no indices, so unlike the host validator this cannot say where.
        HIPDNN_SDK_LOG_ERROR("NaN or Inf detected in the reference or implementation. This may "
                             "indicate an output element was not written by the operation.");
        return false;
    }

    const double maxMagnitude = std::max({magnitudeFromBits(totals.maxRefMagnitudeBits),
                                          magnitudeFromBits(totals.maxImplMagnitudeBits),
                                          std::numeric_limits<double>::min()});

    const double relativeRmsError
        = std::sqrt(totals.squareDifference)
          / (std::sqrt(static_cast<double>(totalElements)) * maxMagnitude);

    // Logged in the host validator's words: the mismatch report shows the threshold and
    // the element drift, but this ratio is what decided the verdict.
    if(relativeRmsError > _relativeTolerance)
    {
        HIPDNN_SDK_LOG_ERROR("Validation failed: relative rms error = " << relativeRmsError
                                                                        << ", relative tolerance = "
                                                                        << _relativeTolerance);
    }

    return relativeRmsError <= _relativeTolerance;
}

template class GpuFpReferenceRmsValidation<float>;
template class GpuFpReferenceRmsValidation<hipdnn_data_sdk::types::half>;
template class GpuFpReferenceRmsValidation<hipdnn_data_sdk::types::bfloat16>;
template class GpuFpReferenceRmsValidation<double>;

} // namespace hipdnn_gpu_ref
