// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>

#include <limits>
#include <optional>

namespace hipdnn_gpu_ref
{

// GPU-based floating-point tensor validator implementing IReferenceValidation.
// Launches a HipRTC kernel to perform element-wise tolerance comparison on the GPU
// using a single atomic failure flag. Supports packed, strided and ragged (RFC 0014)
// tensors. A ragged pair must share one ragged layout; positions past a batch's stored
// rows and positions where the reference is NaN are skipped, and a ragged comparison
// that grades no element fails, matching CpuFpReferenceValidation.
template <class T>
class GpuFpReferenceValidation : public hipdnn_test_sdk::utilities::IReferenceValidation
{
public:
    // NOLINTNEXTLINE(readability-redundant-casting) - cast needed for non-float T types
    explicit GpuFpReferenceValidation(float absoluteTolerance
                                      = float(std::numeric_limits<T>::epsilon()),
                                      // NOLINTNEXTLINE(readability-redundant-casting)
                                      float relativeTolerance
                                      = float(std::numeric_limits<T>::epsilon()));

    ~GpuFpReferenceValidation() override = default;

    // NOLINTNEXTLINE(portability-template-virtual-member-function) - explicit instantiation in .cpp
    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override;

private:
    bool gpuAllClose(
        hipdnn_data_sdk::utilities::ITensor& reference,
        hipdnn_data_sdk::utilities::ITensor& implementation,
        const std::optional<hipdnn_data_sdk::utilities::RaggedIterationInfo>& raggedLayout) const;

    float _absoluteTolerance;
    float _relativeTolerance;
};

// Suppress implicit instantiation — definitions are in GpuFpReferenceValidation.cpp
extern template class GpuFpReferenceValidation<float>;
extern template class GpuFpReferenceValidation<hipdnn_data_sdk::types::half>;
extern template class GpuFpReferenceValidation<hipdnn_data_sdk::types::bfloat16>;
extern template class GpuFpReferenceValidation<double>;

} // namespace hipdnn_gpu_ref
