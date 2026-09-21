// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>

#include <cstdint>

namespace hipdnn_gpu_ref
{

// GPU-based integer tensor validator implementing IReferenceValidation.
// Requires exact equality between reference and implementation tensors.
// Supports both packed and strided tensors.
template <class T>
class GpuIntReferenceValidation : public hipdnn_test_sdk::utilities::IReferenceValidation
{
public:
    GpuIntReferenceValidation() = default;
    ~GpuIntReferenceValidation() override = default;

    // NOLINTNEXTLINE(portability-template-virtual-member-function) - explicit instantiation in .cpp
    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override;

private:
    bool gpuExact(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const;
};

extern template class GpuIntReferenceValidation<int8_t>;
extern template class GpuIntReferenceValidation<uint8_t>;
extern template class GpuIntReferenceValidation<int32_t>;

} // namespace hipdnn_gpu_ref
