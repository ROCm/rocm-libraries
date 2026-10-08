// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>

namespace hipdnn_gpu_ref
{

// GPU counterpart of hipdnn_test_sdk::utilities::CpuFpReferenceMiopenRmsValidation:
// MIOpen's aggregate relative-RMS check,
//   sqrt(sum((ref - impl)^2)) / (sqrt(n) * max(max|ref|, max|impl|)) <= relativeTolerance,
// failing on any NaN/Inf. The sums are reduced on the device, so only four numbers
// come back to the host. Supports both packed and strided tensors.
//
// Takes the same tolerance the CPU validator does, rounded through T the same way, so
// the two agree on a verdict wherever they run.
template <class T>
class GpuFpReferenceRmsValidation : public hipdnn_test_sdk::utilities::IReferenceValidation
{
public:
    explicit GpuFpReferenceRmsValidation(float relativeTolerance);

    ~GpuFpReferenceRmsValidation() override = default;

    // NOLINTNEXTLINE(portability-template-virtual-member-function) - explicit instantiation in .cpp
    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override;

private:
    double _relativeTolerance;
};

// Suppress implicit instantiation — definitions are in GpuFpReferenceRmsValidation.cpp
extern template class GpuFpReferenceRmsValidation<float>;
extern template class GpuFpReferenceRmsValidation<hipdnn_data_sdk::types::half>;
extern template class GpuFpReferenceRmsValidation<hipdnn_data_sdk::types::bfloat16>;
extern template class GpuFpReferenceRmsValidation<double>;

} // namespace hipdnn_gpu_ref
