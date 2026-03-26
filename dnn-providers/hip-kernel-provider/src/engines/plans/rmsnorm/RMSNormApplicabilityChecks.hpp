// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider
{

// --- Tensor Descriptor Value Object ---

// TODO Split into own header? Reuse BN validation code
struct RMSnormTensorDescriptor
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    std::vector<int64_t> strideOrder;

    explicit RMSnormTensorDescriptor(const hipdnn_data_sdk::data_objects::TensorAttributes* attr);

    size_t numDims() const
    {
        return dims.size();
    }
    bool isPacked() const;
};

// --- High-Level Configuration Validators ---
void checkRMSnormBwdTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::RMSNormBackwardAttributes& rmsNormBwdAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace hip_kernel_provider
