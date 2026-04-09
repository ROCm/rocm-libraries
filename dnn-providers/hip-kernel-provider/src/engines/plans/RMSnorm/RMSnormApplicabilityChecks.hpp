// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider::rmsnorm
{

// --- Tensor Descriptor Value Object ---

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
void checkRMSnormTensorConfigSupported(
    const hipdnn_data_sdk::data_objects::RMSNormAttributes& rmsNormAttr,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap);

} // namespace hip_kernel_provider::rmsnorm
