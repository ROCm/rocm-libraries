// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <unordered_map>

#include <hipdnn_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

namespace miopen_legacy_plugin
{

void checkBatchnormTensorConfigSupported(
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap);

void checkBatchnormTensorConfigSupported(
        const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap);

void checkBatchnormTensorConfigSupported(
        const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap);

void checkBatchnormTensorConfigSupported(
        const hipdnn_sdk::data_objects::BatchnormInferenceAttributes& bnInfAttr,
        const hipdnn_sdk::data_objects::PointwiseAttributes& actAttr,
        const hipdnn_sdk::data_objects::BatchnormBackwardAttributes& bnBwdAttr,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap);

void checkBatchnormFwdActivationModeSupported(const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr);

void checkBatchnormBwdActivationModeSupported(const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr);

} // namespace miopen_legacy_plugin
