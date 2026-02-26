// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelUtils.hpp"
#include "HipdnnHipKernelHandle.hpp"
#include "HipdnnHipKernelSettings.hpp"

namespace hip_kernel_plugin
{

class BatchnormFwdInferenceParams
{
public:
    BatchnormFwdInferenceParams(
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceParams(
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& inferenceAttributes,
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceParams(const BatchnormFwdInferenceParams&) = delete;
    BatchnormFwdInferenceParams& operator=(const BatchnormFwdInferenceParams&) = delete;

    BatchnormFwdInferenceParams(BatchnormFwdInferenceParams&&) = default;
    BatchnormFwdInferenceParams& operator=(BatchnormFwdInferenceParams&&) = default;

    const hipdnn_data_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* estMean() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* invVariance() const;

    const std::optional<hip_kernel_utils::ActivationParams>& optActivation() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* activationOut() const;

private:
    const hipdnn_data_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _estMean;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _invVariance;

    std::optional<hip_kernel_utils::ActivationParams> _optActivation;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _activationOut;
};

class BatchnormFwdInferencePlan : public hipdnn_plugin_sdk::IPlan<HipdnnHipKernelHandle>
{
public:
    BatchnormFwdInferencePlan(BatchnormFwdInferenceParams&& inferenceParams,
                              const HipdnnHipKernelSettings& executionSettings);

    BatchnormFwdInferencePlan(const BatchnormFwdInferencePlan&) = delete;
    BatchnormFwdInferencePlan& operator=(const BatchnormFwdInferencePlan&) = delete;

    BatchnormFwdInferencePlan(BatchnormFwdInferencePlan&&) = default;
    BatchnormFwdInferencePlan& operator=(BatchnormFwdInferencePlan&&) = default;

    size_t getWorkspaceSize(const HipdnnHipKernelHandle& handle) const override;

    void execute(const HipdnnHipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdInferenceParams _inferenceParams;
    HipdnnHipKernelSettings _executionSettings;
};

}
