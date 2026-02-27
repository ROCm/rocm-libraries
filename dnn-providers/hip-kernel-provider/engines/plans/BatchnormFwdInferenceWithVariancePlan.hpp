// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "HipKernelUtils.hpp"
#include "HipdnnHipKernelHandle.hpp"
#include "HipdnnHipKernelSettings.hpp"

namespace hip_kernel_plugin
{

class BatchnormFwdInferenceWithVarianceParams
{
public:
    BatchnormFwdInferenceWithVarianceParams(
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceWithVarianceParams(
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            inferenceAttributes,
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceWithVarianceParams(const BatchnormFwdInferenceWithVarianceParams&)
        = delete;
    BatchnormFwdInferenceWithVarianceParams&
        operator=(const BatchnormFwdInferenceWithVarianceParams&)
        = delete;

    BatchnormFwdInferenceWithVarianceParams(BatchnormFwdInferenceWithVarianceParams&&) = default;
    BatchnormFwdInferenceWithVarianceParams& operator=(BatchnormFwdInferenceWithVarianceParams&&)
        = default;

    const hipdnn_data_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* estMean() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* estVariance() const;
    double epsilonValue() const;

    const std::optional<hip_kernel_utils::ActivationParams>& optActivation() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* activationOut() const;

private:
    const hipdnn_data_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _estMean;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _estVariance;
    double _epsilonValue;

    std::optional<hip_kernel_utils::ActivationParams> _optActivation;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _activationOut;
};

class BatchnormFwdInferenceWithVariancePlan : public hipdnn_plugin_sdk::IPlan<HipdnnHipKernelHandle>
{
public:
    BatchnormFwdInferenceWithVariancePlan(BatchnormFwdInferenceWithVarianceParams&& inferenceParams,
                                          const HipdnnHipKernelSettings& executionSettings);

    BatchnormFwdInferenceWithVariancePlan(const BatchnormFwdInferenceWithVariancePlan&) = delete;
    BatchnormFwdInferenceWithVariancePlan& operator=(const BatchnormFwdInferenceWithVariancePlan&)
        = delete;

    BatchnormFwdInferenceWithVariancePlan(BatchnormFwdInferenceWithVariancePlan&&) = default;
    BatchnormFwdInferenceWithVariancePlan& operator=(BatchnormFwdInferenceWithVariancePlan&&)
        = default;

    size_t getWorkspaceSize(const HipdnnHipKernelHandle& handle) const override;

    void execute(const HipdnnHipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdInferenceWithVarianceParams _inferenceParams;
    HipdnnHipKernelSettings _executionSettings;
};

}
