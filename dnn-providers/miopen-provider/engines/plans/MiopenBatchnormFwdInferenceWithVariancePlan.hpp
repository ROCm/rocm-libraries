// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "MiopenActivationDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

class BatchnormFwdInferenceWithVarianceParams
{
public:
    BatchnormFwdInferenceWithVarianceParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceWithVarianceParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            inferenceAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceWithVarianceParams(const BatchnormFwdInferenceWithVarianceParams&)
        = delete;
    BatchnormFwdInferenceWithVarianceParams&
        operator=(const BatchnormFwdInferenceWithVarianceParams&)
        = delete;

    BatchnormFwdInferenceWithVarianceParams(BatchnormFwdInferenceWithVarianceParams&&) = default;
    BatchnormFwdInferenceWithVarianceParams& operator=(BatchnormFwdInferenceWithVarianceParams&&)
        = default;

    const MiopenTensor& x() const;
    const MiopenTensor& y() const;
    const MiopenTensor& scale() const;
    const MiopenTensor& bias() const;
    const MiopenTensor& estMean() const;
    const MiopenTensor& variance() const;
    double epsilonValue() const;

    const std::optional<MiopenActivationDescriptor>& optActivation() const;
    const std::optional<MiopenTensor>& activationOut() const;

private:
    MiopenTensor _x;
    MiopenTensor _y;
    MiopenTensor _scale;
    MiopenTensor _bias;
    MiopenTensor _estMean;
    MiopenTensor _variance;
    double _epsilonValue;

    std::optional<MiopenActivationDescriptor> _optActivation;
    std::optional<MiopenTensor> _activationOut;
};

class BatchnormFwdInferenceWithVariancePlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    BatchnormFwdInferenceWithVariancePlan(BatchnormFwdInferenceWithVarianceParams&& inferenceParams,
                                          const HipdnnMiopenSettings& executionSettings);

    BatchnormFwdInferenceWithVariancePlan(const BatchnormFwdInferenceWithVariancePlan&) = delete;
    BatchnormFwdInferenceWithVariancePlan& operator=(const BatchnormFwdInferenceWithVariancePlan&)
        = delete;

    BatchnormFwdInferenceWithVariancePlan(BatchnormFwdInferenceWithVariancePlan&&) = default;
    BatchnormFwdInferenceWithVariancePlan& operator=(BatchnormFwdInferenceWithVariancePlan&&)
        = default;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdInferenceWithVarianceParams _inferenceParams;
    HipdnnMiopenSettings _executionSettings;
};

} // namespace miopen_plugin
