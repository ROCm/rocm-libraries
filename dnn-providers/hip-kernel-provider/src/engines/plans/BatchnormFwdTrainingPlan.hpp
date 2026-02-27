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

class BatchnormFwdTrainingParams
{
public:
    BatchnormFwdTrainingParams(
        const hipdnn_data_sdk::data_objects::BatchnormAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdTrainingParams(
        const hipdnn_data_sdk::data_objects::BatchnormAttributes& attributes,
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdTrainingParams(const BatchnormFwdTrainingParams&) = delete;
    BatchnormFwdTrainingParams& operator=(const BatchnormFwdTrainingParams&) = delete;

    BatchnormFwdTrainingParams(BatchnormFwdTrainingParams&&) = default;
    BatchnormFwdTrainingParams& operator=(BatchnormFwdTrainingParams&&) = default;

    const hipdnn_data_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* bias() const;
    double epsilonValue() const;

    bool hasSaveMeanVariance() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* mean() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* invVariance() const;

    bool hasRunningStats() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* prevRunningMean() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* prevRunningVariance() const;
    double momentumValue() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* nextRunningMean() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* nextRunningVariance() const;

    const std::optional<hip_kernel_utils::ActivationParams>& optActivation() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* activationOut() const;

private:
    const hipdnn_data_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _bias;
    double _epsilonValue;

    // Optional save mean/variance
    std::optional<const hipdnn_data_sdk::data_objects::TensorAttributes*> _mean;
    std::optional<const hipdnn_data_sdk::data_objects::TensorAttributes*> _invVariance;

    // Optional running statistics
    std::optional<const hipdnn_data_sdk::data_objects::TensorAttributes*> _prevRunningMean;
    std::optional<const hipdnn_data_sdk::data_objects::TensorAttributes*> _prevRunningVariance;
    std::optional<double> _momentumValue;
    std::optional<const hipdnn_data_sdk::data_objects::TensorAttributes*> _nextRunningMean;
    std::optional<const hipdnn_data_sdk::data_objects::TensorAttributes*> _nextRunningVariance;
    bool _hasRunningStats{false};

    // Optional activation fusion
    std::optional<hip_kernel_utils::ActivationParams> _optActivation;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _activationOut;
};

class BatchnormFwdTrainingPlan : public hipdnn_plugin_sdk::IPlan<HipdnnHipKernelHandle>
{
public:
    BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams,
                             const HipdnnHipKernelSettings& executionSettings);

    BatchnormFwdTrainingPlan(const BatchnormFwdTrainingPlan&) = delete;
    BatchnormFwdTrainingPlan& operator=(const BatchnormFwdTrainingPlan&) = delete;

    BatchnormFwdTrainingPlan(BatchnormFwdTrainingPlan&&) = default;
    BatchnormFwdTrainingPlan& operator=(BatchnormFwdTrainingPlan&&) = default;

    size_t getWorkspaceSize(const HipdnnHipKernelHandle& handle) const override;

    void execute(const HipdnnHipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdTrainingParams _trainingParams;
    HipdnnHipKernelSettings _executionSettings;
};

}
