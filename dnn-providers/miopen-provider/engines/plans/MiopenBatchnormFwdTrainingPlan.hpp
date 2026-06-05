// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "HipdnnMiopenSettings.hpp"
#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <optional>

namespace miopen_plugin
{

class BatchnormFwdTrainingParams
{
public:
    BatchnormFwdTrainingParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdTrainingParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdTrainingParams(const BatchnormFwdTrainingParams&) = delete;
    BatchnormFwdTrainingParams& operator=(const BatchnormFwdTrainingParams&) = delete;

    BatchnormFwdTrainingParams(BatchnormFwdTrainingParams&&) = default;
    BatchnormFwdTrainingParams& operator=(BatchnormFwdTrainingParams&&) = default;

    const MiopenTensor& x() const;
    const MiopenTensor& y() const;
    const MiopenTensor& scale() const;
    const MiopenTensor& bias() const;
    double epsilonValue() const;

    bool hasSaveMeanVariance() const;
    const MiopenTensor& mean() const;
    const MiopenTensor& invVariance() const;

    bool hasRunningStats() const;
    const MiopenTensor& prevRunningMean() const;
    const MiopenTensor& prevRunningVariance() const;
    double momentumValue() const;
    const MiopenTensor& nextRunningMean() const;
    const MiopenTensor& nextRunningVariance() const;

    const std::optional<miopen_utils::ActivationParams>& optActivation() const;
    const std::optional<MiopenTensor>& activationOut() const;

private:
    MiopenTensor _x;
    MiopenTensor _y;
    MiopenTensor _scale;
    MiopenTensor _bias;
    double _epsilonValue;

    // Optional save mean/variance
    std::optional<MiopenTensor> _mean;
    std::optional<MiopenTensor> _invVariance;

    // Optional running statistics
    std::optional<MiopenTensor> _prevRunningMean;
    std::optional<MiopenTensor> _prevRunningVariance;
    std::optional<double> _momentumValue;
    std::optional<MiopenTensor> _nextRunningMean;
    std::optional<MiopenTensor> _nextRunningVariance;
    bool _hasRunningStats{false};

    // Optional activation fusion
    std::optional<miopen_utils::ActivationParams> _optActivation;
    std::optional<MiopenTensor> _activationOut;
};

class BatchnormFwdTrainingPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams,
                             const HipdnnMiopenSettings& executionSettings);

    BatchnormFwdTrainingPlan(const BatchnormFwdTrainingPlan&) = delete;
    BatchnormFwdTrainingPlan& operator=(const BatchnormFwdTrainingPlan&) = delete;

    BatchnormFwdTrainingPlan(BatchnormFwdTrainingPlan&&) = default;
    BatchnormFwdTrainingPlan& operator=(BatchnormFwdTrainingPlan&&) = default;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdTrainingParams _trainingParams;
    HipdnnMiopenSettings _executionSettings;
};

}
