// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "MiopenTensor.hpp"
#include "MiopenUtils.hpp"
#include "PlanBuilderInterface.hpp"
#include "PlanInterface.hpp"
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <optional>

namespace miopen_legacy_plugin
{

class BatchnormFwdTrainingActivParams
{
public:
    BatchnormFwdTrainingActivParams(
        const hipdnn_sdk::data_objects::BatchnormAttributes& bnAttr,
        const hipdnn_sdk::data_objects::PointwiseAttributes& activAttr,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdTrainingActivParams(const BatchnormFwdTrainingActivParams&) = delete;
    BatchnormFwdTrainingActivParams& operator=(const BatchnormFwdTrainingActivParams&) = delete;

    BatchnormFwdTrainingActivParams(BatchnormFwdTrainingActivParams&&) = default;
    BatchnormFwdTrainingActivParams& operator=(BatchnormFwdTrainingActivParams&&) = default;

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

    const miopen_utils::ActivationParams& activParams() const;

private:
    MiopenTensor _x;
    MiopenTensor _y;
    MiopenTensor _scale;
    MiopenTensor _bias;
    double _epsilonValue;
    miopen_utils::ActivationParams _activParams;

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
};

class BatchnormFwdTrainingActivPlan : public IPlan
{
public:
    BatchnormFwdTrainingActivPlan(const HipdnnEnginePluginHandle& handle,
                                  BatchnormFwdTrainingActivParams&& params);

    BatchnormFwdTrainingActivPlan(const BatchnormFwdTrainingActivPlan&) = delete;
    BatchnormFwdTrainingActivPlan& operator=(const BatchnormFwdTrainingActivPlan&) = delete;

    BatchnormFwdTrainingActivPlan(BatchnormFwdTrainingActivPlan&&) = default;
    BatchnormFwdTrainingActivPlan& operator=(BatchnormFwdTrainingActivPlan&&) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdTrainingActivParams _params;
};

}
