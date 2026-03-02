// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "HipKernelUtils.hpp"
#include "PlanInterface.hpp"

namespace hip_kernel_plugin
{

class BatchnormBwdParams
{
public:
    BatchnormBwdParams(
        const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormBwdParams(
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& inferenceAttributes,
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const hipdnn_data_sdk::data_objects::BatchnormBackwardAttributes& backwardAttributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormBwdParams(const BatchnormBwdParams&) = delete;
    BatchnormBwdParams& operator=(const BatchnormBwdParams&) = delete;

    BatchnormBwdParams(BatchnormBwdParams&&) = default;
    BatchnormBwdParams& operator=(BatchnormBwdParams&&) = default;

    const hipdnn_data_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* dy() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* dx() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* dscale() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* dbias() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* savedMean() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* savedInvVariance() const;

    const hipdnn_data_sdk::data_objects::TensorAttributes* bias() const;
    const std::optional<hip_kernel_utils::ActivationParams>& optActivation() const;

private:
    const hipdnn_data_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _dy;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _dx;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _dscale;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _dbias;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _savedMean;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _savedInvVariance;

    const hipdnn_data_sdk::data_objects::TensorAttributes* _bias;
    std::optional<hip_kernel_utils::ActivationParams> _optActivation;
};

class BatchnormBwdPlan : public IPlan
{
public:
    BatchnormBwdPlan(BatchnormBwdParams&& bwdParams, bool benchmarkingEnabled = false);

    BatchnormBwdPlan(const BatchnormBwdPlan&) = delete;
    BatchnormBwdPlan& operator=(const BatchnormBwdPlan&) = delete;

    BatchnormBwdPlan(BatchnormBwdPlan&&) = default;
    BatchnormBwdPlan& operator=(BatchnormBwdPlan&&) = default;

    size_t getWorkspaceSize(const HipdnnEnginePluginHandle& handle) const override;

    void execute(const HipdnnEnginePluginHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    void executeSpatial(const HipdnnEnginePluginHandle& handle,
                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    void executePerActivation(const HipdnnEnginePluginHandle& handle,
                              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              uint32_t numDeviceBuffers) const;

    bool isSpatialMode() const;

    BatchnormBwdParams _bwdParams;
    [[maybe_unused]] bool _benchmarkingEnabled;
};

}
