// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "MiopenActivationDescriptor.hpp"
#include "MiopenTensor.hpp"

namespace miopen_plugin
{

// Shared execution plan for all unary pointwise activations (ReLU family, Sigmoid, Tanh, ...).
// All per-op behavior lives in MiopenActivationDescriptor / mapPointwiseModeToMiopenActivation;
// there is no activation-specific logic left to differentiate at this layer.
class MiopenUnaryActivationPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    MiopenUnaryActivationPlan(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    MiopenUnaryActivationPlan(const MiopenUnaryActivationPlan&) = delete;
    MiopenUnaryActivationPlan& operator=(const MiopenUnaryActivationPlan&) = delete;

    MiopenUnaryActivationPlan(MiopenUnaryActivationPlan&&) = delete;
    MiopenUnaryActivationPlan& operator=(MiopenUnaryActivationPlan&&) = delete;

    ~MiopenUnaryActivationPlan() override = default;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    MiopenTensor _input;
    MiopenTensor _output;
    MiopenActivationDescriptor _activation;
};

} // namespace miopen_plugin
