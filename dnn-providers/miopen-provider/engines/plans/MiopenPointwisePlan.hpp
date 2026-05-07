// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>

#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "MiopenActivationDescriptor.hpp"
#include "MiopenTensor.hpp"

namespace miopen_plugin
{

class MiopenPointwisePlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    MiopenPointwisePlan(
        const hipdnn_data_sdk::data_objects::PointwiseAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    MiopenPointwisePlan(const MiopenPointwisePlan&) = delete;
    MiopenPointwisePlan& operator=(const MiopenPointwisePlan&) = delete;

    MiopenPointwisePlan(MiopenPointwisePlan&&) = delete;
    MiopenPointwisePlan& operator=(MiopenPointwisePlan&&) = delete;

    ~MiopenPointwisePlan() override = default;

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
