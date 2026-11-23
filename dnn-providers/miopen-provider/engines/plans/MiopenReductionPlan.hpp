// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "MiopenReduceTensorDescriptor.hpp"
#include "MiopenTensor.hpp"

namespace miopen_plugin
{

// Execution plan for a single reduction node, backed by MIOpen's legacy miopenReduceTensor.
// The applicability gate lives in MiopenReductionChecks; this plan assumes it has passed.
class MiopenReductionPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    // Queries the workspace size, which miopenGetReductionWorkspaceSize needs a handle for.
    MiopenReductionPlan(
        const HipdnnMiopenHandle& handle,
        const hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    MiopenReductionPlan(const MiopenReductionPlan&) = delete;
    MiopenReductionPlan& operator=(const MiopenReductionPlan&) = delete;

    MiopenReductionPlan(MiopenReductionPlan&&) = delete;
    MiopenReductionPlan& operator=(MiopenReductionPlan&&) = delete;

    ~MiopenReductionPlan() override = default;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    MiopenTensor _input;
    MiopenTensor _output;
    MiopenReduceTensorDescriptor _reduceDescriptor;
    // Bytes from the first to one past the last element, used to reject overlapping buffers.
    size_t _inputSpanBytes;
    size_t _outputSpanBytes;
    size_t _workspaceSize{0};
};

} // namespace miopen_plugin
