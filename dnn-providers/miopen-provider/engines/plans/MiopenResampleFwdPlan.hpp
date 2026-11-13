// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipdnnMiopenHandle.hpp"
#include "MiopenPoolingDescriptor.hpp"
#include "MiopenTensor.hpp"
#include "engines/plans/MiopenResampleFwdChecks.hpp"

namespace miopen_plugin
{

// Execution plan for a standalone resample forward (pooling) node, backed by
// miopenPoolingForward.
//
// The workspace is MIOpen's own (miopenPoolingGetWorkSpaceSizeV2): room for the indices plus,
// for channels-last tensors, the transposed copies of X and Y. When the node generates an
// index, MIOpen leaves it in the first yElements * sizeof(index) bytes of the workspace and the
// plan copies it into the index tensor.
class MiopenResampleFwdPlan : public hipdnn_plugin_sdk::IPlan<HipdnnMiopenHandle>
{
public:
    explicit MiopenResampleFwdPlan(const resample_fwd_applicability::ResampleFwdConfig& config);

    MiopenResampleFwdPlan(const MiopenResampleFwdPlan&) = delete;
    MiopenResampleFwdPlan& operator=(const MiopenResampleFwdPlan&) = delete;

    MiopenResampleFwdPlan(MiopenResampleFwdPlan&&) = delete;
    MiopenResampleFwdPlan& operator=(MiopenResampleFwdPlan&&) = delete;

    ~MiopenResampleFwdPlan() override = default;

    size_t getWorkspaceSize(const HipdnnMiopenHandle& handle) const override;

    void execute(const HipdnnMiopenHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    MiopenTensor _x;
    MiopenTensor _y;
    MiopenPoolingDescriptor _pooling;
    std::optional<int64_t> _indexUid;
    size_t _indexBytes{0};
    size_t _workspaceSize{0};
};

} // namespace miopen_plugin
