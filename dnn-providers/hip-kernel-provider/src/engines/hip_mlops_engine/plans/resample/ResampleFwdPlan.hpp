// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"
#include "core/Handle.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider
{
using namespace compilation;

namespace resample
{

class ResampleFwdParams
{
public:
    ResampleFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType);

    ResampleFwdParams(const ResampleFwdParams&) = delete;
    ResampleFwdParams& operator=(const ResampleFwdParams&) = delete;

    ResampleFwdParams(ResampleFwdParams&&) = default;
    ResampleFwdParams& operator=(ResampleFwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* index() const;
    const std::vector<int64_t>& prePadding() const;
    const std::vector<int64_t>& postPadding() const;
    const std::vector<int64_t>& stride() const;
    const std::vector<int64_t>& window() const;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode mode() const;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode() const;
    bool generateIndex() const;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _index;
    std::vector<int64_t> _prePadding;
    std::vector<int64_t> _postPadding;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _window;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode _mode;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode _paddingMode;
    hipdnn_flatbuffers_sdk::data_objects::DataType _computeDataType;
};

class ResampleFwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    explicit ResampleFwdPlan(ResampleFwdParams&& params);

    ResampleFwdPlan(const ResampleFwdPlan&) = delete;
    ResampleFwdPlan& operator=(const ResampleFwdPlan&) = delete;

    ResampleFwdPlan(ResampleFwdPlan&&) = default;
    ResampleFwdPlan& operator=(ResampleFwdPlan&&) = delete;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    uint64_t outputElementCount() const;

    ResampleFwdParams _params;
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _runnableKernel;
};

} // namespace resample
} // namespace hip_kernel_provider
