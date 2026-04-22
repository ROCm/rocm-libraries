// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelHandle.hpp"
#include "hip/ICompiledProgram.hpp"
#include "hip/IRunnableKernel.hpp"

#include <memory>

namespace hip_kernel_provider
{

class IKernelCompiler;

namespace layernorm
{

class LayernormFwdParams
{
public:
    LayernormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    LayernormFwdParams(const LayernormFwdParams&) = delete;
    LayernormFwdParams& operator=(const LayernormFwdParams&) = delete;

    LayernormFwdParams(LayernormFwdParams&&) = default;
    LayernormFwdParams& operator=(LayernormFwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* mean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVariance() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* epsilon() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _mean;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invVariance;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _epsilon;
};

class LayernormFwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    explicit LayernormFwdPlan(LayernormFwdParams&& params);

    LayernormFwdPlan(const LayernormFwdPlan&) = delete;
    LayernormFwdPlan& operator=(const LayernormFwdPlan&) = delete;

    LayernormFwdPlan(LayernormFwdPlan&&) = default;
    LayernormFwdPlan& operator=(LayernormFwdPlan&&) = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    LayernormFwdParams _params;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _runnableKernel;
};

} // namespace layernorm

} // namespace hip_kernel_provider
