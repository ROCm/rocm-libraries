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

namespace resample
{

class ResampleBwdParams
{
public:
    ResampleBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    ResampleBwdParams(const ResampleBwdParams&) = delete;
    ResampleBwdParams& operator=(const ResampleBwdParams&) = delete;

    ResampleBwdParams(ResampleBwdParams&&) = default;
    ResampleBwdParams& operator=(ResampleBwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dy() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dx() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* index() const;
    const flatbuffers::Vector<int64_t>* prePadding() const;
    const flatbuffers::Vector<int64_t>* postPadding() const;
    const flatbuffers::Vector<int64_t>* stride() const;
    const flatbuffers::Vector<int64_t>* window() const;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode resampleMode() const;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode() const;
    bool generateIndex() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dy;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dx;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _index;
    const flatbuffers::Vector<int64_t>* _prePadding;
    const flatbuffers::Vector<int64_t>* _postPadding;
    const flatbuffers::Vector<int64_t>* _stride;
    const flatbuffers::Vector<int64_t>* _window;
    const hipdnn_flatbuffers_sdk::data_objects::ResampleMode _resampleMode;
    const hipdnn_flatbuffers_sdk::data_objects::PaddingMode _paddingMode;
    const bool _generateIndex;
};

class ResampleBwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    explicit ResampleBwdPlan(ResampleBwdParams&& params);

    ResampleBwdPlan(const ResampleBwdPlan&) = delete;
    ResampleBwdPlan& operator=(const ResampleBwdPlan&) = delete;

    ResampleBwdPlan(ResampleBwdPlan&&) = default;
    ResampleBwdPlan& operator=(ResampleBwdPlan&&) = default;

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    ResampleBwdParams _params;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _runnableKernel;
};

} // namespace hip_kernel_provider::resample
} // namespace hip_kernel_provider
