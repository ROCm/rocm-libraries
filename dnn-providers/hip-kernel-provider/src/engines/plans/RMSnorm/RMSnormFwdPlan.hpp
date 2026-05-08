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

namespace rmsnorm
{

class RMSnormFwdParams
{
public:
    RMSnormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    RMSnormFwdParams(const RMSnormFwdParams&) = delete;
    RMSnormFwdParams& operator=(const RMSnormFwdParams&) = delete;

    RMSnormFwdParams(RMSnormFwdParams&&) = default;
    RMSnormFwdParams& operator=(RMSnormFwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invRMS() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* epsilon() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invRMS;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _epsilon;
};

class RMSnormFwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    explicit RMSnormFwdPlan(RMSnormFwdParams&& params);

    RMSnormFwdPlan(const RMSnormFwdPlan&) = delete;
    RMSnormFwdPlan& operator=(const RMSnormFwdPlan&) = delete;

    RMSnormFwdPlan(RMSnormFwdPlan&&) = default;
    RMSnormFwdPlan& operator=(RMSnormFwdPlan&&) = delete;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    int64_t getOuterSize(unsigned normalizeDim) const;
    int64_t getInnerSize(unsigned normalizeDim) const;
    unsigned getNormalizeDim() const;

    RMSnormFwdParams _params;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _runnableKernel;
};

}
}
