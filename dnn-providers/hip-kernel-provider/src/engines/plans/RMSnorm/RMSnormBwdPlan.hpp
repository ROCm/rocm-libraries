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

class RMSnormBwdParams
{
public:
    RMSnormBwdParams(
        const hipdnn_data_sdk::data_objects::RMSNormBackwardAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    RMSnormBwdParams(const RMSnormBwdParams&) = delete;
    RMSnormBwdParams& operator=(const RMSnormBwdParams&) = delete;

    RMSnormBwdParams(RMSnormBwdParams&&) = default;
    RMSnormBwdParams& operator=(RMSnormBwdParams&&) = default;

    const hipdnn_data_sdk::data_objects::TensorAttributes* dy() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* invRMS() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* dx() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* dscale() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* dbias() const;

private:
    const hipdnn_data_sdk::data_objects::TensorAttributes* _dy;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _invRMS;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _dx;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _dscale;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _dbias;
};

class RMSnormBwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    explicit RMSnormBwdPlan(RMSnormBwdParams&& params);

    RMSnormBwdPlan(const RMSnormBwdPlan&) = delete;
    RMSnormBwdPlan& operator=(const RMSnormBwdPlan&) = delete;

    RMSnormBwdPlan(RMSnormBwdPlan&&) = default;
    RMSnormBwdPlan& operator=(RMSnormBwdPlan&&) = delete;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    RMSnormBwdParams _params;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _runnableKernel;
};

} // namespace hip_kernel_provider::rmsnorm
} // namespace hip_kernel_provider
