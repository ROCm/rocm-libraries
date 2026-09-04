// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "compilation/IRunnableKernel.hpp"
#include "core/Handle.hpp"
#include "core/Utils.hpp"

#include <memory>

namespace hip_kernel_provider
{

using namespace core::utils;
using namespace compilation;

namespace rmsnorm
{

class RMSnormBwdParams
{
public:
    RMSnormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);
    RMSnormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& attributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    RMSnormBwdParams(const RMSnormBwdParams&) = delete;
    RMSnormBwdParams& operator=(const RMSnormBwdParams&) = delete;

    RMSnormBwdParams(RMSnormBwdParams&&) = default;
    RMSnormBwdParams& operator=(RMSnormBwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dy() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invRMS() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dx() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dscale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dbias() const;

    const std::optional<ActivationParams>& optActivation() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dy;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invRMS;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dx;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dscale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dbias;

    std::optional<ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y = nullptr;
};

class RMSnormBwdPlan : public hipdnn_plugin_sdk::IPlan<Handle>
{
public:
    explicit RMSnormBwdPlan(RMSnormBwdParams&& params);

    RMSnormBwdPlan(const RMSnormBwdPlan&) = delete;
    RMSnormBwdPlan& operator=(const RMSnormBwdPlan&) = delete;

    RMSnormBwdPlan(RMSnormBwdPlan&&) = default;
    RMSnormBwdPlan& operator=(RMSnormBwdPlan&&) = delete;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const Handle& handle) const override;

    void execute(const Handle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    RMSnormBwdParams _params;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::vector<std::unique_ptr<IRunnableKernel>> _runnableKernels;

    float _activationAlpha = 0.0f;
    float _activationBeta = 0.0f;
};

} // namespace hip_kernel_provider::rmsnorm
} // namespace hip_kernel_provider
