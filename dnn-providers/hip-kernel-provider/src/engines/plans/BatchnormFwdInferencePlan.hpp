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

class BatchnormFwdInferenceParams
{
public:
    BatchnormFwdInferenceParams(
        const hipdnn_data_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceParams(const BatchnormFwdInferenceParams&) = delete;
    BatchnormFwdInferenceParams& operator=(const BatchnormFwdInferenceParams&) = delete;

    BatchnormFwdInferenceParams(BatchnormFwdInferenceParams&&) = default;
    BatchnormFwdInferenceParams& operator=(BatchnormFwdInferenceParams&&) = default;

    const hipdnn_data_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* estMean() const;
    const hipdnn_data_sdk::data_objects::TensorAttributes* invVariance() const;

private:
    const hipdnn_data_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _estMean;
    const hipdnn_data_sdk::data_objects::TensorAttributes* _invVariance;
};

class BatchnormFwdInferencePlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    explicit BatchnormFwdInferencePlan(BatchnormFwdInferenceParams&& inferenceParams);

    BatchnormFwdInferencePlan(const BatchnormFwdInferencePlan&) = delete;
    BatchnormFwdInferencePlan& operator=(const BatchnormFwdInferencePlan&) = delete;

    BatchnormFwdInferencePlan(BatchnormFwdInferencePlan&&) = default;
    BatchnormFwdInferencePlan& operator=(BatchnormFwdInferencePlan&&) = delete;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdInferenceParams _inferenceParams;

    // Populated by compile()
    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::unique_ptr<IRunnableKernel> _runnableKernel;

    // Kernel launch parameters computed during compile()
    unsigned int _channels = 0;
    unsigned int _inCstride = 0;
    unsigned int _batchCount = 0;
    unsigned int _cStride = 0;
    unsigned int _hwStride = 0;
    unsigned int _batchStride = 0;
};

}
