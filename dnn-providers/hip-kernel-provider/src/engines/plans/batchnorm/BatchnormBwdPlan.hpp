// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelHandle.hpp"
#include "HipKernelUtils.hpp"
#include "hip/ICompiledProgram.hpp"
#include "hip/IRunnableKernel.hpp"

namespace hip_kernel_provider
{

class IKernelCompiler;

namespace batchnorm
{

class BatchnormBwdParams
{
public:
    BatchnormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes&
            batchnormBackwardAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes&
            batchnormInferenceAttributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormBwdParams(const BatchnormBwdParams&) = delete;
    BatchnormBwdParams& operator=(const BatchnormBwdParams&) = delete;

    BatchnormBwdParams(BatchnormBwdParams&&) = default;
    BatchnormBwdParams& operator=(BatchnormBwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dy() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dx() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dscale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dbias() const;

    bool hasSavedStats() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* savedMean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* savedInvVariance() const;

    const std::optional<hip_kernel_utils::ActivationParams>& optActivation() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dy;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dx;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dscale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dbias;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _savedMean = nullptr;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _savedInvVariance = nullptr;

    std::optional<hip_kernel_utils::ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias = nullptr;
};

class BatchnormBwdPlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    explicit BatchnormBwdPlan(BatchnormBwdParams&& params);

    BatchnormBwdPlan(const BatchnormBwdPlan&) = delete;
    BatchnormBwdPlan& operator=(const BatchnormBwdPlan&) = delete;

    BatchnormBwdPlan(BatchnormBwdPlan&&) = default;
    BatchnormBwdPlan& operator=(BatchnormBwdPlan&&) = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    void compileSpatial(const IKernelCompiler& kernelCompiler,
                        const hipDeviceProp_t& deviceProperties,
                        const struct ProblemDims& dims);

    void executeSpatial(const HipKernelHandle& handle,
                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    void executePerActivation(const HipKernelHandle& handle,
                              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              uint32_t numDeviceBuffers) const;

    BatchnormBwdParams _params;

    std::unique_ptr<ICompiledProgram> _compiledProgram;
    std::vector<std::unique_ptr<IRunnableKernel>> _runnableKernels;

    bool _isSpatialMode = true;
    bool _usesSavedStats = false;
    int _kernelVariant = -1;
    double _epsilon = 0.0;
    float _invInNhw = 0.0f;
    unsigned int _batchCount = 0;
    unsigned int _batchStride = 0;
    unsigned int _inCstride = 0;
    float _activationAlpha = 0.0f;
    float _activationBeta = 0.0f;
};

} // namespace batchnorm

} // namespace hip_kernel_provider
