// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "HipKernelHandle.hpp"
#include "HipKernelUtils.hpp"
#include "hip/ICompiledProgram.hpp"
#include "hip/IRunnableKernel.hpp"

#include <memory>

namespace hip_kernel_provider
{

class IKernelCompiler;

namespace batchnorm
{

class BatchnormFwdInferenceWithVarianceParams
{
public:
    BatchnormFwdInferenceWithVarianceParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceWithVarianceParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt&
            inferenceAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceWithVarianceParams(const BatchnormFwdInferenceWithVarianceParams&)
        = delete;
    BatchnormFwdInferenceWithVarianceParams&
        operator=(const BatchnormFwdInferenceWithVarianceParams&)
        = delete;

    BatchnormFwdInferenceWithVarianceParams(BatchnormFwdInferenceWithVarianceParams&&) = default;
    BatchnormFwdInferenceWithVarianceParams& operator=(BatchnormFwdInferenceWithVarianceParams&&)
        = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* estMean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* estVariance() const;
    double epsilonValue() const;

    const std::optional<hip_kernel_utils::ActivationParams>& optActivation() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* activationOut() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _estMean;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _estVariance;
    double _epsilonValue;

    std::optional<hip_kernel_utils::ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _activationOut;
};

class BatchnormFwdInferenceWithVariancePlan : public hipdnn_plugin_sdk::IPlan<HipKernelHandle>
{
public:
    explicit BatchnormFwdInferenceWithVariancePlan(
        BatchnormFwdInferenceWithVarianceParams&& inferenceParams);

    BatchnormFwdInferenceWithVariancePlan(const BatchnormFwdInferenceWithVariancePlan&) = delete;
    BatchnormFwdInferenceWithVariancePlan& operator=(const BatchnormFwdInferenceWithVariancePlan&)
        = delete;

    BatchnormFwdInferenceWithVariancePlan(BatchnormFwdInferenceWithVariancePlan&&) = default;
    BatchnormFwdInferenceWithVariancePlan& operator=(BatchnormFwdInferenceWithVariancePlan&&)
        = default;

    void compile(const IKernelCompiler& kernelCompiler, const hipDeviceProp_t& deviceProperties);

    size_t getWorkspaceSize(const HipKernelHandle& handle) const override;

    void execute(const HipKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    BatchnormFwdInferenceWithVarianceParams _inferenceParams;

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

} // namespace batchnorm

} // namespace hip_kernel_provider
