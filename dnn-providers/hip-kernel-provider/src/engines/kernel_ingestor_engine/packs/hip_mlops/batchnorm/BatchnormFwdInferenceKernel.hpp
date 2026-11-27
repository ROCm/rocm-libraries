// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "core/Utils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/batchnorm/BatchnormCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/batchnorm/BatchnormKernelCompileOptions.hpp"

namespace hip_kernel_provider
{

using namespace compilation;

namespace batchnorm
{

class BatchnormFwdInferenceParams
{
public:
    BatchnormFwdInferenceParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceParams(
        const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes&
            inferenceAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    BatchnormFwdInferenceParams(const BatchnormFwdInferenceParams&) = delete;
    BatchnormFwdInferenceParams& operator=(const BatchnormFwdInferenceParams&) = delete;

    BatchnormFwdInferenceParams(BatchnormFwdInferenceParams&&) = default;
    BatchnormFwdInferenceParams& operator=(BatchnormFwdInferenceParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* estMean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVariance() const;

    const std::optional<core::utils::ActivationParams>& optActivation() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* activationOut() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _estMean;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invVariance;

    std::optional<core::utils::ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _activationOut;
};

/// The vector size the inference kernel defaults to for @p params: the largest of 4, 2 and 1
/// that divides the vectorized dimension.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
BatchnormInferenceConfig
    defaultBatchnormFwdInferenceConfig(const BatchnormFwdInferenceParams& params);

/// Whether @p config can run @p params (isInferenceConfigApplicable()).
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
bool isBatchnormFwdInferenceConfigApplicable(const BatchnormFwdInferenceParams& params,
                                             const BatchnormInferenceConfig& config);

/// The compile options, entry point, 3D geometry and launch strides of the BatchNormFwdInferSpatialEstInvVar
/// kernel for @p params on @p gcnArchName with the vector size of @p config. The batch
/// dimension's grid fills the device: @p multiProcessorCount CUs of @p warpSize lanes.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
BatchnormInferenceKernelBuild
    buildBatchnormFwdInferenceKernel(const BatchnormFwdInferenceParams& params,
                                     const std::string& gcnArchName,
                                     int multiProcessorCount,
                                     int warpSize,
                                     const BatchnormInferenceConfig& config);

} // namespace batchnorm

} // namespace hip_kernel_provider
