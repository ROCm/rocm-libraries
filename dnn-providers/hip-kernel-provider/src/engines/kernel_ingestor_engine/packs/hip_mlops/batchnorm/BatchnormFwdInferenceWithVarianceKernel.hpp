// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
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
    double epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    const std::optional<core::utils::ActivationParams>& optActivation() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* activationOut() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _estMean;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _estVariance;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;

    std::optional<core::utils::ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _activationOut;
};

/// The vector size the inference kernel defaults to for @p params: the largest of 4, 2 and 1
/// that divides the vectorized dimension.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
BatchnormInferenceConfig defaultBatchnormFwdInferenceWithVarianceConfig(
    const BatchnormFwdInferenceWithVarianceParams& params);

/// Whether @p config can run @p params (isInferenceConfigApplicable()).
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
bool isBatchnormFwdInferenceWithVarianceConfigApplicable(
    const BatchnormFwdInferenceWithVarianceParams& params, const BatchnormInferenceConfig& config);

/// The compile options, entry point, 3D geometry and launch strides of the BatchNormFwdInferSpatialEst
/// kernel for @p params on @p gcnArchName with the vector size of @p config. The batch
/// dimension's grid fills the device: @p multiProcessorCount CUs of @p warpSize lanes.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
BatchnormInferenceKernelBuild buildBatchnormFwdInferenceWithVarianceKernel(
    const BatchnormFwdInferenceWithVarianceParams& params,
    const std::string& gcnArchName,
    int multiProcessorCount,
    int warpSize,
    const BatchnormInferenceConfig& config);

} // namespace batchnorm

} // namespace hip_kernel_provider
