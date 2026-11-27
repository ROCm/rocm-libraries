// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
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

    const std::optional<core::utils::ActivationParams>& optActivation() const;
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

    std::optional<core::utils::ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias = nullptr;
};

/// The variant, workgroup size and vector size the launch heuristics choose for @p params.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
BatchnormSpatialConfig defaultBatchnormBwdConfig(const BatchnormBwdParams& params);

/// Whether @p config can run @p params (isSpatialConfigApplicable()).
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
bool isBatchnormBwdConfigApplicable(const BatchnormBwdParams& params,
                                    const BatchnormSpatialConfig& config);

/// The compile options, variant and launches of the BatchNormBwdSpatial program for @p params
/// on @p gcnArchName with the variant, workgroup size and vector size of @p config: one launch,
/// or for variant 2 five launches, of which a run with saved statistics makes the first, second
/// and fifth and one without makes all five, stashing in dx.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 3D, 4D or 5D.
BatchnormKernelBuild buildBatchnormBwdKernel(const BatchnormBwdParams& params,
                                             const std::string& gcnArchName,
                                             const BatchnormSpatialConfig& config);

} // namespace batchnorm

} // namespace hip_kernel_provider
