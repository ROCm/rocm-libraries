// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "core/Utils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/PlanUtils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/rmsnorm/RMSnormCommon.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider
{

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

    const std::optional<core::utils::ActivationParams>& optActivation() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dy;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invRMS;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dx;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dscale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _dbias;

    std::optional<core::utils::ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y = nullptr;
};

/// The compile options and launches of the RMSNormBwd program for @p params on
/// @p gcnArchName: RMSnormBwdData, then RMSnormBwdScaleBias, both in workgroups of @p config's
/// size. A fused activation's backward is compiled in; its alpha and beta are launch
/// arguments.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 4D or 5D, or a grid
///         that does not fit one launch.
RMSnormKernelBuild buildRMSnormBwdKernel(const RMSnormBwdParams& params,
                                         const std::string& gcnArchName,
                                         const RMSnormConfig& config);

/// Whether the backward launches of @p config can run @p params: a supported workgroup
/// (isLocalSizeSupported()) whose launches stay below the 2^32 threads HIP accepts.
/// checkLaunchShapesFitThreadLimit() checks the default's.
bool isRMSnormBwdConfigApplicable(const RMSnormBwdParams& params, const RMSnormConfig& config);

/// Every kernel launch of the RMSnorm backward for a problem of the given outer size, stride
/// and inner size, in workgroups of @p localSize threads.
std::vector<KernelLaunchShape>
    rmsnormBwdLaunchShapes(int64_t outerSize, int64_t stride, int64_t innerSize, int64_t localSize);

} // namespace rmsnorm
} // namespace hip_kernel_provider
