// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
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

class RMSnormFwdParams
{
public:
    RMSnormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    RMSnormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& attributes,
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
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
    double epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    const std::optional<core::utils::ActivationParams>& optActivation() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* activationOut() const;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invRMS;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;

    std::optional<core::utils::ActivationParams> _optActivation;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _activationOut;
};

/// The compile options and single launch of the RMSnormFwd kernel for @p params on
/// @p gcnArchName, in workgroups of @p config's size. A fused activation is compiled in; its
/// alpha and beta are launch arguments.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 4D or 5D, or a grid
///         that does not fit one launch.
RMSnormKernelBuild buildRMSnormFwdKernel(const RMSnormFwdParams& params,
                                         const std::string& gcnArchName,
                                         const RMSnormConfig& config);

/// Whether the forward launch of @p config can run @p params: a supported workgroup
/// (isLocalSizeSupported()) whose launch stays below the 2^32 threads HIP accepts.
/// checkLaunchShapesFitThreadLimit() checks the default's.
bool isRMSnormFwdConfigApplicable(const RMSnormFwdParams& params, const RMSnormConfig& config);

/// Every kernel launch of the RMSnorm forward for a problem of the given outer size and stride,
/// in workgroups of @p localSize threads.
std::vector<KernelLaunchShape>
    rmsnormFwdLaunchShapes(int64_t outerSize, int64_t stride, int64_t localSize);

} // namespace rmsnorm

} // namespace hip_kernel_provider
