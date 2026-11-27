// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

#include "core/Utils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/PlanUtils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormUtilities.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider
{

using namespace compilation;

namespace layernorm
{

class LayernormFwdParams
{
public:
    LayernormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& attributes,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap);

    LayernormFwdParams(const LayernormFwdParams&) = delete;
    LayernormFwdParams& operator=(const LayernormFwdParams&) = delete;

    LayernormFwdParams(LayernormFwdParams&&) = default;
    LayernormFwdParams& operator=(LayernormFwdParams&&) = default;

    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* x() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* y() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scale() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* bias() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* mean() const;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVariance() const;
    double epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                        uint32_t numDeviceBuffers) const;

    static constexpr int64_t LOCAL_SIZE = 1024;

private:
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _x;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _y;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _scale;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _bias;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _mean;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* _invVariance;
    hipdnn_plugin_sdk::ScalarOperand _epsilon;
};

/// The default configuration: LOCAL_SIZE threads per row.
LayernormFwdConfig defaultLayernormFwdConfig();

/// The launch of a problem of the given outer size and stride, a workgroup of @p localSize
/// threads per row.
std::vector<KernelLaunchShape>
    layernormFwdLaunchShapes(int64_t outerSize, int64_t stride, int64_t localSize);

/// The launch buildLayernormFwdKernel() makes for @p params with @p config (the overload
/// above at the config's local size).
std::vector<KernelLaunchShape> layernormFwdLaunchShapes(const LayernormFwdParams& params,
                                                        const LayernormFwdConfig& config);

/// The compile options and single launch of the LayernormFwd kernel for @p params on
/// @p gcnArchName, with the workgroup of @p config.
///
/// @throws HipdnnPluginException with BAD_PARAM for a tensor that is not 4D or 5D, or a grid
///         that does not fit one launch.
LayernormKernelBuild buildLayernormFwdKernel(const LayernormFwdParams& params,
                                             const std::string& gcnArchName,
                                             const LayernormFwdConfig& config);

} // namespace layernorm

} // namespace hip_kernel_provider
