// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "RMSnormBwdPlan.hpp"
#include "RMSnormApplicabilityChecks.hpp"

#include "HipKernelUtils.hpp"
#include "hip/IKernelCompiler.hpp"

#include <cstdint>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::rmsnorm
{

RMSnormBwdParams::RMSnormBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _dy(tensorMap.at(attributes.dy_tensor_uid()))
    , _x(tensorMap.at(attributes.x_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _invRMS(tensorMap.at(attributes.inv_rms_tensor_uid()))
    , _dx(tensorMap.at(attributes.dx_tensor_uid()))
    , _dscale(tensorMap.at(attributes.dscale_tensor_uid()))
    , _dbias(attributes.dbias_tensor_uid().has_value()
                 ? tensorMap.at(attributes.dbias_tensor_uid().value())
                 : nullptr)
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::dy() const
{
    return _dy;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::invRMS() const
{
    return _invRMS;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::dx() const
{
    return _dx;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::dscale() const
{
    return _dscale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::dbias() const
{
    return _dbias;
}

RMSnormBwdPlan::RMSnormBwdPlan(RMSnormBwdParams&& params)
    : _params(std::move(params))
{
}

size_t RMSnormBwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for RMS norm
    return 0;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void RMSnormBwdPlan::compile([[maybe_unused]] const IKernelCompiler& kernelCompiler,
                             [[maybe_unused]] const hipDeviceProp_t& deviceProperties)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "RMSNorm backward compile not yet implemented");
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void RMSnormBwdPlan::execute([[maybe_unused]] const HipKernelHandle& handle,
                             [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                             [[maybe_unused]] uint32_t numDeviceBuffers,
                             [[maybe_unused]] void* workspace) const
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "RMSNorm backward execute not yet implemented");
}

} // namespace hip_kernel_provider::rmsnorm
