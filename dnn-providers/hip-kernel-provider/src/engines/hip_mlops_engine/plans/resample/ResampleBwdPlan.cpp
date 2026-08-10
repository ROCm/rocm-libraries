// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResampleBwdPlan.hpp"

#include "compilation/KernelCompileOptions.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResamplePlanUtils.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <cstdint>
#include <limits>
#include <numeric>

namespace hip_kernel_provider::resample
{
using namespace hip_kernel_provider::compilation;
using namespace hip_kernel_provider::core::utils;

ResampleBwdParams::ResampleBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    : _dy(tensorMap.at(attributes.dy_tensor_uid()))
    , _dx(tensorMap.at(attributes.dx_tensor_uid()))
    , _index(attributes.index_tensor_uid().has_value()
                 ? tensorMap.at(attributes.index_tensor_uid().value())
                 : nullptr)
    , _prePadding(toStdVector(attributes.pre_padding()))
    , _postPadding(toStdVector(attributes.post_padding()))
    , _stride(toStdVector(attributes.stride()))
    , _window(toStdVector(attributes.window()))
    , _resampleMode(attributes.resample_mode())
    , _paddingMode(attributes.padding_mode())
    , _computeDataType(computeDataType)
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ResampleBwdParams::dy() const
{
    return _dy;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ResampleBwdParams::dx() const
{
    return _dx;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ResampleBwdParams::index() const
{
    return _index;
}

const std::vector<int64_t>& ResampleBwdParams::prePadding() const
{
    return _prePadding;
}

const std::vector<int64_t>& ResampleBwdParams::postPadding() const
{
    return _postPadding;
}

const std::vector<int64_t>& ResampleBwdParams::stride() const
{
    return _stride;
}

const std::vector<int64_t>& ResampleBwdParams::window() const
{
    return _window;
}

hipdnn_flatbuffers_sdk::data_objects::ResampleMode ResampleBwdParams::resampleMode() const
{
    return _resampleMode;
}

hipdnn_flatbuffers_sdk::data_objects::PaddingMode ResampleBwdParams::paddingMode() const
{
    return _paddingMode;
}

hipdnn_flatbuffers_sdk::data_objects::DataType ResampleBwdParams::computeDataType() const
{
    return _computeDataType;
}

ResampleBwdPlan::ResampleBwdPlan(ResampleBwdParams&& params)
    : _params(std::move(params))
{
}

size_t ResampleBwdPlan::getWorkspaceSize([[maybe_unused]] const Handle& handle) const
{
    // No workspace needed for resample backward
    return 0;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void ResampleBwdPlan::compile([[maybe_unused]] const IKernelCompiler& kernelCompiler,
                              [[maybe_unused]] const hipDeviceProp_t& deviceProperties)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Resample backward compile not yet implemented");
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void ResampleBwdPlan::execute([[maybe_unused]] const Handle& handle,
                              [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              [[maybe_unused]] uint32_t numDeviceBuffers,
                              [[maybe_unused]] void* workspace) const
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Resample backward execute not yet implemented");
}

} // namespace hip_kernel_provider::resample
