// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResampleBwdPlan.hpp"

#include "hip/IKernelCompiler.hpp"

#include <cstddef>
#include <cstdint>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::resample
{

namespace
{

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>* values)
{
    if(values == nullptr)
    {
        return {};
    }
    return {values->begin(), values->end()};
}

} // namespace

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
    , _prePadding(toVector(attributes.pre_padding()))
    , _postPadding(toVector(attributes.post_padding()))
    , _stride(toVector(attributes.stride()))
    , _window(toVector(attributes.window()))
    , _resampleMode(attributes.resample_mode())
    , _paddingMode(attributes.padding_mode())
    , _generateIndex(attributes.generate_index().has_value() ? attributes.generate_index().value()
                                                             : false)
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

bool ResampleBwdParams::generateIndex() const
{
    return _generateIndex;
}

hipdnn_flatbuffers_sdk::data_objects::DataType ResampleBwdParams::computeDataType() const
{
    return _computeDataType;
}

ResampleBwdPlan::ResampleBwdPlan(ResampleBwdParams&& params)
    : _params(std::move(params))
{
}

size_t ResampleBwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
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
void ResampleBwdPlan::execute([[maybe_unused]] const HipKernelHandle& handle,
                              [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              [[maybe_unused]] uint32_t numDeviceBuffers,
                              [[maybe_unused]] void* workspace) const
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Resample backward execute not yet implemented");
}

} // namespace hip_kernel_provider::resample
