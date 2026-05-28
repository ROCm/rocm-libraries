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
ResampleBwdParams::ResampleBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _dy(tensorMap.at(attributes.dy_tensor_uid()))
    , _dx(tensorMap.at(attributes.dx_tensor_uid()))
    , _index(attributes.index_tensor_uid().has_value()
                 ? tensorMap.at(attributes.index_tensor_uid().value())
                 : nullptr)
    , _prePadding(attributes.pre_padding())
    , _postPadding(attributes.post_padding())
    , _stride(attributes.stride())
    , _window(attributes.window())
    , _resampleMode(attributes.resample_mode())
    , _paddingMode(attributes.padding_mode())
    , _generateIndex(attributes.generate_index().has_value() ? attributes.generate_index().value()
                                                             : false)
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

const flatbuffers::Vector<int64_t>* ResampleBwdParams::prePadding() const
{
    return _prePadding;
}

const flatbuffers::Vector<int64_t>* ResampleBwdParams::postPadding() const
{
    return _postPadding;
}

const flatbuffers::Vector<int64_t>* ResampleBwdParams::stride() const
{
    return _stride;
}

const flatbuffers::Vector<int64_t>* ResampleBwdParams::window() const
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

ResampleBwdPlan::ResampleBwdPlan(ResampleBwdParams&& params)
    : _params(std::move(params))
{
}

size_t ResampleBwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for resample backward
    return 0;
}

void ResampleBwdPlan::compile([[maybe_unused]] const IKernelCompiler& kernelCompiler,
                              [[maybe_unused]] const hipDeviceProp_t& deviceProperties)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Resample backward compile not yet implemented");
}

void ResampleBwdPlan::execute([[maybe_unused]] const HipKernelHandle& handle,
                              [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              [[maybe_unused]] uint32_t numDeviceBuffers,
                              [[maybe_unused]] void* workspace) const
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Resample backward execute not yet implemented");
}

} // namespace hip_kernel_provider::resample
