// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenReluPlan.hpp"

namespace miopen_plugin
{

MiopenReluPlan::MiopenReluPlan(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _input(miopen_utils::createTensor(tensorMap, attributes.in_0_tensor_uid()))
    , _output(miopen_utils::createTensor(tensorMap, attributes.out_0_tensor_uid()))
    , _activation(attributes)
{
}

size_t MiopenReluPlan::getWorkspaceSize([[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    return 0;
}

void MiopenReluPlan::execute(const HipdnnMiopenHandle& handle,
                             const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                             uint32_t numDeviceBuffers,
                             [[maybe_unused]] void* workspace) const
{
    const auto inputBuffer
        = miopen_utils::findDeviceBuffer(_input.uid(), deviceBuffers, numDeviceBuffers);
    const auto outputBuffer
        = miopen_utils::findDeviceBuffer(_output.uid(), deviceBuffers, numDeviceBuffers);

    float alpha = 1.0f;
    float beta = 0.0f;

    THROW_ON_MIOPEN_FAILURE(miopenActivationForward(handle.miopenHandle,
                                                    _activation.activationDescriptor(),
                                                    &alpha,
                                                    _input.tensorDescriptor(),
                                                    inputBuffer.ptr,
                                                    &beta,
                                                    _output.tensorDescriptor(),
                                                    outputBuffer.ptr));
}

} // namespace miopen_plugin
