// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenBinaryPointwisePlan.hpp"

namespace miopen_plugin
{

namespace
{
int64_t getIn1TensorUidOrThrow(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attributes)
{
    if(!attributes.in_1_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "MiopenBinaryPointwisePlan: missing in_1_tensor_uid for binary pointwise operation");
    }
    return *attributes.in_1_tensor_uid();
}
} // namespace

MiopenBinaryPointwisePlan::MiopenBinaryPointwisePlan(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _mode(attributes.operation())
    , _input0(miopen_utils::createTensor(tensorMap, attributes.in_0_tensor_uid()))
    , _input1(miopen_utils::createTensor(tensorMap, getIn1TensorUidOrThrow(attributes)))
    , _output(miopen_utils::createTensor(tensorMap, attributes.out_0_tensor_uid()))
{
}

size_t MiopenBinaryPointwisePlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    return 0;
}

void MiopenBinaryPointwisePlan::execute(const HipdnnMiopenHandle& handle,
                                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                        uint32_t numDeviceBuffers,
                                        [[maybe_unused]] void* workspace) const
{
    const auto input0Buffer
        = miopen_utils::findDeviceBuffer(_input0.uid(), deviceBuffers, numDeviceBuffers);
    const auto input1Buffer
        = miopen_utils::findDeviceBuffer(_input1.uid(), deviceBuffers, numDeviceBuffers);
    const auto outputBuffer
        = miopen_utils::findDeviceBuffer(_output.uid(), deviceBuffers, numDeviceBuffers);

    float alpha1 = 1.0f;
    float alpha2 = 1.0f;
    float beta = 0.0f;
    miopenTensorOp_t miopenOp;
    switch(_mode)
    {
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD:
        miopenOp = miopenTensorOpAdd;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB:
        miopenOp = miopenTensorOpAdd; // Subtraction emulated via addition: A + (-1 * B)
        alpha2 = -1.0f; // Flip sign of second input
        break;
    case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL:
        miopenOp = miopenTensorOpMul;
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Binary pointwise execution: unsupported operation mode encountered.");
    }

    // Implements MIOpen scaling contract: C = op(alpha1 * A, alpha2 * B) + beta * C
    // Subtraction is emulated using miopenTensorOpAdd with alpha2 = -1.0f
    THROW_ON_MIOPEN_FAILURE(miopenOpTensor(handle.miopenHandle,
                                           miopenOp,
                                           &alpha1,
                                           _input0.tensorDescriptor(),
                                           input0Buffer.ptr,
                                           &alpha2,
                                           _input1.tensorDescriptor(),
                                           input1Buffer.ptr,
                                           &beta,
                                           _output.tensorDescriptor(),
                                           outputBuffer.ptr));
}

} // namespace miopen_plugin
