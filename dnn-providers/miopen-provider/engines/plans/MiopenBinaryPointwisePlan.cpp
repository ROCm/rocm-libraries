// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenBinaryPointwisePlan.hpp"

namespace
{

int64_t
    resolveIn1TensorUid(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attributes)
{
    if(!attributes.in_1_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "PointwiseAttributes is missing in_1_tensor_uid: isApplicable should have already "
            "rejected this graph");
    }

    return *attributes.in_1_tensor_uid();
}

} // namespace

namespace miopen_plugin
{

MiopenBinaryPointwisePlan::MiopenBinaryPointwisePlan(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _inputA(miopen_utils::createTensor(tensorMap, attributes.in_0_tensor_uid()))
    , _inputB(miopen_utils::createTensor(tensorMap, resolveIn1TensorUid(attributes)))
    , _output(miopen_utils::createTensor(tensorMap, attributes.out_0_tensor_uid()))
    , _mode(attributes.operation())
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
    const auto aBuf
        = hipdnn_plugin_sdk::findDeviceBuffer(_inputA.uid(), deviceBuffers, numDeviceBuffers);
    const auto bBuf
        = hipdnn_plugin_sdk::findDeviceBuffer(_inputB.uid(), deviceBuffers, numDeviceBuffers);
    const auto cBuf
        = hipdnn_plugin_sdk::findDeviceBuffer(_output.uid(), deviceBuffers, numDeviceBuffers);

    // C = op(scaleA * A, scaleB * B) + beta * C.
    miopenTensorOp_t op{};
    float scaleA = 1.0f;
    float scaleB = 1.0f;

    using PM = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    switch(_mode)
    {
    case PM::ADD:
        op = miopenTensorOpAdd;
        break;
    case PM::SUB:
        // there is no miopenTensorOpSub; A - B is Add(A, -B), so negate B's scale instead.
        op = miopenTensorOpAdd;
        scaleB = -1.0f;
        break;
    case PM::MUL:
        op = miopenTensorOpMul;
        break;
    case PM::MAX_OP:
        op = miopenTensorOpMax;
        break;
    case PM::MIN_OP:
        op = miopenTensorOpMin;
        break;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "MiopenBinaryPointwisePlan::execute: mode does not map to a miopenTensorOp_t; "
            "isApplicable should have already rejected this graph");
    }

    float beta = 0.0f;

    THROW_ON_MIOPEN_FAILURE(miopenOpTensor(handle.miopenHandle,
                                           op,
                                           &scaleA,
                                           _inputA.tensorDescriptor(),
                                           aBuf.ptr,
                                           &scaleB,
                                           _inputB.tensorDescriptor(),
                                           bBuf.ptr,
                                           &beta,
                                           _output.tensorDescriptor(),
                                           cBuf.ptr));
}

} // namespace miopen_plugin
