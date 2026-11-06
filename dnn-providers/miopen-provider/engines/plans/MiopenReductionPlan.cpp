// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenReductionChecks.hpp"
#include "engines/plans/MiopenReductionPlan.hpp"

namespace
{

miopenReduceTensorOp_t
    resolveReduceOp(const hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes& attributes)
{
    const auto reduceOp
        = miopen_plugin::reduction_applicability::toMiopenReduceTensorOp(attributes.mode());
    if(!reduceOp.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "ReductionAttributes mode does not map to a miopenReduceTensorOp_t: isApplicable "
            "should have already rejected this graph");
    }
    return *reduceOp;
}

} // namespace

namespace miopen_plugin
{

MiopenReductionPlan::MiopenReductionPlan(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _input(miopen_utils::createTensor(tensorMap, attributes.in_tensor_uid()))
    , _output(miopen_utils::createTensor(tensorMap, attributes.out_tensor_uid()))
    , _reduceDescriptor(resolveReduceOp(attributes))
    , _inputSpanBytes(static_cast<size_t>(reduction_applicability::tensorSpanBytes(
          miopen_utils::findTensorAttributes(tensorMap, attributes.in_tensor_uid()))))
    , _outputSpanBytes(static_cast<size_t>(reduction_applicability::tensorSpanBytes(
          miopen_utils::findTensorAttributes(tensorMap, attributes.out_tensor_uid()))))
{
    // miopenReduceTensor throws on an undersized workspace, so report exactly what it asks for.
    // Legacy always asks for at least one descriptor page, even for a single-pass reduction.
    THROW_ON_MIOPEN_FAILURE(
        miopenGetReductionWorkspaceSize(handle.miopenHandle,
                                        _reduceDescriptor.reduceTensorDescriptor(),
                                        _input.tensorDescriptor(),
                                        _output.tensorDescriptor(),
                                        &_workspaceSize));

    HIPDNN_PLUGIN_LOG_INFO("Reduction: queried workspace size " << _workspaceSize << " bytes");
}

size_t
    MiopenReductionPlan::getWorkspaceSize([[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    return _workspaceSize;
}

void MiopenReductionPlan::execute(const HipdnnMiopenHandle& handle,
                                  const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                  uint32_t numDeviceBuffers,
                                  void* workspace) const
{
    const auto xBuf
        = hipdnn_plugin_sdk::findDeviceBuffer(_input.uid(), deviceBuffers, numDeviceBuffers);
    const auto yBuf
        = hipdnn_plugin_sdk::findDeviceBuffer(_output.uid(), deviceBuffers, numDeviceBuffers);

    // The gate rejects a graph whose x and y share a uid, but two uids can still be bound to
    // overlapping memory. MIOpen does not check: its kernels take A and C as __restrict__.
    const auto xBegin = reinterpret_cast<uintptr_t>(xBuf.ptr);
    const auto yBegin = reinterpret_cast<uintptr_t>(yBuf.ptr);
    if(xBegin < yBegin + _outputSpanBytes && yBegin < xBegin + _inputSpanBytes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Reduction: input and output device buffers overlap; in-place reduction is not "
            "supported");
    }

    // A null workspace makes MIOpen fail its alignment step with an internal error.
    if(workspace == nullptr && _workspaceSize > 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Reduction: a workspace of " + std::to_string(_workspaceSize)
                + " bytes is required, but none was provided");
    }

    // MIOpen reads alpha and beta as float unless the input is miopenDouble, which the gate
    // excludes. beta = 0 means C is never read.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // No indices are requested, so the indices buffer is null with size 0.
    THROW_ON_MIOPEN_FAILURE(miopenReduceTensor(handle.miopenHandle,
                                               _reduceDescriptor.reduceTensorDescriptor(),
                                               nullptr,
                                               0,
                                               workspace,
                                               _workspaceSize,
                                               &alpha,
                                               _input.tensorDescriptor(),
                                               xBuf.ptr,
                                               &beta,
                                               _output.tensorDescriptor(),
                                               yBuf.ptr));
}

} // namespace miopen_plugin
