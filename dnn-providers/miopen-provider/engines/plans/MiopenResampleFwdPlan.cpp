// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <functional>
#include <numeric>
#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenUtils.hpp"
#include "engines/plans/MiopenResampleFwdPlan.hpp"

namespace
{

size_t indexTypeSize(miopenIndexType_t indexType)
{
    switch(indexType)
    {
    case miopenIndexUint8:
        return sizeof(uint8_t);
    case miopenIndexUint16:
        return sizeof(uint16_t);
    case miopenIndexUint32:
        return sizeof(uint32_t);
    case miopenIndexUint64:
        return sizeof(uint64_t);
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "MiopenResampleFwdPlan: unknown index type");
    }
}

size_t elementCount(const std::vector<int64_t>& dims)
{
    return static_cast<size_t>(
        std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<>()));
}

} // namespace

namespace miopen_plugin
{

MiopenResampleFwdPlan::MiopenResampleFwdPlan(
    const resample_fwd_applicability::ResampleFwdConfig& config)
    : _x(config.xUid, config.dataType, config.xDims, config.xStrides)
    , _y(config.yUid, config.dataType, config.yDims, config.yStrides)
    , _pooling(config.mode, config.window, config.padding, config.stride, config.indexType)
    , _indexUid(config.indexUid)
{
    if(_indexUid.has_value())
    {
        _indexBytes = elementCount(config.yDims) * indexTypeSize(config.indexType);
    }

    THROW_ON_MIOPEN_FAILURE(miopenPoolingGetWorkSpaceSizeV2(
        _pooling.poolingDescriptor(), _y.tensorDescriptor(), &_workspaceSize));

    if(_workspaceSize < _indexBytes)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "MiopenResampleFwdPlan: MIOpen's pooling workspace (" + std::to_string(_workspaceSize)
                + " bytes) cannot hold the " + std::to_string(_indexBytes) + " bytes of indices");
    }
}

size_t
    MiopenResampleFwdPlan::getWorkspaceSize([[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    return _workspaceSize;
}

void MiopenResampleFwdPlan::execute(const HipdnnMiopenHandle& handle,
                                    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                    uint32_t numDeviceBuffers,
                                    void* workspace) const
{
    const auto xBuf
        = hipdnn_plugin_sdk::findDeviceBuffer(_x.uid(), deviceBuffers, numDeviceBuffers);
    const auto yBuf
        = hipdnn_plugin_sdk::findDeviceBuffer(_y.uid(), deviceBuffers, numDeviceBuffers);

    void* indexPtr = nullptr;
    if(_indexUid.has_value())
    {
        indexPtr
            = hipdnn_plugin_sdk::findDeviceBuffer(*_indexUid, deviceBuffers, numDeviceBuffers).ptr;
        if(indexPtr == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                "MiopenResampleFwdPlan: the index tensor has no device buffer");
        }
    }

    // MIOpen never checks the size of the workspace it is given, and rejects a null one only for
    // max pooling that saves indices: a missing or short workspace is otherwise an out-of-bounds
    // write on the device, not an error.
    if(_workspaceSize > 0 && workspace == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "MiopenResampleFwdPlan: requires " + std::to_string(_workspaceSize)
                + " workspace bytes but none was provided");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    THROW_ON_MIOPEN_FAILURE(miopenPoolingForward(handle.miopenHandle,
                                                 _pooling.poolingDescriptor(),
                                                 &alpha,
                                                 _x.tensorDescriptor(),
                                                 xBuf.ptr,
                                                 &beta,
                                                 _y.tensorDescriptor(),
                                                 yBuf.ptr,
                                                 _indexUid.has_value(),
                                                 workspace,
                                                 _workspaceSize));

    if(_indexUid.has_value())
    {
        // Enqueue on MIOpen's stream so the copy is ordered after the pooling kernel.
        hipStream_t stream = nullptr;
        THROW_ON_MIOPEN_FAILURE(miopenGetStream(handle.miopenHandle, &stream));

        const auto status
            = hipMemcpyAsync(indexPtr, workspace, _indexBytes, hipMemcpyDeviceToDevice, stream);
        if(status != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string("MiopenResampleFwdPlan: copying the pooling indices failed: ")
                    + hipGetErrorString(status));
        }
    }
}

} // namespace miopen_plugin
