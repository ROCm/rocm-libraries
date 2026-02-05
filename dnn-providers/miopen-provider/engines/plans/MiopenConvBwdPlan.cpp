// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenConvBwdPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

ConvBwdParams::ConvBwdParams(
    const hipdnn_data_sdk::data_objects::ConvolutionBwdAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, attributes.dx_tensor_uid())))
    , _dx(miopen_utils::createTensor(tensorMap, attributes.dx_tensor_uid()))
    , _w(miopen_utils::createTensor(tensorMap, attributes.w_tensor_uid()))
    , _dy(miopen_utils::createTensor(tensorMap, attributes.dy_tensor_uid()))
{
    const auto& attrDX = miopen_utils::findTensorAttributes(tensorMap, _dx.uid());
    const auto& attrW = miopen_utils::findTensorAttributes(tensorMap, _w.uid());
    const auto& attrDY = miopen_utils::findTensorAttributes(tensorMap, _dy.uid());

    const auto inputDims
        = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(attrDX.dims());
    const auto weightDims
        = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_data_sdk::utilities::calculateGroupCount(inputDims, weightDims);

    _conv = MiopenConvDescriptor(_spatialDimCount, attributes, static_cast<int>(groupCount));

    _tensorsValid = (!attrDX.virtual_() && !attrW.virtual_() && !attrDY.virtual_());
}

const MiopenTensor& ConvBwdParams::dx() const
{
    return _dx;
}

const MiopenTensor& ConvBwdParams::w() const
{
    return _w;
}

const MiopenTensor& ConvBwdParams::dy() const
{
    return _dy;
}

const MiopenConvDescriptor& ConvBwdParams::conv() const
{
    return _conv;
}

bool ConvBwdParams::validTensors() const
{
    return _tensorsValid;
}

ConvBwdPlan::ConvBwdPlan(const HipdnnEnginePluginHandle& handle,
                         ConvBwdParams&& params,
                         const HipdnnEnginePluginExecutionContext& executionContext)
    : _params(std::move(params))
    , _benchmarkingEnabled(executionContext.benchmarkingEnabled())
    , _debugMode(executionContext.debugMode())
{
    // Validate that there are solutions available for this configuration.
    size_t solutionCount;
    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardDataGetSolutionCount(handle.miopenHandle,
                                                      _params.dy().tensorDescriptor(),
                                                      _params.w().tensorDescriptor(),
                                                      _params.conv().convDescriptor(),
                                                      _params.dx().tensorDescriptor(),
                                                      &solutionCount));

    if(solutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "miopenConvolutionBackwardDataGetSolutionCount returned no solutions");
    }

    // Determine initial workspace size
    if(executionContext.workspaceSizeLimit().has_value())
    {
        _workspaceSize = executionContext.workspaceSizeLimit().value();
    }
    else
    {
        THROW_ON_MIOPEN_FAILURE(
            miopenConvolutionBackwardDataGetWorkSpaceSize(handle.miopenHandle,
                                                          _params.dy().tensorDescriptor(),
                                                          _params.w().tensorDescriptor(),
                                                          _params.conv().convDescriptor(),
                                                          _params.dx().tensorDescriptor(),
                                                          &_workspaceSize));
    }
}

size_t ConvBwdPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _workspaceSize;
}

void ConvBwdPlan::execute(const HipdnnEnginePluginHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.dx().uid(), deviceBuffers, numDeviceBuffers);
    auto wBuffer
        = miopen_utils::findDeviceBuffer(_params.w().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = miopen_utils::findDeviceBuffer(_params.dy().uid(), deviceBuffers, numDeviceBuffers);

    size_t workspaceSize = 0;
    if(workspace != nullptr)
    {
        // Assume the provided workspace is large enough
        workspaceSize = _workspaceSize;
    }

    ScopedTuningPolicy tuningGuard(handle.miopenHandle, _benchmarkingEnabled);

    // Algorithm selection is performed on first execute() call rather than in constructor
    // because miopenFindConvolutionBackwardDataAlgorithm requires device memory buffers.
    // These buffers are only available during execute(), not during plan construction.
    // The selected algorithm is cached to avoid redundant find calls on subsequent executions.
    if(!_algorithm.has_value())
    {
        int requestCount
            = (_debugMode
               == HipdnnEnginePluginExecutionContext::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS)
                  ? 10
                  : 1;

        std::vector<miopenConvAlgoPerf_t> perfResults(static_cast<size_t>(requestCount));
        int returnedAlgoCount;

        THROW_ON_MIOPEN_FAILURE(
            miopenFindConvolutionBackwardDataAlgorithm(handle.miopenHandle,
                                                       _params.dy().tensorDescriptor(),
                                                       yBuffer.ptr,
                                                       _params.w().tensorDescriptor(),
                                                       wBuffer.ptr,
                                                       _params.conv().convDescriptor(),
                                                       _params.dx().tensorDescriptor(),
                                                       xBuffer.ptr,
                                                       requestCount,
                                                       &returnedAlgoCount,
                                                       perfResults.data(),
                                                       workspace,
                                                       workspaceSize,
                                                       false));

        if(returnedAlgoCount <= 0)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "miopenFindConvolutionBackwardDataAlgorithm returned no algorithms");
        }

        if(_debugMode
           == HipdnnEnginePluginExecutionContext::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS)
        {
            HIPDNN_LOG_INFO("Convolution Bwd: Found {} algorithms", returnedAlgoCount);
            for(size_t i = 0; i < static_cast<size_t>(returnedAlgoCount); ++i)
            {
                HIPDNN_LOG_INFO("  Algorithm {}: algorithm={}, time={}, workspace_size={}",
                                i,
                                static_cast<int>(perfResults[i].bwd_data_algo),
                                perfResults[i].time,
                                perfResults[i].memory);
            }
        }

        HIPDNN_LOG_INFO("Convolution Bwd: Selected algorithm={}, time={}, workspace_size={}",
                        static_cast<int>(perfResults[0].bwd_data_algo),
                        perfResults[0].time,
                        perfResults[0].memory);

        _algorithm = perfResults[0].bwd_data_algo;
        // Update workspace size with the actual requirement from the selected algorithm.
        // This may differ from the initial estimate.
        _workspaceSize = perfResults[0].memory;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    THROW_ON_MIOPEN_FAILURE(miopenConvolutionBackwardData(handle.miopenHandle,
                                                          &alpha,
                                                          _params.dy().tensorDescriptor(),
                                                          yBuffer.ptr,
                                                          _params.w().tensorDescriptor(),
                                                          wBuffer.ptr,
                                                          _params.conv().convDescriptor(),
                                                          _algorithm.value(),
                                                          &beta,
                                                          _params.dx().tensorDescriptor(),
                                                          xBuffer.ptr,
                                                          workspace,
                                                          workspaceSize));
}

} // namespace miopen_plugin
