// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenConvFwdPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

ConvFwdParams::ConvFwdParams(
    const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, attributes.x_tensor_uid())))
    , _x(miopen_utils::createTensor(tensorMap, attributes.x_tensor_uid()))
    , _w(miopen_utils::createTensor(tensorMap, attributes.w_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, attributes.y_tensor_uid()))
{
    const auto& attrX = miopen_utils::findTensorAttributes(tensorMap, _x.uid());
    const auto& attrW = miopen_utils::findTensorAttributes(tensorMap, _w.uid());
    const auto& attrY = miopen_utils::findTensorAttributes(tensorMap, _y.uid());

    const auto inputDims
        = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(attrX.dims());
    const auto weightDims
        = hipdnn_data_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_data_sdk::utilities::calculateGroupCount(inputDims, weightDims);

    _conv = MiopenConvDescriptor(_spatialDimCount, attributes, static_cast<int>(groupCount));

    _tensorsValid = (!attrX.virtual_() && !attrW.virtual_() && !attrY.virtual_());
}

const MiopenTensor& ConvFwdParams::x() const
{
    return _x;
}

const MiopenTensor& ConvFwdParams::w() const
{
    return _w;
}

const MiopenTensor& ConvFwdParams::y() const
{
    return _y;
}

const MiopenConvDescriptor& ConvFwdParams::conv() const
{
    return _conv;
}

bool ConvFwdParams::validTensors() const
{
    return _tensorsValid;
}

ConvFwdPlan::ConvFwdPlan(const HipdnnEnginePluginHandle& handle,
                         ConvFwdParams&& params,
                         const MiopenExecutionSettings& executionSettings)
    : _params(std::move(params))
    , _executionSettings(executionSettings)
{
    // Validate that there are solutions available for this configuration.
    size_t solutionCount;
    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionForwardGetSolutionCount(handle.miopenHandle,
                                                 _params.w().tensorDescriptor(),
                                                 _params.x().tensorDescriptor(),
                                                 _params.conv().convDescriptor(),
                                                 _params.y().tensorDescriptor(),
                                                 &solutionCount));

    if(solutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "miopenConvolutionForwardGetSolutionCount returned no solutions");
    }

    // Determine initial workspace size
    if(_executionSettings.workspaceSizeLimit().has_value())
    {
        _workspaceSize = _executionSettings.workspaceSizeLimit().value();
    }
    else
    {
        THROW_ON_MIOPEN_FAILURE(
            miopenConvolutionForwardGetWorkSpaceSize(handle.miopenHandle,
                                                     _params.w().tensorDescriptor(),
                                                     _params.x().tensorDescriptor(),
                                                     _params.conv().convDescriptor(),
                                                     _params.y().tensorDescriptor(),
                                                     &_workspaceSize));
    }
}

size_t ConvFwdPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _workspaceSize;
}

void ConvFwdPlan::execute(const HipdnnEnginePluginHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.x().uid(), deviceBuffers, numDeviceBuffers);
    auto wBuffer
        = miopen_utils::findDeviceBuffer(_params.w().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = miopen_utils::findDeviceBuffer(_params.y().uid(), deviceBuffers, numDeviceBuffers);

    size_t workspaceSize = 0;
    if(workspace != nullptr)
    {
        // Assume the provided workspace is large enough
        workspaceSize = _workspaceSize;
    }

    ScopedTuningPolicy tuningGuard(handle.miopenHandle, _executionSettings.benchmarkingEnabled());

    // Algorithm selection is performed on first execute() call rather than in constructor
    // because miopenFindConvolutionForwardAlgorithm requires device memory buffers.
    // These buffers are only available during execute(), not during plan construction.
    // The selected algorithm is cached to avoid redundant find calls on subsequent executions.
    if(!_algorithm.has_value())
    {
        int requestCount
            = (_executionSettings.debugMode()
               == MiopenExecutionSettings::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS)
                  ? 10
                  : 1;

        std::vector<miopenConvAlgoPerf_t> perfResults(static_cast<size_t>(requestCount));
        int returnedAlgoCount;

        THROW_ON_MIOPEN_FAILURE(
            miopenFindConvolutionForwardAlgorithm(handle.miopenHandle,
                                                  _params.x().tensorDescriptor(),
                                                  xBuffer.ptr,
                                                  _params.w().tensorDescriptor(),
                                                  wBuffer.ptr,
                                                  _params.conv().convDescriptor(),
                                                  _params.y().tensorDescriptor(),
                                                  yBuffer.ptr,
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
                "miopenFindConvolutionForwardAlgorithm returned no algorithms");
        }

        if(_executionSettings.debugMode()
           == MiopenExecutionSettings::DebugMode::LOG_ALL_FOUND_PLAN_ALGORITHMS)
        {
            HIPDNN_LOG_INFO("Convolution Fwd: Found {} algorithms", returnedAlgoCount);
            for(size_t i = 0; i < static_cast<size_t>(returnedAlgoCount); ++i)
            {
                HIPDNN_LOG_INFO("  Algorithm {}: algorithm={}, time={}, workspace_size={}",
                                i,
                                static_cast<int>(perfResults[i].fwd_algo),
                                perfResults[i].time,
                                perfResults[i].memory);
            }
        }

        HIPDNN_LOG_INFO("Convolution Fwd: Selected algorithm={}, time={}, workspace_size={}",
                        static_cast<int>(perfResults[0].fwd_algo),
                        perfResults[0].time,
                        perfResults[0].memory);

        _algorithm = perfResults[0].fwd_algo;
        // Update workspace size with the actual requirement from the selected algorithm.
        // This may differ from the initial estimate.
        _workspaceSize = perfResults[0].memory;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    THROW_ON_MIOPEN_FAILURE(miopenConvolutionForward(handle.miopenHandle,
                                                     &alpha,
                                                     _params.x().tensorDescriptor(),
                                                     xBuffer.ptr,
                                                     _params.w().tensorDescriptor(),
                                                     wBuffer.ptr,
                                                     _params.conv().convDescriptor(),
                                                     _algorithm.value(),
                                                     &beta,
                                                     _params.y().tensorDescriptor(),
                                                     yBuffer.ptr,
                                                     workspace,
                                                     workspaceSize));
}

} // namespace miopen_plugin
