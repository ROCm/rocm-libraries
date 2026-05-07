// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenConvBwdPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

ConvBwdParams::ConvBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool deterministicEnabled)
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
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrDX.dims());
    const auto weightDims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_data_sdk::utilities::calculateGroupCount(inputDims, weightDims);

    _conv = MiopenConvDescriptor(
        _spatialDimCount, attributes, static_cast<int>(groupCount), deterministicEnabled);

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

ConvBwdPlan::ConvBwdPlan(const HipdnnMiopenHandle& handle,
                         ConvBwdParams&& params,
                         const HipdnnMiopenSettings& executionSettings)
    : _params(std::move(params))
    , _executionSettings(executionSettings)
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
    if(_executionSettings.workspaceSizeLimit().has_value())
    {
        _workspaceSize = _executionSettings.workspaceSizeLimit().value();
        HIPDNN_PLUGIN_LOG_INFO(
            "Convolution Bwd: Using knob settings workspace size: " << _workspaceSize);
    }
    else if(_executionSettings.defaultWorkspaceSize().has_value())
    {
        _workspaceSize = _executionSettings.defaultWorkspaceSize().value();
        HIPDNN_PLUGIN_LOG_INFO("Convolution Bwd: Default max workspace size: " << _workspaceSize);
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
        HIPDNN_PLUGIN_LOG_WARN("Convolution Bwd: Using queried workspace size: " << _workspaceSize);
    }
}

size_t ConvBwdPlan::getWorkspaceSize([[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    return _workspaceSize;
}

void ConvBwdPlan::execute(const HipdnnMiopenHandle& handle,
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

    ScopedTuningPolicy tuningGuard(handle.miopenHandle, _executionSettings.benchmarkingEnabled());

    // Algorithm selection is performed on first execute() call rather than in constructor
    // because miopenFindConvolutionBackwardDataAlgorithm requires device memory buffers.
    // These buffers are only available during execute(), not during plan construction.
    // The selected algorithm is cached to avoid redundant find calls on subsequent executions.
    {
        std::lock_guard<std::mutex> lock(_algorithmMutex);

        if(!_algorithm.has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Convolution Bwd: Performing algorithm selection (first execution)");

            bool traceEnabled = HIPDNN_PLUGIN_LOG_IS_TRACE_ENABLED();
            int requestCount = traceEnabled ? 10 : 1;

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

            if(traceEnabled)
            {
                HIPDNN_PLUGIN_LOG_TRACE("Convolution Bwd: Found " << returnedAlgoCount
                                                                  << " algorithms");
                for(size_t i = 0; i < static_cast<size_t>(returnedAlgoCount); ++i)
                {
                    HIPDNN_PLUGIN_LOG_TRACE("  Algorithm "
                                            << i << ": algorithm="
                                            << static_cast<int>(perfResults[i].bwd_data_algo)
                                            << ", time=" << perfResults[i].time
                                            << ", workspace_size=" << perfResults[i].memory);
                }
            }

            HIPDNN_PLUGIN_LOG_INFO("Convolution Bwd: Selected algorithm="
                                   << static_cast<int>(perfResults[0].bwd_data_algo)
                                   << ", time=" << perfResults[0].time
                                   << ", workspace_size=" << perfResults[0].memory);

            _algorithm = perfResults[0].bwd_data_algo;
            // Update workspace size with the actual requirement from the selected algorithm.
            // This may differ from the initial estimate.
            _workspaceSize = perfResults[0].memory;
        }
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
