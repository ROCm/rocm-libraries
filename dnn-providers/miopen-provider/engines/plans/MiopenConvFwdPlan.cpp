// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenConvFwdPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

ConvFwdParams::ConvFwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool deterministicEnabled)
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
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrX.dims());
    const auto weightDims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_data_sdk::utilities::calculateGroupCount(inputDims, weightDims);

    _conv = MiopenConvDescriptor(
        _spatialDimCount, attributes, static_cast<int>(groupCount), deterministicEnabled);

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

ConvFwdPlan::ConvFwdPlan(const HipdnnMiopenHandle& handle,
                         ConvFwdParams&& params,
                         const HipdnnMiopenSettings& executionSettings)
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
        HIPDNN_PLUGIN_LOG_INFO(
            "Convolution Fwd: Using knob settings workspace size: " << _workspaceSize);
    }
    else if(_executionSettings.defaultWorkspaceSize().has_value())
    {
        _workspaceSize = _executionSettings.defaultWorkspaceSize().value();
        HIPDNN_PLUGIN_LOG_INFO(
            "Convolution Fwd: Using default max workspace size: " << _workspaceSize);
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
        HIPDNN_PLUGIN_LOG_WARN("Convolution Fwd: Using queried workspace size: " << _workspaceSize);
    }
}

size_t ConvFwdPlan::getWorkspaceSize([[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    return _workspaceSize;
}

void ConvFwdPlan::execute(const HipdnnMiopenHandle& handle,
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
    {
        std::lock_guard<std::mutex> lock(_algorithmMutex);

        if(!_algorithm.has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Convolution Fwd: Performing algorithm selection (first execution)");

            bool traceEnabled = HIPDNN_PLUGIN_LOG_IS_TRACE_ENABLED();
            // Find dedupes by algorithm class (ShrinkToFind10Results in
            // projects/miopen/src/ocl/convolutionocl.cpp:238), so it returns at most one
            // entry per value of miopenConvFwdAlgorithm_t (5 enumerators).
            int requestCount = traceEnabled ? 5 : 1;

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

            if(traceEnabled)
            {
                HIPDNN_PLUGIN_LOG_TRACE("Convolution Fwd: Found " << returnedAlgoCount
                                                                  << " algorithms");
                for(size_t i = 0; i < static_cast<size_t>(returnedAlgoCount); ++i)
                {
                    HIPDNN_PLUGIN_LOG_TRACE("  Algorithm "
                                            << i << ": algorithm="
                                            << static_cast<int>(perfResults[i].fwd_algo)
                                            << ", time=" << perfResults[i].time
                                            << ", workspace_size=" << perfResults[i].memory);
                }
            }

            HIPDNN_PLUGIN_LOG_INFO("Convolution Fwd: Selected algorithm="
                                   << static_cast<int>(perfResults[0].fwd_algo)
                                   << ", time=" << perfResults[0].time
                                   << ", workspace_size=" << perfResults[0].memory);

            _algorithm = perfResults[0].fwd_algo;
            // Update workspace size with the actual requirement from the selected algorithm.
            // This may differ from the initial estimate.
            _workspaceSize = perfResults[0].memory;
        }
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
