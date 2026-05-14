// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "MiopenConvWrwPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

ConvWrwParams::ConvWrwParams(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool deterministicEnabled)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, attributes.x_tensor_uid())))
    , _x(miopen_utils::createTensor(tensorMap, attributes.x_tensor_uid()))
    , _dw(miopen_utils::createTensor(tensorMap, attributes.dw_tensor_uid()))
    , _dy(miopen_utils::createTensor(tensorMap, attributes.dy_tensor_uid()))
{
    const auto& attrX = miopen_utils::findTensorAttributes(tensorMap, _x.uid());
    const auto& attrDW = miopen_utils::findTensorAttributes(tensorMap, _dw.uid());
    const auto& attrDY = miopen_utils::findTensorAttributes(tensorMap, _dy.uid());

    const auto inputDims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrX.dims());
    const auto weightDims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrDW.dims());
    const auto groupCount = hipdnn_data_sdk::utilities::calculateGroupCount(inputDims, weightDims);

    _conv = MiopenConvDescriptor(
        _spatialDimCount, attributes, static_cast<int>(groupCount), deterministicEnabled);

    _tensorsValid = (!attrX.virtual_() && !attrDW.virtual_() && !attrDY.virtual_());
}

const MiopenTensor& ConvWrwParams::x() const
{
    return _x;
}

const MiopenTensor& ConvWrwParams::dw() const
{
    return _dw;
}

const MiopenTensor& ConvWrwParams::dy() const
{
    return _dy;
}

const MiopenConvDescriptor& ConvWrwParams::conv() const
{
    return _conv;
}

bool ConvWrwParams::validTensors() const
{
    return _tensorsValid;
}

ConvWrwPlan::ConvWrwPlan(const HipdnnMiopenHandle& handle,
                         ConvWrwParams&& params,
                         const HipdnnMiopenSettings& executionSettings)
    : _params(std::move(params))
    , _executionSettings(executionSettings)
{
    // Validate that there are solutions available for this configuration.
    size_t solutionCount;
    THROW_ON_MIOPEN_FAILURE(
        miopenConvolutionBackwardWeightsGetSolutionCount(handle.miopenHandle,
                                                         _params.dy().tensorDescriptor(),
                                                         _params.x().tensorDescriptor(),
                                                         _params.conv().convDescriptor(),
                                                         _params.dw().tensorDescriptor(),
                                                         &solutionCount));

    if(solutionCount == 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "miopenConvolutionBackwardWeightsGetSolutionCount returned no solutions");
    }

    // Determine initial workspace size
    if(_executionSettings.workspaceSizeLimit().has_value())
    {
        _workspaceSize = _executionSettings.workspaceSizeLimit().value();
        HIPDNN_PLUGIN_LOG_INFO(
            "Convolution Wrw: Using knob settings workspace size: " << _workspaceSize);
    }
    else if(_executionSettings.defaultWorkspaceSize().has_value())
    {
        _workspaceSize = _executionSettings.defaultWorkspaceSize().value();
        HIPDNN_PLUGIN_LOG_INFO(
            "Convolution Wrw: Using default max workspace size: " << _workspaceSize);
    }
    else
    {
        THROW_ON_MIOPEN_FAILURE(
            miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle.miopenHandle,
                                                             _params.dy().tensorDescriptor(),
                                                             _params.x().tensorDescriptor(),
                                                             _params.conv().convDescriptor(),
                                                             _params.dw().tensorDescriptor(),
                                                             &_workspaceSize));
        HIPDNN_PLUGIN_LOG_WARN("Convolution Wrw: Using queried workspace size: " << _workspaceSize);
    }
}

size_t ConvWrwPlan::getWorkspaceSize([[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    return _workspaceSize;
}

void ConvWrwPlan::execute(const HipdnnMiopenHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.x().uid(), deviceBuffers, numDeviceBuffers);
    auto wBuffer
        = miopen_utils::findDeviceBuffer(_params.dw().uid(), deviceBuffers, numDeviceBuffers);
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
    // because miopenFindConvolutionBackwardWeightsAlgorithm requires device memory buffers.
    // These buffers are only available during execute(), not during plan construction.
    // The selected algorithm is cached to avoid redundant find calls on subsequent executions.
    {
        std::lock_guard<std::mutex> lock(_algorithmMutex);

        if(!_algorithm.has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO(
                "Convolution Wrw: Performing algorithm selection (first execution)");

            bool traceEnabled = HIPDNN_PLUGIN_LOG_IS_TRACE_ENABLED();
            // Find dedupes by algorithm class (ShrinkToFind10Results in
            // projects/miopen/src/ocl/convolutionocl.cpp:238), so it returns at most one
            // entry per value of miopenConvBwdWeightsAlgorithm_t (4 enumerators).
            int requestCount = traceEnabled ? 4 : 1;

            std::vector<miopenConvAlgoPerf_t> perfResults(static_cast<size_t>(requestCount));
            int returnedAlgoCount;

            THROW_ON_MIOPEN_FAILURE(
                miopenFindConvolutionBackwardWeightsAlgorithm(handle.miopenHandle,
                                                              _params.dy().tensorDescriptor(),
                                                              yBuffer.ptr,
                                                              _params.x().tensorDescriptor(),
                                                              xBuffer.ptr,
                                                              _params.conv().convDescriptor(),
                                                              _params.dw().tensorDescriptor(),
                                                              wBuffer.ptr,
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
                    "miopenFindConvolutionBackwardWeightsAlgorithm returned no algorithms");
            }

            if(traceEnabled)
            {
                HIPDNN_PLUGIN_LOG_TRACE("Convolution Wrw: Found " << returnedAlgoCount
                                                                  << " algorithms");
                for(size_t i = 0; i < static_cast<size_t>(returnedAlgoCount); ++i)
                {
                    HIPDNN_PLUGIN_LOG_TRACE("  Algorithm "
                                            << i << ": algorithm="
                                            << static_cast<int>(perfResults[i].bwd_weights_algo)
                                            << ", time=" << perfResults[i].time
                                            << ", workspace_size=" << perfResults[i].memory);
                }
            }

            HIPDNN_PLUGIN_LOG_INFO("Convolution Wrw: Selected algorithm="
                                   << static_cast<int>(perfResults[0].bwd_weights_algo)
                                   << ", time=" << perfResults[0].time
                                   << ", workspace_size=" << perfResults[0].memory);

            _algorithm = perfResults[0].bwd_weights_algo;
            // Update workspace size with the actual requirement from the selected algorithm.
            // This may differ from the initial estimate.
            _workspaceSize = perfResults[0].memory;
        }
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    THROW_ON_MIOPEN_FAILURE(miopenConvolutionBackwardWeights(handle.miopenHandle,
                                                             &alpha,
                                                             _params.dy().tensorDescriptor(),
                                                             yBuffer.ptr,
                                                             _params.x().tensorDescriptor(),
                                                             xBuffer.ptr,
                                                             _params.conv().convDescriptor(),
                                                             _algorithm.value(),
                                                             &beta,
                                                             _params.dw().tensorDescriptor(),
                                                             wBuffer.ptr,
                                                             workspace,
                                                             workspaceSize));
}

} // namespace miopen_plugin
