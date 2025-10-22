// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenConvFwdPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

ConvFwdParams::ConvFwdParams(
    const hipdnn_sdk::data_objects::ConvolutionFwdAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, attributes.x_tensor_uid())))
    , _x(miopen_utils::createTensor(tensorMap, attributes.x_tensor_uid()))
    , _w(miopen_utils::createTensor(tensorMap, attributes.w_tensor_uid()))
    , _y(miopen_utils::createTensor(tensorMap, attributes.y_tensor_uid()))
{
    const auto& attrX = miopen_utils::findTensorAttributes(tensorMap, _x.uid());
    const auto& attrW = miopen_utils::findTensorAttributes(tensorMap, _w.uid());
    const auto& attrY = miopen_utils::findTensorAttributes(tensorMap, _y.uid());

    const auto inputDims = hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(attrX.dims());
    const auto weightDims = hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_sdk::utilities::calculateGroupCount(inputDims, weightDims);

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

ConvFwdPlan::ConvFwdPlan(const HipdnnEnginePluginHandle& handle, ConvFwdParams&& params)
    : _params(std::move(params))
{
    // MIOpen Find 2.0 API
    miopenProblem_t problem;
    THROW_ON_MIOPEN_FAILURE(miopenCreateConvProblem(
        &problem, _params.conv().convDescriptor(), miopenProblemDirectionForward));
    hipdnn_sdk::utilities::ScopedResource problemRes(
        problem, [](miopenProblem_t p) { std::ignore = miopenDestroyProblem(p); });

    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionX, _params.x().tensorDescriptor()));
    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionW, _params.w().tensorDescriptor()));
    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionY, _params.y().tensorDescriptor()));

    size_t numSolutions;
    miopenSolution_t solution = nullptr;
    // Requesting only the best solution
    THROW_ON_MIOPEN_FAILURE(
        miopenFindSolutions(handle.miopenHandle, problem, nullptr, &solution, &numSolutions, 1));

    if(solution != nullptr)
    {
        _solution = hipdnn_sdk::utilities::ScopedResource<miopenSolution_t>(
            solution, [](miopenSolution_t s) {
                auto status = miopenDestroySolution(s);
                if(status != miopenStatusSuccess)
                {
                    HIPDNN_LOG_ERROR("miopenDestroySolution failed in ConvFwdPlan destructor");
                }
            });
    }

    if(numSolutions != 1)
    {
        throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "miopenFindSolutions returned no solutions");
    }

    // Get workspace size
    THROW_ON_MIOPEN_FAILURE(miopenGetSolutionWorkspaceSize(_solution.get(), &_workspaceSize));
}

ConvFwdPlan::ConvFwdPlan(ConvFwdPlan&& other) noexcept
    : _params(std::move(other._params))
    , _solution(std::move(other._solution))
    , _workspaceSize(other._workspaceSize)
{
    other._workspaceSize = 0;
}

ConvFwdPlan& ConvFwdPlan::operator=(ConvFwdPlan&& other) noexcept
{
    if(this != &other)
    {
        _params = std::move(other._params);
        _solution = std::move(other._solution);
        _workspaceSize = other._workspaceSize;
        other._workspaceSize = 0;
    }
    return *this;
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
    auto xDesc = _params.x().tensorDescriptor();
    auto wDesc = _params.w().tensorDescriptor();
    auto yDesc = _params.y().tensorDescriptor();

    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.x().uid(), deviceBuffers, numDeviceBuffers);
    auto wBuffer
        = miopen_utils::findDeviceBuffer(_params.w().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = miopen_utils::findDeviceBuffer(_params.y().uid(), deviceBuffers, numDeviceBuffers);

    std::array<miopenTensorArgument_t, 3> tensors
        = {miopenTensorArgument_t{miopenTensorConvolutionX, &xDesc, xBuffer.ptr},
           miopenTensorArgument_t{miopenTensorConvolutionW, &wDesc, wBuffer.ptr},
           miopenTensorArgument_t{miopenTensorConvolutionY, &yDesc, yBuffer.ptr}};

    size_t workspaceSize = 0;
    if(workspace != nullptr)
    {
        // Assume the provided workspace is large enough
        workspaceSize = _workspaceSize;
    }

    THROW_ON_MIOPEN_FAILURE(miopenRunSolution(handle.miopenHandle,
                                              _solution.get(),
                                              tensors.size(),
                                              tensors.data(),
                                              workspace,
                                              workspaceSize));
}

} // namespace miopen_legacy_plugin
