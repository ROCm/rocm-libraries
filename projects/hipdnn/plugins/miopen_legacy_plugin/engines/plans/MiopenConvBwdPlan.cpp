// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>

#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/utilities/ScopedResource.hpp>

#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenConvBwdPlan.hpp"
#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

ConvBwdParams::ConvBwdParams(
    const hipdnn_sdk::data_objects::ConvolutionBwdAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, attributes.dx_tensor_uid())))
    , _dx(miopen_utils::createTensor(tensorMap, attributes.dx_tensor_uid()))
    , _w(miopen_utils::createTensor(tensorMap, attributes.w_tensor_uid()))
    , _dy(miopen_utils::createTensor(tensorMap, attributes.dy_tensor_uid()))
    , _conv(_spatialDimCount, attributes)
{
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

ConvBwdPlan::ConvBwdPlan(const HipdnnEnginePluginHandle& handle, ConvBwdParams&& params)
    : _params(std::move(params))
{
    // MIOpen Find 2.0 API
    miopenProblem_t problem;
    THROW_ON_MIOPEN_FAILURE(miopenCreateConvProblem(
        &problem, _params.conv().convDescriptor(), miopenProblemDirectionBackward));
    hipdnn_sdk::utilities::ScopedResource problemRes(
        problem, [](miopenProblem_t p) { std::ignore = miopenDestroyProblem(p); });

    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionX, _params.dx().tensorDescriptor()));
    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionW, _params.w().tensorDescriptor()));
    THROW_ON_MIOPEN_FAILURE(miopenSetProblemTensorDescriptor(
        problem, miopenTensorConvolutionY, _params.dy().tensorDescriptor()));

    size_t numSolutions;
    // Requesting only the best solution
    THROW_ON_MIOPEN_FAILURE(
        miopenFindSolutions(handle.miopenHandle, problem, nullptr, &_solution, &numSolutions, 1));

    if(numSolutions != 1)
    {
        throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "miopenFindSolutions returned no solutions");
    }
}

ConvBwdPlan::~ConvBwdPlan()
{
    if(_solution != nullptr)
    {
        auto status = miopenDestroySolution(_solution);
        if(status != miopenStatusSuccess)
        {
            HIPDNN_LOG_ERROR("miopenDestroySolution failed in ConvBwdPlan destructor");
        }
    }
}

ConvBwdPlan::ConvBwdPlan(ConvBwdPlan&& other) noexcept
    : _params(std::move(other._params))
    , _solution(other._solution)
{
    other._solution = nullptr;
}

ConvBwdPlan& ConvBwdPlan::operator=(ConvBwdPlan&& other) noexcept
{
    if(this != &other)
    {
        if(_solution != nullptr)
        {
            auto status = miopenDestroySolution(_solution);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_LOG_ERROR("miopenDestroySolution failed in ConvBwdPlan move assignment");
            }
        }
        _params = std::move(other._params);
        _solution = other._solution;
        other._solution = nullptr;
    }
    return *this;
}

void ConvBwdPlan::execute(const HipdnnEnginePluginHandle& handle,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers,
                          void* workspace) const
{
    auto dxDesc = _params.dx().tensorDescriptor();
    auto wDesc = _params.w().tensorDescriptor();
    auto dyDesc = _params.dy().tensorDescriptor();

    auto xBuffer
        = miopen_utils::findDeviceBuffer(_params.dx().uid(), deviceBuffers, numDeviceBuffers);
    auto wBuffer
        = miopen_utils::findDeviceBuffer(_params.w().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = miopen_utils::findDeviceBuffer(_params.dy().uid(), deviceBuffers, numDeviceBuffers);

    std::array<miopenTensorArgument_t, 3> tensors
        = {miopenTensorArgument_t{miopenTensorConvolutionX, &dxDesc, xBuffer.ptr},
           miopenTensorArgument_t{miopenTensorConvolutionW, &wDesc, wBuffer.ptr},
           miopenTensorArgument_t{miopenTensorConvolutionY, &dyDesc, yBuffer.ptr}};

    size_t workspaceSize = 0;
    if(workspace != nullptr)
    {
        // Assume the provided workspace is large enough
        THROW_ON_MIOPEN_FAILURE(miopenGetSolutionWorkspaceSize(_solution, &workspaceSize));
    }

    THROW_ON_MIOPEN_FAILURE(miopenRunSolution(
        handle.miopenHandle, _solution, tensors.size(), tensors.data(), workspace, workspaceSize));
}

} // namespace miopen_legacy_plugin
