// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResampleBwdPlan.hpp"

#include "compilation/KernelCompileOptions.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResamplePlanUtils.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <cstdint>
#include <limits>
#include <numeric>

namespace hip_kernel_provider::resample
{
using namespace hip_kernel_provider::compilation;
using namespace hip_kernel_provider::core::utils;

namespace
{

void addDimOptions(KernelCompileOptions& options,
                   const ResampleBwdParams& params,
                   size_t spatialDims)
{
    options.add("HIP_PLUGIN_RESAMPLE_N", dimAt(params.dx(), 0));
    options.add("HIP_PLUGIN_RESAMPLE_C", dimAt(params.dx(), 1));

    if(spatialDims == 2)
    {
        options.add("HIP_PLUGIN_RESAMPLE_DX_D", 1);
        options.add("HIP_PLUGIN_RESAMPLE_DX_H", dimAt(params.dx(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_DX_W", dimAt(params.dx(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_DY_D", 1);
        options.add("HIP_PLUGIN_RESAMPLE_DY_H", dimAt(params.dy(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_DY_W", dimAt(params.dy(), 3));
    }
    else
    {
        options.add("HIP_PLUGIN_RESAMPLE_DX_D", dimAt(params.dx(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_DX_H", dimAt(params.dx(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_DX_W", dimAt(params.dx(), 4));
        options.add("HIP_PLUGIN_RESAMPLE_DY_D", dimAt(params.dy(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_DY_H", dimAt(params.dy(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_DY_W", dimAt(params.dy(), 4));
    }
}

void addStrideOptions(KernelCompileOptions& options,
                      const ResampleBwdParams& params,
                      size_t spatialDims)
{
    options.add("HIP_PLUGIN_RESAMPLE_DX_STRIDE_N", strideAt(params.dx(), 0));
    options.add("HIP_PLUGIN_RESAMPLE_DX_STRIDE_C", strideAt(params.dx(), 1));
    options.add("HIP_PLUGIN_RESAMPLE_DY_STRIDE_N", strideAt(params.dy(), 0));
    options.add("HIP_PLUGIN_RESAMPLE_DY_STRIDE_C", strideAt(params.dy(), 1));

    const auto* index = params.index();
    options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_N", index == nullptr ? 0 : strideAt(index, 0));
    options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_C", index == nullptr ? 0 : strideAt(index, 1));

    if(spatialDims == 2)
    {
        options.add("HIP_PLUGIN_RESAMPLE_DX_STRIDE_D", 0);
        options.add("HIP_PLUGIN_RESAMPLE_DX_STRIDE_H", strideAt(params.dx(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_DX_STRIDE_W", strideAt(params.dx(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_DY_STRIDE_D", 0);
        options.add("HIP_PLUGIN_RESAMPLE_DY_STRIDE_H", strideAt(params.dy(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_DY_STRIDE_W", strideAt(params.dy(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_D", 0);
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_H",
                    index == nullptr ? 0 : strideAt(index, 2));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_W",
                    index == nullptr ? 0 : strideAt(index, 3));
    }
    else
    {
        options.add("HIP_PLUGIN_RESAMPLE_DX_STRIDE_D", strideAt(params.dx(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_DX_STRIDE_H", strideAt(params.dx(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_DX_STRIDE_W", strideAt(params.dx(), 4));
        options.add("HIP_PLUGIN_RESAMPLE_DY_STRIDE_D", strideAt(params.dy(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_DY_STRIDE_H", strideAt(params.dy(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_DY_STRIDE_W", strideAt(params.dy(), 4));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_D",
                    index == nullptr ? 0 : strideAt(index, 2));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_H",
                    index == nullptr ? 0 : strideAt(index, 3));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_W",
                    index == nullptr ? 0 : strideAt(index, 4));
    }
}

void addSpatialOptions(KernelCompileOptions& options,
                       const ResampleBwdParams& params,
                       size_t spatialDims)
{
    if(spatialDims == 2)
    {
        options.add("HIP_PLUGIN_RESAMPLE_PRE_PAD_D", 0);
        options.add("HIP_PLUGIN_RESAMPLE_PRE_PAD_H", params.prePadding()[0]);
        options.add("HIP_PLUGIN_RESAMPLE_PRE_PAD_W", params.prePadding()[1]);
        options.add("HIP_PLUGIN_RESAMPLE_STRIDE_D", 1);
        options.add("HIP_PLUGIN_RESAMPLE_STRIDE_H", params.stride()[0]);
        options.add("HIP_PLUGIN_RESAMPLE_STRIDE_W", params.stride()[1]);
        options.add("HIP_PLUGIN_RESAMPLE_WINDOW_D", 1);
        options.add("HIP_PLUGIN_RESAMPLE_WINDOW_H", params.window()[0]);
        options.add("HIP_PLUGIN_RESAMPLE_WINDOW_W", params.window()[1]);
    }
    else
    {
        options.add("HIP_PLUGIN_RESAMPLE_PRE_PAD_D", params.prePadding()[0]);
        options.add("HIP_PLUGIN_RESAMPLE_PRE_PAD_H", params.prePadding()[1]);
        options.add("HIP_PLUGIN_RESAMPLE_PRE_PAD_W", params.prePadding()[2]);
        options.add("HIP_PLUGIN_RESAMPLE_STRIDE_D", params.stride()[0]);
        options.add("HIP_PLUGIN_RESAMPLE_STRIDE_H", params.stride()[1]);
        options.add("HIP_PLUGIN_RESAMPLE_STRIDE_W", params.stride()[2]);
        options.add("HIP_PLUGIN_RESAMPLE_WINDOW_D", params.window()[0]);
        options.add("HIP_PLUGIN_RESAMPLE_WINDOW_H", params.window()[1]);
        options.add("HIP_PLUGIN_RESAMPLE_WINDOW_W", params.window()[2]);
    }
}

} // namespace

ResampleBwdParams::ResampleBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
    : _dy(tensorMap.at(attributes.dy_tensor_uid()))
    , _dx(tensorMap.at(attributes.dx_tensor_uid()))
    , _index(attributes.index_tensor_uid().has_value()
                 ? tensorMap.at(attributes.index_tensor_uid().value())
                 : nullptr)
    , _prePadding(toStdVector(attributes.pre_padding()))
    , _postPadding(toStdVector(attributes.post_padding()))
    , _stride(toStdVector(attributes.stride()))
    , _window(toStdVector(attributes.window()))
    , _resampleMode(attributes.resample_mode())
    , _paddingMode(attributes.padding_mode())
    , _computeDataType(computeDataType)
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ResampleBwdParams::dy() const
{
    return _dy;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ResampleBwdParams::dx() const
{
    return _dx;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* ResampleBwdParams::index() const
{
    return _index;
}

const std::vector<int64_t>& ResampleBwdParams::prePadding() const
{
    return _prePadding;
}

const std::vector<int64_t>& ResampleBwdParams::postPadding() const
{
    return _postPadding;
}

const std::vector<int64_t>& ResampleBwdParams::stride() const
{
    return _stride;
}

const std::vector<int64_t>& ResampleBwdParams::window() const
{
    return _window;
}

hipdnn_flatbuffers_sdk::data_objects::ResampleMode ResampleBwdParams::resampleMode() const
{
    return _resampleMode;
}

hipdnn_flatbuffers_sdk::data_objects::PaddingMode ResampleBwdParams::paddingMode() const
{
    return _paddingMode;
}

hipdnn_flatbuffers_sdk::data_objects::DataType ResampleBwdParams::computeDataType() const
{
    return _computeDataType;
}

ResampleBwdPlan::ResampleBwdPlan(ResampleBwdParams&& params)
    : _params(std::move(params))
{
}

size_t ResampleBwdPlan::getWorkspaceSize([[maybe_unused]] const Handle& handle) const
{
    // No workspace needed for resample backward
    return 0;
}

void ResampleBwdPlan::compile([[maybe_unused]] const IKernelCompiler& kernelCompiler,
                              [[maybe_unused]] const hipDeviceProp_t& deviceProperties)
{
    const auto spatialDims = _params.dy()->dims()->size() - 2;
    const auto dyDims = tensorDims(*_params.dy());
    const auto dxDims = tensorDims(*_params.dx());

    validateResampleBwdOutputShape(dyDims,
                                   dxDims,
                                   _params.prePadding(),
                                   _params.postPadding(),
                                   _params.stride(),
                                   _params.window());
    if(_params.index() != nullptr)
    {
        validateResampleIndexShape(*_params.index(), dyDims, "ResampleBwd");
    }

    constexpr uint64_t BLOCK_SIZE = 256;
    const auto numDxElements
        = std::accumulate(dxDims.begin(), dxDims.end(), 1ULL, std::multiplies<>());
    const auto gridSize = (numDxElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if(gridSize > std::numeric_limits<unsigned int>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleBwd output is too large for one kernel launch.");
    }

    const std::string dyTypeString = getKernelParamTypeString(_params.dy()->data_type());
    const std::string dxTypeString = getKernelParamTypeString(_params.dx()->data_type());
    const std::string computeTypeString = getKernelParamTypeString(_params.computeDataType());
    const std::string indexTypeString = getIndexTypeString(_params.index());

    KernelCompileOptions options(_params.dy(), deviceProperties);
    options.add("HIP_PLUGIN_RESAMPLE_DY_TYPE", dyTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_DX_TYPE", dxTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_COMPUTE_TYPE", computeTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_INDEX_TYPE", indexTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_SPATIAL_DIMS", spatialDims);
    options.add("HIP_PLUGIN_RESAMPLE_MODE", static_cast<int64_t>(_params.resampleMode()));
    options.add("HIP_PLUGIN_RESAMPLE_DX_ELEMENT_COUNT", numDxElements);

    addDimOptions(options, _params, spatialDims);
    addStrideOptions(options, _params, spatialDims);
    addSpatialOptions(options, _params, spatialDims);

    _compiledProgram = kernelCompiler.compile("ResampleBwd.cpp", options);
    _runnableKernel = _compiledProgram->getKernel("ResampleBwd");
    _runnableKernel->setBlockSize(static_cast<unsigned int>(BLOCK_SIZE), 1, 1);
    _runnableKernel->setGridSize(static_cast<unsigned int>(gridSize), 1, 1);
}

void ResampleBwdPlan::execute([[maybe_unused]] const Handle& handle,
                              [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              [[maybe_unused]] uint32_t numDeviceBuffers,
                              [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "ResampleBwdPlan::execute() called before compile()");
    }

    auto dyBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.dy()->uid(), deviceBuffers, numDeviceBuffers);
    auto dxBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.dx()->uid(), deviceBuffers, numDeviceBuffers);
    void* indexBufferPtr = nullptr;
    if(_params.index() != nullptr)
    {
        indexBufferPtr = hipdnn_plugin_sdk::findDeviceBuffer(
                             _params.index()->uid(), deviceBuffers, numDeviceBuffers)
                             .ptr;
    }

    _runnableKernel->launch(handle.getStream(), dyBuffer.ptr, indexBufferPtr, dxBuffer.ptr);
}

} // namespace hip_kernel_provider::resample
