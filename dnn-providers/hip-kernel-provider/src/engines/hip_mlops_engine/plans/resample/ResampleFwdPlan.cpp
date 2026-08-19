// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResampleFwdPlan.hpp"

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
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

int64_t dimAt(const data_objects::TensorAttributes* tensor, size_t index)
{
    return tensor->dims()->Get(static_cast<flatbuffers::uoffset_t>(index));
}

int64_t strideAt(const data_objects::TensorAttributes* tensor, size_t index)
{
    return tensor->strides()->Get(static_cast<flatbuffers::uoffset_t>(index));
}

const char* getIndexTypeString(const data_objects::TensorAttributes* index)
{
    if(index == nullptr || index->data_type() == data_objects::DataType::INT32)
    {
        return "int32_t";
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "ResampleFwd index tensor must have INT32 data type.");
}

void addDimOptions(KernelCompileOptions& options,
                   const ResampleFwdParams& params,
                   size_t spatialDims)
{
    options.add("HIP_PLUGIN_RESAMPLE_N", dimAt(params.x(), 0));
    options.add("HIP_PLUGIN_RESAMPLE_C", dimAt(params.x(), 1));

    if(spatialDims == 2)
    {
        options.add("HIP_PLUGIN_RESAMPLE_X_D", 1);
        options.add("HIP_PLUGIN_RESAMPLE_X_H", dimAt(params.x(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_X_W", dimAt(params.x(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_Y_D", 1);
        options.add("HIP_PLUGIN_RESAMPLE_Y_H", dimAt(params.y(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_Y_W", dimAt(params.y(), 3));
    }
    else
    {
        options.add("HIP_PLUGIN_RESAMPLE_X_D", dimAt(params.x(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_X_H", dimAt(params.x(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_X_W", dimAt(params.x(), 4));
        options.add("HIP_PLUGIN_RESAMPLE_Y_D", dimAt(params.y(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_Y_H", dimAt(params.y(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_Y_W", dimAt(params.y(), 4));
    }
}

void addStrideOptions(KernelCompileOptions& options,
                      const ResampleFwdParams& params,
                      size_t spatialDims)
{
    options.add("HIP_PLUGIN_RESAMPLE_X_STRIDE_N", strideAt(params.x(), 0));
    options.add("HIP_PLUGIN_RESAMPLE_X_STRIDE_C", strideAt(params.x(), 1));
    options.add("HIP_PLUGIN_RESAMPLE_Y_STRIDE_N", strideAt(params.y(), 0));
    options.add("HIP_PLUGIN_RESAMPLE_Y_STRIDE_C", strideAt(params.y(), 1));

    const auto* index = params.index();
    options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_N", index == nullptr ? 0 : strideAt(index, 0));
    options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_C", index == nullptr ? 0 : strideAt(index, 1));

    if(spatialDims == 2)
    {
        options.add("HIP_PLUGIN_RESAMPLE_X_STRIDE_D", 0);
        options.add("HIP_PLUGIN_RESAMPLE_X_STRIDE_H", strideAt(params.x(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_X_STRIDE_W", strideAt(params.x(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_Y_STRIDE_D", 0);
        options.add("HIP_PLUGIN_RESAMPLE_Y_STRIDE_H", strideAt(params.y(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_Y_STRIDE_W", strideAt(params.y(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_D", 0);
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_H",
                    index == nullptr ? 0 : strideAt(index, 2));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_W",
                    index == nullptr ? 0 : strideAt(index, 3));
    }
    else
    {
        options.add("HIP_PLUGIN_RESAMPLE_X_STRIDE_D", strideAt(params.x(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_X_STRIDE_H", strideAt(params.x(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_X_STRIDE_W", strideAt(params.x(), 4));
        options.add("HIP_PLUGIN_RESAMPLE_Y_STRIDE_D", strideAt(params.y(), 2));
        options.add("HIP_PLUGIN_RESAMPLE_Y_STRIDE_H", strideAt(params.y(), 3));
        options.add("HIP_PLUGIN_RESAMPLE_Y_STRIDE_W", strideAt(params.y(), 4));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_D",
                    index == nullptr ? 0 : strideAt(index, 2));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_H",
                    index == nullptr ? 0 : strideAt(index, 3));
        options.add("HIP_PLUGIN_RESAMPLE_INDEX_STRIDE_W",
                    index == nullptr ? 0 : strideAt(index, 4));
    }
}

void addSpatialOptions(KernelCompileOptions& options,
                       const ResampleFwdParams& params,
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

ResampleFwdParams::ResampleFwdParams(
    const data_objects::ResampleFwdAttributes& attributes,
    const std::unordered_map<int64_t, const data_objects::TensorAttributes*>& tensorMap,
    data_objects::DataType computeDataType)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _index(attributes.index_tensor_uid().has_value()
                 ? tensorMap.at(attributes.index_tensor_uid().value())
                 : nullptr)
    , _prePadding(toStdVector(attributes.pre_padding()))
    , _postPadding(toStdVector(attributes.post_padding()))
    , _stride(toStdVector(attributes.stride()))
    , _window(toStdVector(attributes.window()))
    , _mode(attributes.resample_mode())
    , _paddingMode(attributes.padding_mode())
    , _computeDataType(computeDataType)
{
}

const data_objects::TensorAttributes* ResampleFwdParams::x() const
{
    return _x;
}

const data_objects::TensorAttributes* ResampleFwdParams::y() const
{
    return _y;
}

const data_objects::TensorAttributes* ResampleFwdParams::index() const
{
    return _index;
}

const std::vector<int64_t>& ResampleFwdParams::prePadding() const
{
    return _prePadding;
}

const std::vector<int64_t>& ResampleFwdParams::postPadding() const
{
    return _postPadding;
}

const std::vector<int64_t>& ResampleFwdParams::stride() const
{
    return _stride;
}

const std::vector<int64_t>& ResampleFwdParams::window() const
{
    return _window;
}

data_objects::ResampleMode ResampleFwdParams::mode() const
{
    return _mode;
}

data_objects::PaddingMode ResampleFwdParams::paddingMode() const
{
    return _paddingMode;
}

bool ResampleFwdParams::generateIndex() const
{
    return _index != nullptr;
}

data_objects::DataType ResampleFwdParams::computeDataType() const
{
    return _computeDataType;
}

ResampleFwdPlan::ResampleFwdPlan(ResampleFwdParams&& params)
    : _params(std::move(params))
{
}

size_t ResampleFwdPlan::getWorkspaceSize([[maybe_unused]] const Handle& handle) const
{
    return 0;
}

uint64_t ResampleFwdPlan::outputElementCount() const
{
    const auto* yDims = _params.y()->dims();
    uint64_t total = 1;
    for(auto dim : *yDims)
    {
        total *= static_cast<uint64_t>(dim);
    }
    return total;
}

void ResampleFwdPlan::compile(const IKernelCompiler& kernelCompiler,
                              const hipDeviceProp_t& deviceProperties)
{
    const auto rank = _params.x()->dims()->size();
    const auto xDims = tensorDims(*_params.x());
    const auto yDims = tensorDims(*_params.y());
    validateResampleOutputShape(xDims,
                                yDims,
                                _params.prePadding(),
                                _params.postPadding(),
                                _params.stride(),
                                _params.window(),
                                "ResampleFwd");
    if(_params.index() != nullptr)
    {
        validateResampleIndexShape(*_params.index(), yDims, "ResampleFwd");
    }

    const auto totalElements = outputElementCount();
    constexpr uint64_t BLOCK_SIZE = 256;
    const auto gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if(gridSize > std::numeric_limits<unsigned int>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ResampleFwd output is too large for one kernel launch.");
    }

    const size_t spatialDims = rank - 2;
    const std::string inputTypeString = getKernelParamTypeString(_params.x()->data_type());
    const std::string outputTypeString = getKernelParamTypeString(_params.y()->data_type());
    const std::string computeTypeString = getKernelParamTypeString(_params.computeDataType());
    const std::string indexTypeString = getIndexTypeString(_params.index());

    KernelCompileOptions options(_params.x(), deviceProperties);
    options.add("HIP_PLUGIN_RESAMPLE_INPUT_TYPE", inputTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_OUTPUT_TYPE", outputTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_COMPUTE_TYPE", computeTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_INDEX_TYPE", indexTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_SPATIAL_DIMS", static_cast<int64_t>(spatialDims));
    options.add("HIP_PLUGIN_RESAMPLE_MODE", static_cast<int64_t>(_params.mode()));
    options.add("HIP_PLUGIN_RESAMPLE_PADDING_MODE", static_cast<int64_t>(_params.paddingMode()));
    options.add("HIP_PLUGIN_RESAMPLE_HAS_INDEX", _params.index() != nullptr);
    options.add("HIP_PLUGIN_RESAMPLE_GENERATE_INDEX", _params.generateIndex());
    options.add("HIP_PLUGIN_RESAMPLE_OUTPUT_ELEMENT_COUNT", totalElements);

    addDimOptions(options, _params, spatialDims);
    addStrideOptions(options, _params, spatialDims);
    addSpatialOptions(options, _params, spatialDims);

    _compiledProgram = kernelCompiler.compile("ResampleFwd.cpp", options);
    _runnableKernel = _compiledProgram->getKernel("ResampleFwd");
    _runnableKernel->setBlockSize(static_cast<unsigned int>(BLOCK_SIZE), 1, 1);
    _runnableKernel->setGridSize(static_cast<unsigned int>(gridSize), 1, 1);
}

void ResampleFwdPlan::execute(const Handle& handle,
                              const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                              uint32_t numDeviceBuffers,
                              [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "ResampleFwdPlan::execute() called before compile()");
    }

    auto xBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.y()->uid(), deviceBuffers, numDeviceBuffers);

    void* indexBufferPtr = nullptr;
    if(_params.index() != nullptr)
    {
        indexBufferPtr = hipdnn_plugin_sdk::findDeviceBuffer(
                             _params.index()->uid(), deviceBuffers, numDeviceBuffers)
                             .ptr;
    }

    _runnableKernel->launch(handle.getStream(), xBuffer.ptr, yBuffer.ptr, indexBufferPtr);
}

} // namespace hip_kernel_provider::resample
