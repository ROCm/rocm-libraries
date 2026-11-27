// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ResampleFwdKernel.hpp"

#include "compilation/KernelCompileOptions.hpp"
#include "core/Utils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/PlanUtils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/resample/ResamplePlanUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>

namespace hip_kernel_provider::resample
{
using namespace hip_kernel_provider::compilation;
using namespace hip_kernel_provider::core::utils;

namespace
{

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

std::vector<KernelLaunchShape> resampleFwdLaunchShapes(int64_t numElements, int64_t blockSize)
{
    return {{"ResampleFwd",
             numElements / blockSize + (numElements % blockSize != 0 ? 1 : 0),
             blockSize}};
}

bool isResampleFwdConfigApplicable(const ResampleFwdParams& params, const ResampleConfig& config)
{
    if(!isBlockSizeSupported(config))
    {
        return false;
    }
    uint64_t totalElements = 1;
    for(auto dim : *params.y()->dims())
    {
        totalElements *= static_cast<uint64_t>(dim);
    }
    const auto shapes
        = resampleFwdLaunchShapes(static_cast<int64_t>(totalElements), config.blockSize);
    return std::all_of(shapes.begin(), shapes.end(), [](const KernelLaunchShape& shape) {
        return launchFitsThreadLimit(shape.blocks, shape.threadsPerBlock);
    });
}

ResampleKernelBuild buildResampleFwdKernel(const ResampleFwdParams& params,
                                           const std::string& gcnArchName,
                                           const ResampleConfig& config)
{
    const auto rank = params.x()->dims()->size();
    const auto xDims = tensorDims(*params.x());
    const auto yDims = tensorDims(*params.y());
    validateResampleOutputShape(
        xDims, yDims, params.prePadding(), params.postPadding(), params.stride(), params.window());
    if(params.index() != nullptr)
    {
        validateResampleIndexShape(*params.index(), yDims, "ResampleFwd");
    }

    uint64_t totalElements = 1;
    for(auto dim : *params.y()->dims())
    {
        totalElements *= static_cast<uint64_t>(dim);
    }
    const auto shapes
        = resampleFwdLaunchShapes(static_cast<int64_t>(totalElements), config.blockSize);
    for(const auto& shape : shapes)
    {
        checkLaunchFitsThreadLimit(shape);
    }

    const size_t spatialDims = rank - 2;
    const std::string inputTypeString = getKernelParamTypeString(params.x()->data_type());
    const std::string outputTypeString = getKernelParamTypeString(params.y()->data_type());
    const std::string computeTypeString = getKernelParamTypeString(params.computeDataType());
    const std::string indexTypeString = getIndexTypeString(params.index());

    KernelCompileOptions options(params.x(), gcnArchName);
    options.add("HIP_PLUGIN_RESAMPLE_INPUT_TYPE", inputTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_OUTPUT_TYPE", outputTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_COMPUTE_TYPE", computeTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_INDEX_TYPE", indexTypeString);
    options.add("HIP_PLUGIN_RESAMPLE_SPATIAL_DIMS", static_cast<int64_t>(spatialDims));
    options.add("HIP_PLUGIN_RESAMPLE_MODE", static_cast<int64_t>(params.mode()));
    options.add("HIP_PLUGIN_RESAMPLE_PADDING_MODE", static_cast<int64_t>(params.paddingMode()));
    options.add("HIP_PLUGIN_RESAMPLE_HAS_INDEX", params.index() != nullptr);
    options.add("HIP_PLUGIN_RESAMPLE_GENERATE_INDEX", params.generateIndex());
    options.add("HIP_PLUGIN_RESAMPLE_OUTPUT_ELEMENT_COUNT", totalElements);

    addDimOptions(options, params, spatialDims);
    addStrideOptions(options, params, spatialDims);
    addSpatialOptions(options, params, spatialDims);

    return {std::move(options), config.blockSize, static_cast<unsigned int>(shapes[0].blocks)};
}

} // namespace hip_kernel_provider::resample
