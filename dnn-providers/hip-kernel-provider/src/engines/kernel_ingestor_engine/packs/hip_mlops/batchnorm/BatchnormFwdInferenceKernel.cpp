// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdInferenceKernel.hpp"

#include <algorithm>

#include "BatchnormKernelCompileOptions.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/PlanUtils.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::batchnorm
{

BatchnormFwdInferenceParams::BatchnormFwdInferenceParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(tensorMap.at(attributes.bias_tensor_uid()))
    , _estMean(tensorMap.at(attributes.mean_tensor_uid()))
    , _invVariance(tensorMap.at(attributes.inv_variance_tensor_uid()))
    , _activationOut(nullptr)
{
}

BatchnormFwdInferenceParams::BatchnormFwdInferenceParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& inferenceAttributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(inferenceAttributes.x_tensor_uid()))
    , _y(tensorMap.at(inferenceAttributes.y_tensor_uid()))
    , _scale(tensorMap.at(inferenceAttributes.scale_tensor_uid()))
    , _bias(tensorMap.at(inferenceAttributes.bias_tensor_uid()))
    , _estMean(tensorMap.at(inferenceAttributes.mean_tensor_uid()))
    , _invVariance(tensorMap.at(inferenceAttributes.inv_variance_tensor_uid()))
    , _optActivation(parseActivation(pointwiseAttributes))
    , _activationOut(tensorMap.at(pointwiseAttributes.out_0_tensor_uid()))
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormFwdInferenceParams::y() const
{
    return _y;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::bias() const
{
    return _bias;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::estMean() const
{
    return _estMean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::invVariance() const
{
    return _invVariance;
}

const std::optional<ActivationParams>& BatchnormFwdInferenceParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdInferenceParams::activationOut() const
{
    return _activationOut;
}

namespace
{

// The dimensions and strides of x, its spatial dimensions combined.
struct InferenceShape
{
    int batches = 0;
    int channels = 0;
    int spatialSize = 0;
    int batchStride = 0;
    int channelStride = 0;
    int spatialStride = 0;
};

InferenceShape inferenceShape(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& x)
{
    const auto* xDims = x.dims();
    const auto* xStrides = x.strides();

    InferenceShape shape;

    // Check if 3D (NCL/NLC), 4D (NCHW/NHWC) or 5D (NCDHW/NDHWC)
    if(xDims->size() == 3)
    {
        shape.batches = checkedNarrowToInt(xDims->Get(0), "batches");
        shape.channels = checkedNarrowToInt(xDims->Get(1), "channels");
        shape.spatialSize = checkedNarrowToInt(xDims->Get(2), "spatialSize");

        shape.batchStride = checkedNarrowToInt(xStrides->Get(0), "batchStride");
        shape.channelStride = checkedNarrowToInt(xStrides->Get(1), "channelStride");
        shape.spatialStride = checkedNarrowToInt(xStrides->Get(2), "spatialStride");
    }
    else if(xDims->size() == 4)
    {
        shape.batches = checkedNarrowToInt(xDims->Get(0), "batches");
        shape.channels = checkedNarrowToInt(xDims->Get(1), "channels");
        auto height = checkedNarrowToInt(xDims->Get(2), "height");
        auto width = checkedNarrowToInt(xDims->Get(3), "width");
        // For 4D, combine H*W into spatial dimension
        shape.spatialSize = height * width;

        shape.batchStride = checkedNarrowToInt(xStrides->Get(0), "batchStride");
        shape.channelStride = checkedNarrowToInt(xStrides->Get(1), "channelStride");
        shape.spatialStride = checkedNarrowToInt(xStrides->Get(3), "spatialStride");
    }
    else if(xDims->size() == 5)
    {
        shape.batches = checkedNarrowToInt(xDims->Get(0), "batches");
        shape.channels = checkedNarrowToInt(xDims->Get(1), "channels");
        auto depth = checkedNarrowToInt(xDims->Get(2), "depth");
        auto height = checkedNarrowToInt(xDims->Get(3), "height");
        auto width = checkedNarrowToInt(xDims->Get(4), "width");
        // For 5D, combine D*H*W into spatial dimension
        shape.spatialSize = depth * height * width;

        shape.batchStride = checkedNarrowToInt(xStrides->Get(0), "batchStride");
        shape.channelStride = checkedNarrowToInt(xStrides->Get(1), "channelStride");
        shape.spatialStride = checkedNarrowToInt(xStrides->Get(4), "spatialStride");
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    return shape;
}

} // namespace

BatchnormInferenceConfig
    defaultBatchnormFwdInferenceConfig(const BatchnormFwdInferenceParams& params)
{
    const auto shape = inferenceShape(*params.x());
    return defaultInferenceConfig(isChannelLastLayout(params.x()),
                                  shape.channels,
                                  static_cast<unsigned int>(shape.spatialSize));
}

bool isBatchnormFwdInferenceConfigApplicable(const BatchnormFwdInferenceParams& params,
                                             const BatchnormInferenceConfig& config)
{
    const auto shape = inferenceShape(*params.x());
    return isInferenceConfigApplicable(isChannelLastLayout(params.x()),
                                       shape.channels,
                                       static_cast<unsigned int>(shape.spatialSize),
                                       config);
}

BatchnormInferenceKernelBuild
    buildBatchnormFwdInferenceKernel(const BatchnormFwdInferenceParams& params,
                                     const std::string& gcnArchName,
                                     int multiProcessorCount,
                                     int warpSize,
                                     const BatchnormInferenceConfig& config)
{
    // Determine data type configuration
    auto xDataType = params.x()->data_type();
    auto yDataType = params.optActivation().has_value() ? params.activationOut()->data_type()
                                                        : params.y()->data_type();
    auto scaleDataType = params.scale()->data_type();

    const bool useFp16Mix
        = xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF
          && yDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF
          && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    const bool useBfp16Mix
        = xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16
          && yDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16
          && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    const bool useFp16Upcast
        = xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF
          && yDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT
          && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    const bool useBfp16Upcast
        = xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16
          && yDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT
          && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;

    // Extract dimensions from x tensor
    const auto shape = inferenceShape(*params.x());
    const int batches = shape.batches;
    const int channels = shape.channels;
    const int spatialSize = shape.spatialSize;
    const int batchStride = shape.batchStride;
    const int channelStride = shape.channelStride;
    const int spatialStride = shape.spatialStride;

    auto inSpatialSize = static_cast<unsigned int>(spatialSize);

    const bool isLayoutChannelLast = isChannelLastLayout(params.x());

    // The vector size along C (channel last) or the spatial dimension
    const size_t vectorsize = config.vectorSize;

    // Calculate block and grid dimensions
    size_t xlocalsize = 0;
    size_t xgridsize = 0;
    size_t ylocalsize = 0;
    size_t ygridsize = 0;
    size_t zlocalsize = 0;
    size_t zgridsize = 0;
    const size_t maxLocalsize = 256;

    if(isLayoutChannelLast)
    {
        xlocalsize = std::min(static_cast<size_t>(channels) / vectorsize, maxLocalsize);
        xgridsize = ((static_cast<size_t>(channels) / vectorsize) + xlocalsize - 1) / xlocalsize
                    * xlocalsize;

        ylocalsize = maxLocalsize / xlocalsize;
        ygridsize = (inSpatialSize + ylocalsize - 1) / ylocalsize * ylocalsize;
    }
    else
    {
        xlocalsize = 1;
        xgridsize = ((static_cast<size_t>(channels) + xlocalsize - 1) / xlocalsize) * xlocalsize;

        ylocalsize = maxLocalsize;
        ygridsize = ((inSpatialSize / vectorsize + ylocalsize - 1) / ylocalsize) * ylocalsize;
    }

    zlocalsize = 1;
    const size_t activeThreadsXy = xgridsize * ygridsize;
    auto maxActiveThreads
        = static_cast<size_t>(multiProcessorCount) * 32 * static_cast<size_t>(warpSize);

    if(activeThreadsXy < maxActiveThreads)
    {
        zgridsize = std::min(maxActiveThreads / activeThreadsXy, static_cast<size_t>(batches));
    }
    else
    {
        zgridsize = 1;
    }

    // Get activation mode
    auto activationMode = ActivationMode::PASTHRU;
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* outputTensor = params.y();
    if(params.optActivation().has_value() && params.activationOut() != nullptr)
    {
        activationMode = (*params.optActivation()).mode;
        outputTensor = params.activationOut();
    }

    // Prepare compilation options
    BatchnormKernelCompileOptions options(
        params.x(), outputTensor, params.estMean(), params.scale(), gcnArchName, activationMode);
    options.update("HIP_PLUGIN_USE_FPMIX", useFp16Mix);
    options.update("HIP_PLUGIN_USE_BFPMIX", useBfp16Mix);
    options.update("HIP_PLUGIN_USE_FPUPCAST", useFp16Upcast);
    options.update("HIP_PLUGIN_USE_BFPUPCAST", useBfp16Upcast);
    options.update("HIP_PLUGIN_BN_GRP0", xlocalsize);
    options.update("HIP_PLUGIN_BN_GRP1", ylocalsize);
    options.update("HIP_PLUGIN_BN_GRP2", zlocalsize);
    options.update("HIP_PLUGIN_BN_VEC_SIZE", vectorsize);
    options.update("HIP_PLUGIN_BN_BATCHES", batches);
    options.update("HIP_PLUGIN_BN_CHANNELS", channels);
    options.update("HIP_PLUGIN_BN_SPATIAL_SIZE", spatialSize);

    return {std::move(options),
            "BatchNormFwdInferSpatialEstInvVar",
            {static_cast<unsigned int>(xlocalsize),
             static_cast<unsigned int>(ylocalsize),
             static_cast<unsigned int>(zlocalsize)},
            {static_cast<unsigned int>(xgridsize / xlocalsize),
             static_cast<unsigned int>(ygridsize / ylocalsize),
             static_cast<unsigned int>(zgridsize / zlocalsize)},
            static_cast<unsigned int>(channelStride),
            static_cast<unsigned int>(spatialStride),
            static_cast<unsigned int>(batchStride)};
}

} // namespace hip_kernel_provider::batchnorm
