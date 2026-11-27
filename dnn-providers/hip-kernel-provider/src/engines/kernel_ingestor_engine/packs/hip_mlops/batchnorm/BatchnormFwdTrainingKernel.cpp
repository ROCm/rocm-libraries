// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdTrainingKernel.hpp"

#include <algorithm>

#include "BatchnormApplicabilityChecks.hpp"
#include "BatchnormCommon.hpp"
#include "BatchnormKernelCompileOptions.hpp"
#include "core/Utils.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::batchnorm
{

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(&(findTensorAttributes(tensorMap, attributes.x_tensor_uid())))
    , _y(&(findTensorAttributes(tensorMap, attributes.y_tensor_uid())))
    , _scale(&(findTensorAttributes(tensorMap, attributes.scale_tensor_uid())))
    , _bias(&(findTensorAttributes(tensorMap, attributes.bias_tensor_uid())))
    , _activationOut(nullptr)
{
    _epsilon = hipdnn_plugin_sdk::makeScalarOperand(
        tensorMap, attributes.epsilon_tensor_uid(), "Epsilon");

    // Save mean and inv_variance are optional; a pair that is virtual in both tensors is not
    // requested and stays unset.
    if(!hasUnrequestedSavedPair(attributes, tensorMap))
    {
        if(attributes.mean_tensor_uid().has_value())
        {
            _mean = &(findTensorAttributes(tensorMap, attributes.mean_tensor_uid().value()));
        }

        if(attributes.inv_variance_tensor_uid().has_value())
        {
            _invVariance
                = &(findTensorAttributes(tensorMap, attributes.inv_variance_tensor_uid().value()));
        }
    }

    if(attributes.prev_running_mean_tensor_uid().has_value()
       && attributes.prev_running_variance_tensor_uid().has_value()
       && attributes.momentum_tensor_uid().has_value()
       && attributes.next_running_mean_tensor_uid().has_value()
       && attributes.next_running_variance_tensor_uid().has_value())
    {
        _momentum = hipdnn_plugin_sdk::makeScalarOperand(
            tensorMap, attributes.momentum_tensor_uid().value(), "Momentum");

        _prevRunningMean
            = &(findTensorAttributes(tensorMap, attributes.prev_running_mean_tensor_uid().value()));
        _prevRunningVariance = &(
            findTensorAttributes(tensorMap, attributes.prev_running_variance_tensor_uid().value()));
        _nextRunningMean
            = &(findTensorAttributes(tensorMap, attributes.next_running_mean_tensor_uid().value()));
        _nextRunningVariance = &(
            findTensorAttributes(tensorMap, attributes.next_running_variance_tensor_uid().value()));
        _hasRunningStats = true;
    }
}

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : BatchnormFwdTrainingParams(attributes, tensorMap)
{
    // Initialize activation attributes
    _optActivation = parseActivation(pointwiseAttributes);
    _activationOut = tensorMap.at(pointwiseAttributes.out_0_tensor_uid());

    // Validate that activation input matches batchnorm output
    if(pointwiseAttributes.in_0_tensor_uid() != attributes.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "BatchnormFwdTrainingParams: Activation input must match batchnorm output");
    }
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormFwdTrainingParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormFwdTrainingParams::y() const
{
    return _y;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::bias() const
{
    return _bias;
}

double BatchnormFwdTrainingParams::epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                uint32_t numDeviceBuffers) const
{
    return hipdnn_plugin_sdk::toDouble(
        hipdnn_plugin_sdk::resolveScalarOperand(_epsilon, deviceBuffers, numDeviceBuffers));
}

bool BatchnormFwdTrainingParams::hasSaveMeanVariance() const
{
    return (_mean != nullptr) && (_invVariance != nullptr);
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::mean() const
{
    return _mean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::invVariance() const
{
    return _invVariance;
}

bool BatchnormFwdTrainingParams::hasRunningStats() const
{
    return _hasRunningStats;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::prevRunningMean() const
{
    return _prevRunningMean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::prevRunningVariance() const
{
    return _prevRunningVariance;
}

double BatchnormFwdTrainingParams::momentumValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                 uint32_t numDeviceBuffers) const
{
    if(!_momentum.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "momentumValue() called but momentum was not set");
    }
    return hipdnn_plugin_sdk::toDouble(
        hipdnn_plugin_sdk::resolveScalarOperand(*_momentum, deviceBuffers, numDeviceBuffers));
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::nextRunningMean() const
{
    return _nextRunningMean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::nextRunningVariance() const
{
    return _nextRunningVariance;
}

const std::optional<ActivationParams>& BatchnormFwdTrainingParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::activationOut() const
{
    return _activationOut;
}

namespace
{

// Spatial multiple needs space for 2 fp32 elements
// per each x thread (including the last workgroup)
// to stash intermediate mean and variance
constexpr unsigned int FWD_STASH_VALUES = 2;

ProblemDescription fwdTrainingProblem(const BatchnormFwdTrainingParams& params,
                                      int multiProcessorCount)
{
    // Determine data type configuration
    auto xDataType = params.x()->data_type();
    auto yDataType = params.optActivation().has_value() ? params.activationOut()->data_type()
                                                        : params.y()->data_type();
    auto scaleDataType = params.scale()->data_type();

    // NOTE: Although the batchnorm spatial training kernel support the
    // FP16 IO and FP16 scale/bias data types, the hip kernel plugin
    // applicability checks require the scale and bias tensors to be FP32.
    // So we are not using the USE_FP16 path in the kernel for now.
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
    const auto* xDims = params.x()->dims();

    size_t batches = 0;
    size_t channels = 0;
    size_t spatialSize = 0;
    // Check if 3D (NCL/NLC), 4D (NCHW/NHWC) or 5D (NCDHW/NDHWC)
    if(xDims->size() == 3)
    {
        batches = static_cast<size_t>(xDims->Get(0));
        channels = static_cast<size_t>(xDims->Get(1));
        spatialSize = static_cast<size_t>(xDims->Get(2));
    }
    else if(xDims->size() == 4)
    {
        batches = static_cast<size_t>(xDims->Get(0));
        channels = static_cast<size_t>(xDims->Get(1));
        auto height = static_cast<size_t>(xDims->Get(2));
        auto width = static_cast<size_t>(xDims->Get(3));
        // For 4D, combine H*W into spatial dimension
        spatialSize = height * width;
    }
    else if(xDims->size() == 5)
    {
        batches = static_cast<size_t>(xDims->Get(0));
        channels = static_cast<size_t>(xDims->Get(1));
        auto depth = static_cast<size_t>(xDims->Get(2));
        auto height = static_cast<size_t>(xDims->Get(3));
        auto width = static_cast<size_t>(xDims->Get(4));
        // For 5D, combine D*H*W into spatial dimension
        spatialSize = depth * height * width;
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    // Spatial multiple sizes its workgroups to keep at least 60% of the CUs busy
    const size_t minWorkgroups
        = std::max(size_t(1), size_t(0.6f * static_cast<float>(multiProcessorCount)));
    return {batches,
            channels,
            spatialSize,
            isChannelLastLayout(params.x()),
            useFp16Mix,
            useBfp16Mix,
            useFp16Upcast,
            useBfp16Upcast,
            Direction::FORWARD_TRAINING,
            minWorkgroups};
}

} // namespace

BatchnormSpatialConfig defaultBatchnormFwdTrainingConfig(const BatchnormFwdTrainingParams& params,
                                                         int multiProcessorCount)
{
    return defaultSpatialConfig(fwdTrainingProblem(params, multiProcessorCount), FWD_STASH_VALUES);
}

bool isBatchnormFwdTrainingConfigApplicable(const BatchnormFwdTrainingParams& params,
                                            int multiProcessorCount,
                                            const BatchnormSpatialConfig& config)
{
    return isSpatialConfigApplicable(
        fwdTrainingProblem(params, multiProcessorCount), FWD_STASH_VALUES, config);
}

BatchnormKernelBuild buildBatchnormFwdTrainingKernel(const BatchnormFwdTrainingParams& params,
                                                     const std::string& gcnArchName,
                                                     int multiProcessorCount,
                                                     int warpSize,
                                                     const BatchnormSpatialConfig& config)
{
    const auto problem = fwdTrainingProblem(params, multiProcessorCount);
    const bool useFp16Mix = problem.useFp16Mix();
    const bool useBfp16Mix = problem.useBfp16Mix();
    const bool useFp16Upcast = problem.useFp16Upcast();
    const bool useBfp16Upcast = problem.useBfp16Upcast();
    const size_t batches = problem.batches();
    const size_t channels = problem.channels();
    const size_t spatialSize = problem.spatialSize();
    const auto inCstride = problem.inCstride();
    const bool isLayoutChannelLast = problem.isLayoutChannelLast();

    // Kernel launch parameters: the variant, its workgroup size or vector size come from
    // config, the rest derives from them and the problem.
    // NOTE: defaultBatchnormFwdTrainingConfig() picks config with the heuristics that worked
    // well in MIOpen. Tuning infrastructure can be added in the future to further optimize it.
    const auto kernelConfig = spatialKernelConfig(problem, config);
    const int variant = kernelConfig.variant;
    const size_t vectorsize = kernelConfig.vectorsize;
    size_t xlocalsize = kernelConfig.xlocalsize;
    size_t xgridsize = 1;
    const size_t ylocalsize = kernelConfig.ylocalsize;
    size_t ygridsize = 1;
    const size_t zlocalsize = kernelConfig.zlocalsize;
    size_t zgridsize = 1;
    unsigned int ldsgcn = 0;
    unsigned int ldsnogcn = 0;
    int stashMethod = 0;
    const size_t batchElements = kernelConfig.batchElements;

    const auto waveSize = static_cast<size_t>(warpSize);

    size_t xlocalsizeFinal = xlocalsize;
    size_t ylocalsizeFinal = ylocalsize;
    size_t zlocalsizeFinal = zlocalsize;
    if(variant != 2)
    {
        xlocalsize = config.groupSize;
        xgridsize = channels * xlocalsize;
        ldsgcn = static_cast<unsigned int>(xlocalsize / waveSize);
        ldsnogcn = static_cast<unsigned int>(xlocalsize);
    }
    else
    {
        // Compute grid size
        if(isLayoutChannelLast)
        {
            xgridsize = xlocalsize * ((channels / vectorsize + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((inCstride + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((channels + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((inCstride / vectorsize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize * ((batches / batchElements + zlocalsize - 1) / zlocalsize);

        // Get the stash method based on problem size and WG size
        stashMethod = getStashMethod(
            problem, FWD_STASH_VALUES, ylocalsize, zlocalsize, batchElements, vectorsize);

        // WG size for Final kernels (NLC/NHWC/NDHWC)
        if(isLayoutChannelLast && channels % 2 == 0 && xlocalsize % 2 == 0)
        {
            // increase number of blocks (xgridsize does not change for final kernels)
            // 2 is the lower bound because of stashing
            xlocalsizeFinal = 2;
            // increase the number of threads in the y and z direction to decrease the number of
            // loads/stores for each thread
            zlocalsizeFinal = zgridsize / zlocalsize * zlocalsize;
            ylocalsizeFinal
                = (xlocalsize * ylocalsize * zlocalsize) / xlocalsizeFinal / zlocalsizeFinal;
        }
        ldsnogcn = static_cast<unsigned int>(xlocalsize * ylocalsize * zlocalsize);
        ldsgcn = static_cast<unsigned int>(xlocalsize * ylocalsize * zlocalsize / waveSize);
    }

    // Get activation mode
    auto activationMode = ActivationMode::PASTHRU;
    if(params.optActivation().has_value() && params.activationOut() != nullptr)
    {
        activationMode = params.optActivation()->mode;
    }

    // Prepare compilation options
    BatchnormKernelCompileOptions options(
        params.x(),
        params.optActivation().has_value() ? params.activationOut() : params.y(),
        params.mean(),
        params.scale(),
        gcnArchName,
        activationMode);
    options.update("HIP_PLUGIN_USE_FPMIX", useFp16Mix);
    options.update("HIP_PLUGIN_USE_BFPMIX", useBfp16Mix);
    options.update("HIP_PLUGIN_USE_FPUPCAST", useFp16Upcast);
    options.update("HIP_PLUGIN_USE_BFPUPCAST", useBfp16Upcast);
    options.update("HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE", params.hasSaveMeanVariance());
    options.update("HIP_PLUGIN_BN_RUNNING_RESULT", params.hasRunningStats());
    options.update("HIP_PLUGIN_BN_VARIANT", variant);
    options.update("HIP_PLUGIN_BN_LDS_SIZE", ldsnogcn);
    options.update("HIP_PLUGIN_BN_LDSGCN_SIZE", ldsgcn);
    options.update("HIP_PLUGIN_BN_BATCHES", batches);
    options.update("HIP_PLUGIN_BN_CHANNELS", channels);
    options.update("HIP_PLUGIN_BN_SPATIAL_SIZE", spatialSize);
    options.update("HIP_PLUGIN_BN_BATCH_ELEMENTS", batchElements);
    options.update("HIP_PLUGIN_BN_GRP0", xlocalsize);
    options.update("HIP_PLUGIN_BN_GRP1", ylocalsize);
    options.update("HIP_PLUGIN_BN_GRP2", zlocalsize);
    options.update("HIP_PLUGIN_BN_VEC_SIZE", vectorsize);
    options.update("HIP_PLUGIN_BN_STASH_METHOD", stashMethod);
    options.update("HIP_PLUGIN_BN_GRPS0", xgridsize / xlocalsize);
    options.update("HIP_PLUGIN_BN_GRPS1", ygridsize / ylocalsize);
    options.update("HIP_PLUGIN_BN_GRPS2", zgridsize / zlocalsize);
    options.update("HIP_PLUGIN_BN_GRP0_FINAL", xlocalsizeFinal);
    options.update("HIP_PLUGIN_BN_GRP1_FINAL", ylocalsizeFinal);
    options.update("HIP_PLUGIN_BN_GRP2_FINAL", zlocalsizeFinal);

    const std::array<unsigned int, 3> block{static_cast<unsigned int>(xlocalsize),
                                            static_cast<unsigned int>(ylocalsize),
                                            static_cast<unsigned int>(zlocalsize)};
    const std::array<unsigned int, 3> grid{static_cast<unsigned int>(xgridsize / xlocalsize),
                                           static_cast<unsigned int>(ygridsize / ylocalsize),
                                           static_cast<unsigned int>(zgridsize / zlocalsize)};
    std::vector<BatchnormLaunch> launches;
    if(variant != 2)
    {
        launches.push_back({"BatchNormFwdTrainSpatial", block, grid});
    }
    else
    {
        // For variant 2, three kernels
        launches.push_back({"BatchNormFwdTrainSpatialMeanVariance", block, grid});
        launches.push_back({"BatchNormFwdTrainSpatialFinalMeanVariance",
                            {static_cast<unsigned int>(xlocalsizeFinal),
                             static_cast<unsigned int>(ylocalsizeFinal),
                             static_cast<unsigned int>(zlocalsizeFinal)},
                            {static_cast<unsigned int>(xgridsize / xlocalsizeFinal), 1, 1}});
        launches.push_back({"BatchNormFwdTrainSpatialNorm", block, grid});
    }

    HIPDNN_PLUGIN_LOG_INFO("BatchnormFwdTraining: selected variant: "
                           << variant << ", stash method: " << stashMethod
                           << ", and vectorsize: " << vectorsize);

    return {std::move(options), variant, std::move(launches)};
}

} // namespace hip_kernel_provider::batchnorm
