// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdTrainingPlan.hpp"
#include "BatchnormCommon.hpp"
#include "hip/HipKernelCompileOptions.hpp"

#include "HipKernelUtils.hpp"
#include "hip/IKernelCompiler.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::batchnorm
{

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(&(hip_kernel_utils::findTensorAttributes(tensorMap, attributes.x_tensor_uid())))
    , _y(&(hip_kernel_utils::findTensorAttributes(tensorMap, attributes.y_tensor_uid())))
    , _scale(&(hip_kernel_utils::findTensorAttributes(tensorMap, attributes.scale_tensor_uid())))
    , _bias(&(hip_kernel_utils::findTensorAttributes(tensorMap, attributes.bias_tensor_uid())))
    , _activationOut(nullptr)
{
    // Extract epsilon value from pass-by-value tensor (cast to double for kernel compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
        epsilonTensorAttr, "Epsilon");

    // Save mean and inv_variance are optional
    if(attributes.mean_tensor_uid().has_value())
    {
        _mean = &(hip_kernel_utils::findTensorAttributes(tensorMap,
                                                         attributes.mean_tensor_uid().value()));
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _invVariance = &(hip_kernel_utils::findTensorAttributes(
            tensorMap, attributes.inv_variance_tensor_uid().value()));
    }

    if(attributes.prev_running_mean_tensor_uid().has_value()
       && attributes.prev_running_variance_tensor_uid().has_value()
       && attributes.momentum_tensor_uid().has_value()
       && attributes.next_running_mean_tensor_uid().has_value()
       && attributes.next_running_variance_tensor_uid().has_value())
    {
        // Extract momentum value from pass-by-value tensor (cast to double for kernel compatibility)
        auto momentumTensorAttr = tensorMap.at(attributes.momentum_tensor_uid().value());
        _momentumValue = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
            momentumTensorAttr, "Momentum");

        _prevRunningMean = &(hip_kernel_utils::findTensorAttributes(
            tensorMap, attributes.prev_running_mean_tensor_uid().value()));
        _prevRunningVariance = &(hip_kernel_utils::findTensorAttributes(
            tensorMap, attributes.prev_running_variance_tensor_uid().value()));
        _nextRunningMean = &(hip_kernel_utils::findTensorAttributes(
            tensorMap, attributes.next_running_mean_tensor_uid().value()));
        _nextRunningVariance = &(hip_kernel_utils::findTensorAttributes(
            tensorMap, attributes.next_running_variance_tensor_uid().value()));
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
    _optActivation = hip_kernel_utils::parseActivation(pointwiseAttributes);
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

double BatchnormFwdTrainingParams::epsilonValue() const
{
    return _epsilonValue;
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

double BatchnormFwdTrainingParams::momentumValue() const
{
    return _momentumValue.value();
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

const std::optional<hip_kernel_utils::ActivationParams>&
    BatchnormFwdTrainingParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdTrainingPlan::BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams)
    : _trainingParams(std::move(trainingParams))
{
}

size_t
    BatchnormFwdTrainingPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for batchnorm training
    return 0;
}

void BatchnormFwdTrainingPlan::compile(const IKernelCompiler& kernelCompiler,
                                       const hipDeviceProp_t& deviceProperties)
{
    // Determine data type configuration
    auto xDataType = _trainingParams.x()->data_type();
    auto scaleDataType = _trainingParams.scale()->data_type();

    // NOTE: Although the batchnorm spatial training kernel support the
    // FP16 IO and FP16 scale/bias data types, the hip kernel plugin
    // applicability checks require the scale and bias tensors to be FP32.
    // So we are not using the USE_FP16 path in the kernel for now.
    bool useFp16Mix = (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF
                       && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    bool useBfp16Mix = (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16
                        && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    bool useFp32 = !useFp16Mix && !useBfp16Mix;

    // Extract dimensions from x tensor
    const auto* xDims = _trainingParams.x()->dims();

    size_t n = 0;
    size_t c = 0;
    size_t h = 0;
    size_t w = 0;
    // Check if 4D (NCHW/NHWC) or 5D (NCDHW/NDHWC)
    if(xDims->size() == 4)
    {
        n = static_cast<size_t>(xDims->Get(0));
        c = static_cast<size_t>(xDims->Get(1));
        h = static_cast<size_t>(xDims->Get(2));
        w = static_cast<size_t>(xDims->Get(3));
    }
    else if(xDims->size() == 5)
    {
        n = static_cast<size_t>(xDims->Get(0));
        c = static_cast<size_t>(xDims->Get(1));
        auto d = static_cast<size_t>(xDims->Get(2));
        h = static_cast<size_t>(xDims->Get(3));
        w = static_cast<size_t>(xDims->Get(4));
        // For 5D, combine D*H*W into spatial dimension
        h = d * h;
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    auto inCstride = static_cast<unsigned int>(h * w);
    auto inNhw = static_cast<unsigned int>(n * h * w);
    auto invInNhw = static_cast<float>(1.0 / inNhw);

    // Detect layout
    bool isLayoutNHWC = hip_kernel_utils::isChannelLastLayout(_trainingParams.x());

    // Kernel launch parameters
    // NOTE: These are generally selected based on heuristics and tuning,
    // but here we are starting with initial values that worked well in
    // MIOpen. Tuning infrastructure can be added in the future to further
    // optimize these parameters.
    int variant = -1;
    size_t vectorsize = 1;
    size_t xlocalsize = 1;
    size_t xgridsize = 1;
    size_t ylocalsize = 1;
    size_t ygridsize = 1;
    size_t zlocalsize = 1;
    size_t zgridsize = 1;
    unsigned int ldsgcn = 0;
    unsigned int ldsnogcn = 0;
    int stashMethod = 0;
    size_t nelements = 1;

    // Spatial multiple needs space for 2 fp32 elements
    // per each x thread (including the last workgroup)
    // to stash intermediate mean and variance
    const unsigned int stashValuesFwd = 2;

    // Get the kernel launch configuration based on heuristics
    hip_kernel_provider::batchnorm::KernelConfig config;
    // Define default configuration based on heuristics and
    // add all other valid configurations for the given problem
    if(hip_kernel_provider::batchnorm::useMultiple(
           n,
           h,
           w,
           useFp16Mix || useBfp16Mix,
           isLayoutNHWC,
           hip_kernel_provider::batchnorm::Direction::FORWARD_TRAINING))
    {
        // Determine the minimum number of workgroups
        const size_t minWorkgroups = std::max(
            size_t(1), size_t(0.6f * static_cast<float>(deviceProperties.multiProcessorCount)));
        hip_kernel_provider::batchnorm::defaultConfigSpatialMultiple(
            n, c, h, w, isLayoutNHWC, useFp32, minWorkgroups, stashValuesFwd, config);
        if(config.variant == -1)
        {
            // If the default spatial multiple function failed to select a valid configuration,
            // get a default spatial single configuration as fallback
            hip_kernel_provider::batchnorm::defaultConfigSpatialSingle(
                n,
                h,
                w,
                useFp16Mix,
                useBfp16Mix,
                isLayoutNHWC,
                hip_kernel_provider::batchnorm::Direction::FORWARD_TRAINING,
                config);
        }
    }
    else
    {
        hip_kernel_provider::batchnorm::defaultConfigSpatialSingle(
            n,
            h,
            w,
            useFp16Mix,
            useBfp16Mix,
            isLayoutNHWC,
            hip_kernel_provider::batchnorm::Direction::FORWARD_TRAINING,
            config);
    }

    variant = config.variant;
    vectorsize = config.vectorsize;
    xlocalsize = config.xlocalsize;
    ylocalsize = config.ylocalsize;
    zlocalsize = config.zlocalsize;
    nelements = config.nelements;

    size_t xlocalsizeFinal = xlocalsize;
    size_t ylocalsizeFinal = ylocalsize;
    size_t zlocalsizeFinal = zlocalsize;
    if(variant != 2)
    {
        xlocalsize = 1024;
        if(((inCstride < 256) && (n < 256)) || ((inCstride < 100) && (n <= 256)))
        {
            xlocalsize = 256;
        }
        xgridsize = c * xlocalsize;
        ldsgcn = static_cast<unsigned int>(xlocalsize / 64);
        ldsnogcn = static_cast<unsigned int>(xlocalsize);
    }
    else
    {
        // Compute grid size
        if(isLayoutNHWC)
        {
            xgridsize = xlocalsize * ((c / vectorsize + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((inCstride + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((c + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((inCstride / vectorsize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize * ((n / nelements + zlocalsize - 1) / zlocalsize);

        // Get the stash method based on problem size and WG size
        stashMethod = hip_kernel_provider::batchnorm::getStashMethod(isLayoutNHWC,
                                                                     useFp32,
                                                                     stashValuesFwd,
                                                                     c,
                                                                     n,
                                                                     inCstride,
                                                                     ylocalsize,
                                                                     zlocalsize,
                                                                     nelements);

        // WG size for Final kernels (NHWC)
        if(isLayoutNHWC && c % 2 == 0 && xlocalsize % 2 == 0)
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
        ldsgcn = static_cast<unsigned int>(xlocalsize * ylocalsize * zlocalsize / 64);
    }

    // Detect GPU architecture
    std::string archName(deviceProperties.gcnArchName);
    bool isGfx103X = (archName.find("gfx103") == 0);
    bool isGfx110X = (archName.find("gfx110") == 0);
    bool isGfx120X = (archName.find("gfx120") == 0);
    bool isGfx115X = (archName.find("gfx115") == 0);

    // Get activation mode
    auto activationMode = hip_kernel_utils::ActivationMode::PASTHRU;
    if(_trainingParams.optActivation().has_value() && _trainingParams.activationOut() != nullptr)
    {
        activationMode = (*_trainingParams.optActivation()).mode;
    }

    // Prepare compilation options
    HipKernelCompileOptions options(_trainingParams.x(), deviceProperties, activationMode);
    options.add("HIP_PLUGIN_USE_FPMIX", useFp16Mix);
    options.add("HIP_PLUGIN_USE_BFPMIX", useBfp16Mix);
    // Not using FP16 and BFP16 paths due to affine data type requirements
    options.update("HIP_PLUGIN_USE_FP16", 0);
    options.update("HIP_PLUGIN_USE_BFP16", 0);
    options.add("HIP_PLUGIN_SAVE_MEAN_VARIANCE", _trainingParams.hasSaveMeanVariance());
    options.add("HIP_PLUGIN_RUNNING_RESULT", _trainingParams.hasRunningStats());
    options.add("HIP_PLUGIN_BN_VARIANT", variant);
    options.add("HIP_PLUGIN_BN_LDS_SIZE", ldsnogcn);
    options.add("HIP_PLUGIN_BN_LDSGCN_SIZE", ldsgcn);
    options.add("HIP_PLUGIN_BN_N", n);
    options.add("HIP_PLUGIN_BN_C", c);
    options.add("HIP_PLUGIN_BN_HW", inCstride);
    options.add("HIP_PLUGIN_BN_NHW", inNhw);
    options.add("HIP_PLUGIN_BN_CHW", c * inCstride);
    options.add("HIP_PLUGIN_BN_NCHW", c * inNhw);
    options.add("HIP_PLUGIN_BN_NGRPS", ygridsize / ylocalsize);
    options.add("HIP_PLUGIN_BN_NGRPS2", zgridsize / zlocalsize);
    options.add("HIP_PLUGIN_BN_N_ELEMENTS", nelements);
    options.add("HIP_PLUGIN_BN_GRP0", xlocalsize);
    options.add("HIP_PLUGIN_BN_GRP1", ylocalsize);
    options.add("HIP_PLUGIN_BN_GRP2", zlocalsize);
    options.add("HIP_PLUGIN_BN_GRP0_FINAL", xlocalsizeFinal);
    options.add("HIP_PLUGIN_BN_GRP1_FINAL", ylocalsizeFinal);
    options.add("HIP_PLUGIN_BN_GRP2_FINAL", zlocalsizeFinal);
    options.add("HIP_PLUGIN_BN_GFX103X", isGfx103X);
    options.add("HIP_PLUGIN_BN_GFX110X", isGfx110X);
    options.add("HIP_PLUGIN_BN_GFX120X", isGfx120X);
    options.add("HIP_PLUGIN_BN_GFX115X", isGfx115X);
    options.add("HIP_PLUGIN_BN_VECTORIZE", vectorsize > 1);
    options.add("HIP_PLUGIN_BN_VEC_SIZE", vectorsize);
    options.add("HIP_PLUGIN_BN_STASH_METHOD", stashMethod);

    // Compile the kernel and configure launch parameters based on the selected variant
    _compiledProgram = kernelCompiler.compile("BatchNormFwdTrainSpatial.cpp", options);
    if(variant != 2)
    {
        _runnableKernels.push_back(_compiledProgram->getKernel("BatchNormFwdTrainSpatial"));
        _runnableKernels[0]->setBlockSize(static_cast<unsigned int>(xlocalsize),
                                          static_cast<unsigned int>(ylocalsize),
                                          static_cast<unsigned int>(zlocalsize));
        _runnableKernels[0]->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                                         static_cast<unsigned int>(ygridsize / ylocalsize),
                                         static_cast<unsigned int>(zgridsize / zlocalsize));
    }
    else
    {
        // For variant 2, we need to configure three kernels
        _runnableKernels.push_back(
            _compiledProgram->getKernel("BatchNormFwdTrainSpatialMeanVariance"));
        _runnableKernels.push_back(
            _compiledProgram->getKernel("BatchNormFwdTrainSpatialFinalMeanVariance"));
        _runnableKernels.push_back(_compiledProgram->getKernel("BatchNormFwdTrainSpatialNorm"));

        _runnableKernels[0]->setBlockSize(static_cast<unsigned int>(xlocalsize),
                                          static_cast<unsigned int>(ylocalsize),
                                          static_cast<unsigned int>(zlocalsize));
        _runnableKernels[0]->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                                         static_cast<unsigned int>(ygridsize / ylocalsize),
                                         static_cast<unsigned int>(zgridsize / zlocalsize));
        _runnableKernels[1]->setBlockSize(static_cast<unsigned int>(xlocalsizeFinal),
                                          static_cast<unsigned int>(ylocalsizeFinal),
                                          static_cast<unsigned int>(zlocalsizeFinal));
        _runnableKernels[1]->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsizeFinal),
                                         static_cast<unsigned int>(1),
                                         static_cast<unsigned int>(1));
        _runnableKernels[2]->setBlockSize(static_cast<unsigned int>(xlocalsize),
                                          static_cast<unsigned int>(ylocalsize),
                                          static_cast<unsigned int>(zlocalsize));
        _runnableKernels[2]->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                                         static_cast<unsigned int>(ygridsize / ylocalsize),
                                         static_cast<unsigned int>(zgridsize / zlocalsize));
    }

    HIPDNN_PLUGIN_LOG_INFO("BatchnormFwdTrainingPlan: Compiled kernel with variant: "
                           << variant << ", stash method: " << stashMethod
                           << ", and vectorsize: " << vectorsize);

    // Store kernel launch parameters
    _kernelVariant = variant;
    _invInNhw = invInNhw;
}

void BatchnormFwdTrainingPlan::execute(const HipKernelHandle& handle,
                                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                       uint32_t numDeviceBuffers,
                                       [[maybe_unused]] void* workspace) const
{
    if(_runnableKernels.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "BatchnormFwdTrainingPlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto xBuffer = hip_kernel_utils::findDeviceBuffer(
        _trainingParams.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _trainingParams.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = hip_kernel_utils::findDeviceBuffer(
        _trainingParams.bias()->uid(), deviceBuffers, numDeviceBuffers);

    // Handle save mean/variance if provided (optional)
    void* resultSaveMeanPtr = nullptr;
    void* resultSaveInvVariancePtr = nullptr;

    if(_trainingParams.hasSaveMeanVariance())
    {
        resultSaveMeanPtr = hip_kernel_utils::findDeviceBuffer(
                                _trainingParams.mean()->uid(), deviceBuffers, numDeviceBuffers)
                                .ptr;
        resultSaveInvVariancePtr
            = hip_kernel_utils::findDeviceBuffer(
                  _trainingParams.invVariance()->uid(), deviceBuffers, numDeviceBuffers)
                  .ptr;
    }

    // Handle running stats if provided (optional)
    void* prevRunningMeanPtr = nullptr;
    void* prevRunningVariancePtr = nullptr;
    void* nextRunningMeanPtr = nullptr;
    void* nextRunningVariancePtr = nullptr;

    if(_trainingParams.hasRunningStats())
    {
        prevRunningMeanPtr
            = hip_kernel_utils::findDeviceBuffer(
                  _trainingParams.prevRunningMean()->uid(), deviceBuffers, numDeviceBuffers)
                  .ptr;
        prevRunningVariancePtr
            = hip_kernel_utils::findDeviceBuffer(
                  _trainingParams.prevRunningVariance()->uid(), deviceBuffers, numDeviceBuffers)
                  .ptr;
        nextRunningMeanPtr
            = hip_kernel_utils::findDeviceBuffer(
                  _trainingParams.nextRunningMean()->uid(), deviceBuffers, numDeviceBuffers)
                  .ptr;
        nextRunningVariancePtr
            = hip_kernel_utils::findDeviceBuffer(
                  _trainingParams.nextRunningVariance()->uid(), deviceBuffers, numDeviceBuffers)
                  .ptr;
    }

    // Get epsilon value from training parameters
    // Note: Type validation already done in constructor
    double epsilon = _trainingParams.epsilonValue();

    // Extract momentum from pass-by-value tensor attribute if running stats exist
    double expAvgFactor = 0.0;
    if(_trainingParams.hasRunningStats())
    {
        expAvgFactor = _trainingParams.momentumValue();
        HIPDNN_PLUGIN_LOG_INFO(
            "BatchnormFwdTrainingPlan: expAvgFactor (momentum) = " << expAvgFactor);
    }

    // Get output buffer and activation parameters
    float activationAlpha = 0.0f;
    float activationBeta = 0.0f;
    hipdnnPluginDeviceBuffer_t yBuffer = {-1, nullptr};
    if(_trainingParams.optActivation().has_value() && _trainingParams.activationOut() != nullptr)
    {
        yBuffer = hip_kernel_utils::findDeviceBuffer(
            _trainingParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

        const auto& activation = *_trainingParams.optActivation();
        activationAlpha = static_cast<float>(activation.alpha);
        activationBeta = static_cast<float>(activation.beta);
    }
    else
    {
        yBuffer = hip_kernel_utils::findDeviceBuffer(
            _trainingParams.y()->uid(), deviceBuffers, numDeviceBuffers);
    }

    if(_kernelVariant != 2)
    {
        // Launch the kernel
        if(_trainingParams.hasSaveMeanVariance() && _trainingParams.hasRunningStats())
        {
            _runnableKernels[0]->launch(handle.getStream(),
                                        xBuffer.ptr,
                                        yBuffer.ptr,
                                        scaleBuffer.ptr,
                                        biasBuffer.ptr,
                                        _invInNhw,
                                        expAvgFactor,
                                        prevRunningMeanPtr,
                                        prevRunningVariancePtr,
                                        nextRunningMeanPtr,
                                        nextRunningVariancePtr,
                                        epsilon,
                                        resultSaveMeanPtr,
                                        resultSaveInvVariancePtr,
                                        activationAlpha,
                                        activationBeta);
        }
        else if(_trainingParams.hasSaveMeanVariance())
        {
            _runnableKernels[0]->launch(handle.getStream(),
                                        xBuffer.ptr,
                                        yBuffer.ptr,
                                        scaleBuffer.ptr,
                                        biasBuffer.ptr,
                                        _invInNhw,
                                        epsilon,
                                        resultSaveMeanPtr,
                                        resultSaveInvVariancePtr,
                                        activationAlpha,
                                        activationBeta);
        }
        else if(_trainingParams.hasRunningStats())
        {
            _runnableKernels[0]->launch(handle.getStream(),
                                        xBuffer.ptr,
                                        yBuffer.ptr,
                                        scaleBuffer.ptr,
                                        biasBuffer.ptr,
                                        _invInNhw,
                                        expAvgFactor,
                                        prevRunningMeanPtr,
                                        prevRunningVariancePtr,
                                        nextRunningMeanPtr,
                                        nextRunningVariancePtr,
                                        epsilon,
                                        activationAlpha,
                                        activationBeta);
        }
        else
        {
            _runnableKernels[0]->launch(handle.getStream(),
                                        xBuffer.ptr,
                                        yBuffer.ptr,
                                        scaleBuffer.ptr,
                                        biasBuffer.ptr,
                                        _invInNhw,
                                        epsilon,
                                        activationAlpha,
                                        activationBeta);
        }
    }
    else
    {
        // Launch the kernels
        // 1. BatchNormFwdTrainSpatialMeanVariance kernel
        _runnableKernels[0]->launch(handle.getStream(), xBuffer.ptr, yBuffer.ptr);
        // 2. BatchNormFwdTrainSpatialFinalMeanVariance kernel
        if(_trainingParams.hasSaveMeanVariance() && _trainingParams.hasRunningStats())
        {
            _runnableKernels[1]->launch(handle.getStream(),
                                        yBuffer.ptr,
                                        _invInNhw,
                                        expAvgFactor,
                                        prevRunningMeanPtr,
                                        prevRunningVariancePtr,
                                        nextRunningMeanPtr,
                                        nextRunningVariancePtr,
                                        epsilon,
                                        resultSaveMeanPtr,
                                        resultSaveInvVariancePtr);
        }
        else if(_trainingParams.hasSaveMeanVariance())
        {
            _runnableKernels[1]->launch(handle.getStream(),
                                        yBuffer.ptr,
                                        _invInNhw,
                                        epsilon,
                                        resultSaveMeanPtr,
                                        resultSaveInvVariancePtr);
        }
        else if(_trainingParams.hasRunningStats())
        {
            _runnableKernels[1]->launch(handle.getStream(),
                                        yBuffer.ptr,
                                        _invInNhw,
                                        expAvgFactor,
                                        prevRunningMeanPtr,
                                        prevRunningVariancePtr,
                                        nextRunningMeanPtr,
                                        nextRunningVariancePtr,
                                        epsilon);
        }
        else
        {
            _runnableKernels[1]->launch(handle.getStream(), yBuffer.ptr, _invInNhw, epsilon);
        }
        // 3. BatchNormFwdTrainSpatialNorm kernel
        _runnableKernels[2]->launch(handle.getStream(),
                                    xBuffer.ptr,
                                    yBuffer.ptr,
                                    scaleBuffer.ptr,
                                    biasBuffer.ptr,
                                    activationAlpha,
                                    activationBeta);
    }
}

} // namespace hip_kernel_provider::batchnorm
