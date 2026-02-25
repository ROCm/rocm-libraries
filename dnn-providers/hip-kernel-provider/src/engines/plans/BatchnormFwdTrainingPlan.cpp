// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormFwdTrainingPlan.hpp"
#include "BatchnormCommon.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "hip/HipKernel.hpp"
#include "hip/HipProgram.hpp"
#include "hip/HipUtils.hpp"

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <sstream>
#include <stdexcept>

namespace hip_kernel_plugin
{

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_data_sdk::data_objects::BatchnormAttributes& attributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(tensorMap.at(attributes.bias_tensor_uid()))
    , _activationOut(nullptr)
{
    // Extract epsilon value from pass-by-value tensor (cast to double for kernel compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue
        = hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");

    // Save mean and inv_variance are optional
    if(attributes.mean_tensor_uid().has_value())
    {
        _mean = tensorMap.at(attributes.mean_tensor_uid().value());
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _invVariance = tensorMap.at(attributes.inv_variance_tensor_uid().value());
    }

    if(attributes.prev_running_mean_tensor_uid().has_value()
       && attributes.prev_running_variance_tensor_uid().has_value()
       && attributes.momentum_tensor_uid().has_value()
       && attributes.next_running_mean_tensor_uid().has_value()
       && attributes.next_running_variance_tensor_uid().has_value())
    {
        // Extract momentum value from pass-by-value tensor (cast to double for kernel compatibility)
        auto momentumTensorAttr = tensorMap.at(attributes.momentum_tensor_uid().value());
        _momentumValue = hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(
            momentumTensorAttr, "Momentum");

        _prevRunningMean = tensorMap.at(attributes.prev_running_mean_tensor_uid().value());
        _prevRunningVariance = tensorMap.at(attributes.prev_running_variance_tensor_uid().value());
        _nextRunningMean = tensorMap.at(attributes.next_running_mean_tensor_uid().value());
        _nextRunningVariance = tensorMap.at(attributes.next_running_variance_tensor_uid().value());
        _hasRunningStats = true;
    }
}

BatchnormFwdTrainingParams::BatchnormFwdTrainingParams(
    const hipdnn_data_sdk::data_objects::BatchnormAttributes& attributes,
    const hipdnn_data_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(tensorMap.at(attributes.bias_tensor_uid()))
    , _optActivation(hip_kernel_utils::parseActivation(pointwiseAttributes))
    , _activationOut(tensorMap.at(pointwiseAttributes.out_0_tensor_uid()))
{
    // Extract epsilon value from pass-by-value tensor (cast to double for kernel compatibility)
    auto epsilonTensorAttr = tensorMap.at(attributes.epsilon_tensor_uid());
    _epsilonValue
        = hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(epsilonTensorAttr, "Epsilon");

    // Validate that activation input matches batchnorm output
    if(pointwiseAttributes.in_0_tensor_uid() != attributes.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "BatchnormFwdTrainingParams: Activation input must match batchnorm output");
    }

    // Save mean and inv_variance are optional
    if(attributes.mean_tensor_uid().has_value())
    {
        _mean = tensorMap.at(attributes.mean_tensor_uid().value());
    }

    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _invVariance = tensorMap.at(attributes.inv_variance_tensor_uid().value());
    }

    if(attributes.prev_running_mean_tensor_uid().has_value()
       && attributes.prev_running_variance_tensor_uid().has_value()
       && attributes.momentum_tensor_uid().has_value()
       && attributes.next_running_mean_tensor_uid().has_value()
       && attributes.next_running_variance_tensor_uid().has_value())
    {
        // Extract momentum value from pass-by-value tensor (cast to double for kernel compatibility)
        auto momentumTensorAttr = tensorMap.at(attributes.momentum_tensor_uid().value());
        _momentumValue = hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(
            momentumTensorAttr, "Momentum");

        _prevRunningMean = tensorMap.at(attributes.prev_running_mean_tensor_uid().value());
        _prevRunningVariance = tensorMap.at(attributes.prev_running_variance_tensor_uid().value());
        _nextRunningMean = tensorMap.at(attributes.next_running_mean_tensor_uid().value());
        _nextRunningVariance = tensorMap.at(attributes.next_running_variance_tensor_uid().value());
        _hasRunningStats = true;
    }
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdTrainingParams::x() const
{
    return _x;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdTrainingParams::y() const
{
    return _y;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdTrainingParams::scale() const
{
    return _scale;
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdTrainingParams::bias() const
{
    return _bias;
}

double BatchnormFwdTrainingParams::epsilonValue() const
{
    return _epsilonValue;
}

bool BatchnormFwdTrainingParams::hasSaveMeanVariance() const
{
    return _mean.has_value() && _invVariance.has_value();
}

const hipdnn_data_sdk::data_objects::TensorAttributes* BatchnormFwdTrainingParams::mean() const
{
    return _mean.value();
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::invVariance() const
{
    return _invVariance.value();
}

bool BatchnormFwdTrainingParams::hasRunningStats() const
{
    return _hasRunningStats;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::prevRunningMean() const
{
    return _prevRunningMean.value();
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::prevRunningVariance() const
{
    return _prevRunningVariance.value();
}

double BatchnormFwdTrainingParams::momentumValue() const
{
    return _momentumValue.value();
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::nextRunningMean() const
{
    return _nextRunningMean.value();
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::nextRunningVariance() const
{
    return _nextRunningVariance.value();
}

const std::optional<hip_kernel_utils::ActivationParams>&
    BatchnormFwdTrainingParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_data_sdk::data_objects::TensorAttributes*
    BatchnormFwdTrainingParams::activationOut() const
{
    return _activationOut;
}

BatchnormFwdTrainingPlan::BatchnormFwdTrainingPlan(BatchnormFwdTrainingParams&& trainingParams,
                                                   bool benchmarkingEnabled)
    : _trainingParams(std::move(trainingParams))
    , _benchmarkingEnabled(benchmarkingEnabled)
{
}

size_t BatchnormFwdTrainingPlan::getWorkspaceSize(
    [[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    // No workspace needed for batchnorm training
    return 0;
}

void BatchnormFwdTrainingPlan::execute(const HipdnnEnginePluginHandle& handle,
                                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                       uint32_t numDeviceBuffers,
                                       [[maybe_unused]] void* workspace) const
{
    // Extract epsilon from pass-by-value tensor attribute (type-safe, no buffer lookup needed)
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

    // Get device and properties
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    // Determine data type configuration
    auto xDataType = _trainingParams.x()->data_type();
    auto scaleDataType = _trainingParams.scale()->data_type();

    // NOTE: Although the batchnorm spatial training kernel support the
    // FP16 IO and FP16 scale/bias data types, the hip kernel plugin
    // applicability checks require the scale and bias tensors to be FP32.
    // So we are not using the USE_FP16 path in the kernel for now.
    bool useFp16Mix = (xDataType == hipdnn_data_sdk::data_objects::DataType::HALF
                       && scaleDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
    bool useBfp16Mix = (xDataType == hipdnn_data_sdk::data_objects::DataType::BFLOAT16
                        && scaleDataType == hipdnn_data_sdk::data_objects::DataType::FLOAT);
    bool useFp32 = !useFp16Mix && !useBfp16Mix;

    // Extract dimensions from x tensor
    const auto* xDims = _trainingParams.x()->dims();
    const auto* xStrides = _trainingParams.x()->strides();

    size_t n, c, h, w;
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
        size_t d = static_cast<size_t>(xDims->Get(2));
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

    unsigned int in_cstride = static_cast<unsigned int>(h * w);
    unsigned int in_nhw = static_cast<unsigned int>(n * h * w);
    float inhw = static_cast<float>(1.0 / in_nhw);

    // Detect layout: NHWC has C dimension (index 1) with stride 1, NCHW has stride H*W
    bool isLayoutNHWC = (xStrides->Get(1) == 1);

    // Kernel launch parameters
    // NOTE: These are generally selected based on heuristics and tuning,
    // but here we are starting with initial values that worked well in
    // MIOpen. Tuning infrastructure can be added in the future to further
    // optimize these parameters.
    int variant = -1;
    size_t vectorsize = 1;
    size_t xlocalsize = 1, xgridsize = 1;
    size_t ylocalsize = 1, ygridsize = 1;
    size_t zlocalsize = 1, zgridsize = 1;
    unsigned int ldsgcn = 0, ldsnogcn = 0;
    int stash_method = 0;
    size_t nelements = 1;

    // Spatial multiple needs space for 2 fp32 elements
    // per each x thread (including the last workgroup)
    // to stash intermediate mean and variance
    const unsigned int stash_values_fwd = 2;

    // Get the kernel launch configuration based on heuristics
    hip_kernel_plugin::batchnorm::KernelConfig config;
    // Define default configuration based on heuristics and
    // add all other valid configurations for the given problem
    hip_kernel_plugin::batchnorm::
        DefaultConfigSpatialSingle( // Will only select variants 0, 1, or 3!
            n,
            h,
            w,
            useFp16Mix,
            useBfp16Mix,
            isLayoutNHWC,
            hip_kernel_plugin::batchnorm::Direction::FORWARD_TRAINING,
            config);

    variant = config.variant;
    vectorsize = config.vectorsize;
    // Activate these only for variant 2!
    // xlocalsize = config.xlocalsize;
    // ylocalsize = config.ylocalsize;
    // zlocalsize = config.zlocalsize;
    // nelements = config.nelements;

    size_t xlocalsize_final = xlocalsize, ylocalsize_final = ylocalsize,
           zlocalsize_final = zlocalsize;
    if(variant != 2)
    {
        xlocalsize = 1024;
        if(((in_cstride < 256) && (n < 256)) || ((in_cstride < 100) && (n <= 256)))
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
            ygridsize = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((c + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((in_cstride / vectorsize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize * ((n / nelements + zlocalsize - 1) / zlocalsize);

        // Get the stash method based on problem size and WG size
        stash_method = hip_kernel_plugin::batchnorm::GetStashMethod(isLayoutNHWC,
                                                                    useFp32,
                                                                    stash_values_fwd,
                                                                    c,
                                                                    n,
                                                                    in_cstride,
                                                                    ylocalsize,
                                                                    zlocalsize,
                                                                    nelements);

        // WG size for Final kernels (NHWC)
        if(isLayoutNHWC && c % 2 == 0 && xlocalsize % 2 == 0)
        {
            // increase number of blocks (xgridsize does not change for final kernels)
            // 2 is the lower bound because of stashing
            xlocalsize_final = 2;
            // increase the number of threads in the y and z direction to decrease the number of
            // loads/stores for each thread
            zlocalsize_final = zgridsize / zlocalsize * zlocalsize;
            ylocalsize_final
                = (xlocalsize * ylocalsize * zlocalsize) / xlocalsize_final / zlocalsize_final;
        }
        ldsnogcn = static_cast<unsigned int>(xlocalsize * ylocalsize * zlocalsize);
        ldsgcn = static_cast<unsigned int>(xlocalsize * ylocalsize * zlocalsize / 64);
    }

    // Detect GPU architecture
    std::string archName(props.gcnArchName);
    bool isGfx103X = (archName.find("gfx103") == 0);
    bool isGfx110X = (archName.find("gfx110") == 0);
    bool isGfx120X = (archName.find("gfx120") == 0);
    bool isGfx115X = (archName.find("gfx115") == 0);

    // Get activation parameters
    int nrnOpId = 0;
    float alpha = 0.0f;
    float beta = 0.0f;

    if(_trainingParams.optActivation().has_value() && _trainingParams.activationOut() != nullptr)
    {
        const auto& activation = *(_trainingParams.optActivation());
        nrnOpId = static_cast<int>(activation.mode);
        alpha = static_cast<float>(activation.alpha);
        beta = static_cast<float>(activation.beta);
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

    // Prepare compilation options
    std::vector<std::string> options;
    options.emplace_back("-I/opt/rocm/include");
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FP32=") + (useFp32 ? "1" : "0"));
    options.emplace_back(std::string(
        "-DHIP_PLUGIN_USE_FP16=0")); // Not using this path due to scale/bias data type requirements
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_FPMIX=") + (useFp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_USE_BFPMIX=") + (useBfp16Mix ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_SAVE_MEAN_VARIANCE=")
                         + (_trainingParams.hasSaveMeanVariance() ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_RUNNING_RESULT=")
                         + (_trainingParams.hasRunningStats() ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_VARIANT=") + std::to_string(variant));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_LDS_SIZE=") + std::to_string(ldsnogcn));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_LDSGCN_SIZE=") + std::to_string(ldsgcn));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_N=") + std::to_string(n));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_C=") + std::to_string(c));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_HW=") + std::to_string(in_cstride));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_NHW=") + std::to_string(in_nhw));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_CHW=") + std::to_string(c * in_cstride));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_NCHW=") + std::to_string(c * in_nhw));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_NGRPS=")
                         + std::to_string(ygridsize / ylocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_NGRPS2=")
                         + std::to_string(zgridsize / zlocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_N_ELEMENTS=") + std::to_string(nelements));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP0=") + std::to_string(xlocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP1=") + std::to_string(ylocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP2=") + std::to_string(zlocalsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP0_FINAL=")
                         + std::to_string(xlocalsize_final));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP1_FINAL=")
                         + std::to_string(ylocalsize_final));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GRP2_FINAL=")
                         + std::to_string(zlocalsize_final));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX103X=") + (isGfx103X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX110X=") + (isGfx110X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX120X=") + (isGfx120X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_GFX115X=") + (isGfx115X ? "1" : "0"));
    options.emplace_back(std::string("-DHIP_PLUGIN_LAYOUT_NHWC=")
                         + std::to_string(static_cast<int>(isLayoutNHWC)));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_VECTORIZE=")
                         + std::to_string(static_cast<int>(vectorsize > 1)));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_VEC_SIZE=") + std::to_string(vectorsize));
    options.emplace_back(std::string("-DHIP_PLUGIN_BN_STASH_METHOD=")
                         + std::to_string(stash_method));
    options.emplace_back(std::string("-DHIP_PLUGIN_NRN_OP_ID=") + std::to_string(nrnOpId));
    options.emplace_back(std::string("--offload-arch=") + props.gcnArchName);

    // Create and configure kernel - Should be implemented differently for variant 2!
    auto hipProgram = HipProgram("BatchNormFwdTrainSpatial.cpp", options);
    auto hipKernel = HipKernel(hipProgram, "BatchNormFwdTrainSpatial");

    hipKernel.SetBlockSize(static_cast<unsigned int>(xlocalsize),
                           static_cast<unsigned int>(ylocalsize),
                           static_cast<unsigned int>(zlocalsize));
    hipKernel.SetGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                          static_cast<unsigned int>(ygridsize / ylocalsize),
                          static_cast<unsigned int>(zgridsize / zlocalsize));

    // Launch kernel with appropriate output buffer
    if(_trainingParams.optActivation().has_value() && _trainingParams.activationOut() != nullptr)
    {
        auto activationOutBuffer = hip_kernel_utils::findDeviceBuffer(
            _trainingParams.activationOut()->uid(), deviceBuffers, numDeviceBuffers);

        if(variant != 2)
        {
            if(_trainingParams.hasSaveMeanVariance() && _trainingParams.hasRunningStats())
            {
                hipKernel.Launch(handle.getStream(),
                                 xBuffer.ptr,
                                 activationOutBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasBuffer.ptr,
                                 inhw,
                                 expAvgFactor,
                                 prevRunningMeanPtr,
                                 prevRunningVariancePtr,
                                 nextRunningMeanPtr,
                                 nextRunningVariancePtr,
                                 epsilon,
                                 resultSaveMeanPtr,
                                 resultSaveInvVariancePtr,
                                 alpha,
                                 beta);
            }
            else if(_trainingParams.hasSaveMeanVariance())
            {
                hipKernel.Launch(handle.getStream(),
                                 xBuffer.ptr,
                                 activationOutBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasBuffer.ptr,
                                 inhw,
                                 epsilon,
                                 resultSaveMeanPtr,
                                 resultSaveInvVariancePtr,
                                 alpha,
                                 beta);
            }
            else if(_trainingParams.hasRunningStats())
            {
                hipKernel.Launch(handle.getStream(),
                                 xBuffer.ptr,
                                 activationOutBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasBuffer.ptr,
                                 inhw,
                                 expAvgFactor,
                                 prevRunningMeanPtr,
                                 prevRunningVariancePtr,
                                 nextRunningMeanPtr,
                                 nextRunningVariancePtr,
                                 epsilon,
                                 alpha,
                                 beta);
            }
            else
            {
                hipKernel.Launch(handle.getStream(),
                                 xBuffer.ptr,
                                 activationOutBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasBuffer.ptr,
                                 inhw,
                                 epsilon,
                                 alpha,
                                 beta);
            }
        }
        // For variant 2, we still need to implement the launch code!
    }
    else
    {
        auto yBuffer = hip_kernel_utils::findDeviceBuffer(
            _trainingParams.y()->uid(), deviceBuffers, numDeviceBuffers);

        if(variant != 2)
        {
            if(_trainingParams.hasSaveMeanVariance() && _trainingParams.hasRunningStats())
            {
                hipKernel.Launch(handle.getStream(),
                                 xBuffer.ptr,
                                 yBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasBuffer.ptr,
                                 inhw,
                                 expAvgFactor,
                                 prevRunningMeanPtr,
                                 prevRunningVariancePtr,
                                 nextRunningMeanPtr,
                                 nextRunningVariancePtr,
                                 epsilon,
                                 resultSaveMeanPtr,
                                 resultSaveInvVariancePtr,
                                 alpha,
                                 beta);
            }
            else if(_trainingParams.hasSaveMeanVariance())
            {
                hipKernel.Launch(handle.getStream(),
                                 xBuffer.ptr,
                                 yBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasBuffer.ptr,
                                 inhw,
                                 epsilon,
                                 resultSaveMeanPtr,
                                 resultSaveInvVariancePtr,
                                 alpha,
                                 beta);
            }
            else if(_trainingParams.hasRunningStats())
            {
                hipKernel.Launch(handle.getStream(),
                                 xBuffer.ptr,
                                 yBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasBuffer.ptr,
                                 inhw,
                                 expAvgFactor,
                                 prevRunningMeanPtr,
                                 prevRunningVariancePtr,
                                 nextRunningMeanPtr,
                                 nextRunningVariancePtr,
                                 epsilon,
                                 alpha,
                                 beta);
            }
            else
            {
                hipKernel.Launch(handle.getStream(),
                                 xBuffer.ptr,
                                 yBuffer.ptr,
                                 scaleBuffer.ptr,
                                 biasBuffer.ptr,
                                 inhw,
                                 epsilon,
                                 alpha,
                                 beta);
            }
        }
        // For variant 2, we still need to implement the launch code!
    }
}

}
