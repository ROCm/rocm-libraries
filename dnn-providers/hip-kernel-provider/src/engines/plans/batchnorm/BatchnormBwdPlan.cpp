// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormBwdPlan.hpp"

#include <string>
#include <utility>

#include <flatbuffers/base.h>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "BatchnormCommon.hpp"
#include "HipKernelUtils.hpp"
#include "hip/HipKernelCompileOptions.hpp"
#include "hip/IKernelCompiler.hpp"

namespace hip_kernel_provider::batchnorm
{

namespace
{
int64_t getRequiredOptionalUid(const flatbuffers::Optional<int64_t>& opt,
                               const std::string& fieldName)
{
    if(!opt.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            fieldName + std::string{" tensor uid is required but not set"});
    }
    return *opt;
}
} // namespace

struct ProblemDims
{
    size_t n = 0;
    size_t c = 0;
    size_t h = 0;
    size_t w = 0;
    unsigned int inCstride = 0;
    unsigned int inNhw = 0;
    unsigned int inChw = 0;
    unsigned int inNchw = 0;
    bool isLayoutNHWC = false;
    bool useFp16Mix = false;
    bool useBfp16Mix = false;
    bool useFp32 = true;
    bool isGfx103X = false;
    bool isGfx110X = false;
    bool isGfx115X = false;
    bool isGfx120X = false;
};

ProblemDims extractProblemDims(const BatchnormBwdParams& params,
                               const hipDeviceProp_t& deviceProperties)
{
    ProblemDims dims{};

    const auto xDataType = params.x()->data_type();
    const auto scaleDataType = params.scale()->data_type();
    dims.useFp16Mix = (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF
                       && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    dims.useBfp16Mix = (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16
                        && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    dims.useFp32 = !dims.useFp16Mix && !dims.useBfp16Mix;

    const auto* xDims = params.x()->dims();
    if(xDims->size() == 4)
    {
        dims.n = static_cast<size_t>(xDims->Get(0));
        dims.c = static_cast<size_t>(xDims->Get(1));
        dims.h = static_cast<size_t>(xDims->Get(2));
        dims.w = static_cast<size_t>(xDims->Get(3));
    }
    else if(xDims->size() == 5)
    {
        dims.n = static_cast<size_t>(xDims->Get(0));
        dims.c = static_cast<size_t>(xDims->Get(1));
        const auto d = static_cast<size_t>(xDims->Get(2));
        dims.h = d * static_cast<size_t>(xDims->Get(3));
        dims.w = static_cast<size_t>(xDims->Get(4));
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    dims.inCstride = static_cast<unsigned int>(dims.h * dims.w);
    dims.inNhw = static_cast<unsigned int>(dims.n) * dims.inCstride;
    dims.inChw = static_cast<unsigned int>(dims.c) * dims.inCstride;
    dims.inNchw = static_cast<unsigned int>(dims.n) * dims.inChw;
    dims.isLayoutNHWC = hip_kernel_utils::isChannelLastLayout(params.x());

    const std::string archName(deviceProperties.gcnArchName);
    dims.isGfx103X = archName.find("gfx103") == 0;
    dims.isGfx110X = archName.find("gfx110") == 0;
    dims.isGfx115X = archName.find("gfx115") == 0;
    dims.isGfx120X = archName.find("gfx120") == 0;

    return dims;
}

BatchnormBwdParams::BatchnormBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(&hip_kernel_utils::findTensorAttributes(tensorMap, attributes.x_tensor_uid()))
    , _dy(&hip_kernel_utils::findTensorAttributes(tensorMap, attributes.dy_tensor_uid()))
    , _dx(&hip_kernel_utils::findTensorAttributes(tensorMap, attributes.dx_tensor_uid()))
    , _scale(&hip_kernel_utils::findTensorAttributes(tensorMap, attributes.scale_tensor_uid()))
    , _dscale(&hip_kernel_utils::findTensorAttributes(tensorMap, attributes.dscale_tensor_uid()))
    , _dbias(&hip_kernel_utils::findTensorAttributes(tensorMap, attributes.dbias_tensor_uid()))
{
    if(attributes.mean_tensor_uid().has_value())
    {
        _savedMean = &hip_kernel_utils::findTensorAttributes(tensorMap,
                                                             attributes.mean_tensor_uid().value());
    }
    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _savedInvVariance = &hip_kernel_utils::findTensorAttributes(
            tensorMap, attributes.inv_variance_tensor_uid().value());
    }
}

BatchnormBwdParams::BatchnormBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes&
        batchnormBackwardAttributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes&
        batchnormInferenceAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(&hip_kernel_utils::findTensorAttributes(tensorMap,
                                                 batchnormBackwardAttributes.x_tensor_uid()))
    , _dy(&hip_kernel_utils::findTensorAttributes(
          tensorMap,
          getRequiredOptionalUid(pointwiseAttributes.in_1_tensor_uid(), "Pointwise in_1")))
    , _dx(&hip_kernel_utils::findTensorAttributes(tensorMap,
                                                  batchnormBackwardAttributes.dx_tensor_uid()))
    , _scale(&hip_kernel_utils::findTensorAttributes(
          tensorMap, batchnormBackwardAttributes.scale_tensor_uid()))
    , _dscale(&hip_kernel_utils::findTensorAttributes(
          tensorMap, batchnormBackwardAttributes.dscale_tensor_uid()))
    , _dbias(&hip_kernel_utils::findTensorAttributes(
          tensorMap, batchnormBackwardAttributes.dbias_tensor_uid()))
    , _optActivation(hip_kernel_utils::parseActivation(pointwiseAttributes))
    , _bias(&hip_kernel_utils::findTensorAttributes(tensorMap,
                                                    batchnormInferenceAttributes.bias_tensor_uid()))
{
    if(batchnormBackwardAttributes.mean_tensor_uid().has_value())
    {
        _savedMean = &hip_kernel_utils::findTensorAttributes(
            tensorMap, batchnormBackwardAttributes.mean_tensor_uid().value());
    }
    if(batchnormBackwardAttributes.inv_variance_tensor_uid().has_value())
    {
        _savedInvVariance = &hip_kernel_utils::findTensorAttributes(
            tensorMap, batchnormBackwardAttributes.inv_variance_tensor_uid().value());
    }
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormBwdParams::x() const
{
    return _x;
}
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormBwdParams::dy() const
{
    return _dy;
}
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormBwdParams::dx() const
{
    return _dx;
}
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormBwdParams::scale() const
{
    return _scale;
}
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormBwdParams::dscale() const
{
    return _dscale;
}
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormBwdParams::dbias() const
{
    return _dbias;
}
bool BatchnormBwdParams::hasSavedStats() const
{
    return _savedMean != nullptr && _savedInvVariance != nullptr;
}
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormBwdParams::savedMean() const
{
    return _savedMean;
}
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    BatchnormBwdParams::savedInvVariance() const
{
    return _savedInvVariance;
}
const std::optional<hip_kernel_utils::ActivationParams>& BatchnormBwdParams::optActivation() const
{
    return _optActivation;
}
const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* BatchnormBwdParams::bias() const
{
    return _bias;
}

BatchnormBwdPlan::BatchnormBwdPlan(BatchnormBwdParams&& params)
    : _params(std::move(params))
    , _usesSavedStats(_params.hasSavedStats())
    , _epsilon(hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON)
{
}

size_t BatchnormBwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    return 0;
}

void BatchnormBwdPlan::compile(const IKernelCompiler& kernelCompiler,
                               const hipDeviceProp_t& deviceProperties)
{
    const auto dims = extractProblemDims(_params, deviceProperties);

    if(_params.optActivation().has_value())
    {
        _activationAlpha = static_cast<float>(_params.optActivation()->alpha);
        _activationBeta = static_cast<float>(_params.optActivation()->beta);
    }

    constexpr unsigned int STASH_VALUES_BWD = 2;
    KernelConfig config;
    if(useMultiple(dims.n,
                   dims.h,
                   dims.w,
                   dims.useFp16Mix || dims.useBfp16Mix,
                   dims.isLayoutNHWC,
                   Direction::BACKWARD))
    {
        const size_t minWorkgroups = 1;
        defaultConfigSpatialMultiple(dims.n,
                                     dims.c,
                                     dims.h,
                                     dims.w,
                                     dims.isLayoutNHWC,
                                     dims.useFp32,
                                     minWorkgroups,
                                     STASH_VALUES_BWD,
                                     config);
        if(config.variant == -1)
        {
            defaultConfigSpatialSingle(dims.n,
                                       dims.h,
                                       dims.w,
                                       dims.useFp16Mix,
                                       dims.useBfp16Mix,
                                       dims.isLayoutNHWC,
                                       Direction::BACKWARD,
                                       config);
        }
    }
    else
    {
        defaultConfigSpatialSingle(dims.n,
                                   dims.h,
                                   dims.w,
                                   dims.useFp16Mix,
                                   dims.useBfp16Mix,
                                   dims.isLayoutNHWC,
                                   Direction::BACKWARD,
                                   config);
    }

    _kernelVariant = config.variant;
    _invInNhw = 1.0f / static_cast<float>(dims.inNhw);

    size_t xlocalsize = config.xlocalsize;
    size_t ylocalsize = config.ylocalsize;
    size_t zlocalsize = config.zlocalsize;
    size_t xgridsize = 1;
    size_t ygridsize = 1;
    size_t zgridsize = 1;
    size_t xlocalsizeFinal = xlocalsize;
    size_t ylocalsizeFinal = ylocalsize;
    size_t zlocalsizeFinal = zlocalsize;
    int stashMethod = 0;
    unsigned int ldsSize = 0;

    // Get activation mode
    auto activationMode = hip_kernel_utils::ActivationMode::PASTHRU;
    if(_params.optActivation().has_value())
    {
        activationMode = (*_params.optActivation()).mode;
    }

    HipKernelCompileOptions options(_params.x(), deviceProperties, activationMode);
    options.add("HIP_PLUGIN_USE_FPMIX", dims.useFp16Mix);
    options.add("HIP_PLUGIN_USE_BFPMIX", dims.useBfp16Mix);
    // Not using FP16 and BFP16 paths due to affine data type requirements
    options.update("HIP_PLUGIN_USE_FP16", 0);
    options.update("HIP_PLUGIN_USE_BFP16", 0);
    options.add("HIP_PLUGIN_BN_GFX103X", dims.isGfx103X);
    options.add("HIP_PLUGIN_BN_GFX110X", dims.isGfx110X);
    options.add("HIP_PLUGIN_BN_GFX115X", dims.isGfx115X);
    options.add("HIP_PLUGIN_BN_GFX120X", dims.isGfx120X);
    options.add("HIP_PLUGIN_BN_USESAVED", _usesSavedStats);
    options.add("HIP_PLUGIN_BN_N", dims.n);
    options.add("HIP_PLUGIN_BN_C", dims.c);
    options.add("HIP_PLUGIN_BN_HW", dims.inCstride);
    options.add("HIP_PLUGIN_BN_NHW", dims.inNhw);
    options.add("HIP_PLUGIN_BN_CHW", dims.inChw);
    options.add("HIP_PLUGIN_BN_NCHW", dims.inNchw);
    options.add("HIP_PLUGIN_BN_VARIANT", _kernelVariant);

    if(_kernelVariant != 2)
    {
        xlocalsize = 1024;
        if(((dims.inCstride < 256) && (dims.n < 256))
           || ((dims.inCstride < 100) && (dims.n <= 256)))
        {
            xlocalsize = 256;
        }
        xgridsize = dims.c * xlocalsize;
        ldsSize = static_cast<unsigned int>(xlocalsize);

        options.add("HIP_PLUGIN_BN_GRP0", xlocalsize);
        options.add("HIP_PLUGIN_BN_GRP1", ylocalsize);
        options.add("HIP_PLUGIN_BN_GRP2", zlocalsize);
        options.add("HIP_PLUGIN_BN_LDS_SIZE", ldsSize);
        options.add("HIP_PLUGIN_BN_MAXN", 65);
        options.add("HIP_PLUGIN_BN_VEC_SIZE", config.vectorsize);

        _compiledProgram = kernelCompiler.compile("BatchNormBwdSpatial.cpp", options);
        _runnableKernels.push_back(_compiledProgram->getKernel("BatchNormBwdSpatial"));
        _runnableKernels[0]->setBlockSize(static_cast<unsigned int>(xlocalsize), 1, 1);
        _runnableKernels[0]->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize), 1, 1);
    }
    else
    {
        if(dims.isLayoutNHWC)
        {
            xgridsize = xlocalsize * ((dims.c / config.vectorsize + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((dims.inCstride + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((dims.c + xlocalsize - 1) / xlocalsize);
            ygridsize
                = ylocalsize * ((dims.inCstride / config.vectorsize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize * ((dims.n / config.nelements + zlocalsize - 1) / zlocalsize);

        stashMethod = getStashMethod(dims.isLayoutNHWC,
                                     dims.useFp32,
                                     STASH_VALUES_BWD,
                                     dims.c,
                                     dims.n,
                                     dims.inCstride,
                                     ylocalsize,
                                     zlocalsize,
                                     config.nelements);

        if(dims.isLayoutNHWC && dims.c % 2 == 0 && xlocalsize % 2 == 0)
        {
            xlocalsizeFinal = 2;
            zlocalsizeFinal = zgridsize / zlocalsize * zlocalsize;
            ylocalsizeFinal
                = (xlocalsize * ylocalsize * zlocalsize) / xlocalsizeFinal / zlocalsizeFinal;
            if(ylocalsizeFinal == 0)
            {
                ylocalsizeFinal = 1;
            }
        }
        ldsSize = static_cast<unsigned int>(xlocalsize * ylocalsize * zlocalsize);

        options.add("HIP_PLUGIN_BN_GRP0", xlocalsize);
        options.add("HIP_PLUGIN_BN_GRP1", ylocalsize);
        options.add("HIP_PLUGIN_BN_GRP2", zlocalsize);
        options.add("HIP_PLUGIN_BN_GRP0_FINAL", xlocalsizeFinal);
        options.add("HIP_PLUGIN_BN_GRP1_FINAL", ylocalsizeFinal);
        options.add("HIP_PLUGIN_BN_GRP2_FINAL", zlocalsizeFinal);
        options.add("HIP_PLUGIN_BN_NGRPS", ygridsize / ylocalsize);
        options.add("HIP_PLUGIN_BN_NGRPS2", zgridsize / zlocalsize);
        options.add("HIP_PLUGIN_BN_N_ELEMENTS", config.nelements);
        options.add("HIP_PLUGIN_BN_LDS_SIZE", ldsSize);
        options.add("HIP_PLUGIN_BN_VEC_SIZE", config.vectorsize);
        options.add("HIP_PLUGIN_BN_STASH_METHOD", stashMethod);

        _compiledProgram = kernelCompiler.compile("BatchNormBwdSpatial.cpp", options);
        _runnableKernels.push_back(
            _compiledProgram->getKernel(_usesSavedStats ? "BatchNormBwdSpatialDScaleDBias"
                                                        : "BatchNormBwdSpatialMeanVariance"));
        _runnableKernels.push_back(
            _compiledProgram->getKernel(_usesSavedStats ? "BatchNormBwdSpatialFinalDScaleDBias"
                                                        : "BatchNormBwdSpatialFinalMeanVariance"));
        _runnableKernels.push_back(_compiledProgram->getKernel(
            _usesSavedStats ? "BatchNormBwdSpatialDX" : "BatchNormBwdSpatialDScaleDBias"));
        _runnableKernels.push_back(_compiledProgram->getKernel(
            _usesSavedStats ? "BatchNormBwdSpatialDX" : "BatchNormBwdSpatialFinalDScaleDBias"));
        _runnableKernels.push_back(_compiledProgram->getKernel("BatchNormBwdSpatialDX"));

        for(size_t i = 0; i < 5; ++i)
        {
            if(i == 1 || i == 3)
            {
                _runnableKernels[i]->setBlockSize(static_cast<unsigned int>(xlocalsizeFinal),
                                                  static_cast<unsigned int>(ylocalsizeFinal),
                                                  static_cast<unsigned int>(zlocalsizeFinal));
                _runnableKernels[i]->setGridSize(
                    static_cast<unsigned int>(xgridsize / xlocalsizeFinal), 1, 1);
            }
            else
            {
                _runnableKernels[i]->setBlockSize(static_cast<unsigned int>(xlocalsize),
                                                  static_cast<unsigned int>(ylocalsize),
                                                  static_cast<unsigned int>(zlocalsize));
                _runnableKernels[i]->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize),
                                                 static_cast<unsigned int>(ygridsize / ylocalsize),
                                                 static_cast<unsigned int>(zgridsize / zlocalsize));
            }
        }
    }
}

void BatchnormBwdPlan::execute(const HipKernelHandle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               [[maybe_unused]] void* workspace) const
{
    if(_runnableKernels.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "BatchnormBwdPlan::execute() called before compile()");
    }

    auto xBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto dyBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.dy()->uid(), deviceBuffers, numDeviceBuffers);
    auto dxBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.dx()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dscaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.dscale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dbiasBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.dbias()->uid(), deviceBuffers, numDeviceBuffers);

    void* biasPtr = nullptr;
    if(_params.bias() != nullptr)
    {
        biasPtr = hip_kernel_utils::findDeviceBuffer(
                      _params.bias()->uid(), deviceBuffers, numDeviceBuffers)
                      .ptr;
    }

    void* savedMeanPtr = nullptr;
    void* savedInvVariancePtr = nullptr;
    if(_usesSavedStats)
    {
        savedMeanPtr = hip_kernel_utils::findDeviceBuffer(
                           _params.savedMean()->uid(), deviceBuffers, numDeviceBuffers)
                           .ptr;
        savedInvVariancePtr = hip_kernel_utils::findDeviceBuffer(_params.savedInvVariance()->uid(),
                                                                 deviceBuffers,
                                                                 numDeviceBuffers)
                                  .ptr;
    }

    if(_kernelVariant != 2)
    {
        if(_usesSavedStats)
        {
            _runnableKernels[0]->launch(handle.getStream(),
                                        xBuffer.ptr,
                                        dyBuffer.ptr,
                                        dxBuffer.ptr,
                                        scaleBuffer.ptr,
                                        biasPtr,
                                        dscaleBuffer.ptr,
                                        dbiasBuffer.ptr,
                                        savedMeanPtr,
                                        savedInvVariancePtr,
                                        _invInNhw,
                                        _activationAlpha,
                                        _activationBeta);
        }
        else
        {
            _runnableKernels[0]->launch(handle.getStream(),
                                        xBuffer.ptr,
                                        dyBuffer.ptr,
                                        dxBuffer.ptr,
                                        scaleBuffer.ptr,
                                        biasPtr,
                                        dscaleBuffer.ptr,
                                        dbiasBuffer.ptr,
                                        _epsilon,
                                        _invInNhw,
                                        _activationAlpha,
                                        _activationBeta);
        }
        return;
    }

    if(_usesSavedStats)
    {
        _runnableKernels[0]->launch(handle.getStream(),
                                    xBuffer.ptr,
                                    dyBuffer.ptr,
                                    dxBuffer.ptr,
                                    scaleBuffer.ptr,
                                    biasPtr,
                                    savedMeanPtr,
                                    savedInvVariancePtr,
                                    _activationAlpha,
                                    _activationBeta);
        _runnableKernels[1]->launch(
            handle.getStream(), dxBuffer.ptr, dscaleBuffer.ptr, dbiasBuffer.ptr);
        _runnableKernels[4]->launch(handle.getStream(),
                                    xBuffer.ptr,
                                    dyBuffer.ptr,
                                    dxBuffer.ptr,
                                    scaleBuffer.ptr,
                                    biasPtr,
                                    dscaleBuffer.ptr,
                                    dbiasBuffer.ptr,
                                    savedMeanPtr,
                                    savedInvVariancePtr,
                                    _invInNhw,
                                    _activationAlpha,
                                    _activationBeta);
    }
    else
    {
        _runnableKernels[0]->launch(handle.getStream(), xBuffer.ptr, dxBuffer.ptr);
        _runnableKernels[1]->launch(handle.getStream(), dxBuffer.ptr, _invInNhw, _epsilon);
        _runnableKernels[2]->launch(handle.getStream(),
                                    xBuffer.ptr,
                                    dyBuffer.ptr,
                                    dxBuffer.ptr,
                                    scaleBuffer.ptr,
                                    biasPtr,
                                    _activationAlpha,
                                    _activationBeta);
        _runnableKernels[3]->launch(
            handle.getStream(), dxBuffer.ptr, dscaleBuffer.ptr, dbiasBuffer.ptr);
        _runnableKernels[4]->launch(handle.getStream(),
                                    xBuffer.ptr,
                                    dyBuffer.ptr,
                                    dxBuffer.ptr,
                                    scaleBuffer.ptr,
                                    biasPtr,
                                    dscaleBuffer.ptr,
                                    dbiasBuffer.ptr,
                                    _invInNhw,
                                    _activationAlpha,
                                    _activationBeta);
    }
}

} // namespace hip_kernel_provider::batchnorm
