// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormBwdPlan.hpp"

#include <string>
#include <utility>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "BatchnormCommon.hpp"
#include "BatchnormKernelCompileOptions.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "core/Utils.hpp"

namespace hip_kernel_provider::batchnorm
{

static ProblemDescription extractProblemDescription(const BatchnormBwdParams& params)
{
    const auto xDataType = params.x()->data_type();
    const auto scaleDataType = params.scale()->data_type();
    const bool useFp16Mix
        = (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::HALF
           && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const bool useBfp16Mix
        = (xDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16
           && scaleDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const auto* xDims = params.x()->dims();
    size_t n = 0;
    size_t c = 0;
    size_t h = 0;
    size_t w = 0;
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
        const auto d = static_cast<size_t>(xDims->Get(2));
        h = d * static_cast<size_t>(xDims->Get(3));
        w = static_cast<size_t>(xDims->Get(4));
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    return ProblemDescription(n,
                              c,
                              h,
                              w,
                              isChannelLastLayout(params.x()),
                              useFp16Mix,
                              useBfp16Mix,
                              Direction::BACKWARD,
                              1);
}

BatchnormBwdParams::BatchnormBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(&findTensorAttributes(tensorMap, attributes.x_tensor_uid()))
    , _dy(&findTensorAttributes(tensorMap, attributes.dy_tensor_uid()))
    , _dx(&findTensorAttributes(tensorMap, attributes.dx_tensor_uid()))
    , _scale(&findTensorAttributes(tensorMap, attributes.scale_tensor_uid()))
    , _dscale(&findTensorAttributes(tensorMap, attributes.dscale_tensor_uid()))
    , _dbias(&findTensorAttributes(tensorMap, attributes.dbias_tensor_uid()))
{
    if(attributes.mean_tensor_uid().has_value())
    {
        _savedMean = &findTensorAttributes(tensorMap, attributes.mean_tensor_uid().value());
    }
    if(attributes.inv_variance_tensor_uid().has_value())
    {
        _savedInvVariance
            = &findTensorAttributes(tensorMap, attributes.inv_variance_tensor_uid().value());
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
    : _x(&findTensorAttributes(tensorMap, batchnormBackwardAttributes.x_tensor_uid()))
    , _dy(&findTensorAttributes(tensorMap, pointwiseAttributes.in_0_tensor_uid()))
    , _dx(&findTensorAttributes(tensorMap, batchnormBackwardAttributes.dx_tensor_uid()))
    , _scale(&findTensorAttributes(tensorMap, batchnormBackwardAttributes.scale_tensor_uid()))
    , _dscale(&findTensorAttributes(tensorMap, batchnormBackwardAttributes.dscale_tensor_uid()))
    , _dbias(&findTensorAttributes(tensorMap, batchnormBackwardAttributes.dbias_tensor_uid()))
    , _optActivation(parseActivation(pointwiseAttributes))
    , _bias(&findTensorAttributes(tensorMap, batchnormInferenceAttributes.bias_tensor_uid()))
{
    if(batchnormBackwardAttributes.mean_tensor_uid().has_value())
    {
        _savedMean = &findTensorAttributes(tensorMap,
                                           batchnormBackwardAttributes.mean_tensor_uid().value());
    }
    if(batchnormBackwardAttributes.inv_variance_tensor_uid().has_value())
    {
        _savedInvVariance = &findTensorAttributes(
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
const std::optional<ActivationParams>& BatchnormBwdParams::optActivation() const
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

size_t BatchnormBwdPlan::getWorkspaceSize([[maybe_unused]] const Handle& handle) const
{
    return 0;
}

void BatchnormBwdPlan::compile(const IKernelCompiler& kernelCompiler,
                               const hipDeviceProp_t& deviceProperties)
{
    const auto problem = extractProblemDescription(_params);

    if(_params.optActivation().has_value())
    {
        _activationAlpha = static_cast<float>(_params.optActivation()->alpha);
        _activationBeta = static_cast<float>(_params.optActivation()->beta);
    }

    // _usesSavedStats is true when the caller supplied mean/variance, so the backward
    // pass need not recompute them. That sets how many per-channel fields the
    // multi-workgroup reduction path stashes into the dx-aliased buffer: 4 when it must
    // compute stats (mean, variance, dscale, dbias), 2 when they are supplied (dscale,
    // dbias only). The applicability gate selects that path only when this many fields
    // fit in dx, so this count must not be too small. This matches the logic captured
    // in MIOpen backward_spatial.cpp.
    const unsigned int stashValuesBwd = !_usesSavedStats ? 4u : 2u;
    KernelConfig config;
    if(useMultiple(problem))
    {
        defaultConfigSpatialMultiple(problem, stashValuesBwd, config);
        if(config.variant == -1)
        {
            defaultConfigSpatialSingle(problem, config);
        }
    }
    else
    {
        defaultConfigSpatialSingle(problem, config);
    }

    _kernelVariant = config.variant;
    _invInNhw = 1.0f / static_cast<float>(problem.inNhw());

    size_t xlocalsize = config.xlocalsize;
    const size_t ylocalsize = config.ylocalsize;
    const size_t zlocalsize = config.zlocalsize;
    size_t xgridsize = 1;
    size_t ygridsize = 1;
    size_t zgridsize = 1;
    size_t xlocalsizeFinal = xlocalsize;
    size_t ylocalsizeFinal = ylocalsize;
    size_t zlocalsizeFinal = zlocalsize;
    int stashMethod = 0;
    unsigned int ldsSize = 0;

    // Get activation mode
    auto activationMode = ActivationMode::PASTHRU;
    if(_params.optActivation().has_value())
    {
        activationMode = (*_params.optActivation()).mode;
    }

    BatchnormKernelCompileOptions options(_params.x(),
                                          _params.dx(),
                                          _params.savedMean(),
                                          _params.scale(),
                                          deviceProperties,
                                          activationMode);
    options.update("HIP_PLUGIN_USE_FPMIX", problem.useFp16Mix());
    options.update("HIP_PLUGIN_USE_BFPMIX", problem.useBfp16Mix());
    // Not using FP16 and BFP16 paths due to affine data type requirements
    options.update("HIP_PLUGIN_USE_FP16", 0);
    options.update("HIP_PLUGIN_USE_BFP16", 0);
    options.update("HIP_PLUGIN_BN_USESAVED", _usesSavedStats);
    options.update("HIP_PLUGIN_BN_N", problem.n());
    options.update("HIP_PLUGIN_BN_C", problem.c());
    options.update("HIP_PLUGIN_BN_HW", problem.inCstride());
    options.update("HIP_PLUGIN_BN_NHW", problem.inNhw());
    options.update("HIP_PLUGIN_BN_CHW", problem.inChw());
    options.update("HIP_PLUGIN_BN_NCHW", problem.inNchw());
    options.update("HIP_PLUGIN_BN_VARIANT", _kernelVariant);

    if(_kernelVariant != 2)
    {
        xlocalsize = 1024;
        if(((problem.inCstride() < 256) && (problem.n() < 256))
           || ((problem.inCstride() < 100) && (problem.n() <= 256)))
        {
            xlocalsize = 256;
        }
        xgridsize = problem.c() * xlocalsize;
        ldsSize = static_cast<unsigned int>(xlocalsize);

        options.update("HIP_PLUGIN_BN_GRP0", xlocalsize);
        options.update("HIP_PLUGIN_BN_GRP1", ylocalsize);
        options.update("HIP_PLUGIN_BN_GRP2", zlocalsize);
        options.update("HIP_PLUGIN_BN_LDS_SIZE", ldsSize);
        options.update("HIP_PLUGIN_BN_VEC_SIZE", config.vectorsize);

        _compiledProgram = kernelCompiler.compile("BatchNormBwdSpatial.cpp", options);
        _runnableKernels.push_back(_compiledProgram->getKernel("BatchNormBwdSpatial"));
        _runnableKernels[0]->setBlockSize(static_cast<unsigned int>(xlocalsize), 1, 1);
        _runnableKernels[0]->setGridSize(static_cast<unsigned int>(xgridsize / xlocalsize), 1, 1);
    }
    else
    {
        if(problem.isLayoutNHWC())
        {
            xgridsize
                = xlocalsize * ((problem.c() / config.vectorsize + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((problem.inCstride() + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((problem.c() + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize
                        * ((problem.inCstride() / config.vectorsize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize * ((problem.n() / config.nelements + zlocalsize - 1) / zlocalsize);

        stashMethod
            = getStashMethod(problem, stashValuesBwd, ylocalsize, zlocalsize, config.nelements);

        if(problem.isLayoutNHWC() && problem.c() % 2 == 0 && xlocalsize % 2 == 0)
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

        options.update("HIP_PLUGIN_BN_GRP0", xlocalsize);
        options.update("HIP_PLUGIN_BN_GRP1", ylocalsize);
        options.update("HIP_PLUGIN_BN_GRP2", zlocalsize);
        options.update("HIP_PLUGIN_BN_N_ELEMENTS", config.nelements);
        options.update("HIP_PLUGIN_BN_LDS_SIZE", ldsSize);
        options.update("HIP_PLUGIN_BN_VEC_SIZE", config.vectorsize);
        options.update("HIP_PLUGIN_BN_STASH_METHOD", stashMethod);

        options.add("HIP_PLUGIN_BN_NGRPS", ygridsize / ylocalsize);
        options.add("HIP_PLUGIN_BN_NGRPS2", zgridsize / zlocalsize);
        options.add("HIP_PLUGIN_BN_GRP0_FINAL", xlocalsizeFinal);
        options.add("HIP_PLUGIN_BN_GRP1_FINAL", ylocalsizeFinal);
        options.add("HIP_PLUGIN_BN_GRP2_FINAL", zlocalsizeFinal);

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

void BatchnormBwdPlan::execute(const Handle& handle,
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
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto dyBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.dy()->uid(), deviceBuffers, numDeviceBuffers);
    auto dxBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.dx()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
        _params.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dscaleBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
        _params.dscale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dbiasBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
        _params.dbias()->uid(), deviceBuffers, numDeviceBuffers);

    void* biasPtr = nullptr;
    if(_params.bias() != nullptr)
    {
        biasPtr = hipdnn_plugin_sdk::findDeviceBuffer(
                      _params.bias()->uid(), deviceBuffers, numDeviceBuffers)
                      .ptr;
    }

    void* savedMeanPtr = nullptr;
    void* savedInvVariancePtr = nullptr;
    if(_usesSavedStats)
    {
        savedMeanPtr = hipdnn_plugin_sdk::findDeviceBuffer(
                           _params.savedMean()->uid(), deviceBuffers, numDeviceBuffers)
                           .ptr;
        savedInvVariancePtr = hipdnn_plugin_sdk::findDeviceBuffer(_params.savedInvVariance()->uid(),
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
