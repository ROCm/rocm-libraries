// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormBwdPlan.hpp"

#include <string>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "BatchnormCommon.hpp"
#include "BatchnormKernelCompileOptions.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "core/Utils.hpp"

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::batchnorm
{

static ProblemDescription extractProblemDescription(const BatchnormBwdParams& params)
{
    const auto xDataType = params.x()->data_type();
    const auto yDataType = params.dy()->data_type();
    const auto scaleDataType = params.scale()->data_type();
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

    const auto* xDims = params.x()->dims();
    size_t batches = 0;
    size_t channels = 0;
    size_t spatialSize = 0;
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
        const auto height = static_cast<size_t>(xDims->Get(2));
        const auto width = static_cast<size_t>(xDims->Get(3));
        spatialSize = height * width;
    }
    else if(xDims->size() == 5)
    {
        batches = static_cast<size_t>(xDims->Get(0));
        channels = static_cast<size_t>(xDims->Get(1));
        const auto depth = static_cast<size_t>(xDims->Get(2));
        const auto height = static_cast<size_t>(xDims->Get(3));
        const auto width = static_cast<size_t>(xDims->Get(4));
        spatialSize = depth * height * width;
    }
    else
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    return {batches,
            channels,
            spatialSize,
            isChannelLastLayout(params.x()),
            useFp16Mix,
            useBfp16Mix,
            useFp16Upcast,
            useBfp16Upcast,
            Direction::BACKWARD,
            1};
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

BatchnormKernelBuild buildBatchnormBwdKernel(const BatchnormBwdParams& params,
                                             const std::string& gcnArchName)
{
    const auto problem = extractProblemDescription(params);

    // usesSavedStats is true when the caller supplied mean/variance, so the backward
    // pass need not recompute them. That sets how many per-channel fields the
    // multi-workgroup reduction path stashes into the dx-aliased buffer: 4 when it must
    // compute stats (mean, variance, dscale, dbias), 2 when they are supplied (dscale,
    // dbias only). The applicability gate selects that path only when this many fields
    // fit in dx, so this count must not be too small. This matches the logic captured
    // in MIOpen backward_spatial.cpp.
    const bool usesSavedStats = params.hasSavedStats();
    const unsigned int stashValuesBwd = !usesSavedStats ? 4u : 2u;
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

    const int variant = config.variant;

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
    if(params.optActivation().has_value())
    {
        activationMode = (*params.optActivation()).mode;
    }

    BatchnormKernelCompileOptions options(
        params.x(), params.dy(), params.savedMean(), params.scale(), gcnArchName, activationMode);
    options.update("HIP_PLUGIN_USE_FPMIX", problem.useFp16Mix());
    options.update("HIP_PLUGIN_USE_BFPMIX", problem.useBfp16Mix());
    options.update("HIP_PLUGIN_USE_FPUPCAST", problem.useFp16Upcast());
    options.update("HIP_PLUGIN_USE_BFPUPCAST", problem.useBfp16Upcast());
    options.update("HIP_PLUGIN_BN_USESAVED", usesSavedStats);
    options.update("HIP_PLUGIN_BN_BATCHES", problem.batches());
    options.update("HIP_PLUGIN_BN_CHANNELS", problem.channels());
    options.update("HIP_PLUGIN_BN_SPATIAL_SIZE", problem.spatialSize());
    options.update("HIP_PLUGIN_BN_VARIANT", variant);

    if(variant != 2)
    {
        xlocalsize = 1024;
        if(((problem.inCstride() < 256) && (problem.batches() < 256))
           || ((problem.inCstride() < 100) && (problem.batches() <= 256)))
        {
            xlocalsize = 256;
        }
        xgridsize = problem.channels() * xlocalsize;
        ldsSize = static_cast<unsigned int>(xlocalsize);

        options.update("HIP_PLUGIN_BN_GRP0", xlocalsize);
        options.update("HIP_PLUGIN_BN_GRP1", ylocalsize);
        options.update("HIP_PLUGIN_BN_GRP2", zlocalsize);
        options.update("HIP_PLUGIN_BN_LDS_SIZE", ldsSize);
        options.update("HIP_PLUGIN_BN_VEC_SIZE", config.vectorsize);

        std::vector<BatchnormLaunch> launches{
            {"BatchNormBwdSpatial",
             {static_cast<unsigned int>(xlocalsize), 1, 1},
             {static_cast<unsigned int>(xgridsize / xlocalsize), 1, 1}}};
        return {std::move(options), variant, std::move(launches)};
    }
    else
    {
        if(problem.isLayoutChannelLast())
        {
            xgridsize = xlocalsize
                        * ((problem.channels() / config.vectorsize + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize * ((problem.inCstride() + ylocalsize - 1) / ylocalsize);
        }
        else
        {
            xgridsize = xlocalsize * ((problem.channels() + xlocalsize - 1) / xlocalsize);
            ygridsize = ylocalsize
                        * ((problem.inCstride() / config.vectorsize + ylocalsize - 1) / ylocalsize);
        }
        zgridsize = zlocalsize
                    * ((problem.batches() / config.batchElements + zlocalsize - 1) / zlocalsize);

        stashMethod = getStashMethod(problem,
                                     stashValuesBwd,
                                     ylocalsize,
                                     zlocalsize,
                                     config.batchElements,
                                     config.vectorsize);

        if(problem.isLayoutChannelLast() && problem.channels() % 2 == 0 && xlocalsize % 2 == 0)
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
        options.update("HIP_PLUGIN_BN_BATCH_ELEMENTS", config.batchElements);
        options.update("HIP_PLUGIN_BN_LDS_SIZE", ldsSize);
        options.update("HIP_PLUGIN_BN_VEC_SIZE", config.vectorsize);
        options.update("HIP_PLUGIN_BN_STASH_METHOD", stashMethod);
        options.update("HIP_PLUGIN_BN_GRPS0", xgridsize / xlocalsize);
        options.update("HIP_PLUGIN_BN_GRPS1", ygridsize / ylocalsize);
        options.update("HIP_PLUGIN_BN_GRPS2", zgridsize / zlocalsize);
        options.update("HIP_PLUGIN_BN_GRP0_FINAL", xlocalsizeFinal);
        options.update("HIP_PLUGIN_BN_GRP1_FINAL", ylocalsizeFinal);
        options.update("HIP_PLUGIN_BN_GRP2_FINAL", zlocalsizeFinal);

        // The five launches execute() indexes; with saved statistics it runs 0, 1 and 4.
        const std::array<const char*, 5> entryPoints = {
            usesSavedStats ? "BatchNormBwdSpatialDScaleDBias" : "BatchNormBwdSpatialMeanVariance",
            usesSavedStats ? "BatchNormBwdSpatialFinalDScaleDBias"
                           : "BatchNormBwdSpatialFinalMeanVariance",
            usesSavedStats ? "BatchNormBwdSpatialDX" : "BatchNormBwdSpatialDScaleDBias",
            usesSavedStats ? "BatchNormBwdSpatialDX" : "BatchNormBwdSpatialFinalDScaleDBias",
            "BatchNormBwdSpatialDX"};

        std::vector<BatchnormLaunch> launches;
        for(size_t i = 0; i < 5; ++i)
        {
            if(i == 1 || i == 3)
            {
                launches.push_back(
                    {entryPoints[i],
                     {static_cast<unsigned int>(xlocalsizeFinal),
                      static_cast<unsigned int>(ylocalsizeFinal),
                      static_cast<unsigned int>(zlocalsizeFinal)},
                     {static_cast<unsigned int>(xgridsize / xlocalsizeFinal), 1, 1}});
            }
            else
            {
                launches.push_back({entryPoints[i],
                                    {static_cast<unsigned int>(xlocalsize),
                                     static_cast<unsigned int>(ylocalsize),
                                     static_cast<unsigned int>(zlocalsize)},
                                    {static_cast<unsigned int>(xgridsize / xlocalsize),
                                     static_cast<unsigned int>(ygridsize / ylocalsize),
                                     static_cast<unsigned int>(zgridsize / zlocalsize)}});
            }
        }
        return {std::move(options), variant, std::move(launches)};
    }
}

void BatchnormBwdPlan::compile(const IKernelCompiler& kernelCompiler,
                               const hipDeviceProp_t& deviceProperties)
{
    if(_params.optActivation().has_value())
    {
        _activationAlpha = static_cast<float>(_params.optActivation()->alpha);
        _activationBeta = static_cast<float>(_params.optActivation()->beta);
    }

    const auto build = buildBatchnormBwdKernel(_params, std::string(deviceProperties.gcnArchName));
    _kernelVariant = build.variant;

    _compiledProgram = kernelCompiler.compile("BatchNormBwdSpatial.cpp", build.options);
    for(const auto& launch : build.launches)
    {
        _runnableKernels.push_back(_compiledProgram->getKernel(launch.entryPoint));
        _runnableKernels.back()->setBlockSize(
            launch.blockSize[0], launch.blockSize[1], launch.blockSize[2]);
        _runnableKernels.back()->setGridSize(
            launch.gridSize[0], launch.gridSize[1], launch.gridSize[2]);
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
                                    _activationAlpha,
                                    _activationBeta);
    }
    else
    {
        _runnableKernels[0]->launch(handle.getStream(), xBuffer.ptr, dxBuffer.ptr);
        _runnableKernels[1]->launch(handle.getStream(), dxBuffer.ptr, _epsilon);
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
                                    _activationAlpha,
                                    _activationBeta);
    }
}

} // namespace hip_kernel_provider::batchnorm
