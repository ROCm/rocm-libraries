// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormBwdPlan.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"

#include "compilation/IKernelCompiler.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormUtilities.hpp"
#include "hip_kernel_provider_common/HipDeviceUtils.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::layernorm
{

LayernormBwdParams::LayernormBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _dy(tensorMap.at(attributes.dy_tensor_uid()))
    , _x(tensorMap.at(attributes.x_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _mean(attributes.mean_tensor_uid().has_value()
                ? tensorMap.at(attributes.mean_tensor_uid().value())
                : nullptr)
    , _invVariance(attributes.inv_variance_tensor_uid().has_value()
                       ? tensorMap.at(attributes.inv_variance_tensor_uid().value())
                       : nullptr)
    , _epsilon(attributes.epsilon_tensor_uid().has_value()
                   ? std::make_optional(hipdnn_plugin_sdk::makeScalarOperand(
                         tensorMap, attributes.epsilon_tensor_uid().value(), "Epsilon"))
                   : std::nullopt)
    , _dx(tensorMap.at(attributes.dx_tensor_uid()))
    , _dscale(tensorMap.at(attributes.dscale_tensor_uid()))
    , _dbias(tensorMap.at(attributes.dbias_tensor_uid()))
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormBwdParams::dy() const
{
    return _dy;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormBwdParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormBwdParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormBwdParams::mean() const
{
    return _mean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    LayernormBwdParams::invVariance() const
{
    return _invVariance;
}

std::optional<double>
    LayernormBwdParams::epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers) const
{
    return _epsilon.has_value() ? std::make_optional(hipdnn_plugin_sdk::toDouble(
                                      hipdnn_plugin_sdk::resolveScalarOperand(
                                          _epsilon.value(), deviceBuffers, numDeviceBuffers)))
                                : std::nullopt;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormBwdParams::dx() const
{
    return _dx;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormBwdParams::dscale() const
{
    return _dscale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormBwdParams::dbias() const
{
    return _dbias;
}

LayernormBwdPlan::LayernormBwdPlan(LayernormBwdParams&& params)
    : _params(std::move(params))
{
    // Extract dimensions from x tensor
    const auto* xDims = _params.x()->dims();
    const auto* xStrides = _params.x()->strides();
    const auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(
        std::vector<int64_t>(xStrides->begin(), xStrides->end()));

    const size_t normalizedDim
        = layernorm::guessNormalizedDim(_params.x(), _params.scale(), _params.mean());
    int64_t outerSize = 1;
    int64_t innerSize = 1;
    int64_t stride = 1;
    const auto layoutNHWC = hipdnn_data_sdk::utilities::TensorLayout::NHWC;
    const auto layoutNDHWC = hipdnn_data_sdk::utilities::TensorLayout::NDHWC;

    if(normalizedDim > 1
       && (strideOrder == layoutNHWC.strideOrder || strideOrder == layoutNDHWC.strideOrder))
    {
        stride = static_cast<int64_t>(xDims->Get(1));
    }

    for(unsigned int i = 0; i < xDims->size(); ++i)
    {
        if(i < normalizedDim)
        {
            if(stride == 1 || i != 1) // Don't add C to outerSize if there is a stride
            {
                outerSize *= static_cast<int64_t>(xDims->Get(i));
            }
        }
        else
        {
            innerSize *= static_cast<int64_t>(xDims->Get(i));
        }
    }

    _outerSize = outerSize;
    _innerSize = innerSize;
    _stride = stride;
    _localSize = LayernormBwdParams::MAX_LOCAL_SIZE;
}

size_t LayernormBwdPlan::getWorkspaceSize([[maybe_unused]] const Handle& handle) const
{
    auto deviceProperties = hip_kernel_provider_common::getDeviceProperties(handle.getStream());

    // Workspace only needed for the parallel backward pass or for fallback mean/rstd calculations
    const bool parallel = isParallel(deviceProperties,
                                     static_cast<size_t>(_localSize),
                                     static_cast<size_t>(_innerSize),
                                     static_cast<size_t>(_outerSize));
    const size_t parallelSize = getParallelSize(deviceProperties,
                                                static_cast<size_t>(_localSize),
                                                static_cast<size_t>(_innerSize),
                                                static_cast<size_t>(_outerSize));
    const size_t sizeFromParallel
        = parallel ? 2 * static_cast<size_t>(_innerSize) * parallelSize : 0;

    const size_t sizeFromFallbackMeanRstd
        = _params.mean() == nullptr || _params.invVariance() == nullptr
              ? 2 * static_cast<size_t>(_outerSize) * static_cast<size_t>(_stride)
              : 0;

    return 4 * (sizeFromParallel + sizeFromFallbackMeanRstd);
}

void LayernormBwdPlan::compile(const IKernelCompiler& kernelCompiler,
                               const hipDeviceProp_t& deviceProperties)
{
    // Ensure that the input tensor is either 4D or 5D
    const auto* xDims = _params.x()->dims();
    if(xDims->size() != 4 && xDims->size() != 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    // Determine input/output data type configuration
    const auto inputDataType = _params.x()->data_type();
    const auto outputDataType = _params.dy()->data_type();
    const auto scaleBiasDataType = _params.scale()->data_type();
    const auto meanInvVarianceDataType
        = (_params.mean() == nullptr) ? inputDataType : _params.mean()->data_type();
    const std::string inputTypeString = getKernelParamTypeString(inputDataType);
    const std::string outputTypeString = getKernelParamTypeString(outputDataType);
    const std::string scaleBiasTypeString = getKernelParamTypeString(scaleBiasDataType);
    const std::string meanInvVarianceTypeString = getKernelParamTypeString(meanInvVarianceDataType);

    const int64_t gridSize = _outerSize * _stride;
    if(gridSize >= UINT32_MAX)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported number of workgroups: "
                                                           + std::to_string(gridSize));
    }

    _isParallel = isParallel(deviceProperties,
                             static_cast<size_t>(_localSize),
                             static_cast<size_t>(_innerSize),
                             static_cast<size_t>(_outerSize));
    const size_t parallelSize = _isParallel ? getParallelSize(deviceProperties,
                                                              static_cast<size_t>(_localSize),
                                                              static_cast<size_t>(_innerSize),
                                                              static_cast<size_t>(_outerSize))
                                            : 0;

    // Prepare compilation options
    KernelCompileOptions options(_params.x(), deviceProperties);
    options.add("HIP_PLUGIN_LAYERNORM_OUTER_SIZE", _outerSize);
    options.add("HIP_PLUGIN_LAYERNORM_INNER_SIZE", _innerSize);
    options.add("HIP_PLUGIN_LAYERNORM_STRIDE", _stride);
    options.add("HIP_PLUGIN_LAYERNORM_LOCAL_SIZE", _localSize);
    options.add("HIP_PLUGIN_LAYERNORM_PARALLEL_SIZE", parallelSize);
    options.add("HIP_PLUGIN_LAYERNORM_INPUT_TYPE", inputTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_OUTPUT_TYPE", outputTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE", scaleBiasTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE", meanInvVarianceTypeString);

    // Compile kernel and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("LayernormBwd.cpp", options);
    _runnableKernels.push_back(_compiledProgram->getKernel("LayernormBwd"));
    _runnableKernels[0]->setBlockSize(static_cast<unsigned int>(_localSize),
                                      static_cast<unsigned int>(1),
                                      static_cast<unsigned int>(1));
    _runnableKernels[0]->setGridSize(static_cast<unsigned int>(gridSize),
                                     static_cast<unsigned int>(1),
                                     static_cast<unsigned int>(1));

    if(_isParallel)
    {
        HIPDNN_PLUGIN_LOG_INFO("LayernormBwdPlan: parallel");

        _runnableKernels.push_back(_compiledProgram->getKernel("LayernormBwdScaleBiasParallel"));
        _runnableKernels[1]->setBlockSize(static_cast<unsigned int>(_localSize),
                                          static_cast<unsigned int>(1),
                                          static_cast<unsigned int>(1));
        _runnableKernels[1]->setGridSize(
            static_cast<unsigned int>(align(parallelSize * static_cast<size_t>(_innerSize),
                                            static_cast<size_t>(_localSize))),
            static_cast<unsigned int>(1),
            static_cast<unsigned int>(1));

        _runnableKernels.push_back(_compiledProgram->getKernel("LayernormBwdScaleBiasReduceSum"));
        _runnableKernels[2]->setBlockSize(static_cast<unsigned int>(_localSize),
                                          static_cast<unsigned int>(1),
                                          static_cast<unsigned int>(1));
        _runnableKernels[2]->setGridSize(static_cast<unsigned int>(align(_innerSize, _localSize)),
                                         static_cast<unsigned int>(1),
                                         static_cast<unsigned int>(1));
    }
    else
    {
        HIPDNN_PLUGIN_LOG_INFO("LayernormBwdPlan: not parallel");

        _runnableKernels.push_back(_compiledProgram->getKernel("LayernormBwdScaleBias"));
        _runnableKernels[1]->setBlockSize(static_cast<unsigned int>(_localSize),
                                          static_cast<unsigned int>(1),
                                          static_cast<unsigned int>(1));
        _runnableKernels[1]->setGridSize(static_cast<unsigned int>(align(_innerSize, _localSize)),
                                         static_cast<unsigned int>(1),
                                         static_cast<unsigned int>(1));
    }
}

void LayernormBwdPlan::execute(const Handle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               [[maybe_unused]] void* workspace) const
{
    if(_runnableKernels.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "LayernormBwdPlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto dyBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.dy()->uid(), deviceBuffers, numDeviceBuffers);
    auto xBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
        _params.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto meanBuffer = _params.mean() != nullptr
                          ? hipdnn_plugin_sdk::findDeviceBuffer(
                                _params.mean()->uid(), deviceBuffers, numDeviceBuffers)
                          : hipdnnPluginDeviceBuffer_t{-1, nullptr};
    auto invVarianceBuffer = _params.invVariance() != nullptr
                                 ? hipdnn_plugin_sdk::findDeviceBuffer(_params.invVariance()->uid(),
                                                                       deviceBuffers,
                                                                       numDeviceBuffers)
                                 : hipdnnPluginDeviceBuffer_t{-1, nullptr};
    auto dxBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.dx()->uid(), deviceBuffers, numDeviceBuffers);
    auto dscaleBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
        _params.dscale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dbiasBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
        _params.dbias()->uid(), deviceBuffers, numDeviceBuffers);

    auto optionalEpsilon = _params.epsilonValue(deviceBuffers, numDeviceBuffers);
    auto epsilon = optionalEpsilon.has_value()
                       ? optionalEpsilon.value()
                       : hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON;

    // Launch kernels
    _runnableKernels[0]->launch(handle.getStream(),
                                dyBuffer.ptr,
                                xBuffer.ptr,
                                scaleBuffer.ptr,
                                meanBuffer.ptr,
                                invVarianceBuffer.ptr,
                                dxBuffer.ptr,
                                workspace,
                                static_cast<float>(epsilon));
    if(_isParallel)
    {
        _runnableKernels[1]->launch(handle.getStream(),
                                    dyBuffer.ptr,
                                    xBuffer.ptr,
                                    meanBuffer.ptr,
                                    invVarianceBuffer.ptr,
                                    workspace);
        _runnableKernels[2]->launch(
            handle.getStream(), workspace, dscaleBuffer.ptr, dbiasBuffer.ptr);
    }
    else
    {
        _runnableKernels[1]->launch(handle.getStream(),
                                    dyBuffer.ptr,
                                    xBuffer.ptr,
                                    meanBuffer.ptr,
                                    invVarianceBuffer.ptr,
                                    dscaleBuffer.ptr,
                                    dbiasBuffer.ptr,
                                    workspace);
    }
}

size_t LayernormBwdPlan::getReqdWorkItemCount(const hipDeviceProp_t& deviceProperties,
                                              size_t localSize)
{
    return 4 * localSize * static_cast<size_t>(deviceProperties.multiProcessorCount);
}

bool LayernormBwdPlan::isParallel(const hipDeviceProp_t& deviceProperties,
                                  size_t localSize,
                                  size_t innerSize,
                                  size_t outerSize)
{
    const size_t reqdWorkItemCount = getReqdWorkItemCount(deviceProperties, localSize);
    return innerSize <= reqdWorkItemCount && innerSize * outerSize > reqdWorkItemCount;
}

size_t LayernormBwdPlan::getParallelSize(const hipDeviceProp_t& deviceProperties,
                                         size_t localSize,
                                         size_t innerSize,
                                         size_t outerSize)
{
    const size_t reqdWorkItemCount = getReqdWorkItemCount(deviceProperties, localSize);
    size_t parallelSize = 1;
    while(parallelSize * innerSize < reqdWorkItemCount
          && static_cast<double>(parallelSize) < std::sqrt(outerSize))
    {
        parallelSize *= 2;
    }
    return parallelSize;
}

} // namespace hip_kernel_provider::layernorm
