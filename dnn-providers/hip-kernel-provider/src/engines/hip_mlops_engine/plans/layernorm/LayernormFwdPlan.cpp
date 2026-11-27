// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormFwdPlan.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"

#include "compilation/IKernelCompiler.hpp"
#include "core/Utils.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormUtilities.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::layernorm
{

LayernormFwdParams::LayernormFwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(tensorMap.at(attributes.bias_tensor_uid()))
    , _mean(attributes.mean_tensor_uid().has_value()
                ? tensorMap.at(attributes.mean_tensor_uid().value())
                : nullptr)
    , _invVariance(attributes.inv_variance_tensor_uid().has_value()
                       ? tensorMap.at(attributes.inv_variance_tensor_uid().value())
                       : nullptr)
    , _epsilon(hipdnn_plugin_sdk::makeScalarOperand(
          tensorMap, attributes.epsilon_tensor_uid(), "Epsilon"))
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::y() const
{
    return _y;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::bias() const
{
    return _bias;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* LayernormFwdParams::mean() const
{
    return _mean;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    LayernormFwdParams::invVariance() const
{
    return _invVariance;
}

double LayernormFwdParams::epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                        uint32_t numDeviceBuffers) const
{
    return hipdnn_plugin_sdk::toDouble(
        hipdnn_plugin_sdk::resolveScalarOperand(_epsilon, deviceBuffers, numDeviceBuffers));
}

LayernormFwdPlan::LayernormFwdPlan(LayernormFwdParams&& params)
    : _params(std::move(params))
{
}

std::vector<KernelLaunchShape>
    LayernormFwdPlan::launchShapes(int64_t outerSize, int64_t stride, int64_t localSize)
{
    return {{"LayernormFwd", outerSize * stride, localSize}};
}

size_t LayernormFwdPlan::getWorkspaceSize([[maybe_unused]] const Handle& handle) const
{
    // No workspace needed for layernorm
    return 0;
}

namespace
{

ProblemDescription fwdProblem(const LayernormFwdParams& params)
{
    const auto statAttr
        = params.mean() == nullptr ? std::nullopt : std::make_optional(params.mean());
    return {params.x(), params.scale(), statAttr, Direction::FORWARD};
}

} // namespace

LayernormFwdConfig defaultLayernormFwdConfig()
{
    return {LayernormFwdParams::LOCAL_SIZE};
}

std::vector<KernelLaunchShape> layernormFwdLaunchShapes(const LayernormFwdParams& params,
                                                        const LayernormFwdConfig& config)
{
    const auto problem = fwdProblem(params);
    return LayernormFwdPlan::launchShapes(problem.outerSize(), problem.stride(), config.localSize);
}

LayernormKernelBuild buildLayernormFwdKernel(const LayernormFwdParams& params,
                                             const std::string& gcnArchName,
                                             const LayernormFwdConfig& config)
{
    // Extract dimensions from x tensor
    const auto* xDims = params.x()->dims();
    // Ensure that the input tensor is either 4D or 5D
    if(xDims->size() != 4 && xDims->size() != 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    const auto problem = fwdProblem(params);
    const int64_t outerSize = problem.outerSize();
    const int64_t innerSize = problem.innerSize();
    const int64_t stride = problem.stride();

    const auto shapes = LayernormFwdPlan::launchShapes(outerSize, stride, config.localSize);
    for(const auto& shape : shapes)
    {
        checkLaunchFitsThreadLimit(shape);
    }

    const int64_t xlocalsize = shapes[0].threadsPerBlock;
    const int64_t xgridsize = shapes[0].blocks;

    // Determine input/output data type configuration
    const auto inputDataType = params.x()->data_type();
    const auto outputDataType = params.y()->data_type();
    const auto scaleBiasDataType = params.scale()->data_type();
    const auto meanInvVarianceDataType
        = (params.mean() == nullptr) ? inputDataType : params.mean()->data_type();
    const std::string inputTypeString = getKernelParamTypeString(inputDataType);
    const std::string outputTypeString = getKernelParamTypeString(outputDataType);
    const std::string scaleBiasTypeString = getKernelParamTypeString(scaleBiasDataType);
    const std::string meanInvVarianceTypeString = getKernelParamTypeString(meanInvVarianceDataType);

    // Prepare compilation options
    KernelCompileOptions options(params.x(), gcnArchName);
    options.add("HIP_PLUGIN_LAYERNORM_OUTER_SIZE", outerSize);
    options.add("HIP_PLUGIN_LAYERNORM_INNER_SIZE", innerSize);
    options.add("HIP_PLUGIN_LAYERNORM_STRIDE", stride);
    options.add("HIP_PLUGIN_LAYERNORM_LOCAL_SIZE", xlocalsize);
    options.add("HIP_PLUGIN_LAYERNORM_INPUT_TYPE", inputTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_OUTPUT_TYPE", outputTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE", scaleBiasTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE", meanInvVarianceTypeString);

    return {std::move(options),
            {{"LayernormFwd",
              static_cast<unsigned int>(xlocalsize),
              static_cast<unsigned int>(xgridsize)}}};
}

void LayernormFwdPlan::compile(const IKernelCompiler& kernelCompiler,
                               const hipDeviceProp_t& deviceProperties)
{
    const auto build = buildLayernormFwdKernel(
        _params, std::string(deviceProperties.gcnArchName), defaultLayernormFwdConfig());
    const auto& launch = build.launches.front();

    // Compile kernel and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("LayernormFwd.cpp", build.options);
    _runnableKernel = _compiledProgram->getKernel(launch.entryPoint);
    _runnableKernel->setBlockSize(launch.blockSize, 1, 1);
    _runnableKernel->setGridSize(launch.gridSize, 1, 1);
}

void LayernormFwdPlan::execute(const Handle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "LayernormFwdPlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto xBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.y()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
        _params.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto biasBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
        _params.bias()->uid(), deviceBuffers, numDeviceBuffers);
    auto meanBuffer = _params.mean() != nullptr
                          ? hipdnn_plugin_sdk::findDeviceBuffer(
                                _params.mean()->uid(), deviceBuffers, numDeviceBuffers)
                          : hipdnnPluginDeviceBuffer_t{-1, nullptr};
    auto invVarianceBuffer = _params.invVariance() != nullptr
                                 ? hipdnn_plugin_sdk::findDeviceBuffer(_params.invVariance()->uid(),
                                                                       deviceBuffers,
                                                                       numDeviceBuffers)
                                 : hipdnnPluginDeviceBuffer_t{-1, nullptr};

    double epsilon = _params.epsilonValue(deviceBuffers, numDeviceBuffers);

    // Launch kernel
    _runnableKernel->launch(handle.getStream(),
                            xBuffer.ptr,
                            yBuffer.ptr,
                            scaleBuffer.ptr,
                            biasBuffer.ptr,
                            meanBuffer.ptr,
                            invVarianceBuffer.ptr,
                            static_cast<float>(epsilon));
}

} // namespace hip_kernel_provider::layernorm
