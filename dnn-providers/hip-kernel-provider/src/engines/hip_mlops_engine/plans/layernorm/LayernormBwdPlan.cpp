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

namespace
{

/// The work-items one parallel pass keeps busy on a device with @p multiProcessorCount CUs.
size_t reqdWorkItemCount(int multiProcessorCount, size_t localSize)
{
    return 4 * localSize * static_cast<size_t>(multiProcessorCount);
}

size_t parallelSize(int multiProcessorCount, size_t localSize, size_t innerSize, size_t outerSize)
{
    const size_t reqd = reqdWorkItemCount(multiProcessorCount, localSize);
    size_t size = 1;
    while(size * innerSize < reqd && static_cast<double>(size) < std::sqrt(outerSize))
    {
        size *= 2;
    }
    return size;
}

ProblemDescription bwdProblem(const LayernormBwdParams& params)
{
    const auto statAttr
        = params.mean() == nullptr ? std::nullopt : std::make_optional(params.mean());
    return {params.x(), params.scale(), statAttr, Direction::BACKWARD};
}

/// The partial sums per column the scale/bias reduction of @p config keeps on a device with
/// @p multiProcessorCount CUs: parallelSize() when it splits across workgroups, else 0.
size_t partialSumCount(const ProblemDescription& problem,
                       int multiProcessorCount,
                       const LayernormBwdConfig& config)
{
    return config.parallel ? parallelSize(multiProcessorCount,
                                          static_cast<size_t>(LayernormBwdParams::MAX_LOCAL_SIZE),
                                          static_cast<size_t>(problem.innerSize()),
                                          static_cast<size_t>(problem.outerSize()))
                           : 0;
}

} // namespace

LayernormBwdConfig defaultLayernormBwdConfig(const LayernormBwdParams& params,
                                             int multiProcessorCount)
{
    const auto problem = bwdProblem(params);
    const auto innerSize = static_cast<size_t>(problem.innerSize());
    const auto outerSize = static_cast<size_t>(problem.outerSize());
    const size_t reqd = reqdWorkItemCount(multiProcessorCount,
                                          static_cast<size_t>(LayernormBwdParams::MAX_LOCAL_SIZE));
    return {innerSize <= reqd && innerSize * outerSize > reqd};
}

std::vector<KernelLaunchShape> LayernormBwdPlan::launchShapes(int64_t outerSize,
                                                              int64_t innerSize,
                                                              int64_t stride,
                                                              int64_t localSize,
                                                              bool isParallel,
                                                              int64_t parallelSize)
{
    std::vector<KernelLaunchShape> shapes{{"LayernormBwd", outerSize * stride, localSize}};
    if(isParallel)
    {
        shapes.push_back({"LayernormBwdScaleBiasParallel",
                          align(parallelSize * innerSize, localSize),
                          localSize});
        shapes.push_back(
            {"LayernormBwdScaleBiasReduceSum", align(innerSize, localSize), localSize});
    }
    else
    {
        shapes.push_back({"LayernormBwdScaleBias", align(innerSize, localSize), localSize});
    }
    return shapes;
}

LayernormKernelBuild buildLayernormBwdKernel(const LayernormBwdParams& params,
                                             const std::string& gcnArchName,
                                             int multiProcessorCount,
                                             const LayernormBwdConfig& config)
{
    // Ensure that the input tensor is either 4D or 5D
    const auto* xDims = params.x()->dims();
    if(xDims->size() != 4 && xDims->size() != 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xDims->size()));
    }

    const auto problem = bwdProblem(params);
    const int64_t outerSize = problem.outerSize();
    const int64_t innerSize = problem.innerSize();
    const int64_t stride = problem.stride();
    const int64_t localSize = LayernormBwdParams::MAX_LOCAL_SIZE;

    // Determine input/output data type configuration
    const auto inputDataType = params.x()->data_type();
    const auto outputDataType = params.dy()->data_type();
    const auto scaleBiasDataType = params.scale()->data_type();
    const auto meanInvVarianceDataType
        = (params.mean() == nullptr) ? inputDataType : params.mean()->data_type();
    const std::string inputTypeString = getKernelParamTypeString(inputDataType);
    const std::string outputTypeString = getKernelParamTypeString(outputDataType);
    const std::string scaleBiasTypeString = getKernelParamTypeString(scaleBiasDataType);
    const std::string meanInvVarianceTypeString = getKernelParamTypeString(meanInvVarianceDataType);

    const bool parallel = config.parallel;
    const size_t partialSums = partialSumCount(problem, multiProcessorCount, config);

    const auto shapes = layernormBwdLaunchShapes(params, multiProcessorCount, config);
    for(const auto& shape : shapes)
    {
        checkLaunchFitsThreadLimit(shape);
    }

    // Prepare compilation options
    KernelCompileOptions options(params.x(), gcnArchName);
    options.add("HIP_PLUGIN_LAYERNORM_OUTER_SIZE", outerSize);
    options.add("HIP_PLUGIN_LAYERNORM_INNER_SIZE", innerSize);
    options.add("HIP_PLUGIN_LAYERNORM_STRIDE", stride);
    options.add("HIP_PLUGIN_LAYERNORM_LOCAL_SIZE", localSize);
    options.add("HIP_PLUGIN_LAYERNORM_PARALLEL_SIZE", partialSums);
    options.add("HIP_PLUGIN_LAYERNORM_INPUT_TYPE", inputTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_OUTPUT_TYPE", outputTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE", scaleBiasTypeString);
    options.add("HIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE", meanInvVarianceTypeString);

    const auto block = static_cast<unsigned int>(localSize);
    std::vector<LayernormLaunch> launches{
        {"LayernormBwd", block, static_cast<unsigned int>(shapes[0].blocks)}};
    if(parallel)
    {
        launches.push_back({"LayernormBwdScaleBiasParallel",
                            block,
                            static_cast<unsigned int>(shapes[1].blocks)});
        launches.push_back({"LayernormBwdScaleBiasReduceSum",
                            block,
                            static_cast<unsigned int>(shapes[2].blocks)});
    }
    else
    {
        launches.push_back({"LayernormBwdScaleBias",
                            block,
                            static_cast<unsigned int>(align(innerSize, localSize))});
    }
    return {std::move(options), std::move(launches)};
}

std::vector<KernelLaunchShape> layernormBwdLaunchShapes(const LayernormBwdParams& params,
                                                        int multiProcessorCount,
                                                        const LayernormBwdConfig& config)
{
    const auto problem = bwdProblem(params);
    return LayernormBwdPlan::launchShapes(
        problem.outerSize(),
        problem.innerSize(),
        problem.stride(),
        LayernormBwdParams::MAX_LOCAL_SIZE,
        config.parallel,
        static_cast<int64_t>(partialSumCount(problem, multiProcessorCount, config)));
}

size_t layernormBwdWorkspaceBytes(const LayernormBwdParams& params,
                                  int multiProcessorCount,
                                  const LayernormBwdConfig& config)
{
    const auto problem = bwdProblem(params);
    const auto outerSize = static_cast<size_t>(problem.outerSize());
    const auto innerSize = static_cast<size_t>(problem.innerSize());
    const auto stride = static_cast<size_t>(problem.stride());
    const auto localSize = static_cast<size_t>(LayernormBwdParams::MAX_LOCAL_SIZE);

    // Workspace only needed for the parallel backward pass or for fallback mean/rstd calculations
    const bool parallel = config.parallel;
    const size_t sizeFromParallel
        = parallel
              ? 2 * innerSize * parallelSize(multiProcessorCount, localSize, innerSize, outerSize)
              : 0;
    const size_t sizeFromFallbackMeanRstd
        = params.mean() == nullptr || params.invVariance() == nullptr ? 2 * outerSize * stride : 0;

    return 4 * (sizeFromParallel + sizeFromFallbackMeanRstd);
}

LayernormBwdPlan::LayernormBwdPlan(LayernormBwdParams&& params)
    : _params(std::move(params))
    , _isParallel(false)
{
}

size_t LayernormBwdPlan::getWorkspaceSize(const Handle& handle) const
{
    const auto deviceProperties
        = hip_kernel_provider_common::getDeviceProperties(handle.getStream());
    return layernormBwdWorkspaceBytes(
        _params,
        deviceProperties.multiProcessorCount,
        defaultLayernormBwdConfig(_params, deviceProperties.multiProcessorCount));
}

void LayernormBwdPlan::compile(const IKernelCompiler& kernelCompiler,
                               const hipDeviceProp_t& deviceProperties)
{
    const auto build = buildLayernormBwdKernel(
        _params,
        std::string(deviceProperties.gcnArchName),
        deviceProperties.multiProcessorCount,
        defaultLayernormBwdConfig(_params, deviceProperties.multiProcessorCount));
    _isParallel = build.launches.size() == 3;
    HIPDNN_PLUGIN_LOG_INFO(
        (_isParallel ? "LayernormBwdPlan: parallel" : "LayernormBwdPlan: not parallel"));

    // Compile kernel and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("LayernormBwd.cpp", build.options);
    for(const auto& launch : build.launches)
    {
        _runnableKernels.push_back(_compiledProgram->getKernel(launch.entryPoint));
        _runnableKernels.back()->setBlockSize(launch.blockSize, 1, 1);
        _runnableKernels.back()->setGridSize(launch.gridSize, 1, 1);
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

} // namespace hip_kernel_provider::layernorm
