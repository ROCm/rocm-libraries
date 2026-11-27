// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormBwdKernel.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/PlanUtils.hpp"

#include "core/Utils.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormUtilities.hpp"

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

#include <cmath>
#include <cstddef>
#include <string>

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

std::vector<KernelLaunchShape> layernormBwdLaunchShapes(int64_t outerSize,
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
    const int64_t localSize = config.localSize;

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
        launches.push_back(
            {"LayernormBwdScaleBiasParallel", block, static_cast<unsigned int>(shapes[1].blocks)});
        launches.push_back(
            {"LayernormBwdScaleBiasReduceSum", block, static_cast<unsigned int>(shapes[2].blocks)});
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
    return layernormBwdLaunchShapes(
        problem.outerSize(),
        problem.innerSize(),
        problem.stride(),
        config.localSize,
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

} // namespace hip_kernel_provider::layernorm
