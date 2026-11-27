// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "RMSnormFwdKernel.hpp"
#include "../PlanUtils.hpp"
#include "RMSnormCommon.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "core/Utils.hpp"

#include <algorithm>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::rmsnorm
{

RMSnormFwdParams::RMSnormFwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _x(tensorMap.at(attributes.x_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _bias(attributes.bias_tensor_uid().has_value()
                ? tensorMap.at(attributes.bias_tensor_uid().value())
                : nullptr)
    , _y(tensorMap.at(attributes.y_tensor_uid()))
    , _invRMS(attributes.inv_rms_tensor_uid().has_value()
                  ? tensorMap.at(attributes.inv_rms_tensor_uid().value())
                  : nullptr)
    , _epsilon(hipdnn_plugin_sdk::makeScalarOperand(
          tensorMap, attributes.epsilon_tensor_uid(), "Epsilon"))
    , _activationOut(nullptr)
{
}

RMSnormFwdParams::RMSnormFwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& attributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : RMSnormFwdParams(attributes, tensorMap)
{
    // Initialize activation attributes
    _optActivation = parseActivation(pointwiseAttributes);
    _activationOut = tensorMap.at(pointwiseAttributes.out_0_tensor_uid());

    // Validate that activation input matches rmsnorm output
    if(pointwiseAttributes.in_0_tensor_uid() != attributes.y_tensor_uid())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "RMSnormFwdParams: Activation input must match rmsnorm output");
    }
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormFwdParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormFwdParams::scale() const
{
    return _scale;
}

double RMSnormFwdParams::epsilonValue(const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                      uint32_t numDeviceBuffers) const
{
    return hipdnn_plugin_sdk::toDouble(
        hipdnn_plugin_sdk::resolveScalarOperand(_epsilon, deviceBuffers, numDeviceBuffers));
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormFwdParams::bias() const
{
    return _bias;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormFwdParams::y() const
{
    return _y;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormFwdParams::invRMS() const
{
    return _invRMS;
}

const std::optional<ActivationParams>& RMSnormFwdParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*
    RMSnormFwdParams::activationOut() const
{
    return _activationOut;
}

std::vector<KernelLaunchShape>
    rmsnormFwdLaunchShapes(int64_t outerSize, int64_t stride, int64_t localSize)
{
    return {{"RMSnormFwd", outerSize * stride, localSize}};
}

bool isRMSnormFwdConfigApplicable(const RMSnormFwdParams& params, const RMSnormConfig& config)
{
    if(!isLocalSizeSupported(config.localSize))
    {
        return false;
    }
    const ProblemDescription problem(params.x(), params.scale(), Direction::FORWARD);
    const auto shapes
        = rmsnormFwdLaunchShapes(problem.outerSize(), problem.stride(), config.localSize);
    return std::all_of(shapes.begin(), shapes.end(), [](const KernelLaunchShape& shape) {
        return launchFitsThreadLimit(shape.blocks, shape.threadsPerBlock);
    });
}

RMSnormKernelBuild buildRMSnormFwdKernel(const RMSnormFwdParams& params,
                                         const std::string& gcnArchName,
                                         const RMSnormConfig& config)
{
    // Extract dimensions from x tensor
    const auto* xDims = params.x()->dims();
    if(const auto xRank = xDims->size(); xRank < 4 || xRank > 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xRank));
    }

    const ProblemDescription problem(params.x(), params.scale(), Direction::FORWARD);
    const int64_t stride = problem.stride();
    const int64_t outerSize = problem.outerSize();
    const int64_t innerSize = problem.innerSize();
    const auto shapes = rmsnormFwdLaunchShapes(outerSize, stride, config.localSize);
    for(const auto& shape : shapes)
    {
        checkLaunchFitsThreadLimit(shape);
    }

    // Calculate block and grid dimensions
    const auto xlocalsize = static_cast<unsigned int>(shapes[0].threadsPerBlock);
    const auto xgridsize = static_cast<unsigned int>(shapes[0].blocks);

    // Determine input/output data type configuration
    const auto inputDataType = params.x()->data_type();
    auto outputDataType = params.y()->data_type();
    const auto scaleDataType = params.scale()->data_type();
    const auto computeDataType = (params.invRMS() == nullptr)
                                     ? hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT
                                     : params.invRMS()->data_type();

    // Get activation mode
    auto activationMode = ActivationMode::PASTHRU;
    if(params.optActivation().has_value() && params.activationOut() != nullptr)
    {
        activationMode = params.optActivation()->mode;
        outputDataType = params.activationOut()->data_type();
    }

    const std::string inputTypeString = getKernelParamTypeString(inputDataType);
    const std::string outputTypeString = getKernelParamTypeString(outputDataType);
    const std::string scaleTypeString = getKernelParamTypeString(scaleDataType);
    const std::string computeTypeString = getKernelParamTypeString(computeDataType);

    // Prepare compilation options
    KernelCompileOptions options(params.x(), gcnArchName);
    options.add("HIP_PLUGIN_RMSNORM_INNER_SIZE", innerSize);
    options.add("HIP_PLUGIN_RMSNORM_STRIDE", stride);
    options.add("HIP_PLUGIN_RMSNORM_INPUT_TYPE", inputTypeString);
    options.add("HIP_PLUGIN_RMSNORM_OUTPUT_TYPE", outputTypeString);
    options.add("HIP_PLUGIN_RMSNORM_SCALE_TYPE", scaleTypeString);
    options.add("HIP_PLUGIN_RMSNORM_COMPUTE_TYPE", computeTypeString);
    options.add("HIP_PLUGIN_RMSNORM_LOCAL_SIZE", xlocalsize);
    options.add("HIP_PLUGIN_RMSNORM_NRN_OP_ID", static_cast<int>(activationMode));

    std::vector<RMSnormLaunch> launches{{"RMSnormFwd", xlocalsize, xgridsize}};
    return {std::move(options), std::move(launches)};
}

} // namespace hip_kernel_provider::rmsnorm
