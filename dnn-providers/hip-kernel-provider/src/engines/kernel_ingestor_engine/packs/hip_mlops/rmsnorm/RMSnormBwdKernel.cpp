// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "RMSnormBwdKernel.hpp"
#include "../PlanUtils.hpp"
#include "RMSnormCommon.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "core/Utils.hpp"

#include <algorithm>
#include <cstdint>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>

using namespace hip_kernel_provider::core::utils;

namespace hip_kernel_provider::rmsnorm
{

RMSnormBwdParams::RMSnormBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& attributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _dy(tensorMap.at(attributes.dy_tensor_uid()))
    , _x(tensorMap.at(attributes.x_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _invRMS(tensorMap.at(attributes.inv_rms_tensor_uid()))
    , _dx(tensorMap.at(attributes.dx_tensor_uid()))
    , _dscale(tensorMap.at(attributes.dscale_tensor_uid()))
    , _dbias(attributes.dbias_tensor_uid().has_value()
                 ? tensorMap.at(attributes.dbias_tensor_uid().value())
                 : nullptr)
{
}

RMSnormBwdParams::RMSnormBwdParams(
    const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& attributes,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& pointwiseAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
    : _dy(tensorMap.at(pointwiseAttributes.in_0_tensor_uid()))
    , _x(tensorMap.at(attributes.x_tensor_uid()))
    , _scale(tensorMap.at(attributes.scale_tensor_uid()))
    , _invRMS(tensorMap.at(attributes.inv_rms_tensor_uid()))
    , _dx(tensorMap.at(attributes.dx_tensor_uid()))
    , _dscale(tensorMap.at(attributes.dscale_tensor_uid()))
    , _dbias(attributes.dbias_tensor_uid().has_value()
                 ? tensorMap.at(attributes.dbias_tensor_uid().value())
                 : nullptr)
    , _optActivation(parseActivation(pointwiseAttributes))
    // The passthrough gradient ignores the activation input; reuse x for unary IDENTITY.
    , _y(pointwiseAttributes.operation()
                 == hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY
             ? _x
             : tensorMap.at(pointwiseAttributes.in_1_tensor_uid().value()))
{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::dy() const
{
    return _dy;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::invRMS() const
{
    return _invRMS;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::dx() const
{
    return _dx;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::dscale() const
{
    return _dscale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::dbias() const
{
    return _dbias;
}

const std::optional<ActivationParams>& RMSnormBwdParams::optActivation() const
{
    return _optActivation;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormBwdParams::y() const
{
    return _y;
}

std::vector<KernelLaunchShape>
    rmsnormBwdLaunchShapes(int64_t outerSize, int64_t stride, int64_t innerSize, int64_t localSize)
{
    return {{"RMSnormBwdData", outerSize * stride, localSize},
            {"RMSnormBwdScaleBias", (innerSize + localSize - 1) / localSize, localSize}};
}

bool isRMSnormBwdConfigApplicable(const RMSnormBwdParams& params, const RMSnormConfig& config)
{
    if(!isLocalSizeSupported(config.localSize))
    {
        return false;
    }
    const ProblemDescription problem(params.x(), params.scale(), Direction::BACKWARD);
    const auto shapes = rmsnormBwdLaunchShapes(
        problem.outerSize(), problem.stride(), problem.innerSize(), config.localSize);
    return std::all_of(shapes.begin(), shapes.end(), [](const KernelLaunchShape& shape) {
        return launchFitsThreadLimit(shape.blocks, shape.threadsPerBlock);
    });
}

RMSnormKernelBuild buildRMSnormBwdKernel(const RMSnormBwdParams& params,
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

    // Get problem dimensions
    const ProblemDescription problem(params.x(), params.scale(), Direction::BACKWARD);
    const int64_t stride = problem.stride();
    const int64_t outerSize = problem.outerSize();
    const int64_t innerSize = problem.innerSize();
    const auto shapes = rmsnormBwdLaunchShapes(outerSize, stride, innerSize, config.localSize);
    for(const auto& shape : shapes)
    {
        checkLaunchFitsThreadLimit(shape);
    }

    // Get activation mode
    auto activationMode = ActivationMode::PASTHRU;
    if(const auto& activation = params.optActivation(); activation.has_value())
    {
        activationMode = activation->mode;
    }

    // Determine input/output data type configuration
    const auto xDataType = params.x()->data_type();
    const auto dYDataType = params.dy()->data_type();
    const auto dXDataType = params.dx()->data_type();
    const auto scaleDataType = params.scale()->data_type();
    const auto computeDataType = params.invRMS()->data_type();
    const auto yDataType = params.y() == nullptr ? dYDataType : params.y()->data_type();
    const std::string xTypeString = getKernelParamTypeString(xDataType);
    const std::string dYTypeString = getKernelParamTypeString(dYDataType);
    const std::string dXTypeString = getKernelParamTypeString(dXDataType);
    const std::string scaleTypeString = getKernelParamTypeString(scaleDataType);
    const std::string computeTypeString = getKernelParamTypeString(computeDataType);
    const std::string yTypeString = getKernelParamTypeString(yDataType);

    // Calculate block and grid dimensions
    const auto xlocalsize = static_cast<unsigned int>(shapes[0].threadsPerBlock);
    const auto xgridsizeBwdData = static_cast<unsigned int>(shapes[0].blocks);
    const auto xgridsizeBwdScaleBias = static_cast<unsigned int>(shapes[1].blocks);

    // Prepare compilation options
    KernelCompileOptions options(params.x(), gcnArchName);
    options.add("HIP_PLUGIN_RMSNORM_LOCAL_SIZE", xlocalsize);
    options.add("HIP_PLUGIN_RMSNORM_INNER_SIZE", innerSize);
    options.add("HIP_PLUGIN_RMSNORM_OUTER_SIZE", outerSize);
    options.add("HIP_PLUGIN_RMSNORM_STRIDE", stride);
    options.add("HIP_PLUGIN_RMSNORM_X_TYPE", xTypeString);
    options.add("HIP_PLUGIN_RMSNORM_DY_TYPE", dYTypeString);
    options.add("HIP_PLUGIN_RMSNORM_DX_TYPE", dXTypeString);
    options.add("HIP_PLUGIN_RMSNORM_SCALE_TYPE", scaleTypeString);
    options.add("HIP_PLUGIN_RMSNORM_COMPUTE_TYPE", computeTypeString);
    options.add("HIP_PLUGIN_RMSNORM_Y_TYPE", yTypeString);
    options.add("HIP_PLUGIN_RMSNORM_NRN_OP_ID", static_cast<int>(activationMode));

    std::vector<RMSnormLaunch> launches{{"RMSnormBwdData", xlocalsize, xgridsizeBwdData},
                                        {"RMSnormBwdScaleBias", xlocalsize, xgridsizeBwdScaleBias}};
    return {std::move(options), std::move(launches)};
}

} // namespace hip_kernel_provider::rmsnorm
