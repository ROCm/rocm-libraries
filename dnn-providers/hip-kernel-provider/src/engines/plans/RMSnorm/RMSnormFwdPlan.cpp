// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "RMSnormFwdPlan.hpp"
#include "../PlanUtils.hpp"
#include "RMSnormCommon.hpp"
#include "hip/HipKernelCompileOptions.hpp"

#include "HipKernelUtils.hpp"
#include "hip/IKernelCompiler.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>

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
    , _epsilon(tensorMap.at(attributes.epsilon_tensor_uid()))

{
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormFwdParams::x() const
{
    return _x;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormFwdParams::scale() const
{
    return _scale;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* RMSnormFwdParams::epsilon() const
{
    return _epsilon;
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

RMSnormFwdPlan::RMSnormFwdPlan(RMSnormFwdParams&& params)
    : _params(std::move(params))
{
}

size_t RMSnormFwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for RMS norm
    return 0;
}

void RMSnormFwdPlan::compile(const IKernelCompiler& kernelCompiler,
                             const hipDeviceProp_t& deviceProperties)
{
    // Extract dimensions from x tensor
    const auto* xDims = _params.x()->dims();
    if(const auto xRank = xDims->size(); xRank < 4 || xRank > 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xRank));
    }

    // Infer the outer and inner normalization sizes.
    // 1) Work out the normalization dimension, as the index of the first dimension in
    //    scale tensor that is not 1.
    // 2) Work out the outerSize as the size of input dimensions for which scale is 1.
    //    We will have a work-group for each of these dimensions.
    //    When stride is not 1, we are in a channel-last layout and we ignore the
    //    channel dimension when calculating the outer size.
    // 3) Work out the innerSize as the size of the intput dimensions for which scale nor 1.
    //
    // For an input of [N, C, H, W] with scale [1, C, H, W] this will give: a normalization
    // dimension of 1, outerSize of N, and innerSize of CxHxW.
    // The kernel will therefore consist of N workgroups with each workgroup normalizing over
    // CxHxW elements using a fixed number of threads.
    // For an input of [N, H, W, C] with scale [1, H, W, 1] this will give: a normalization
    // dimension of 2, outerSize of N, stride of C, and innerSize of HxW.
    // The kernel will therefore consist of NxC workgroups with each workgroup normalizing
    // over HxW elements using a fixed number of threads.
    const unsigned normalizeDim = getNormalizeDim(_params.x()->dims(), _params.scale()->dims());
    const int64_t stride = getStride(_params.x(), normalizeDim);
    const int64_t outerSize = getOuterSize(_params.x()->dims(), normalizeDim, stride);
    const int64_t innerSize = getInnerSize(_params.x()->dims(), normalizeDim);
    if(outerSize * stride >= UINT32_MAX)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported number of workgroups: "
                                                           + std::to_string(outerSize * stride));
    }

    // Calculate block and grid dimensions
    const unsigned int xlocalsize = 256;
    const auto xgridsize = static_cast<unsigned int>(outerSize * stride);
    const unsigned int ylocalsize = 1;
    const unsigned int ygridsize = 1;
    const unsigned int zlocalsize = 1;
    const unsigned int zgridsize = 1;

    // Determine input/output data type configuration
    const auto inputDataType = _params.x()->data_type();
    const auto outputDataType = _params.y()->data_type();
    const auto scaleDataType = _params.scale()->data_type();
    const auto computeDataType = (_params.invRMS() == nullptr)
                                     ? hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT
                                     : _params.invRMS()->data_type();
    const std::string inputTypeString = getKernelParamTypeString(inputDataType);
    const std::string outputTypeString = getKernelParamTypeString(outputDataType);
    const std::string scaleTypeString = getKernelParamTypeString(scaleDataType);
    const std::string computeTypeString = getKernelParamTypeString(computeDataType);

    // Prepare compilation options
    HipKernelCompileOptions options(_params.x(), deviceProperties);
    options.add("HIP_PLUGIN_RMSNORM_INNER_SIZE", innerSize);
    options.add("HIP_PLUGIN_RMSNORM_STRIDE", stride);
    options.add("HIP_PLUGIN_RMSNORM_INPUT_TYPE", inputTypeString);
    options.add("HIP_PLUGIN_RMSNORM_OUTPUT_TYPE", outputTypeString);
    options.add("HIP_PLUGIN_RMSNORM_SCALE_TYPE", scaleTypeString);
    options.add("HIP_PLUGIN_RMSNORM_COMPUTE_TYPE", computeTypeString);
    options.add("HIP_PLUGIN_RMSNORM_LOCAL_SIZE", xlocalsize);

    // Compile kernel and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("RMSNormFwd.cpp", options);
    _runnableKernel = _compiledProgram->getKernel("RMSnormFwd");

    _runnableKernel->setBlockSize(xlocalsize, ylocalsize, zlocalsize);
    _runnableKernel->setGridSize(xgridsize, ygridsize, zgridsize);
}

void RMSnormFwdPlan::execute(const HipKernelHandle& handle,
                             const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                             uint32_t numDeviceBuffers,
                             [[maybe_unused]] void* workspace) const
{
    if(!_runnableKernel)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "RMSnormFwdPlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto xBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.y()->uid(), deviceBuffers, numDeviceBuffers);

    void* biasBufferPtr = (_params.bias() == nullptr)
                              ? nullptr
                              : hip_kernel_utils::findDeviceBuffer(
                                    _params.bias()->uid(), deviceBuffers, numDeviceBuffers)
                                    .ptr;
    void* invRMSBufferPtr = (_params.invRMS() == nullptr)
                                ? nullptr
                                : hip_kernel_utils::findDeviceBuffer(
                                      _params.invRMS()->uid(), deviceBuffers, numDeviceBuffers)
                                      .ptr;

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT epsilonTensor;
    _params.epsilon()->UnPackTo(&epsilonTensor);
    double epsilon
        = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(epsilonTensor, "Epsilon");

    _runnableKernel->launch(handle.getStream(),
                            xBuffer.ptr,
                            scaleBuffer.ptr,
                            biasBufferPtr,
                            yBuffer.ptr,
                            invRMSBufferPtr,
                            static_cast<float>(epsilon));
}

}
