// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "RMSnormBwdPlan.hpp"
#include "../PlanUtils.hpp"
#include "RMSnormCommon.hpp"
#include "hip/HipKernelCompileOptions.hpp"

#include "HipKernelUtils.hpp"
#include "hip/IKernelCompiler.hpp"

#include <cstdint>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>

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

RMSnormBwdPlan::RMSnormBwdPlan(RMSnormBwdParams&& params)
    : _params(std::move(params))
{
}

size_t RMSnormBwdPlan::getWorkspaceSize([[maybe_unused]] const HipKernelHandle& handle) const
{
    // No workspace needed for RMS norm
    return 0;
}

void RMSnormBwdPlan::compile([[maybe_unused]] const IKernelCompiler& kernelCompiler,
                             [[maybe_unused]] const hipDeviceProp_t& deviceProperties)
{
    // Extract dimensions from x tensor
    const auto* xDims = _params.x()->dims();
    if(const auto xRank = xDims->size(); xRank < 4 || xRank > 5)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported tensor dimension: "
                                                           + std::to_string(xRank));
    }

    // Get problem dimensions
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

    // Determine input/output data type configuration
    const auto xDataType = _params.x()->data_type();
    const auto dYDataType = _params.dy()->data_type();
    const auto dXDataType = _params.dx()->data_type();
    const auto scaleDataType = _params.scale()->data_type();
    const auto computeDataType = _params.invRMS()->data_type();
    const std::string xTypeString = getKernelParamTypeString(xDataType);
    const std::string dYTypeString = getKernelParamTypeString(dYDataType);
    const std::string dXTypeString = getKernelParamTypeString(dXDataType);
    const std::string scaleTypeString = getKernelParamTypeString(scaleDataType);
    const std::string computeTypeString = getKernelParamTypeString(computeDataType);

    // Calculate block and grid dimensions
    const unsigned int xlocalsize = 256;
    const auto xgridsizeBwdData = static_cast<unsigned int>(outerSize * stride);
    const auto xgridsizeBwdWeightBias
        = static_cast<unsigned int>((innerSize + xlocalsize - 1) / xlocalsize);
    const unsigned int ylocalsize = 1;
    const unsigned int ygridsize = 1;
    const unsigned int zlocalsize = 1;
    const unsigned int zgridsize = 1;

    // Prepare compilation options
    HipKernelCompileOptions options(_params.x(), deviceProperties);
    options.add("HIP_PLUGIN_RMSNORM_LOCAL_SIZE", xlocalsize);
    options.add("HIP_PLUGIN_RMSNORM_INNER_SIZE", innerSize);
    options.add("HIP_PLUGIN_RMSNORM_OUTER_SIZE", outerSize);
    options.add("HIP_PLUGIN_RMSNORM_STRIDE", stride);
    options.add("HIP_PLUGIN_RMSNORM_X_TYPE", xTypeString);
    options.add("HIP_PLUGIN_RMSNORM_DY_TYPE", dYTypeString);
    options.add("HIP_PLUGIN_RMSNORM_DX_TYPE", dXTypeString);
    options.add("HIP_PLUGIN_RMSNORM_SCALE_TYPE", scaleTypeString);
    options.add("HIP_PLUGIN_RMSNORM_COMPUTE_TYPE", computeTypeString);

    // Compile kernels and configure launch dimensions
    _compiledProgram = kernelCompiler.compile("RMSNormBwd.cpp", options);
    _runnableKernels.push_back(_compiledProgram->getKernel("RMSnormBwdData"));
    _runnableKernels.push_back(_compiledProgram->getKernel("RMSnormBwdWeightBias"));
    for(auto& kernel : _runnableKernels)
    {
        kernel->setBlockSize(xlocalsize, ylocalsize, zlocalsize);
    }
    _runnableKernels[0]->setGridSize(xgridsizeBwdData, ygridsize, zgridsize);
    _runnableKernels[1]->setGridSize(xgridsizeBwdWeightBias, ygridsize, zgridsize);
}

void RMSnormBwdPlan::execute([[maybe_unused]] const HipKernelHandle& handle,
                             [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                             [[maybe_unused]] uint32_t numDeviceBuffers,
                             [[maybe_unused]] void* workspace) const
{
    if(_runnableKernels.empty())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "RMSnormBwdPlan::execute() called before compile()");
    }

    // Get device buffer pointers
    auto xBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.x()->uid(), deviceBuffers, numDeviceBuffers);
    auto scaleBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.scale()->uid(), deviceBuffers, numDeviceBuffers);
    auto dYBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.dy()->uid(), deviceBuffers, numDeviceBuffers);
    auto invRMSBuffer = hip_kernel_utils::findDeviceBuffer(
        _params.invRMS()->uid(), deviceBuffers, numDeviceBuffers);
    auto dXBuffer
        = hip_kernel_utils::findDeviceBuffer(_params.dx()->uid(), deviceBuffers, numDeviceBuffers);
    auto dScaleBufferPtr = hip_kernel_utils::findDeviceBuffer(
        _params.dscale()->uid(), deviceBuffers, numDeviceBuffers);
    void* dBiasBufferPtr = (_params.dbias() == nullptr)
                               ? nullptr
                               : hip_kernel_utils::findDeviceBuffer(
                                     _params.dbias()->uid(), deviceBuffers, numDeviceBuffers)
                                     .ptr;

    // Run the BwdData kernel
    _runnableKernels[0]->launch(handle.getStream(),
                                dYBuffer.ptr,
                                xBuffer.ptr,
                                scaleBuffer.ptr,
                                invRMSBuffer.ptr,
                                dXBuffer.ptr);

    // Run the BwdWeightBias kernel
    _runnableKernels[1]->launch(handle.getStream(),
                                dYBuffer.ptr,
                                xBuffer.ptr,
                                invRMSBuffer.ptr,
                                dScaleBufferPtr.ptr,
                                dBiasBufferPtr);
}

} // namespace hip_kernel_provider::rmsnorm
