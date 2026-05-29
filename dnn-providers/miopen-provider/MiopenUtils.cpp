// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenUtils.hpp"

namespace miopen_plugin::miopen_utils
{

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers)
{
    for(uint32_t i = 0; i < numDeviceBuffers; i++)
    {
        if(uid == deviceBuffers[i].uid)
        {
            return deviceBuffers[i];
        }
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        "Device buffer with the uid: " + std::to_string(uid)
            + " not found in the provided device buffers.");
}

miopenDataType_t
    tensorDataTypeToMiopenDataType(const hipdnn_flatbuffers_sdk::data_objects::DataType& dataType)
{
    switch(dataType)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return miopenFloat;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
        return miopenHalf;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return miopenBFloat16;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported data type for MIOpen: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::toString(dataType)));
    }
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid)
{
    if(auto tensorAttr = tensorMap.find(uid); tensorAttr != tensorMap.end())
    {
        return *tensorAttr->second;
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Failed to find tensor with UID in tensorMap: "
                                                       + std::to_string(uid));
}

MiopenTensor createTensor(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid)
{
    const auto& tensorAttr = findTensorAttributes(tensorMap, uid);
    return {tensorAttr};
}

MiopenTensor createBatchnormTensor(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid)
{
    const auto& tensorAttr = findTensorAttributes(tensorMap, uid);

    if(tensorAttr.dims() == nullptr || tensorAttr.strides() == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Tensor dims or strides are null for UID: "
                                                           + std::to_string(uid));
    }

    if(tensorAttr.dims()->size() != 3)
    {
        return {tensorAttr};
    }

    if(tensorAttr.dims()->size() != tensorAttr.strides()->size())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Tensor dims and strides size mismatch for UID: " + std::to_string(uid));
    }

    std::vector<int64_t> dims(tensorAttr.dims()->begin(), tensorAttr.dims()->end());
    std::vector<int64_t> strides(tensorAttr.strides()->begin(), tensorAttr.strides()->end());

    // MIOpen requires at least 4D tensors for batchnorm.
    // Pad 3D to 4D by appending W=1 dimension.
    // Stride for W: 1 for channels-first (NCL→NCHW), C for channels-last (NLC→NHWC).
    dims.push_back(1);

    constexpr size_t C_IDX = 1;
    constexpr size_t L_IDX = 2;
    const bool isChannelsLast = strides[C_IDX] < strides[L_IDX];
    strides.push_back(isChannelsLast ? dims[C_IDX] : 1);

    return {uid, tensorAttr.data_type(), dims, strides};
}

size_t getSpatialDimCount(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attr)
{
    if(attr.dims()->size() < 3)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Tensor must have at least 3 dimensions, but got: "
                + std::to_string(attr.dims()->size()));
    }

    return attr.dims()->size() - 2;
}

ActivationParams mapPointwiseModeToMiopenActivation(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs)
{
    using PM = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    switch(attrs.operation())
    {
    case PM::RELU_FWD:
    case PM::RELU_BWD:
    {
        auto lowerClip = attrs.relu_lower_clip();
        auto upperClip = attrs.relu_upper_clip();
        auto lowerClipSlope = attrs.relu_lower_clip_slope();

        if(lowerClip && upperClip)
        {
            // CLAMP
            // act(x) = max(\alpha, min(\beta, x))
            return ActivationParams{miopenActivationCLAMP,
                                    static_cast<double>(*lowerClip),
                                    static_cast<double>(*upperClip),
                                    0.0};
        }
        if(upperClip)
        {
            // Clipped ReLU
            // act(x) = max(0, min(\alpha, x))
            return ActivationParams{
                miopenActivationCLIPPEDRELU, static_cast<double>(*upperClip), 0.0, 0.0};
        }
        if(lowerClipSlope)
        {
            // Leaky ReLU
            return ActivationParams{
                miopenActivationLEAKYRELU, static_cast<double>(*lowerClipSlope), 0.0, 0.0};
        }
        if(lowerClip && *lowerClip != 0.f)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Standard relu with a non-zero lower_clip is not supported");
        }

        // Standard ReLU
        return ActivationParams{miopenActivationRELU, 0.0, 0.0, 0.0};
    }
    case PM::SIGMOID_FWD:
    case PM::SIGMOID_BWD:
        return ActivationParams{miopenActivationLOGISTIC, 0.0, 0.0, 0.0};
    case PM::TANH_FWD:
    case PM::TANH_BWD:
        return ActivationParams{miopenActivationTANH, 1.0, 1.0, 0.0};
    case PM::ELU_FWD:
    case PM::ELU_BWD:
    {
        const double alpha = attrs.elu_alpha() ? static_cast<double>(*attrs.elu_alpha()) : 1.0;
        return ActivationParams{miopenActivationELU, alpha, 0.0, 0.0};
    }
    case PM::SOFTPLUS_FWD:
    case PM::SOFTPLUS_BWD:
        // Softplus is (1/beta) * log(1 + e^(beta*x))
        // However, MIOpen uses:
        // log(1 + e^x)
        // This is valid Softplus only when beta=1
        if(attrs.softplus_beta())
        {
            // Only support beta=1
            if(static_cast<double>(*attrs.softplus_beta()) != 1.0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                               "Softplus only supports beta = 1.0");
            }
        }
        return ActivationParams{miopenActivationSOFTRELU, 0.0, 0.0, 0.0};
    case PM::ABS:
        return ActivationParams{miopenActivationABS, 0.0, 0.0, 0.0};
    case PM::IDENTITY:
        return ActivationParams{miopenActivationPASTHRU, 0.0, 0.0, 0.0};
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported activation operation");
    }
}

std::string getDeviceArch(hipStream_t stream)
{
    hipDevice_t deviceId = -1;
    auto status = hipStreamGetDevice(stream, &deviceId);
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "hipStreamGetDevice failed: "
                                                           + std::to_string(status));
    }
    hipDeviceProp_t props;
    status = hipGetDeviceProperties(&props, deviceId);
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "hipGetDeviceProperties failed: "
                                                           + std::to_string(status));
    }
    const std::string archStr(props.gcnArchName);
    return archStr.substr(0, archStr.find(':'));
}

}
