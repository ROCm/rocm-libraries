// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltUtils.hpp"

namespace hipblaslt_plugin::hipblaslt_utils
{

EpilogueParams mapPointwiseModeToHipblasLtEpilogue(
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* attrs, bool withBias)
{
    if(!attrs)
    {
        return EpilogueParams{
            withBias ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT, 0.0, 0.0};
    }

    using PM = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    switch(attrs->operation())
    {
    case PM::RELU_FWD:
    {
        if(attrs->relu_lower_clip() && attrs->relu_upper_clip())
        {
            if(attrs->relu_lower_clip_slope().has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "Incorrect configuration of Clamp (Relu with lower and upper clips). "
                    "Both lower and upper clips are set, but relu_lower_clip_slope is also set.");
            }

            // CLAMP
            // act(x) = max(\alpha, min(\beta, x))
            return EpilogueParams{withBias ? HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT
                                           : HIPBLASLT_EPILOGUE_CLAMP_EXT,
                                  static_cast<float>(*attrs->relu_lower_clip()),
                                  static_cast<float>(*attrs->relu_upper_clip())};
        }
        if(attrs->relu_lower_clip().has_value() && attrs->relu_lower_clip().value() == 0.f
           && !attrs->relu_upper_clip().has_value())
        {
            // Standard ReLU
            return EpilogueParams{
                withBias ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_RELU, 0.0, 0.0};
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Supports only clamp and standard relu with zero min value");
    }
    case PM::GELU_APPROX_TANH_FWD:
        return EpilogueParams{
            withBias ? HIPBLASLT_EPILOGUE_GELU_BIAS : HIPBLASLT_EPILOGUE_GELU, 0.0, 0.0};
    case PM::SWISH_FWD:
        if(attrs->swish_beta().has_value() && attrs->swish_beta().value() == 1.0f)
        {
            return EpilogueParams{withBias ? HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT
                                           : HIPBLASLT_EPILOGUE_SWISH_EXT,
                                  0.0,
                                  0.0};
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Supports only swish with beta = 1.0");
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported activation operation");
    }
}

hipDataType
    tensorDataTypeToHipDataType(const hipdnn_flatbuffers_sdk::data_objects::DataType& dataType)
{
    switch(dataType)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return HIP_R_32F;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT32:
        return HIP_R_32I;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
        return HIP_R_16F;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return HIP_R_16BF;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT8:
        return HIP_R_8I;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported data type for hipBLASLt: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::toString(dataType)));
    }
}

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

hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid)
{
    if(auto tensorAttr = tensorMap.find(uid); tensorAttr != tensorMap.end())
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::TensorAttributesWrapper(
            tensorAttr->second);
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Failed to find tensor with UID in tensorMap: "
                                                       + std::to_string(uid));
}

} // namespace hipblaslt_plugin::hipblaslt_utils
