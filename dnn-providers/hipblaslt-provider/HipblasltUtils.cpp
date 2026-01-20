// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltUtils.hpp"

namespace hipblaslt_plugin::hipblaslt_utils
{

hipDataType tensorDataTypeToHipDataType(const hipdnn_data_sdk::data_objects::DataType& dataType)
{
    switch(dataType)
    {
    case hipdnn_data_sdk::data_objects::DataType::FLOAT:
        return HIP_R_32F;
    case hipdnn_data_sdk::data_objects::DataType::INT32:
        return HIP_R_32I;
    case hipdnn_data_sdk::data_objects::DataType::HALF:
        return HIP_R_16F;
    case hipdnn_data_sdk::data_objects::DataType::BFLOAT16:
        return HIP_R_16BF;
    case hipdnn_data_sdk::data_objects::DataType::INT8:
        return HIP_R_8I;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported data type for hipBLASLt: "
                + std::string(hipdnn_data_sdk::data_objects::toString(dataType)));
    }
}

} // namespace hipblaslt_plugin::hipblaslt_utils
