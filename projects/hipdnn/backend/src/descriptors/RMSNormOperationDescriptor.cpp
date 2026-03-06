// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RMSNormOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"

namespace hipdnn_backend
{

void RMSNormOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_scaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormOperationDescriptor::finalize() failed: SCALE tensor not set");
    THROW_IF_NULL(_epsilonDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormOperationDescriptor::finalize() failed: EPSILON tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormOperationDescriptor::finalize() failed: Y tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormOperationDescriptor::finalize() failed: compute data type not set");
    THROW_IF_TRUE(_data.forward_phase == hipdnn_data_sdk::data_objects::NormFwdPhase::NOT_SET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormOperationDescriptor::finalize() failed: forward_phase not set");

    // Bias and inv_rms are optional — not required for finalization.

    HipdnnBackendDescriptorImpl<RMSNormOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void RMSNormOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                              hipdnnBackendAttributeType_t attributeType,
                                              int64_t elementCount,
                                              const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "RMSNormOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT:
        setTensorDescriptor(_epsilonDesc,
                            _data.epsilon_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT:
        setOptionalTensorDescriptor(_biasDesc,
                                    _data.bias_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "RMSNormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT:
        setOptionalTensorDescriptor(_invRmsDesc,
                                    _data.inv_rms_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "RMSNormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT:
        setNormFwdPhase(_data.forward_phase,
                        attributeType,
                        elementCount,
                        arrayOfElements,
                        "RMSNormOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RMSNORM_MATH_PREC_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "RMSNormOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "RMSNormOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void RMSNormOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                              hipdnnBackendAttributeType_t attributeType,
                                              int64_t requestedElementCount,
                                              int64_t* elementCount,
                                              void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "RMSNormOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT:
        getTensorDescriptor(_epsilonDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT:
        getOptionalTensorDescriptor(_biasDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "RMSNormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT:
        getOptionalTensorDescriptor(_invRmsDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "RMSNormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT:
        getNormFwdPhase(_data.forward_phase,
                        attributeType,
                        requestedElementCount,
                        elementCount,
                        arrayOfElements,
                        "RMSNormOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RMSNORM_MATH_PREC_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "RMSNormOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "RMSNormOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    RMSNormOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.push_back(_xDesc);
    result.push_back(_scaleDesc);
    result.push_back(_epsilonDesc);
    result.push_back(_yDesc);
    if(_biasDesc)
    {
        result.push_back(_biasDesc);
    }
    if(_invRmsDesc)
    {
        result.push_back(_invRmsDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT> RMSNormOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::RMSNormAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t RMSNormOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT;
}

std::string RMSNormOperationDescriptor::toString() const
{
    std::string str = "RMSNormOperationDescriptor: {";
    str += "x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", epsilon_uid=" + std::to_string(_data.epsilon_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    if(_data.bias_tensor_uid.has_value())
    {
        str += ", bias_uid=" + std::to_string(*_data.bias_tensor_uid);
    }
    if(_data.inv_rms_tensor_uid.has_value())
    {
        str += ", inv_rms_uid=" + std::to_string(*_data.inv_rms_tensor_uid);
    }
    str += ", forward_phase=" + std::to_string(static_cast<int>(_data.forward_phase));
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
