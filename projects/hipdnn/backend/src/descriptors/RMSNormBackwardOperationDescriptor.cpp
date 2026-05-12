// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RMSNormBackwardOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void RMSNormBackwardOperationDescriptor::finalize()
{
    THROW_IF_NULL(_dyDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormBackwardOperationDescriptor::finalize() failed: DY_EXT tensor not set");
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormBackwardOperationDescriptor::finalize() failed: X_EXT tensor not set");
    THROW_IF_NULL(
        _scaleDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "RMSNormBackwardOperationDescriptor::finalize() failed: SCALE_EXT tensor not set");
    THROW_IF_NULL(
        _invRmsDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "RMSNormBackwardOperationDescriptor::finalize() failed: INV_RMS_EXT tensor not set");
    THROW_IF_NULL(_dxDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormBackwardOperationDescriptor::finalize() failed: DX_EXT tensor not set");
    THROW_IF_NULL(
        _dscaleDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "RMSNormBackwardOperationDescriptor::finalize() failed: DSCALE_EXT tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "RMSNormBackwardOperationDescriptor::finalize() failed: compute data type not "
                  "set");

    HipdnnBackendDescriptorImpl<RMSNormBackwardOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void RMSNormBackwardOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                      hipdnnBackendAttributeType_t attributeType,
                                                      int64_t elementCount,
                                                      const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "RMSNormBackwardOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT:
        setTensorDescriptor(_invRmsDesc,
                            _data.inv_rms_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT:
        setTensorDescriptor(_dxDesc,
                            _data.dx_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT:
        setTensorDescriptor(_dscaleDesc,
                            _data.dscale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT:
        setOptionalTensorDescriptor(_dbiasDesc,
                                    _data.dbias_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "RMSNormBackwardOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "RMSNormBackwardOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void RMSNormBackwardOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                      hipdnnBackendAttributeType_t attributeType,
                                                      int64_t requestedElementCount,
                                                      int64_t* elementCount,
                                                      void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "RMSNormBackwardOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT:
        getTensorDescriptor(_invRmsDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT:
        getTensorDescriptor(_dxDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT:
        getTensorDescriptor(_dscaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT:
        getOptionalTensorDescriptor(_dbiasDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_RMSNORM_BACKWARD_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "RMSNormBackwardOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "RMSNormBackwardOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    RMSNormBackwardOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result
        = {_dyDesc, _xDesc, _scaleDesc, _invRmsDesc, _dxDesc, _dscaleDesc};
    if(_dbiasDesc)
    {
        result.push_back(_dbiasDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    RMSNormBackwardOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t RMSNormBackwardOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT;
}

std::string RMSNormBackwardOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "RMSNormBackwardOperationDescriptor: {";
    str += "name=" + _name;
    str += ", dy_uid=" + std::to_string(_data.dy_tensor_uid);
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", inv_rms_uid=" + std::to_string(_data.inv_rms_tensor_uid);
    str += ", dx_uid=" + std::to_string(_data.dx_tensor_uid);
    str += ", dscale_uid=" + std::to_string(_data.dscale_tensor_uid);
    str += ", dbias_uid="
           + (_data.dbias_tensor_uid ? std::to_string(*_data.dbias_tensor_uid) : "nullopt");
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<RMSNormBackwardOperationDescriptor> RMSNormBackwardOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsRMSNormBackwardAttributes();
    THROW_IF_NULL(
        attrs,
        HIPDNN_STATUS_INTERNAL_ERROR,
        "RMSNormBackwardOperationDescriptor::fromNode: RMSNormBackwardAttributes is null");

    auto desc = std::make_shared<RMSNormBackwardOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_dyDesc = findTensorInMap(
        tensorMap, attrs->dy_tensor_uid, "RMSNormBackwardOperationDescriptor::fromNode: Dy");
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "RMSNormBackwardOperationDescriptor::fromNode: X");
    desc->_scaleDesc = findTensorInMap(
        tensorMap, attrs->scale_tensor_uid, "RMSNormBackwardOperationDescriptor::fromNode: Scale");
    desc->_invRmsDesc = findTensorInMap(tensorMap,
                                        attrs->inv_rms_tensor_uid,
                                        "RMSNormBackwardOperationDescriptor::fromNode: InvRms");
    desc->_dxDesc = findTensorInMap(
        tensorMap, attrs->dx_tensor_uid, "RMSNormBackwardOperationDescriptor::fromNode: Dx");
    desc->_dscaleDesc = findTensorInMap(tensorMap,
                                        attrs->dscale_tensor_uid,
                                        "RMSNormBackwardOperationDescriptor::fromNode: Dscale");
    if(attrs->dbias_tensor_uid)
    {
        desc->_dbiasDesc = findTensorInMap(tensorMap,
                                           *attrs->dbias_tensor_uid,
                                           "RMSNormBackwardOperationDescriptor::fromNode: Dbias");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
