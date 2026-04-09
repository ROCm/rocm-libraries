// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "LayernormBackwardOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void LayernormBackwardOperationDescriptor::finalize()
{
    THROW_IF_NULL(_dyDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormBackwardOperationDescriptor::finalize() failed: DY tensor not set");
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormBackwardOperationDescriptor::finalize() failed: X tensor not set");
    THROW_IF_NULL(_scaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormBackwardOperationDescriptor::finalize() failed: SCALE tensor not set");
    THROW_IF_NULL(_dxDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormBackwardOperationDescriptor::finalize() failed: DX tensor not set");
    THROW_IF_NULL(_dscaleDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormBackwardOperationDescriptor::finalize() failed: DSCALE tensor not set");
    THROW_IF_NULL(_dbiasDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormBackwardOperationDescriptor::finalize() failed: DBIAS tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "LayernormBackwardOperationDescriptor::finalize() failed: compute data type not "
                  "set");

    HipdnnBackendDescriptorImpl<LayernormBackwardOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void LayernormBackwardOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                        hipdnnBackendAttributeType_t attributeType,
                                                        int64_t elementCount,
                                                        const void* arrayOfElements)
{
    THROW_IF_TRUE(
        isFinalized(),
        HIPDNN_STATUS_NOT_INITIALIZED,
        "LayernormBackwardOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY:
        setTensorDescriptor(_dyDesc,
                            _data.dy_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN:
        setOptionalTensorDescriptor(_meanDesc,
                                    _data.mean_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE:
        setOptionalTensorDescriptor(_invVarianceDesc,
                                    _data.inv_variance_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON:
        setOptionalTensorDescriptor(_epsilonDesc,
                                    _data.epsilon_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX:
        setTensorDescriptor(_dxDesc,
                            _data.dx_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE:
        setTensorDescriptor(_dscaleDesc,
                            _data.dscale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS:
        setTensorDescriptor(_dbiasDesc,
                            _data.dbias_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_LAYERNORM_BACKWARD_NORMALIZED_DIM_COUNT:
        setScalar(_data.normalized_dim_count,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "LayernormBackwardOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "LayernormBackwardOperationDescriptor::setAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void LayernormBackwardOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                        hipdnnBackendAttributeType_t attributeType,
                                                        int64_t requestedElementCount,
                                                        int64_t* elementCount,
                                                        void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "LayernormBackwardOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY:
        getTensorDescriptor(_dyDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN:
        getOptionalTensorDescriptor(_meanDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE:
        getOptionalTensorDescriptor(_invVarianceDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON:
        getOptionalTensorDescriptor(_epsilonDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX:
        getTensorDescriptor(_dxDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE:
        getTensorDescriptor(_dscaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS:
        getTensorDescriptor(_dbiasDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_LAYERNORM_BACKWARD_NORMALIZED_DIM_COUNT:
        getScalar(_data.normalized_dim_count,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_LAYERNORM_BACKWARD_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "LayernormBackwardOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "LayernormBackwardOperationDescriptor::getAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    LayernormBackwardOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result = {_dyDesc, _xDesc, _scaleDesc};
    if(_meanDesc)
    {
        result.push_back(_meanDesc);
    }
    if(_invVarianceDesc)
    {
        result.push_back(_invVarianceDesc);
    }
    if(_epsilonDesc)
    {
        result.push_back(_epsilonDesc);
    }
    result.push_back(_dxDesc);
    result.push_back(_dscaleDesc);
    result.push_back(_dbiasDesc);
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    LayernormBackwardOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t LayernormBackwardOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_LAYERNORM_BACKWARD_DESCRIPTOR;
}

std::string LayernormBackwardOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "LayernormBackwardOperationDescriptor: {";
    str += "name=" + _name;
    str += ", dy_uid=" + std::to_string(_data.dy_tensor_uid);
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", mean_uid="
           + (_data.mean_tensor_uid ? std::to_string(*_data.mean_tensor_uid) : "nullopt");
    str += ", inv_variance_uid="
           + (_data.inv_variance_tensor_uid ? std::to_string(*_data.inv_variance_tensor_uid)
                                            : "nullopt");
    str += ", epsilon_uid="
           + (_data.epsilon_tensor_uid ? std::to_string(*_data.epsilon_tensor_uid) : "nullopt");
    str += ", dx_uid=" + std::to_string(_data.dx_tensor_uid);
    str += ", dscale_uid=" + std::to_string(_data.dscale_tensor_uid);
    str += ", dbias_uid=" + std::to_string(_data.dbias_tensor_uid);
    str += ", normalized_dim_count=" + std::to_string(_data.normalized_dim_count);
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<LayernormBackwardOperationDescriptor>
    LayernormBackwardOperationDescriptor::fromNode(
        const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
        const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsLayernormBackwardAttributes();
    THROW_IF_NULL(
        attrs,
        HIPDNN_STATUS_INTERNAL_ERROR,
        "LayernormBackwardOperationDescriptor::fromNode: LayernormBackwardAttributes is null");

    auto desc = std::make_shared<LayernormBackwardOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_dyDesc = findTensorInMap(
        tensorMap, attrs->dy_tensor_uid, "LayernormBackwardOperationDescriptor::fromNode: Dy");
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->x_tensor_uid, "LayernormBackwardOperationDescriptor::fromNode: X");
    desc->_scaleDesc = findTensorInMap(tensorMap,
                                       attrs->scale_tensor_uid,
                                       "LayernormBackwardOperationDescriptor::fromNode: Scale");
    if(attrs->mean_tensor_uid)
    {
        desc->_meanDesc = findTensorInMap(tensorMap,
                                          *attrs->mean_tensor_uid,
                                          "LayernormBackwardOperationDescriptor::fromNode: Mean");
    }
    if(attrs->inv_variance_tensor_uid)
    {
        desc->_invVarianceDesc
            = findTensorInMap(tensorMap,
                              *attrs->inv_variance_tensor_uid,
                              "LayernormBackwardOperationDescriptor::fromNode: InvVariance");
    }
    if(attrs->epsilon_tensor_uid)
    {
        desc->_epsilonDesc
            = findTensorInMap(tensorMap,
                              *attrs->epsilon_tensor_uid,
                              "LayernormBackwardOperationDescriptor::fromNode: Epsilon");
    }
    desc->_dxDesc = findTensorInMap(
        tensorMap, attrs->dx_tensor_uid, "LayernormBackwardOperationDescriptor::fromNode: Dx");
    desc->_dscaleDesc = findTensorInMap(tensorMap,
                                        attrs->dscale_tensor_uid,
                                        "LayernormBackwardOperationDescriptor::fromNode: Dscale");
    desc->_dbiasDesc = findTensorInMap(tensorMap,
                                       attrs->dbias_tensor_uid,
                                       "LayernormBackwardOperationDescriptor::fromNode: Dbias");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
