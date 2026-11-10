// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TopkOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void TopkOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "TopkOperationDescriptor::finalize() failed: XDESC_EXT tensor not set");
    THROW_IF_NULL(_indicesDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "TopkOperationDescriptor::finalize() failed: INDICES_DESC_EXT tensor not set");

    HipdnnBackendDescriptorImpl<TopkOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void TopkOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "TopkOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "TopkOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT:
        setTensorDescriptor(_indicesDesc,
                            _data.indices_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "TopkOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT:
        setOptionalTensorDescriptor(_valuesDesc,
                                    _data.values_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "TopkOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TOPK_K_EXT:
        setScalar(_data.k,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "TopkOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT:
        setScalar(_data.largest,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "TopkOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "TopkOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "TopkOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void TopkOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "TopkOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "TopkOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT:
        getTensorDescriptor(_indicesDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "TopkOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT:
        getOptionalTensorDescriptor(_valuesDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "TopkOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TOPK_K_EXT:
        getScalar(_data.k,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "TopkOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT:
        getScalar(_data.largest,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "TopkOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "TopkOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_TOPK_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "TopkOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "TopkOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>> TopkOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(3);
    result.push_back(_xDesc);
    result.push_back(_indicesDesc);
    if(_valuesDesc)
    {
        result.push_back(_valuesDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    TopkOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::TopkAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t TopkOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_TOPK_DESCRIPTOR_EXT;
}

std::string TopkOperationDescriptor::toString() const
{
    std::string str = "TopkOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", indices_uid=" + std::to_string(_data.indices_tensor_uid);
    str += ", values_uid="
           + (_data.values_tensor_uid ? std::to_string(*_data.values_tensor_uid)
                                      : std::string("nullopt"));
    str += ", k=" + std::to_string(_data.k);
    str += ", largest=" + std::string(_data.largest ? "true" : "false");
    str += "}";
    return str;
}

std::shared_ptr<TopkOperationDescriptor> TopkOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsTopkAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "TopkOperationDescriptor::fromNode: TopkAttributes is null");

    auto desc = std::make_shared<TopkOperationDescriptor>();
    desc->_data = *attrs;
    desc->_name = nodeT.name;
    desc->_xDesc
        = findTensorInMap(tensorMap, attrs->x_tensor_uid, "TopkOperationDescriptor::fromNode: X");
    desc->_indicesDesc = findTensorInMap(
        tensorMap, attrs->indices_tensor_uid, "TopkOperationDescriptor::fromNode: Indices");
    if(attrs->values_tensor_uid)
    {
        desc->_valuesDesc = findTensorInMap(
            tensorMap, *attrs->values_tensor_uid, "TopkOperationDescriptor::fromNode: Values");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
