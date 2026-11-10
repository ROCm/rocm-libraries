// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GatherOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void GatherOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GatherOperationDescriptor::finalize() failed: XDESC_EXT tensor not set");
    THROW_IF_NULL(_indicesDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GatherOperationDescriptor::finalize() failed: INDICES_DESC_EXT tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GatherOperationDescriptor::finalize() failed: YDESC_EXT tensor not set");

    HipdnnBackendDescriptorImpl<GatherOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void GatherOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t elementCount,
                                             const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "GatherOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_GATHER_XDESC_EXT:
        setTensorDescriptor(_xDesc,
                            _data.x_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GatherOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GATHER_INDICES_DESC_EXT:
        setTensorDescriptor(_indicesDesc,
                            _data.indices_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GatherOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GATHER_YDESC_EXT:
        setTensorDescriptor(_yDesc,
                            _data.y_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GatherOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GATHER_AXIS_EXT:
        setScalar(_data.axis,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GatherOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GatherOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "GatherOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void GatherOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                             hipdnnBackendAttributeType_t attributeType,
                                             int64_t requestedElementCount,
                                             int64_t* elementCount,
                                             void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "GatherOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_GATHER_XDESC_EXT:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GatherOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GATHER_INDICES_DESC_EXT:
        getTensorDescriptor(_indicesDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GatherOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GATHER_YDESC_EXT:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GatherOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GATHER_AXIS_EXT:
        getScalar(_data.axis,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GatherOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GatherOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_GATHER_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "GatherOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "GatherOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    GatherOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(3);
    result.push_back(_xDesc);
    result.push_back(_indicesDesc);
    result.push_back(_yDesc);
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    GatherOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::GatherAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t GatherOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_GATHER_DESCRIPTOR_EXT;
}

std::string GatherOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "GatherOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.x_tensor_uid);
    str += ", indices_uid=" + std::to_string(_data.indices_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.y_tensor_uid);
    str += ", axis=" + std::to_string(_data.axis);
    str += "}";
    return str;
}

std::shared_ptr<GatherOperationDescriptor> GatherOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsGatherAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "GatherOperationDescriptor::fromNode: GatherAttributes is null");

    auto desc = std::make_shared<GatherOperationDescriptor>();
    desc->_data = *attrs;
    desc->_name = nodeT.name;
    desc->_xDesc
        = findTensorInMap(tensorMap, attrs->x_tensor_uid, "GatherOperationDescriptor::fromNode: X");
    desc->_indicesDesc = findTensorInMap(
        tensorMap, attrs->indices_tensor_uid, "GatherOperationDescriptor::fromNode: Indices");
    desc->_yDesc
        = findTensorInMap(tensorMap, attrs->y_tensor_uid, "GatherOperationDescriptor::fromNode: Y");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
