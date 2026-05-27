// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ReductionOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"

namespace hipdnn_backend
{

void ReductionOperationDescriptor::finalize()
{
    THROW_IF_NULL(_xDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ReductionOperationDescriptor::finalize() failed: X_EXT tensor not set");
    THROW_IF_NULL(_yDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ReductionOperationDescriptor::finalize() failed: Y_EXT tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ReductionOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_data.mode == hipdnn_flatbuffers_sdk::data_objects::ReductionMode::NOT_SET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "ReductionOperationDescriptor::finalize() failed: mode not set");

    HipdnnBackendDescriptorImpl<ReductionOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void ReductionOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t elementCount,
                                                const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "ReductionOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_REDUCTION_XDESC:
        setTensorDescriptor(_xDesc,
                            _data.in_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ReductionOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_REDUCTION_YDESC:
        setTensorDescriptor(_yDesc,
                            _data.out_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "ReductionOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_REDUCTION_OPERATOR:
        setReductionMode(_data.mode,
                         attributeType,
                         elementCount,
                         arrayOfElements,
                         "ReductionOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC:
        setScalar(_data.is_deterministic,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "ReductionOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_REDUCTION_COMP_TYPE:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "ReductionOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "ReductionOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ReductionOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void ReductionOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t requestedElementCount,
                                                int64_t* elementCount,
                                                void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "ReductionOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_REDUCTION_XDESC:
        getTensorDescriptor(_xDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ReductionOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_REDUCTION_YDESC:
        getTensorDescriptor(_yDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "ReductionOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_REDUCTION_OPERATOR:
        getReductionMode(_data.mode,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "ReductionOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC:
        getScalar(_data.is_deterministic,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "ReductionOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_REDUCTION_COMP_TYPE:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "ReductionOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "ReductionOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_REDUCTION_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "ReductionOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "ReductionOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    ReductionOperationDescriptor::getTensorDescriptors() const
{
    return {_xDesc, _yDesc};
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    ReductionOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::ReductionAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t ReductionOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR;
}

std::string ReductionOperationDescriptor::toString() const
{
    std::string str = "ReductionOperationDescriptor: {";
    str += "name=" + _name;
    str += ", x_uid=" + std::to_string(_data.in_tensor_uid);
    str += ", y_uid=" + std::to_string(_data.out_tensor_uid);
    str += ", mode=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameReductionMode(_data.mode);
    str += ", is_deterministic=";
    str += (_data.is_deterministic ? "true" : "false");
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<ReductionOperationDescriptor> ReductionOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsReductionAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "ReductionOperationDescriptor::fromNode: ReductionAttributes is null");

    auto desc = std::make_shared<ReductionOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_xDesc = findTensorInMap(
        tensorMap, attrs->in_tensor_uid, "ReductionOperationDescriptor::fromNode: X");
    desc->_yDesc = findTensorInMap(
        tensorMap, attrs->out_tensor_uid, "ReductionOperationDescriptor::fromNode: Y");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
