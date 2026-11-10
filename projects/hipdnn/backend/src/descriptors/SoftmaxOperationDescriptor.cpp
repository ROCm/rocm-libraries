// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SoftmaxOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void SoftmaxOperationDescriptor::finalize()
{
    THROW_IF_NULL(_pDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SoftmaxOperationDescriptor::finalize() failed: XDESC tensor not set");
    THROW_IF_NULL(_sDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SoftmaxOperationDescriptor::finalize() failed: YDESC tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SoftmaxOperationDescriptor::finalize() failed: compute data type not "
                  "set");

    HipdnnBackendDescriptorImpl<SoftmaxOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void SoftmaxOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                              hipdnnBackendAttributeType_t attributeType,
                                              int64_t elementCount,
                                              const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "SoftmaxOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC:
        setTensorDescriptor(_pDesc,
                            _data.p_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SoftmaxOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC:
        setTensorDescriptor(_sDesc,
                            _data.s_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SoftmaxOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC:
        setOptionalTensorDescriptor(_statsDesc,
                                    _data.stats_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SoftmaxOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2:
        setScalar(_data.stats_use_log2,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SoftmaxOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "SoftmaxOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SoftmaxOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "SoftmaxOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void SoftmaxOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                              hipdnnBackendAttributeType_t attributeType,
                                              int64_t requestedElementCount,
                                              int64_t* elementCount,
                                              void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "SoftmaxOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC:
        getTensorDescriptor(_pDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SoftmaxOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC:
        getTensorDescriptor(_sDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SoftmaxOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC:
        getOptionalTensorDescriptor(_statsDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SoftmaxOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2:
        getScalar(_data.stats_use_log2,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SoftmaxOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "SoftmaxOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SoftmaxOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_SOFTMAX_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "SoftmaxOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "SoftmaxOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    SoftmaxOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(3);
    result.push_back(_pDesc);
    result.push_back(_sDesc);
    if(_statsDesc)
    {
        result.push_back(_statsDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    SoftmaxOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::SoftmaxAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t SoftmaxOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR;
}

std::string SoftmaxOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "SoftmaxOperationDescriptor: {";
    str += "name=" + _name;
    str += ", p_uid=" + std::to_string(_data.p_tensor_uid);
    str += ", s_uid=" + std::to_string(_data.s_tensor_uid);
    str += ", stats_uid="
           + (_data.stats_tensor_uid ? std::to_string(*_data.stats_tensor_uid) : "nullopt");
    str += ", stats_use_log2=" + std::string(_data.stats_use_log2 ? "true" : "false");
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<SoftmaxOperationDescriptor> SoftmaxOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsSoftmaxAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "SoftmaxOperationDescriptor::fromNode: SoftmaxAttributes is null");

    auto desc = std::make_shared<SoftmaxOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_pDesc = findTensorInMap(
        tensorMap, attrs->p_tensor_uid, "SoftmaxOperationDescriptor::fromNode: P");
    desc->_sDesc = findTensorInMap(
        tensorMap, attrs->s_tensor_uid, "SoftmaxOperationDescriptor::fromNode: S");
    if(attrs->stats_tensor_uid)
    {
        desc->_statsDesc = findTensorInMap(
            tensorMap, *attrs->stats_tensor_uid, "SoftmaxOperationDescriptor::fromNode: Stats");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
