// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "MoeRoutingOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void MoeRoutingOperationDescriptor::finalize()
{
    THROW_IF_NULL(
        _topkIdsDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeRoutingOperationDescriptor::finalize() failed: TOPK_IDS_DESC_EXT tensor not set");
    THROW_IF_NULL(_firstTokenOffsetDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MoeRoutingOperationDescriptor::finalize() failed: FIRST_TOKEN_OFFSET_DESC_EXT "
                  "tensor not set");
    THROW_IF_NULL(
        _tokenIndexDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeRoutingOperationDescriptor::finalize() failed: TOKEN_INDEX_DESC_EXT tensor not set");
    THROW_IF_NULL(
        _tokenKsDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeRoutingOperationDescriptor::finalize() failed: TOKEN_KS_DESC_EXT tensor not set");
    THROW_IF_TRUE(
        _topkIdsDesc->getData().data_type != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeRoutingOperationDescriptor::finalize() failed: TOPK_IDS_DESC_EXT tensor must have "
        "INT32 data type");
    THROW_IF_TRUE(
        _tokenIndexDesc->getData().data_type
            != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeRoutingOperationDescriptor::finalize() failed: TOKEN_INDEX_DESC_EXT tensor must have "
        "INT32 data type");
    THROW_IF_TRUE(
        _tokenKsDesc->getData().data_type != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeRoutingOperationDescriptor::finalize() failed: TOKEN_KS_DESC_EXT tensor must have "
        "INT32 data type");
    const auto offsetType = _firstTokenOffsetDesc->getData().data_type;
    THROW_IF_TRUE(offsetType != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32
                      && offsetType != hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MoeRoutingOperationDescriptor::finalize() failed: FIRST_TOKEN_OFFSET_DESC_EXT "
                  "tensor must have INT32 or INT64 data type");
    THROW_IF_TRUE(_data.num_experts < 1,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MoeRoutingOperationDescriptor::finalize() failed: NUM_EXPERTS_EXT must be >= 1");

    HipdnnBackendDescriptorImpl<MoeRoutingOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void MoeRoutingOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                 hipdnnBackendAttributeType_t attributeType,
                                                 int64_t elementCount,
                                                 const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "MoeRoutingOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT:
        setTensorDescriptor(_topkIdsDesc,
                            _data.topk_ids_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeRoutingOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT:
        setTensorDescriptor(_firstTokenOffsetDesc,
                            _data.first_token_offset_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeRoutingOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT:
        setTensorDescriptor(_tokenIndexDesc,
                            _data.token_index_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeRoutingOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT:
        setTensorDescriptor(_tokenKsDesc,
                            _data.token_ks_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeRoutingOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT:
        setScalar(_data.num_experts,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "MoeRoutingOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "MoeRoutingOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "MoeRoutingOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void MoeRoutingOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                 hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "MoeRoutingOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT:
        getTensorDescriptor(_topkIdsDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeRoutingOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT:
        getTensorDescriptor(_firstTokenOffsetDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeRoutingOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT:
        getTensorDescriptor(_tokenIndexDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeRoutingOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT:
        getTensorDescriptor(_tokenKsDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeRoutingOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT:
        getScalar(_data.num_experts,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "MoeRoutingOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "MoeRoutingOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_MOE_ROUTING_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "MoeRoutingOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "MoeRoutingOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    MoeRoutingOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(4);
    result.push_back(_topkIdsDesc);
    result.push_back(_firstTokenOffsetDesc);
    result.push_back(_tokenIndexDesc);
    result.push_back(_tokenKsDesc);
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    MoeRoutingOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::MoeRoutingAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t MoeRoutingOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_MOE_ROUTING_DESCRIPTOR_EXT;
}

std::string MoeRoutingOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "MoeRoutingOperationDescriptor: {";
    str += "name=" + _name;
    str += ", topk_ids_uid=" + std::to_string(_data.topk_ids_tensor_uid);
    str += ", first_token_offset_uid=" + std::to_string(_data.first_token_offset_tensor_uid);
    str += ", token_index_uid=" + std::to_string(_data.token_index_tensor_uid);
    str += ", token_ks_uid=" + std::to_string(_data.token_ks_tensor_uid);
    str += ", num_experts=" + std::to_string(_data.num_experts);
    str += "}";
    return str;
}

std::shared_ptr<MoeRoutingOperationDescriptor> MoeRoutingOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsMoeRoutingAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "MoeRoutingOperationDescriptor::fromNode: MoeRoutingAttributes is null");

    auto desc = std::make_shared<MoeRoutingOperationDescriptor>();
    desc->_data = *attrs;
    desc->_name = nodeT.name;
    desc->_topkIdsDesc = findTensorInMap(
        tensorMap, attrs->topk_ids_tensor_uid, "MoeRoutingOperationDescriptor::fromNode: TopkIds");
    desc->_firstTokenOffsetDesc
        = findTensorInMap(tensorMap,
                          attrs->first_token_offset_tensor_uid,
                          "MoeRoutingOperationDescriptor::fromNode: FirstTokenOffset");
    desc->_tokenIndexDesc = findTensorInMap(tensorMap,
                                            attrs->token_index_tensor_uid,
                                            "MoeRoutingOperationDescriptor::fromNode: TokenIndex");
    desc->_tokenKsDesc = findTensorInMap(
        tensorMap, attrs->token_ks_tensor_uid, "MoeRoutingOperationDescriptor::fromNode: TokenKs");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
