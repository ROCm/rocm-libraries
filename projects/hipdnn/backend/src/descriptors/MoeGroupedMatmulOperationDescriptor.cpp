// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "MoeGroupedMatmulOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void MoeGroupedMatmulOperationDescriptor::finalize()
{
    THROW_IF_NULL(
        _tokenDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeGroupedMatmulOperationDescriptor::finalize() failed: TOKEN_DESC tensor not set");
    THROW_IF_NULL(
        _weightDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeGroupedMatmulOperationDescriptor::finalize() failed: WEIGHT_DESC tensor not set");
    THROW_IF_NULL(_firstTokenOffsetDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MoeGroupedMatmulOperationDescriptor::finalize() failed: FIRST_TOKEN_OFFSET_DESC "
                  "tensor not set");
    THROW_IF_NULL(
        _outputDesc,
        HIPDNN_STATUS_BAD_PARAM,
        "MoeGroupedMatmulOperationDescriptor::finalize() failed: OUTPUT_DESC tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MoeGroupedMatmulOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    THROW_IF_TRUE(_firstTokenOffsetDesc->getData().data_type
                      != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
                  HIPDNN_STATUS_BAD_PARAM,
                  "MoeGroupedMatmulOperationDescriptor::finalize() failed: FIRST_TOKEN_OFFSET_DESC "
                  "tensor must have "
                  "INT32 data type");
    switch(_data.mode)
    {
    case hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode::NONE:
        THROW_IF_TRUE(_tokenIndexDesc != nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "MoeGroupedMatmulOperationDescriptor::finalize() failed: NONE mode forbids "
                      "TOKEN_INDEX_DESC tensor");
        THROW_IF_TRUE(_tokenKsDesc != nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "MoeGroupedMatmulOperationDescriptor::finalize() failed: NONE mode forbids "
                      "TOKEN_KS_DESC tensor");
        THROW_IF_NE(_data.top_k,
                    0,
                    HIPDNN_STATUS_BAD_PARAM,
                    "MoeGroupedMatmulOperationDescriptor::finalize() failed: NONE mode requires "
                    "top_k to equal 0");
        break;
    case hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode::GATHER:
        THROW_IF_NULL(_tokenIndexDesc,
                      HIPDNN_STATUS_BAD_PARAM,
                      "MoeGroupedMatmulOperationDescriptor::finalize() failed: GATHER mode "
                      "requires TOKEN_INDEX_DESC tensor");
        THROW_IF_TRUE(
            _tokenIndexDesc->getData().data_type
                != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
            HIPDNN_STATUS_BAD_PARAM,
            "MoeGroupedMatmulOperationDescriptor::finalize() failed: TOKEN_INDEX_DESC tensor must "
            "have INT32 data type");
        THROW_IF_TRUE(_tokenKsDesc != nullptr,
                      HIPDNN_STATUS_BAD_PARAM,
                      "MoeGroupedMatmulOperationDescriptor::finalize() failed: GATHER mode forbids "
                      "TOKEN_KS_DESC tensor");
        THROW_IF_NE(_data.top_k,
                    0,
                    HIPDNN_STATUS_BAD_PARAM,
                    "MoeGroupedMatmulOperationDescriptor::finalize() failed: GATHER mode requires "
                    "top_k to equal 0");
        break;
    case hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode::SCATTER:
        THROW_IF_NULL(_tokenIndexDesc,
                      HIPDNN_STATUS_BAD_PARAM,
                      "MoeGroupedMatmulOperationDescriptor::finalize() failed: SCATTER mode "
                      "requires TOKEN_INDEX_DESC tensor");
        THROW_IF_TRUE(
            _tokenIndexDesc->getData().data_type
                != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
            HIPDNN_STATUS_BAD_PARAM,
            "MoeGroupedMatmulOperationDescriptor::finalize() failed: TOKEN_INDEX_DESC tensor must "
            "have INT32 data type");
        THROW_IF_NULL(_tokenKsDesc,
                      HIPDNN_STATUS_BAD_PARAM,
                      "MoeGroupedMatmulOperationDescriptor::finalize() failed: SCATTER mode "
                      "requires TOKEN_KS_DESC tensor");
        THROW_IF_TRUE(
            _tokenKsDesc->getData().data_type
                != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32,
            HIPDNN_STATUS_BAD_PARAM,
            "MoeGroupedMatmulOperationDescriptor::finalize() failed: TOKEN_KS_DESC tensor must "
            "have INT32 data type");
        THROW_IF_TRUE(
            _data.top_k < 1,
            HIPDNN_STATUS_BAD_PARAM,
            "MoeGroupedMatmulOperationDescriptor::finalize() failed: SCATTER mode requires "
            "top_k to be at least 1");
        THROW_IF_TRUE(_weightDesc->getData().dims.empty(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "MoeGroupedMatmulOperationDescriptor::finalize() failed: WEIGHT_DESC "
                      "tensor has too few dimensions to bound top_k");
        THROW_IF_TRUE(_data.top_k > _weightDesc->getData().dims[0],
                      HIPDNN_STATUS_BAD_PARAM,
                      "MoeGroupedMatmulOperationDescriptor::finalize() failed: top_k must not "
                      "exceed weight dimension 0");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_BAD_PARAM,
            "MoeGroupedMatmulOperationDescriptor::finalize() failed: unknown routing mode");
    }

    HipdnnBackendDescriptorImpl<MoeGroupedMatmulOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void MoeGroupedMatmulOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                       hipdnnBackendAttributeType_t attributeType,
                                                       int64_t elementCount,
                                                       const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "MoeGroupedMatmulOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC:
        setTensorDescriptor(_tokenDesc,
                            _data.token_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC:
        setTensorDescriptor(_weightDesc,
                            _data.weight_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC:
        setTensorDescriptor(_firstTokenOffsetDesc,
                            _data.first_token_offset_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC:
        setOptionalTensorDescriptor(_tokenIndexDesc,
                                    _data.token_index_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC:
        setOptionalTensorDescriptor(_tokenKsDesc,
                                    _data.token_ks_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC:
        setTensorDescriptor(_outputDesc,
                            _data.output_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE:
        setMoeGroupedMatmulMode(_data.mode,
                                attributeType,
                                elementCount,
                                arrayOfElements,
                                "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K:
        setScalar(_data.top_k,
                  HIPDNN_TYPE_INT32,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "MoeGroupedMatmulOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "MoeGroupedMatmulOperationDescriptor::setAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void MoeGroupedMatmulOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                       hipdnnBackendAttributeType_t attributeType,
                                                       int64_t requestedElementCount,
                                                       int64_t* elementCount,
                                                       void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "MoeGroupedMatmulOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC:
        getTensorDescriptor(_tokenDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC:
        getTensorDescriptor(_weightDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC:
        getTensorDescriptor(_firstTokenOffsetDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC:
        getOptionalTensorDescriptor(_tokenIndexDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC:
        getOptionalTensorDescriptor(_tokenKsDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC:
        getTensorDescriptor(_outputDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE:
        getMoeGroupedMatmulMode(_data.mode,
                                attributeType,
                                requestedElementCount,
                                elementCount,
                                arrayOfElements,
                                "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K:
        getScalar(_data.top_k,
                  HIPDNN_TYPE_INT32,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_MOE_GROUPED_MATMUL_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "MoeGroupedMatmulOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            "MoeGroupedMatmulOperationDescriptor::getAttribute: attributeName not "
            "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    MoeGroupedMatmulOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(6);
    result.push_back(_tokenDesc);
    result.push_back(_weightDesc);
    result.push_back(_firstTokenOffsetDesc);
    if(_tokenIndexDesc)
    {
        result.push_back(_tokenIndexDesc);
    }
    if(_tokenKsDesc)
    {
        result.push_back(_tokenKsDesc);
    }
    result.push_back(_outputDesc);
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    MoeGroupedMatmulOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t MoeGroupedMatmulOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR;
}

std::string MoeGroupedMatmulOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "MoeGroupedMatmulOperationDescriptor: {";
    str += "name=" + _name;
    str += ", token_uid=" + std::to_string(_data.token_tensor_uid);
    str += ", weight_uid=" + std::to_string(_data.weight_tensor_uid);
    str += ", first_token_offset_uid=" + std::to_string(_data.first_token_offset_tensor_uid);
    str += ", token_index_uid="
           + (_data.token_index_tensor_uid ? std::to_string(*_data.token_index_tensor_uid)
                                           : "nullopt");
    str += ", token_ks_uid="
           + (_data.token_ks_tensor_uid ? std::to_string(*_data.token_ks_tensor_uid) : "nullopt");
    str += ", output_uid=" + std::to_string(_data.output_tensor_uid);
    str += ", mode="
           + std::string(
               hipdnn_flatbuffers_sdk::data_objects::EnumNameMoeGroupedMatmulMode(_data.mode));
    str += ", top_k=" + std::to_string(_data.top_k);
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<MoeGroupedMatmulOperationDescriptor> MoeGroupedMatmulOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsMoeGroupedMatmulAttributes();
    THROW_IF_NULL(
        attrs,
        HIPDNN_STATUS_INTERNAL_ERROR,
        "MoeGroupedMatmulOperationDescriptor::fromNode: MoeGroupedMatmulAttributes is null");

    auto desc = std::make_shared<MoeGroupedMatmulOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_tokenDesc = findTensorInMap(
        tensorMap, attrs->token_tensor_uid, "MoeGroupedMatmulOperationDescriptor::fromNode: Token");
    desc->_weightDesc = findTensorInMap(tensorMap,
                                        attrs->weight_tensor_uid,
                                        "MoeGroupedMatmulOperationDescriptor::fromNode: Weight");
    desc->_firstTokenOffsetDesc
        = findTensorInMap(tensorMap,
                          attrs->first_token_offset_tensor_uid,
                          "MoeGroupedMatmulOperationDescriptor::fromNode: FirstTokenOffset");
    if(attrs->token_index_tensor_uid)
    {
        desc->_tokenIndexDesc
            = findTensorInMap(tensorMap,
                              *attrs->token_index_tensor_uid,
                              "MoeGroupedMatmulOperationDescriptor::fromNode: TokenIndex");
    }
    if(attrs->token_ks_tensor_uid)
    {
        desc->_tokenKsDesc
            = findTensorInMap(tensorMap,
                              *attrs->token_ks_tensor_uid,
                              "MoeGroupedMatmulOperationDescriptor::fromNode: TokenKs");
    }
    desc->_outputDesc = findTensorInMap(tensorMap,
                                        attrs->output_tensor_uid,
                                        "MoeGroupedMatmulOperationDescriptor::fromNode: Output");
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
