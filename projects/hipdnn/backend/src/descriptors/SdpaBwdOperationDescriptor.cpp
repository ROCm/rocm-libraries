// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SdpaBwdOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <unordered_map>

namespace hipdnn_backend
{

void SdpaBwdOperationDescriptor::finalize()
{
    THROW_IF_NULL(_qDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: Q tensor not set");
    THROW_IF_NULL(_kDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: K tensor not set");
    THROW_IF_NULL(_vDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: V tensor not set");
    THROW_IF_NULL(_oDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: O tensor not set");
    THROW_IF_NULL(_doDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: dO tensor not set");
    THROW_IF_NULL(_statsDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: Stats tensor not set");
    THROW_IF_NULL(_dqDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: dQ tensor not set");
    THROW_IF_NULL(_dkDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: dK tensor not set");
    THROW_IF_NULL(_dvDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: dV tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaBwdOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    HipdnnBackendDescriptorImpl<SdpaBwdOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void SdpaBwdOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                              hipdnnBackendAttributeType_t attributeType,
                                              int64_t elementCount,
                                              const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "SdpaBwdOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    // Required input tensors
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_Q_EXT:
        setTensorDescriptor(_qDesc,
                            _data.q_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_K_EXT:
        setTensorDescriptor(_kDesc,
                            _data.k_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_V_EXT:
        setTensorDescriptor(_vDesc,
                            _data.v_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_O_EXT:
        setTensorDescriptor(_oDesc,
                            _data.o_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DO_EXT:
        setTensorDescriptor(_doDesc,
                            _data.do_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_STATS_EXT:
        setTensorDescriptor(_statsDesc,
                            _data.stats_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;

    // Required output tensors
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DQ_EXT:
        setTensorDescriptor(_dqDesc,
                            _data.dq_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DK_EXT:
        setTensorDescriptor(_dkDesc,
                            _data.dk_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DV_EXT:
        setTensorDescriptor(_dvDesc,
                            _data.dv_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::setAttribute()");
        break;

    // Optional input tensors
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_SCALE_EXT:
        setOptionalTensorDescriptor(_scaleDesc,
                                    _data.scale_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_ATTN_MASK_EXT:
        setOptionalTensorDescriptor(_attnMaskDesc,
                                    _data.attn_mask_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_Q_EXT:
        setOptionalTensorDescriptor(_seqLenQDesc,
                                    _data.seq_len_q_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KV_EXT:
        setOptionalTensorDescriptor(_seqLenKvDesc,
                                    _data.seq_len_kv_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_SEED_EXT:
        setOptionalTensorDescriptor(_seedDesc,
                                    _data.seed_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_OFFSET_EXT:
        setOptionalTensorDescriptor(_offsetDesc,
                                    _data.offset_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_MASK_EXT:
        setOptionalTensorDescriptor(_dropoutMaskDesc,
                                    _data.dropout_mask_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_EXT:
        setOptionalTensorDescriptor(_dropoutScaleDesc,
                                    _data.dropout_scale_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_INV_EXT:
        setOptionalTensorDescriptor(_dropoutScaleInvDesc,
                                    _data.dropout_scale_inv_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;

    // Optional output tensors
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DBIAS_EXT:
        setOptionalTensorDescriptor(_dbiasDesc,
                                    _data.dbias_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;

    // Boolean flags
    case HIPDNN_ATTR_SDPA_BWD_ALIBI_MASK_EXT:
        setScalar(_data.alibi_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_PADDING_MASK_EXT:
        setScalar(_data.padding_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_EXT:
        setScalar(_data.causal_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT:
        setScalar(_data.causal_mask_bottom_right,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::setAttribute()");
        break;

    // Optional scalar parameters
    case HIPDNN_ATTR_SDPA_BWD_DROPOUT_PROBABILITY_EXT:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.dropout_probability,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_ATTN_SCALE_VALUE_EXT:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.attn_scale_value,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_LEFT_BOUND_EXT:
        setOptionalScalar<HIPDNN_TYPE_INT64>(_data.left_bound,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_RIGHT_BOUND_EXT:
        setOptionalScalar<HIPDNN_TYPE_INT64>(_data.right_bound,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::setAttribute()");
        break;

    // Enum parameters
    case HIPDNN_ATTR_SDPA_BWD_DIAGONAL_ALIGNMENT_EXT:
        hipdnn_backend::setDiagonalAlignment(_data.diagonal_alignment,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::setAttribute()");
        break;

    // Compute data type
    case HIPDNN_ATTR_SDPA_BWD_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "SdpaBwdOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void SdpaBwdOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                              hipdnnBackendAttributeType_t attributeType,
                                              int64_t requestedElementCount,
                                              int64_t* elementCount,
                                              void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "SdpaBwdOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    // Required input tensors
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_Q_EXT:
        getTensorDescriptor(_qDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_K_EXT:
        getTensorDescriptor(_kDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_V_EXT:
        getTensorDescriptor(_vDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_O_EXT:
        getTensorDescriptor(_oDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DO_EXT:
        getTensorDescriptor(_doDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_STATS_EXT:
        getTensorDescriptor(_statsDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;

    // Required output tensors
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DQ_EXT:
        getTensorDescriptor(_dqDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DK_EXT:
        getTensorDescriptor(_dkDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DV_EXT:
        getTensorDescriptor(_dvDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaBwdOperationDescriptor::getAttribute()");
        break;

    // Optional tensors
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_SCALE_EXT:
        getOptionalTensorDescriptor(_scaleDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_ATTN_MASK_EXT:
        getOptionalTensorDescriptor(_attnMaskDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_Q_EXT:
        getOptionalTensorDescriptor(_seqLenQDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KV_EXT:
        getOptionalTensorDescriptor(_seqLenKvDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_SEED_EXT:
        getOptionalTensorDescriptor(_seedDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_OFFSET_EXT:
        getOptionalTensorDescriptor(_offsetDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_MASK_EXT:
        getOptionalTensorDescriptor(_dropoutMaskDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_EXT:
        getOptionalTensorDescriptor(_dropoutScaleDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_INV_EXT:
        getOptionalTensorDescriptor(_dropoutScaleInvDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_BWD_DBIAS_EXT:
        getOptionalTensorDescriptor(_dbiasDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;

    // Boolean flags
    case HIPDNN_ATTR_SDPA_BWD_ALIBI_MASK_EXT:
        getScalar(_data.alibi_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_PADDING_MASK_EXT:
        getScalar(_data.padding_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_EXT:
        getScalar(_data.causal_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT:
        getScalar(_data.causal_mask_bottom_right,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::getAttribute()");
        break;

    // Optional scalars
    case HIPDNN_ATTR_SDPA_BWD_DROPOUT_PROBABILITY_EXT:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.dropout_probability,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_ATTN_SCALE_VALUE_EXT:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.attn_scale_value,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_LEFT_BOUND_EXT:
        getOptionalScalar<HIPDNN_TYPE_INT64>(_data.left_bound,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_BWD_RIGHT_BOUND_EXT:
        getOptionalScalar<HIPDNN_TYPE_INT64>(_data.right_bound,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::getAttribute()");
        break;

    // Enum parameters
    case HIPDNN_ATTR_SDPA_BWD_DIAGONAL_ALIGNMENT_EXT:
        hipdnn_backend::getDiagonalAlignment(_data.diagonal_alignment,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaBwdOperationDescriptor::getAttribute()");
        break;

    // Compute data type
    case HIPDNN_ATTR_SDPA_BWD_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_SDPA_BACKWARD_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "SdpaBwdOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "SdpaBwdOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    SdpaBwdOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> tensors;
    // Required tensors
    tensors.push_back(_qDesc);
    tensors.push_back(_kDesc);
    tensors.push_back(_vDesc);
    tensors.push_back(_oDesc);
    tensors.push_back(_doDesc);
    tensors.push_back(_statsDesc);
    tensors.push_back(_dqDesc);
    tensors.push_back(_dkDesc);
    tensors.push_back(_dvDesc);
    // Optional tensors - only include if set
    addIfSet(tensors, _scaleDesc);
    addIfSet(tensors, _attnMaskDesc);
    addIfSet(tensors, _seqLenQDesc);
    addIfSet(tensors, _seqLenKvDesc);
    addIfSet(tensors, _seedDesc);
    addIfSet(tensors, _offsetDesc);
    addIfSet(tensors, _dropoutMaskDesc);
    addIfSet(tensors, _dropoutScaleDesc);
    addIfSet(tensors, _dropoutScaleInvDesc);
    addIfSet(tensors, _dbiasDesc);
    return tensors;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    SdpaBwdOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;
    node->name = _name;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributesT(_data));
    return node;
}

std::shared_ptr<SdpaBwdOperationDescriptor> SdpaBwdOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsSdpaBackwardAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "SdpaBwdOperationDescriptor::fromNode: SdpaBackwardAttributes is null");

    auto desc = std::make_shared<SdpaBwdOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;

    // Required tensors
    desc->_qDesc = findTensorInMap(
        tensorMap, attrs->q_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: Q");
    desc->_kDesc = findTensorInMap(
        tensorMap, attrs->k_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: K");
    desc->_vDesc = findTensorInMap(
        tensorMap, attrs->v_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: V");
    desc->_oDesc = findTensorInMap(
        tensorMap, attrs->o_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: O");
    desc->_doDesc = findTensorInMap(
        tensorMap, attrs->do_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: dO");
    desc->_statsDesc = findTensorInMap(
        tensorMap, attrs->stats_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: Stats");
    desc->_dqDesc = findTensorInMap(
        tensorMap, attrs->dq_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: dQ");
    desc->_dkDesc = findTensorInMap(
        tensorMap, attrs->dk_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: dK");
    desc->_dvDesc = findTensorInMap(
        tensorMap, attrs->dv_tensor_uid, "SdpaBwdOperationDescriptor::fromNode: dV");

    // Optional tensors
    desc->_scaleDesc = findOptionalTensor(tensorMap, attrs->scale_tensor_uid);
    desc->_attnMaskDesc = findOptionalTensor(tensorMap, attrs->attn_mask_tensor_uid);
    desc->_seqLenQDesc = findOptionalTensor(tensorMap, attrs->seq_len_q_tensor_uid);
    desc->_seqLenKvDesc = findOptionalTensor(tensorMap, attrs->seq_len_kv_tensor_uid);
    desc->_seedDesc = findOptionalTensor(tensorMap, attrs->seed_tensor_uid);
    desc->_offsetDesc = findOptionalTensor(tensorMap, attrs->offset_tensor_uid);
    desc->_dropoutMaskDesc = findOptionalTensor(tensorMap, attrs->dropout_mask_tensor_uid);
    desc->_dropoutScaleDesc = findOptionalTensor(tensorMap, attrs->dropout_scale_tensor_uid);
    desc->_dropoutScaleInvDesc = findOptionalTensor(tensorMap, attrs->dropout_scale_inv_tensor_uid);
    desc->_dbiasDesc = findOptionalTensor(tensorMap, attrs->dbias_tensor_uid);

    desc->finalize();
    return desc;
}

hipdnnBackendDescriptorType_t SdpaBwdOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR_EXT;
}

std::string SdpaBwdOperationDescriptor::toString() const
{
    std::string str = "SdpaBwdOperationDescriptor: {";
    str += "name=" + _name;
    str += ", q_uid=" + std::to_string(_data.q_tensor_uid);
    str += ", k_uid=" + std::to_string(_data.k_tensor_uid);
    str += ", v_uid=" + std::to_string(_data.v_tensor_uid);
    str += ", o_uid=" + std::to_string(_data.o_tensor_uid);
    str += ", do_uid=" + std::to_string(_data.do_tensor_uid);
    str += ", stats_uid=" + std::to_string(_data.stats_tensor_uid);
    str += ", dq_uid=" + std::to_string(_data.dq_tensor_uid);
    str += ", dk_uid=" + std::to_string(_data.dk_tensor_uid);
    str += ", dv_uid=" + std::to_string(_data.dv_tensor_uid);
    str += ", scale_uid=" + optionalToString(_data.scale_tensor_uid);
    str += ", attn_mask_uid=" + optionalToString(_data.attn_mask_tensor_uid);
    str += ", seq_len_q_uid=" + optionalToString(_data.seq_len_q_tensor_uid);
    str += ", seq_len_kv_uid=" + optionalToString(_data.seq_len_kv_tensor_uid);
    str += ", seed_uid=" + optionalToString(_data.seed_tensor_uid);
    str += ", offset_uid=" + optionalToString(_data.offset_tensor_uid);
    str += ", dropout_mask_uid=" + optionalToString(_data.dropout_mask_tensor_uid);
    str += ", dropout_scale_uid=" + optionalToString(_data.dropout_scale_tensor_uid);
    str += ", dropout_scale_inv_uid=" + optionalToString(_data.dropout_scale_inv_tensor_uid);
    str += ", dbias_uid=" + optionalToString(_data.dbias_tensor_uid);
    str += ", alibi_mask=";
    str += _data.alibi_mask ? "true" : "false";
    str += ", padding_mask=";
    str += _data.padding_mask ? "true" : "false";
    str += ", causal_mask=";
    str += _data.causal_mask ? "true" : "false";
    str += ", causal_mask_bottom_right=";
    str += _data.causal_mask_bottom_right ? "true" : "false";
    str += ", dropout_probability=" + optionalToString(_data.dropout_probability);
    str += ", attn_scale_value=" + optionalToString(_data.attn_scale_value);
    str += ", left_bound=" + optionalToString(_data.left_bound);
    str += ", right_bound=" + optionalToString(_data.right_bound);
    str += ", diagonal_alignment=" + std::to_string(static_cast<int>(_data.diagonal_alignment));
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
