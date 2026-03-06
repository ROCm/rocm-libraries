// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SdpaFpropOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void SdpaFpropOperationDescriptor::finalize()
{
    THROW_IF_NULL(_qDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaFpropOperationDescriptor::finalize() failed: Q tensor not set");
    THROW_IF_NULL(_kDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaFpropOperationDescriptor::finalize() failed: K tensor not set");
    THROW_IF_NULL(_vDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaFpropOperationDescriptor::finalize() failed: V tensor not set");
    THROW_IF_NULL(_oDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaFpropOperationDescriptor::finalize() failed: O tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_data_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaFpropOperationDescriptor::finalize() failed: compute data type not "
                  "set");
    HipdnnBackendDescriptorImpl<SdpaFpropOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void SdpaFpropOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t elementCount,
                                                const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "SdpaFpropOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT:
        setTensorDescriptor(_qDesc,
                            _data.q_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT:
        setTensorDescriptor(_kDesc,
                            _data.k_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT:
        setTensorDescriptor(_vDesc,
                            _data.v_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT:
        setTensorDescriptor(_oDesc,
                            _data.o_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT:
        setTensorDescriptor(_attnMaskDesc,
                            _data.attn_mask_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT:
        setTensorDescriptor(_scaleDesc,
                            _data.scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT:
        setTensorDescriptor(_seqLenQDesc,
                            _data.seq_len_q_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT:
        setTensorDescriptor(_seqLenKvDesc,
                            _data.seq_len_kv_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT:
        setTensorDescriptor(_seedDesc,
                            _data.seed_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT:
        setTensorDescriptor(_offsetDesc,
                            _data.offset_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT:
        setTensorDescriptor(_dropoutMaskDesc,
                            _data.dropout_mask_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT:
        setTensorDescriptor(_dropoutScaleDesc,
                            _data.dropout_scale_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT:
        setTensorDescriptor(_pageTableKDesc,
                            _data.page_table_k_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT:
        setTensorDescriptor(_pageTableVDesc,
                            _data.page_table_v_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT:
        setTensorDescriptor(_blockMaskDesc,
                            _data.block_mask_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT:
        setTensorDescriptor(_sinkTokenDesc,
                            _data.sink_token_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT:
        setTensorDescriptor(_descaleQDesc,
                            _data.descale_q_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT:
        setTensorDescriptor(_descaleKDesc,
                            _data.descale_k_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT:
        setTensorDescriptor(_descaleVDesc,
                            _data.descale_v_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT:
        setTensorDescriptor(_descaleSDesc,
                            _data.descale_s_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT:
        setTensorDescriptor(_scaleSDesc,
                            _data.scale_s_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT:
        setTensorDescriptor(_scaleODesc,
                            _data.scale_o_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT:
        setTensorDescriptor(_statsDesc,
                            _data.stats_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT:
        setTensorDescriptor(_maxDesc,
                            _data.max_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT:
        setTensorDescriptor(_sumExpDesc,
                            _data.sum_exp_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT:
        setTensorDescriptor(_rngDumpDesc,
                            _data.rng_dump_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT:
        setTensorDescriptor(_amaxSDesc,
                            _data.amax_s_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT:
        setTensorDescriptor(_amaxODesc,
                            _data.amax_o_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_GENERATE_STATS_EXT:
        setScalar(_data.generate_stats,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_ALIBI_MASK_EXT:
        setScalar(_data.alibi_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_PADDING_MASK_EXT:
        setScalar(_data.padding_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_CAUSAL_MASK_EXT:
        setScalar(_data.causal_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_CAUSAL_MASK_BOTTOM_RIGHT_EXT:
        setScalar(_data.causal_mask_bottom_right,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_DROPOUT_PROBABILITY_EXT:
        setScalar(_data.dropout_probability,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_ATTN_SCALE_VALUE_EXT:
        setScalar(_data.attn_scale_value,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT:
        setScalar(_data.left_bound,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT:
        setScalar(_data.right_bound,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT:
        setScalar(_data.max_seq_len_kv,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT:
        setDiagonalAlignment(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT:
        setMmaCoreMode(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT:
        setImplementation(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "SdpaFpropOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

void SdpaFpropOperationDescriptor::setDiagonalAlignment(hipdnnBackendAttributeType_t attributeType,
                                                        int64_t elementCount,
                                                        const void* arrayOfElements)
{
    checkSetArgs(HIPDNN_TYPE_INT64,
                 attributeType,
                 arrayOfElements,
                 "SdpaFpropOperationDescriptor::setAttribute()");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "SdpaFpropOperationDescriptor::setAttribute(): elementCount is not 1");
    auto mode = static_cast<hipdnn_data_sdk::data_objects::DiagonalAlignment>(
        *static_cast<const int64_t*>(arrayOfElements));
    THROW_IF_TRUE(mode < hipdnn_data_sdk::data_objects::DiagonalAlignment::MIN
                      || mode > hipdnn_data_sdk::data_objects::DiagonalAlignment::MAX,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaFpropOperationDescriptor::setAttribute(): invalid DiagonalAlignment value");
    _data.diagonal_alignment = mode;
}

void SdpaFpropOperationDescriptor::setMmaCoreMode(hipdnnBackendAttributeType_t attributeType,
                                                  int64_t elementCount,
                                                  const void* arrayOfElements)
{
    checkSetArgs(HIPDNN_TYPE_INT64,
                 attributeType,
                 arrayOfElements,
                 "SdpaFpropOperationDescriptor::setAttribute()");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "SdpaFpropOperationDescriptor::setAttribute(): elementCount is not 1");
    auto mode = static_cast<hipdnn_data_sdk::data_objects::DataType>(
        *static_cast<const int64_t*>(arrayOfElements));
    THROW_IF_TRUE(mode < hipdnn_data_sdk::data_objects::DataType::MIN
                      || mode > hipdnn_data_sdk::data_objects::DataType::MAX,
                  HIPDNN_STATUS_BAD_PARAM,
                  "SdpaFpropOperationDescriptor::setAttribute(): invalid DataType value");
    _data.mma_core_mode = mode;
}

void SdpaFpropOperationDescriptor::setImplementation(hipdnnBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     const void* arrayOfElements)
{
    checkSetArgs(HIPDNN_TYPE_INT64,
                 attributeType,
                 arrayOfElements,
                 "SdpaFpropOperationDescriptor::setAttribute()");
    THROW_IF_FALSE(elementCount == 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "SdpaFpropOperationDescriptor::setAttribute(): elementCount is not 1");
    auto mode = static_cast<hipdnn_data_sdk::data_objects::AttentionImplementation>(
        *static_cast<const int64_t*>(arrayOfElements));
    THROW_IF_TRUE(
        mode < hipdnn_data_sdk::data_objects::AttentionImplementation::MIN
            || mode > hipdnn_data_sdk::data_objects::AttentionImplementation::MAX,
        HIPDNN_STATUS_BAD_PARAM,
        "SdpaFpropOperationDescriptor::setAttribute(): invalid AttentionImplementation value");
    _data.implementation = mode;
}

// ============================================================================
// getAttribute
// ============================================================================

void SdpaFpropOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                                hipdnnBackendAttributeType_t attributeType,
                                                int64_t requestedElementCount,
                                                int64_t* elementCount,
                                                void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "SdpaFpropOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT:
        getTensorDescriptor(_qDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT:
        getTensorDescriptor(_kDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT:
        getTensorDescriptor(_vDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT:
        getTensorDescriptor(_oDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT:
        getTensorDescriptor(_attnMaskDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT:
        getTensorDescriptor(_scaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT:
        getTensorDescriptor(_seqLenQDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT:
        getTensorDescriptor(_seqLenKvDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT:
        getTensorDescriptor(_seedDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT:
        getTensorDescriptor(_offsetDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT:
        getTensorDescriptor(_dropoutMaskDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT:
        getTensorDescriptor(_dropoutScaleDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT:
        getTensorDescriptor(_pageTableKDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT:
        getTensorDescriptor(_pageTableVDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT:
        getTensorDescriptor(_blockMaskDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT:
        getTensorDescriptor(_sinkTokenDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT:
        getTensorDescriptor(_descaleQDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT:
        getTensorDescriptor(_descaleKDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT:
        getTensorDescriptor(_descaleVDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT:
        getTensorDescriptor(_descaleSDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT:
        getTensorDescriptor(_scaleSDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT:
        getTensorDescriptor(_scaleODesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT:
        getTensorDescriptor(_statsDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT:
        getTensorDescriptor(_maxDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT:
        getTensorDescriptor(_sumExpDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT:
        getTensorDescriptor(_rngDumpDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT:
        getTensorDescriptor(_amaxSDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT:
        getTensorDescriptor(_amaxODesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_GENERATE_STATS_EXT:
        getScalar(_data.generate_stats,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_ALIBI_MASK_EXT:
        getScalar(_data.alibi_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_PADDING_MASK_EXT:
        getScalar(_data.padding_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_CAUSAL_MASK_EXT:
        getScalar(_data.causal_mask,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_CAUSAL_MASK_BOTTOM_RIGHT_EXT:
        getScalar(_data.causal_mask_bottom_right,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_DROPOUT_PROBABILITY_EXT:
        getScalar(_data.dropout_probability,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_ATTN_SCALE_VALUE_EXT:
        getScalar(_data.attn_scale_value,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT:
        getScalar(_data.left_bound,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT:
        getScalar(_data.right_bound,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT:
        getScalar(_data.max_seq_len_kv,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT:
        getDiagonalAlignment(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT:
        getMmaCoreMode(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT:
        getImplementation(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "SdpaFpropOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

void SdpaFpropOperationDescriptor::getDiagonalAlignment(hipdnnBackendAttributeType_t attributeType,
                                                        int64_t requestedElementCount,
                                                        int64_t* elementCount,
                                                        void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_INT64, attributeType, "SdpaFpropOperationDescriptor::getAttribute()");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "SdpaFpropOperationDescriptor::getAttribute(): elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "SdpaFpropOperationDescriptor::getAttribute(): requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    *static_cast<int64_t*>(arrayOfElements) = static_cast<int64_t>(_data.diagonal_alignment);
}

void SdpaFpropOperationDescriptor::getMmaCoreMode(hipdnnBackendAttributeType_t attributeType,
                                                  int64_t requestedElementCount,
                                                  int64_t* elementCount,
                                                  void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_INT64, attributeType, "SdpaFpropOperationDescriptor::getAttribute()");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "SdpaFpropOperationDescriptor::getAttribute(): elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "SdpaFpropOperationDescriptor::getAttribute(): requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    *static_cast<int64_t*>(arrayOfElements) = static_cast<int64_t>(_data.mma_core_mode);
}

void SdpaFpropOperationDescriptor::getImplementation(hipdnnBackendAttributeType_t attributeType,
                                                     int64_t requestedElementCount,
                                                     int64_t* elementCount,
                                                     void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_INT64, attributeType, "SdpaFpropOperationDescriptor::getAttribute()");

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "SdpaFpropOperationDescriptor::getAttribute(): elementCount is null");
        *elementCount = 1;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= 1,
                   HIPDNN_STATUS_BAD_PARAM,
                   "SdpaFpropOperationDescriptor::getAttribute(): requestedElementCount < 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    *static_cast<int64_t*>(arrayOfElements) = static_cast<int64_t>(_data.implementation);
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>>
    SdpaFpropOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> tensors;
    // Required tensors
    tensors.push_back(_qDesc);
    tensors.push_back(_kDesc);
    tensors.push_back(_vDesc);
    tensors.push_back(_oDesc);
    // Optional tensors - only include if set
    auto addIfSet = [&](const std::shared_ptr<TensorDescriptor>& desc) {
        if(desc)
        {
            tensors.push_back(desc);
        }
    };
    addIfSet(_attnMaskDesc);
    addIfSet(_scaleDesc);
    addIfSet(_seqLenQDesc);
    addIfSet(_seqLenKvDesc);
    addIfSet(_seedDesc);
    addIfSet(_offsetDesc);
    addIfSet(_dropoutMaskDesc);
    addIfSet(_dropoutScaleDesc);
    addIfSet(_pageTableKDesc);
    addIfSet(_pageTableVDesc);
    addIfSet(_blockMaskDesc);
    addIfSet(_sinkTokenDesc);
    addIfSet(_descaleQDesc);
    addIfSet(_descaleKDesc);
    addIfSet(_descaleVDesc);
    addIfSet(_descaleSDesc);
    addIfSet(_scaleSDesc);
    addIfSet(_scaleODesc);
    addIfSet(_statsDesc);
    addIfSet(_maxDesc);
    addIfSet(_sumExpDesc);
    addIfSet(_rngDumpDesc);
    addIfSet(_amaxSDesc);
    addIfSet(_amaxODesc);
    return tensors;
}

std::unique_ptr<hipdnn_data_sdk::data_objects::NodeT>
    SdpaFpropOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_data_sdk::data_objects::NodeT>();
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_data_sdk::data_objects::SdpaAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t SdpaFpropOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_SDPA_FPROP_DESCRIPTOR_EXT;
}

std::string SdpaFpropOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "SdpaFpropOperationDescriptor: {";
    str += "q_uid=" + std::to_string(_data.q_tensor_uid);
    str += ", k_uid=" + std::to_string(_data.k_tensor_uid);
    str += ", v_uid=" + std::to_string(_data.v_tensor_uid);
    str += ", o_uid=" + std::to_string(_data.o_tensor_uid);
    str += ", attn_mask_uid=" + std::to_string(_data.attn_mask_tensor_uid);
    str += ", scale_uid=" + std::to_string(_data.scale_tensor_uid);
    str += ", seq_len_q_uid=" + std::to_string(_data.seq_len_q_tensor_uid);
    str += ", seq_len_kv_uid=" + std::to_string(_data.seq_len_kv_tensor_uid);
    str += ", seed_uid=" + std::to_string(_data.seed_tensor_uid);
    str += ", offset_uid=" + std::to_string(_data.offset_tensor_uid);
    str += ", dropout_mask_uid=" + std::to_string(_data.dropout_mask_tensor_uid);
    str += ", dropout_scale_uid=" + std::to_string(_data.dropout_scale_tensor_uid);
    str += ", page_table_k_uid=" + std::to_string(_data.page_table_k_tensor_uid);
    str += ", page_table_v_uid=" + std::to_string(_data.page_table_v_tensor_uid);
    str += ", block_mask_uid=" + std::to_string(_data.block_mask_tensor_uid);
    str += ", sink_token_uid=" + std::to_string(_data.sink_token_tensor_uid);
    str += ", descale_q_uid=" + std::to_string(_data.descale_q_tensor_uid);
    str += ", descale_k_uid=" + std::to_string(_data.descale_k_tensor_uid);
    str += ", descale_v_uid=" + std::to_string(_data.descale_v_tensor_uid);
    str += ", descale_s_uid=" + std::to_string(_data.descale_s_tensor_uid);
    str += ", scale_s_uid=" + std::to_string(_data.scale_s_tensor_uid);
    str += ", scale_o_uid=" + std::to_string(_data.scale_o_tensor_uid);
    str += ", stats_uid=" + std::to_string(_data.stats_tensor_uid);
    str += ", max_uid=" + std::to_string(_data.max_tensor_uid);
    str += ", sum_exp_uid=" + std::to_string(_data.sum_exp_tensor_uid);
    str += ", rng_dump_uid=" + std::to_string(_data.rng_dump_tensor_uid);
    str += ", amax_s_uid=" + std::to_string(_data.amax_s_tensor_uid);
    str += ", amax_o_uid=" + std::to_string(_data.amax_o_tensor_uid);
    str += ", generate_stats=" + std::to_string(_data.generate_stats);
    str += ", alibi_mask=" + std::to_string(_data.alibi_mask);
    str += ", padding_mask=" + std::to_string(_data.padding_mask);
    str += ", causal_mask=" + std::to_string(_data.causal_mask);
    str += ", causal_mask_bottom_right=" + std::to_string(_data.causal_mask_bottom_right);
    str += ", dropout_probability=" + std::to_string(_data.dropout_probability);
    str += ", attn_scale_value=" + std::to_string(_data.attn_scale_value);
    str += ", left_bound=" + std::to_string(_data.left_bound);
    str += ", right_bound=" + std::to_string(_data.right_bound);
    str += ", max_seq_len_kv=" + std::to_string(_data.max_seq_len_kv);
    str += ", diagonal_alignment=" + std::to_string(static_cast<int>(_data.diagonal_alignment));
    str += ", mma_core_mode=" + std::to_string(static_cast<int>(_data.mma_core_mode));
    str += ", implementation=" + std::to_string(static_cast<int>(_data.implementation));
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
