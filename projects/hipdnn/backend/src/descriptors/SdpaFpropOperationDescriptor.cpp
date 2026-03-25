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
        setOptionalTensorDescriptor(_attnMaskDesc,
                                    _data.attn_mask_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT:
        setOptionalTensorDescriptor(_scaleDesc,
                                    _data.scale_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT:
        setOptionalTensorDescriptor(_seqLenQDesc,
                                    _data.seq_len_q_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT:
        setOptionalTensorDescriptor(_seqLenKvDesc,
                                    _data.seq_len_kv_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT:
        setOptionalTensorDescriptor(_seedDesc,
                                    _data.seed_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT:
        setOptionalTensorDescriptor(_offsetDesc,
                                    _data.offset_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT:
        setOptionalTensorDescriptor(_dropoutMaskDesc,
                                    _data.dropout_mask_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT:
        setOptionalTensorDescriptor(_dropoutScaleDesc,
                                    _data.dropout_scale_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT:
        setOptionalTensorDescriptor(_pageTableKDesc,
                                    _data.page_table_k_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT:
        setOptionalTensorDescriptor(_pageTableVDesc,
                                    _data.page_table_v_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT:
        setOptionalTensorDescriptor(_blockMaskDesc,
                                    _data.block_mask_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT:
        setOptionalTensorDescriptor(_sinkTokenDesc,
                                    _data.sink_token_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT:
        setOptionalTensorDescriptor(_descaleQDesc,
                                    _data.descale_q_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT:
        setOptionalTensorDescriptor(_descaleKDesc,
                                    _data.descale_k_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT:
        setOptionalTensorDescriptor(_descaleVDesc,
                                    _data.descale_v_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT:
        setOptionalTensorDescriptor(_descaleSDesc,
                                    _data.descale_s_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT:
        setOptionalTensorDescriptor(_scaleSDesc,
                                    _data.scale_s_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT:
        setOptionalTensorDescriptor(_scaleODesc,
                                    _data.scale_o_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT:
        setOptionalTensorDescriptor(_statsDesc,
                                    _data.stats_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT:
        setOptionalTensorDescriptor(_maxDesc,
                                    _data.max_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT:
        setOptionalTensorDescriptor(_sumExpDesc,
                                    _data.sum_exp_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT:
        setOptionalTensorDescriptor(_rngDumpDesc,
                                    _data.rng_dump_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT:
        setOptionalTensorDescriptor(_amaxSDesc,
                                    _data.amax_s_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT:
        setOptionalTensorDescriptor(_amaxODesc,
                                    _data.amax_o_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_GENERATE_STATS_EXT:
        setOptionalScalar<HIPDNN_TYPE_BOOLEAN>(_data.generate_stats,
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
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.dropout_probability,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_ATTN_SCALE_VALUE_EXT:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.attn_scale_value,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT:
        setOptionalScalar<HIPDNN_TYPE_INT64>(_data.left_bound,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT:
        setOptionalScalar<HIPDNN_TYPE_INT64>(_data.right_bound,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT:
        setOptionalScalar<HIPDNN_TYPE_INT32>(_data.max_seq_len_kv,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT:
        hipdnn_backend::setDiagonalAlignment(_data.diagonal_alignment,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT:
        setDataType(_data.mma_core_mode,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT:
        hipdnn_backend::setAttentionImplementation(_data.implementation,
                                                   attributeType,
                                                   elementCount,
                                                   arrayOfElements,
                                                   "SdpaFpropOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT:
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
        getOptionalTensorDescriptor(_attnMaskDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT:
        getOptionalTensorDescriptor(_scaleDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT:
        getOptionalTensorDescriptor(_seqLenQDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT:
        getOptionalTensorDescriptor(_seqLenKvDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT:
        getOptionalTensorDescriptor(_seedDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT:
        getOptionalTensorDescriptor(_offsetDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT:
        getOptionalTensorDescriptor(_dropoutMaskDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT:
        getOptionalTensorDescriptor(_dropoutScaleDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT:
        getOptionalTensorDescriptor(_pageTableKDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT:
        getOptionalTensorDescriptor(_pageTableVDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT:
        getOptionalTensorDescriptor(_blockMaskDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT:
        getOptionalTensorDescriptor(_sinkTokenDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT:
        getOptionalTensorDescriptor(_descaleQDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT:
        getOptionalTensorDescriptor(_descaleKDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT:
        getOptionalTensorDescriptor(_descaleVDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT:
        getOptionalTensorDescriptor(_descaleSDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT:
        getOptionalTensorDescriptor(_scaleSDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT:
        getOptionalTensorDescriptor(_scaleODesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT:
        getOptionalTensorDescriptor(_statsDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT:
        getOptionalTensorDescriptor(_maxDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT:
        getOptionalTensorDescriptor(_sumExpDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT:
        getOptionalTensorDescriptor(_rngDumpDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT:
        getOptionalTensorDescriptor(_amaxSDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT:
        getOptionalTensorDescriptor(_amaxODesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_GENERATE_STATS_EXT:
        getOptionalScalar<HIPDNN_TYPE_BOOLEAN>(_data.generate_stats,
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
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.dropout_probability,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_ATTN_SCALE_VALUE_EXT:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.attn_scale_value,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT:
        getOptionalScalar<HIPDNN_TYPE_INT64>(_data.left_bound,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT:
        getOptionalScalar<HIPDNN_TYPE_INT64>(_data.right_bound,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT:
        getOptionalScalar<HIPDNN_TYPE_INT32>(_data.max_seq_len_kv,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT:
        hipdnn_backend::getDiagonalAlignment(_data.diagonal_alignment,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT:
        getDataType(_data.mma_core_mode,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT:
        hipdnn_backend::getAttentionImplementation(_data.implementation,
                                                   attributeType,
                                                   requestedElementCount,
                                                   elementCount,
                                                   arrayOfElements,
                                                   "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT:
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
    addIfSet(tensors, _attnMaskDesc);
    addIfSet(tensors, _scaleDesc);
    addIfSet(tensors, _seqLenQDesc);
    addIfSet(tensors, _seqLenKvDesc);
    addIfSet(tensors, _seedDesc);
    addIfSet(tensors, _offsetDesc);
    addIfSet(tensors, _dropoutMaskDesc);
    addIfSet(tensors, _dropoutScaleDesc);
    addIfSet(tensors, _pageTableKDesc);
    addIfSet(tensors, _pageTableVDesc);
    addIfSet(tensors, _blockMaskDesc);
    addIfSet(tensors, _sinkTokenDesc);
    addIfSet(tensors, _descaleQDesc);
    addIfSet(tensors, _descaleKDesc);
    addIfSet(tensors, _descaleVDesc);
    addIfSet(tensors, _descaleSDesc);
    addIfSet(tensors, _scaleSDesc);
    addIfSet(tensors, _scaleODesc);
    addIfSet(tensors, _statsDesc);
    addIfSet(tensors, _maxDesc);
    addIfSet(tensors, _sumExpDesc);
    addIfSet(tensors, _rngDumpDesc);
    addIfSet(tensors, _amaxSDesc);
    addIfSet(tensors, _amaxODesc);
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
    std::string str = "SdpaFpropOperationDescriptor: {";
    str += "q_uid=" + std::to_string(_data.q_tensor_uid);
    str += ", k_uid=" + std::to_string(_data.k_tensor_uid);
    str += ", v_uid=" + std::to_string(_data.v_tensor_uid);
    str += ", o_uid=" + std::to_string(_data.o_tensor_uid);
    str += ", attn_mask_uid=" + optionalToString(_data.attn_mask_tensor_uid);
    str += ", scale_uid=" + optionalToString(_data.scale_tensor_uid);
    str += ", seq_len_q_uid=" + optionalToString(_data.seq_len_q_tensor_uid);
    str += ", seq_len_kv_uid=" + optionalToString(_data.seq_len_kv_tensor_uid);
    str += ", seed_uid=" + optionalToString(_data.seed_tensor_uid);
    str += ", offset_uid=" + optionalToString(_data.offset_tensor_uid);
    str += ", dropout_mask_uid=" + optionalToString(_data.dropout_mask_tensor_uid);
    str += ", dropout_scale_uid=" + optionalToString(_data.dropout_scale_tensor_uid);
    str += ", page_table_k_uid=" + optionalToString(_data.page_table_k_tensor_uid);
    str += ", page_table_v_uid=" + optionalToString(_data.page_table_v_tensor_uid);
    str += ", block_mask_uid=" + optionalToString(_data.block_mask_tensor_uid);
    str += ", sink_token_uid=" + optionalToString(_data.sink_token_tensor_uid);
    str += ", descale_q_uid=" + optionalToString(_data.descale_q_tensor_uid);
    str += ", descale_k_uid=" + optionalToString(_data.descale_k_tensor_uid);
    str += ", descale_v_uid=" + optionalToString(_data.descale_v_tensor_uid);
    str += ", descale_s_uid=" + optionalToString(_data.descale_s_tensor_uid);
    str += ", scale_s_uid=" + optionalToString(_data.scale_s_tensor_uid);
    str += ", scale_o_uid=" + optionalToString(_data.scale_o_tensor_uid);
    str += ", stats_uid=" + optionalToString(_data.stats_tensor_uid);
    str += ", max_uid=" + optionalToString(_data.max_tensor_uid);
    str += ", sum_exp_uid=" + optionalToString(_data.sum_exp_tensor_uid);
    str += ", rng_dump_uid=" + optionalToString(_data.rng_dump_tensor_uid);
    str += ", amax_s_uid=" + optionalToString(_data.amax_s_tensor_uid);
    str += ", amax_o_uid=" + optionalToString(_data.amax_o_tensor_uid);
    str += ", generate_stats=" + optionalBoolToString(_data.generate_stats);
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
    str += ", max_seq_len_kv=" + optionalToString(_data.max_seq_len_kv);
    str += ", diagonal_alignment=" + std::to_string(static_cast<int>(_data.diagonal_alignment));
    str += ", mma_core_mode=" + std::to_string(static_cast<int>(_data.mma_core_mode));
    str += ", implementation=" + std::to_string(static_cast<int>(_data.implementation));
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
