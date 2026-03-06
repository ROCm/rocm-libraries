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
    {
        int64_t uid = 0;
        setTensorDescriptor(_attnMaskDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.attn_mask_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_scaleDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.scale_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_seqLenQDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.seq_len_q_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_seqLenKvDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.seq_len_kv_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_seedDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.seed_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_offsetDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.offset_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_dropoutMaskDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.dropout_mask_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_dropoutScaleDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.dropout_scale_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_pageTableKDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.page_table_k_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_pageTableVDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.page_table_v_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_blockMaskDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.block_mask_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_sinkTokenDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.sink_token_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_descaleQDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.descale_q_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_descaleKDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.descale_k_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_descaleVDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.descale_v_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_descaleSDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.descale_s_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_scaleSDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.scale_s_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_scaleODesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.scale_o_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_statsDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.stats_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_maxDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.max_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_sumExpDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.sum_exp_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_rngDumpDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.rng_dump_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_amaxSDesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.amax_s_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT:
    {
        int64_t uid = 0;
        setTensorDescriptor(_amaxODesc,
                            uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "SdpaFpropOperationDescriptor::setAttribute()");
        _data.amax_o_tensor_uid = uid;
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_GENERATE_STATS_EXT:
    {
        bool val = false;
        setScalar(val,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        _data.generate_stats = val;
        break;
    }
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
    {
        float val = 0.0f;
        setScalar(val,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        _data.dropout_probability = val;
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_ATTN_SCALE_VALUE_EXT:
    {
        float val = 0.0f;
        setScalar(val,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        _data.attn_scale_value = val;
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT:
    {
        int64_t val = 0;
        setScalar(val,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        _data.left_bound = val;
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT:
    {
        int64_t val = 0;
        setScalar(val,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        _data.right_bound = val;
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT:
    {
        int64_t val = 0;
        setScalar(val,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::setAttribute()");
        _data.max_seq_len_kv
            = static_cast<int32_t>(val); // FlatBuffer schema uses int32 for max_seq_len_kv
        break;
    }
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
                  "SdpaFpropOperationDescriptor::setAttribute(): invalid MMA core mode (DataType) value");
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
    {
        bool val = _data.generate_stats.value_or(false);
        getScalar(val,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    }
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
    {
        float val = _data.dropout_probability.value_or(0.0f);
        getScalar(val,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_ATTN_SCALE_VALUE_EXT:
    {
        float val = _data.attn_scale_value.value_or(0.0f);
        getScalar(val,
                  HIPDNN_TYPE_FLOAT,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT:
    {
        int64_t val = _data.left_bound.value_or(0);
        getScalar(val,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT:
    {
        int64_t val = _data.right_bound.value_or(0);
        getScalar(val,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    }
    case HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT:
    {
        int64_t val = static_cast<int64_t>(_data.max_seq_len_kv.value_or(0));
        getScalar(val,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "SdpaFpropOperationDescriptor::getAttribute()");
        break;
    }
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
    std::string str = "SdpaFpropOperationDescriptor: {";
    str += "q_uid=" + std::to_string(_data.q_tensor_uid);
    str += ", k_uid=" + std::to_string(_data.k_tensor_uid);
    str += ", v_uid=" + std::to_string(_data.v_tensor_uid);
    str += ", o_uid=" + std::to_string(_data.o_tensor_uid);
    auto optInt64Str = [](const ::flatbuffers::Optional<int64_t>& opt) -> std::string {
        return opt.has_value() ? std::to_string(*opt) : "null";
    };
    auto optBoolStr = [](const ::flatbuffers::Optional<bool>& opt) -> std::string {
        if(!opt.has_value())
        {
            return "null";
        }
        return *opt ? "true" : "false";
    };
    auto optFloatStr = [](const ::flatbuffers::Optional<float>& opt) -> std::string {
        return opt.has_value() ? std::to_string(*opt) : "null";
    };
    auto optInt32Str = [](const ::flatbuffers::Optional<int32_t>& opt) -> std::string {
        return opt.has_value() ? std::to_string(*opt) : "null";
    };
    str += ", attn_mask_uid=" + optInt64Str(_data.attn_mask_tensor_uid);
    str += ", scale_uid=" + optInt64Str(_data.scale_tensor_uid);
    str += ", seq_len_q_uid=" + optInt64Str(_data.seq_len_q_tensor_uid);
    str += ", seq_len_kv_uid=" + optInt64Str(_data.seq_len_kv_tensor_uid);
    str += ", seed_uid=" + optInt64Str(_data.seed_tensor_uid);
    str += ", offset_uid=" + optInt64Str(_data.offset_tensor_uid);
    str += ", dropout_mask_uid=" + optInt64Str(_data.dropout_mask_tensor_uid);
    str += ", dropout_scale_uid=" + optInt64Str(_data.dropout_scale_tensor_uid);
    str += ", page_table_k_uid=" + optInt64Str(_data.page_table_k_tensor_uid);
    str += ", page_table_v_uid=" + optInt64Str(_data.page_table_v_tensor_uid);
    str += ", block_mask_uid=" + optInt64Str(_data.block_mask_tensor_uid);
    str += ", sink_token_uid=" + optInt64Str(_data.sink_token_tensor_uid);
    str += ", descale_q_uid=" + optInt64Str(_data.descale_q_tensor_uid);
    str += ", descale_k_uid=" + optInt64Str(_data.descale_k_tensor_uid);
    str += ", descale_v_uid=" + optInt64Str(_data.descale_v_tensor_uid);
    str += ", descale_s_uid=" + optInt64Str(_data.descale_s_tensor_uid);
    str += ", scale_s_uid=" + optInt64Str(_data.scale_s_tensor_uid);
    str += ", scale_o_uid=" + optInt64Str(_data.scale_o_tensor_uid);
    str += ", stats_uid=" + optInt64Str(_data.stats_tensor_uid);
    str += ", max_uid=" + optInt64Str(_data.max_tensor_uid);
    str += ", sum_exp_uid=" + optInt64Str(_data.sum_exp_tensor_uid);
    str += ", rng_dump_uid=" + optInt64Str(_data.rng_dump_tensor_uid);
    str += ", amax_s_uid=" + optInt64Str(_data.amax_s_tensor_uid);
    str += ", amax_o_uid=" + optInt64Str(_data.amax_o_tensor_uid);
    str += ", generate_stats=" + optBoolStr(_data.generate_stats);
    str += ", alibi_mask=";
    str += _data.alibi_mask ? "true" : "false";
    str += ", padding_mask=";
    str += _data.padding_mask ? "true" : "false";
    str += ", causal_mask=";
    str += _data.causal_mask ? "true" : "false";
    str += ", causal_mask_bottom_right=";
    str += _data.causal_mask_bottom_right ? "true" : "false";
    str += ", dropout_probability=" + optFloatStr(_data.dropout_probability);
    str += ", attn_scale_value=" + optFloatStr(_data.attn_scale_value);
    str += ", left_bound=" + optInt64Str(_data.left_bound);
    str += ", right_bound=" + optInt64Str(_data.right_bound);
    str += ", max_seq_len_kv=" + optInt32Str(_data.max_seq_len_kv);
    str += ", diagonal_alignment=" + std::to_string(static_cast<int>(_data.diagonal_alignment));
    str += ", mma_core_mode=" + std::to_string(static_cast<int>(_data.mma_core_mode));
    str += ", implementation=" + std::to_string(static_cast<int>(_data.implementation));
    str += ", compute_data_type=";
    str += hipdnn_data_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

} // namespace hipdnn_backend
