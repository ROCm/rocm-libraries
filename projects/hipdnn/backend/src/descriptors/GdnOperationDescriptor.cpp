// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "GdnOperationDescriptor.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_backend
{

void GdnOperationDescriptor::finalize()
{
    THROW_IF_NULL(_qDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GdnOperationDescriptor::finalize() failed: Q_DESC_EXT tensor not set");
    THROW_IF_NULL(_kDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GdnOperationDescriptor::finalize() failed: K_DESC_EXT tensor not set");
    THROW_IF_NULL(_vDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GdnOperationDescriptor::finalize() failed: V_DESC_EXT tensor not set");
    THROW_IF_NULL(_gDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GdnOperationDescriptor::finalize() failed: G_DESC_EXT tensor not set");
    THROW_IF_NULL(_betaDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GdnOperationDescriptor::finalize() failed: BETA_DESC_EXT tensor not set");
    THROW_IF_NULL(_cuSeqlensDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GdnOperationDescriptor::finalize() failed: CU_SEQLENS_DESC_EXT tensor not set");
    THROW_IF_NULL(_oDesc,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GdnOperationDescriptor::finalize() failed: O_DESC_EXT tensor not set");
    THROW_IF_TRUE(_computeDataType == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                  HIPDNN_STATUS_BAD_PARAM,
                  "GdnOperationDescriptor::finalize() failed: compute data type not "
                  "set");

    HipdnnBackendDescriptorImpl<GdnOperationDescriptor>::finalize();
}

// ============================================================================
// setAttribute
// ============================================================================

void GdnOperationDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                          hipdnnBackendAttributeType_t attributeType,
                                          int64_t elementCount,
                                          const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "GdnOperationDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_GDN_Q_DESC_EXT:
        setTensorDescriptor(_qDesc,
                            _data.q_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_K_DESC_EXT:
        setTensorDescriptor(_kDesc,
                            _data.k_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_V_DESC_EXT:
        setTensorDescriptor(_vDesc,
                            _data.v_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_G_DESC_EXT:
        setTensorDescriptor(_gDesc,
                            _data.g_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_BETA_DESC_EXT:
        setTensorDescriptor(_betaDesc,
                            _data.beta_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_CU_SEQLENS_DESC_EXT:
        setTensorDescriptor(_cuSeqlensDesc,
                            _data.cu_seqlens_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_INITIAL_STATE_DESC_EXT:
        setOptionalTensorDescriptor(_initialStateDesc,
                                    _data.initial_state_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_A_LOG_DESC_EXT:
        setOptionalTensorDescriptor(_aLogDesc,
                                    _data.a_log_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_DT_BIAS_DESC_EXT:
        setOptionalTensorDescriptor(_dtBiasDesc,
                                    _data.dt_bias_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_STATE_INDICES_DESC_EXT:
        setOptionalTensorDescriptor(_stateIndicesDesc,
                                    _data.state_indices_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_O_DESC_EXT:
        setTensorDescriptor(_oDesc,
                            _data.o_tensor_uid,
                            attributeType,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_FINAL_STATE_DESC_EXT:
        setOptionalTensorDescriptor(_finalStateDesc,
                                    _data.final_state_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_STATE_CHECKPOINTS_DESC_EXT:
        setOptionalTensorDescriptor(_stateCheckpointsDesc,
                                    _data.state_checkpoints_tensor_uid,
                                    attributeType,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_SCALE_EXT:
        setOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.scale,
                                             attributeType,
                                             elementCount,
                                             arrayOfElements,
                                             "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_OUTPUT_FINAL_STATE_EXT:
        setScalar(_data.output_final_state,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_USE_QK_L2NORM_EXT:
        setScalar(_data.use_qk_l2norm,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_CHECKPOINT_EVERY_N_TOKENS_EXT:
        setScalar(_data.checkpoint_every_n_tokens,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_USE_BETA_SIGMOID_EXT:
        setScalar(_data.use_beta_sigmoid,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_ALLOW_NEG_EIGVAL_EXT:
        setScalar(_data.allow_neg_eigval,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_SAFE_GATE_EXT:
        setScalar(_data.safe_gate,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_GATE_DOMAIN_EXT:
        setGateDomain(_data.gate_domain,
                      attributeType,
                      elementCount,
                      arrayOfElements,
                      "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_BATCH_INVARIANT_EXT:
        setScalar(_data.batch_invariant,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_OVERWRITE_INITIAL_STATE_EXT:
        setScalar(_data.overwrite_initial_state,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_COMP_TYPE_EXT:
        setDataType(_computeDataType,
                    attributeType,
                    elementCount,
                    arrayOfElements,
                    "GdnOperationDescriptor::setAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        setString(_name,
                  attributeType,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::setAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "GdnOperationDescriptor::setAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// getAttribute
// ============================================================================

void GdnOperationDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                          hipdnnBackendAttributeType_t attributeType,
                                          int64_t requestedElementCount,
                                          int64_t* elementCount,
                                          void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "GdnOperationDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_OPERATION_GDN_Q_DESC_EXT:
        getTensorDescriptor(_qDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_K_DESC_EXT:
        getTensorDescriptor(_kDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_V_DESC_EXT:
        getTensorDescriptor(_vDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_G_DESC_EXT:
        getTensorDescriptor(_gDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_BETA_DESC_EXT:
        getTensorDescriptor(_betaDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_CU_SEQLENS_DESC_EXT:
        getTensorDescriptor(_cuSeqlensDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_INITIAL_STATE_DESC_EXT:
        getOptionalTensorDescriptor(_initialStateDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_A_LOG_DESC_EXT:
        getOptionalTensorDescriptor(_aLogDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_DT_BIAS_DESC_EXT:
        getOptionalTensorDescriptor(_dtBiasDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_STATE_INDICES_DESC_EXT:
        getOptionalTensorDescriptor(_stateIndicesDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_O_DESC_EXT:
        getTensorDescriptor(_oDesc,
                            attributeType,
                            requestedElementCount,
                            elementCount,
                            arrayOfElements,
                            "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_FINAL_STATE_DESC_EXT:
        getOptionalTensorDescriptor(_finalStateDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_STATE_CHECKPOINTS_DESC_EXT:
        getOptionalTensorDescriptor(_stateCheckpointsDesc,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements,
                                    "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_SCALE_EXT:
        getOptionalScalar<HIPDNN_TYPE_FLOAT>(_data.scale,
                                             attributeType,
                                             requestedElementCount,
                                             elementCount,
                                             arrayOfElements,
                                             "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_OUTPUT_FINAL_STATE_EXT:
        getScalar(_data.output_final_state,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_USE_QK_L2NORM_EXT:
        getScalar(_data.use_qk_l2norm,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_CHECKPOINT_EVERY_N_TOKENS_EXT:
        getScalar(_data.checkpoint_every_n_tokens,
                  HIPDNN_TYPE_INT64,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_USE_BETA_SIGMOID_EXT:
        getScalar(_data.use_beta_sigmoid,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_ALLOW_NEG_EIGVAL_EXT:
        getScalar(_data.allow_neg_eigval,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_SAFE_GATE_EXT:
        getScalar(_data.safe_gate,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_GATE_DOMAIN_EXT:
        getGateDomain(_data.gate_domain,
                      attributeType,
                      requestedElementCount,
                      elementCount,
                      arrayOfElements,
                      "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_BATCH_INVARIANT_EXT:
        getScalar(_data.batch_invariant,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_OVERWRITE_INITIAL_STATE_EXT:
        getScalar(_data.overwrite_initial_state,
                  HIPDNN_TYPE_BOOLEAN,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_GDN_COMP_TYPE_EXT:
        getDataType(_computeDataType,
                    attributeType,
                    requestedElementCount,
                    elementCount,
                    arrayOfElements,
                    "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_NAME_EXT:
        getString(_name,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "GdnOperationDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_OPERATION_TYPE_EXT:
        getOperationType(HIPDNN_OPERATION_TYPE_GDN_EXT,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements,
                         "GdnOperationDescriptor::getAttribute()");
        break;
    default:
        throw HipdnnException(HIPDNN_STATUS_NOT_SUPPORTED,
                              "GdnOperationDescriptor::getAttribute: attributeName not "
                              "supported");
    }
}

// ============================================================================
// Other methods
// ============================================================================

std::vector<std::shared_ptr<TensorDescriptor>> GdnOperationDescriptor::getTensorDescriptors() const
{
    std::vector<std::shared_ptr<TensorDescriptor>> result;
    result.reserve(13);
    result.push_back(_qDesc);
    result.push_back(_kDesc);
    result.push_back(_vDesc);
    result.push_back(_gDesc);
    result.push_back(_betaDesc);
    result.push_back(_cuSeqlensDesc);
    if(_initialStateDesc)
    {
        result.push_back(_initialStateDesc);
    }
    if(_aLogDesc)
    {
        result.push_back(_aLogDesc);
    }
    if(_dtBiasDesc)
    {
        result.push_back(_dtBiasDesc);
    }
    if(_stateIndicesDesc)
    {
        result.push_back(_stateIndicesDesc);
    }
    result.push_back(_oDesc);
    if(_finalStateDesc)
    {
        result.push_back(_finalStateDesc);
    }
    if(_stateCheckpointsDesc)
    {
        result.push_back(_stateCheckpointsDesc);
    }
    return result;
}

std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::NodeT>
    GdnOperationDescriptor::buildNode() const
{
    auto node = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::NodeT>();
    node->name = _name;
    node->compute_data_type = _computeDataType;
    node->attributes.Set(hipdnn_flatbuffers_sdk::data_objects::GdnAttributesT(_data));
    return node;
}

hipdnnBackendDescriptorType_t GdnOperationDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_OPERATION_GDN_DESCRIPTOR_EXT;
}

std::string GdnOperationDescriptor::toString() const
{
    using hipdnn_data_sdk::utilities::vecToString;
    std::string str = "GdnOperationDescriptor: {";
    str += "name=" + _name;
    str += ", q_uid=" + std::to_string(_data.q_tensor_uid);
    str += ", k_uid=" + std::to_string(_data.k_tensor_uid);
    str += ", v_uid=" + std::to_string(_data.v_tensor_uid);
    str += ", g_uid=" + std::to_string(_data.g_tensor_uid);
    str += ", beta_uid=" + std::to_string(_data.beta_tensor_uid);
    str += ", cu_seqlens_uid=" + std::to_string(_data.cu_seqlens_tensor_uid);
    str += ", initial_state_uid="
           + (_data.initial_state_tensor_uid ? std::to_string(*_data.initial_state_tensor_uid)
                                             : "nullopt");
    str += ", a_log_uid="
           + (_data.a_log_tensor_uid ? std::to_string(*_data.a_log_tensor_uid) : "nullopt");
    str += ", dt_bias_uid="
           + (_data.dt_bias_tensor_uid ? std::to_string(*_data.dt_bias_tensor_uid) : "nullopt");
    str += ", state_indices_uid="
           + (_data.state_indices_tensor_uid ? std::to_string(*_data.state_indices_tensor_uid)
                                             : "nullopt");
    str += ", o_uid=" + std::to_string(_data.o_tensor_uid);
    str += ", final_state_uid="
           + (_data.final_state_tensor_uid ? std::to_string(*_data.final_state_tensor_uid)
                                           : "nullopt");
    str += ", state_checkpoints_uid="
           + (_data.state_checkpoints_tensor_uid
                  ? std::to_string(*_data.state_checkpoints_tensor_uid)
                  : "nullopt");
    str += ", scale=" + (_data.scale ? std::to_string(*_data.scale) : "nullopt");
    str += ", output_final_state=" + std::string(_data.output_final_state ? "true" : "false");
    str += ", use_qk_l2norm=" + std::string(_data.use_qk_l2norm ? "true" : "false");
    str += ", checkpoint_every_n_tokens=" + std::to_string(_data.checkpoint_every_n_tokens);
    str += ", use_beta_sigmoid=" + std::string(_data.use_beta_sigmoid ? "true" : "false");
    str += ", allow_neg_eigval=" + std::string(_data.allow_neg_eigval ? "true" : "false");
    str += ", safe_gate=" + std::string(_data.safe_gate ? "true" : "false");
    str += ", gate_domain="
           + std::string(
               hipdnn_flatbuffers_sdk::data_objects::EnumNameGateDomain(_data.gate_domain));
    str += ", batch_invariant=" + std::string(_data.batch_invariant ? "true" : "false");
    str += ", overwrite_initial_state="
           + std::string(_data.overwrite_initial_state ? "true" : "false");
    str += ", compute_data_type=";
    str += hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(_computeDataType);
    str += "}";
    return str;
}

std::shared_ptr<GdnOperationDescriptor> GdnOperationDescriptor::fromNode(
    const hipdnn_flatbuffers_sdk::data_objects::NodeT& nodeT,
    const std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>>& tensorMap)
{
    const auto* attrs = nodeT.attributes.AsGdnAttributes();
    THROW_IF_NULL(attrs,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "GdnOperationDescriptor::fromNode: GdnAttributes is null");

    auto desc = std::make_shared<GdnOperationDescriptor>();
    desc->_data = *attrs;
    desc->_computeDataType = nodeT.compute_data_type;
    desc->_name = nodeT.name;
    desc->_qDesc
        = findTensorInMap(tensorMap, attrs->q_tensor_uid, "GdnOperationDescriptor::fromNode: Q");
    desc->_kDesc
        = findTensorInMap(tensorMap, attrs->k_tensor_uid, "GdnOperationDescriptor::fromNode: K");
    desc->_vDesc
        = findTensorInMap(tensorMap, attrs->v_tensor_uid, "GdnOperationDescriptor::fromNode: V");
    desc->_gDesc
        = findTensorInMap(tensorMap, attrs->g_tensor_uid, "GdnOperationDescriptor::fromNode: G");
    desc->_betaDesc = findTensorInMap(
        tensorMap, attrs->beta_tensor_uid, "GdnOperationDescriptor::fromNode: Beta");
    desc->_cuSeqlensDesc = findTensorInMap(
        tensorMap, attrs->cu_seqlens_tensor_uid, "GdnOperationDescriptor::fromNode: CuSeqlens");
    if(attrs->initial_state_tensor_uid)
    {
        desc->_initialStateDesc = findTensorInMap(tensorMap,
                                                  *attrs->initial_state_tensor_uid,
                                                  "GdnOperationDescriptor::fromNode: InitialState");
    }
    if(attrs->a_log_tensor_uid)
    {
        desc->_aLogDesc = findTensorInMap(
            tensorMap, *attrs->a_log_tensor_uid, "GdnOperationDescriptor::fromNode: ALog");
    }
    if(attrs->dt_bias_tensor_uid)
    {
        desc->_dtBiasDesc = findTensorInMap(
            tensorMap, *attrs->dt_bias_tensor_uid, "GdnOperationDescriptor::fromNode: DtBias");
    }
    if(attrs->state_indices_tensor_uid)
    {
        desc->_stateIndicesDesc = findTensorInMap(tensorMap,
                                                  *attrs->state_indices_tensor_uid,
                                                  "GdnOperationDescriptor::fromNode: StateIndices");
    }
    desc->_oDesc
        = findTensorInMap(tensorMap, attrs->o_tensor_uid, "GdnOperationDescriptor::fromNode: O");
    if(attrs->final_state_tensor_uid)
    {
        desc->_finalStateDesc = findTensorInMap(tensorMap,
                                                *attrs->final_state_tensor_uid,
                                                "GdnOperationDescriptor::fromNode: FinalState");
    }
    if(attrs->state_checkpoints_tensor_uid)
    {
        desc->_stateCheckpointsDesc
            = findTensorInMap(tensorMap,
                              *attrs->state_checkpoints_tensor_uid,
                              "GdnOperationDescriptor::fromNode: StateCheckpoints");
    }
    desc->finalize();
    return desc;
}

} // namespace hipdnn_backend
