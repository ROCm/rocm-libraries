// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_frontend::detail
{

// Shape rules shared by the gated delta-rule linear-attention nodes (GDN and KDA), following
// the cudnn-frontend Python ops graph.gdn / graph.kda.
//
// The per-token tensors are token-packed, as in cuDNN: q [T, H, K], k [T, HK, K],
// v [T, HV, V], beta [T, HO], o [T, HO, V], with sequence offsets cu_seqlens [N + 1]. The
// leading "token prefix" [T] is shared by all of them. HK is H or HV, and HO = max(H, HV) is
// the head count of the gate, beta, output and state tensors. GDN's g is [T, HO]; KDA's is
// [T, HO, K]. States are [N, HO, V, K]; with state_indices the initial state is a pool
// [P, HO, V, K].
struct GatedDeltaRuleShape
{
    std::vector<int64_t> prefix; // [T]
    int64_t heads = 0; // H
    int64_t valueHeads = 0; // HV
    int64_t outputHeads = 0; // HO = max(H, HV)
    int64_t keyDim = 0; // K
    int64_t valueDim = 0; // V
    int64_t numSequences = 0; // N
    int64_t totalTokens = 0; // T
};

inline std::vector<int64_t> withSuffix(const std::vector<int64_t>& prefix,
                                       std::initializer_list<int64_t> suffix)
{
    std::vector<int64_t> dims = prefix;
    dims.insert(dims.end(), suffix);
    return dims;
}

inline std::string dimsToString(const std::vector<int64_t>& dims)
{
    std::string text = "[";
    for(size_t i = 0; i < dims.size(); ++i)
    {
        text += (i == 0 ? "" : ", ") + std::to_string(dims[i]);
    }
    return text + "]";
}

inline Error checkDims(const std::shared_ptr<graph::TensorAttributes>& tensor,
                       const std::vector<int64_t>& expected,
                       const std::string& op,
                       const char* name)
{
    HIPDNN_RETURN_IF_NE(tensor->get_dim(),
                        expected,
                        ErrorCode::INVALID_VALUE,
                        op + ": " + name + " must be " + dimsToString(expected) + ", got "
                            + dimsToString(tensor->get_dim()));
    return {};
}

// Validates the inputs and settings of a GDN (perChannelGate = false) or KDA
// (perChannelGate = true) node and, on success, reports the derived shape.
template <typename Attributes>
Error validateGatedDeltaRule(const Attributes& attrs,
                             bool perChannelGate,
                             const std::string& op,
                             GatedDeltaRuleShape& shape)
{
    const auto q = attrs.get_q();
    const auto k = attrs.get_k();
    const auto v = attrs.get_v();
    const auto g = attrs.get_g();
    const auto beta = attrs.get_beta();
    const auto cuSeqlens = attrs.get_cu_seqlens();
    for(const auto& [tensor, name] : {std::pair{q, "q"},
                                      std::pair{k, "k"},
                                      std::pair{v, "v"},
                                      std::pair{g, "g"},
                                      std::pair{beta, "beta"},
                                      std::pair{cuSeqlens, "cu_seqlens"}})
    {
        HIPDNN_RETURN_IF_FALSE(tensor, ErrorCode::ATTRIBUTE_NOT_SET, op + " missing " + name);
        HIPDNN_RETURN_IF_TRUE(tensor->get_dim().empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              op + " missing " + name + " dimensions");
    }
    HIPDNN_RETURN_IF_FALSE(attrs.get_o(), ErrorCode::ATTRIBUTE_NOT_SET, op + " missing o");

    constexpr size_t TOKEN_PACKED_RANK = 3;
    for(const auto& [tensor, name] : {std::pair{q, "q"}, std::pair{k, "k"}, std::pair{v, "v"}})
    {
        HIPDNN_RETURN_IF_NE(tensor->get_dim().size(),
                            TOKEN_PACKED_RANK,
                            ErrorCode::INVALID_VALUE,
                            op + ": " + name + " must be rank 3 ([T, heads, dim], token-packed)");
    }

    const auto& qDims = q->get_dim();
    const auto& kDims = k->get_dim();
    const auto& vDims = v->get_dim();
    shape.prefix.assign(qDims.begin(), qDims.end() - 2);
    shape.heads = qDims[TOKEN_PACKED_RANK - 2];
    shape.keyDim = qDims[TOKEN_PACKED_RANK - 1];
    const int64_t keyHeads = kDims[TOKEN_PACKED_RANK - 2];
    shape.valueHeads = vDims[TOKEN_PACKED_RANK - 2];
    shape.valueDim = vDims[TOKEN_PACKED_RANK - 1];
    shape.outputHeads = std::max(shape.heads, shape.valueHeads);

    for(const auto dim : qDims)
    {
        HIPDNN_RETURN_IF_TRUE(dim <= 0, ErrorCode::INVALID_VALUE, op + ": q dims must be positive");
    }
    HIPDNN_CHECK_ERROR(checkDims(k, withSuffix(shape.prefix, {keyHeads, shape.keyDim}), op, "k"));
    HIPDNN_RETURN_IF_TRUE(keyHeads <= 0 || shape.valueHeads <= 0 || shape.valueDim <= 0,
                          ErrorCode::INVALID_VALUE,
                          op + ": k and v dims must be positive");
    HIPDNN_CHECK_ERROR(
        checkDims(v, withSuffix(shape.prefix, {shape.valueHeads, shape.valueDim}), op, "v"));
    HIPDNN_RETURN_IF_TRUE(keyHeads != shape.heads && keyHeads != shape.valueHeads,
                          ErrorCode::INVALID_VALUE,
                          op + ": k heads must equal q heads or v heads");
    HIPDNN_RETURN_IF_NE(shape.outputHeads % std::min(shape.heads, shape.valueHeads),
                        0,
                        ErrorCode::INVALID_VALUE,
                        op + ": q heads and v heads must divide one another");

    const auto gateDims = perChannelGate
                              ? withSuffix(shape.prefix, {shape.outputHeads, shape.keyDim})
                              : withSuffix(shape.prefix, {shape.outputHeads});
    HIPDNN_CHECK_ERROR(checkDims(g, gateDims, op, "g"));
    HIPDNN_CHECK_ERROR(checkDims(beta, withSuffix(shape.prefix, {shape.outputHeads}), op, "beta"));

    HIPDNN_RETURN_IF_TRUE(cuSeqlens->get_dim().size() != 1 || cuSeqlens->get_dim()[0] < 2,
                          ErrorCode::INVALID_VALUE,
                          op + ": cu_seqlens must be [N + 1] with N >= 1");
    shape.numSequences = cuSeqlens->get_dim()[0] - 1;
    shape.totalTokens = shape.prefix[0];

    const auto stateIndices = attrs.get_state_indices();
    const auto initialState = attrs.get_initial_state();
    if(stateIndices)
    {
        HIPDNN_RETURN_IF_FALSE(initialState,
                               ErrorCode::INVALID_VALUE,
                               op + ": state_indices requires initial_state (the state pool)");
        HIPDNN_CHECK_ERROR(checkDims(stateIndices, {shape.numSequences}, op, "state_indices"));
    }
    if(initialState)
    {
        const auto& stateDims = initialState->get_dim();
        HIPDNN_RETURN_IF_TRUE(stateDims.empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              op + " missing initial_state dimensions");
        const int64_t rows = stateIndices ? stateDims[0] : shape.numSequences;
        HIPDNN_CHECK_ERROR(checkDims(initialState,
                                     {rows, shape.outputHeads, shape.valueDim, shape.keyDim},
                                     op,
                                     "initial_state"));
    }

    const auto aLog = attrs.get_a_log();
    const auto dtBias = attrs.get_dt_bias();
    HIPDNN_RETURN_IF_TRUE((aLog || dtBias) && !attrs.get_safe_gate(),
                          ErrorCode::INVALID_VALUE,
                          op + ": a_log and dt_bias require safe_gate");
    if(aLog)
    {
        HIPDNN_CHECK_ERROR(checkDims(aLog, {shape.outputHeads}, op, "a_log"));
    }
    if(dtBias)
    {
        const std::vector<int64_t> biasDims
            = perChannelGate ? std::vector<int64_t>{shape.outputHeads, shape.keyDim}
                             : std::vector<int64_t>{shape.outputHeads};
        HIPDNN_CHECK_ERROR(checkDims(dtBias, biasDims, op, "dt_bias"));
    }

    HIPDNN_RETURN_IF_TRUE(attrs.get_safe_gate() && attrs.get_gate_domain() == GateDomain::LINEAR,
                          ErrorCode::INVALID_VALUE,
                          op + ": safe_gate cannot be combined with gate_domain LINEAR");
    HIPDNN_RETURN_IF_TRUE(attrs.get_allow_neg_eigval() && !attrs.get_use_beta_sigmoid(),
                          ErrorCode::INVALID_VALUE,
                          op + ": allow_neg_eigval requires use_beta_sigmoid");
    HIPDNN_RETURN_IF_TRUE(attrs.get_overwrite_initial_state()
                              && (!initialState || !attrs.get_final_state()),
                          ErrorCode::INVALID_VALUE,
                          op + ": overwrite_initial_state requires initial_state and final_state");
    HIPDNN_RETURN_IF_TRUE(attrs.get_checkpoint_every_n_tokens() < 0,
                          ErrorCode::INVALID_VALUE,
                          op + ": checkpoint_every_n_tokens must be >= 0");
    HIPDNN_RETURN_IF_TRUE(attrs.get_output_final_state() != (attrs.get_final_state() != nullptr),
                          ErrorCode::INVALID_VALUE,
                          op + ": output_final_state and the final_state tensor must go together");
    HIPDNN_RETURN_IF_TRUE(
        (attrs.get_checkpoint_every_n_tokens() > 0) != (attrs.get_state_checkpoints() != nullptr),
        ErrorCode::INVALID_VALUE,
        op + ": checkpoint_every_n_tokens > 0 and the state_checkpoints tensor must go together");
    HIPDNN_RETURN_IF_TRUE(stateIndices && attrs.get_checkpoint_every_n_tokens() > 0,
                          ErrorCode::INVALID_VALUE,
                          op + ": state_indices cannot be combined with checkpoint_every_n_tokens");
    return {};
}

inline Error inferTensor(const std::shared_ptr<graph::TensorAttributes>& tensor,
                         const std::vector<int64_t>& dims,
                         const std::string& op,
                         const char* name)
{
    if(tensor->get_dim().empty())
    {
        tensor->set_dim(dims);
    }
    else
    {
        HIPDNN_CHECK_ERROR(checkDims(tensor, dims, op, name));
    }
    if(tensor->get_stride().empty())
    {
        tensor->set_stride(hipdnn_data_sdk::utilities::generateStrides(tensor->get_dim()));
    }
    return {};
}

// Infers output dims (and contiguous strides when unset) and state dtypes. States default to
// the initial state's dtype, else fp32, matching cuDNN; o and state_checkpoints use the I/O type.
template <typename Attributes>
Error inferGatedDeltaRuleProperties(Attributes& attrs,
                                    bool perChannelGate,
                                    const graph::GraphAttributes& graphAttrs,
                                    const std::string& op)
{
    const auto o = attrs.get_o();
    HIPDNN_RETURN_IF_FALSE(o, ErrorCode::ATTRIBUTE_NOT_SET, op + " missing o");

    GatedDeltaRuleShape shape;
    HIPDNN_CHECK_ERROR(validateGatedDeltaRule(attrs, perChannelGate, op, shape));

    const auto initialState = attrs.get_initial_state();
    const auto finalState = attrs.get_final_state();
    if(finalState && finalState->get_data_type() == DataType::NOT_SET)
    {
        const bool hasStateType
            = initialState && initialState->get_data_type() != DataType::NOT_SET;
        finalState->set_data_type(hasStateType ? initialState->get_data_type() : DataType::FLOAT);
    }
    HIPDNN_CHECK_ERROR(attrs.fill_from_context(graphAttrs));

    HIPDNN_CHECK_ERROR(
        inferTensor(o, withSuffix(shape.prefix, {shape.outputHeads, shape.valueDim}), op, "o"));
    if(finalState)
    {
        const bool pooled = attrs.get_state_indices() != nullptr;
        const int64_t rows = pooled ? initialState->get_dim()[0] : shape.numSequences;
        HIPDNN_CHECK_ERROR(inferTensor(finalState,
                                       {rows, shape.outputHeads, shape.valueDim, shape.keyDim},
                                       op,
                                       "final_state"));
    }
    if(const auto checkpoints = attrs.get_state_checkpoints())
    {
        const int64_t every = attrs.get_checkpoint_every_n_tokens();
        HIPDNN_RETURN_IF_TRUE(every <= 0,
                              ErrorCode::INVALID_VALUE,
                              op + ": state_checkpoints requires checkpoint_every_n_tokens > 0");
        // Capacity bound used by cuDNN (_pygraph.py): one row per `every` tokens plus one
        // partial-chunk row per sequence.
        const int64_t rows = std::max<int64_t>(shape.totalTokens / every + shape.numSequences, 1);
        HIPDNN_CHECK_ERROR(inferTensor(checkpoints,
                                       {rows, shape.outputHeads, shape.valueDim, shape.keyDim},
                                       op,
                                       "state_checkpoints"));
    }
    return {};
}

} // namespace hipdnn_frontend::detail
