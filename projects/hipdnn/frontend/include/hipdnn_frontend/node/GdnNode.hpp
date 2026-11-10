// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GdnAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/detail/GdnPacker.hpp>
#include <hipdnn_frontend/detail/GdnUnpacker.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace hipdnn_frontend::graph
{
// Gated DeltaNet forward. Validation mirrors cuDNN's graph.gdn / gated_delta_net
// (cudnn-frontend python/cudnn/linear_attention/ops/gdn.py) and the cross-field rules
// documented in gdn_attributes.fbs.
class GdnNode : public BaseNode<GdnNode, NodeType::GDN>
{
public:
    GdnAttributes attributes;

    GdnNode(GdnAttributes&& attrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(attrs))
    {
    }

    Error unpack_from_descriptor(
        hipdnnBackendDescriptor_t opDesc,
        std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap) override
    {
        GdnAttributes attrs;
        HIPDNN_CHECK_ERROR(detail::unpackGdnOperation(opDesc, tensorMap, attrs));
        attributes = std::move(attrs);
        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error pre_validate_node() const override
    {
        const auto q = attributes.get_q();
        const auto k = attributes.get_k();
        const auto v = attributes.get_v();
        const auto g = attributes.get_g();
        const auto beta = attributes.get_beta();
        const auto cuSeqlens = attributes.get_cu_seqlens();
        const auto o = attributes.get_o();
        const auto initialState = attributes.get_initial_state();
        const auto aLog = attributes.get_a_log();
        const auto dtBias = attributes.get_dt_bias();
        const auto stateIndices = attributes.get_state_indices();
        const auto finalState = attributes.get_final_state();
        const auto stateCheckpoints = attributes.get_state_checkpoints();

        HIPDNN_RETURN_IF_FALSE(q, ErrorCode::ATTRIBUTE_NOT_SET, "GdnNode missing q (input)");
        HIPDNN_RETURN_IF_FALSE(k, ErrorCode::ATTRIBUTE_NOT_SET, "GdnNode missing k (input)");
        HIPDNN_RETURN_IF_FALSE(v, ErrorCode::ATTRIBUTE_NOT_SET, "GdnNode missing v (input)");
        HIPDNN_RETURN_IF_FALSE(g, ErrorCode::ATTRIBUTE_NOT_SET, "GdnNode missing g (input)");
        HIPDNN_RETURN_IF_FALSE(beta, ErrorCode::ATTRIBUTE_NOT_SET, "GdnNode missing beta (input)");
        HIPDNN_RETURN_IF_FALSE(
            cuSeqlens, ErrorCode::ATTRIBUTE_NOT_SET, "GdnNode missing cu_seqlens (input)");
        HIPDNN_RETURN_IF_FALSE(o, ErrorCode::ATTRIBUTE_NOT_SET, "GdnNode missing o (output)");

        // Ranks: q/k/v [T, heads, dim], g/beta [T, HO], cu_seqlens [N + 1].
        HIPDNN_RETURN_IF_NE(
            q->get_dim().size(), 3u, ErrorCode::INVALID_VALUE, "Gdn q must have shape [T, H, K]");
        HIPDNN_RETURN_IF_NE(
            k->get_dim().size(), 3u, ErrorCode::INVALID_VALUE, "Gdn k must have shape [T, HK, K]");
        HIPDNN_RETURN_IF_NE(
            v->get_dim().size(), 3u, ErrorCode::INVALID_VALUE, "Gdn v must have shape [T, HV, V]");
        HIPDNN_RETURN_IF_NE(
            g->get_dim().size(), 2u, ErrorCode::INVALID_VALUE, "Gdn g must have shape [T, HO]");
        HIPDNN_RETURN_IF_NE(beta->get_dim().size(),
                            2u,
                            ErrorCode::INVALID_VALUE,
                            "Gdn beta must have shape [T, HO]");
        HIPDNN_RETURN_IF_NE(cuSeqlens->get_dim().size(),
                            1u,
                            ErrorCode::INVALID_VALUE,
                            "Gdn cu_seqlens must have shape [N + 1]");

        const auto tokens = q->get_dim()[0];
        const auto numHeadsQ = q->get_dim()[1];
        const auto headDimK = q->get_dim()[2];
        const auto numHeadsK = k->get_dim()[1];
        const auto numHeadsV = v->get_dim()[1];
        const auto headDimV = v->get_dim()[2];
        const auto numHeadsOut = std::max(numHeadsQ, numHeadsV);
        const auto numSequences = cuSeqlens->get_dim()[0] - 1;

        HIPDNN_RETURN_IF_TRUE(tokens < 1 || numHeadsQ < 1 || headDimK < 1 || numHeadsK < 1
                                  || numHeadsV < 1 || headDimV < 1,
                              ErrorCode::INVALID_VALUE,
                              "Gdn q, k and v dimensions must be positive");
        HIPDNN_RETURN_IF_NE(
            k->get_dim()[0], tokens, ErrorCode::INVALID_VALUE, "Gdn k token count must match q");
        HIPDNN_RETURN_IF_NE(
            v->get_dim()[0], tokens, ErrorCode::INVALID_VALUE, "Gdn v token count must match q");
        HIPDNN_RETURN_IF_NE(k->get_dim()[2],
                            headDimK,
                            ErrorCode::INVALID_VALUE,
                            "Gdn k head dim must match q head dim K");
        HIPDNN_RETURN_IF_TRUE(numHeadsK != numHeadsQ && numHeadsK != numHeadsV,
                              ErrorCode::INVALID_VALUE,
                              "Gdn k head count must match q's (H) or v's (HV)");
        HIPDNN_RETURN_IF_TRUE(numHeadsOut % numHeadsQ != 0 || numHeadsOut % numHeadsV != 0,
                              ErrorCode::INVALID_VALUE,
                              "Gdn q and v head counts must divide HO = max(H, HV)");
        HIPDNN_RETURN_IF_NE(g->get_dim(),
                            (std::vector<int64_t>{tokens, numHeadsOut}),
                            ErrorCode::INVALID_VALUE,
                            "Gdn g must have shape [T, HO] with HO = max(H, HV)");
        HIPDNN_RETURN_IF_NE(beta->get_dim(),
                            (std::vector<int64_t>{tokens, numHeadsOut}),
                            ErrorCode::INVALID_VALUE,
                            "Gdn beta must have shape [T, HO] with HO = max(H, HV)");
        HIPDNN_RETURN_IF_TRUE(numSequences < 1,
                              ErrorCode::INVALID_VALUE,
                              "Gdn cu_seqlens must hold at least two offsets");
        HIPDNN_RETURN_IF_FALSE(isIndexType(cuSeqlens->get_data_type()),
                               ErrorCode::INVALID_VALUE,
                               "Gdn cu_seqlens must have INT32 or INT64 data type");

        if(!o->get_dim().empty())
        {
            HIPDNN_RETURN_IF_NE(o->get_dim(),
                                (std::vector<int64_t>{tokens, numHeadsOut, headDimV}),
                                ErrorCode::INVALID_VALUE,
                                "Gdn o must have shape [T, HO, V]");
        }

        // Gate cross rules.
        HIPDNN_RETURN_IF_TRUE(attributes.get_gate_domain() == GateDomain::LINEAR
                                  && attributes.get_safe_gate(),
                              ErrorCode::INVALID_VALUE,
                              "Gdn gate_domain LINEAR cannot combine with safe_gate");
        HIPDNN_RETURN_IF_TRUE((aLog || dtBias) && !attributes.get_safe_gate(),
                              ErrorCode::INVALID_VALUE,
                              "Gdn a_log and dt_bias require safe_gate");
        HIPDNN_RETURN_IF_TRUE(attributes.get_allow_neg_eigval()
                                  && !attributes.get_use_beta_sigmoid(),
                              ErrorCode::INVALID_VALUE,
                              "Gdn allow_neg_eigval requires use_beta_sigmoid");
        if(aLog)
        {
            HIPDNN_RETURN_IF_NE(aLog->get_dim(),
                                std::vector<int64_t>{numHeadsOut},
                                ErrorCode::INVALID_VALUE,
                                "Gdn a_log must have shape [HO]");
        }
        if(dtBias)
        {
            HIPDNN_RETURN_IF_NE(dtBias->get_dim(),
                                std::vector<int64_t>{numHeadsOut},
                                ErrorCode::INVALID_VALUE,
                                "Gdn dt_bias must have shape [HO]");
        }

        // State cross rules.
        const auto checkpointEvery = attributes.get_checkpoint_every_n_tokens();
        HIPDNN_RETURN_IF_TRUE(checkpointEvery < 0,
                              ErrorCode::INVALID_VALUE,
                              "Gdn checkpoint_every_n_tokens must not be negative");
        if(stateIndices)
        {
            HIPDNN_RETURN_IF_FALSE(
                initialState, ErrorCode::INVALID_VALUE, "Gdn state_indices requires initial_state");
            HIPDNN_RETURN_IF_NE(stateIndices->get_dim(),
                                std::vector<int64_t>{numSequences},
                                ErrorCode::INVALID_VALUE,
                                "Gdn state_indices must have shape [N]");
            HIPDNN_RETURN_IF_NE(stateIndices->get_data_type(),
                                DataType::INT32,
                                ErrorCode::INVALID_VALUE,
                                "Gdn state_indices must have INT32 data type");
            HIPDNN_RETURN_IF_TRUE(checkpointEvery > 0,
                                  ErrorCode::INVALID_VALUE,
                                  "Gdn state_indices cannot combine with "
                                  "checkpoint_every_n_tokens > 0");
        }
        HIPDNN_RETURN_IF_TRUE(attributes.get_overwrite_initial_state()
                                  && (!initialState || !finalState),
                              ErrorCode::INVALID_VALUE,
                              "Gdn overwrite_initial_state requires initial_state and final_state");
        HIPDNN_RETURN_IF_NE(attributes.get_output_final_state(),
                            finalState != nullptr,
                            ErrorCode::INVALID_VALUE,
                            "Gdn final_state must be set exactly when output_final_state is "
                            "requested");
        HIPDNN_RETURN_IF_NE(checkpointEvery > 0,
                            stateCheckpoints != nullptr,
                            ErrorCode::INVALID_VALUE,
                            "Gdn state_checkpoints must be set exactly when "
                            "checkpoint_every_n_tokens > 0");

        const std::vector<int64_t> stateTail{numHeadsOut, headDimV, headDimK};
        if(initialState)
        {
            HIPDNN_CHECK_ERROR(validateStateDims(
                initialState->get_dim(), "initial_state", stateTail, numSequences));
        }
        if(finalState && !finalState->get_dim().empty())
        {
            HIPDNN_CHECK_ERROR(
                validateStateDims(finalState->get_dim(), "final_state", stateTail, numSequences));
            if(stateIndices || attributes.get_overwrite_initial_state())
            {
                HIPDNN_RETURN_IF_NE(finalState->get_dim()[0],
                                    initialState->get_dim()[0],
                                    ErrorCode::INVALID_VALUE,
                                    "Gdn final_state must have as many rows as initial_state");
            }
        }
        if(stateCheckpoints && !stateCheckpoints->get_dim().empty())
        {
            const std::vector<int64_t> expected{
                checkpointRows(tokens, numSequences, checkpointEvery),
                numHeadsOut,
                headDimV,
                headDimK};
            HIPDNN_RETURN_IF_NE(stateCheckpoints->get_dim(),
                                expected,
                                ErrorCode::INVALID_VALUE,
                                "Gdn state_checkpoints must have shape [max(T / n + N, 1), HO, V, "
                                "K]");
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error infer_properties_node() override
    {
        const auto q = attributes.get_q();
        const auto initialState = attributes.get_initial_state();
        const auto finalState = attributes.get_final_state();
        const auto stateCheckpoints = attributes.get_state_checkpoints();

        // cuDNN: final_state follows initial_state's dtype (FLOAT without one); checkpoints
        // default to the io (q) dtype. Set before fill_from_context applies the graph default.
        if(finalState && finalState->get_data_type() == DataType::NOT_SET)
        {
            finalState->set_data_type(initialState
                                              && initialState->get_data_type() != DataType::NOT_SET
                                          ? initialState->get_data_type()
                                          : DataType::FLOAT);
        }
        if(stateCheckpoints && stateCheckpoints->get_data_type() == DataType::NOT_SET
           && q->get_data_type() != DataType::NOT_SET)
        {
            stateCheckpoints->set_data_type(q->get_data_type());
        }

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));
        HIPDNN_CHECK_ERROR(validateDataTypes());

        const auto v = attributes.get_v();
        const auto o = attributes.get_o();
        const auto tokens = q->get_dim()[0];
        const auto headDimK = q->get_dim()[2];
        const auto headDimV = v->get_dim()[2];
        const auto numHeadsOut = std::max(q->get_dim()[1], v->get_dim()[1]);
        const auto numSequences = attributes.get_cu_seqlens()->get_dim()[0] - 1;

        if(o->get_dim().empty())
        {
            o->set_dim({tokens, numHeadsOut, headDimV});
        }
        if(o->get_stride().empty())
        {
            if(v->get_stride().size() == v->get_dim().size())
            {
                const auto strideOrder
                    = hipdnn_data_sdk::utilities::extractStrideOrder(v->get_stride());
                o->set_stride(
                    hipdnn_data_sdk::utilities::generateStrides(o->get_dim(), strideOrder));
            }
            else
            {
                o->set_stride(hipdnn_data_sdk::utilities::generateStrides(o->get_dim()));
            }
        }

        if(finalState)
        {
            if(finalState->get_dim().empty())
            {
                const auto stateRows = initialState
                                               && (attributes.get_state_indices()
                                                   || attributes.get_overwrite_initial_state())
                                           ? initialState->get_dim()[0]
                                           : numSequences;
                finalState->set_dim({stateRows, numHeadsOut, headDimV, headDimK});
            }
            if(finalState->get_stride().empty())
            {
                finalState->set_stride(
                    hipdnn_data_sdk::utilities::generateStrides(finalState->get_dim()));
            }
        }

        if(stateCheckpoints)
        {
            if(stateCheckpoints->get_dim().empty())
            {
                stateCheckpoints->set_dim(
                    {checkpointRows(
                         tokens, numSequences, attributes.get_checkpoint_every_n_tokens()),
                     numHeadsOut,
                     headDimV,
                     headDimK});
            }
            if(stateCheckpoints->get_stride().empty())
            {
                stateCheckpoints->set_stride(
                    hipdnn_data_sdk::utilities::generateStrides(stateCheckpoints->get_dim()));
            }
        }

        return {};
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error create_operation(
        std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor>& tensorDescs,
        std::vector<detail::ScopedHipdnnBackendDescriptor>& operations) const override
    {
        return detail::createGdnOperation(attributes, tensorDescs, operations);
    }

    // Rows allocated for state_checkpoints: C = max(T / n + N, 1) (cuDNN run_gdn_fwd).
    static int64_t checkpointRows(int64_t tokens, int64_t numSequences, int64_t every)
    {
        return std::max(tokens / every + numSequences, int64_t{1});
    }

private:
    static bool isIndexType(DataType dataType)
    {
        return dataType == DataType::INT32 || dataType == DataType::INT64;
    }

    static bool isFloatType(DataType dataType)
    {
        return dataType == DataType::FLOAT || dataType == DataType::HALF
               || dataType == DataType::BFLOAT16;
    }

    // Float operand dtypes, as cuDNN gated_delta_net checks them: k and v match q (the io
    // dtype), beta is FLOAT or the io dtype, g / a_log / dt_bias / states are float types.
    // Run after fill_from_context so unset dtypes carry the graph default.
    Error validateDataTypes() const
    {
        const auto ioType = attributes.get_q()->get_data_type();
        HIPDNN_RETURN_IF_FALSE(isFloatType(ioType),
                               ErrorCode::INVALID_VALUE,
                               "Gdn q must have FLOAT, HALF or BFLOAT16 data type");
        HIPDNN_RETURN_IF_NE(attributes.get_k()->get_data_type(),
                            ioType,
                            ErrorCode::INVALID_VALUE,
                            "Gdn k must have the data type of q");
        HIPDNN_RETURN_IF_NE(attributes.get_v()->get_data_type(),
                            ioType,
                            ErrorCode::INVALID_VALUE,
                            "Gdn v must have the data type of q");
        const auto betaType = attributes.get_beta()->get_data_type();
        HIPDNN_RETURN_IF_TRUE(betaType != DataType::FLOAT && betaType != ioType,
                              ErrorCode::INVALID_VALUE,
                              "Gdn beta must have FLOAT data type or the data type of q");
        HIPDNN_RETURN_IF_FALSE(isFloatType(attributes.get_g()->get_data_type()),
                               ErrorCode::INVALID_VALUE,
                               "Gdn g must have FLOAT, HALF or BFLOAT16 data type");
        for(const auto& tensor : {attributes.get_a_log(),
                                  attributes.get_dt_bias(),
                                  attributes.get_initial_state(),
                                  attributes.get_final_state(),
                                  attributes.get_state_checkpoints()})
        {
            if(tensor)
            {
                HIPDNN_RETURN_IF_FALSE(isFloatType(tensor->get_data_type()),
                                       ErrorCode::INVALID_VALUE,
                                       "Gdn a_log, dt_bias and state tensors must have FLOAT, "
                                       "HALF or BFLOAT16 data type");
            }
        }
        // cuDNN: final_state comes back in initial_state's data type.
        const auto initialState = attributes.get_initial_state();
        const auto finalState = attributes.get_final_state();
        HIPDNN_RETURN_IF_TRUE(initialState && finalState
                                  && initialState->get_data_type() != finalState->get_data_type(),
                              ErrorCode::INVALID_VALUE,
                              "Gdn final_state must have the data type of initial_state");
        return {};
    }

    // A state is [N, HO, V, K], or a pool [P, HO, V, K] with P >= 1 under state_indices.
    Error validateStateDims(const std::vector<int64_t>& dims,
                            const std::string& name,
                            const std::vector<int64_t>& tail,
                            int64_t numSequences) const
    {
        HIPDNN_RETURN_IF_NE(
            dims.size(), 4u, ErrorCode::INVALID_VALUE, "Gdn " + name + " must have rank 4");
        if(attributes.get_state_indices())
        {
            HIPDNN_RETURN_IF_TRUE(dims[0] < 1,
                                  ErrorCode::INVALID_VALUE,
                                  "Gdn " + name + " pool must have at least one row");
        }
        else
        {
            HIPDNN_RETURN_IF_NE(dims[0],
                                numSequences,
                                ErrorCode::INVALID_VALUE,
                                "Gdn " + name
                                    + " leading dimension must match the sequence count N");
        }
        HIPDNN_RETURN_IF_FALSE(std::equal(tail.begin(), tail.end(), dims.begin() + 1),
                               ErrorCode::INVALID_VALUE,
                               "Gdn " + name + " must have shape [N or P, HO, V, K]");
        return {};
    }
};
} // namespace hipdnn_frontend::graph
