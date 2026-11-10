// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/linear_attention_common_generated.h>

namespace hipdnn_test_sdk::utilities
{

/// Scalar attributes of a GDN node (gdn_attributes.fbs).
///
/// batch_invariant and overwrite_initial_state have no field: the reference processes every
/// sequence on its own (so its results never depend on the batch) and reads every initial state
/// before it writes any final state (so final_state may alias initial_state's buffer).
struct GdnConfig
{
    using GateDomain = hipdnn_flatbuffers_sdk::data_objects::GateDomain;

    std::optional<float> scale; ///< Unset: 1 / sqrt(K)
    bool useQkL2norm = false;
    int64_t checkpointEveryNTokens = 0;
    bool useBetaSigmoid = false;
    bool allowNegEigval = false;
    bool safeGate = false;
    GateDomain gateDomain = GateDomain::LOG;
};

/// Operands of one GDN node. q, k, v and o share the io type; every other floating operand
/// (gates, states, checkpoints) has AuxType. Optional operands are null when absent.
template <class IoType,
          class AuxType = float,
          class SeqlensType = int32_t,
          class IndexType = int32_t>
struct GdnTensors
{
    template <class T>
    using TensorBase = hipdnn_data_sdk::utilities::TensorBase<T>;

    const TensorBase<IoType>* q = nullptr; ///< [T, H, K]
    const TensorBase<IoType>* k = nullptr; ///< [T, HK, K], HK in {H, HV}
    const TensorBase<IoType>* v = nullptr; ///< [T, HV, V]
    const TensorBase<AuxType>* g = nullptr; ///< [T, HO]
    const TensorBase<AuxType>* beta = nullptr; ///< [T, HO]
    const TensorBase<SeqlensType>* cuSeqlens = nullptr; ///< [N + 1]
    TensorBase<IoType>* o = nullptr; ///< [T, HO, V]
    const TensorBase<AuxType>* initialState = nullptr; ///< [N or P, HO, V, K]
    const TensorBase<AuxType>* aLog = nullptr; ///< [HO]
    const TensorBase<AuxType>* dtBias = nullptr; ///< [HO]
    const TensorBase<IndexType>* stateIndices = nullptr; ///< [N]
    TensorBase<AuxType>* finalState = nullptr; ///< [N or P, HO, V, K]
    TensorBase<AuxType>* stateCheckpoints = nullptr; ///< [C, HO, V, K]
};

/// CPU reference for Gated DeltaNet forward (gdn_attributes.fbs; cuDNN graph.gdn,
/// cudnn-frontend test/python/linear_attention/reference_gdn.py).
///
/// Token-packed: N sequences with offsets cuSeqlens (0 = cu[0] <= ... <= cu[N] = T). Output
/// head h of HO = max(H, HV) reads q head h / (HO / H), k head h / (HO / HK) and v head
/// h / (HO / HV) (heads replicated onto HO, as cuDNN's grouped-head reference). Per sequence,
/// head and token t, with state S [V, K] held in ComputeType (FP32 by default):
///   S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
///       = alpha_t S_{t-1} + beta_t (v_t - alpha_t S_{t-1} k_t) k_t^T
///   o_t = scale S_t q_t
/// - use_qk_l2norm: q_t and k_t become x * rsqrt(sum(x^2) + 1e-6) (cuDNN L2_NORM_EPS) first;
///   scale (default 1 / sqrt(K)) multiplies the normalized q.
/// - alpha_t: exp(g) (LOG), max(g, 1e-10) (LINEAR; the cuDNN in-kernel floor), or
///   exp(-exp(a_log[h]) * softplus(g + dt_bias[h])) with safe_gate (a_log absent: unit
///   amplitude; dt_bias absent: zero).
/// - beta_t: beta, sigmoid(beta) (use_beta_sigmoid), or 2 * sigmoid(beta) (allow_neg_eigval).
/// - State rows: initial_state / final_state row n, or row stateIndices[n] of a pool. Absent
///   initial_state is a zero state; an empty sequence's final state is its initial state.
/// - Checkpoints (n = checkpoint_every_n_tokens > 0): a non-empty sequence of length L stores
///   (L - 1) / n + 1 rows, row j being the state after its first j * n tokens (row 0 = the
///   incoming state); rows pack by sequence in order and rows past the last are untouched.
/// batch_invariant and overwrite_initial_state do not change the result (see GdnConfig).
class CpuFpReferenceGdn
{
public:
    using GateDomain = hipdnn_flatbuffers_sdk::data_objects::GateDomain;

    static constexpr double L2_NORM_EPS = 1e-6;
    static constexpr double LINEAR_GATE_FLOOR = 1e-10;

    template <class ComputeType = float,
              class IoType,
              class AuxType,
              class SeqlensType,
              class IndexType>
    static void forward(const GdnTensors<IoType, AuxType, SeqlensType, IndexType>& t,
                        const GdnConfig& config)
    {
        const auto shape = validate(t, config);
        const auto offsets = readOffsets(*t.cuSeqlens, shape.sequences, shape.tokens);
        const auto rows = stateRows(t, shape);

        const int64_t stateSize = shape.valueDim * shape.keyDim;
        const auto stateOffset = [&](int64_t n, int64_t h) {
            return static_cast<size_t>((n * shape.headsOut + h) * stateSize);
        };

        // Read every incoming state before anything is written: finalState may alias it.
        std::vector<ComputeType> states(
            static_cast<size_t>(shape.sequences * shape.headsOut * stateSize), ComputeType(0));
        if(t.initialState != nullptr)
        {
            for(int64_t n = 0; n < shape.sequences; ++n)
            {
                for(int64_t h = 0; h < shape.headsOut; ++h)
                {
                    auto* state = &states[stateOffset(n, h)];
                    for(int64_t i = 0; i < shape.valueDim; ++i)
                    {
                        for(int64_t j = 0; j < shape.keyDim; ++j)
                        {
                            state[i * shape.keyDim + j]
                                = static_cast<ComputeType>(t.initialState->getHostValue(
                                    rows[static_cast<size_t>(n)], h, i, j));
                        }
                    }
                }
            }
        }

        const auto scale
            = config.scale.has_value()
                  ? static_cast<ComputeType>(*config.scale)
                  : static_cast<ComputeType>(1.0 / std::sqrt(static_cast<double>(shape.keyDim)));
        const int64_t every = config.checkpointEveryNTokens;

        std::vector<ComputeType> query(static_cast<size_t>(shape.keyDim));
        std::vector<ComputeType> key(static_cast<size_t>(shape.keyDim));
        std::vector<ComputeType> keyState(static_cast<size_t>(shape.valueDim));

        int64_t checkpointBase = 0;
        for(int64_t n = 0; n < shape.sequences; ++n)
        {
            const int64_t start = offsets[static_cast<size_t>(n)];
            const int64_t length = offsets[static_cast<size_t>(n) + 1] - start;
            for(int64_t h = 0; h < shape.headsOut; ++h)
            {
                const int64_t qHead = h / (shape.headsOut / shape.headsQ);
                const int64_t kHead = h / (shape.headsOut / shape.headsK);
                const int64_t vHead = h / (shape.headsOut / shape.headsV);
                auto* state = &states[stateOffset(n, h)];

                for(int64_t step = 0; step < length; ++step)
                {
                    const int64_t token = start + step;
                    if(t.stateCheckpoints != nullptr && step % every == 0)
                    {
                        storeState(
                            *t.stateCheckpoints, checkpointBase + step / every, h, state, shape);
                    }

                    for(int64_t j = 0; j < shape.keyDim; ++j)
                    {
                        query[static_cast<size_t>(j)]
                            = static_cast<ComputeType>(t.q->getHostValue(token, qHead, j));
                        key[static_cast<size_t>(j)]
                            = static_cast<ComputeType>(t.k->getHostValue(token, kHead, j));
                    }
                    if(config.useQkL2norm)
                    {
                        l2Normalize(query);
                        l2Normalize(key);
                    }

                    const auto alpha = decay<ComputeType>(t, config, token, h);
                    const auto betaValue = writeStrength<ComputeType>(t, config, token, h);

                    // keyState = S k; S = alpha S + beta (v - alpha S k) k^T.
                    for(int64_t i = 0; i < shape.valueDim; ++i)
                    {
                        auto sum = ComputeType(0);
                        for(int64_t j = 0; j < shape.keyDim; ++j)
                        {
                            sum += state[i * shape.keyDim + j] * key[static_cast<size_t>(j)];
                        }
                        keyState[static_cast<size_t>(i)] = sum;
                    }
                    for(int64_t i = 0; i < shape.valueDim; ++i)
                    {
                        const auto value
                            = static_cast<ComputeType>(t.v->getHostValue(token, vHead, i));
                        const auto update
                            = betaValue * (value - alpha * keyState[static_cast<size_t>(i)]);
                        for(int64_t j = 0; j < shape.keyDim; ++j)
                        {
                            auto& entry = state[i * shape.keyDim + j];
                            entry = alpha * entry + update * key[static_cast<size_t>(j)];
                        }
                    }

                    // o = scale S q.
                    for(int64_t i = 0; i < shape.valueDim; ++i)
                    {
                        auto sum = ComputeType(0);
                        for(int64_t j = 0; j < shape.keyDim; ++j)
                        {
                            sum += state[i * shape.keyDim + j] * query[static_cast<size_t>(j)];
                        }
                        t.o->setHostValue(static_cast<IoType>(scale * sum), token, h, i);
                    }
                }
            }
            if(t.stateCheckpoints != nullptr && length > 0)
            {
                checkpointBase += (length - 1) / every + 1;
            }
        }
        t.o->memory().markHostModified();

        if(t.stateCheckpoints != nullptr)
        {
            t.stateCheckpoints->memory().markHostModified();
        }

        if(t.finalState != nullptr)
        {
            for(int64_t n = 0; n < shape.sequences; ++n)
            {
                for(int64_t h = 0; h < shape.headsOut; ++h)
                {
                    storeState(*t.finalState,
                               rows[static_cast<size_t>(n)],
                               h,
                               &states[stateOffset(n, h)],
                               shape);
                }
            }
            t.finalState->memory().markHostModified();
        }
    }

    /// Rows allocated for state_checkpoints: C = max(T / n + N, 1).
    static int64_t checkpointCapacity(int64_t tokens, int64_t sequences, int64_t every)
    {
        return std::max(tokens / every + sequences, int64_t{1});
    }

private:
    struct Shape
    {
        int64_t tokens = 0;
        int64_t sequences = 0;
        int64_t headsQ = 0;
        int64_t headsK = 0;
        int64_t headsV = 0;
        int64_t headsOut = 0;
        int64_t keyDim = 0;
        int64_t valueDim = 0;
        int64_t stateRowCount = 0;
    };

    template <class ComputeType>
    static void l2Normalize(std::vector<ComputeType>& row)
    {
        auto sumSquares = ComputeType(0);
        for(const auto value : row)
        {
            sumSquares += value * value;
        }
        const auto inverseNorm
            = ComputeType(1) / std::sqrt(sumSquares + static_cast<ComputeType>(L2_NORM_EPS));
        for(auto& value : row)
        {
            value *= inverseNorm;
        }
    }

    template <class ComputeType>
    static ComputeType softplus(ComputeType value)
    {
        // torch.nn.functional.softplus: linear above the threshold 20.
        return value > ComputeType(20) ? value : std::log1p(std::exp(value));
    }

    template <class ComputeType>
    static ComputeType sigmoid(ComputeType value)
    {
        return ComputeType(1) / (ComputeType(1) + std::exp(-value));
    }

    template <class ComputeType, class Tensors>
    static ComputeType decay(const Tensors& t, const GdnConfig& config, int64_t token, int64_t head)
    {
        const auto gate = static_cast<ComputeType>(t.g->getHostValue(token, head));
        if(config.safeGate)
        {
            const auto amplitude
                = t.aLog != nullptr ? std::exp(static_cast<ComputeType>(t.aLog->getHostValue(head)))
                                    : ComputeType(1);
            const auto bias = t.dtBias != nullptr
                                  ? static_cast<ComputeType>(t.dtBias->getHostValue(head))
                                  : ComputeType(0);
            return std::exp(-amplitude * softplus(gate + bias));
        }
        if(config.gateDomain == GateDomain::LINEAR)
        {
            return std::max(gate, static_cast<ComputeType>(LINEAR_GATE_FLOOR));
        }
        return std::exp(gate);
    }

    template <class ComputeType, class Tensors>
    static ComputeType
        writeStrength(const Tensors& t, const GdnConfig& config, int64_t token, int64_t head)
    {
        auto value = static_cast<ComputeType>(t.beta->getHostValue(token, head));
        if(config.useBetaSigmoid)
        {
            value = sigmoid(value);
            if(config.allowNegEigval)
            {
                value *= ComputeType(2);
            }
        }
        return value;
    }

    template <class AuxType, class ComputeType>
    static void storeState(hipdnn_data_sdk::utilities::TensorBase<AuxType>& target,
                           int64_t row,
                           int64_t head,
                           const ComputeType* state,
                           const Shape& shape)
    {
        for(int64_t i = 0; i < shape.valueDim; ++i)
        {
            for(int64_t j = 0; j < shape.keyDim; ++j)
            {
                target.setHostValue(
                    static_cast<AuxType>(state[i * shape.keyDim + j]), row, head, i, j);
            }
        }
    }

    template <class SeqlensType>
    static std::vector<int64_t>
        readOffsets(const hipdnn_data_sdk::utilities::TensorBase<SeqlensType>& cuSeqlens,
                    int64_t sequences,
                    int64_t tokens)
    {
        std::vector<int64_t> offsets(static_cast<size_t>(sequences) + 1);
        for(int64_t n = 0; n <= sequences; ++n)
        {
            offsets[static_cast<size_t>(n)] = static_cast<int64_t>(cuSeqlens.getHostValue(n));
        }
        if(offsets.front() != 0 || offsets.back() != tokens)
        {
            throw std::invalid_argument(
                "CpuFpReferenceGdn: cu_seqlens must start at 0 and end at T");
        }
        for(size_t n = 1; n < offsets.size(); ++n)
        {
            if(offsets[n] < offsets[n - 1])
            {
                throw std::invalid_argument("CpuFpReferenceGdn: cu_seqlens must be non-decreasing");
            }
        }
        return offsets;
    }

    /// State row of each sequence: stateIndices[n] when given, else n. Rows must lie in
    /// [0, stateRowCount), and rows written to finalState must be distinct.
    template <class Tensors>
    static std::vector<int64_t> stateRows(const Tensors& t, const Shape& shape)
    {
        std::vector<int64_t> rows(static_cast<size_t>(shape.sequences));
        for(int64_t n = 0; n < shape.sequences; ++n)
        {
            rows[static_cast<size_t>(n)]
                = t.stateIndices != nullptr ? static_cast<int64_t>(t.stateIndices->getHostValue(n))
                                            : n;
        }
        std::vector<bool> written(static_cast<size_t>(shape.stateRowCount), false);
        for(const auto row : rows)
        {
            if(row < 0 || row >= shape.stateRowCount)
            {
                throw std::out_of_range("CpuFpReferenceGdn: state row " + std::to_string(row)
                                        + " is outside the " + std::to_string(shape.stateRowCount)
                                        + " state rows");
            }
            if(t.finalState != nullptr)
            {
                if(written[static_cast<size_t>(row)])
                {
                    throw std::invalid_argument("CpuFpReferenceGdn: state row "
                                                + std::to_string(row)
                                                + " is written by two sequences");
                }
                written[static_cast<size_t>(row)] = true;
            }
        }
        return rows;
    }

    static void require(bool condition, const char* message)
    {
        if(!condition)
        {
            throw std::invalid_argument(std::string("CpuFpReferenceGdn: ") + message);
        }
    }

    template <class Tensors>
    static Shape validate(const Tensors& t, const GdnConfig& config)
    {
        require(t.q != nullptr && t.k != nullptr && t.v != nullptr && t.g != nullptr
                    && t.beta != nullptr && t.cuSeqlens != nullptr && t.o != nullptr,
                "q, k, v, g, beta, cu_seqlens and o are required");

        const auto& qDims = t.q->dims();
        const auto& kDims = t.k->dims();
        const auto& vDims = t.v->dims();
        require(qDims.size() == 3 && kDims.size() == 3 && vDims.size() == 3,
                "q, k and v must be [T, heads, dim]");

        Shape shape;
        shape.tokens = qDims[0];
        shape.headsQ = qDims[1];
        shape.keyDim = qDims[2];
        shape.headsK = kDims[1];
        shape.headsV = vDims[1];
        shape.valueDim = vDims[2];
        shape.headsOut = std::max(shape.headsQ, shape.headsV);

        require(shape.headsQ > 0 && shape.headsV > 0 && shape.keyDim > 0 && shape.valueDim > 0,
                "head counts and head dims must be positive");
        require(kDims[0] == shape.tokens && vDims[0] == shape.tokens,
                "q, k and v must have the same token count");
        require(kDims[2] == shape.keyDim, "k head dim must equal q's");
        require(shape.headsK == shape.headsQ || shape.headsK == shape.headsV,
                "k head count must be H or HV");
        require(shape.headsOut % shape.headsQ == 0 && shape.headsOut % shape.headsV == 0,
                "q and v head counts must divide HO = max(H, HV)");

        const std::vector<int64_t> perHead{shape.tokens, shape.headsOut};
        require(t.g->dims() == perHead && t.beta->dims() == perHead, "g and beta must be [T, HO]");
        require(t.o->dims() == std::vector<int64_t>{shape.tokens, shape.headsOut, shape.valueDim},
                "o must be [T, HO, V]");

        const auto& cuDims = t.cuSeqlens->dims();
        require(cuDims.size() == 1 && cuDims[0] >= 2, "cu_seqlens must be [N + 1] with N >= 1");
        shape.sequences = cuDims[0] - 1;

        require(!(config.safeGate && config.gateDomain == GateDomain::LINEAR),
                "gate_domain LINEAR cannot combine with safe_gate");
        require((t.aLog == nullptr && t.dtBias == nullptr) || config.safeGate,
                "a_log and dt_bias require safe_gate");
        require(!config.allowNegEigval || config.useBetaSigmoid,
                "allow_neg_eigval requires use_beta_sigmoid");
        const std::vector<int64_t> headsOnly{shape.headsOut};
        require(t.aLog == nullptr || t.aLog->dims() == headsOnly, "a_log must be [HO]");
        require(t.dtBias == nullptr || t.dtBias->dims() == headsOnly, "dt_bias must be [HO]");

        if(t.stateIndices != nullptr)
        {
            require(t.initialState != nullptr, "state_indices requires initial_state");
            require(t.stateIndices->dims() == std::vector<int64_t>{shape.sequences},
                    "state_indices must be [N]");
            require(config.checkpointEveryNTokens == 0,
                    "state_indices cannot combine with checkpoint_every_n_tokens");
        }

        const auto checkState = [&](const auto* state) {
            if(state == nullptr)
            {
                return;
            }
            const auto& dims = state->dims();
            require(dims.size() == 4 && dims[1] == shape.headsOut && dims[2] == shape.valueDim
                        && dims[3] == shape.keyDim,
                    "states must be [N or P, HO, V, K]");
            require(t.stateIndices != nullptr ? dims[0] >= 1 : dims[0] == shape.sequences,
                    "states must have N rows, or P >= 1 rows with state_indices");
        };
        checkState(t.initialState);
        checkState(t.finalState);
        if(t.initialState != nullptr && t.finalState != nullptr)
        {
            require(t.finalState->dims()[0] == t.initialState->dims()[0],
                    "final_state must have as many rows as initial_state");
        }
        shape.stateRowCount = t.initialState != nullptr ? t.initialState->dims()[0]
                              : t.finalState != nullptr ? t.finalState->dims()[0]
                                                        : shape.sequences;

        require(config.checkpointEveryNTokens >= 0, "checkpoint_every_n_tokens must be >= 0");
        require((config.checkpointEveryNTokens > 0) == (t.stateCheckpoints != nullptr),
                "state_checkpoints must be given exactly when checkpoint_every_n_tokens > 0");
        if(t.stateCheckpoints != nullptr)
        {
            const auto& dims = t.stateCheckpoints->dims();
            require(dims.size() == 4 && dims[1] == shape.headsOut && dims[2] == shape.valueDim
                        && dims[3] == shape.keyDim
                        && dims[0] >= checkpointCapacity(
                               shape.tokens, shape.sequences, config.checkpointEveryNTokens),
                    "state_checkpoints must be [C >= max(T / n + N, 1), HO, V, K]");
        }
        return shape;
    }
};

} // namespace hipdnn_test_sdk::utilities
