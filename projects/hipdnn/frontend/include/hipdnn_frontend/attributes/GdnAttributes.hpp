// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file GdnAttributes.hpp
 * @brief Attributes for Gdn operations
 *
 * This file defines the GdnAttributes class for configuring
 * Gdn operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <cstdint>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class GdnAttributes
 * @brief Configuration for Gdn operations
 */
class GdnAttributes : public Attributes<GdnAttributes>
{
public:
    GdnAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        Q = 0,
        K = 1,
        V = 2,
        G = 3,
        BETA = 4,
        CU_SEQLENS = 5,
        INITIAL_STATE = 6,
        A_LOG = 7,
        DT_BIAS = 8,
        STATE_INDICES = 9
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        O = 0,
        FINAL_STATE = 1,
        STATE_CHECKPOINTS = 2
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    std::optional<float> scale = std::nullopt; ///< Scale
    bool output_final_state = false; ///< Output Final State
    bool use_qk_l2norm = false; ///< Use Qk L2norm
    int64_t checkpoint_every_n_tokens = 0; ///< Checkpoint Every N Tokens
    bool use_beta_sigmoid = false; ///< Use Beta Sigmoid
    bool allow_neg_eigval = false; ///< Allow Neg Eigval
    bool safe_gate = false; ///< Safe Gate
    GateDomain gate_domain = GateDomain::LOG; ///< Gate Domain
    bool batch_invariant = false; ///< Batch Invariant
    bool overwrite_initial_state = false; ///< Overwrite Initial State
    // NOLINTEND(readability-identifier-naming)

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the q input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_q() const
    {
        return getInput(InputNames::Q);
    }
    /// @brief Get the k input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_k() const
    {
        return getInput(InputNames::K);
    }
    /// @brief Get the v input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_v() const
    {
        return getInput(InputNames::V);
    }
    /// @brief Get the g input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_g() const
    {
        return getInput(InputNames::G);
    }
    /// @brief Get the beta input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_beta() const
    {
        return getInput(InputNames::BETA);
    }
    /// @brief Get the cu_seqlens input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_cu_seqlens() const
    {
        return getInput(InputNames::CU_SEQLENS);
    }
    /// @brief Get the initial_state input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_initial_state() const
    {
        return getInput(InputNames::INITIAL_STATE);
    }
    /// @brief Get the a_log input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_a_log() const
    {
        return getInput(InputNames::A_LOG);
    }
    /// @brief Get the dt_bias input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dt_bias() const
    {
        return getInput(InputNames::DT_BIAS);
    }
    /// @brief Get the state_indices input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_state_indices() const
    {
        return getInput(InputNames::STATE_INDICES);
    }
    /// @brief Get the o output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_o() const
    {
        return getOutput(OutputNames::O);
    }
    /// @brief Get the final_state output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_final_state() const
    {
        return getOutput(OutputNames::FINAL_STATE);
    }
    /// @brief Get the state_checkpoints output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_state_checkpoints() const
    {
        return getOutput(OutputNames::STATE_CHECKPOINTS);
    }

    /// @brief Set the q input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_q(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Q, std::move(value));
    }
    /// @brief Set the q input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_q(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Q, value);
    }
    /// @brief Set the k input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_k(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::K, std::move(value));
    }
    /// @brief Set the k input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_k(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::K, value);
    }
    /// @brief Set the v input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_v(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::V, std::move(value));
    }
    /// @brief Set the v input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_v(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::V, value);
    }
    /// @brief Set the g input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_g(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::G, std::move(value));
    }
    /// @brief Set the g input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_g(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::G, value);
    }
    /// @brief Set the beta input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_beta(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BETA, std::move(value));
    }
    /// @brief Set the beta input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_beta(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BETA, value);
    }
    /// @brief Set the cu_seqlens input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_cu_seqlens(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::CU_SEQLENS, std::move(value));
    }
    /// @brief Set the cu_seqlens input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_cu_seqlens(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::CU_SEQLENS, value);
    }
    /// @brief Set the initial_state input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_initial_state(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INITIAL_STATE, std::move(value));
    }
    /// @brief Set the initial_state input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_initial_state(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INITIAL_STATE, value);
    }
    /// @brief Set the a_log input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_a_log(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::A_LOG, std::move(value));
    }
    /// @brief Set the a_log input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_a_log(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::A_LOG, value);
    }
    /// @brief Set the dt_bias input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_dt_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DT_BIAS, std::move(value));
    }
    /// @brief Set the dt_bias input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_dt_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DT_BIAS, value);
    }
    /// @brief Set the state_indices input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_state_indices(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::STATE_INDICES, std::move(value));
    }
    /// @brief Set the state_indices input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_state_indices(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::STATE_INDICES, value);
    }
    /// @brief Set the o output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_o(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::O, std::move(value));
    }
    /// @brief Set the o output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_o(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::O, value);
    }
    /// @brief Set the final_state output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_final_state(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::FINAL_STATE, std::move(value));
    }
    /// @brief Set the final_state output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_final_state(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::FINAL_STATE, value);
    }
    /// @brief Set the state_checkpoints output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_state_checkpoints(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::STATE_CHECKPOINTS, std::move(value));
    }
    /// @brief Set the state_checkpoints output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_state_checkpoints(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::STATE_CHECKPOINTS, value);
    }
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_scale(float value)
    {
        scale = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_scale() const
    {
        return scale;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_output_final_state(bool value)
    {
        output_final_state = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_output_final_state() const
    {
        return output_final_state;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_use_qk_l2norm(bool value)
    {
        use_qk_l2norm = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_use_qk_l2norm() const
    {
        return use_qk_l2norm;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_checkpoint_every_n_tokens(int64_t value)
    {
        checkpoint_every_n_tokens = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int64_t get_checkpoint_every_n_tokens() const
    {
        return checkpoint_every_n_tokens;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_use_beta_sigmoid(bool value)
    {
        use_beta_sigmoid = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_use_beta_sigmoid() const
    {
        return use_beta_sigmoid;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_allow_neg_eigval(bool value)
    {
        allow_neg_eigval = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_allow_neg_eigval() const
    {
        return allow_neg_eigval;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_safe_gate(bool value)
    {
        safe_gate = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_safe_gate() const
    {
        return safe_gate;
    }

    /**
     * @brief Set gate domain
     * @param value Gate Domain
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_gate_domain(GateDomain value)
    {
        gate_domain = value;
        return *this;
    }

    /// @brief Get gate domain
    // NOLINTNEXTLINE(readability-identifier-naming)
    GateDomain get_gate_domain() const
    {
        return gate_domain;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_batch_invariant(bool value)
    {
        batch_invariant = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_batch_invariant() const
    {
        return batch_invariant;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GdnAttributes& set_overwrite_initial_state(bool value)
    {
        overwrite_initial_state = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_overwrite_initial_state() const
    {
        return overwrite_initial_state;
    }

    /**
     * @brief Custom equality hook for Gdn-specific attributes
     *
     * Compares the operation's scalar/enum/vector configuration — the
     * fields that define its mathematical semantics rather than tensor
     * layout, so logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const GdnAttributes& other) const
    {
        return scale == other.scale && output_final_state == other.output_final_state
               && use_qk_l2norm == other.use_qk_l2norm
               && checkpoint_every_n_tokens == other.checkpoint_every_n_tokens
               && use_beta_sigmoid == other.use_beta_sigmoid
               && allow_neg_eigval == other.allow_neg_eigval && safe_gate == other.safe_gate
               && gate_domain == other.gate_domain && batch_invariant == other.batch_invariant
               && overwrite_initial_state == other.overwrite_initial_state;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const GdnAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }
};
} // namespace hipdnn_frontend::graph
