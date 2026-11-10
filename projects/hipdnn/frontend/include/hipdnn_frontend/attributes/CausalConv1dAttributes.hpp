// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file CausalConv1dAttributes.hpp
 * @brief Attributes for CausalConv1d operations
 *
 * This file defines the CausalConv1dAttributes class for configuring
 * CausalConv1d operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <cstdint>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class CausalConv1dAttributes
 * @brief Configuration for depthwise causal 1-D convolution with a carried state
 *
 * Inputs are token-packed: x [T, C] holds N sequences with offsets cu_seqlens [N + 1]
 * (INT32 or INT64); weight is [C, W] with W >= 2 and y is [T, C]. For sequence n and
 * channel c, with xs the W - 1 state columns followed by the sequence's tokens:
 * y[t, c] = act(bias[c] + sum_{j < W} weight[c, j] * xs[t + j, c]) and the final state is
 * the last W - 1 entries of xs. Optional: bias [C], initial_state [N, C, W - 1] (absent:
 * zeros), state_indices [N] (INT32 or INT64; requires initial_state and turns both states
 * into a pool [P, C, W - 1]), and the final_state output, requested with
 * set_output_final_state(true). final_state may share initial_state's device buffer.
 */
class CausalConv1dAttributes : public Attributes<CausalConv1dAttributes>
{
public:
    CausalConv1dAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        X = 0,
        WEIGHT = 1,
        CU_SEQLENS = 2,
        BIAS = 3,
        INITIAL_STATE = 4,
        STATE_INDICES = 5
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        Y = 0,
        FINAL_STATE = 1
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    CausalConv1dActivation activation = CausalConv1dActivation::IDENTITY; ///< Activation
    bool output_final_state = false; ///< Emit the final_state output
    // NOLINTEND(readability-identifier-naming)

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the x input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the weight input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_weight() const
    {
        return getInput(InputNames::WEIGHT);
    }
    /// @brief Get the cu_seqlens input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_cu_seqlens() const
    {
        return getInput(InputNames::CU_SEQLENS);
    }
    /// @brief Get the bias input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(InputNames::BIAS);
    }
    /// @brief Get the initial_state input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_initial_state() const
    {
        return getInput(InputNames::INITIAL_STATE);
    }
    /// @brief Get the state_indices input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_state_indices() const
    {
        return getInput(InputNames::STATE_INDICES);
    }
    /// @brief Get the y output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }
    /// @brief Get the final_state output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_final_state() const
    {
        return getOutput(OutputNames::FINAL_STATE);
    }

    /// @brief Set the x input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    /// @brief Set the x input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    /// @brief Set the weight input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_weight(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::WEIGHT, std::move(value));
    }
    /// @brief Set the weight input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_weight(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::WEIGHT, value);
    }
    /// @brief Set the cu_seqlens input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_cu_seqlens(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::CU_SEQLENS, std::move(value));
    }
    /// @brief Set the cu_seqlens input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_cu_seqlens(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::CU_SEQLENS, value);
    }
    /// @brief Set the bias input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    /// @brief Set the bias input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    /// @brief Set the initial_state input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_initial_state(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INITIAL_STATE, std::move(value));
    }
    /// @brief Set the initial_state input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_initial_state(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INITIAL_STATE, value);
    }
    /// @brief Set the state_indices input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_state_indices(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::STATE_INDICES, std::move(value));
    }
    /// @brief Set the state_indices input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_state_indices(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::STATE_INDICES, value);
    }
    /// @brief Set the y output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    /// @brief Set the y output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    /// @brief Set the final_state output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_final_state(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::FINAL_STATE, std::move(value));
    }
    /// @brief Set the final_state output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_final_state(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::FINAL_STATE, value);
    }
    // NOLINTEND(readability-identifier-naming)

    /**
     * @brief Set activation
     * @param value Activation
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_activation(CausalConv1dActivation value)
    {
        activation = value;
        return *this;
    }

    /// @brief Get activation
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dActivation get_activation() const
    {
        return activation;
    }

    /**
     * @brief Request the final_state output [N or P, C, W - 1]
     * @param value True to emit final_state
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    CausalConv1dAttributes& set_output_final_state(bool value)
    {
        output_final_state = value;
        return *this;
    }

    /// @brief Whether the final_state output is requested
    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_output_final_state() const
    {
        return output_final_state;
    }

    /**
     * @brief Custom equality hook for CausalConv1d-specific attributes
     *
     * Compares the operation's scalar/enum/vector configuration — the
     * fields that define its mathematical semantics rather than tensor
     * layout, so logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const CausalConv1dAttributes& other) const
    {
        return activation == other.activation && output_final_state == other.output_final_state;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const CausalConv1dAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }
};
} // namespace hipdnn_frontend::graph
