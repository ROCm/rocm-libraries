// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file MoeRoutingAttributes.hpp
 * @brief Attributes for MoeRouting operations
 *
 * This file defines the MoeRoutingAttributes class for configuring
 * MoeRouting operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class MoeRoutingAttributes
 * @brief Configuration for MoeRouting operations
 */
class MoeRoutingAttributes : public Attributes<MoeRoutingAttributes>
{
public:
    MoeRoutingAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        TOPK_IDS = 0
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        FIRST_TOKEN_OFFSET = 0,
        TOKEN_INDEX = 1,
        TOKEN_KS = 2
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    int64_t num_experts = 0; ///< Num Experts
    // NOLINTEND(readability-identifier-naming)

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the topk_ids input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_topk_ids() const
    {
        return getInput(InputNames::TOPK_IDS);
    }
    /// @brief Get the first_token_offset output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_first_token_offset() const
    {
        return getOutput(OutputNames::FIRST_TOKEN_OFFSET);
    }
    /// @brief Get the token_index output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_token_index() const
    {
        return getOutput(OutputNames::TOKEN_INDEX);
    }
    /// @brief Get the token_ks output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_token_ks() const
    {
        return getOutput(OutputNames::TOKEN_KS);
    }

    /// @brief Set the topk_ids input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_topk_ids(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::TOPK_IDS, std::move(value));
    }
    /// @brief Set the topk_ids input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_topk_ids(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::TOPK_IDS, value);
    }
    /// @brief Set the first_token_offset output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_first_token_offset(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::FIRST_TOKEN_OFFSET, std::move(value));
    }
    /// @brief Set the first_token_offset output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_first_token_offset(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::FIRST_TOKEN_OFFSET, value);
    }
    /// @brief Set the token_index output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_token_index(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::TOKEN_INDEX, std::move(value));
    }
    /// @brief Set the token_index output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_token_index(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::TOKEN_INDEX, value);
    }
    /// @brief Set the token_ks output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_token_ks(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::TOKEN_KS, std::move(value));
    }
    /// @brief Set the token_ks output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_token_ks(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::TOKEN_KS, value);
    }
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeRoutingAttributes& set_num_experts(int64_t value)
    {
        num_experts = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int64_t get_num_experts() const
    {
        return num_experts;
    }

    /**
     * @brief Custom equality hook for MoeRouting-specific attributes
     *
     * Compares the operation's scalar/enum/vector configuration — the
     * fields that define its mathematical semantics rather than tensor
     * layout, so logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const MoeRoutingAttributes& other) const
    {
        return num_experts == other.num_experts;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const MoeRoutingAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }
};
} // namespace hipdnn_frontend::graph
