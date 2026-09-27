// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file MoeGroupedMatmulAttributes.hpp
 * @brief Attributes for MoeGroupedMatmul operations
 *
 * This file defines the MoeGroupedMatmulAttributes class for configuring
 * MoeGroupedMatmul operations in hipDNN computational graphs.
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
 * @class MoeGroupedMatmulAttributes
 * @brief Configuration for MoeGroupedMatmul operations
 */
class MoeGroupedMatmulAttributes : public Attributes<MoeGroupedMatmulAttributes>
{
public:
    MoeGroupedMatmulAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        TOKEN = 0,
        WEIGHT = 1,
        FIRST_TOKEN_OFFSET = 2,
        TOKEN_INDEX = 3,
        TOKEN_KS = 4
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        OUTPUT = 0
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    MoeGroupedMatmulMode mode = MoeGroupedMatmulMode::NONE; ///< Mode
    int32_t top_k = 0; ///< Top K
    // NOLINTEND(readability-identifier-naming)

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the token input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_token() const
    {
        return getInput(InputNames::TOKEN);
    }
    /// @brief Get the weight input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_weight() const
    {
        return getInput(InputNames::WEIGHT);
    }
    /// @brief Get the first_token_offset input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_first_token_offset() const
    {
        return getInput(InputNames::FIRST_TOKEN_OFFSET);
    }
    /// @brief Get the token_index input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_token_index() const
    {
        return getInput(InputNames::TOKEN_INDEX);
    }
    /// @brief Get the token_ks input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_token_ks() const
    {
        return getInput(InputNames::TOKEN_KS);
    }
    /// @brief Get the output output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_output() const
    {
        return getOutput(OutputNames::OUTPUT);
    }

    /// @brief Set the token input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_token(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::TOKEN, std::move(value));
    }
    /// @brief Set the token input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_token(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::TOKEN, value);
    }
    /// @brief Set the weight input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_weight(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::WEIGHT, std::move(value));
    }
    /// @brief Set the weight input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_weight(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::WEIGHT, value);
    }
    /// @brief Set the first_token_offset input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_first_token_offset(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::FIRST_TOKEN_OFFSET, std::move(value));
    }
    /// @brief Set the first_token_offset input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes&
        set_first_token_offset(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::FIRST_TOKEN_OFFSET, value);
    }
    /// @brief Set the token_index input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_token_index(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::TOKEN_INDEX, std::move(value));
    }
    /// @brief Set the token_index input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_token_index(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::TOKEN_INDEX, value);
    }
    /// @brief Set the token_ks input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_token_ks(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::TOKEN_KS, std::move(value));
    }
    /// @brief Set the token_ks input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_token_ks(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::TOKEN_KS, value);
    }
    /// @brief Set the output output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_output(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::OUTPUT, std::move(value));
    }
    /// @brief Set the output output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_output(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::OUTPUT, value);
    }
    // NOLINTEND(readability-identifier-naming)

    /**
     * @brief Set mode
     * @param value Mode
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_mode(MoeGroupedMatmulMode value)
    {
        mode = value;
        return *this;
    }

    /// @brief Get mode
    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulMode get_mode() const
    {
        return mode;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    MoeGroupedMatmulAttributes& set_top_k(int32_t value)
    {
        top_k = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int32_t get_top_k() const
    {
        return top_k;
    }

    /**
     * @brief Custom equality hook for MoE-grouped-matmul-specific attributes
     *
     * Compares the routing mode and top-k count — the fields that define the
     * mathematical semantics of the operation rather than tensor layout, so
     * logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const MoeGroupedMatmulAttributes& other) const
    {
        return mode == other.mode && top_k == other.top_k;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const MoeGroupedMatmulAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }
};
typedef MoeGroupedMatmulAttributes Moe_grouped_matmul_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
