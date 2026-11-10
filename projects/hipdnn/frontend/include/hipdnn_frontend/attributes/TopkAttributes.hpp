// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TopkAttributes.hpp
 * @brief Attributes for Topk operations
 *
 * This file defines the TopkAttributes class for configuring
 * Topk operations in hipDNN computational graphs.
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
 * @class TopkAttributes
 * @brief Configuration for top-k selection along the last dimension of x.
 *
 * Selects the k largest (largest = true, default) or smallest entries of each row, sorted
 * best-first; equal values keep the lower index first and NaN compares greater than every
 * number. Outputs: indices (required; INT32 or INT64, INT64 when unset) and values
 * (optional; data type of x when unset). Both have the dims of x with the last dim k.
 */
class TopkAttributes : public Attributes<TopkAttributes>
{
public:
    TopkAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        X = 0
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        VALUES = 0,
        INDICES = 1
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    int64_t k = 0; ///< Entries selected per row, 1 <= k <= x last dim
    bool largest = true; ///< Select the largest (true) or smallest entries
    // NOLINTEND(readability-identifier-naming)

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the x input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the values output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_values() const
    {
        return getOutput(OutputNames::VALUES);
    }
    /// @brief Get the indices output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_indices() const
    {
        return getOutput(OutputNames::INDICES);
    }

    /// @brief Set the x input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    TopkAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    /// @brief Set the x input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    TopkAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    /// @brief Set the values output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    TopkAttributes& set_values(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::VALUES, std::move(value));
    }
    /// @brief Set the values output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    TopkAttributes& set_values(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::VALUES, value);
    }
    /// @brief Set the indices output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    TopkAttributes& set_indices(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::INDICES, std::move(value));
    }
    /// @brief Set the indices output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    TopkAttributes& set_indices(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::INDICES, value);
    }
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    TopkAttributes& set_k(int64_t value)
    {
        k = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int64_t get_k() const
    {
        return k;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    TopkAttributes& set_largest(bool value)
    {
        largest = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_largest() const
    {
        return largest;
    }

    /**
     * @brief Custom equality hook for Topk-specific attributes
     *
     * Compares the operation's scalar/enum/vector configuration — the
     * fields that define its mathematical semantics rather than tensor
     * layout, so logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const TopkAttributes& other) const
    {
        return k == other.k && largest == other.largest;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const TopkAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }
};
} // namespace hipdnn_frontend::graph
