// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file GatherAttributes.hpp
 * @brief Attributes for Gather operations
 *
 * This file defines the GatherAttributes class for configuring
 * Gather operations in hipDNN computational graphs.
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
 * @class GatherAttributes
 * @brief Configuration for Gather operations
 */
class GatherAttributes : public Attributes<GatherAttributes>
{
public:
    GatherAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        X = 0,
        INDICES = 1
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        Y = 0
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    int64_t axis = 0; ///< Axis
    // NOLINTEND(readability-identifier-naming)

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the x input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the indices input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_indices() const
    {
        return getInput(InputNames::INDICES);
    }
    /// @brief Get the y output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }

    /// @brief Set the x input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GatherAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    /// @brief Set the x input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GatherAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    /// @brief Set the indices input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GatherAttributes& set_indices(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INDICES, std::move(value));
    }
    /// @brief Set the indices input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GatherAttributes& set_indices(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INDICES, value);
    }
    /// @brief Set the y output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GatherAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    /// @brief Set the y output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    GatherAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    GatherAttributes& set_axis(int64_t value)
    {
        axis = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int64_t get_axis() const
    {
        return axis;
    }

    /**
     * @brief Custom equality hook for Gather-specific attributes
     *
     * Compares the operation's scalar/enum/vector configuration — the
     * fields that define its mathematical semantics rather than tensor
     * layout, so logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const GatherAttributes& other) const
    {
        return axis == other.axis;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const GatherAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }
};
} // namespace hipdnn_frontend::graph
