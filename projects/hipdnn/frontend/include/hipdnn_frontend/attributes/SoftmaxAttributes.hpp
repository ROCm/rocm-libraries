// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file SoftmaxAttributes.hpp
 * @brief Attributes for Softmax operations
 *
 * This file defines the SoftmaxAttributes class for configuring
 * Softmax operations in hipDNN computational graphs.
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
 * @class SoftmaxAttributes
 * @brief Configuration for Softmax operations
 */
class SoftmaxAttributes : public Attributes<SoftmaxAttributes>
{
public:
    SoftmaxAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        P = 0
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        S = 0,
        // cuDNN frontend spells this Softmax_attributes::output_names::Stats; the cuDNN shim
        // inherits this enum, so the name must stay as is.
        Stats = 1 // NOLINT(readability-identifier-naming)
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    /// Report the log-sum-exp stats in log2 units (stats / ln 2) instead of natural log.
    bool stats_use_log2 = false;
    /// Graph::softmax() creates the optional Stats output when true. Frontend-only; a lifted
    /// node sets it from the presence of the Stats tensor.
    bool generate_stats = false;
    // NOLINTEND(readability-identifier-naming)

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the p input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_p() const
    {
        return getInput(InputNames::P);
    }
    /// @brief Get the s output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_s() const
    {
        return getOutput(OutputNames::S);
    }
    /// @brief Get the stats output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_stats() const
    {
        return getOutput(OutputNames::Stats);
    }

    /// @brief Set the p input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    SoftmaxAttributes& set_p(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::P, std::move(value));
    }
    /// @brief Set the p input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    SoftmaxAttributes& set_p(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::P, value);
    }
    /// @brief Set the s output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    SoftmaxAttributes& set_s(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::S, std::move(value));
    }
    /// @brief Set the s output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    SoftmaxAttributes& set_s(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::S, value);
    }
    /// @brief Set the stats output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    SoftmaxAttributes& set_stats(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Stats, std::move(value));
    }
    /// @brief Set the stats output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    SoftmaxAttributes& set_stats(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Stats, value);
    }
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    SoftmaxAttributes& set_stats_use_log2(bool value)
    {
        stats_use_log2 = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_stats_use_log2() const
    {
        return stats_use_log2;
    }

    /// @brief Request the optional log-sum-exp Stats output from Graph::softmax()
    // NOLINTNEXTLINE(readability-identifier-naming)
    SoftmaxAttributes& set_generate_stats(bool value)
    {
        generate_stats = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_generate_stats() const
    {
        return generate_stats;
    }

    /**
     * @brief Custom equality hook for Softmax-specific attributes
     *
     * Compares the operation's scalar/enum/vector configuration — the
     * fields that define its mathematical semantics rather than tensor
     * layout, so logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const SoftmaxAttributes& other) const
    {
        return stats_use_log2 == other.stats_use_log2;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const SoftmaxAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }
};
typedef SoftmaxAttributes Softmax_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
