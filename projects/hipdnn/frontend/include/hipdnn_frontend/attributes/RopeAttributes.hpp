// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file RopeAttributes.hpp
 * @brief Attributes for Rope operations
 *
 * This file defines the RopeAttributes class for configuring
 * Rope operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class RopeAttributes
 * @brief Configuration for rotary position embedding (RoPE) forward operations
 *
 * Mirrors cuDNN's RoPE_attributes. Rotate-half (non-interleaved) rotation of the segment
 * x[..., o : o + r] of the last dim D, with r = rope_dim (0 means D) and o = rope_offset.
 * freqs holds raw angles in FLOAT; it has x's rank, each non-last dim equal to x's or 1, and
 * last dim r. Dims outside the segment are multiplied by output_scale and passed through.
 */
class RopeAttributes : public Attributes<RopeAttributes>
{
public:
    RopeAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        X = 0,
        FREQS = 1
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
    float output_scale = 1.0f; ///< Scale applied to every output element (rotated or not)
    int64_t rope_dim = 0; ///< Rotation width r; 0 rotates the full last dim
    /// Start o of the rotated segment (hipDNN extension, no cuDNN equivalent).
    /// Unset: o = D - r (cuDNN, trailing segment); 0: Llama/Qwen-style partial rotary.
    std::optional<int64_t> rope_offset = std::nullopt;
    // NOLINTEND(readability-identifier-naming)

    // NOLINTBEGIN(readability-identifier-naming)
    /// @brief Get the x input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the freqs input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_freqs() const
    {
        return getInput(InputNames::FREQS);
    }
    /// @brief Get the y output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }

    /// @brief Set the x input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    /// @brief Set the x input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    /// @brief Set the freqs input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_freqs(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::FREQS, std::move(value));
    }
    /// @brief Set the freqs input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_freqs(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::FREQS, value);
    }
    /// @brief Set the y output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    /// @brief Set the y output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_output_scale(float value)
    {
        output_scale = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    float get_output_scale() const
    {
        return output_scale;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_rope_dim(int64_t value)
    {
        rope_dim = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int64_t get_rope_dim() const
    {
        return rope_dim;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    RopeAttributes& set_rope_offset(int64_t value)
    {
        rope_offset = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int64_t> get_rope_offset() const
    {
        return rope_offset;
    }

    /**
     * @brief Custom equality hook for Rope-specific attributes
     *
     * Compares the operation's scalar/enum/vector configuration — the
     * fields that define its mathematical semantics rather than tensor
     * layout, so logical and strict equality coincide here.
     */
    bool logicallyEqualsImpl(const RopeAttributes& other) const
    {
        return output_scale == other.output_scale && rope_dim == other.rope_dim
               && rope_offset == other.rope_offset;
    }

    /// @brief Strict equality delegates to logical equality; no layout-only
    ///        fields exist in this class to distinguish the two checks.
    bool strictEqualsImpl(const RopeAttributes& other) const
    {
        return logicallyEqualsImpl(other);
    }
};
typedef RopeAttributes RoPE_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
