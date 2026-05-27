// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file ResampleBwdAttributes.hpp
 * @brief Attributes for ResampleBwd operations
 *
 * This file defines the ResampleBwdAttributes class for configuring
 * ResampleBwd operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class ResampleBwdAttributes
 * @brief Configuration for ResampleBwd operations
 */
class ResampleBwdAttributes : public Attributes<ResampleBwdAttributes>
{
public:
    ResampleBwdAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        DY = 0,
        INDEX = 1
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        DX = 0
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    std::vector<int64_t> pre_padding; ///< Pre Padding
    std::vector<int64_t> post_padding; ///< Post Padding
    std::vector<int64_t> stride; ///< Stride
    std::vector<int64_t> window; ///< Window
    ResampleMode resample_mode = ResampleMode::MAXPOOL; ///< Resample Mode
    PaddingMode padding_mode = PaddingMode::ZERO_PAD; ///< Padding Mode
    std::optional<bool> generate_index = std::nullopt; ///< Generate Index
    // NOLINTEND(readability-identifier-naming)

    /// @brief Get the dy input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
    /// @brief Get the index input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_index() const
    {
        return getInput(InputNames::INDEX);
    }
    /// @brief Get the dx output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dx() const
    {
        return getOutput(OutputNames::DX);
    }

    /// @brief Set the dy input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    /// @brief Set the dy input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    /// @brief Set the index input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_index(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INDEX, std::move(value));
    }
    /// @brief Set the index input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_index(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INDEX, value);
    }
    /// @brief Set the dx output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_dx(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DX, std::move(value));
    }
    /// @brief Set the dx output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_dx(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DX, value);
    }

    /**
     * @brief Set pre padding
     * @param value Pre Padding values
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_pre_padding(const std::vector<int64_t>& value)
    {
        pre_padding = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_pre_padding(std::vector<int64_t>&& value)
    {
        pre_padding = std::move(value);
        return *this;
    }

    /// @brief Get pre padding
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_pre_padding() const
    {
        return pre_padding;
    }

    /**
     * @brief Set post padding
     * @param value Post Padding values
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_post_padding(const std::vector<int64_t>& value)
    {
        post_padding = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_post_padding(std::vector<int64_t>&& value)
    {
        post_padding = std::move(value);
        return *this;
    }

    /// @brief Get post padding
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_post_padding() const
    {
        return post_padding;
    }

    /**
     * @brief Set stride
     * @param value Stride values
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_stride(const std::vector<int64_t>& value)
    {
        stride = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_stride(std::vector<int64_t>&& value)
    {
        stride = std::move(value);
        return *this;
    }

    /// @brief Get stride
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_stride() const
    {
        return stride;
    }

    /**
     * @brief Set window
     * @param value Window values
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_window(const std::vector<int64_t>& value)
    {
        window = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_window(std::vector<int64_t>&& value)
    {
        window = std::move(value);
        return *this;
    }

    /// @brief Get window
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_window() const
    {
        return window;
    }

    /**
     * @brief Set resample mode
     * @param value Resample Mode
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_resample_mode(ResampleMode value)
    {
        resample_mode = value;
        return *this;
    }

    /// @brief Get resample mode
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleMode get_resample_mode() const
    {
        return resample_mode;
    }

    /**
     * @brief Set padding mode
     * @param value Padding Mode
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_padding_mode(PaddingMode value)
    {
        padding_mode = value;
        return *this;
    }

    /// @brief Get padding mode
    // NOLINTNEXTLINE(readability-identifier-naming)
    PaddingMode get_padding_mode() const
    {
        return padding_mode;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleBwdAttributes& set_generate_index(bool value)
    {
        generate_index = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<bool> get_generate_index() const
    {
        return generate_index;
    }
};
typedef ResampleBwdAttributes Resample_bwd_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
