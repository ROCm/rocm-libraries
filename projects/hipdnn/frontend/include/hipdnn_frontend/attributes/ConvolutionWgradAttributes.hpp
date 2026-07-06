// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file ConvolutionWgradAttributes.hpp
 * @brief Attributes for convolution weight gradient (backward filter) operations
 *
 * This file defines the ConvWgradAttributes class for configuring backward
 * convolution operations that compute the gradient with respect to weights.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class ConvWgradAttributes
 * @brief Configuration for convolution weight gradient (backward filter) operations
 *
 * ConvWgradAttributes specifies the parameters for computing the gradient of
 * the convolution with respect to the weights (wgrad operation):
 * dw = conv_wrw(x, dy)
 *
 * @code{.cpp}
 * auto dw = graph.conv_wgrad(dy, x, ConvWgradAttributes()
 *              .set_padding({1, 1})
 *              .set_stride({1, 1}));
 * @endcode
 *
 * @see Graph::conv_wgrad(), ConvFpropAttributes, ConvDgradAttributes
 */
class ConvWgradAttributes : public Attributes<ConvWgradAttributes>
{
public:
    ConvWgradAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        DY = 0, ///< Gradient of output tensor
        X = 1 ///< Input activation tensor
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        DW = 0 ///< Gradient of weights tensor
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    std::vector<int64_t> pre_padding; ///< Padding before convolution (per spatial dim)
    std::vector<int64_t> post_padding; ///< Padding after convolution (per spatial dim)
    std::vector<int64_t> stride; ///< Stride (per spatial dim)
    std::vector<int64_t> dilation; ///< Dilation (per spatial dim)
    /// Convolution mode (default: CROSS_CORRELATION)
    ConvolutionMode math_mode = ConvolutionMode::CROSS_CORRELATION;
    // NOLINTEND(readability-identifier-naming)

    /// @brief Get the input activation tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the output gradient tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
    /// @brief Get the weight gradient tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dw() const
    {
        return getOutput(OutputNames::DW);
    }

    // Setters for tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dw(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DW, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dw(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DW, value);
    }

    /**
     * @brief Set symmetric padding (same for pre and post)
     * @param padding Padding values for each spatial dimension
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_padding(const std::vector<int64_t>& padding)
    {
        set_pre_padding(padding);
        set_post_padding(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_pre_padding(const std::vector<int64_t>& padding)
    {
        pre_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_pre_padding(std::vector<int64_t>&& padding)
    {
        pre_padding = std::move(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_post_padding(const std::vector<int64_t>& padding)
    {
        post_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_post_padding(std::vector<int64_t>&& padding)
    {
        post_padding = std::move(padding);
        return *this;
    }

    /**
     * @brief Set convolution stride
     * @param strideVals Stride values for each spatial dimension
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_stride(const std::vector<int64_t>& strideVals)
    {
        stride = strideVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_stride(std::vector<int64_t>&& strideVals)
    {
        stride = std::move(strideVals);
        return *this;
    }

    /**
     * @brief Set filter dilation
     * @param dilationVals Dilation values for each spatial dimension
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dilation(const std::vector<int64_t>& dilationVals)
    {
        dilation = dilationVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dilation(std::vector<int64_t>&& dilationVals)
    {
        dilation = std::move(dilationVals);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_convolution_mode(ConvolutionMode mode)
    {
        math_mode = mode;
        return *this;
    }

    /// @brief Get pre-convolution padding
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_pre_padding() const
    {
        return pre_padding;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_post_padding() const
    {
        return post_padding;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_stride() const
    {
        return stride;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_dilation() const
    {
        return dilation;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvolutionMode get_convolution_mode() const
    {
        return math_mode;
    }
};

typedef ConvWgradAttributes Conv_wgrad_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
