// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file ConvolutionDgradAttributes.hpp
 * @brief Attributes for convolution data gradient (backward data) operations
 *
 * This file defines the ConvDgradAttributes class for configuring backward
 * convolution operations that compute the gradient with respect to input data.
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
 * @class ConvDgradAttributes
 * @brief Configuration for convolution data gradient (backward data) operations
 *
 * ConvDgradAttributes specifies the parameters for computing the gradient of
 * the convolution with respect to the input data (dgrad operation):
 * dx = conv_bwd(dy, w)
 *
 * @code{.cpp}
 * auto dx = graph.conv_dgrad(dy, w, ConvDgradAttributes()
 *              .set_padding({1, 1})
 *              .set_stride({1, 1}));
 * @endcode
 *
 * @see Graph::conv_dgrad(), ConvFpropAttributes, ConvWgradAttributes
 */
class ConvDgradAttributes : public Attributes<ConvDgradAttributes>
{
public:
    ConvDgradAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        DY = 0, ///< Gradient of output tensor
        W = 1 ///< Weight/filter tensor
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        DX = 0 ///< Gradient of input tensor
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

    /// @brief Get the output gradient tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
    /// @brief Get the weight/filter tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_w() const
    {
        return getInput(InputNames::W);
    }
    /// @brief Get the input gradient tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dx() const
    {
        return getOutput(OutputNames::DX);
    }

    // Setters for tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_w(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::W, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_w(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::W, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dx(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DX, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dx(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DX, value);
    }

    /**
     * @brief Set symmetric padding (same for pre and post)
     * @param padding Padding values for each spatial dimension
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_padding(const std::vector<int64_t>& padding)
    {
        set_pre_padding(padding);
        set_post_padding(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_pre_padding(const std::vector<int64_t>& padding)
    {
        pre_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_pre_padding(std::vector<int64_t>&& padding)
    {
        pre_padding = std::move(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_post_padding(const std::vector<int64_t>& padding)
    {
        post_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_post_padding(std::vector<int64_t>&& padding)
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
    ConvDgradAttributes& set_stride(const std::vector<int64_t>& strideVals)
    {
        stride = strideVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_stride(std::vector<int64_t>&& strideVals)
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
    ConvDgradAttributes& set_dilation(const std::vector<int64_t>& dilationVals)
    {
        dilation = dilationVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dilation(std::vector<int64_t>&& dilationVals)
    {
        dilation = std::move(dilationVals);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_convolution_mode(ConvolutionMode mode)
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

typedef ConvDgradAttributes Conv_dgrad_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
