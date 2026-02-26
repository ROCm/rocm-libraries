// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file ConvolutionFpropAttributes.hpp
 * @brief Attributes for forward convolution operations
 *
 * This file defines the ConvFpropAttributes class for configuring forward
 * convolution (fprop) operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class ConvFpropAttributes
 * @brief Configuration for forward convolution operations
 *
 * ConvFpropAttributes specifies the parameters for a forward convolution (fprop)
 * operation: y = conv(x, w). The operation computes the convolution of input tensor
 * x with filter tensor w.
 *
 * @code{.cpp}
 * // Create a 2D convolution with 3x3 kernel, padding=1, stride=1
 * auto y = graph.conv_fprop(x, w, ConvFpropAttributes()
 *              .set_padding({1, 1})
 *              .set_stride({1, 1})
 *              .set_dilation({1, 1}));
 * @endcode
 *
 * @see Graph::conv_fprop(), ConvDgradAttributes, ConvWgradAttributes
 */
class ConvFpropAttributes : public Attributes<ConvFpropAttributes>
{
public:
    /// Input tensor identifiers
    enum class InputNames
    {
        X = 0, ///< Input activation tensor (NCHW or NHWC)
        W = 1 ///< Weight/filter tensor (KCRS or KRSC)
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        Y = 0 ///< Output activation tensor
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    std::vector<int64_t> pre_padding; ///< Padding before convolution (per spatial dim)
    std::vector<int64_t> post_padding; ///< Padding after convolution (per spatial dim)
    std::vector<int64_t> stride; ///< Stride (per spatial dim)
    std::vector<int64_t> dilation; ///< Dilation (per spatial dim)
    /// Convolution mode (CROSS_CORRELATION or CONVOLUTION)
    ConvolutionMode math_mode = ConvolutionMode::CROSS_CORRELATION;
    // NOLINTEND(readability-identifier-naming)

    /// @brief Get the input activation tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the weight/filter tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_w() const
    {
        return getInput(InputNames::W);
    }
    /// @brief Get the output activation tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }

    /// @brief Set the input activation tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_w(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::W, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_w(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::W, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    /**
     * @brief Set symmetric padding (same for pre and post)
     * @param padding Padding values for each spatial dimension
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_padding(std::vector<int64_t> padding)
    {
        pre_padding = padding;
        post_padding = std::move(padding);
        return *this;
    }

    /**
     * @brief Set pre-convolution padding
     * @param padding Padding before the input (per spatial dimension)
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_pre_padding(const std::vector<int64_t>& padding)
    {
        pre_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_pre_padding(std::vector<int64_t>&& padding)
    {
        pre_padding = std::move(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_post_padding(const std::vector<int64_t>& padding)
    {
        post_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_post_padding(std::vector<int64_t>&& padding)
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
    ConvFpropAttributes& set_stride(const std::vector<int64_t>& strideVals)
    {
        stride = strideVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_stride(std::vector<int64_t>&& strideVals)
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
    ConvFpropAttributes& set_dilation(const std::vector<int64_t>& dilationVals)
    {
        dilation = dilationVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_dilation(std::vector<int64_t>&& dilationVals)
    {
        dilation = std::move(dilationVals);
        return *this;
    }

    /**
     * @brief Set convolution mode
     * @param mode CROSS_CORRELATION (default) or CONVOLUTION
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvFpropAttributes& set_convolution_mode(ConvolutionMode mode)
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

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        return hipdnn_data_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
            builder,
            get_x()->get_uid(),
            get_w()->get_uid(),
            get_y()->get_uid(),
            &pre_padding,
            &post_padding,
            &stride,
            &dilation,
            toSdkType(math_mode));
    }

    static ConvFpropAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        ConvFpropAttributes attr;

        attr.set_x(tensorMap.at(fb->x_tensor_uid()));
        attr.set_w(tensorMap.at(fb->w_tensor_uid()));
        attr.set_y(tensorMap.at(fb->y_tensor_uid()));

        if(fb->pre_padding() != nullptr)
        {
            std::vector<int64_t> prePadding(fb->pre_padding()->begin(), fb->pre_padding()->end());
            attr.set_pre_padding(std::move(prePadding));
        }
        if(fb->post_padding() != nullptr)
        {
            std::vector<int64_t> postPadding(fb->post_padding()->begin(),
                                             fb->post_padding()->end());
            attr.set_post_padding(std::move(postPadding));
        }
        if(fb->stride() != nullptr)
        {
            std::vector<int64_t> strideVec(fb->stride()->begin(), fb->stride()->end());
            attr.set_stride(std::move(strideVec));
        }
        if(fb->dilation() != nullptr)
        {
            std::vector<int64_t> dilationVec(fb->dilation()->begin(), fb->dilation()->end());
            attr.set_dilation(std::move(dilationVec));
        }
        attr.set_convolution_mode(fromSdkType(fb->conv_mode()));

        return attr;
    }
};
typedef ConvFpropAttributes Conv_fprop_attributes; ///< @brief cuDNN compatibility alias
} // namespace hipdnn_frontend::graph
