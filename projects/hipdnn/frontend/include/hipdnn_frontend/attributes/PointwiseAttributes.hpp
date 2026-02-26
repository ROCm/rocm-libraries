// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file PointwiseAttributes.hpp
 * @brief Attributes for element-wise (pointwise) operations
 *
 * This file defines the PointwiseAttributes class for configuring pointwise
 * operations including activation functions, arithmetic, and other element-wise ops.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class PointwiseAttributes
 * @brief Configuration for element-wise (pointwise) operations
 *
 * PointwiseAttributes specifies the parameters for pointwise operations that
 * operate element-wise on tensors. This includes:
 * - Activation functions: RELU, SIGMOID, TANH, GELU, ELU, SWISH, etc.
 * - Arithmetic operations: ADD, MUL, SUB, DIV, etc.
 * - Comparison operations: CMP_EQ, CMP_GT, etc.
 * - Other supported pointwise operations as defined in PointwiseMode
 *
 * @code{.cpp}
 * // ReLU activation
 * auto y = graph.pointwise(x, PointwiseAttributes()
 *              .set_mode(PointwiseMode::RELU));
 *
 * // Leaky ReLU with negative slope
 * auto y = graph.pointwise(x, PointwiseAttributes()
 *              .set_mode(PointwiseMode::RELU_FWD)
 *              .set_relu_lower_clip_slope(0.01f));
 *
 * // Element-wise addition
 * auto c = graph.pointwise(a, b, PointwiseAttributes()
 *              .set_mode(PointwiseMode::ADD));
 * @endcode
 *
 * @see Graph::pointwise(), PointwiseMode
 */
class PointwiseAttributes : public Attributes<PointwiseAttributes>
{
public:
    /// @brief Get the pointwise operation mode
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseMode get_mode() const
    {
        return mode;
    }
    /// @brief Get ReLU lower clipping bound
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_lower_clip() const
    {
        return relu_lower_clip;
    }
    /// @brief Get ReLU upper clipping bound
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_upper_clip() const
    {
        return relu_upper_clip;
    }
    /// @brief Get slope for Leaky ReLU (negative region)
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_lower_clip_slope() const
    {
        return relu_lower_clip_slope;
    }
    /// @brief Get Swish activation beta parameter
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_swish_beta() const
    {
        return swish_beta;
    }
    /// @brief Get ELU activation alpha parameter
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_elu_alpha() const
    {
        return elu_alpha;
    }
    /// @brief Get Softplus activation beta parameter
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_softplus_beta() const
    {
        return softplus_beta;
    }
    /// @brief Get axis for operations like softmax
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int64_t> get_axis() const
    {
        return axis;
    }
    /// @brief Get the first input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_input_0() const
    {
        return getInput(InputNames::IN_0);
    }
    /// @brief Get the second input tensor (for binary ops)
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_input_1() const
    {
        return getInput(InputNames::IN_1);
    }
    /// @brief Get the third input tensor (for ternary ops)
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_input_2() const
    {
        return getInput(InputNames::IN_2);
    }
    /// @brief Get the output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_output_0() const
    {
        return getOutput(OutputNames::OUT_0);
    }

    /**
     * @brief Set the pointwise operation mode
     * @param value The operation to perform (see PointwiseMode enum)
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_mode(PointwiseMode value)
    {
        mode = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_lower_clip(float value)
    {
        relu_lower_clip = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_upper_clip(float value)
    {
        relu_upper_clip = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_lower_clip_slope(float value)
    {
        relu_lower_clip_slope = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_swish_beta(float value)
    {
        swish_beta = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_elu_alpha(float value)
    {
        elu_alpha = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_softplus_beta(float value)
    {
        softplus_beta = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_axis(int64_t value)
    {
        axis = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_0(const std::shared_ptr<TensorAttributes>& input0)
    {
        return setInput(InputNames::IN_0, input0);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_0(std::shared_ptr<TensorAttributes>&& input0)
    {
        return setInput(InputNames::IN_0, std::move(input0));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_1(const std::shared_ptr<TensorAttributes>& input1)
    {
        return setInput(InputNames::IN_1, input1);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_1(std::shared_ptr<TensorAttributes>&& input1)
    {
        return setInput(InputNames::IN_1, std::move(input1));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_2(const std::shared_ptr<TensorAttributes>& input2)
    {
        return setInput(InputNames::IN_2, input2);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_2(std::shared_ptr<TensorAttributes>&& input2)
    {
        return setInput(InputNames::IN_2, std::move(input2));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_output_0(const std::shared_ptr<TensorAttributes>& output0)
    {
        return setOutput(OutputNames::OUT_0, output0);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_output_0(std::shared_ptr<TensorAttributes>&& output0)
    {
        return setOutput(OutputNames::OUT_0, std::move(output0));
    }

    /// Input tensor identifiers
    enum class InputNames
    {
        IN_0 = 0, ///< First input tensor (required)
        IN_1 = 1, ///< Second input tensor (for binary ops)
        IN_2 = 2, ///< Third input tensor (for ternary ops)
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        OUT_0 = 0, ///< Output tensor
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    PointwiseMode mode = PointwiseMode::NOT_SET; ///< The operation mode
    std::optional<float> relu_lower_clip = std::nullopt; ///< ReLU lower clip bound
    std::optional<float> relu_upper_clip = std::nullopt; ///< ReLU upper clip bound
    std::optional<float> relu_lower_clip_slope = std::nullopt; ///< Leaky ReLU slope
    std::optional<int64_t> axis = std::nullopt; ///< Axis for reductions
    std::optional<float> swish_beta = std::nullopt; ///< Swish beta parameter
    std::optional<float> elu_alpha = std::nullopt; ///< ELU alpha parameter
    std::optional<float> softplus_beta = std::nullopt; ///< Softplus beta parameter
    // NOLINTEND(readability-identifier-naming)

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::PointwiseAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto in0 = get_input_0();
        auto in1 = get_input_1();
        auto in2 = get_input_2();
        auto ot0 = get_output_0();

        return hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            toSdkType(mode),
            relu_lower_clip,
            relu_upper_clip,
            relu_lower_clip_slope,
            axis,
            in0->get_uid(),
            in1 ? flatbuffers::Optional<int64_t>(in1->get_uid()) : flatbuffers::nullopt,
            in2 ? flatbuffers::Optional<int64_t>(in2->get_uid()) : flatbuffers::nullopt,
            ot0->get_uid(),
            swish_beta,
            elu_alpha,
            softplus_beta);
    }

    static PointwiseAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::PointwiseAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        PointwiseAttributes attr;

        attr.set_mode(fromSdkType(fb->operation()));

        if(fb->relu_lower_clip().has_value())
        {
            attr.set_relu_lower_clip(fb->relu_lower_clip().value());
        }
        if(fb->relu_upper_clip().has_value())
        {
            attr.set_relu_upper_clip(fb->relu_upper_clip().value());
        }
        if(fb->relu_lower_clip_slope().has_value())
        {
            attr.set_relu_lower_clip_slope(fb->relu_lower_clip_slope().value());
        }
        if(fb->swish_beta().has_value())
        {
            attr.set_swish_beta(fb->swish_beta().value());
        }
        if(fb->elu_alpha().has_value())
        {
            attr.set_elu_alpha(fb->elu_alpha().value());
        }
        if(fb->softplus_beta().has_value())
        {
            attr.set_softplus_beta(fb->softplus_beta().value());
        }
        if(fb->axis_tensor_uid().has_value())
        {
            attr.set_axis(fb->axis_tensor_uid().value());
        }

        attr.set_input_0(tensorMap.at(fb->in_0_tensor_uid()));
        if(fb->in_1_tensor_uid().has_value())
        {
            attr.set_input_1(tensorMap.at(fb->in_1_tensor_uid().value()));
        }
        if(fb->in_2_tensor_uid().has_value())
        {
            attr.set_input_2(tensorMap.at(fb->in_2_tensor_uid().value()));
        }

        attr.set_output_0(tensorMap.at(fb->out_0_tensor_uid()));

        return attr;
    }
};
typedef PointwiseAttributes Pointwise_attributes; ///< @brief cuDNN compatibility alias
} // namespace hipdnn_frontend::graph
