// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BatchnormInferenceAttributes.hpp
 * @brief Attributes for batch normalization inference operations
 *
 * This file defines the BatchnormInferenceAttributes class for configuring
 * batch normalization during inference (using pre-computed mean/variance).
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class BatchnormInferenceAttributes
 * @brief Configuration for batch normalization inference
 *
 * BatchnormInferenceAttributes specifies the parameters for batch normalization
 * during inference, using pre-computed running mean and inverse variance:
 *
 * y = scale * (x - mean) * inv_variance + bias
 *
 * @code{.cpp}
 * auto y = graph.batchnorm_inference(x, mean, inv_variance, scale, bias,
 *              BatchnormInferenceAttributes());
 * @endcode
 *
 * @see Graph::batchnorm_inference(), BatchnormAttributes
 */
class BatchnormInferenceAttributes : public Attributes<BatchnormInferenceAttributes>
{
public:
    BatchnormInferenceAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        X = 0, ///< Input activation tensor
        MEAN = 1, ///< Pre-computed mean
        INV_VARIANCE = 2, ///< Pre-computed inverse variance (1/sqrt(var + epsilon))
        SCALE = 3, ///< Scale (gamma) parameter
        BIAS = 4 ///< Bias (beta) parameter
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        Y = 0 ///< Normalized output tensor
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    /// @brief Get the input activation tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the mean tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getInput(InputNames::MEAN);
    }
    /// @brief Get the inverse variance tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getInput(InputNames::INV_VARIANCE);
    }
    /// @brief Get the scale (gamma) tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    /// @brief Get the bias (beta) tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(InputNames::BIAS);
    }
    /// @brief Get the output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INV_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INV_VARIANCE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
};

typedef BatchnormInferenceAttributes Batchnorm_inference_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
