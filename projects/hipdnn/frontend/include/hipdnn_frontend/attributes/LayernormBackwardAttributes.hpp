// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file LayernormBackwardAttributes.hpp
 * @brief Attributes for LayernormBackward operations
 *
 * This file defines the LayernormBackwardAttributes class for configuring
 * LayernormBackward operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class LayernormBackwardAttributes
 * @brief Configuration for LayernormBackward operations
 */
class LayernormBackwardAttributes : public Attributes<LayernormBackwardAttributes>
{
public:
    LayernormBackwardAttributes() = default;

    /// Input tensor identifiers
    enum class InputNames
    {
        DY = 0,
        X = 1,
        SCALE = 2,
        MEAN = 3,
        INV_VARIANCE = 4,
        EPSILON = 5
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        DX = 0,
        DSCALE = 1,
        DBIAS = 2
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    int64_t normalized_dim_count = 0; ///< Normalized Dim Count
    // NOLINTEND(readability-identifier-naming)

    /// @brief Get the dy input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
    /// @brief Get the x input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the scale input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    /// @brief Get the mean input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getInput(InputNames::MEAN);
    }
    /// @brief Get the inv_variance input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getInput(InputNames::INV_VARIANCE);
    }
    /// @brief Get the epsilon input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_epsilon() const
    {
        return getInput(InputNames::EPSILON);
    }
    /// @brief Get the dx output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dx() const
    {
        return getOutput(OutputNames::DX);
    }
    /// @brief Get the dscale output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dscale() const
    {
        return getOutput(OutputNames::DSCALE);
    }
    /// @brief Get the dbias output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dbias() const
    {
        return getOutput(OutputNames::DBIAS);
    }

    /// @brief Set the dy input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    /// @brief Set the dy input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    /// @brief Set the x input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    /// @brief Set the x input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    /// @brief Set the scale input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    /// @brief Set the scale input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    /// @brief Set the mean input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::MEAN, std::move(value));
    }
    /// @brief Set the mean input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::MEAN, value);
    }
    /// @brief Set the inv_variance input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INV_VARIANCE, std::move(value));
    }
    /// @brief Set the inv_variance input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INV_VARIANCE, value);
    }
    /// @brief Set the epsilon input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }
    /// @brief Set the epsilon input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::EPSILON, value);
    }
    /// @brief Set the dx output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_dx(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DX, std::move(value));
    }
    /// @brief Set the dx output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_dx(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DX, value);
    }
    /// @brief Set the dscale output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_dscale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DSCALE, std::move(value));
    }
    /// @brief Set the dscale output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_dscale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DSCALE, value);
    }
    /// @brief Set the dbias output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_dbias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DBIAS, std::move(value));
    }
    /// @brief Set the dbias output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_dbias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DBIAS, value);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormBackwardAttributes& set_normalized_dim_count(int64_t value)
    {
        normalized_dim_count = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int64_t get_normalized_dim_count() const
    {
        return normalized_dim_count;
    }

    LayernormBackwardAttributes&
        set_saved_mean_and_inv_variance(const std::shared_ptr<TensorAttributes>& mean, // NOLINT
                                        const std::shared_ptr<TensorAttributes>& invVariance)
    {
        return set_mean(mean).set_inv_variance(invVariance);
    }

    LayernormBackwardAttributes&
        set_saved_mean_and_inv_variance(std::shared_ptr<TensorAttributes>&& mean, // NOLINT
                                        std::shared_ptr<TensorAttributes>&& invVariance)
    {
        return set_mean(std::move(mean)).set_inv_variance(std::move(invVariance));
    }
};
} // namespace hipdnn_frontend::graph
