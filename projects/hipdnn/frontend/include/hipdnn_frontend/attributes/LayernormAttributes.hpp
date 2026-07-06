// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file LayernormAttributes.hpp
 * @brief Attributes for layer normalization operation
 *
 * This file defines the LayernormAttributes class used to configure
 * layer normalization operations, which normalize across the feature dimension.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class LayernormAttributes
 * @brief Configuration attributes for layer normalization
 *
 * LayernormAttributes configures a layer normalization operation.
 * Unlike batch normalization which normalizes across the batch dimension,
 * layer normalization normalizes across the last k feature dimensions.
 *
 * The number of normalized dimensions (k) is determined by:
 * 1. **From scale shape**: k is inferred from the scale tensor. Dimensions
 *    where scale != 1 are normalized. If scale has fewer dims than X, all
 *    scale dims are normalized dims.
 * 2. **Default**: If scale dims are also unset, they are inferred as
 *    `[1, D1, ..., Dk]` (batch dim = 1), normalizing all non-batch dims.
 *
 * **Tensor Shapes:**
 * - **X** (input): `(N, D1, D2, ..., Dk)` — batch first, then feature dims
 * - **Scale, Bias**: Full-rank `(1, D1, ..., Dk)` with batch dim = 1,
 *   or reduced-rank `(D1, ..., Dk)` with batch dims omitted
 * - **Y** (output): Same shape as X
 * - **Mean, InvVariance**: Batch dims from input, normalized dims = 1
 *
 * **Required inputs:**
 * - X: Input tensor to normalize
 * - Scale: Per-feature scale (gamma) tensor
 * - Bias: Per-feature bias (beta) tensor
 * - Epsilon: Small constant for numerical stability (scalar tensor)
 *
 * **Outputs:**
 * - Y: Normalized output tensor
 * - Mean: Computed mean (optional, training phase only)
 * - InvVariance: Computed inverse variance (1/sqrt(var + epsilon)) (optional, training phase only)
 *
 * @code{.cpp}
 * LayernormAttributes attr;
 * attr.set_forward_phase(NormFwdPhase::INFERENCE);
 * attr.set_epsilon(epsilon);
 *
 * auto [y, mean, inv_variance] = graph.layernorm(x, scale, bias, attr);
 * @endcode
 */
class LayernormAttributes : public Attributes<LayernormAttributes>
{
public:
    LayernormAttributes() = default;

    enum class InputNames
    {
        X = 0,
        SCALE = 1,
        BIAS = 2,
        EPSILON = 3
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0,
        MEAN = 1,
        INV_VARIANCE = 2
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(InputNames::BIAS);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_epsilon() const
    {
        return getInput(InputNames::EPSILON);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getOutput(OutputNames::MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getOutput(OutputNames::INV_VARIANCE);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::EPSILON, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::INV_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::INV_VARIANCE, std::move(value));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_forward_phase(NormFwdPhase value)
    {
        _forwardPhase = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    NormFwdPhase get_forward_phase() const
    {
        return _forwardPhase;
    }

    /// @cond INTERNAL
    // NOLINTNEXTLINE(readability-identifier-naming)
    LayernormAttributes& set_normalized_dim_count(int64_t value)
    {
        _normalizedDimCount = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    int64_t get_normalized_dim_count() const
    {
        return _normalizedDimCount;
    }
    /// @endcond

private:
    int64_t _normalizedDimCount = 0;
    NormFwdPhase _forwardPhase = NormFwdPhase::NOT_SET;
};

typedef LayernormAttributes Layernorm_attributes; // NOLINT(readability-identifier-naming)
} // namespace hipdnn_frontend::graph
