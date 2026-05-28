// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BatchnormAttributes.hpp
 * @brief Attributes for batch normalization forward training operation
 *
 * This file defines the BatchnormAttributes class used to configure
 * batch normalization operations during training, including computing
 * mean and variance statistics.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class BatchnormAttributes
 * @brief Configuration attributes for batch normalization forward training
 *
 * BatchnormAttributes configures a batch normalization operation for training.
 * This operation normalizes the input tensor across the batch dimension and
 * computes running statistics for inference.
 *
 * **Tensor Shapes:**
 * - **X** (input): `(N, C, H, W)` or `(N, C, D, H, W)` — batch, channels, spatial dims
 * - **Scale, Bias**: `(1, C, 1, 1)` or `(1, C, 1, 1, 1)` — per-channel parameters
 * - **Y** (output): Same shape as X
 * - **Mean, Inv_variance**: `(1, C, 1, 1)` or `(1, C, 1, 1, 1)` — per-channel statistics
 *
 * **Required inputs:**
 * - X: Input tensor to normalize
 * - Scale: Per-channel scale (gamma) tensor
 * - Bias: Per-channel bias (beta) tensor
 *
 * **Optional inputs (for running statistics):**
 * - prev_running_mean: Previous running mean (for exponential moving average)
 * - prev_running_variance: Previous running variance
 * - momentum: Momentum for running statistics update
 *
 * **Outputs:**
 * - Y: Normalized output tensor
 * - Mean: Computed batch mean
 * - Inv_variance: Computed inverse variance (1/sqrt(var + epsilon))
 * - next_running_mean: Updated running mean (optional)
 * - next_running_variance: Updated running variance (optional)
 *
 * @code{.cpp}
 * BatchnormAttributes attr;
 * attr.set_x(inputTensor)
 *     .set_scale(scaleTensor)
 *     .set_bias(biasTensor)
 *     .set_epsilon(epsilonTensor);
 *
 * auto [y, mean, invVar, nextMean, nextVar] = graph.batchnorm(x, scale, bias, attr);
 * @endcode
 *
 * @see BatchnormInferenceAttributes for inference-only batch normalization
 * @see BatchnormBackwardAttributes for backward pass
 */
class BatchnormAttributes : public Attributes<BatchnormAttributes>
{
public:
    BatchnormAttributes() = default;

    enum class InputNames
    {
        X = 0,
        SCALE = 1,
        BIAS = 2,
        PREV_RUNNING_MEAN = 3,
        PREV_RUNNING_VARIANCE = 4,
        MOMENTUM = 5,
        EPSILON = 6
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0,
        MEAN = 1,
        INV_VARIANCE = 2,
        NEXT_RUNNING_MEAN = 3,
        NEXT_RUNNING_VARIANCE = 4
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::vector<std::shared_ptr<TensorAttributes>> peer_stats;

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
    const std::vector<std::shared_ptr<TensorAttributes>>& get_peer_stats() const
    {
        return peer_stats;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_prev_running_mean() const
    {
        return getInput(InputNames::PREV_RUNNING_MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_prev_running_variance() const
    {
        return getInput(InputNames::PREV_RUNNING_VARIANCE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_momentum() const
    {
        return getInput(InputNames::MOMENTUM);
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
    std::shared_ptr<TensorAttributes> get_next_running_mean() const
    {
        return getOutput(OutputNames::NEXT_RUNNING_MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_next_running_variance() const
    {
        return getOutput(OutputNames::NEXT_RUNNING_VARIANCE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::EPSILON, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_peer_stats(const std::vector<std::shared_ptr<TensorAttributes>>& value)
    {
        peer_stats = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_peer_stats(std::vector<std::shared_ptr<TensorAttributes>>&& value)
    {
        peer_stats = std::move(value);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_prev_running_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::PREV_RUNNING_MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_prev_running_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::PREV_RUNNING_MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_prev_running_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::PREV_RUNNING_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_prev_running_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::PREV_RUNNING_VARIANCE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_momentum(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::MOMENTUM, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_momentum(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::MOMENTUM, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::INV_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::INV_VARIANCE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_next_running_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::NEXT_RUNNING_MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_next_running_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::NEXT_RUNNING_MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_next_running_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::NEXT_RUNNING_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_next_running_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::NEXT_RUNNING_VARIANCE, std::move(value));
    }
    // NOLINTBEGIN(readability-identifier-naming)
    BatchnormAttributes&
        set_previous_running_stats(const std::shared_ptr<TensorAttributes>& mean,
                                   const std::shared_ptr<TensorAttributes>& variance,
                                   const std::shared_ptr<TensorAttributes>& momentum)
    // NOLINTEND(readability-identifier-naming)
    {
        return set_prev_running_mean(mean).set_prev_running_variance(variance).set_momentum(
            momentum);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormAttributes& set_previous_running_stats(std::shared_ptr<TensorAttributes>&& mean,
                                                    std::shared_ptr<TensorAttributes>&& variance,
                                                    std::shared_ptr<TensorAttributes>&& momentum)
    {
        return set_prev_running_mean(std::move(mean))
            .set_prev_running_variance(std::move(variance))
            .set_momentum(std::move(momentum));
    }
};

typedef BatchnormAttributes Batchnorm_attributes;
} // namespace hipdnn_frontend::graph
