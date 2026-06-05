// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file RMSNormAttributes.hpp
 * @brief Attributes for RMS normalization forward operation
 *
 * This file defines the RMSNormAttributes class used to configure
 * RMS normalization operations, supporting both training and inference phases.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class RMSNormAttributes
 * @brief Configuration attributes for RMS normalization forward operation
 *
 * RMSNormAttributes configures an RMS normalization operation. Unlike batch
 * normalization, RMSNorm normalizes using only the root mean square of the
 * input without subtracting the mean, and operates per-channel rather than
 * across the batch dimension.
 *
 * **Required inputs:**
 * - X: Input tensor to normalize
 * - Scale: Per-channel scale (gamma) tensor with shape [1, C, 1, 1, ...]
 * - Epsilon: Scalar numerical stability constant
 *
 * **Optional inputs:**
 * - Bias: Per-channel bias (beta) tensor with shape [1, C, 1, 1, ...]
 *
 * **Required outputs:**
 * - Y: Normalized output tensor (same shape as X)
 *
 * **Optional outputs (training only):**
 * - Inv_rms: Inverse RMS values saved for the backward pass
 *
 * **Required parameters:**
 * - forward_phase: Must be set to TRAINING or INFERENCE. In TRAINING mode,
 *   the inv_rms output is computed and saved. In INFERENCE mode, only Y
 *   is produced.
 *
 * @code{.cpp}
 * RMSNormAttributes attr;
 * attr.set_x(inputTensor)
 *     .set_scale(scaleTensor)
 *     .set_epsilon(epsilonTensor)
 *     .set_forward_phase(NormFwdPhase::TRAINING);
 *
 * auto [y, invRms] = graph.rmsnorm(attr);
 * @endcode
 *
 * @see BatchnormAttributes for batch normalization
 */
class RMSNormAttributes : public Attributes<RMSNormAttributes>
{
public:
    RMSNormAttributes() = default;

    enum class InputNames
    {
        X = 0,
        SCALE = 1,
        EPSILON = 2,
        BIAS = 3
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0,
        INV_RMS = 1
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    NormFwdPhase forward_phase = NormFwdPhase::NOT_SET; // NOLINT(readability-identifier-naming)

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
    std::shared_ptr<TensorAttributes> get_epsilon() const
    {
        return getInput(InputNames::EPSILON);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(InputNames::BIAS);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_rms() const
    {
        return getOutput(OutputNames::INV_RMS);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::EPSILON, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_inv_rms(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::INV_RMS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_inv_rms(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::INV_RMS, std::move(value));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    NormFwdPhase get_forward_phase() const
    {
        return forward_phase;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormAttributes& set_forward_phase(NormFwdPhase phase)
    {
        forward_phase = phase;
        return *this;
    }
};

using Rmsnorm_attributes = RMSNormAttributes; // NOLINT(readability-identifier-naming)

} // namespace hipdnn_frontend::graph
