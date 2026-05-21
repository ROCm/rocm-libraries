// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file RMSNormBackwardAttributes.hpp
 * @brief Attributes for RMS normalization backward pass operation
 *
 * This file defines the RMSNormBackwardAttributes class used to configure
 * the backward pass (gradient computation) of RMS normalization.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class RMSNormBackwardAttributes
 * @brief Configuration attributes for RMS normalization backward pass
 *
 * RMSNormBackwardAttributes configures the backward pass of RMS normalization,
 * computing gradients with respect to the input, scale, and optionally bias.
 *
 * **Required inputs:**
 * - DY: Gradient of the loss with respect to the output (upstream gradient)
 * - X: Original input tensor from forward pass
 * - Scale: Per-channel scale (gamma) tensor
 * - Inv_rms: Saved inverse RMS from forward pass
 *
 * **Outputs:**
 * - DX: Gradient with respect to input
 * - DScale: Gradient with respect to scale (gamma)
 * - DBias: Gradient with respect to bias (beta), optional
 */
class RMSNormBackwardAttributes : public Attributes<RMSNormBackwardAttributes>
{
public:
    RMSNormBackwardAttributes() = default;

    enum class InputNames
    {
        DY = 0,
        X = 1,
        SCALE = 2,
        INV_RMS = 3
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        DX = 0,
        DSCALE = 1,
        DBIAS = 2
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
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
    std::shared_ptr<TensorAttributes> get_inv_rms() const
    {
        return getInput(InputNames::INV_RMS);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dx() const
    {
        return getOutput(OutputNames::DX);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dscale() const
    {
        return getOutput(OutputNames::DSCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dbias() const
    {
        return getOutput(OutputNames::DBIAS);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_inv_rms(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INV_RMS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_inv_rms(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INV_RMS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dx(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DX, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dx(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DX, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dscale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DSCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dscale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DSCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dbias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DBIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dbias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DBIAS, std::move(value));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_compute_dbias() const
    {
        return compute_dbias;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_compute_dbias(bool value)
    {
        compute_dbias = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& has_dbias(bool value)
    {
        return set_compute_dbias(value);
    }

private:
    // NOLINTNEXTLINE(readability-identifier-naming)
    bool compute_dbias = false;
};

using Rmsnorm_backward_attributes
    = RMSNormBackwardAttributes; // NOLINT(readability-identifier-naming)

} // namespace hipdnn_frontend::graph
