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
#include <hipdnn_data_sdk/data_objects/rmsnorm_attributes_generated.h>
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

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::RMSNormAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto invRms = get_inv_rms();
        auto bias = get_bias();

        return hipdnn_data_sdk::data_objects::CreateRMSNormAttributes(
            builder,
            get_x()->get_uid(),
            get_scale()->get_uid(),
            get_epsilon()->get_uid(),
            get_y()->get_uid(),
            bias ? flatbuffers::Optional<int64_t>(bias->get_uid()) : flatbuffers::nullopt,
            invRms ? flatbuffers::Optional<int64_t>(invRms->get_uid()) : flatbuffers::nullopt,
            toSdkType(forward_phase));
    }

    static RMSNormAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::RMSNormAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        RMSNormAttributes attr;

        attr.set_x(tensorMap.at(fb->x_tensor_uid()));
        attr.set_scale(tensorMap.at(fb->scale_tensor_uid()));
        attr.set_epsilon(tensorMap.at(fb->epsilon_tensor_uid()));
        attr.set_y(tensorMap.at(fb->y_tensor_uid()));

        if(fb->inv_rms_tensor_uid().has_value())
        {
            attr.set_inv_rms(tensorMap.at(fb->inv_rms_tensor_uid().value()));
        }

        if(fb->bias_tensor_uid().has_value())
        {
            attr.set_bias(tensorMap.at(fb->bias_tensor_uid().value()));
        }

        attr.set_forward_phase(fromSdkType(fb->forward_phase()));

        return attr;
    }
};

using Rmsnorm_attributes = RMSNormAttributes; // NOLINT(readability-identifier-naming)

} // namespace hipdnn_frontend::graph
