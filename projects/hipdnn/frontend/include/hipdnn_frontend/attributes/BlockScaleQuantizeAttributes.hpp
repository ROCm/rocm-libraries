// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BlockScaleQuantizeAttributes.hpp
 * @brief Attributes for block scale quantize operation
 *
 * This file defines the BlockScaleQuantizeAttributes class used to configure
 * block scaling (quantize) operations. This is a generic scaling operation
 * that supports all MX blocked low-precision data-types (mxfp8, mxbfp8, mxfp6, mxfp4).
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class BlockScaleQuantizeAttributes
 * @brief Configuration attributes for block scale quantize operation
 *
 * BlockScaleQuantizeAttributes configures a block scaling quantize operation
 * that supports all MX blocked low-precision data-types. The operation takes an
 * input tensor and produces a quantized output tensor and a scale tensor.
 *
 * **Required inputs:**
 * - X: Input tensor to quantize
 *
 * **Required outputs:**
 * - Y: Quantized output tensor
 * - Scale: Scale tensor computed during quantization
 *
 * **Optional parameters:**
 * - block_size: Block size for quantization (default: not set)
 * - axis: Axis along which to apply block scaling (default: not set)
 * - transpose: Whether to transpose (default: false)
 *
 * @code{.cpp}
 * BlockScaleQuantizeAttributes attr;
 * attr.set_block_size(32)
 *     .set_axis(1)
 *     .set_transpose(false);
 *
 * auto [y, scale] = graph.block_scale_quantize(x, attr);
 * @endcode
 */
class BlockScaleQuantizeAttributes : public Attributes<BlockScaleQuantizeAttributes>
{
public:
    BlockScaleQuantizeAttributes() = default;

    enum class InputNames
    {
        X = 0
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0,
        SCALE = 1
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
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getOutput(OutputNames::SCALE);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::SCALE, std::move(value));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int32_t> get_block_size() const
    {
        return block_size;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_block_size(int32_t value)
    {
        block_size = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int64_t> get_axis() const
    {
        return axis;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_axis(int64_t value)
    {
        axis = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_transpose() const
    {
        return transpose;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_transpose(bool value)
    {
        transpose = value;
        return *this;
    }

private:
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int32_t> block_size;
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int64_t> axis;
    // NOLINTNEXTLINE(readability-identifier-naming)
    bool transpose = false;
};

using Block_scale_quantize_attributes // NOLINT(readability-identifier-naming)
    = BlockScaleQuantizeAttributes;

} // namespace hipdnn_frontend::graph
