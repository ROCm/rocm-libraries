// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BlockScaleDequantizeAttributes.hpp
 * @brief Attributes for block scale dequantize operation
 *
 * This file defines the BlockScaleDequantizeAttributes class used to configure
 * block scaling (dequantize) operations. This is a generic scaling operation
 * that supports all MX blocked low-precision data-types (mxfp8, mxbfp8, mxfp6, mxfp4).
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class BlockScaleDequantizeAttributes
 * @brief Configuration attributes for block scale dequantize operation
 *
 * BlockScaleDequantizeAttributes configures a block scaling dequantize operation
 * that supports all MX blocked low-precision data-types. The operation takes a
 * blocked input tensor and a scale tensor, producing a dequantized output tensor.
 *
 * **Required inputs:**
 * - X: Input blocked tensor to dequantize
 * - Scale: Scale tensor for block dequantization
 *
 * **Required outputs:**
 * - Y: Dequantized output tensor
 *
 * **Required parameters:**
 * - block_size: Vector of block sizes for each dimension
 *
 * **Optional parameters:**
 * - is_negative_scale: Whether the scale is negative (default: false)
 *
 * @code{.cpp}
 * BlockScaleDequantizeAttributes attr;
 * attr.set_x(inputTensor)
 *     .set_scale(scaleTensor)
 *     .set_block_size({32})
 *     .set_is_negative_scale(false);
 *
 * auto y = graph.block_scale_dequantize(x, scale, attr);
 * @endcode
 */
class BlockScaleDequantizeAttributes : public Attributes<BlockScaleDequantizeAttributes>
{
public:
    BlockScaleDequantizeAttributes() = default;

    enum class InputNames
    {
        X = 0,
        SCALE = 1
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0
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
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int32_t>& get_block_size() const
    {
        return block_size;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_block_size(int32_t value, int32_t idx = 0)
    {
        if(idx < 0)
        {
            return *this;
        }
        if(static_cast<int32_t>(block_size.size()) < idx + 1)
        {
            block_size.resize(static_cast<size_t>(idx) + 1, 1);
        }
        block_size[static_cast<size_t>(idx)] = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_block_size(const int32_t* values, int32_t len = 1)
    {
        if(len < 1)
        {
            return *this;
        }
        block_size.resize(static_cast<size_t>(len));
        std::copy(values, values + len, block_size.begin());
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_block_size(const std::vector<int32_t>& values)
    {
        block_size = values;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_is_negative_scale() const
    {
        return is_negative_scale;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleDequantizeAttributes& set_is_negative_scale(bool value)
    {
        is_negative_scale = value;
        return *this;
    }

private:
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::vector<int32_t> block_size;
    // NOLINTNEXTLINE(readability-identifier-naming)
    bool is_negative_scale = false;
};

using Block_scale_dequantize_attributes // NOLINT(readability-identifier-naming)
    = BlockScaleDequantizeAttributes;

} // namespace hipdnn_frontend::graph
