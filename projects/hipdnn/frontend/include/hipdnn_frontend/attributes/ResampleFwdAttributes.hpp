// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{

class ResampleFwdAttributes : public Attributes<ResampleFwdAttributes>
{
public:
    ResampleFwdAttributes() = default;

    /// Input tensor identifiers
    // NOLINTNEXTLINE(readability-identifier-naming)
    enum class input_names
    {
        X = 0, ///< Input tensor
    };

    /// Output tensor identifiers
    // NOLINTNEXTLINE(readability-identifier-naming)
    enum class output_names
    {
        Y = 0, ///< Output tensor
        INDEX = 1, ///< Optional index tensor (for max resample)
    };

    std::unordered_map<input_names, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<output_names, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    // NOLINTBEGIN(readability-identifier-naming)
    std::vector<int64_t> pre_padding;
    std::vector<int64_t> post_padding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    ResampleMode resample_mode = ResampleMode::NOT_SET;
    PaddingMode padding_mode = PaddingMode::NOT_SET;
    std::optional<bool> generate_index = std::nullopt;
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(input_names::X);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(output_names::Y);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_index() const
    {
        return getOutput(output_names::INDEX);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_x(const std::shared_ptr<TensorAttributes>& x)
    {
        return setInput(input_names::X, x);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_x(std::shared_ptr<TensorAttributes>&& x)
    {
        return setInput(input_names::X, std::move(x));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_y(const std::shared_ptr<TensorAttributes>& y)
    {
        return setOutput(output_names::Y, y);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_y(std::shared_ptr<TensorAttributes>&& y)
    {
        return setOutput(output_names::Y, std::move(y));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_index(const std::shared_ptr<TensorAttributes>& idx)
    {
        return setOutput(output_names::INDEX, idx);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_index(std::shared_ptr<TensorAttributes>&& idx)
    {
        return setOutput(output_names::INDEX, std::move(idx));
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_pre_padding() const
    {
        return pre_padding;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_pre_padding(std::vector<int64_t> value)
    {
        pre_padding = std::move(value);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_post_padding() const
    {
        return post_padding;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_post_padding(std::vector<int64_t> value)
    {
        post_padding = std::move(value);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_stride() const
    {
        return stride;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_stride(std::vector<int64_t> value)
    {
        stride = std::move(value);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_window() const
    {
        return window;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_window(std::vector<int64_t> value)
    {
        window = std::move(value);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleMode get_resample_mode() const
    {
        return resample_mode;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_resample_mode(ResampleMode value)
    {
        resample_mode = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    PaddingMode get_padding_mode() const
    {
        return padding_mode;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_padding_mode(PaddingMode value)
    {
        padding_mode = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<bool> get_generate_index() const
    {
        return generate_index;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ResampleFwdAttributes& set_generate_index(bool value)
    {
        generate_index = value;
        return *this;
    }
};

typedef ResampleFwdAttributes Resample_fwd_attributes; // NOLINT(readability-identifier-naming)

} // namespace hipdnn_frontend::graph
