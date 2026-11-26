// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file CustomOpAttributes.hpp
 * @brief Attributes for custom (plugin-provided) operations
 *
 * This file defines the CustomOpAttributes class used to configure opaque
 * plugin operations. Custom ops carry tensor I/O structure (UIDs for graph
 * topology) and an opaque byte payload that a plugin interprets.
 */

#pragma once

#include "GraphAttributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_frontend::graph
{

/**
 * @class CustomOpAttributes
 * @brief Configuration attributes for a custom (plugin-provided) operation
 *
 * Custom ops let users coordinate directly with plugins without requiring
 * hipDNN to understand the operation. hipDNN transports the tensor I/O
 * topology and an opaque byte payload (`data`), and the target plugin
 * interprets the payload at execution time.
 *
 * Unlike other attributes classes, CustomOpAttributes does not inherit from
 * Attributes<> because custom ops use variable-length positional tensor
 * vectors rather than named tensor maps.
 *
 * The `custom_op_id` identifies which plugin handles the operation, using
 * dotted namespace notation: `"<plugin>.<operation>"` (e.g. `"example.rope"`).
 *
 * @see Graph::custom_op()
 * @see CustomOpNode
 */
class CustomOpAttributes
{
public:
    CustomOpAttributes() = default;

    // NOLINTBEGIN(readability-identifier-naming)
    std::string name;
    DataType compute_data_type = DataType::NOT_SET;
    /// Unique identifier for the plugin operation, using dotted namespace
    /// notation: "<plugin>.<operation>" (e.g. "example.rope", "mylib.fused_add").
    std::string custom_op_id;
    std::vector<std::shared_ptr<TensorAttributes>> inputs;
    std::vector<std::shared_ptr<TensorAttributes>> outputs;
    std::vector<uint8_t> data;
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    CustomOpAttributes& set_name(const std::string& nameValue)
    {
        name = nameValue;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::string& get_name() const
    {
        return name;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    CustomOpAttributes& set_compute_data_type(DataType value)
    {
        compute_data_type = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_compute_data_type() const
    {
        return compute_data_type;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    CustomOpAttributes& set_custom_op_id(const std::string& value)
    {
        custom_op_id = value;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::string& get_custom_op_id() const
    {
        return custom_op_id;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    CustomOpAttributes& set_inputs(std::vector<std::shared_ptr<TensorAttributes>> value)
    {
        inputs = std::move(value);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<std::shared_ptr<TensorAttributes>>& get_inputs() const
    {
        return inputs;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    CustomOpAttributes& set_outputs(std::vector<std::shared_ptr<TensorAttributes>> value)
    {
        outputs = std::move(value);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<std::shared_ptr<TensorAttributes>>& get_outputs() const
    {
        return outputs;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    CustomOpAttributes& set_data(std::vector<uint8_t> value)
    {
        data = std::move(value);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<uint8_t>& get_data() const
    {
        return data;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error fill_from_context(const GraphAttributes& graphAttributes)
    {
        for(auto& tensor : inputs)
        {
            if(tensor)
            {
                tensor->fill_from_context(graphAttributes);
            }
        }

        for(auto& tensor : outputs)
        {
            if(tensor)
            {
                tensor->fill_from_context(graphAttributes);
            }
        }

        if(get_compute_data_type() == DataType::NOT_SET)
        {
            set_compute_data_type(graphAttributes.get_compute_data_type());
        }

        return {};
    }

    /**
     * @brief Logical equality: same operation semantics, ignoring the node name and
     * tensor names and UIDs.
     *
     * Compares compute_data_type, custom_op_id and the opaque data payload (both define
     * what the plugin computes), and the positional input and output tensors with
     * TensorAttributes::logicallyEquals. Tensor order matters; a null entry equals only
     * a null entry at the same position.
     */
    bool logicallyEquals(const CustomOpAttributes& other) const
    {
        if(compute_data_type != other.compute_data_type || custom_op_id != other.custom_op_id
           || data != other.data)
        {
            return false;
        }

        const auto logical = [](const TensorAttributes& t1, const TensorAttributes& t2) {
            return t1.logicallyEquals(t2);
        };
        return compareTensorVectors(inputs, other.inputs, logical)
               && compareTensorVectors(outputs, other.outputs, logical);
    }

    /**
     * @brief Strict equality: logical equality plus the node name and strict
     * TensorAttributes equality for every positional tensor.
     */
    bool operator==(const CustomOpAttributes& other) const
    {
        if(name != other.name || compute_data_type != other.compute_data_type
           || custom_op_id != other.custom_op_id || data != other.data)
        {
            return false;
        }

        const auto strict
            = [](const TensorAttributes& t1, const TensorAttributes& t2) { return t1 == t2; };
        return compareTensorVectors(inputs, other.inputs, strict)
               && compareTensorVectors(outputs, other.outputs, strict);
    }

    bool operator!=(const CustomOpAttributes& other) const
    {
        return !(*this == other);
    }

private:
    // True if v1 and v2 have the same length and, position by position, both entries are
    // null or both are non-null and `equal` holds for the pointed-to tensors.
    template <typename EqualT>
    static bool compareTensorVectors(const std::vector<std::shared_ptr<TensorAttributes>>& v1,
                                     const std::vector<std::shared_ptr<TensorAttributes>>& v2,
                                     EqualT equal)
    {
        if(v1.size() != v2.size())
        {
            return false;
        }

        for(size_t i = 0; i < v1.size(); ++i)
        {
            if(!v1[i] || !v2[i])
            {
                if(v1[i] != v2[i])
                {
                    return false;
                }
                continue;
            }

            if(!equal(*v1[i], *v2[i]))
            {
                return false;
            }
        }

        return true;
    }
};

typedef CustomOpAttributes Custom_op_attributes; // NOLINT(readability-identifier-naming)

} // namespace hipdnn_frontend::graph
