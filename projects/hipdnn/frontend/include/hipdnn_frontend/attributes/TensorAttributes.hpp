// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TensorAttributes.hpp
 * @brief Tensor configuration and attributes for hipDNN Frontend operations
 *
 * This file defines the TensorAttributes class which is used to configure
 * tensor properties like dimensions, strides, data type, and unique identifiers.
 * Tensors are the fundamental data containers in hipDNN computational graphs.
 */

#pragma once

#include "GraphAttributes.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace hipdnn_frontend::graph
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/**
 * @class TensorAttributes
 * @brief Describes the properties and configuration of a tensor
 *
 * TensorAttributes is used to define tensors that participate in hipDNN operations.
 * Each tensor has dimensions, strides, a data type, and optionally a unique identifier
 * for mapping to device memory during execution.
 *
 * Tensors can be:
 * - **Physical tensors**: Have a UID and map to actual device memory
 * - **Virtual tensors**: Intermediate results that don't require explicit memory allocation
 * - **Pass-by-value tensors**: Scalar values embedded directly in the tensor
 *
 * @note **Dimension Ordering**: The expected dimension ordering depends on the operation.
 * Convolution and batch normalization tensors use `(N, C, H, W)` or `(N, C, D, H, W)`.
 * Matmul tensors use `(...batch, M, K)` for A and `(...batch, K, N)` for B.
 * Pointwise operations accept any shape with broadcasting support.
 * In all cases, the memory layout is controlled by strides, not by dimension order in the tensor shape vector.
 * Use `hipdnn_data_sdk::utilities::generateStrides()` to compute strides from a TensorLayout.
 *
 * @code{.cpp}
 * // Create a 4D convolution input tensor
 * // For convolution, dimensions follow (N, C, H, W) ordering
 * auto x = Graph::tensor(TensorAttributes()
 *              .set_dim({1, 64, 28, 28})   // dims: N=1, C=64, H=28, W=28
 *              .set_stride({50176, 784, 28, 1})  // NCHW layout strides
 *              .set_data_type(DataType::HALF)
 *              .set_uid(0)
 *              .set_name("input_x"));
 *
 * // Same dimensions with NHWC (channel-last) layout
 * auto x_nhwc = Graph::tensor(TensorAttributes()
 *              .set_dim({1, 64, 28, 28})   // dims: N=1, C=64, H=28, W=28
 *              .set_stride({50176, 1, 1792, 64})  // NHWC layout strides
 *              .set_data_type(DataType::HALF)
 *              .set_uid(1)
 *              .set_name("input_x_nhwc"));
 *
 * // Create a scalar tensor
 * TensorAttributes scalar(2.0f);  // Pass-by-value float
 * @endcode
 */
class TensorAttributes
{
public:
    /// Variant type for storing pass-by-value scalar values
    using ValueVariant
        = std::variant<std::monostate, double, float, half, bfloat16, uint8_t, int32_t, int64_t>;

    /// @brief Default constructor
    TensorAttributes() = default;

    /**
     * @brief Construct a pass-by-value tensor from a scalar
     * @tparam T Scalar type (float, double, half, hip_bfloat16, uint8_t, int32_t, int64_t)
     * @param scalar The scalar value to store in the tensor
     */
    template <typename T>
    TensorAttributes(const T& scalar)
    {
        set_value(scalar);
    }

    /**
     * @brief Check if this tensor is a pass-by-value tensor
     * @return true if the tensor contains an embedded scalar value
     */
    bool get_pass_by_value() const // NOLINT(readability-identifier-naming)
    {
        return !std::holds_alternative<std::monostate>(_value);
    }

    /**
     * @brief Get the pass-by-value scalar of a specific type
     * @tparam T The expected scalar type
     * @return The scalar value if it matches type T, std::nullopt otherwise
     */
    template <typename T>
    std::optional<T> get_pass_by_value() const // NOLINT(readability-identifier-naming)
    {
        if(auto p = std::get_if<T>(&_value))
        {
            return *p;
        }
        return std::nullopt;
    }

    /**
     * @brief Set a pass-by-value scalar in this tensor
     * @tparam T Scalar type (float, double, half, hip_bfloat16, uint8_t, int32_t, int64_t)
     * @param v The scalar value
     * @return Reference to this for method chaining
     */
    template <typename T>
    TensorAttributes& set_value(T v) // NOLINT(readability-identifier-naming)
    {

        static_assert(std::disjunction_v<std::is_same<T, float>,
                                         std::is_same<T, double>,
                                         std::is_same<T, half>,
                                         std::is_same<T, bfloat16>,
                                         std::is_same<T, uint8_t>,
                                         std::is_same<T, int32_t>,
                                         std::is_same<T, int64_t>>,
                      "Unsupported type for Tensor_attributes::set_value");
        _value = v;
        _dataType = getDataTypeEnumFromType<T>();
        _dim = _stride = {1};
        return *this;
    }

    /**
     * @brief Get the raw value variant for type-erased access to the scalar value
     * @return Const reference to the internal ValueVariant
     */
    const ValueVariant& get_value_variant() const // NOLINT(readability-identifier-naming)
    {
        return _value;
    }

    /**
     * @brief Clear the pass-by-value scalar
     * @return Reference to this for method chaining
     */
    TensorAttributes& clear_value() // NOLINT(readability-identifier-naming)
    {
        _value = {};
        return *this;
    }

    /**
     * @brief Get the unique identifier of this tensor
     * @return The tensor UID
     */
    int64_t get_uid() const // NOLINT(readability-identifier-naming)
    {
        return _uid;
    }

    /**
     * @brief Get the name of this tensor
     * @return The tensor name
     */
    const std::string& get_name() const // NOLINT(readability-identifier-naming)
    {
        return _name;
    }

    /**
     * @brief Get the data type of this tensor
     * @return The DataType enum value
     */
    DataType get_data_type() const // NOLINT(readability-identifier-naming)
    {
        return _dataType;
    }

    /**
     * @brief Get the strides of this tensor
     * @return Vector of strides for each dimension
     */
    const std::vector<int64_t>& get_stride() const // NOLINT(readability-identifier-naming)
    {
        return _stride;
    }

    /**
     * @brief Get the dimensions of this tensor
     * @return Vector of dimension sizes
     */
    const std::vector<int64_t>& get_dim() const // NOLINT(readability-identifier-naming)
    {
        return _dim;
    }

    /**
     * @brief Get the total number of elements in this tensor
     * @return Product of all dimension sizes
     */
    int64_t get_volume() const // NOLINT(readability-identifier-naming)
    {
        int64_t volume = 1;
        for(const auto& d : _dim)
        {
            volume *= d;
        }
        return volume;
    }

    /**
     * @brief Check if this tensor is virtual (intermediate result)
     * @return true if virtual, false if physical (requires memory allocation)
     */
    bool get_is_virtual() const // NOLINT(readability-identifier-naming)
    {
        return _isVirtual;
    }

    /**
     * @brief Check if this tensor has a UID assigned
     * @return true if a UID has been set
     */
    bool has_uid() const // NOLINT(readability-identifier-naming)
    {
        return _uidSet;
    }

    /**
     * @brief Set the unique identifier for this tensor
     * @param uid The unique identifier (used for memory mapping during execution)
     * @return Reference to this for method chaining
     */
    TensorAttributes& set_uid(int64_t uid) // NOLINT(readability-identifier-naming)
    {
        _uid = uid;
        _uidSet = true;
        return *this;
    }

    /**
     * @brief Set a human-readable name for this tensor
     * @param name The tensor name (for debugging and logging)
     * @return Reference to this for method chaining
     */
    TensorAttributes& set_name(const std::string& name) // NOLINT(readability-identifier-naming)
    {
        _name = name;
        return *this;
    }

    /**
     * @brief Set the data type of this tensor
     * @param dataType The DataType enum value
     * @return Reference to this for method chaining
     */
    TensorAttributes& set_data_type(DataType dataType) // NOLINT(readability-identifier-naming)
    {
        _dataType = dataType;
        return *this;
    }

    /**
     * @brief Set the strides for this tensor
     * @param stride Vector of strides for each dimension
     * @return Reference to this for method chaining
     *
     * @note Strides must have the same size as dimensions
     */
    TensorAttributes&
        set_stride(const std::vector<int64_t>& stride) // NOLINT(readability-identifier-naming)
    {
        _stride = stride;
        return *this;
    }

    /**
     * @brief Set the dimensions for this tensor
     * @param dim Vector of dimension sizes
     * @return Reference to this for method chaining
     *
     * @note The expected dimension ordering depends on the operation type:
     *       convolution and batch normalization use (N, C, H, W) / (N, C, D, H, W),
     *       matmul uses (...batch, M, K) / (...batch, K, N),
     *       and pointwise operations accept any shape.
     *       Memory layout is always controlled by strides and stride order, not by dimension order in the tensor shape vector.
     */
    TensorAttributes&
        set_dim(const std::vector<int64_t>& dim) // NOLINT(readability-identifier-naming)
    {
        _dim = dim;
        return *this;
    }

    /**
     * @brief Set whether this tensor is virtual
     * @param isVirtual true for virtual tensors (intermediates), false for physical
     * @return Reference to this for method chaining
     */
    TensorAttributes& set_is_virtual(bool isVirtual) // NOLINT(readability-identifier-naming)
    {
        _isVirtual = isVirtual;
        return *this;
    }

    /**
     * @brief Convenience method to mark tensor as output (non-virtual)
     * @param output true to mark as output, false to mark as virtual
     * @return Reference to this for method chaining
     */
    TensorAttributes& set_output(bool output) // NOLINT(readability-identifier-naming)
    {
        return set_is_virtual(!output);
    }

    /**
     * @brief Clear the UID from this tensor
     * @return Reference to this for method chaining
     */
    TensorAttributes& clear_uid() // NOLINT(readability-identifier-naming)
    {
        _uid = 0;
        _uidSet = false;
        return *this;
    }

    /**
     * @brief Fill unset attributes from graph context
     * @param graphAttributes The graph attributes to inherit from
     * @return Reference to this for method chaining
     *
     * If data type is not set, it will be inferred from the graph's
     * io_data_type (for physical tensors) or intermediate_data_type (for virtual tensors).
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    TensorAttributes& fill_from_context(const GraphAttributes& graphAttributes)
    {
        if(_dataType == DataType::NOT_SET)
        {
            if(_isVirtual)
            {
                _dataType = graphAttributes.get_intermediate_data_type();
            }
            else
            {
                _dataType = graphAttributes.get_io_data_type();
            }
        }

        return *this;
    }

    /**
     * @brief Validate tensor attributes
     * @return Error indicating success or describing what is invalid
     *
     * Checks that:
     * - Data type is set
     * - Virtual tensors are not pass-by-value
     * - Dimensions and strides have matching sizes
     * - Dimensions are non-empty and positive
     */
    Error validate() const
    {
        if(_dataType == DataType::NOT_SET)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "Tensor " + _name + " does not have a data type set"};
        }

        HIPDNN_RETURN_IF_TRUE(_isVirtual && get_pass_by_value(),
                              ErrorCode::INVALID_VALUE,
                              "Tensor " + _name + " cannot be virtual and pass by value");
        HIPDNN_RETURN_IF_NE(_dim.size(),
                            _stride.size(),
                            ErrorCode::INVALID_VALUE,
                            "Tensor " + _name + " dims and strides have different sizes");

        HIPDNN_RETURN_IF_TRUE(_dim.empty(),
                              ErrorCode::ATTRIBUTE_NOT_SET,
                              "Tensor " + _name + " dims must be non-empty");

        auto isPositive = [](int64_t value) constexpr { return value > 0; };
        HIPDNN_RETURN_IF_FALSE(std::all_of(_dim.begin(), _dim.end(), isPositive),
                               ErrorCode::INVALID_VALUE,
                               "Tensor " + _name + " must have only positive dimensions");

        return {ErrorCode::OK, ""};
    }

private:
    int64_t _uid = 0;
    bool _uidSet = false;
    std::string _name;
    DataType _dataType = DataType::NOT_SET;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _dim;
    bool _isVirtual = false;
    ValueVariant _value;
};
typedef TensorAttributes Tensor_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
