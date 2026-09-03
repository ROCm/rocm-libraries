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
#include <hipdnn_frontend/detail/TensorConstants.hpp>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace hipdnn_frontend::graph
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

/// @brief Selects how a pass-by-value scalar is treated: a runtime parameter
/// (supplied via the variant pack at execute time) or a compile-time constant.
enum class ScalarType
{
    RUNTIME_PARAM,
    COMPILE_TIME_CONST
};

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
    using ValueVariant = std::
        variant<std::monostate, double, float, half, bfloat16, uint8_t, int32_t, int64_t, bool>;

    /// cuDNN-parity alias for the pass-by-value scalar variant type.
    using pass_by_values_t = ValueVariant; // NOLINT(readability-identifier-naming)

    /// cuDNN-parity alias for the tensor UID type; consumers spell variant-pack
    /// map keys as `TensorAttributes::uid_t`.
    using uid_t = int64_t; // NOLINT(readability-identifier-naming)

    /// @brief Default constructor
    TensorAttributes() = default;

    /**
     * @brief Construct a compile-time-constant pass-by-value tensor from a scalar
     * @tparam T Scalar type (float, double, half, hip_bfloat16, uint8_t, int32_t, int64_t, bool)
     * @param scalar The scalar value to bake into the tensor
     *
     * Delegates to set_value(), which bakes a baseline-1.0.0 compile-time
     * constant. See RFC 0016 §4.3. Use the (scalar, ScalarType::RUNTIME_PARAM)
     * constructor, set_as_runtime_parameter(), or set_is_pass_by_value(true)
     * for a runtime-with-default scalar that floors the provider at 1.2.0.
     */
    template <typename T>
    TensorAttributes(const T& scalar)
    {
        set_value(scalar);
    }

    /**
     * @brief Construct a pass-by-value tensor from a scalar with an explicit mode
     * @tparam T Scalar type
     * @param scalar The scalar value to store in the tensor
     * @param type RUNTIME_PARAM => runtime-with-default; COMPILE_TIME_CONST => compile-time constant
     */
    template <typename T>
    TensorAttributes(const T& scalar, ScalarType type)
    {
        set_value(scalar);
        if(type == ScalarType::RUNTIME_PARAM)
        {
            _isRuntimePassByValue = true;
        }
    }

    /**
     * @brief Check if this tensor is a pass-by-value tensor
     * @return true if the tensor contains an embedded scalar value
     */
    bool get_is_pass_by_value() const // NOLINT(readability-identifier-naming)
    {
        return _isRuntimePassByValue || hasValue();
    }

    /**
     * @brief Get the runtime pass-by-value scalar (cuDNN parity).
     * @return The value variant iff runtime-with-default (runtime flag set and a
     *         value present); std::nullopt otherwise. Inspect the active type with
     *         std::holds_alternative / std::get on the returned variant.
     */
    std::optional<pass_by_values_t>
        get_pass_by_value() const // NOLINT(readability-identifier-naming)
    {
        return _isRuntimePassByValue && hasValue() ? std::optional<pass_by_values_t>{_value}
                                                   : std::nullopt;
    }

    /**
     * @brief Typed convenience wrapper over get_pass_by_value().
     * @tparam T The expected scalar type
     * @return The value iff runtime-with-default and the stored scalar is a T;
     *         std::nullopt otherwise (absent, or present but a different type).
     */
    template <typename T>
    std::optional<T> get_pass_by_value() const // NOLINT(readability-identifier-naming)
    {
        const std::optional<pass_by_values_t> value = get_pass_by_value();
        if(value && std::holds_alternative<T>(*value))
        {
            return std::get<T>(*value);
        }
        return std::nullopt;
    }

    /**
     * @brief Get the compile-time constant scalar (cuDNN parity).
     * @return The value variant iff compile-time constant (runtime flag clear and
     *         a value present); std::nullopt otherwise.
     */
    std::optional<pass_by_values_t>
        get_compile_time_constant() const // NOLINT(readability-identifier-naming)
    {
        return !_isRuntimePassByValue && hasValue() ? std::optional<pass_by_values_t>{_value}
                                                    : std::nullopt;
    }

    /**
     * @brief Typed convenience wrapper over get_compile_time_constant().
     * @tparam T The expected scalar type
     * @return The value iff a compile-time constant whose stored scalar is a T;
     *         std::nullopt otherwise (absent, or present but a different type).
     */
    template <typename T>
    std::optional<T> get_compile_time_constant() const // NOLINT(readability-identifier-naming)
    {
        const std::optional<pass_by_values_t> value = get_compile_time_constant();
        if(value && std::holds_alternative<T>(*value))
        {
            return std::get<T>(*value);
        }
        return std::nullopt;
    }

    /**
     * @brief Check whether this tensor carries a compile-time constant value
     */
    bool get_has_compile_time_constant() const // NOLINT(readability-identifier-naming)
    {
        return !_isRuntimePassByValue && hasValue();
    }

    /**
     * @brief Raw accessor for the stored runtime pass-by-value flag
     */
    bool get_is_runtime_pass_by_value() const // NOLINT(readability-identifier-naming)
    {
        return _isRuntimePassByValue;
    }

    /**
     * @brief Set a compile-time-constant scalar in this tensor
     * @tparam T Scalar type (float, double, half, hip_bfloat16, uint8_t, int32_t, int64_t, bool)
     * @param v The scalar value to bake into the tensor
     * @return Reference to this for method chaining
     *
     * Bakes a baseline-1.0.0 compile-time constant (clears the runtime
     * pass-by-value flag). Use set_as_runtime_parameter(),
     * set_is_pass_by_value(true), or the (scalar, ScalarType::RUNTIME_PARAM)
     * constructor for a runtime-with-default scalar that floors the provider
     * at 1.2.0.
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
                                         std::is_same<T, int64_t>,
                                         std::is_same<T, bool>>,
                      "Unsupported type for Tensor_attributes::set_value");
        _value = v;
        _isRuntimePassByValue = false;
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
     * @brief Set a compile-time constant scalar in this tensor (cuDNN parity).
     * @param v The scalar value, as a pass_by_values_t variant
     * @return Reference to this for method chaining
     *
     * Delegates to the typed set_value (via std::visit) so the per-scalar data
     * type is derived from the active alternative; set_value() itself bakes a
     * baseline-1.0.0 compile-time constant, so this is now a thin cuDNN-parity
     * alias with an ergonomic pass_by_values_t-variant signature. A
     * std::monostate (empty) variant is a no-op guarded here.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    TensorAttributes& set_compile_time_constant(const pass_by_values_t& v)
    {
        std::visit(
            [this](const auto& scalar) {
                if constexpr(!std::is_same_v<std::decay_t<decltype(scalar)>, std::monostate>)
                {
                    set_value(scalar);
                }
            },
            v);
        return *this;
    }

    /**
     * @brief Mark this tensor as a runtime pass-by-value parameter, clearing any stored value
     * @return Reference to this for method chaining
     *
     * Plugin-side resolution (hipdnn_plugin_sdk::ScalarOperand/resolveScalarOperand) supports
     * DOUBLE, FLOAT, HALF, BFLOAT16, INT32, INT64, and BOOLEAN. UINT8, INT8, and the FP8/FP6/FP4
     * families are accepted by the frontend/flatbuffer value union but are not yet wired into
     * the plugin SDK's scalar resolution; a runtime pass-by-value tensor set to one of those
     * types throws HIPDNN_PLUGIN_STATUS_BAD_PARAM at plan-build time.
     */
    TensorAttributes& set_as_runtime_parameter() // NOLINT(readability-identifier-naming)
    {
        clear_value();
        _isRuntimePassByValue = true;
        return *this;
    }

    /**
     * @brief Set only the runtime pass-by-value flag, leaving any stored value intact
     * @param b Whether this tensor is a runtime pass-by-value scalar
     * @return Reference to this for method chaining
     */
    TensorAttributes& set_is_pass_by_value(bool b) // NOLINT(readability-identifier-naming)
    {
        _isRuntimePassByValue = b;
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
     * @brief Get the ragged-offset aux tensor for this tensor
     * @return Shared pointer to the ragged-offset TensorAttributes, or nullptr if not ragged
     */
    std::shared_ptr<TensorAttributes> get_ragged_offset() // NOLINT(readability-identifier-naming)
    {
        return _raggedOffset;
    }

    /**
     * @brief Set the ragged-offset aux tensor for this tensor
     * @param value Shared pointer to the TensorAttributes of the ragged-offset aux tensor
     * @return Reference to this for method chaining
     */
    TensorAttributes& set_ragged_offset( // NOLINT(readability-identifier-naming)
        const std::shared_ptr<TensorAttributes>& value)
    {
        _raggedOffset = value;
        return *this;
    }

    /**
     * @brief Check whether this tensor has a ragged-offset aux tensor set
     * @return true if a ragged-offset aux has been set
     */
    bool has_ragged_offset() const // NOLINT(readability-identifier-naming)
    {
        return _raggedOffset != nullptr;
    }

    /**
     * @brief Get the required byte alignment of the tensor's physical buffer pointer
     * @return Alignment in bytes (default 16)
     */
    int64_t get_alignment() const // NOLINT(readability-identifier-naming)
    {
        return _alignment;
    }

    /**
     * @brief Set the required byte alignment of the tensor's physical buffer pointer
     * @param value Alignment in bytes; must be >= 1
     * @return Reference to this for method chaining
     */
    TensorAttributes& set_alignment(const int64_t value) // NOLINT(readability-identifier-naming)
    {
        _alignment = value;
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

        HIPDNN_RETURN_IF_TRUE(_isVirtual && _isRuntimePassByValue,
                              ErrorCode::INVALID_VALUE,
                              "Tensor " + _name + " cannot be virtual and runtime pass by value");
        HIPDNN_RETURN_IF_TRUE(_isVirtual && hasValue(),
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

        HIPDNN_RETURN_IF_TRUE(_alignment < 1,
                              ErrorCode::INVALID_VALUE,
                              "Tensor " + _name + " alignment must be >= 1");

        return {ErrorCode::OK, ""};
    }

    /// @brief Checks if two tensors are logically identical in terms of shape,
    /// layout, data type, value, and structural role in the graph.
    /// @note This intentionally ignores the human-readable string name.
    bool logicallyEquals(const TensorAttributes& other) const
    {
        if(this->_dataType != other._dataType)
        {
            return false;
        }
        if(this->_dim != other._dim)
        {
            return false;
        }
        if(this->_stride != other._stride)
        {
            return false;
        }
        if(this->_isVirtual != other._isVirtual)
        {
            return false;
        }
        if(this->_isRuntimePassByValue != other._isRuntimePassByValue)
        {
            return false;
        }
        // Compare pass-by-value scalar variants
        if(this->_value != other._value)
        {
            return false;
        }

        return true;
    }

    /// @brief Absolute equality check including non-functional metadata like names.
    bool operator==(const TensorAttributes& other) const
    {

        if(!logicallyEquals(other))
        {
            return false;
        }
        if(this->_name != other._name)
        {
            return false;
        }
        if(this->_uidSet != other._uidSet)
        {
            return false;
        }
        if(this->_uidSet && (this->_uid != other._uid))
        {
            return false;
        }

        return true;
    }

    bool operator!=(const TensorAttributes& other) const
    {
        return !(*this == other);
    }

private:
    bool hasValue() const
    {
        return !std::holds_alternative<std::monostate>(_value);
    }

    int64_t _uid = 0;
    bool _uidSet = false;
    std::string _name;
    DataType _dataType = DataType::NOT_SET;
    std::vector<int64_t> _stride;
    std::vector<int64_t> _dim;
    bool _isVirtual = false;
    bool _isRuntimePassByValue = false;
    ValueVariant _value;
    std::shared_ptr<TensorAttributes> _raggedOffset; ///< nullptr = non-ragged
    int64_t _alignment
        = detail::DEFAULT_TENSOR_ALIGNMENT; ///< byte alignment of the physical buffer pointer
};
typedef TensorAttributes Tensor_attributes; ///< @brief Compatibility alias
} // namespace hipdnn_frontend::graph
