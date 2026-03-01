// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file HipdnnPointwiseMode.h
 * @brief Pointwise mode enumeration for hipDNN backend operations
 *
 * Defines the pointwise mode used when setting the
 * HIPDNN_ATTR_POINTWISE_MODE attribute on pointwise descriptors.
 */

#pragma once

/**
 * @enum hipdnnPointwiseMode_t
 * @brief Pointwise mode for backend pointwise operations
 *
 * Determines the pointwise mathematical operation applied element-wise.
 * Pointwise operations can be unary (1 input), binary (2 inputs), or
 * ternary (3 inputs).
 *
 * @see HIPDNN_ATTR_POINTWISE_MODE
 * @see hipdnnBackendSetAttribute()
 */
typedef enum
{
    HIPDNN_POINTWISE_MODE_ABS = 1, ///< Absolute value: |x|
    HIPDNN_POINTWISE_MODE_ADD = 2, ///< Addition: x + y
    HIPDNN_POINTWISE_MODE_ADD_SQUARE = 3, ///< Add and square: (x + y)^2
    HIPDNN_POINTWISE_MODE_BINARY_SELECT = 4, ///< Ternary select based on condition
    HIPDNN_POINTWISE_MODE_CEIL = 5, ///< Ceiling function
    HIPDNN_POINTWISE_MODE_CMP_EQ = 6, ///< Compare equal: x == y
    HIPDNN_POINTWISE_MODE_CMP_GE = 7, ///< Compare greater or equal: x >= y
    HIPDNN_POINTWISE_MODE_CMP_GT = 8, ///< Compare greater than: x > y
    HIPDNN_POINTWISE_MODE_CMP_LE = 9, ///< Compare less or equal: x <= y
    HIPDNN_POINTWISE_MODE_CMP_LT = 10, ///< Compare less than: x < y
    HIPDNN_POINTWISE_MODE_CMP_NEQ = 11, ///< Compare not equal: x != y
    HIPDNN_POINTWISE_MODE_DIV = 12, ///< Division: x / y
    HIPDNN_POINTWISE_MODE_ELU_BWD = 13, ///< ELU activation backward pass
    HIPDNN_POINTWISE_MODE_ELU_FWD = 14, ///< ELU activation forward pass
    HIPDNN_POINTWISE_MODE_ERF = 15, ///< Error function
    HIPDNN_POINTWISE_MODE_EXP = 16, ///< Exponential: e^x
    HIPDNN_POINTWISE_MODE_FLOOR = 17, ///< Floor function
    HIPDNN_POINTWISE_MODE_GELU_APPROX_TANH_BWD = 18, ///< GELU (tanh approximation) backward
    HIPDNN_POINTWISE_MODE_GELU_APPROX_TANH_FWD = 19, ///< GELU (tanh approximation) forward
    HIPDNN_POINTWISE_MODE_GELU_BWD = 20, ///< GELU activation backward
    HIPDNN_POINTWISE_MODE_GELU_FWD = 21, ///< GELU activation forward
    HIPDNN_POINTWISE_MODE_GEN_INDEX = 22, ///< Generate index tensor
    HIPDNN_POINTWISE_MODE_IDENTITY = 23, ///< Identity: y = x
    HIPDNN_POINTWISE_MODE_LOG = 24, ///< Natural logarithm
    HIPDNN_POINTWISE_MODE_LOGICAL_AND = 25, ///< Logical AND
    HIPDNN_POINTWISE_MODE_LOGICAL_NOT = 26, ///< Logical NOT
    HIPDNN_POINTWISE_MODE_LOGICAL_OR = 27, ///< Logical OR
    HIPDNN_POINTWISE_MODE_MAX = 28, ///< Element-wise maximum
    HIPDNN_POINTWISE_MODE_MIN = 29, ///< Element-wise minimum
    HIPDNN_POINTWISE_MODE_MUL = 30, ///< Multiplication: x * y
    HIPDNN_POINTWISE_MODE_NEG = 31, ///< Negation: -x
    HIPDNN_POINTWISE_MODE_RECIPROCAL = 32, ///< Reciprocal: 1/x
    HIPDNN_POINTWISE_MODE_RELU_BWD = 33, ///< ReLU backward pass
    HIPDNN_POINTWISE_MODE_RELU_FWD = 34, ///< ReLU forward pass
    HIPDNN_POINTWISE_MODE_RSQRT = 35, ///< Reciprocal square root: 1/sqrt(x)
    HIPDNN_POINTWISE_MODE_SIGMOID_BWD = 36, ///< Sigmoid backward pass
    HIPDNN_POINTWISE_MODE_SIGMOID_FWD = 37, ///< Sigmoid forward pass
    HIPDNN_POINTWISE_MODE_SIN = 38, ///< Sine function
    HIPDNN_POINTWISE_MODE_SOFTPLUS_BWD = 39, ///< Softplus backward pass
    HIPDNN_POINTWISE_MODE_SOFTPLUS_FWD = 40, ///< Softplus forward pass
    HIPDNN_POINTWISE_MODE_SQRT = 41, ///< Square root
    HIPDNN_POINTWISE_MODE_SUB = 42, ///< Subtraction: x - y
    HIPDNN_POINTWISE_MODE_SWISH_BWD = 43, ///< Swish activation backward
    HIPDNN_POINTWISE_MODE_SWISH_FWD = 44, ///< Swish activation forward
    HIPDNN_POINTWISE_MODE_TAN = 45, ///< Tangent function
    HIPDNN_POINTWISE_MODE_TANH_BWD = 46, ///< Tanh backward pass
    HIPDNN_POINTWISE_MODE_TANH_FWD = 47 ///< Tanh forward pass
} hipdnnPointwiseMode_t;
