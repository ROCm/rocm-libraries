// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <bitset>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>

namespace hipdnn_sdk
{
namespace utilities
{

// Bitset size based on the maximum PointwiseMode value + 1
static constexpr size_t POINTWISE_MODE_COUNT
    = static_cast<size_t>(hipdnn_sdk::data_objects::PointwiseMode::MAX) + 1;

using PointwiseModeBitset = std::bitset<POINTWISE_MODE_COUNT>;

constexpr size_t toBitPosition(hipdnn_sdk::data_objects::PointwiseMode mode)
{
    return static_cast<size_t>(mode);
}

// Get operations that have implemented functors for unary operations
inline const PointwiseModeBitset& getImplementedUnaryModesBitset()
{
    static const PointwiseModeBitset s_implementedUnaryModes = []() {
        PointwiseModeBitset bitset;
        // Only include operations with implemented functors in ReferencePointwiseBase
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::ABS));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::NEG));
        return bitset;
    }();
    return s_implementedUnaryModes;
}

// Get operations that have implemented functors for binary operations
inline const PointwiseModeBitset& getImplementedBinaryModesBitset()
{
    static const PointwiseModeBitset s_implementedBinaryModes = []() {
        PointwiseModeBitset bitset;
        // Only include operations with implemented functors in ReferencePointwiseBase
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::ADD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SUB));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::MUL));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::TANH_BWD));
        return bitset;
    }();
    return s_implementedBinaryModes;
}

// Get all unary operations (for frontend compatibility)
inline const PointwiseModeBitset& getUnaryModesBitset()
{
    static const PointwiseModeBitset s_unaryModes = []() {
        PointwiseModeBitset bitset;
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::ABS));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::CEIL));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::ELU_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::ERF));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::EXP));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::FLOOR));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::GELU_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::GEN_INDEX));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::IDENTITY));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::LOG));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::LOGICAL_NOT));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::NEG));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::RECIPROCAL));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::RSQRT));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SIN));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SOFTPLUS_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SQRT));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SWISH_FWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::TAN));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD));
        return bitset;
    }();
    return s_unaryModes;
}

// Get all binary operations (for frontend compatibility)
inline const PointwiseModeBitset& getBinaryModesBitset()
{
    static const PointwiseModeBitset s_binaryModes = []() {
        PointwiseModeBitset bitset;
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::ADD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::ADD_SQUARE));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::CMP_EQ));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::CMP_GE));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::CMP_GT));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::CMP_LE));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::CMP_LT));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::CMP_NEQ));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::DIV));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::ELU_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::GELU_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::LOGICAL_AND));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::LOGICAL_OR));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::MAX_OP));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::MIN_OP));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::MUL));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SOFTPLUS_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SUB));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::SWISH_BWD));
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::TANH_BWD));
        return bitset;
    }();
    return s_binaryModes;
}

// Get all ternary operations (for frontend compatibility)
inline const PointwiseModeBitset& getTernaryModesBitset()
{
    static const PointwiseModeBitset s_ternaryModes = []() {
        PointwiseModeBitset bitset;
        bitset.set(toBitPosition(hipdnn_sdk::data_objects::PointwiseMode::BINARY_SELECT));
        return bitset;
    }();
    return s_ternaryModes;
}

inline bool isUnaryPointwiseMode(hipdnn_sdk::data_objects::PointwiseMode mode)
{
    auto position = toBitPosition(mode);
    return position < POINTWISE_MODE_COUNT && getUnaryModesBitset().test(position);
}

inline bool isBinaryPointwiseMode(hipdnn_sdk::data_objects::PointwiseMode mode)
{
    auto position = toBitPosition(mode);
    return position < POINTWISE_MODE_COUNT && getBinaryModesBitset().test(position);
}

inline bool isTernaryPointwiseMode(hipdnn_sdk::data_objects::PointwiseMode mode)
{
    auto position = toBitPosition(mode);
    return position < POINTWISE_MODE_COUNT && getTernaryModesBitset().test(position);
}

// Check if operations have implemented functors (for ReferencePointwiseBase usage)
inline bool isImplementedUnaryPointwiseMode(hipdnn_sdk::data_objects::PointwiseMode mode)
{
    auto position = toBitPosition(mode);
    return position < POINTWISE_MODE_COUNT && getImplementedUnaryModesBitset().test(position);
}

inline bool isImplementedBinaryPointwiseMode(hipdnn_sdk::data_objects::PointwiseMode mode)
{
    auto position = toBitPosition(mode);
    return position < POINTWISE_MODE_COUNT && getImplementedBinaryModesBitset().test(position);
}

inline bool isImplementedTernaryPointwiseMode(hipdnn_sdk::data_objects::PointwiseMode /* mode */)
{
    // Currently no ternary operations are implemented
    return false;
}

} // namespace utilities
} // namespace hipdnn_sdk
