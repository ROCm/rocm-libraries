// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>

namespace hipdnn_test_sdk::utilities::pointwise
{

/// Maps a PointwiseMode to its corresponding PointwiseErrorClass.
///
/// Classification follows the error model documented in DynamicTolerances.hpp:
/// - BITWISE (A): exact ops — comparisons, select, relu, abs, neg, identity, floor, ceil
/// - LINEAR (B): IEEE correctly-rounded — add, sub, mul, add_square
/// - RATIONAL (C): Newton-Raphson — div, reciprocal, sqrt, rsqrt
/// - TRANSCENDENTAL_FWD (D fwd): polynomial/table approx — exp, log, sin, tan, tanh, sigmoid, erf,
///   elu
/// - TRANSCENDENTAL_BWD (D bwd): chain rule amplification — tanh_bwd, sigmoid_bwd, elu_bwd
/// - COMPOSITE_FWD (E fwd): compound nonlinear — gelu, gelu_approx_tanh, softplus, swish
/// - COMPOSITE_BWD (E bwd): compound backward — gelu_bwd, gelu_approx_tanh_bwd, softplus_bwd,
///   swish_bwd
inline PointwiseErrorClass
    classifyPointwiseOp(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    switch(mode)
    {
    // Class A — Bitwise / Selection / Predicate
    case PointwiseMode::ABS:
    case PointwiseMode::NEG:
    case PointwiseMode::IDENTITY:
    case PointwiseMode::CEIL:
    case PointwiseMode::FLOOR:
    case PointwiseMode::CMP_EQ:
    case PointwiseMode::CMP_GE:
    case PointwiseMode::CMP_GT:
    case PointwiseMode::CMP_LE:
    case PointwiseMode::CMP_LT:
    case PointwiseMode::CMP_NEQ:
    case PointwiseMode::LOGICAL_AND:
    case PointwiseMode::LOGICAL_OR:
    case PointwiseMode::LOGICAL_NOT:
    case PointwiseMode::BINARY_SELECT:
    case PointwiseMode::GEN_INDEX:
    case PointwiseMode::MAX_OP:
    case PointwiseMode::MIN_OP:
    case PointwiseMode::RELU_FWD:
    case PointwiseMode::RELU_BWD:
        return PointwiseErrorClass::BITWISE;

    // Class B — Linear Arithmetic
    case PointwiseMode::ADD:
    case PointwiseMode::SUB:
    case PointwiseMode::MUL:
    case PointwiseMode::ADD_SQUARE:
        return PointwiseErrorClass::LINEAR;

    // Class C — Rational / Simple Nonlinear
    case PointwiseMode::DIV:
    case PointwiseMode::RECIPROCAL:
    case PointwiseMode::SQRT:
    case PointwiseMode::RSQRT:
        return PointwiseErrorClass::RATIONAL;

    // Class D fwd — Transcendentals (forward)
    case PointwiseMode::EXP:
    case PointwiseMode::LOG:
    case PointwiseMode::SIN:
    case PointwiseMode::TAN:
    case PointwiseMode::TANH_FWD:
    case PointwiseMode::SIGMOID_FWD:
    case PointwiseMode::ERF:
    case PointwiseMode::ELU_FWD:
        return PointwiseErrorClass::TRANSCENDENTAL_FWD;

    // Class D bwd — Transcendentals (backward)
    case PointwiseMode::TANH_BWD:
    case PointwiseMode::SIGMOID_BWD:
    case PointwiseMode::ELU_BWD:
        return PointwiseErrorClass::TRANSCENDENTAL_BWD;

    // Class E fwd — Composite (forward)
    case PointwiseMode::GELU_FWD:
    case PointwiseMode::GELU_APPROX_TANH_FWD:
    case PointwiseMode::SOFTPLUS_FWD:
    case PointwiseMode::SWISH_FWD:
        return PointwiseErrorClass::COMPOSITE_FWD;

    // Class E bwd — Composite (backward)
    case PointwiseMode::GELU_BWD:
    case PointwiseMode::GELU_APPROX_TANH_BWD:
    case PointwiseMode::SOFTPLUS_BWD:
    case PointwiseMode::SWISH_BWD:
        return PointwiseErrorClass::COMPOSITE_BWD;
    default:
        throw std::logic_error("Unclassified PointwiseMode — add it to classifyPointwiseOp");
    }
}

/// Returns true if the pointwise operation has bounded output, meaning the caller
/// should pass scale=1.0 rather than max(|input|) to calculatePointwiseTolerance.
///
/// Bounded-output ops:
/// - SIGMOID_FWD: output in [0, 1]
/// - TANH_FWD: output in [-1, 1]
/// - ERF: output in [-1, 1]
inline bool isBoundedOutput(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    switch(mode)
    {
    case PointwiseMode::SIGMOID_FWD:
    case PointwiseMode::TANH_FWD:
    case PointwiseMode::ERF:
        return true;
    default:
        return false;
    }
}

} // namespace hipdnn_test_sdk::utilities::pointwise
