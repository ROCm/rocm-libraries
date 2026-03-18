// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 *
 */

#pragma once

#include <concepts>
#include <functional>
#include <memory>
#include <variant>

#include <rocRoller/AssemblyKernelArgument_fwd.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension_fwd.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct Add;
        struct MatrixMultiply;
        struct ScaledMatrixMultiply;
        struct Multiply;
        struct MultiplyAdd;
        struct MultiplyHigh;
        struct Subtract;
        struct Divide;
        struct Modulo;
        struct ShiftL;
        struct LogicalShiftR;
        struct ArithmeticShiftR;
        struct BitfieldCombine;
        struct BitwiseNegate;
        struct BitwiseAnd;
        struct BitwiseOr;
        struct BitwiseXor;
        struct GreaterThan;
        struct GreaterThanEqual;
        struct LessThan;
        struct LessThanEqual;
        struct Equal;
        struct NotEqual;
        struct LogicalAnd;
        struct LogicalOr;
        struct LogicalNot;

        struct Exponential2;
        struct Exponential;

        struct MagicMultiple;
        struct MagicShifts;
        struct MagicShiftAndSign;
        struct Negate;

        struct RandomNumber;

        struct ToScalar;

        struct BitFieldExtract;

        struct AddShiftL;
        struct ShiftLAdd;
        struct Conditional;

        struct Convert;
        struct Reinterpret;

        struct Concatenate;

        template <DataType DATATYPE>
        struct SRConvert;

        struct DataFlowTag;
        struct PositionalArgument;

        using WaveTilePtr = std::shared_ptr<KernelGraph::CoordinateGraph::WaveTile>;

        using Expression = std::variant<
            // --- Binary Operations ---
            Add,
            Subtract,
            Multiply,
            MultiplyHigh,
            Divide,
            Modulo,

            ShiftL,
            LogicalShiftR,
            ArithmeticShiftR,

            BitfieldCombine,
            BitwiseAnd,
            BitwiseOr,
            BitwiseXor,
            LogicalAnd,
            LogicalOr,

            GreaterThan,
            GreaterThanEqual,
            LessThan,
            LessThanEqual,
            Equal,
            NotEqual,

            Concatenate,

            // --- Stochastic Rounding Convert (also binary) ---
            SRConvert<DataType::FP8>,
            SRConvert<DataType::BF8>,

            // --- Unary Operations ---
            MagicMultiple,
            MagicShifts,
            MagicShiftAndSign,
            Negate,
            BitwiseNegate,
            LogicalNot,
            Exponential2,
            Exponential,
            RandomNumber,
            ToScalar,
            BitFieldExtract,
            Convert,
            Reinterpret,

            // --- Ternary Operations ---
            AddShiftL,
            ShiftLAdd,
            MatrixMultiply,
            Conditional,
            MultiplyAdd,

            // ---Quinary Operation(s) ---
            ScaledMatrixMultiply,

            // --- Values ---
            PositionalArgument,

            // Literal: Always available
            CommandArgumentValue,

            // Available at kernel launch
            CommandArgumentPtr,

            // Available at kernel execute (i.e. on the GPU), and at kernel launch.
            AssemblyKernelArgumentPtr,

            // Available at kernel execute (i.e. on the GPU)
            Register::ValuePtr,
            DataFlowTag,
            WaveTilePtr>;
        using ExpressionPtr = std::shared_ptr<Expression>;

        using ExpressionTransducer = std::function<ExpressionPtr(ExpressionPtr)>;

        template <typename T>
        concept CExpression = std::constructible_from<Expression, T>;

        enum class EvaluationTime : int
        {
            Translate = 0, //< An expression where all the values come from CommandArgumentValues.
            KernelLaunch, //< An expression where values may come from CommandArguments or CommandArgumentValues.
            KernelExecute, // An expression that depends on at least one Register::Value.
            Count
        };

        enum class Category : int;

    }
}
