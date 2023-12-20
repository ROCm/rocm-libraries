/**
 *
 */

#pragma once

#include <concepts>
#include <functional>
#include <memory>
#include <variant>

#include "AssemblyKernelArgument_fwd.hpp"
#include "InstructionValues/Register_fwd.hpp"
#include "KernelGraph/CoordinateGraph/Dimension_fwd.hpp"
#include "Operations/CommandArgument_fwd.hpp"

namespace rocRoller
{
    namespace Expression
    {
        struct Add;
        struct MatrixMultiply;
        struct Multiply;
        struct MultiplyAdd;
        struct MultiplyHigh;
        struct Subtract;
        struct Divide;
        struct Modulo;
        struct ShiftL;
        struct LogicalShiftR;
        struct ArithmeticShiftR;
        struct BitwiseAnd;
        struct BitwiseOr;
        struct BitwiseXor;
        struct GreaterThan;
        struct GreaterThanEqual;
        struct LessThan;
        struct LessThanEqual;
        struct Equal;
        struct LogicalAnd;
        struct LogicalOr;

        struct MagicMultiple;
        struct MagicShifts;
        struct MagicSign;
        struct Negate;

        struct AddShiftL;
        struct ShiftLAdd;
        struct Conditional;

        template <DataType DATATYPE>
        struct Convert;

        struct DataFlowTag;
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
            BitwiseAnd,
            BitwiseOr,
            BitwiseXor,
            GreaterThan,
            GreaterThanEqual,
            LessThan,
            LessThanEqual,
            Equal,
            LogicalAnd,
            LogicalOr,

            // --- Unary Operations ---
            MagicMultiple,
            MagicShifts,
            MagicSign,
            Negate,

            // --- Ternary Operations ---
            AddShiftL,
            ShiftLAdd,
            MatrixMultiply,
            Conditional,

            // --- TernaryMixed Operations ---
            MultiplyAdd,

            // --- Convert Operations ---
            Convert<DataType::Half>,
            Convert<DataType::Halfx2>,
            Convert<DataType::Float>,
            Convert<DataType::Double>,
            Convert<DataType::Int32>,
            Convert<DataType::Int64>,
            Convert<DataType::UInt32>,
            Convert<DataType::UInt64>,

            // --- Values ---
            // Literal: Always available
            CommandArgumentValue,

            // Available at kernel launch
            CommandArgumentPtr,

            // Available at kernel execute (i.e. on the GPU)
            AssemblyKernelArgumentPtr,
            Register::ValuePtr,
            DataFlowTag,
            WaveTilePtr>;
        using ExpressionPtr = std::shared_ptr<Expression>;

        using ExpressionTransducer = std::function<ExpressionPtr(ExpressionPtr)>;

        template <typename T>
        concept CExpression = std::constructible_from<Expression, T>;

        enum class EvaluationTime : int;

        enum class Category : int;

    }
}
