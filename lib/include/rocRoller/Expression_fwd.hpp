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
        struct MagicSign;
        struct Negate;

        struct RandomNumber;

        struct BitFieldExtract;

        struct AddShiftL;
        struct ShiftLAdd;
        struct Conditional;

        struct Convert;

        template <DataType DATATYPE>
        struct SRConvert;

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
            NotEqual,
            LogicalAnd,
            LogicalOr,

            // --- Unary Operations ---
            MagicMultiple,
            MagicShifts,
            MagicSign,
            Negate,
            BitwiseNegate,
            LogicalNot,
            Exponential2,
            Exponential,
            RandomNumber,
            BitFieldExtract,

            // --- Ternary Operations ---
            AddShiftL,
            ShiftLAdd,
            MatrixMultiply,
            Conditional,

            // --- TernaryMixed Operations ---
            MultiplyAdd,

            // ---Quinary Operation(s) ---
            ScaledMatrixMultiply,

            // --- Convert Operations ---
            Convert,

            // --- Stochastic Rounding Convert ---
            SRConvert<DataType::FP8>,
            SRConvert<DataType::BF8>,

            // --- Values ---
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

        enum class EvaluationTime : int;

        enum class Category : int;

    }
}
