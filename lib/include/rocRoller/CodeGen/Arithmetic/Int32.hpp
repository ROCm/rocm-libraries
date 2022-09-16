/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "ArithmeticBase.hpp"

#include <rocRoller/CodeGen/CopyGenerator.hpp>

namespace rocRoller
{
    struct Arithmetic_Vector_Int32
        : public ArithmeticBase<Register::Type::Vector, DataType::Int32, PointerType::Value>
    {
        using Base = Arithmetic;
        using DirectBase
            = ArithmeticBase<Register::Type::Vector, DataType::Int32, PointerType::Value>;

    public:
        static const std::string Name;
        static const std::string Basename;

        virtual std::string name() final override;

        Arithmetic_Vector_Int32(std::shared_ptr<Context> context);

        Arithmetic_Vector_Int32(Arithmetic_Vector_Int32 const& rhs) = default;
        Arithmetic_Vector_Int32(Arithmetic_Vector_Int32&& rhs)      = default;
        ~Arithmetic_Vector_Int32()                                  = default;

        static bool                  Match(Argument const& arg);
        static std::shared_ptr<Base> Build(Argument const& arg);

        virtual Generator<Instruction> add(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> sub(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction>
            addShiftL(std::shared_ptr<Register::Value> dest,
                      std::shared_ptr<Register::Value> lhs,
                      std::shared_ptr<Register::Value> rhs,
                      std::shared_ptr<Register::Value> shiftAmount) final override;

        virtual Generator<Instruction>
            shiftLAdd(std::shared_ptr<Register::Value> dest,
                      std::shared_ptr<Register::Value> lhs,
                      std::shared_ptr<Register::Value> shiftAmount,
                      std::shared_ptr<Register::Value> rhs) final override;
    };

    static_assert(Component::Component<Arithmetic_Vector_Int32>);
    static_assert(
        std::is_base_of<Arithmetic_Vector_Int32::DirectBase, Arithmetic_Vector_Int32>::value);
    static_assert(std::is_base_of<Arithmetic_Vector_Int32::Base, Arithmetic_Vector_Int32>::value);

    struct Arithmetic_Scalar_Int32
        : public ArithmeticBase<Register::Type::Scalar, DataType::Int32, PointerType::Value>
    {
        using Base = Arithmetic;
        using DirectBase
            = ArithmeticBase<Register::Type::Scalar, DataType::Int32, PointerType::Value>;

    public:
        static const std::string Name;
        static const std::string Basename;

        virtual std::string name() final override;

        Arithmetic_Scalar_Int32(std::shared_ptr<Context> context);

        Arithmetic_Scalar_Int32(Arithmetic_Scalar_Int32 const& rhs) = default;
        Arithmetic_Scalar_Int32(Arithmetic_Scalar_Int32&& rhs)      = default;
        ~Arithmetic_Scalar_Int32()                                  = default;

        static bool                  Match(Argument const& arg);
        static std::shared_ptr<Base> Build(Argument const& arg);

        virtual Generator<Instruction> add(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> sub(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction>
            addShiftL(std::shared_ptr<Register::Value> dest,
                      std::shared_ptr<Register::Value> lhs,
                      std::shared_ptr<Register::Value> rhs,
                      std::shared_ptr<Register::Value> shiftAmount) final override;

        virtual Generator<Instruction>
            shiftLAdd(std::shared_ptr<Register::Value> dest,
                      std::shared_ptr<Register::Value> lhs,
                      std::shared_ptr<Register::Value> shiftAmount,
                      std::shared_ptr<Register::Value> rhs) final override;
    };

    static_assert(Component::Component<Arithmetic_Scalar_Int32>);
    static_assert(
        std::is_base_of<Arithmetic_Scalar_Int32::DirectBase, Arithmetic_Scalar_Int32>::value);
    static_assert(std::is_base_of<Arithmetic_Scalar_Int32::Base, Arithmetic_Scalar_Int32>::value);
}
