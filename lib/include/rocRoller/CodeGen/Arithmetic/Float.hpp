/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "ArithmeticBase.hpp"

namespace rocRoller
{
    struct Arithmetic_Vector_Float
        : public ArithmeticBase<Register::Type::Vector, DataType::Float, PointerType::Value>
    {
        using Base = Arithmetic;
        using DirectBase
            = ArithmeticBase<Register::Type::Vector, DataType::Float, PointerType::Value>;

        static const std::string Name;
        static const std::string Basename;

        virtual std::string name() final override;

        Arithmetic_Vector_Float(std::shared_ptr<Context> context);

        Arithmetic_Vector_Float(Arithmetic_Vector_Float const& rhs) = default;
        Arithmetic_Vector_Float(Arithmetic_Vector_Float&& rhs)      = default;
        ~Arithmetic_Vector_Float()                                  = default;

        static bool                  Match(Argument const& arg);
        static std::shared_ptr<Base> Build(Argument const& arg);

        virtual Generator<Instruction> add(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> sub(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> mul(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> gt(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> ge(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> lt(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> le(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> eq(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs) final override;
    };

    static_assert(Component::Component<Arithmetic_Vector_Float>);
    static_assert(
        std::is_base_of<Arithmetic_Vector_Float::DirectBase, Arithmetic_Vector_Float>::value);
    static_assert(std::is_base_of<Arithmetic_Vector_Float::Base, Arithmetic_Vector_Float>::value);

}
