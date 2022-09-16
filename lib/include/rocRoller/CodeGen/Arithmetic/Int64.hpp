/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "ArithmeticBase.hpp"

#include <rocRoller/CodeGen/CopyGenerator.hpp>

namespace rocRoller
{
    struct Arithmetic_Vector_Int64
        : public ArithmeticBase<Register::Type::Vector, DataType::Int64, PointerType::Value>
    {
        using Base = Arithmetic;
        using DirectBase
            = ArithmeticBase<Register::Type::Vector, DataType::Int64, PointerType::Value>;

    public:
        static const std::string Name;
        static const std::string Basename;

        virtual std::string name() final override;

        Arithmetic_Vector_Int64(std::shared_ptr<Context> context);

        Arithmetic_Vector_Int64(Arithmetic_Vector_Int64 const& rhs) = default;
        Arithmetic_Vector_Int64(Arithmetic_Vector_Int64&& rhs)      = default;
        ~Arithmetic_Vector_Int64()                                  = default;

        static bool                  Match(Argument const& arg);
        static std::shared_ptr<Base> Build(Argument const& arg);

        virtual Generator<Instruction> getDwords(std::vector<Register::ValuePtr>& dwords,
                                                 Register::ValuePtr input) override final;

        /**
         * Separates input into separate Values by dword which contain a 64-bit
         * representation of the value in `input`.
         *
         * `input` may be a 32- or 64-bit Vector, Scalar or Literal integral value.
         *
         * `lsd` and `msd` may be returned as mixed register types (specifically if
         * `input` is a 32-bit unsigned int, msd will be a literal 0) and not
         * necessarily adjacent registers.
         */
        Generator<Instruction>
            get2Dwords(Register::ValuePtr& lsd, Register::ValuePtr& msd, Register::ValuePtr input);

        virtual Generator<Instruction> add(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> sub(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction>
            shiftL(std::shared_ptr<Register::Value> dest,
                   std::shared_ptr<Register::Value> value,
                   std::shared_ptr<Register::Value> shiftAmount) final override;

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

    static_assert(Component::Component<Arithmetic_Vector_Int64>);
    static_assert(
        std::is_base_of<Arithmetic_Vector_Int64::DirectBase, Arithmetic_Vector_Int64>::value);
    static_assert(std::is_base_of<Arithmetic_Vector_Int64::Base, Arithmetic_Vector_Int64>::value);

    struct Arithmetic_Scalar_Int64
        : public ArithmeticBase<Register::Type::Scalar, DataType::Int64, PointerType::Value>
    {
        using Base = Arithmetic;
        using DirectBase
            = ArithmeticBase<Register::Type::Scalar, DataType::Int64, PointerType::Value>;

    public:
        static const std::string Name;
        static const std::string Basename;

        virtual std::string name() final override;

        Arithmetic_Scalar_Int64(std::shared_ptr<Context> context);

        Arithmetic_Scalar_Int64(Arithmetic_Scalar_Int64 const& rhs) = default;
        Arithmetic_Scalar_Int64(Arithmetic_Scalar_Int64&& rhs)      = default;
        ~Arithmetic_Scalar_Int64()                                  = default;

        virtual Generator<Instruction> getDwords(std::vector<Register::ValuePtr>& dwords,
                                                 Register::ValuePtr input) override final;

        /**
         * Separates input into separate Values by dword which contain a 64-bit
         * representation of the value in `input`.
         *
         * `input` may be a 32- or 64-bit Vector, Scalar or Literal integral value.
         *
         * `lsd` and `msd` may be returned as mixed register types (specifically if
         * `input` is a 32-bit unsigned int, msd will be a literal 0) and not
         * necessarily adjacent registers.
         */
        Generator<Instruction>
            get2Dwords(Register::ValuePtr& msw, Register::ValuePtr& lsw, Register::ValuePtr input);

        static bool                  Match(Argument const& arg);
        static std::shared_ptr<Base> Build(Argument const& arg);

        virtual Generator<Instruction> add(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction> sub(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs) final override;

        virtual Generator<Instruction>
            shiftL(std::shared_ptr<Register::Value> dest,
                   std::shared_ptr<Register::Value> value,
                   std::shared_ptr<Register::Value> shiftAmount) final override;

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

    static_assert(Component::Component<Arithmetic_Scalar_Int64>);
    static_assert(
        std::is_base_of<Arithmetic_Scalar_Int64::DirectBase, Arithmetic_Scalar_Int64>::value);
    static_assert(std::is_base_of<Arithmetic_Scalar_Int64::Base, Arithmetic_Scalar_Int64>::value);
}
