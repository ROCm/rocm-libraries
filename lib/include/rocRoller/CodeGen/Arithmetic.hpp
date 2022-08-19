/**
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include <functional>
#include <memory>

#include "Instruction.hpp"

#include "../DataTypes/DataTypes.hpp"
#include "../InstructionValues/Register_fwd.hpp"
#include "../Utilities/Component.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{

    struct Arithmetic
    {
        using Argument = std::tuple<std::shared_ptr<Context>, Register::Type, VariableType>;

        Arithmetic(std::shared_ptr<Context> context);

        virtual ~Arithmetic();

        /**
         * Returns an Arithmetic object that matches the register/data types of `reg`.
         */
        static std::shared_ptr<Arithmetic> Get(Register::ValuePtr reg);

        /**
         * Returns an Arithmetic object that corresponds to a unary operation on `src`, stored into `dst`
         */
        static std::shared_ptr<Arithmetic> Get(Register::ValuePtr dst, Register::ValuePtr src);

        /**
         * Returns an Arithmetic object that corresponds to a binary arithmetic operation on `lhs` and `rhs`, stored into `dst`.
         */
        static std::shared_ptr<Arithmetic>
            Get(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

        /**
         * Returns an Arithmetic object that corresponds to a binary comparison operation on `lhs` and `rhs`, stored into `dst`.
         */
        static std::shared_ptr<Arithmetic>
            GetComparison(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs);

        /**
         * Returns an Arithmetic object that corresponds to a ternary arithmetic operation on `lhs` and `rhs`, stored into `dst`.
         */
        static std::shared_ptr<Arithmetic> Get(Register::ValuePtr dst,
                                               Register::ValuePtr lhs,
                                               Register::ValuePtr r1hs,
                                               Register::ValuePtr r2hs);

        static const std::string Name;
        virtual std::string      name() = 0;

        virtual DataType       dataType()     = 0;
        virtual Register::Type registerType() = 0;

        virtual Register::ValuePtr placeholder() = 0;

        /**
         * Splits `inputs` into individual dword-sized Register::Value objects
         * appropriate for this kind of arithmetic.

         * Generic interface to get2DWords which allows us to call in to other
         * Arithmetic implementations.
         *
         * We may or may not be able to simplify this into just the get2Dwords
         * interface.
         */
        virtual Generator<Instruction> getDwords(std::vector<Register::ValuePtr>& dwords,
                                                 Register::ValuePtr               inputs)
            = 0;

        virtual Generator<Instruction> negate(std::shared_ptr<Register::Value> dest,
                                              std::shared_ptr<Register::Value> src)
            = 0;

        virtual Generator<Instruction> add(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs)
            = 0;

        virtual Generator<Instruction> sub(std::shared_ptr<Register::Value> dest,
                                           std::shared_ptr<Register::Value> lhs,
                                           std::shared_ptr<Register::Value> rhs)
            = 0;

        virtual Generator<Instruction> shiftL(std::shared_ptr<Register::Value> dest,
                                              std::shared_ptr<Register::Value> value,
                                              std::shared_ptr<Register::Value> shiftAmount)
        {
            throw std::runtime_error("ShiftL unsupported for this datatype");
        }

        virtual Generator<Instruction> shiftR(std::shared_ptr<Register::Value> dest,
                                              std::shared_ptr<Register::Value> value,
                                              std::shared_ptr<Register::Value> shiftAmount)
        {
            throw std::runtime_error("ShiftR unsupported for this datatype");
        }

        virtual Generator<Instruction> addShiftL(std::shared_ptr<Register::Value> dest,
                                                 std::shared_ptr<Register::Value> lhs,
                                                 std::shared_ptr<Register::Value> rhs,
                                                 std::shared_ptr<Register::Value> shiftAmount)
        {
            throw std::runtime_error("AddShiftL unsupported for this datatype");
        }

        virtual Generator<Instruction> shiftLAdd(std::shared_ptr<Register::Value> dest,
                                                 std::shared_ptr<Register::Value> lhs,
                                                 std::shared_ptr<Register::Value> shiftAmount,
                                                 std::shared_ptr<Register::Value> rhs)
        {
            throw std::runtime_error("shiftAddL unsupported for this datatype");
        }

        virtual Generator<Instruction> bitwiseAnd(std::shared_ptr<Register::Value> dest,
                                                  std::shared_ptr<Register::Value> lhs,
                                                  std::shared_ptr<Register::Value> rhs)
        {
            throw std::runtime_error("BitwiseAnd unsupported for this datatype");
        }

        virtual Generator<Instruction> bitwiseXor(std::shared_ptr<Register::Value> dest,
                                                  std::shared_ptr<Register::Value> lhs,
                                                  std::shared_ptr<Register::Value> rhs)
        {
            throw std::runtime_error("BitwiseXor unsupported for this datatype");
        }

        virtual Generator<Instruction> gt(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs)
            = 0;

        virtual Generator<Instruction> ge(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs)
            = 0;

        virtual Generator<Instruction> lt(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs)
            = 0;

        virtual Generator<Instruction> le(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs)
            = 0;

        virtual Generator<Instruction> eq(std::shared_ptr<Register::Value> dest,
                                          std::shared_ptr<Register::Value> lhs,
                                          std::shared_ptr<Register::Value> rhs)
            = 0;

    protected:
        std::shared_ptr<Context> m_context;
    };

    static_assert(Component::ComponentBase<Arithmetic>);

}
