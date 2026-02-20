// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>

#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    namespace Arithmetic
    {
        struct LiteralDWordsVisitor
        {
            Register::ValuePtr& lsd;
            Register::ValuePtr& msd;

            void operator()(Buffer const&) const
            {
                Throw<FatalError>("Buffer type is not valid for get2LiteralDwords.");
            }

            template <typename T>
            void operator()(T v) const
            {
                using U = std::decay_t<T>;
                AssertFatal((std::is_integral_v<U>));
                uint64_t value = 0;
                if constexpr(std::is_pointer_v<U>)
                {
                    value = reinterpret_cast<uint64_t>(v);
                }
                else
                {
                    value = static_cast<uint64_t>(v);
                }
                lsd = Register::Value::Literal(static_cast<uint32_t>(value));
                msd = Register::Value::Literal(static_cast<uint32_t>(value >> 32));
            }
        };

        void get2LiteralDwords(Register::ValuePtr& lsd,
                               Register::ValuePtr& msd,
                               Register::ValuePtr  input)
        {
            AssertFatal(input->regType() == Register::Type::Literal, ShowValue(input->regType()));
            std::visit(LiteralDWordsVisitor{lsd, msd}, input->getLiteralValue());
        }

        std::string getModifier(DataType dtype)
        {
            switch(dtype)
            {
            case DataType::FP8x4:
                return "0b000";
            case DataType::BF8x4:
                return "0b001";
            case DataType::FP6x16:
                return "0b010";
            case DataType::BF6x16:
                return "0b011";
            case DataType::FP4x8:
                return "0b100";
            default:
                Throw<FatalError>("Unable to determine MI modifier: unhandled data type.",
                                  ShowValue(dtype));
            }
        }

        std::tuple<std::string, std::string> getOpselModifiers2xByte(uint lhsByte, uint rhsByte)
        {
            AssertFatal(lhsByte < 4, ShowValue(lhsByte));
            AssertFatal(rhsByte < 4, ShowValue(rhsByte));

            auto lhsLo = lhsByte & 1;
            auto rhsLo = rhsByte & 1;

            auto lhsHi = (lhsByte >> 1) & 1;
            auto rhsHi = (rhsByte >> 1) & 1;

            auto modLo = fmt::format("op_sel:[{},{}]", lhsLo, rhsLo);
            auto modHi = fmt::format("op_sel_hi:[{},{}]", lhsHi, rhsHi);

            return {modLo, modHi};
        }
    }
}
