#include <rocRoller/CodeGen/Arithmetic/Utility.hpp>

namespace rocRoller
{
    namespace Arithmetic
    {
        void get2LiteralDwords(Register::ValuePtr& lsd,
                               Register::ValuePtr& msd,
                               Register::ValuePtr  input)
        {
            assert(input->regType() == Register::Type::Literal);
            uint64_t value = std::visit(
                [](auto v) {
                    using T = std::decay_t<decltype(v)>;
                    AssertFatal((std::is_integral_v<T>));
                    if constexpr(std::is_pointer_v<T>)
                    {
                        return reinterpret_cast<uint64_t>(v);
                    }
                    else
                    {
                        return static_cast<uint64_t>(v);
                    }
                },
                input->getLiteralValue());

            lsd = Register::Value::Literal(static_cast<uint32_t>(value));
            msd = Register::Value::Literal(static_cast<uint32_t>(value >> 32));
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
                Throw<FatalError>("Unable to determine MFMA modifier: unhandled data type.",
                                  ShowValue(dtype));
            }
        }
    }
}
