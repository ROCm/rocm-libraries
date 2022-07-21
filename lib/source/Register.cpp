#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    std::shared_ptr<Register::Value> Register::Value::WavefrontPlaceholder(ContextPtr context)
    {
        int count = 1;
        if(context->kernel()->wavefront_size() == 64)
        {
            count = 2;
        }

        return Placeholder(context, Register::Type::Scalar, DataType::Raw32, count);
    }

    bool Register::Value::isVCC() const
    {
        auto context = m_context.lock();
        if(context && m_regType == Type::Special)
        {
            return ((context->kernel()->wavefront_size() == 64 && m_specialName == "vcc")
                    || (context->kernel()->wavefront_size() == 32 && m_specialName == "vcc_lo"));
        }
        return false;
    }

    Expression::ExpressionPtr Register::Value::expression()
    {
        return std::make_shared<Expression::Expression>(shared_from_this());
    }
}
