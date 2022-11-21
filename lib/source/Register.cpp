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
            return ((context->kernel()->wavefront_size() == 64
                     && m_specialName == Register::SpecialType::VCC)
                    || (context->kernel()->wavefront_size() == 32
                        && m_specialName == Register::SpecialType::VCC_LO));
        }
        return false;
    }

    /**
     * @brief Yields RegisterId for the registers associated with this allocation
     *
     * Note: This function does not yield any ids for Literals, Labels, or unallocated registers
     *
     * @return Generator<RegisterId>
     */
    Generator<Register::RegisterId> Register::Value::getRegisterIds() const
    {
        if(m_regType == Register::Type::Literal || m_regType == Register::Type::Label)
        {
            co_return;
        }
        if(m_regType == Register::Type::Special)
        {
            auto context = m_context.lock();
            if(context->kernel()->wavefront_size() == 64
               && m_specialName == Register::SpecialType::VCC)
            {
                co_yield RegisterId(m_regType, static_cast<int>(Register::SpecialType::VCC_LO));
                co_yield RegisterId(m_regType, static_cast<int>(Register::SpecialType::VCC_HI));
            }
            else
            {
                co_yield RegisterId(m_regType, static_cast<int>(m_specialName));
            }
        }
        if(!m_allocation || m_allocation->allocationState() != Register::AllocationState::Allocated)
        {
            co_return;
        }
        for(int coord : m_allocationCoord)
        {
            co_yield RegisterId(m_regType, m_allocation->m_registerIndices.at(coord));
        }
    }

    Expression::ExpressionPtr Register::Value::expression()
    {
        AssertFatal(this, "Null expression accessed");
        return std::make_shared<Expression::Expression>(shared_from_this());
    }
}
