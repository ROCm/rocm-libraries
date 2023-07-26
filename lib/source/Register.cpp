#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    namespace Register
    {
        ValuePtr Value::WavefrontPlaceholder(ContextPtr context)
        {
            int count = 1;
            if(context->kernel()->wavefront_size() == 64)
            {
                count = 2;
            }

            return Placeholder(context, Type::Scalar, DataType::Raw32, count);
        }

        bool Value::isVCC() const
        {
            auto context = m_context.lock();
            if(context && m_regType == Type::Special)
            {
                return (
                    (context->kernel()->wavefront_size() == 64 && m_specialName == SpecialType::VCC)
                    || (context->kernel()->wavefront_size() == 32
                        && m_specialName == SpecialType::VCC_LO));
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
        Generator<RegisterId> Value::getRegisterIds() const
        {
            if(m_regType == Type::Literal || m_regType == Type::Label)
            {
                co_return;
            }
            if(m_regType == Type::Special)
            {
                auto context = m_context.lock();
                if(context->kernel()->wavefront_size() == 64 && m_specialName == SpecialType::VCC)
                {
                    co_yield RegisterId(m_regType, static_cast<int>(SpecialType::VCC_LO));
                    co_yield RegisterId(m_regType, static_cast<int>(SpecialType::VCC_HI));
                }
                else
                {
                    co_yield RegisterId(m_regType, static_cast<int>(m_specialName));
                }
            }
            else
            {
                AssertFatal(allocationState() == AllocationState::Allocated,
                            ShowValue(allocationState()));

                for(int coord : m_allocationCoord)
                {
                    co_yield RegisterId(m_regType, m_allocation->m_registerIndices.at(coord));
                }
            }
        }

        Expression::ExpressionPtr Value::expression()
        {
            AssertFatal(this, "Null expression accessed");
            return std::make_shared<Expression::Expression>(shared_from_this());
        }

        std::vector<ValuePtr> Value::split(std::vector<std::vector<int>> const& indices)
        {
            // All the indices must be within the allocation.
            {
                std::unordered_set<int> seenIndices;
                for(auto const& segment : indices)
                {
                    for(int idx : segment)
                    {
                        AssertFatal(idx < m_allocationCoord.size());
                        AssertFatal(!seenIndices.contains(idx), ShowValue(idx));

                        seenIndices.insert(idx);
                    }
                }
            }

            std::vector<ValuePtr> rv;
            rv.reserve(indices.size());

            for(auto const& segment : indices)
            {
                rv.push_back(subset(segment));
                rv.back()->takeAllocation();
            }

            m_allocation.reset();

            return rv;
        }

        void Value::takeAllocation()
        {
            AssertFatal(allocationState() == AllocationState::Allocated);
            AssertFatal(!m_ldsAllocation);

            std::vector<int> coords, indices;

            coords.reserve(m_allocationCoord.size());
            indices.reserve(m_allocationCoord.size());

            for(int i = 0; i < m_allocationCoord.size(); i++)
            {
                coords.push_back(i);
                indices.push_back(m_allocation->m_registerIndices[m_allocationCoord[i]]);
            }

            auto newAlloc     = m_context.lock()->allocator(m_regType)->reassign(indices);
            m_allocation      = newAlloc;
            m_allocationCoord = coords;
            m_contiguousIndices.reset();
        }

        bool Value::intersects(ValuePtr input) const
        {
            if(regType() != input->regType())
                return false;

            if(allocationState() != AllocationState::Allocated
               || input->allocationState() != AllocationState::Allocated)
                return false;

            for(int a : registerIndices())
            {
                for(int b : input->registerIndices())
                {
                    if(a == b)
                    {
                        return true;
                    }
                }
            }

            return false;
        }
    }
}
