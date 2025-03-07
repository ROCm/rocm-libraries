/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

namespace rocRoller
{
    namespace Register
    {
        ValuePtr Value::WavefrontPlaceholder(ContextPtr context, int count)
        {
            auto dataType = DataType::Bool32;
            if(context->kernel()->wavefront_size() == 64)
            {
                dataType = DataType::Bool64;
            }

            return Placeholder(context,
                               Type::Scalar,
                               dataType,
                               count,
                               Register::AllocationOptions::FullyContiguous());
        }

        bool Value::isVCC() const
        {
            auto context = m_context.lock();
            if(context && IsSpecial(m_regType))
            {
                return (
                    (context->kernel()->wavefront_size() == 64 && m_regType == Type::VCC)
                    || (context->kernel()->wavefront_size() == 32 && m_regType == Type::VCC_LO));
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
            if(IsSpecial(m_regType))
            {
                auto context = m_context.lock();
                if(context->kernel()->wavefront_size() == 64 && m_regType == Type::VCC)
                {
                    co_yield RegisterId(Type::VCC_LO, static_cast<int>(Type::VCC_LO));
                    co_yield RegisterId(Type::VCC_HI, static_cast<int>(Type::VCC_HI));
                }
                else
                {
                    co_yield RegisterId(m_regType, static_cast<int>(m_regType));
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

        bool Value::sameAs(ValuePtr b) const
        {
            return this->regType() == b->regType() && this->variableType() == b->variableType()
                   && this->registerCount() == b->registerCount()
                   && this->allocationCoord() == b->allocationCoord()
                   && this->allocation() == b->allocation();
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
