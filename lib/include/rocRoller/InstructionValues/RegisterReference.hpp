#pragma once

#include "Register.hpp"

namespace rocRoller
{
    namespace Register
    {
        /**
         * @brief Copy for a register Value without the associated allocation lifecycle
         *
         * This is a structure to record how a register was setup with an instruction.
         * This is useful for referencing instructions that were previously generated.
         */
        class RegisterReference
        {
        public:
            RegisterReference() {}
            RegisterReference(ValuePtr const value)
            {
                m_registerType = value->regType();
                m_isExec       = value->isExec();

                if(value->allocationState() == AllocationState::Allocated
                   && m_registerType != Type::Label && m_registerType != Type::Literal
                   && m_registerType != Type::Special)
                {
                    for(int index : value->registerIndices())
                    {
                        m_registerIndices.push_back(index);
                    }
                }
            }

            bool isExec() const
            {
                return m_isExec;
            }

            std::vector<int> getRegisterIndices() const
            {
                return m_registerIndices;
            }

            Type getRegisterType() const
            {
                return m_registerType;
            }

            bool exactlyOverlaps(RegisterReference const& other) const
            {
                return m_registerType == other.getRegisterType()
                       && m_registerIndices == other.getRegisterIndices();
            }

            bool intersects(RegisterReference const& other) const
            {
                if(m_registerType == other.getRegisterType())
                {
                    for(int const regA : m_registerIndices)
                    {
                        for(int const regB : other.getRegisterIndices())
                        {
                            if(regA == regB)
                            {
                                return true;
                            }
                        }
                    }
                }
                return false;
            }

        private:
            bool             m_isExec;
            Type             m_registerType;
            std::vector<int> m_registerIndices;
        };
    }
}
