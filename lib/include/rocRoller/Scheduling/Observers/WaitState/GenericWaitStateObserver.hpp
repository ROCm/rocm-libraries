#pragma once

#include <rocRoller/Scheduling/Scheduling.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>
#include <rocRoller/Context.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        class GenericWaitStateObserver
        {
        public:
            GenericWaitStateObserver() {}
            GenericWaitStateObserver(std::shared_ptr<Context> context)
                : m_context(context){};

            InstructionStatus peek(Instruction const& inst) const;
            void              modify(Instruction& inst) const;
            InstructionStatus observe(Instruction const& inst);

        protected:
            std::weak_ptr<Context> m_context;

            /**
             * @brief Get the maximum number of NOPs for the worst case.
             *
             * This amount is used to populate the RegisterHazardMap.
             *
             * @param inst The instruction reference that is analyzed for hazard potential.
             * @return int Worst case NOPs required if the hazard is discovered.
             */
            virtual int getMaxNops(std::shared_ptr<InstructionRef> inst) const = 0;

            /**
             * @brief The condition that could trigger the hazard for this rule.
             *
             * @param inst The instruction reference that is analyzed for hazard potential.
             * @return True if the instruction could cause a hazard according to this rule.
             */
            virtual bool trigger(std::shared_ptr<InstructionRef> inst) const = 0;

            /**
             * @brief Is this hazard caused by writing a register?
             *
             * Note: Create seperate rules if reading and writing both cause hazards.
             *
             * @return True if the hazard is caused by writing registers, False if by reading.
             */
            virtual bool writeTrigger() const = 0;

            /**
             * @brief Get the number of NOPs for the instruction according to this hazard rule.
             *
             * Holds the logic for determining whether an instruction causes a hazard according
             * to this hazard rule. Should check the RegisterHazardMap to determine if the conditions
             * for the rule are met. If so, a number of NOPs are returned according to the number in
             * the RegisterHazardMap. If not, return 0 NOPs.
             *
             * @param inst The instruction to be analyzed
             * @return int The number of NOPs this instruction should have according to this rule.
             */
            virtual int getNops(Instruction const& inst) const = 0;
        };
    }
}
