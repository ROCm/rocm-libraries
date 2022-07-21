
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "../CodeGen/WaitCount.hpp"

namespace rocRoller
{

#if 0
    namespace Instructions
    {

        template <DataType D>
        InstructionGenerator Add(std::shared_ptr<Register::Value> & a, std::shared_ptr<Register::Value> & b)
        {
            co_yield Instruction("v_add_u32", a.placeholder(), {a, b}, {});
        }

    }
#endif

    namespace Scheduling
    {

        struct InstructionStatus
        {
            unsigned int             stallCycles             = 0;
            unsigned int             unscheduledDependencies = 0;
            WaitCount                waitCount;
            std::vector<std::string> errors;

            static InstructionStatus StallCycles(unsigned int value);
            static InstructionStatus UnscheduledDependencies(unsigned int value);
            static InstructionStatus Wait(WaitCount const& value);
            static InstructionStatus Error(std::string const& msg);

            void combine(InstructionStatus const& other);
        };

        template <typename T>
        concept CObserver = requires(T a, Instruction inst)
        {
            //> Speculatively predict stalls if this instruction were scheduled now.
            {
                a.peek(inst)
                } -> std::convertible_to<InstructionStatus>;

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            {a.modify(inst)};

            //> This instruction _will_ be scheduled now, record any side effects.
            //> This is after all observers have had the opportunity to modify the instruction.
            {
                a.observe(inst)
                } -> std::convertible_to<InstructionStatus>;
        };

        struct IObserver
        {
            virtual ~IObserver();

            //> Speculatively predict stalls if this instruction were scheduled now.
            virtual InstructionStatus peek(Instruction const& inst) const = 0;

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            virtual void modify(Instruction& inst) const = 0;

            //> This instruction _will_ be scheduled now, record any side effects.
            virtual InstructionStatus observe(Instruction const& inst) = 0;
        };

    }

}

#include "Scheduling_impl.hpp"
