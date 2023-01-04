
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "../CodeGen/Instruction_fwd.hpp"
#include "../CodeGen/WaitCount.hpp"
#include "../Context_fwd.hpp"

namespace rocRoller
{
    namespace Scheduling
    {

        struct InstructionStatus
        {
            unsigned int             stallCycles = 0;
            WaitCount                waitCount;
            unsigned int             nops = 0;
            std::vector<std::string> errors;

            static InstructionStatus StallCycles(unsigned int const value);
            static InstructionStatus Wait(WaitCount const& value);
            static InstructionStatus Nops(unsigned int const value);
            static InstructionStatus Error(std::string const& msg);

            void combine(InstructionStatus const& other);
        };

        template <typename T>
        concept CObserver = requires(T a, Instruction inst, ContextPtr const& ctx)
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
            {a.observe(inst)};

            //> This observer is required in ctx.
            {
                a.required(ctx)
                } -> std::convertible_to<bool>;
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
            virtual void observe(Instruction const& inst) = 0;
        };

    }

}

#include "Scheduling_impl.hpp"
