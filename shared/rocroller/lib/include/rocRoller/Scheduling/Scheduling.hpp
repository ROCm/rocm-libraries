// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/CodeGen/Instruction_fwd.hpp>
#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Utilities/EnumArray.hpp>
#include <rocRoller/Utilities/EnumBitset.hpp>

namespace rocRoller
{
    using DisallowedCycles = std::map<int, EnumBitset<CoexecCategory>>;
    void        combineCoexec(DisallowedCycles& dst, DisallowedCycles const& src, int offset);
    std::string toString(DisallowedCycles const& cycles);

    namespace Scheduling
    {

        /**
         * Struct that describes modeled properties of an instruction if it were to be scheduled now.
         *
         * Note that any new members need to:
         * 1. Have a neutral default value, either in the declaration or in the constructor
         * 2. Be added to the combine() function which is used to merge values from different observers.
         */
        struct InstructionStatus
        {
            unsigned int stallCycles      = 0;
            unsigned int additionalCycles = 0;
            WaitCount    waitCount;
            unsigned int nops = 0;

            unsigned int reusedOperands = 0;

            /// The new length of each of the queues.
            EnumArray<int, GPUWaitQueueType> waitLengths;

            /// How many new registers of each type must be allocated?
            EnumArray<int, Register::Type> allocatedRegisters;

            /// How many free registers of each type will remain?
            EnumArray<int, Register::Type> remainingRegisters;

            /// How much does this instruction add to the high water mark of allocated registers?
            EnumArray<int, Register::Type> highWaterMarkRegistersDelta;

            /// Will this cause an out-of-registers error?
            EnumBitset<Register::Type> outOfRegisters;

            DisallowedCycles disallowedCoexec;

            std::vector<std::string> errors;

            static InstructionStatus StallCycles(unsigned int const value);
            static InstructionStatus Wait(WaitCount const& value);
            static InstructionStatus Nops(unsigned int const value);
            static InstructionStatus Error(std::string const& msg);

            /**
             * Merge values of members from `other` into `this`.
             */
            void combine(InstructionStatus const& other);

            std::string coexecString() const;

            InstructionStatus();

            std::string toString() const;
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
            {a.observe(inst)};
        };

        template <typename T>
        concept CObserverConst = requires(T a, GPUArchitectureTarget const& target)
        {
            requires CObserver<T>;

            //> This observer is required in ctx, checking GPUArchitectureTarget if needed.
            {
                a.required(target)
                } -> std::convertible_to<bool>;
        };

        template <typename T>
        concept CObserverRuntime = requires(T a)
        {
            requires CObserver<T>;

            //> This observer is required in ctx, determined at runtime.
            {
                a.runtimeRequired()
                } -> std::convertible_to<bool>;
        };

        template <typename T>
        concept CObserverRuntimeWithContext = requires(T a, ContextPtr const& ctx)
        {
            requires CObserver<T>;

            //> This observer requires ctx to determine if it's required at runtime.
            {
                a.runtimeRequired(ctx)
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

#include <rocRoller/Scheduling/Scheduling_impl.hpp>
