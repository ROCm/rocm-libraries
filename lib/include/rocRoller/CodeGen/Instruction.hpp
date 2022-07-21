/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include <memory>
#include <string>

#include "WaitCount.hpp"

#include "../Context_fwd.hpp"
#include "../InstructionValues/Register_fwd.hpp"

namespace rocRoller
{
    struct Instruction
    {
        Instruction();
        Instruction(std::string const&                                      opcode,
                    std::initializer_list<std::shared_ptr<Register::Value>> dst,
                    std::initializer_list<std::shared_ptr<Register::Value>> src,
                    std::initializer_list<std::string>                      modifiers,
                    std::string const&                                      comment);

        static Generator<std::string> EscapeComment(std::string comment, int indent = 0);

        static Instruction Allocate(std::shared_ptr<Register::Value> reg);
        static Instruction Allocate(std::shared_ptr<Register::Allocation> reg);
        static Instruction
            Allocate(std::initializer_list<std::shared_ptr<Register::Allocation>> regs);

        static Instruction Directive(std::string const& directive);
        static Instruction Directive(std::string const& directive, std::string const& comment);

        static Instruction Comment(std::string const& comment);

        static Instruction Warning(std::string const& warning);

        static Instruction Nop();
        static Instruction Nop(int nopCount);
        static Instruction Nop(std::string const& comment);
        static Instruction Nop(int nopCount, std::string const& comment);

        static Instruction Label(std::string const& name);
        static Instruction Label(std::string&& name);

        static Instruction Label(Register::ValuePtr label);

        static Instruction Wait(WaitCount const& wait);
        static Instruction Wait(WaitCount&& wait);

        std::tuple<std::vector<std::shared_ptr<Register::Value>>,
                   std::vector<std::shared_ptr<Register::Value>>>
                  getRegisters() const;
        bool      hasRegisters() const;
        WaitCount getWaitCount() const;

        bool registersIntersect(std::vector<std::shared_ptr<Register::Value>>,
                                std::vector<std::shared_ptr<Register::Value>>) const;

        void        toStream(std::ostream&, LogLevel level) const;
        std::string toString(LogLevel level) const;

        // Temporary functions to stub out testing. These were added as part
        // of a proof of concept, and aren't intended to be functional.
        // TODO: Remove once the test doesn't call this anymore.
        //{

        template <typename T>
        static Instruction Normal(T&&, std::string const& comment, int = 0)
        {
            return Comment(comment);
        }

        template <typename T>
        static Instruction Allocate(T&&, std::string const& comment, int = 0)
        {
            return Comment("Allocate " + comment);
        }

        int stallCycles() const
        {
            return 0;
        }

        std::string opCode() const;

        void addAllocation(std::shared_ptr<Register::Allocation> alloc);
        void addWaitCount(WaitCount const& wait);
        void addComment(std::string const& comment);
        void addWarning(std::string const& warning);
        void addNop();
        void addNop(int count);
        void setNopMin(int count);

        void allocateNow();

    private:
        enum
        {
            MaxDstRegisters = 2,
            MaxSrcRegisters = 4,
            MaxModifiers    = 10,
            MaxAllocations  = 4
        };

        /**
         * toString = preamble + functional + coda
         */
        void preambleString(std::ostream& oss, LogLevel level) const;
        void functionalString(std::ostream& oss, LogLevel level) const;
        void codaString(std::ostream& oss, LogLevel level) const;

        /**
         * A comment detailing allocations that happened when scheduling this instruction.
         */
        void allocationString(std::ostream& oss, LogLevel level) const;

        /**
         * Assembler directive(s), if this instruction contains an any.
         */
        void directiveString(std::ostream& oss, LogLevel level) const;

        /**
         * Just the main instruction.
         */
        void coreInstructionString(std::ostream& oss, LogLevel level) const;

        /**
         * When this instruction is scheduled, perform this register allocation.
         */
        std::array<std::shared_ptr<Register::Allocation>, MaxAllocations> m_allocations;

        int m_nopCount = 0;

        std::string m_directive;

        std::vector<std::string> m_warnings;

        std::vector<std::string> m_comments;

        std::string m_label;

        WaitCount m_waitCount;

        std::string m_opcode;

        std::array<std::shared_ptr<Register::Value>, MaxDstRegisters> m_dst;
        std::array<std::shared_ptr<Register::Value>, MaxSrcRegisters> m_src;

        std::array<std::string, MaxModifiers> m_modifiers;
    };

}

#include "Instruction_impl.hpp"
