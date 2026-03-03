// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 */

#pragma once

#include <string>
#include <unordered_map>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRollerTest
{
    class ArgumentLoaderTest_loadArgExtra_Test;
}

namespace rocRoller
{
    /**
     * Generates code to load argument values from the kernel argument buffer into SGPRs.
     * Supports two main strategies: all-at-once, and on-demand.
     *
     * All-at-once can be faster since we can use wider load instructions, but so far this
     * requires allocating all the argument SGPRs in one block so they can't be freed one by one
     * to allow reuse.
     *
     * On-demand is more flexible, but also potentially slower as it will require more load
     * instructions as well as possibly more synchronization.
     *
     */
    class ArgumentLoader
    {
    public:
        ArgumentLoader(AssemblyKernelPtr kernel);

        /**
         * Loads all manually loaded arguments into a single allocation of SGPRs.  Uses the
         * widest load instructions possible given alignment constraints.
         */
        Generator<Instruction> eagerLoadArguments();

        /**
         * Obtain the `Value` for a given argument.  If the argument has not been loaded yet,
         * emits instructions to load it into SGPRs.
         *
         * @param argName
         * @param value
         */
        Generator<Instruction> getValue(std::string const& argName, Register::ValuePtr& value);

        void releaseArgument(std::string const& argName);
        void releaseAllArguments();

        Generator<Instruction>
            loadRange(int offset, int sizeBytes, Register::ValuePtr& value) const;

        /**
         * Allocates the block of registers that will be preloaded with kernel arguments at the
         * beginning of the kernel execution. This must be called at the correct time within 
         * AllocateInitialRegisters as it is part of the initial register state.
         *
         * Returns the appropriate values for the kernel description fields
         * kernarg_preload_offset and kernarg_preload_length through preloadOffset and
         * preloadLength respectively.
         */
        Generator<Instruction> allocatePreloadedRegisters(int& preloadOffset, int& preloadLength);

        bool anyPreloadedArguments() const;
        bool anyManuallyLoadedArguments() const;

        /**
         * Decides which if any kernel arguments can be preloaded based on architecture and
         * kernel options.
         * 
         * The priority is to pick the earliest arguments first, but it will pick out-of-order
         * arguments if alignment prevents earlier arguments from being preloaded.
         *
         * `args` will be partitioned into preloaded and non-preloaded arguments, and then
         * sorted descending by size.
         */
        void decidePreloadedKernargs(std::vector<AssemblyKernelArgument>& args);

        /**
         * Splits the block allocations of kernel arguments into individual registers and clears
         * the block allocations so that individual kernel arguments can be deallocated individually. 
         */
        Generator<Instruction> splitOutArgumentRegisters();

    private:
        friend class rocRollerTest::ArgumentLoaderTest_loadArgExtra_Test;

        std::weak_ptr<Context> m_context;

        AssemblyKernelPtr m_kernel;

        Generator<Instruction> loadArgument(std::string const& argName);
        Generator<Instruction> loadArgument(AssemblyKernelArgument const& arg);

        /// Call the one from the AssemblyKernel instead.
        Register::ValuePtr argumentPointer() const;

        Generator<Instruction> splitOutArgs(Register::ValuePtr rawRegs, int beginOffset);

        std::unordered_map<std::string, Register::ValuePtr> m_loadedValues;

        Register::ValuePtr m_preloadedBlock;
        Register::ValuePtr m_manuallyLoadedBlock;
        int                m_manuallyLoadedOffset = 0;

        void populateAnyArgumentsFlags() const;

        mutable std::optional<bool> m_anyPreloadedArguments;
        mutable std::optional<bool> m_anyManuallyLoadedArguments;
    };

    /**
     * Picks the widest load instruction to load some of argPtr[offset:endOffset]
     * into s[*beginReg:*endReg]. Returns the width of that load instruction in
     * bytes, or 0 if offset == endOffset.
     *
     * Reasons to decrease the width of the load:
     *  - offset is not aligned to the width of the load
     *  - endOffset-offset is less than the width of the load
     *  - *beginReg is not aligned to the width of the load
     *  - destination registers are not contiguous
     *
     * @return int Width of the load in bytes
     */
    template <typename Iter, typename End>
    inline int PickInstructionWidthBytes(int offset, int endOffset, Iter beginReg, End endReg);

}

#include <rocRoller/CodeGen/ArgumentLoader_impl.hpp>
