/**
 */

#pragma once

#include <string>
#include <unordered_map>

#include "Instruction.hpp"

#include "../AssemblyKernel.hpp"
#include "../InstructionValues/Register_fwd.hpp"
#include "../Utilities/Generator.hpp"

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
        ArgumentLoader(std::shared_ptr<AssemblyKernel> kernel);

        /**
         * Loads all arguments into a single allocation of SGPRs.  Uses the widest load
         * instructions possible given alignment constraints.
         */
        Generator<Instruction> loadAllArguments();

        /**
         * Obtain the `Value` for a given argument.  If the argument has not been loaded yet,
         * emits instructions to load it into SGPRs.
         *
         * @param argName
         * @param value
         */
        Generator<Instruction> getValue(std::string const& argName, Register::ValuePtr& value);

        Generator<Instruction> loadArgument(std::string const& argName);
        Generator<Instruction> loadArgument(AssemblyKernelArgument const& arg);

        void releaseArgument(std::string const& argName);
        void releaseAllArguments();

        Generator<Instruction>
            loadRange(int offset, int sizeBytes, Register::ValuePtr& value) const;

    private:
        std::weak_ptr<Context>          m_context;
        std::shared_ptr<AssemblyKernel> m_kernel;

        /// Call the one from the AssemblyKernel instead.
        Register::ValuePtr argumentPointer() const;

        std::unordered_map<std::string, Register::ValuePtr> m_loadedValues;
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

#include "ArgumentLoader_impl.hpp"
