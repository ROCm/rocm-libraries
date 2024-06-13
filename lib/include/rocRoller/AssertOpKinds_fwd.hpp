#pragma once

namespace rocRoller
{
    /**
     * Dictates which instructions are generated when lowering an AssertOp.
     *
     * NoOp - no instructions are generated. Used to allow execution of kernels with AssertOps without impacting performance.
     * MemoryViolation - generates instructions to write to nullptr and trigger a memory violation in kernel code.
     * STrap - generates a s_trap instruction interpreted by HSA's trap handler as a debug trap.
     */
    enum class AssertOpKind
    {
        NoOp,
        MemoryViolation,
        STrap,
        Count,
    };
}
