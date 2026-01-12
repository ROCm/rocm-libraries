/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "stinkytofu.hpp"
#include <memory>
#include <string>
#include <vector>

namespace stinkytofu
{
    // Forward declarations
    class IRInstruction;
    class IRListModule;

    // ========================================================================
    // TODO: Remove IRListModule - it's no longer needed!
    // ========================================================================
    //
    // IRListModule was originally created to mimic rocisa's Module behavior
    // for exposing assembly instructions to Python. Now with the new high-level
    // IR system, we should:
    //
    // 1. Remove IRListModule entirely
    // 2. Expose only IRModule to Python bindings
    // 3. Assembly instructions (StinkyInstruction*) are internal implementation
    //    details handled by lowering passes
    //
    // New architecture:
    //   Python -> IRModule (high-level IR)
    //          ? (IRInstPassManager with lowering passes)
    //          -> Function/BasicBlock with StinkyInstruction* (internal assembly)
    //          ? (assembly passes: peephole, scheduling, etc.)
    //          -> Final assembly string
    //
    // This simplifies the system:
    //   - Python users only see IRModule + IRInstruction (high-level)
    //   - No need to expose assembly IR (StinkyInstruction*) to Python
    //   - IRListModule becomes obsolete
    //
    // Refactor steps:
    //   1. Update Python bindings to use IRModule instead of IRListModule
    //   2. Update all C++ tests to use IRModule + lowering passes
    //   3. Remove IRListModule class and all references
    //   4. Keep IRList (IntrusiveList) for internal assembly representation
    // ========================================================================

    /**
     * @brief High-level IR Module container (LLVM-style)
     *
     * IRModule holds high-level, architecture-independent IR instructions
     * (stinkytofu::IRInstruction*) and provides a pass pipeline to lower
     * them to assembly instructions (StinkyInstruction*).
     *
     * Architecture:
     *   IRModule (high-level IR) -> Lowering Passes -> IRListModule (assembly IR)
     *
     * The IRModule itself is architecture-independent. The target architecture
     * is specified only when lowering to assembly.
     *
     * Example usage:
     * @code
     *   auto module = std::make_shared<IRModule>("myKernel");
     *
     *   // Add high-level IRInstructions (architecture-independent)
     *   module->add(new VAddF32(vgpr(0), vgpr(1), vgpr(2)));
     *   module->add(new VAddPKF32(vgpr(3), vgpr(4), vgpr(5)));  // Composite
     *
     *   // Run optimization and lowering passes
     *   IRInstPassManager pm(GfxArchID::Gfx942);
     *   // pm.addPass(createConstantFoldingPass());              // Future: IR optimization
     *   pm.addPass(createCompositeInstructionLoweringPass());    // Lowering
     *   pm.addPass(createToStinkyAsmPass());                      // IRInst->StinkyInst
     *   auto asmModule = pm.run(module.get());
     *   std::string asm = asmModule->emitAssembly();
     * @endcode
     */
    class IRModule
    {
    public:
        /**
         * @brief Construct a new IRModule
         * @param name Module/kernel name
         */
        IRModule(const std::string& name = "");

        /**
         * @brief Destructor - owns and deletes all IRInstructions
         */
        ~IRModule();

        // Disable copy (we own the instructions)
        IRModule(const IRModule&)            = delete;
        IRModule& operator=(const IRModule&) = delete;

        // Enable move
        IRModule(IRModule&&) noexcept;
        IRModule& operator=(IRModule&&) noexcept;

        /**
         * @brief Get the module name
         * @return Module name string
         */
        std::string getName() const;

        /**
         * @brief Add a single IR instruction to this module
         *
         * The module shares ownership of the instruction via shared_ptr.
         *
         * @param inst IR instruction to add
         * @return The same instruction (for chaining)
         */
        std::shared_ptr<IRInstruction> add(std::shared_ptr<IRInstruction> inst);

        /**
         * @brief Get all IR instructions in this module (const version)
         * @return Vector of IR instructions (shared ownership)
         */
        const std::vector<std::shared_ptr<IRInstruction>>& getInstructions() const;

        /**
         * @brief Get all IR instructions in this module (non-const version)
         * @return Vector of IR instructions (shared ownership)
         */
        std::vector<std::shared_ptr<IRInstruction>>& getMutableInstructions();

        /**
         * @brief Remove an instruction from the module
         * @param inst Pointer to the instruction to remove
         * @return true if instruction was found and removed, false otherwise
         */
        bool removeInstruction(IRInstruction* inst);

        /**
         * @brief Get number of instructions in this module
         * @return Instruction count
         */
        size_t size() const;

        /**
         * @brief Dump the IR instructions for debugging
         * @param out Output stream
         */
        void dump(std::ostream& out) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace stinkytofu
