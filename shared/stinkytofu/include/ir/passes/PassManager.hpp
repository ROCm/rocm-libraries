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
#include <vector>

namespace stinkytofu
{
    // Forward declarations
    class IRModule;
    class IRInstruction;
    struct StinkyInstruction;

    /**
     * @brief Base class for high-level IRInstruction passes
     *
     * Operates on IRInstruction* (high-level, architecture-independent IR).
     * Passes can be composed into pipelines using IRInstPassManager for
     * optimization and lowering sequences.
     *
     * Naming convention:
     * - IRInstPass - operates on IRInstruction (high-level IR)
     * - Pass       - operates on StinkyInstruction (assembly IR, stinkytofu.hpp)
     */
    class IRInstPass
    {
    public:
        virtual ~IRInstPass() = default;

        /**
         * @brief Get the name of this pass (for debugging/logging)
         */
        virtual const char* getName() const = 0;

        /**
         * @brief Check if this pass produces assembly instructions
         *
         * Most passes produce IRInstruction*, but the final lowering pass
         * (ToStinkyAsmPass) produces StinkyInstruction*.
         */
        virtual bool producesAsm() const
        {
            return false;
        }
    };

    /**
     * @brief Pass that transforms IRInstructions to other IRInstructions
     *
     * Example: CompositeInstructionLoweringPass expands VAddPKF32 -> VAddF32
     */
    class IRInstTransformPass : public IRInstPass
    {
    public:
        /**
         * @brief Transform an IR instruction
         * @param irInst Input IR instruction
         * @param arch Target architecture for capability queries
         * @return Vector of output IR instructions (may be 0, 1, or many)
         */
        virtual std::vector<IRInstruction*> transform(IRInstruction* irInst, GfxArchID arch) = 0;

        bool producesAsm() const override
        {
            return false;
        }
    };

    /**
     * @brief Pass that converts IRInstructions to assembly instructions
     *
     * This is the final lowering pass in the pipeline.
     * Example: ToStinkyAsmPass converts VAddF32 (IRInstruction) -> v_add_f32 (StinkyInstruction)
     */
    class IRInstToAsmPass : public IRInstPass
    {
    public:
        /**
         * @brief Lower an IR instruction to assembly
         * @param irInst Input IR instruction
         * @param arch Target architecture for mnemonic mapping
         * @return Vector of output assembly instructions (typically 1)
         */
        virtual std::vector<StinkyInstruction*> lower(IRInstruction* irInst, GfxArchID arch) = 0;

        bool producesAsm() const override
        {
            return true;
        }
    };

    /**
     * @brief Manages a pipeline of IRInstruction transformation and lowering passes
     *
     * IRInstPassManager operates on high-level IRInstruction (IRModule -> IRListModule).
     * It can run optimization passes (constant folding, peephole, etc.) and
     * lowering passes (composite expansion, IRInstruction->StinkyInstruction conversion).
     *
     * Naming convention:
     * - IRInstPassManager - manages IRInstPass (operates on IRInstruction)
     * - PassManager       - manages Pass (operates on StinkyInstruction, stinkytofu.hpp)
     *
     * TODO: Unify with PassManager using LLVM-style template approach:
     *       template<typename IRUnitT> class PassManager;
     *       using ModulePassManager = PassManager<IRModule>;
     *       using FunctionPassManager = PassManager<Function>;
     *       This would provide a single unified pass infrastructure across
     *       all IR levels (Module, Function, BasicBlock, Instruction).
     *
     * Example usage:
     * @code
     *   IRInstPassManager pm(GfxArchID::Gfx942);
     *   pm.addPass(createCompositeInstructionLoweringPass());  // Lowering
     *   pm.addPass(createToStinkyAsmPass());                    // IRInst->Asm
     *   auto asmModule = pm.run(irModule.get(), "output_kernel");
     * @endcode
     */
    class IRInstPassManager
    {
    public:
        /**
         * @brief Construct an IRInstPassManager for a target architecture
         * @param arch Target architecture for all passes
         */
        explicit IRInstPassManager(GfxArchID arch);

        /**
         * @brief Add an IRInstruction pass to the pipeline using a factory function
         *
         * @param pass Unique pointer to the pass (from factory function)
         */
        void addPass(std::unique_ptr<IRInstPass> pass)
        {
            passes.push_back(std::move(pass));
        }

        /**
         * @brief Run the pass pipeline on an IRModule
         *
         * Executes all registered passes in order, producing assembly instructions.
         *
         * @param module Input IRModule (high-level IRInstruction*)
         * @return Unique pointer to IRList containing lowered assembly instructions (StinkyInstruction*)
         *
         * @note The pipeline must end with an IRInstToAsmPass (like ToStinkyAsmPass)
         */
        std::unique_ptr<IRList> run(IRModule* module);

    private:
        GfxArchID                                arch;
        std::vector<std::unique_ptr<IRInstPass>> passes;
    };

} // namespace stinkytofu
