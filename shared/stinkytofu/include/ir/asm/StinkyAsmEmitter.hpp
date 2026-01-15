/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <ostream>
#include <sstream>
#include <string>

#include "ir/asm/StinkyAsmDirectives.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "ir/asm/StinkyMacro.hpp"
#include "stinkytofu.hpp"

namespace stinkytofu
{
    /**
     * Configuration options for assembly code emission.
     */
    struct AsmEmitterOptions
    {
        // Whether to include comments in the output
        bool emitComments = true;

        // Whether to emit cycle count information as comments
        bool emitCycleInfo = false;

        // Indentation for instructions (spaces)
        int indent = 4;

        // Whether to emit blank lines between instruction groups
        bool emitBlankLines = false;

        // Whether to use symbolic register names when available
        // If true: v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+3]
        // If false: v10, v[46:49]
        bool useSymbolicNames = false;

        // Column position for aligning comments (0 = no alignment)
        // When > 0, comments will be aligned at this column position
        // e.g., 50 will pad instruction to column 50 before adding comment
        int commentAlignColumn = 51;
    };

    /**
     * StinkyAsmEmitter - Converts StinkyTofu IR to actual assembly code.
     *
     * This class takes MLIR-style StinkyTofu IR and emits actual GPU assembly
     * code that can be assembled by the GPU assembler.
     *
     * Example usage:
     * ```cpp
     * IRList irlist;
     * // ... populate irlist ...
     *
     * AsmEmitterOptions options;
     * options.emitComments = true;
     * options.emitCycleInfo = true;
     *
     * StinkyAsmEmitter emitter(options);
     * std::string assembly = emitter.emit(irlist);
     * // or
     * emitter.emit(irlist, std::cout);
     * ```
     */
    class StinkyAsmEmitter
    {
    public:
        /**
         * Constructor with default options.
         */
        StinkyAsmEmitter()
            : options(AsmEmitterOptions())
        {
        }

        /**
         * Constructor with custom options.
         *
         * @param opts Configuration options for assembly emission
         */
        StinkyAsmEmitter(const AsmEmitterOptions& opts)
            : options(opts)
        {
        }

        /**
         * Emit assembly code for a single instruction to a stream.
         *
         * @param os Output stream
         * @param inst The StinkyInstruction to emit
         */
        void emit(std::ostream& os, const StinkyInstruction& inst);

        /**
         * Emit assembly code for an entire IRList to a stream.
         *
         * @param os Output stream
         * @param irlist The IRList to emit
         */
        void emit(std::ostream& os, const IRList& irlist);

        /**
         * Emit assembly code for a single instruction as a string.
         *
         * @param inst The StinkyInstruction to emit
         * @return Assembly code as string
         */
        std::string emit(const StinkyInstruction& inst);

        /**
         * Emit assembly code for an entire IRList as a string.
         *
         * @param irlist The IRList to emit
         * @return Assembly code as string
         */
        std::string emit(const IRList& irlist);

        /**
         * Emit assembly code for a directive to a stream.
         */
        void emit(std::ostream& os, const AsmDirective& directive);

        /**
         * Emit assembly code for a directive as a string.
         */
        std::string emit(const AsmDirective& directive);

        /**
         * Emit assembly code for a macro to a stream (MLIR-like syntax).
         */
        void emit(std::ostream& os, const MacroInstruction& macro);

        /**
         * Emit assembly code for a macro as a string (MLIR-like syntax).
         */
        std::string emit(const MacroInstruction& macro);

        /**
         * Get the current options.
         */
        const AsmEmitterOptions& getOptions() const
        {
            return options;
        }

        /**
         * Set new options.
         */
        void setOptions(const AsmEmitterOptions& opts)
        {
            options = opts;
        }

    private:
        AsmEmitterOptions options;

        /**
         * Emit a register in assembly syntax.
         */
        void emitRegister(std::ostream& os, const StinkyRegister& reg);

        /**
         * Emit instruction mnemonic.
         */
        void emitMnemonic(std::ostream& os, const StinkyInstruction& inst);

        /**
         * Emit operands for an instruction.
         */
        void emitOperands(std::ostream& os, const StinkyInstruction& inst);

        /**
         * Emit memory modifiers (DS, FLAT, MUBUF, SMEM) after operands.
         */
        void emitMemoryModifiers(std::ostream& os, const StinkyInstruction& inst);

        /**
         * Emit cycle information as a comment.
         * @param os Output stream
         * @param inst The instruction
         * @param currentColumn Current column position (for alignment)
         */
        void emitCycleComment(std::ostream& os, const StinkyInstruction& inst, int currentColumn);
    };

    /**
     * Utility function to convert IRList to assembly code string.
     *
     * @param irlist The IRList to convert
     * @param options Emission options
     * @return Assembly code as string
     */
    inline std::string toAssembly(const IRList&            irlist,
                                  const AsmEmitterOptions& options = AsmEmitterOptions())
    {
        StinkyAsmEmitter emitter(options);
        return emitter.emit(irlist);
    }

    /**
     * Utility function to convert a single instruction to assembly code string.
     *
     * @param inst The instruction to convert
     * @param options Emission options
     * @return Assembly code as string
     */
    inline std::string toAssembly(const StinkyInstruction& inst,
                                  const AsmEmitterOptions& options = AsmEmitterOptions())
    {
        StinkyAsmEmitter emitter(options);
        return emitter.emit(inst);
    }

} // namespace stinkytofu
