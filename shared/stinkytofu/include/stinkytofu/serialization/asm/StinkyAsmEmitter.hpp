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

#include <iosfwd>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {
struct StinkyInstruction;

/// Configuration options for assembly code emission.
struct AsmEmitterOptions {
    /// Whether to include comments in the output
    bool emitComments = true;

    /// Whether to emit cycle count information as comments
    bool emitCycleInfo = false;

    /// Indentation for instructions (spaces)
    int indent = 4;

    /// Whether to emit blank lines between instruction groups
    bool emitBlankLines = false;

    /// Whether to use symbolic register names when available
    /// If true: v[vgprLocalWriteAddrA+0], v[vgprG2LA+0:vgprG2LA+3]
    /// If false: v10, v[46:49]
    bool useSymbolicNames = false;

    /// Column position for aligning comments (0 = no alignment)
    /// When > 0, comments will be aligned at this column position
    /// e.g., 50 will pad instruction to column 50 before adding comment
    int commentAlignColumn = 51;
};

/// StinkyAsmEmitter - Converts StinkyTofu IR to actual GPU assembly code.
///
/// Example usage:
/// \code
/// Function func("my_kernel");
/// ... populate Function ...
///
/// AsmEmitterOptions options;
/// options.emitComments = true;
/// options.emitCycleInfo = true;
///
/// StinkyAsmEmitter emitter(options);
/// std::string assembly = emitter.emit(func);
///
/// (or) emitter.emit(func, std::cout);
/// \endcode
class STINKYTOFU_EXPORT StinkyAsmEmitter {
   public:
    StinkyAsmEmitter() : options(AsmEmitterOptions()) {}

    StinkyAsmEmitter(const AsmEmitterOptions& opts) : options(opts) {}

    /// Emit assembly for a single instruction or an entire Function,
    /// either to a stream or as a string.
    void emit(std::ostream& os, const StinkyInstruction& inst);
    void emit(std::ostream& os, const Function& function);

    std::string emit(const StinkyInstruction& inst);
    std::string emit(const Function& function);

    const AsmEmitterOptions& getOptions() const {
        return options;
    }

    void setOptions(const AsmEmitterOptions& opts) {
        options = opts;
    }

   private:
    AsmEmitterOptions options;
};

/// Utility function to convert Function to assembly code string.
inline std::string toAssembly(const Function& function,
                              const AsmEmitterOptions& options = AsmEmitterOptions()) {
    StinkyAsmEmitter emitter(options);
    return emitter.emit(function);
}

/// Utility function to convert a single instruction to assembly code string.
inline std::string toAssembly(const StinkyInstruction& inst,
                              const AsmEmitterOptions& options = AsmEmitterOptions()) {
    StinkyAsmEmitter emitter(options);
    return emitter.emit(inst);
}

}  // namespace stinkytofu
