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

#include <array>
#include <memory>
#include <string>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {
enum class GfxArchID : uint32_t;
struct ParsedFunction;

/**
 * StinkyIRConverter - A utility class for converting raw instruction strings to a Function.
 *
 * This class provides a simple interface to parse MLIR-style StinkyTofu IR text and
 * convert it to a Function that can be used with the StinkyTofu pass infrastructure.
 *
 * Example usage:
 * ```cpp
 * std::string raw_stinky_inst = R"(
 *   v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }
 *   v[4:7] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }
 * )";
 *
 * StinkyIRConverter converter;
 * Function* func = converter.convertToFunction(raw_stinky_inst);
 *
 * // Use the function with passes...
 * // Access the entry BasicBlock and its IR:
 * BasicBlock* entryBB = func->getEntryBlock();
 *
 * // Don't forget to clean up when done
 * converter.cleanup();
 * ```
 */
class STINKYTOFU_EXPORT StinkyIRConverter {
   public:
    /**
     * Constructor with default architecture (gfx1250).
     */
    StinkyIRConverter();

    /**
     * Constructor with specified architecture.
     *
     * @param targetArch The target GPU architecture (e.g., {12, 5, 0} for gfx1250)
     */
    StinkyIRConverter(const std::array<int, 3>& targetArch);

    /**
     * Convert a raw instruction string to a Function.
     *
     * The string should be in MLIR-style StinkyTofu IR format:
     * destRegs = "st.mnemonic"(srcRegs) { attributes }
     *
     * @param rawInstructions The raw instruction string to parse
     * @return Pointer to the Function (owned by this converter until cleanup()), or nullptr if
     * conversion fails
     */
    Function* convertToFunction(const std::string& rawInstructions);

    /**
     * Static helper to populate a Function from an IR text string.
     * This is the core conversion logic shared by both StinkyIRConverter and stinkytofu-opt.
     *
     * @param irText The IR source text to parse
     * @param func The Function to populate with a BasicBlock containing instructions
     * @param passCtx The PassContext for resource management
     * @param arch The target GPU architecture
     * @return StinkyErrorCode indicating success or failure
     */
    static StinkyErrorCode populateFunctionFromString(const std::string& irText, Function& func,
                                                      PassContext& passCtx, GfxArchID arch);

    /// Populate a Function from a pre-parsed ParsedFunction.
    static StinkyErrorCode populateFunctionFromParsed(ParsedFunction& parsedFunc, Function& func,
                                                      GfxArchID arch);

    /**
     * Get the PassContext associated with the last conversion.
     * This is useful if you need to access the context for running passes.
     *
     * @return Pointer to the PassContext, or nullptr if not available
     */
    PassContext* getPassContext();

    /**
     * Cleanup resources. Call this when done with the Function.
     * Releases the converted Function and PassContext.
     */
    void cleanup();

    /**
     * Destructor - automatically cleans up resources.
     */
    ~StinkyIRConverter();

   private:
    std::array<int, 3> arch;
    std::unique_ptr<Function> function;
    std::unique_ptr<stinkytofu::PassContext> passCtx;
};
}  // namespace stinkytofu
