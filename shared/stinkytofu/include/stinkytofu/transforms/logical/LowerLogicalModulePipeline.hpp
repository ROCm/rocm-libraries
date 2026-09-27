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

#include "stinkytofu/Export.hpp"
#include "stinkytofu/bindings/python/Module.hpp"

namespace stinkytofu {
class Function;
class PyLogicalModule;
struct GemmTileConfig;

/**
 * @brief Run the standard logical-IR lowering pipeline on a Function in-place.
 *
 * Equivalent to the inline boilerplate that already appears in many unit tests
 * (e.g. LogicalToAsmPipelineTest, RegisterWidthValidationTest):
 *
 * @code
 *   PassManager pm;
 *   pm.setGemmTileConfig(config);
 *   pm.addPass(createCompositeInstructionLoweringPass());
 *   pm.addPass(createToStinkyAsmPass());
 *   pm.run(func);
 * @endcode
 *
 * After this returns, every reachable LogicalInstruction in @p func has been
 * replaced with a StinkyInstruction (asm IR), and the Function is ready to be
 * emitted via StinkyAsmModule::emitAssembly() or fed into the asm-side
 * Backend optimization pipeline.
 *
 * @param func   Function to lower (mutated in place).
 * @param config GemmTileConfig used by the passes. @c config.arch must be set;
 *               tile / wave fields can be left zero for trivial bring-up cases.
 */
STINKYTOFU_EXPORT void runLogicalLoweringPipeline(Function& func, const GemmTileConfig& config);

/**
 * @brief One-shot helper: build a StinkyAsmModule from a Python-side
 *        PyLogicalModule by running the standard logical-IR lowering pipeline.
 *
 * This is the "left-path" sibling of
 * @c stinkytofu::toStinkyTofuModule() (which converts the right path,
 * rocisa::Module → StinkyAsmModule). It is the recommended entry point for
 * KernelWriter-style code that builds logical IR in Python and just wants a
 * StinkyAsmModule it can call @c emitAssembly() on.
 *
 * Steps performed internally:
 *  1. Construct a fresh StinkyAsmModule (which already owns an "entry" block).
 *  2. Append the externally-owned LogicalInstruction* nodes from
 *     @p module into that entry block.
 *  3. Run @c runLogicalLoweringPipeline() on the underlying Function.
 *  4. Detach any LogicalInstruction nodes whose lifetime is owned by Python
 *     (so the C++ IRList does not delete them when the StinkyAsmModule dies).
 *
 * The caller must keep @p module alive for as long as the returned
 * StinkyAsmModule is in use, mirroring the lifetime contract of
 * @c PyLogicalFunction.
 *
 * @param module        Source PyLogicalModule (Python-side high-level IR).
 * @param arch          Target GPU architecture [major, minor, stepping].
 * @param moduleOptions Module-level options propagated to the asm module and
 *                      used to populate the GemmTileConfig for the lowering
 *                      passes. Defaults to a zero-initialized options struct
 *                      suitable for trivial vertical-slice tests.
 * @return shared_ptr to the produced StinkyAsmModule.
 */
STINKYTOFU_EXPORT std::shared_ptr<StinkyAsmModule> lowerLogicalModuleToAsm(
    PyLogicalModule& module, std::array<int, 3> arch,
    const StinkyAsmModule::ModuleOptions& moduleOptions = {});

}  // namespace stinkytofu
