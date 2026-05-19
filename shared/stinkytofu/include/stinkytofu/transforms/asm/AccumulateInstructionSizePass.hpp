/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

#include "stinkytofu/core/IRBase.hpp"

namespace stinkytofu {
class Pass;
class StinkyAsmModule;
class BasicBlock;

/// \brief Accumulate instruction **cycle cost** and **encoded byte size** for a
/// single basic block.
///
/// **Walk / iteration.**  Same style as
/// **`scheduleFirstLocalReadWithLatency`**: linear
/// **`bb.begin()`** … **`bb.end()`** as **`IRList::iterator`**, using
/// **`getStinkyInst`** on
/// **`StinkyTofu`** nodes.  For each node,
/// **`addAlignmentPaddingFromDirectiveNode`** may advance
/// **`totalBytes`** for **`.align`**-style directives; bare
/// **`StinkyAsmDirective`** nodes are then skipped.  Non-**`StinkyTofu`** nodes
/// are skipped.
///
/// **PHI / LABEL.**
/// - **`GFX::PHI`**: skipped (no cycle or byte contribution).
/// - **`GFX::LABEL`**: if **`LabelData`** is present,
/// **`addAlignmentPaddingForLabelInstruction`**
///   runs first, then **`outLabelByteOffset[label] = baseByteOffset +
///   totalBytes`** — the global byte address of the **next** instruction in
///   program order (the label pseudo-op itself has **no** instruction encoding
///   size).  Labels without **`LabelData`** are only noted when **`debugOut`**
///   is set.
///
/// **Counted instructions.**  For every other Stinky instruction: add
/// **MFMA/WMMA** **`latencyCycles`** or, otherwise, **`issueCycles`** to the
/// running cycle total.  **Bytes:** **`instBytes =
/// getEffectiveBaseSizeInBytes(inst) + getLiteralExtraBytes(inst,
/// &outLabelByteOffset, pcBefore, asmSetSymbols)`** with **`pcBefore =
/// baseByteOffset + totalBytes`** (this instruction’s global start, used for
/// PC-relative literal sizing).  Then **`totalBytes += instBytes`**.
///
/// **Callers.**  **`AccumulateInstructionSizePass`** (kernel-wide totals and
/// optional debug dump) and
/// **`SwPrefetchInsertionPass`** (post-insertion walk so
/// **`m_labelByteOffset`** / byte totals match IR after **`s_mov_b32` +
/// `s_prefetch_inst_pc_rel`**) share this one implementation.
///
/// **SW prefetch / thresholds.**  **`pcBefore`** and **`outLabelByteOffset`**
/// line up with SW prefetch
/// **`labelOff`** usage and tail-flush semantics (**P(k)** vs block end); see
/// prefetch pass docs and
/// **`shared/stinkytofu/docs/StinkyTofu-Prefetch-Passes-Report.md`** for **`L
/// == P(k)`** and placement rules.
///
/// \param bb                    Basic block to walk (read-only aside from
/// **`debugOut`** side effects). \param outLabelByteOffset    **In/out:** label
/// name → global byte offset after label alignment;
///                              fed into **`getLiteralExtraBytes`** for
///                              **`label*`** operands (may already hold
///                              pseudo-keys such as
///                              **`label_SWprefetch_<k>`**).
/// \param debugOut              If non-null, prints per-instruction **`[cost=…
/// cycles, size=…]`** lines
///                              and **`inst.dump`** output.
/// \param outCount              If non-null, set to the number of instructions
/// that contributed cycles
///                              (excludes PHI, LABEL, directives).
/// \param outTotalBytes         If non-null, set to **sum of `instBytes`** for
/// this block (local sum,
///                              not including **`baseByteOffset`**).
/// \param baseByteOffset        Global byte offset of the **first** byte of
/// this block in the linked
///                              function image (e.g. pass
///                              **`m_byteOffsetBase`**).
/// \param asmSetSymbols         Optional **`.set` / asm-set** map for
/// **`getLiteralExtraBytes`**; may
///                              be **nullptr**.
///
/// \return Total **cycles** accumulated for this block (MFMA/WMMA latency, else
/// issue cycles).
int64_t accumulateInstructionSize(
    BasicBlock& bb, std::unordered_map<std::string, int64_t>& outLabelByteOffset,
    std::ostream* debugOut = nullptr, int* outCount = nullptr, int64_t* outTotalBytes = nullptr,
    int64_t baseByteOffset = 0,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols = nullptr);

/// Creates a pass that accumulates instruction size (cycles) over all
/// instructions in the IR.
///
/// Iterates instructions the same way as scheduleFirstLocalReadWithLatency
/// (IRList::iterator with getStinkyInst). For each instruction:
/// - MFMA/WMMA: adds latencyCycles to the total
/// - Other instructions: adds issueCycles to the total
///
/// Usage:
/// ```cpp
/// PassManager pm;
/// pm.addPass(createAccumulateInstructionSizePass(module));  // typical: wires
/// module + optional debug file pm.run();
/// ```
///
/// For debugging: cast the pass to AccumulateInstructionSizePass and call
/// setDebug(true) before run() to print each instruction and its cost (to
/// stderr by default). Call setDebugOutputPath(path) before run() to dump
/// instruction cost into a file instead. By default the pass processes all
/// basic blocks (whole program); use setProcessAllBlocks(false) to respect the
/// pipeline's basic block filter.
///
/// Enables debug and dumps pass output (per-instruction lines + summary) to the
/// given file path when \p debugOutputPath is non-empty.
std::unique_ptr<Pass> createAccumulateInstructionSizePass(const std::string& debugOutputPath);

/// Binds the pass to \p module: after the pipeline runs, sets
/// StinkyAsmModule::setTotalInstructionBytes from the pass total (whole
/// function, all basic blocks). When \p module has a non-empty output directory
/// (see StinkyAsmModule::setOutputDir), also enables debug and writes pass
/// output to
/// `<outputDir>/<kernel_basename>/accumulate_instruction_size_pass_debug.txt`.
std::unique_ptr<Pass> createAccumulateInstructionSizePass(StinkyAsmModule& module);

}  // namespace stinkytofu
