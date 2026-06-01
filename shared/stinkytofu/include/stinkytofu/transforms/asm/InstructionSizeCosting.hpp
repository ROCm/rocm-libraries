/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Instruction byte-size helpers shared by AccumulateInstructionSizePass and
 * unit tests.
 * ************************************************************************ */
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {

/// Hardware encoding size in bytes from **`HwInstDesc::encoding`** (bits),
/// filled by tablegen from resolved **`Gfx1250Formats.def`** **`.encoding`**
/// (including **`.parent`** merge, e.g. TENSOR → VIMAGE → 96 b).  If
/// **`encoding == 0`**, returns **4** (defensive default).
inline int hardwareEncodingBytes(const StinkyInstruction& inst) {
    const HwInstDesc* d = inst.getHwInstDesc();
    if (d != nullptr && d->encoding > 0) return static_cast<int>(d->encoding) / 8;
    return 4;
}

/// Base size after VALU 4B→8B promotion rules (see
/// `InstructionSizeCosting.cpp`).
STINKYTOFU_EXPORT int getEffectiveBaseSizeInBytes(const StinkyInstruction& inst);

/// Additional bytes for literal-pool words (see `InstructionSizeCosting.cpp`).
STINKYTOFU_EXPORT int getLiteralExtraBytes(
    const StinkyInstruction& inst,
    const std::unordered_map<std::string, int64_t>* labelByteOffset = nullptr,
    int64_t currentByteOffsetBeforeInst = 0,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols = nullptr);

/// `getEffectiveBaseSizeInBytes` + `getLiteralExtraBytes` (total encoding cost
/// used by the pass).
STINKYTOFU_EXPORT int totalInstructionEncodingBytes(
    const StinkyInstruction& inst,
    const std::unordered_map<std::string, int64_t>* labelByteOffset = nullptr,
    int64_t currentByteOffsetBeforeInst = 0,
    const std::unordered_map<std::string, int64_t>* asmSetSymbols = nullptr);

/// Bytes the assembler inserts before the next code byte so the PC is a
/// multiple of \p alignmentBytes (e.g. `.align 16` / `LabelData::alignment`).
/// Typically encoded as `s_nop` (4 B each) on GCN. Returns 0 if alignment is 1
/// or less.
STINKYTOFU_EXPORT int64_t paddingBytesForCodeAlignment(int64_t offsetBytes, int64_t alignmentBytes);

/// If \p node is an `AsmDirective` with `ALIGN`, add padding bytes to \p
/// totalBytes (see `paddingBytesForCodeAlignment`). No-op for other node types.
STINKYTOFU_EXPORT void addAlignmentPaddingFromDirectiveNode(IRBase* node, int64_t baseByteOffset,
                                                            int64_t& totalBytes,
                                                            std::ostream* debugOut = nullptr);

/// If \p inst is a `LABEL` with `LabelData::alignment` above 1, add padding
/// before the label address. No-op for non-LABEL.
STINKYTOFU_EXPORT void addAlignmentPaddingForLabelInstruction(const StinkyInstruction& inst,
                                                              int64_t baseByteOffset,
                                                              int64_t& totalBytes,
                                                              std::ostream* debugOut = nullptr);

}  // namespace stinkytofu
