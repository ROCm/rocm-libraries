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
#include <bitset>
#include <cassert>
#include <cstdint>

#include "stinkytofu/support/Span.hpp"

namespace stinkytofu {
enum class GfxArchID : uint32_t {
#define STINKYTOFU_ARCH(archName) archName,
#include "Config/Archs.def"
};

enum InstFlag : uint8_t {
#define MACRO(flag) flag,
#include "stinkytofu/hardware/Flags.def"

#undef MACRO

    // Total count of flags.
    // Order does matter, define count immediately after the last flag.
    IF_COUNT,

    // The beginning of the flags is always 0.
    IF_BEGIN = 0,  // = IF_MUBUFLoad
};

// Note: Try to keep the capacity <= 64.
// If gfx instruction needs more than 64 flags, there are two options:
// 1. Implement a custom bitset type that can support constexpr
//    initialization for more than 64 bits.
//
// 2. (simpler) Just add another std::bitset<32> flag.
constexpr size_t flagCapacity = 64;

// Helper function to convert flags to a bit pattern at compile time
constexpr uint64_t makeFlagBits(std::initializer_list<InstFlag> flags) {
    static_assert(flagCapacity <= 64, "flagCapacity exceeds uint64_t bit width");

    uint64_t bits = 0;
    for (auto f : flags) bits |= (1ULL << static_cast<size_t>(f));
    return bits;
}

constexpr std::bitset<flagCapacity> makeFlagSet(std::initializer_list<InstFlag> flags) {
    return std::bitset<flagCapacity>(makeFlagBits(flags));
}

using IsaOpcode = uint16_t;
using UnifiedOpcode = uint16_t;
using InstFlagSet = std::bitset<flagCapacity>;

/// Microcode format: the ISA-level encoding format for an instruction.
/// Values correspond to hardware microcode encoding formats (e.g., MC_VOP3P = 64-bit packed
/// vector).
enum class MicrocodeFormat : uint8_t {
    NONE = 0,

    // Scalar ALU and Control Formats
    MC_SOP2,
    MC_SOP1,
    MC_SOPK,
    MC_SOPP,
    MC_SOPC,

    // Scalar Memory Format
    MC_SMEM,

    // Vector ALU Formats
    MC_VOP1,
    MC_VOP2,
    MC_VOPC,
    MC_VOP3,
    MC_VOP3SD,
    MC_VOP3P,
    MC_VOP3PX2,
    MC_VOP3PX3,
    MC_VOPD,
    MC_VOPD3,
    MC_DPP16,
    MC_DPP8,

    // Data Share Format
    MC_VDS,

    // Vector Memory Buffer Format
    MC_VBUFFER,

    // Vector Memory Image Format
    MC_VIMAGE,

    // Flat, Global and Scratch Formats
    MC_VFLAT,
    MC_VGLOBAL,
    MC_VSCRATCH,

    COUNT
};

/// Execution unit that processes an instruction.
enum class ExecUnit : uint8_t {
    NONE = 0,
    VALU,
    SALU,
    BranchUnit,
    MatrixUnit,
    BufferMemory,
    LDS,
    GlobalMemory,
    ScalarMemory,
    TensorUnit,
    COUNT
};

/// Encoding field in the instruction format (hardware-level field name).
enum class EncodeField : uint8_t {
    None = 0,
    // Buffer memory
    vdata,
    vaddr,
    rsrc,
    soffset,
    // Flat / global memory
    vdst,
    vsrc,
    saddr,
    // DS (LDS)
    addr,
    data0,
    data1,
    // Scalar ALU
    sdst,
    ssrc1,
    literal,
    // Scalar memory (SMEM)
    sdata,
    sbase,
    // Scalar / control-flow
    simm16,
    simm32,
    ssrc0,
    // Vector ALU (VOP2 specific)
    vsrc1,
    // WMMA / MXWMMA
    src0,
    src1,
    src2,
    scale_src0,
    scale_src1,
    // Tensor / VIMAGE
    vaddr0,
    vaddr1,
    vaddr2,
    vaddr3,
    COUNT
};

/// Operand register type classification.
enum class FieldType : uint8_t {
    None = 0,
    // Register types
    vgpr,
    sreg,
    sreg_m0,
    // Scalar ALU types
    sdst,
    ssrc,
    hwreg,
    delay,
    set_vgpr_msb,
    sleep,
    // Scalar memory types
    smem_offset,
    // Immediate / control-flow types
    label,
    simm16,
    simm32,
    ssrc_barrier_id,
    // Vector ALU source types
    src,
    vcc,
    exec,
    sgpr,
    // WMMA / MXWMMA operand types
    src_vgpr,
    src_vgpr_or_inline,
    src_simple,
    wait_alu,
    wait_mem_ds,
    COUNT
};

// General hardware instruction description for all ISAs.
struct HwInstDesc {
    // instruction opcode in specific ISA.
    IsaOpcode isaOpcode = 0;

    // unified instruction opcode for all ISAs.
    UnifiedOpcode unifiedOpcode = 0;

    // issue cycles and latency cycles in specific ISA.
    uint16_t issue = 0;
    uint16_t latency = 0;

    // per-cycle VALU co-issue bitmask: bit i = 1 means VALU can co-issue at cycle i
    // after this instruction is issued (used for matrix instructions).
    uint16_t coIssueMask = 0;

    // mnemonic string for the instruction.
    const char* mnemonic = nullptr;

    std::bitset<flagCapacity> flags;

    // microcode format (ISA-level encoding format, e.g., MC_VOP3P, MC_SMEM)
    MicrocodeFormat microcode = MicrocodeFormat::NONE;

    // instruction encoding size in bits (e.g., 32, 64, 96, 128)
    uint16_t encoding = 0;

    // execution unit (e.g., VALU, SALU, MatrixUnit)
    ExecUnit unit = ExecUnit::NONE;

    /// Per-operand encoding field description (dest/src, encoding field,
    /// register type, and size in bits).
    struct OperandFieldDesc {
        EncodeField encodeField = EncodeField::None;
        FieldType fieldType = FieldType::None;

        // fieldSizeBits means the size of the operand in bits, for example,
        // if the source is a vreg, it field size is 32.
        // 12 bits supports up to 4095; current max is 512 (WMMA 16×VGPR).
        uint16_t fieldSizeBits : 12 = 0;
        uint16_t isDest : 1 = 0;
        uint16_t isReadWrite : 1 = 0;
        // M64: 64-bit lane mask that may be truncated to 32 bits in wave32.
        uint16_t isM64 : 1 = 0;
        uint16_t reserved : 1 = 0;
    };

    /// Operand field descriptions for this instruction (primary encoding).
    /// Empty for instructions without field metadata.
    span<const OperandFieldDesc> operandFields;

    /// Promoted (wider) encoding format, e.g., VOP2 instructions can be promoted
    /// to VOP3.  NONE when no promoted encoding exists.
    MicrocodeFormat promotedFormat = MicrocodeFormat::NONE;

    /// Operand field descriptions for the promoted encoding.
    /// Empty when no promoted encoding exists.
    span<const OperandFieldDesc> promotedFields;

    bool has(InstFlag f) const {
        return flags.test((size_t)f);
    }

    EncodeField getEncodeField(unsigned idx) const {
        return idx < operandFields.size() ? operandFields[idx].encodeField : EncodeField::None;
    }

    FieldType getFieldType(unsigned idx) const {
        return idx < operandFields.size() ? operandFields[idx].fieldType : FieldType::None;
    }

    uint16_t getFieldTypeSize(unsigned idx) const {
        return idx < operandFields.size() ? operandFields[idx].fieldSizeBits : 0;
    }
};

}  // namespace stinkytofu
