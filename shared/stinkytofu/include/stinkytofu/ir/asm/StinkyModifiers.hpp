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
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
// Enum for selecting high or low 16 bits in True16 instructions
enum class HighBitSel : int { NONE = -1, LOW = 0, HIGH = 1 };

// Matrix data format for MFMA/WMMA F8F6F4 instructions.
enum class MatrixFmt : uint8_t {
    FP8 = 0,
    BF8 = 1,
    FP6 = 2,
    BF6 = 3,
    FP4 = 4,
    NONE = 0xFF,
};

// Scale format for MXMFMA / scaled-WMMA instructions.
enum class MatrixScaleFmt : uint8_t {
    E8 = 0,
    E5M3 = 1,
    E4M3 = 2,
    NONE = 0xFF,
};

// Convert MatrixFmt enum to assembly string (e.g. FP8 -> "MATRIX_FMT_FP8").
STINKYTOFU_EXPORT std::string matrixFmtToStr(MatrixFmt fmt);

// Parse assembly string to MatrixFmt enum (e.g. "MATRIX_FMT_FP8" -> FP8).
STINKYTOFU_EXPORT MatrixFmt parseMatrixFmt(std::string_view s);

// Convert MatrixScaleFmt enum to assembly string.
STINKYTOFU_EXPORT std::string matrixScaleFmtToStr(MatrixScaleFmt fmt);

// Parse assembly string to MatrixScaleFmt enum.
STINKYTOFU_EXPORT MatrixScaleFmt parseMatrixScaleFmt(std::string_view s);

enum class MUBUFScope : uint8_t {
    SCOPE_NONE = 0,
    SCOPE_CU = 1,
    SCOPE_SE = 2,
    SCOPE_DEV = 3,
    SCOPE_SYS = 4
};

inline std::string_view toString(MUBUFScope scope) {
    switch (scope) {
        case MUBUFScope::SCOPE_CU:
            return "SCOPE_CU";
        case MUBUFScope::SCOPE_SE:
            return "SCOPE_SE";
        case MUBUFScope::SCOPE_DEV:
            return "SCOPE_DEV";
        case MUBUFScope::SCOPE_SYS:
            return "SCOPE_SYS";
        default:
            return "";
    }
}

inline MUBUFScope parseMUBUFScope(std::string_view scope) {
    if (scope == "SCOPE_CU") return MUBUFScope::SCOPE_CU;
    if (scope == "SCOPE_SE") return MUBUFScope::SCOPE_SE;
    if (scope == "SCOPE_DEV") return MUBUFScope::SCOPE_DEV;
    if (scope == "SCOPE_SYS") return MUBUFScope::SCOPE_SYS;
    return MUBUFScope::SCOPE_NONE;
}

// 9-bit DPP permutation control selector (matches the hardware dpp_ctrl field).
// Three encoding shapes:
//   singleton     — the named value IS the encoding   (e.g. ROW_MIRROR = 0x140)
//   parameterized — base + amount                     (e.g. row_shl:N = ROW_SHL0 + N;
//                                                      see dppRowShl() / dppRowShr() / ...)
//   bit-packed    — quad_perm:[p0..p3] packs 4x2 bits (see dppQuadPerm())
// *_FIRST / *_LAST constants mark each parameterized range for validity checks.
enum class DppCtrl : uint16_t {
    // clang-format off

    // --- quad_perm:[p0,p1,p2,p3] ---
    // Encoding: 8-bit value = p0 | (p1<<2) | (p2<<4) | (p3<<6)
    QUAD_PERM_FIRST    = 0x000,
    QUAD_PERM_ID       = 0x0E4,  // identity [0,1,2,3]
    QUAD_PERM_LAST     = 0x0FF,

    // --- row_shl:[1..15] --- shifts lanes left within each 16-lane row
    ROW_SHL0           = 0x100,  // base; row_shl:0 itself is no-op (reserved)
    ROW_SHL_FIRST      = 0x101,
    ROW_SHL_LAST       = 0x10F,

    // --- row_shr:[1..15] --- shifts lanes right within each 16-lane row
    ROW_SHR0           = 0x110,  // base; row_shr:0 itself is no-op (reserved)
    ROW_SHR_FIRST      = 0x111,
    ROW_SHR_LAST       = 0x11F,

    // --- row_ror:[1..15] --- rotates lanes right within each 16-lane row
    ROW_ROR0           = 0x120,  // base; row_ror:0 itself is no-op (reserved)
    ROW_ROR_FIRST      = 0x121,
    ROW_ROR_LAST       = 0x12F,

    // --- wave_shl:1, wave_rol:1, wave_shr:1, wave_ror:1 ---
    // GFX8 (VI) and GFX9 only.
    // Slots 0x131-0x133, 0x135-0x137, 0x139-0x13B, 0x13D-0x13F are reserved.
    WAVE_SHL1          = 0x130,
    WAVE_ROL1          = 0x134,
    WAVE_SHR1          = 0x138,
    WAVE_ROR1          = 0x13C,

    // --- row_mirror --- mirrors lanes within each row.
    // Deprecated; equivalent to row_xmask:15.
    ROW_MIRROR         = 0x140,

    // --- row_half_mirror --- mirrors lanes within each 8-lane half-row.
    // Deprecated; equivalent to row_xmask:7.
    ROW_HALF_MIRROR    = 0x141,

    // --- row_bcast:15, row_bcast:31 ---
    // GFX8 (VI) and GFX9 only.
    // Slots 0x144-0x14F are reserved.
    BCAST15            = 0x142,  // broadcast lane 15 to row
    BCAST31            = 0x143,  // broadcast lane 31 to row

    // --- row_newbcast:[0..15] (GFX90A) / row_share:[0..15] (GFX10+) ---
    // Same encoding, different mnemonic per arch. Selects one lane within each
    // row and shares its value with all lanes in the row.
    ROW_NEWBCAST_FIRST = 0x150,  // GFX90A name for ROW_SHARE
    ROW_NEWBCAST_LAST  = 0x15F,
    ROW_SHARE0         = 0x150,
    ROW_SHARE_FIRST    = 0x150,
    ROW_SHARE_LAST     = 0x15F,

    // --- row_xmask:[0..15] (GFX10+) ---
    // Lane reads from lane (self XOR mask) within each row.
    ROW_XMASK0         = 0x160,
    ROW_XMASK_FIRST    = 0x160,
    ROW_XMASK_LAST     = 0x16F,

    DPP_LAST           = ROW_XMASK_LAST,
    NONE               = 0xFFFF,
    // clang-format on
};

// Classify a DppCtrl value into a human-readable assembly string.
// E.g. DppCtrl(0x113) -> "row_shr:3", DppCtrl(0x140) -> "row_mirror".
STINKYTOFU_EXPORT std::string dppCtrlToAsmStr(DppCtrl ctrl);

// Parse an assembly DPP control token like "row_shr:3" or "quad_perm:[0,1,2,3]".
STINKYTOFU_EXPORT DppCtrl parseDppCtrlFromAsm(std::string_view s);

struct Modifier {
    enum class Type : uint8_t {
        DS,
        FLAT,
        GLOBAL,
        MUBUF,
        CACHE_SCOPE,
        SMEM,
        SDWA,
        DPP,
        VOP3,
        VOP3P,
        TRUE16,
        EXEC,
        VCC,
        SWAITCNT_DATA,
        SWAITTENSORCNT_DATA,
        SWAITSTORECNT_DATA,
        SDELAYALU_DATA,
        SWAITALU_DATA,
        LABEL_NAME,
        MFMA_DATA,
        COMMENT,
        MATRIX_FMT,
        MEM_TOKEN,
    };

    Modifier(Type type) : type(type) {}

    virtual ~Modifier() = default;

    /// Deep copy; each TypedModifier<Derived> returns std::make_unique<Derived>(*this).
    virtual std::unique_ptr<Modifier> clone() const = 0;

    const Type& getType() const {
        return type;
    }

    /// For isa<> / dyn_cast<> (see stinkytofu/support/Casting.hpp).
    static bool classof(const Modifier* m) {
        return m != nullptr;
    }

   private:
    Type type;
};

/// CRTP base: provides classof for isa<>/dyn_cast<> without repeating it in each modifier type.
/// Derived must define: static constexpr Modifier::Type Type = Modifier::Type::...;
template <typename Derived>
struct TypedModifier : public Modifier {
    static bool classof(const Modifier* m) {
        return m && m->getType() == Derived::Type;
    }

    std::unique_ptr<Modifier> clone() const override {
        return std::make_unique<Derived>(*static_cast<const Derived*>(this));
    }

   protected:
    explicit TypedModifier()  // NOLINT(bugprone-crtp-constructor-accessibility)
        : Modifier(Derived::Type) {}
};

struct DSModifiers : public TypedModifier<DSModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::DS;

    DSModifiers(int na = 1, int offset = 0, int offset0 = 0, int offset1 = 0, bool gds = false)
        : TypedModifier<DSModifiers>(),
          na(na),
          offset(offset),
          offset0(offset0),
          offset1(offset1),
          gds(gds) {}

    int na;
    int offset;
    int offset0;
    int offset1;
    bool gds;
};

struct FLATModifiers : public TypedModifier<FLATModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::FLAT;

    FLATModifiers(int offset12 = 0, bool glc = false, bool slc = false, bool lds = false,
                  bool isStore = false, bool hasGLCModifier = false, bool hasSC0Modifier = false)
        : TypedModifier<FLATModifiers>(),
          offset12(offset12),
          glc(glc),
          slc(slc),
          lds(lds),
          isStore(isStore),
          hasGLCModifier(hasGLCModifier),
          hasSC0Modifier(hasSC0Modifier) {}

    int offset12;
    uint32_t glc : 1;
    uint32_t slc : 1;
    uint32_t lds : 1;
    uint32_t isStore : 1;
    uint32_t hasGLCModifier : 1;
    uint32_t hasSC0Modifier : 1;
};

struct GLOBALModifiers : public TypedModifier<GLOBALModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::GLOBAL;

    GLOBALModifiers(int offset = 0) : TypedModifier<GLOBALModifiers>(), offset(offset) {}

    int offset;
};

struct MUBUFModifiers : public TypedModifier<MUBUFModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::MUBUF;

    MUBUFModifiers(bool offen = false, int offset12 = 0, bool glc = false, bool slc = false,
                   bool nt = false, bool lds = false, bool isStore = false,
                   bool hasMUBUFConst = false, bool hasGLCModifier = false,
                   bool hasSC0Modifier = false, MUBUFScope scope = MUBUFScope::SCOPE_NONE)
        : TypedModifier<MUBUFModifiers>(),
          offset12(offset12),
          offen(offen),
          glc(glc),
          slc(slc),
          nt(nt),
          lds(lds),
          isStore(isStore),
          hasMUBUFConst(hasMUBUFConst),
          hasGLCModifier(hasGLCModifier),
          hasSC0Modifier(hasSC0Modifier),
          scope(scope) {}

    int offset12;
    uint32_t offen : 1;
    uint32_t glc : 1;
    uint32_t slc : 1;
    uint32_t nt : 1;
    uint32_t lds : 1;
    uint32_t isStore : 1;
    uint32_t hasMUBUFConst : 1;
    uint32_t hasGLCModifier : 1;
    uint32_t hasSC0Modifier : 1;
    MUBUFScope scope;
};

// Carries just the cache scope token for SOPP-format memory fences such as
// global_wb / global_inv on gfx1250+. These instructions are not buffer ops
// and do not need offen/glc/slc/lds/etc., so they cannot reuse MUBUFModifiers
// without coupling to fields that may diverge in future MUBUF refactors.
struct CacheScopeModifiers : public TypedModifier<CacheScopeModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::CACHE_SCOPE;

    CacheScopeModifiers(MUBUFScope scope = MUBUFScope::SCOPE_NONE)
        : TypedModifier<CacheScopeModifiers>(), scope(scope) {}

    MUBUFScope scope;
};

struct SMEMModifiers : public TypedModifier<SMEMModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::SMEM;

    SMEMModifiers(bool glc = false, bool nv = false, int offset = 0, bool hasSCOPEModifier = false)
        : TypedModifier<SMEMModifiers>(),
          glc(glc),
          nv(nv),
          offset(offset)  // 20u 21s shaes the same
          ,
          hasSCOPEModifier(hasSCOPEModifier) {}

    uint32_t glc : 1;
    uint32_t nv : 1;
    int offset;
    uint32_t hasSCOPEModifier : 1;
};

struct SDWAModifiers : public TypedModifier<SDWAModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::SDWA;

    enum class SelectBit : uint8_t {
        SEL_NONE = 0,
        DWORD = 1,
        BYTE_0 = 2,
        BYTE_1 = 3,
        BYTE_2 = 4,
        BYTE_3 = 5,
        WORD_0 = 6,
        WORD_1 = 7
    };

    enum class UnusedBit : uint8_t {
        UNUSED_NONE = 0,
        UNUSED_PAD = 1,
        UNUSED_SEXT = 2,
        UNUSED_PRESERVE = 3
    };

    SDWAModifiers(SelectBit dst_sel = SelectBit::SEL_NONE,
                  UnusedBit dst_unused = UnusedBit::UNUSED_NONE,
                  SelectBit src0_sel = SelectBit::SEL_NONE,
                  SelectBit src1_sel = SelectBit::SEL_NONE)
        : TypedModifier<SDWAModifiers>(),
          dst_sel(dst_sel),
          dst_unused(dst_unused),
          src0_sel(src0_sel),
          src1_sel(src1_sel) {}

    SelectBit dst_sel;
    UnusedBit dst_unused;
    SelectBit src0_sel;
    SelectBit src1_sel;
};

// DPP (Data Parallel Primitives) modifier: cross-lane permutation for VOP ALU instructions.
struct DPPModifiers : public TypedModifier<DPPModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::DPP;

    DppCtrl dppCtrl = DppCtrl::NONE;  // 9-bit permutation selector (DPP16)
    uint8_t rowMask = 0xF;            // 4-bit row participation mask
    uint8_t bankMask = 0xF;           // 4-bit bank participation mask
    uint8_t boundCtrl = 0;            // 0 = keep old value, 1 = zero
    uint8_t fi = 0;                   // fetch-inactive lane data

    // DPP8 permutation (8 x 3-bit lane selectors). Empty means DPP16 mode.
    std::array<uint8_t, 8> dpp8 = {0, 0, 0, 0, 0, 0, 0, 0};
    bool isDPP8 = false;  // true = DPP8 mode (use dpp8 array), false = DPP16 (use dppCtrl)

    // Default constructor (empty / no-DPP sentinel).
    DPPModifiers() : TypedModifier<DPPModifiers>() {}

    // DPP16 constructor.
    DPPModifiers(DppCtrl ctrl, uint8_t rowMask = 0xF, uint8_t bankMask = 0xF, uint8_t boundCtrl = 0,
                 uint8_t fi = 0)
        : TypedModifier<DPPModifiers>(),
          dppCtrl(ctrl),
          rowMask(rowMask),
          bankMask(bankMask),
          boundCtrl(boundCtrl),
          fi(fi) {}

    // DPP8 constructor.
    DPPModifiers(const std::array<uint8_t, 8>& dpp8Perm, uint8_t fi = 0)
        : TypedModifier<DPPModifiers>(), fi(fi), dpp8(dpp8Perm), isDPP8(true) {}

    bool empty() const {
        return !isDPP8 && dppCtrl == DppCtrl::NONE;
    }
};

// Build a DPP16 DppCtrl for row_shr:N (N in 1..15).
inline DppCtrl dppRowShr(unsigned n) {
    assert(n >= 1 && n <= 15 && "row_shr amount must be in [1,15]");
    return static_cast<DppCtrl>(static_cast<uint16_t>(DppCtrl::ROW_SHR0) + n);
}
// Build a DPP16 DppCtrl for row_shl:N (N in 1..15).
inline DppCtrl dppRowShl(unsigned n) {
    assert(n >= 1 && n <= 15 && "row_shl amount must be in [1,15]");
    return static_cast<DppCtrl>(static_cast<uint16_t>(DppCtrl::ROW_SHL0) + n);
}
// Build a DPP16 DppCtrl for row_ror:N (N in 1..15).
inline DppCtrl dppRowRor(unsigned n) {
    assert(n >= 1 && n <= 15 && "row_ror amount must be in [1,15]");
    return static_cast<DppCtrl>(static_cast<uint16_t>(DppCtrl::ROW_ROR0) + n);
}
// Encode a quad_perm:[a,b,c,d] into a DppCtrl value (each lane in 0..3).
inline DppCtrl dppQuadPerm(uint8_t p0, uint8_t p1, uint8_t p2, uint8_t p3) {
    assert(p0 <= 3 && p1 <= 3 && p2 <= 3 && p3 <= 3 && "quad_perm lane must be in [0,3]");
    return static_cast<DppCtrl>((p0 & 3) | ((p1 & 3) << 2) | ((p2 & 3) << 4) | ((p3 & 3) << 6));
}
// Build a DPP16 DppCtrl for row_share:N (N in 0..15).
inline DppCtrl dppRowShare(unsigned n) {
    assert(n <= 15 && "row_share lane must be in [0,15]");
    return static_cast<DppCtrl>(static_cast<uint16_t>(DppCtrl::ROW_SHARE_FIRST) + n);
}
// Build a DPP16 DppCtrl for row_xmask:N (N in 0..15).
inline DppCtrl dppRowXmask(unsigned n) {
    assert(n <= 15 && "row_xmask must be in [0,15]");
    return static_cast<DppCtrl>(static_cast<uint16_t>(DppCtrl::ROW_XMASK_FIRST) + n);
}

struct VOP3Modifiers : public TypedModifier<VOP3Modifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::VOP3;

    VOP3Modifiers(bool neg_src0 = false, bool neg_src1 = false, bool neg_src2 = false,
                  bool abs_src0 = false, bool abs_src1 = false, bool abs_src2 = false,
                  bool clamp = false, int omod = 0)
        : TypedModifier<VOP3Modifiers>(),
          neg_src0(neg_src0),
          neg_src1(neg_src1),
          neg_src2(neg_src2),
          abs_src0(abs_src0),
          abs_src1(abs_src1),
          abs_src2(abs_src2),
          clamp(clamp),
          omod(omod) {}

    bool neg_src0;  // Negate source operand 0
    bool neg_src1;  // Negate source operand 1
    bool neg_src2;  // Negate source operand 2
    bool abs_src0;  // Absolute value of source operand 0
    bool abs_src1;  // Absolute value of source operand 1
    bool abs_src2;  // Absolute value of source operand 2
    bool clamp;     // Clamp result to [0.0, 1.0]
    int omod;       // Output modifier: 0=*1, 1=*2, 2=*4, 3=*0.5
};

struct VOP3PModifiers : public TypedModifier<VOP3PModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::VOP3P;

    VOP3PModifiers(const std::vector<int>& op_sel = {}, const std::vector<int>& op_sel_hi = {},
                   const std::vector<int>& byte_sel = {})
        : TypedModifier<VOP3PModifiers>(),
          op_sel(op_sel),
          op_sel_hi(op_sel_hi),
          byte_sel(byte_sel) {}

    std::vector<int> op_sel;
    std::vector<int> op_sel_hi;
    std::vector<int> byte_sel;
};

struct True16Modifiers : public TypedModifier<True16Modifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::TRUE16;

    // Default constructor with explicit destination and source modifiers
    True16Modifiers(HighBitSel dst0 = HighBitSel::NONE, HighBitSel dst1 = HighBitSel::NONE,
                    const std::vector<HighBitSel>& srcs = {})
        : TypedModifier<True16Modifiers>(), encoded_(0), srcCount_(0) {
        setDst0(dst0);
        setDst1(dst1);
        setSrcs(srcs);
    }

    // Getters
    HighBitSel getDst0() const {
        return decode(0);
    }
    HighBitSel getDst1() const {
        return decode(1);
    }
    HighBitSel getSrc(size_t index) const {
        return (index < srcCount_) ? decode(2 + index) : HighBitSel::NONE;
    }
    size_t getSrcCount() const {
        return srcCount_;
    }

    // Setters
    void setDst0(HighBitSel val) {
        encode(0, val);
    }
    void setDst1(HighBitSel val) {
        encode(1, val);
    }
    void setSrcs(const std::vector<HighBitSel>& srcs) {
        srcCount_ = srcs.size();
        for (size_t i = 0; i < srcs.size(); ++i) {
            encode(2 + i, srcs[i]);
        }
    }

   private:
    // Encoding: 2 bits per operand
    // Bits [0-1]:   dst0
    // Bits [2-3]:   dst1
    // Bits [4-5]:   src0
    // Bits [6-7]:   src1
    // Bits [8-9]:   src2
    // Bits [10-11]: src3
    // Bits [12-13]: src4
    // Bits [14-15]: src5
    // Maximum 2 dsts + 6 srcs = 8 operands * 2 bits = 16 bits (fits in uint16_t)
    uint16_t encoded_;
    uint8_t srcCount_;  // Number of source operands (0-6)

    // Encode HighBitSel to 2-bit value
    static uint8_t encodeValue(HighBitSel val) {
        switch (val) {
            case HighBitSel::NONE:
                return 0;
            case HighBitSel::LOW:
                return 1;
            case HighBitSel::HIGH:
                return 2;
            default:
                return 0;
        }
    }

    // Decode 2-bit value to HighBitSel
    static HighBitSel decodeValue(uint8_t val) {
        switch (val & 0x3) {
            case 0:
                return HighBitSel::NONE;
            case 1:
                return HighBitSel::LOW;
            case 2:
                return HighBitSel::HIGH;
            default:
                return HighBitSel::NONE;
        }
    }

    // Encode value at operand index (0=dst0, 1=dst1, 2=src0, ...)
    void encode(size_t index, HighBitSel val) {
        uint16_t bits = encodeValue(val);
        size_t bitOffset = index * 2;
        encoded_ &= ~(uint16_t(0x3) << bitOffset);  // Clear existing bits
        encoded_ |= (bits << bitOffset);            // Set new bits
    }

    // Decode value at operand index
    HighBitSel decode(size_t index) const {
        size_t bitOffset = index * 2;
        uint16_t bits = (encoded_ >> bitOffset) & 0x3;
        return decodeValue(static_cast<uint8_t>(bits));
    }
};

struct EXEC : public TypedModifier<EXEC> {
    static constexpr Modifier::Type Type = Modifier::Type::EXEC;

    EXEC(bool setHi = false) : TypedModifier<EXEC>(), setHi(setHi) {}

    bool setHi;
};

struct VCC : public TypedModifier<VCC> {
    static constexpr Modifier::Type Type = Modifier::Type::VCC;

    VCC(bool setHi = false) : TypedModifier<VCC>(), setHi(setHi) {}

    bool setHi;
};

struct SWaitCntData : public TypedModifier<SWaitCntData> {
    static constexpr Modifier::Type Type = Modifier::Type::SWAITCNT_DATA;

    SWaitCntData(int vlcnt = -1, int vscnt = -1, int dlcnt = -1, int dscnt = -1, int kmcnt = -1,
                 int maxLgkmcnt = -1, int maxVmcnt = -1)
        : TypedModifier<SWaitCntData>(),
          vlcnt(vlcnt),
          vscnt(vscnt),
          dlcnt(dlcnt),
          dscnt(dscnt),
          kmcnt(kmcnt),
          maxLgkmcnt(maxLgkmcnt),
          maxVmcnt(maxVmcnt) {}

    // vlcnt: Number of VMEM load (and atomic with return value) instructions issued but not yet
    // completed. vscnt: Number of VMEM store (and atomic w/o return value) instructions issued but
    // not yet completed. dlcnt: Number of LDS load instructions issued but not yet completed.
    // dscnt: Number of LDS store instructions issued but not yet completed.
    // kmcnt: Number of constant-fetch (scalar memory read), and message instructions issued but not
    // yet completed.
    //
    // In some ISA, VMEM load/store share the same counter(vmcnt). LDS, scalar memory read and
    // message share the same counter(lgkmcnt). These counters are combined from the 4 counters
    // above as:
    //     vmcnt   = vlcnt + vscnt
    //     lgkmcnt = dlcnt + dscnt + kmcnt
    //
    // Example: 4 VMEM instructions vl_0, vl_1, vs_0, vl_2 are issued and we want to wait for vl_1
    // to complete.
    //
    //     If the target ISA has separate counters for load and store, use SWaitCnt(vlcnt=1),
    //     which means vl_2 is not completed yet.
    //
    //     If the target ISA has a single counter for load and store, use SWaitCnt(vlcnt=1,
    //     vscnt=1), which means vs_0, vl_2 are not completed yet.
    int vlcnt;
    int vscnt;
    int dlcnt;
    int dscnt;
    int kmcnt;
    int maxLgkmcnt;
    int maxVmcnt;
};

struct LabelData : public TypedModifier<LabelData> {
    static constexpr Modifier::Type Type = Modifier::Type::LABEL_NAME;

    LabelData(const std::string& label, uint16_t alignment = 1)
        : TypedModifier<LabelData>(), label(label), alignment(alignment) {}
    std::string label;
    uint16_t alignment;
};

struct SWaitTensorCntData : public TypedModifier<SWaitTensorCntData> {
    static constexpr Modifier::Type Type = Modifier::Type::SWAITTENSORCNT_DATA;

    SWaitTensorCntData(int8_t tlcnt = -1) : TypedModifier<SWaitTensorCntData>(), tlcnt(tlcnt) {}

    int8_t tlcnt;
};

struct SWaitStoreCntData : public TypedModifier<SWaitStoreCntData> {
    static constexpr Modifier::Type Type = Modifier::Type::SWAITSTORECNT_DATA;

    SWaitStoreCntData(int8_t storecnt = -1)
        : TypedModifier<SWaitStoreCntData>(), storecnt(storecnt) {}

    int8_t storecnt;
};

/// Delay ALU instruction data for RDNA3/RDNA4 (gfx11xx/gfx12xx)
/// Specifies dependencies between ALU instructions
/// Encoding: SIMM16[3:0] = InstID0, SIMM16[6:4] = InstSkip, SIMM16[10:7] = InstID1
struct SDelayAluData : public TypedModifier<SDelayAluData> {
    static constexpr Modifier::Type Type = Modifier::Type::SDELAYALU_DATA;

    enum class InstType : uint8_t {
        VALU,   ///< Vector ALU instruction
        SALU,   ///< Scalar ALU instruction
        TRANS,  ///< Transcendental instruction
        NO_DEP  ///< No dependency
    };

    // Constructor for single dependency (instid0 only)
    SDelayAluData(InstType type, int8_t distance = 0)
        : TypedModifier<SDelayAluData>(),
          instid0Type(type),
          instid0Distance(distance),
          hasInstId1(false),
          instSkip(0),
          instid1Type(InstType::NO_DEP),
          instid1Distance(0) {}

    // Constructor for dual dependency (instid0 and instid1)
    SDelayAluData(InstType id0Type, int8_t id0Distance, int8_t skip, InstType id1Type,
                  int8_t id1Distance)
        : TypedModifier<SDelayAluData>(),
          instid0Type(id0Type),
          instid0Distance(id0Distance),
          hasInstId1(true),
          instSkip(skip),
          instid1Type(id1Type),
          instid1Distance(id1Distance) {}

    // Primary dependency
    InstType instid0Type;    ///< InstID0 type (VALU/SALU/TRANS/NO_DEP)
    int8_t instid0Distance;  ///< InstID0 dependency distance

    // Optional secondary dependency
    bool hasInstId1;         ///< Whether InstID1 is present
    int8_t instSkip;         ///< Number of instructions to skip (0-7)
    InstType instid1Type;    ///< InstID1 type (VALU/SALU/TRANS/NO_DEP)
    int8_t instid1Distance;  ///< InstID1 dependency distance

    // Legacy accessors for backward compatibility
    InstType type() const {
        return instid0Type;
    }
    int8_t distance() const {
        return instid0Distance;
    }
};

// SWaitAlu instruction data
struct SWaitAluData : public TypedModifier<SWaitAluData> {
    static constexpr Modifier::Type Type = Modifier::Type::SWAITALU_DATA;

    struct HwValue {
        uint16_t sa_sdst : 1;   // Bit 0
        uint16_t va_vcc : 1;    // Bit 1
        uint16_t vm_vsrc : 3;   // Bits 4-2
        uint16_t reserved : 2;  // Bits 6-5
        uint16_t hold_cnt : 1;  // Bit 7
        uint16_t va_ssrc : 1;   // Bit 8
        uint16_t va_sdst : 3;   // Bits 11-9
        uint16_t va_vdst : 4;   // Bits 15-12

        HwValue& operator=(HwValue const& other) = default;
    };

    // Field enumeration
    enum Field : uint8_t {
        VA_VDST = 0,
        VA_SDST = 1,
        VA_SSRC = 2,
        HOLD_CNT = 3,
        VM_VSRC = 4,
        VA_VCC = 5,
        SA_SDST = 6,
        FIELD_COUNT = 7
    };

    // Constructor from separate fields
    SWaitAluData(int va_vdst_val, int va_sdst_val, int va_ssrc_val, int hold_cnt_val,
                 int vm_vsrc_val, int va_vcc_val, int sa_sdst_val)
        : TypedModifier<SWaitAluData>() {
        auto [packed, mask] = encodeWithMask(va_vdst_val, va_sdst_val, va_ssrc_val, hold_cnt_val,
                                             vm_vsrc_val, va_vcc_val, sa_sdst_val);

        fieldsValue = packed;
        validFields = mask;
    }

    // Static utility function to encode separate fields into hw format with validity mask
    static std::pair<HwValue, uint8_t> encodeWithMask(int va_vdst, int va_sdst, int va_ssrc,
                                                      int hold_cnt, int vm_vsrc, int va_vcc,
                                                      int sa_sdst) {
        HwValue hwVal = {};
        uint8_t mask = 0;

#define SET_FIELD(val, field, bitmask, fieldEnum) \
    if ((val) != -1) {                            \
        hwVal.field = (val) & (bitmask);          \
        mask |= (1 << (fieldEnum));               \
    }

        SET_FIELD(va_vdst, va_vdst, 0xF, VA_VDST)
        SET_FIELD(va_sdst, va_sdst, 0x7, VA_SDST)
        SET_FIELD(va_ssrc, va_ssrc, 0x1, VA_SSRC)
        SET_FIELD(hold_cnt, hold_cnt, 0x1, HOLD_CNT)
        SET_FIELD(vm_vsrc, vm_vsrc, 0x7, VM_VSRC)
        SET_FIELD(va_vcc, va_vcc, 0x1, VA_VCC)
        SET_FIELD(sa_sdst, sa_sdst, 0x1, SA_SDST)

#undef SET_FIELD

        return {hwVal, mask};
    }

    // Unified field accessor - always returns the raw value
    unsigned getField(Field field) const {
        switch (field) {
                // clang-format off
            case VA_VDST: return fieldsValue.va_vdst;
            case VA_SDST: return fieldsValue.va_sdst;
            case VA_SSRC: return fieldsValue.va_ssrc;
            case HOLD_CNT: return fieldsValue.hold_cnt;
            case VM_VSRC: return fieldsValue.vm_vsrc;
            case VA_VCC: return fieldsValue.va_vcc;
            case SA_SDST: return fieldsValue.sa_sdst;
            default: return 0;  // clang-format on
        }
    }

    // Unified validity checker - check if field is used/set
    bool hasField(Field field) const {
        return validFields & (1 << field);
    }

   private:
    HwValue fieldsValue;
    uint8_t validFields;  // Bitmask indicating which fields are valid (not -1)
};

// Per-source neg_lo/neg_hi bits for MFMA/WMMA instructions.
// neg_lo and neg_hi are 3-bit fields (one bit per source: src0, src1, src2).
// For IU8/IU4 data types, negation bits indicate signed vs unsigned.
struct MFMANegBits {
    std::array<uint8_t, 3> negLo = {0, 0, 0};
    std::array<uint8_t, 3> negHi = {0, 0, 0};
    uint8_t numSrcs = 0;  // 0 = unset, 2 or 3

    bool hasNegLo() const {
        return numSrcs > 0 && (negLo[0] || negLo[1] || negLo[2]);
    }
    bool hasNegHi() const {
        return numSrcs > 0 && (negHi[0] || negHi[1] || negHi[2]);
    }
    bool empty() const {
        return numSrcs == 0;
    }
};

// MFMA modifiers - per-source neg_lo/neg_hi bits and matrix reuse hints.
// Matrix data/scale formats live in MatrixFmtModifiers (its own TypedModifier)
// so the cost-override table can match on it as a first-class Modifier type.
struct MFMAModifiers : public TypedModifier<MFMAModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::MFMA_DATA;

    MFMAModifiers() : TypedModifier<MFMAModifiers>() {}

    MFMANegBits negBits;  // Per-source neg_lo/neg_hi bits
    bool reuseA = false;
    bool reuseB = false;
};

// Matrix format modifier for F8F6F4-family WMMA/MFMA instructions.
// Carries the per-matrix data format (fmtA/fmtB) and, for scaled (MXMFMA)
// instructions, the per-matrix scale numeric format (scaleFmtA/scaleFmtB).
// Used both for assembly emission (matrix_a_fmt:..., matrix_a_scale_fmt:...)
// and as the match key for cost-override entries in *_Instructions.def.
struct MatrixFmtModifiers : public TypedModifier<MatrixFmtModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::MATRIX_FMT;

    MatrixFmt fmtA = MatrixFmt::NONE;
    MatrixFmt fmtB = MatrixFmt::NONE;
    MatrixScaleFmt scaleFmtA = MatrixScaleFmt::NONE;
    MatrixScaleFmt scaleFmtB = MatrixScaleFmt::NONE;

    MatrixFmtModifiers() : TypedModifier<MatrixFmtModifiers>() {}
    MatrixFmtModifiers(MatrixFmt a, MatrixFmt b)
        : TypedModifier<MatrixFmtModifiers>(), fmtA(a), fmtB(b) {}
    MatrixFmtModifiers(MatrixFmt a, MatrixFmt b, MatrixScaleFmt sa, MatrixScaleFmt sb)
        : TypedModifier<MatrixFmtModifiers>(), fmtA(a), fmtB(b), scaleFmtA(sa), scaleFmtB(sb) {}

    // True for scaled F8F6F4 / MXMFMA (scale_fmt fields set).
    bool isMXMFMA() const {
        return scaleFmtA != MatrixScaleFmt::NONE;
    }
    // True when no format info is set (instance carries nothing useful).
    bool empty() const {
        return fmtA == MatrixFmt::NONE && fmtB == MatrixFmt::NONE;
    }
};

struct CommentData : public TypedModifier<CommentData> {
    static constexpr Modifier::Type Type = Modifier::Type::COMMENT;

    CommentData(const std::string& comment) : TypedModifier<CommentData>(), comment(comment) {}

    std::string comment;
};

struct MemTokenData : public TypedModifier<MemTokenData> {
    static constexpr Modifier::Type Type = Modifier::Type::MEM_TOKEN;

    std::vector<int> tokens;

    MemTokenData(const std::vector<int>& tokens = {})
        : TypedModifier<MemTokenData>(), tokens(tokens) {}
};

}  // namespace stinkytofu
