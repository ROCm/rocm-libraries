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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace stinkytofu {
// Enum for selecting high or low 16 bits in True16 instructions
enum class HighBitSel : int { NONE = -1, LOW = 0, HIGH = 1 };

enum class MatrixFmt : uint8_t { FP4 = 0, FP6 = 1, FP8 = 2 };

struct Modifier {
    enum class Type : uint8_t {
        DS,
        FLAT,
        GLOBAL,
        MUBUF,
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
    explicit TypedModifier() : Modifier(Derived::Type) {}
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
                   bool hasSC0Modifier = false)
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
          hasSC0Modifier(hasSC0Modifier) {}

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

// dot2: for WaveSplitK reduction. Only a subset of DPP modifiers are used here
struct DPPModifiers : public TypedModifier<DPPModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::DPP;

    int row_shr;
    int row_bcast;
    int bound_ctrl;

    DPPModifiers(int row_shr = -1, int row_bcast = -1, int bound_ctrl = -1)
        : TypedModifier<DPPModifiers>(),
          row_shr(row_shr),
          row_bcast(row_bcast),
          bound_ctrl(bound_ctrl) {}
};

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

        HwValue& operator=(HwValue const& other) {
            sa_sdst = other.sa_sdst;
            va_vcc = other.va_vcc;
            vm_vsrc = other.vm_vsrc;
            reserved = other.reserved;
            hold_cnt = other.hold_cnt;
            va_ssrc = other.va_ssrc;
            va_sdst = other.va_sdst;
            va_vdst = other.va_vdst;
            return *this;
        }
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

// MFMA modifiers
struct MFMAModifiers : public TypedModifier<MFMAModifiers> {
    static constexpr Modifier::Type Type = Modifier::Type::MFMA_DATA;

    MFMAModifiers(const std::string& inputPermute = "", const std::string& scaleStr = "",
                  const std::string& negStr = "", bool reuseA = false, bool reuseB = false,
                  bool neg_lo = false, bool neg_hi = false)
        : TypedModifier<MFMAModifiers>(),
          inputPermute(inputPermute),
          scaleStr(scaleStr),
          negStr(negStr),
          reuseA(reuseA),
          reuseB(reuseB),
          neg_lo(neg_lo),
          neg_hi(neg_hi),
          isMXMFMA(false),
          mxInstType(0),
          mxScaleAType(0),
          mxScaleBType(0) {}

    // Constructor for MXMFMA instructions
    MFMAModifiers(const std::string& inputPermute, const std::string& scaleStr,
                  const std::string& negStr, bool reuseA, bool reuseB, int mxInstType,
                  int mxScaleAType, int mxScaleBType, bool neg_lo = false, bool neg_hi = false)
        : TypedModifier<MFMAModifiers>(),
          inputPermute(inputPermute),
          scaleStr(scaleStr),
          negStr(negStr),
          reuseA(reuseA),
          reuseB(reuseB),
          neg_lo(neg_lo),
          neg_hi(neg_hi),
          isMXMFMA(true),
          mxInstType(mxInstType),
          mxScaleAType(mxScaleAType),
          mxScaleBType(mxScaleBType) {}

    std::string inputPermute;
    std::string scaleStr;
    std::string negStr;
    bool reuseA;
    bool reuseB;
    bool neg_lo;  // Indicates if neg_lo modifier is present
    bool neg_hi;  // Indicates if neg_hi modifier is present

    // MXMFMA-specific fields
    bool isMXMFMA;     // Flag to indicate if this is a MXMFMA modifier
    int mxInstType;    // MXMFMA instruction type (rocisa::InstType)
    int mxScaleAType;  // Scale type for matrix A (rocisa::InstType)
    int mxScaleBType;  // Scale type for matrix B (rocisa::InstType)
};

struct CommentData : public TypedModifier<CommentData> {
    static constexpr Modifier::Type Type = Modifier::Type::COMMENT;

    CommentData(const std::string& comment) : TypedModifier<CommentData>(), comment(comment) {}

    std::string comment;
};

struct MatrixFmtData : public TypedModifier<MatrixFmtData> {
    static constexpr Modifier::Type Type = Modifier::Type::MATRIX_FMT;

    MatrixFmtData(MatrixFmt fmtA, MatrixFmt fmtB)
        : TypedModifier<MatrixFmtData>(), a(fmtA), b(fmtB) {}

    MatrixFmt a;
    MatrixFmt b;
};

struct MemTokenData : public TypedModifier<MemTokenData> {
    static constexpr Modifier::Type Type = Modifier::Type::MEM_TOKEN;

    std::vector<int> tokens;

    MemTokenData(const std::vector<int>& tokens = {})
        : TypedModifier<MemTokenData>(), tokens(tokens) {}
};

}  // namespace stinkytofu
