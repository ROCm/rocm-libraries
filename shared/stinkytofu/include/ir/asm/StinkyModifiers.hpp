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
#include <string>
#include <vector>

namespace stinkytofu
{
    // Enum for selecting high or low 16 bits in True16 instructions
    enum class HighBitSel : int
    {
        NONE = -1,
        LOW  = 0,
        HIGH = 1
    };

    struct Modifier
    {
        enum class Type : uint8_t
        {
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
            SDELAYALU_DATA,
            LABEL_NAME,
            COMMENT,
        };

        Modifier(Type type)
            : type(type)
        {
        }

        virtual ~Modifier() = default;

        const Type& getType() const
        {
            return type;
        }

    private:
        Type type;
    };

    struct DSModifiers : public Modifier
    {
        DSModifiers(int na = 1, int offset = 0, int offset0 = 0, int offset1 = 0, bool gds = false)
            : Modifier(Type::DS)
            , na(na)
            , offset(offset)
            , offset0(offset0)
            , offset1(offset1)
            , gds(gds)
        {
        }

        int  na;
        int  offset;
        int  offset0;
        int  offset1;
        bool gds;
    };

    struct FLATModifiers : public Modifier
    {
        FLATModifiers(int  offset12 = 0,
                      bool glc      = false,
                      bool slc      = false,
                      bool lds      = false,
                      bool isStore  = false)
            : Modifier(Type::FLAT)
            , offset12(offset12)
            , glc(glc)
            , slc(slc)
            , lds(lds)
            , isStore(isStore)
        {
        }

        int      offset12;
        uint32_t glc : 1;
        uint32_t slc : 1;
        uint32_t lds : 1;
        uint32_t isStore : 1;
    };

    struct GLOBALModifiers : public Modifier
    {
        GLOBALModifiers(int offset = 0)
            : Modifier(Type::GLOBAL)
            , offset(offset)
        {
        }

        int offset;
    };

    struct MUBUFModifiers : public Modifier
    {
        MUBUFModifiers(bool offen    = false,
                       int  offset12 = 0,
                       bool glc      = false,
                       bool slc      = false,
                       bool nt       = false,
                       bool lds      = false,
                       bool isStore  = false)
            : Modifier(Type::MUBUF)
            , offset12(offset12)
            , offen(offen)
            , glc(glc)
            , slc(slc)
            , nt(nt)
            , lds(lds)
            , isStore(isStore)
        {
        }

        int      offset12;
        uint32_t offen : 1;
        uint32_t glc : 1;
        uint32_t slc : 1;
        uint32_t nt : 1;
        uint32_t lds : 1;
        uint32_t isStore : 1;
    };

    struct SMEMModifiers : public Modifier
    {
        SMEMModifiers(bool glc = false, bool nv = false, int offset = 0)
            : Modifier(Type::SMEM)
            , glc(glc)
            , nv(nv)
            , offset(offset) // 20u 21s shaes the same
        {
        }

        uint32_t glc : 1;
        uint32_t nv : 1;
        int      offset;
    };

    struct SDWAModifiers : public Modifier
    {
        enum class SelectBit : uint8_t
        {
            SEL_NONE = 0,
            DWORD    = 1,
            BYTE_0   = 2,
            BYTE_1   = 3,
            BYTE_2   = 4,
            BYTE_3   = 5,
            WORD_0   = 6,
            WORD_1   = 7
        };

        enum class UnusedBit : uint8_t
        {
            UNUSED_NONE     = 0,
            UNUSED_PAD      = 1,
            UNUSED_SEXT     = 2,
            UNUSED_PRESERVE = 3
        };

        SDWAModifiers(SelectBit dst_sel    = SelectBit::SEL_NONE,
                      UnusedBit dst_unused = UnusedBit::UNUSED_NONE,
                      SelectBit src0_sel   = SelectBit::SEL_NONE,
                      SelectBit src1_sel   = SelectBit::SEL_NONE)
            : Modifier(Type::SDWA)
            , dst_sel(dst_sel)
            , dst_unused(dst_unused)
            , src0_sel(src0_sel)
            , src1_sel(src1_sel)
        {
        }

        SelectBit dst_sel;
        UnusedBit dst_unused;
        SelectBit src0_sel;
        SelectBit src1_sel;
    };

    // dot2: for WaveSplitK reduction. Only a subset of DPP modifiers are used here
    struct DPPModifiers : public Modifier
    {
        int row_shr;
        int row_bcast;
        int bound_ctrl;

        DPPModifiers(int row_shr = -1, int row_bcast = -1, int bound_ctrl = -1)
            : Modifier(Type::DPP)
            , row_shr(row_shr)
            , row_bcast(row_bcast)
            , bound_ctrl(bound_ctrl)
        {
        }
    };

    struct VOP3Modifiers : public Modifier
    {
        VOP3Modifiers(bool neg_src0 = false,
                      bool neg_src1 = false,
                      bool neg_src2 = false,
                      bool abs_src0 = false,
                      bool abs_src1 = false,
                      bool abs_src2 = false,
                      bool clamp    = false,
                      int  omod     = 0)
            : Modifier(Type::VOP3)
            , neg_src0(neg_src0)
            , neg_src1(neg_src1)
            , neg_src2(neg_src2)
            , abs_src0(abs_src0)
            , abs_src1(abs_src1)
            , abs_src2(abs_src2)
            , clamp(clamp)
            , omod(omod)
        {
        }

        bool neg_src0; // Negate source operand 0
        bool neg_src1; // Negate source operand 1
        bool neg_src2; // Negate source operand 2
        bool abs_src0; // Absolute value of source operand 0
        bool abs_src1; // Absolute value of source operand 1
        bool abs_src2; // Absolute value of source operand 2
        bool clamp; // Clamp result to [0.0, 1.0]
        int  omod; // Output modifier: 0=*1, 1=*2, 2=*4, 3=*0.5
    };

    struct VOP3PModifiers : public Modifier
    {
        VOP3PModifiers(const std::vector<int>& op_sel    = {},
                       const std::vector<int>& op_sel_hi = {},
                       const std::vector<int>& byte_sel  = {})
            : Modifier(Type::VOP3P)
            , op_sel(op_sel)
            , op_sel_hi(op_sel_hi)
            , byte_sel(byte_sel)
        {
        }

        std::vector<int> op_sel;
        std::vector<int> op_sel_hi;
        std::vector<int> byte_sel;
    };

    struct True16Modifiers : public Modifier
    {
        True16Modifiers(HighBitSel high_bit = HighBitSel::NONE)
            : Modifier(Type::TRUE16)
            , high_bit(high_bit)
        {
        }

        True16Modifiers(int high_bit_int)
            : Modifier(Type::TRUE16)
            , high_bit(static_cast<HighBitSel>(high_bit_int))
        {
        }

        HighBitSel high_bit;
    };

    struct EXEC : public Modifier
    {
        EXEC(bool setHi = false)
            : Modifier(Type::EXEC)
            , setHi(setHi)
        {
        }

        bool setHi;
    };

    struct VCC : public Modifier
    {
        VCC(bool setHi = false)
            : Modifier(Type::VCC)
            , setHi(setHi)
        {
        }

        bool setHi;
    };

    struct SWaitCntData : public Modifier
    {
        SWaitCntData(int8_t vlcnt = -1,
                     int8_t vscnt = -1,
                     int8_t dlcnt = -1,
                     int8_t dscnt = -1,
                     int8_t kmcnt = -1)
            : Modifier(Type::SWAITCNT_DATA)
            , vlcnt(vlcnt)
            , vscnt(vscnt)
            , dlcnt(dlcnt)
            , dscnt(dscnt)
            , kmcnt(kmcnt)
        {
        }

        // vlcnt: Number of VMEM load (and atomic with return value) instructions issued but not yet completed.
        // vscnt: Number of VMEM store (and atomic w/o return value) instructions issued but not yet completed.
        // dlcnt: Number of LDS load instructions issued but not yet completed.
        // dscnt: Number of LDS store instructions issued but not yet completed.
        // kmcnt: Number of constant-fetch (scalar memory read), and message instructions issued but not yet completed.
        //
        // In some ISA, VMEM load/store share the same counter(vmcnt). LDS, scalar memory read and message share
        // the same counter(lgkmcnt). These counters are combined from the 4 counters above as:
        //     vmcnt   = vlcnt + vscnt
        //     lgkmcnt = dlcnt + dscnt + kmcnt
        //
        // Example: 4 VMEM instructions vl_0, vl_1, vs_0, vl_2 are issued and we want to wait for vl_1 to complete.
        //
        //     If the target ISA has separate counters for load and store, use SWaitCnt(vlcnt=1),
        //     which means vl_2 is not completed yet.
        //
        //     If the target ISA has a single counter for load and store, use SWaitCnt(vlcnt=1, vscnt=1),
        //     which means vs_0, vl_2 are not completed yet.
        int8_t vlcnt;
        int8_t vscnt;
        int8_t dlcnt;
        int8_t dscnt;
        int8_t kmcnt;
    };

    struct LabelData : public Modifier
    {
        LabelData(const Modifier::Type& type, const std::string& label)
            : Modifier(type)
            , label(label)
        {
        }
        std::string label;
    };

    struct SWaitTensorCntData : public Modifier
    {
        SWaitTensorCntData(int8_t tlcnt = -1)
            : Modifier(Type::SWAITTENSORCNT_DATA)
            , tlcnt(tlcnt)
        {
        }

        int8_t tlcnt;
    };

    /// Delay ALU instruction data for RDNA3/RDNA4 (gfx11xx/gfx12xx)
    /// Specifies dependencies between ALU instructions
    struct SDelayAluData : public Modifier
    {
        enum class InstType : uint8_t
        {
            VALU, ///< Vector ALU instruction
            SALU, ///< Scalar ALU instruction
            TRANS, ///< Transcendental instruction
            NO_DEP ///< No dependency
        };

        // Simple constructor for common case (single dependency)
        SDelayAluData(InstType type, int8_t distance = 0)
            : Modifier(Type::SDELAYALU_DATA)
            , type(type)
            , distance(distance)
        {
        }

        InstType type; ///< Instruction type (VALU/SALU/TRANS/NO_DEP)
        int8_t   distance; ///< Dependency distance (1-4 for VALU, 1 for SALU, 1-3 for TRANS)

        // TODO: Add support for dual dependencies (instid1) if needed in the future
        // For now, we only support single dependency which covers 99% of cases
    };

    struct CommentData : public Modifier
    {
        CommentData(const std::string& comment)
            : Modifier(Type::COMMENT)
            , comment(comment)
        {
        }

        std::string comment;
    };

    inline std::ostream& operator<<(std::ostream& os, const SWaitCntData& waitCntData)
    {
        os << "vlcnt=" << (int)waitCntData.vlcnt << ", vscnt=" << (int)waitCntData.vscnt
           << ", dlcnt=" << (int)waitCntData.dlcnt << ", dscnt=" << (int)waitCntData.dscnt
           << ", kmcnt=" << (int)waitCntData.kmcnt;
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const SWaitTensorCntData& waitTensorCntData)
    {
        os << "tlcnt=" << (int)waitTensorCntData.tlcnt;
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const SDelayAluData& delayAluData)
    {
        auto typeToString = [](SDelayAluData::InstType type) -> const char* {
            switch(type)
            {
            case SDelayAluData::InstType::VALU:
                return "VALU";
            case SDelayAluData::InstType::SALU:
                return "SALU";
            case SDelayAluData::InstType::TRANS:
                return "TRANS";
            case SDelayAluData::InstType::NO_DEP:
                return "NO_DEP";
            default:
                return "UNKNOWN";
            }
        };

        os << "instid0(" << typeToString(delayAluData.type);

        // SALU always uses CYCLE_1, others use DEP_N
        if(delayAluData.type == SDelayAluData::InstType::SALU)
        {
            os << "_CYCLE_1";
        }
        else if(delayAluData.type != SDelayAluData::InstType::NO_DEP)
        {
            os << "_DEP_" << (int)delayAluData.distance;
        }

        os << ")";
        return os;
    }

    // clang-format off
    // Helper template for type mapping
    template<typename T> constexpr Modifier::Type getModifierType();
    template<> constexpr Modifier::Type getModifierType<DSModifiers>() { return Modifier::Type::DS; }
    template<> constexpr Modifier::Type getModifierType<FLATModifiers>() { return Modifier::Type::FLAT; }
    template<> constexpr Modifier::Type getModifierType<GLOBALModifiers>() { return Modifier::Type::GLOBAL; }
    template<> constexpr Modifier::Type getModifierType<MUBUFModifiers>() { return Modifier::Type::MUBUF; }
    template<> constexpr Modifier::Type getModifierType<SMEMModifiers>() { return Modifier::Type::SMEM; }
    template<> constexpr Modifier::Type getModifierType<SDWAModifiers>() { return Modifier::Type::SDWA; }
    template<> constexpr Modifier::Type getModifierType<DPPModifiers>() { return Modifier::Type::DPP; }
    template<> constexpr Modifier::Type getModifierType<VOP3Modifiers>() { return Modifier::Type::VOP3; }
    template<> constexpr Modifier::Type getModifierType<VOP3PModifiers>() { return Modifier::Type::VOP3P; }
    template<> constexpr Modifier::Type getModifierType<True16Modifiers>() { return Modifier::Type::TRUE16; }
    template<> constexpr Modifier::Type getModifierType<EXEC>() { return Modifier::Type::EXEC; }
    template<> constexpr Modifier::Type getModifierType<VCC>() { return Modifier::Type::VCC; }
    template<> constexpr Modifier::Type getModifierType<SWaitCntData>() { return Modifier::Type::SWAITCNT_DATA; }
    template<> constexpr Modifier::Type getModifierType<SWaitTensorCntData>() { return Modifier::Type::SWAITTENSORCNT_DATA; }
    template<> constexpr Modifier::Type getModifierType<SDelayAluData>() { return Modifier::Type::SDELAYALU_DATA; }
    template<> constexpr Modifier::Type getModifierType<LabelData>() { return Modifier::Type::LABEL_NAME; }
    template<> constexpr Modifier::Type getModifierType<CommentData>() { return Modifier::Type::COMMENT; }
    // clang-format on

} // namespace stinkytofu
