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
#include <string>

namespace rocisa
{
    enum class RegisterType : int
    {
        Vgpr,
        Sgpr,
        Accvgpr,
        mgpr
    };

    enum class DataType : int
    {
        Float,
        Double,
        ComplexFloat,
        ComplexDouble,
        Half,
        Int8x4,
        Int32,
        BFloat16,
        Int8,
        Int64,
        XFloat32,
        Float8_fnuz,
        BFloat8_fnuz,
        Float8BFloat8_fnuz,
        BFloat8Float8_fnuz,
        Float8,
        BFloat8,
        Float8BFloat8,
        BFloat8Float8,
        Float6,
        BFloat6,
        Float4,
        E8,
        E5M3,
        Count,
        None = Count
    };

    inline float dataTypeToBytes(DataType type)
    {
        switch(type)
        {
        case DataType::Float:
            return 4;
        case DataType::Double:
            return 8;
        case DataType::ComplexFloat:
            return 8;
        case DataType::ComplexDouble:
            return 16;
        case DataType::Half:
            return 2;
        case DataType::Int8x4:
            return 4;
        case DataType::Int32:
            return 4;
        case DataType::BFloat16:
            return 2;
        case DataType::Int8:
            return 1;
        case DataType::Int64:
            return 8;
        case DataType::XFloat32:
            return 4;
        case DataType::Float8_fnuz:
            return 1;
        case DataType::BFloat8_fnuz:
            return 1;
        case DataType::Float8BFloat8_fnuz:
            return 1;
        case DataType::BFloat8Float8_fnuz:
            return 1;
        case DataType::Float8:
            return 1;
        case DataType::BFloat8:
            return 1;
        case DataType::Float8BFloat8:
            return 1;
        case DataType::BFloat8Float8:
            return 1;
        case DataType::Float6:
            return 0.75;
        case DataType::BFloat6:
            return 0.75;
        case DataType::Float4:
            return 0.5;
        default:
            return -1; // Invalid type
        }
    }

    inline std::string toString(DataType type)
    {
        switch(type)
        {
        case DataType::Float:
            return "Float";
        case DataType::Double:
            return "Double";
        case DataType::ComplexFloat:
            return "ComplexFloat";
        case DataType::ComplexDouble:
            return "ComplexDouble";
        case DataType::Half:
            return "Half";
        case DataType::Int8x4:
            return "Int8x4";
        case DataType::Int32:
            return "Int32";
        case DataType::BFloat16:
            return "BFloat16";
        case DataType::Int8:
            return "Int8";
        case DataType::Int64:
            return "Int64";
        case DataType::XFloat32:
            return "XFloat32";
        case DataType::Float8_fnuz:
            return "Float8_fnuz";
        case DataType::BFloat8_fnuz:
            return "BFloat8_fnuz";
        case DataType::Float8BFloat8_fnuz:
            return "Float8BFloat8_fnuz";
        case DataType::BFloat8Float8_fnuz:
            return "BFloat8Float8_fnuz";
        case DataType::Float8:
            return "Float8";
        case DataType::BFloat8:
            return "BFloat8";
        case DataType::Float8BFloat8:
            return "Float8BFloat8";
        case DataType::BFloat8Float8:
            return "BFloat8Float8";
        case DataType::Float6:
            return "Float6";
        case DataType::BFloat6:
            return "BFloat6";
        case DataType::Float4:
            return "Float4";
        case DataType::E8:
            return "E8";
        case DataType::E5M3:
            return "E5M3";
        default:
            return "Invalid";
        }
        return "Invalid";
    }

    enum class SignatureValueKind : int
    {
        SIG_VALUE        = 1,
        SIG_GLOBALBUFFER = 2
    };

    enum class InstType : int
    {
        INST_F8         = 1,
        INST_F16        = 2,
        INST_F32        = 3,
        INST_F64        = 4,
        INST_I8         = 5,
        INST_I16        = 6,
        INST_I32        = 7,
        INST_U8         = 8,
        INST_U16        = 9,
        INST_U32        = 10,
        INST_U64        = 11,
        INST_LO_I32     = 12,
        INST_HI_I32     = 13,
        INST_LO_U32     = 14,
        INST_HI_U32     = 15,
        INST_BF16       = 16,
        INST_B8         = 17,
        INST_B16        = 18,
        INST_B32        = 19,
        INST_B64        = 20,
        INST_B128       = 21,
        INST_B192       = 22,
        INST_B256       = 23,
        INST_B512       = 24,
        INST_B8_HI_D16  = 25,
        INST_D16_U8     = 26,
        INST_D16_HI_U8  = 27,
        INST_D16_U16    = 28,
        INST_D16_HI_U16 = 29,
        INST_D16_B8     = 30,
        INST_D16_HI_B8  = 31,
        INST_D16_B16    = 32,
        INST_D16_HI_B16 = 33,
        INST_XF32       = 34,
        INST_BF8        = 35,
        INST_F8_BF8     = 36,
        INST_BF8_F8     = 37,
        INST_TR8_B64    = 38,
        INST_TR16_B128  = 39,
        INST_CVT        = 40,
        INST_MACRO      = 41,
        INST_F6         = 42,
        INST_BF6        = 43,
        INST_F4         = 44,
        INST_F8_F4      = 45,
        INST_F4_F8      = 46,
        INST_F6_F4      = 47,
        INST_F4_F6      = 48,
        INST_F8_F6      = 49,
        INST_F6_F8      = 50,
        INST_F8_B6      = 51,
        INST_B6_F8      = 52,
        INST_B8_F4      = 53,
        INST_F4_B8      = 54,
        INST_B6_F4      = 55,
        INST_F4_B6      = 56,
        INST_B8_F6      = 57,
        INST_F6_B8      = 58,
        INST_B8_B6      = 59,
        INST_B6_B8      = 60,
        INST_F6_B6      = 61,
        INST_B6_F6      = 62,
        INST_B96        = 63,
        INST_E8         = 64,
        INST_E5M3       = 65,
        INST_TDM        = 66,
        INST_SWAIT      = 67,
        INST_NOTYPE     = 68
    };

    enum class SelectBit : int
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

    enum class UnusedBit : int
    {
        UNUSED_NONE     = 0,
        UNUSED_PAD      = 1,
        UNUSED_SEXT     = 2,
        UNUSED_PRESERVE = 3
    };

    enum class CacheScope : int
    {
        SCOPE_NONE = 0,
        SCOPE_CU   = 1,
        SCOPE_SE   = 2,
        SCOPE_DEV  = 3,
        SCOPE_SYS  = 4,
    };

    // Temporal Hint encoding for gfx1250 memory ops.
    // Values match the ISA TH[2:0] field. LOAD and STORE share the same field
    // values but use different assembled names for TH3 and TH7.
    enum class TemporalHint : int
    {
        TH_NONE     = -1, // no th modifier
        TH_RT       = 0, // regular temporal (default for both near and far caches)
        TH_NT       = 1, // non-temporal (re-use not expected, both caches)
        TH_HT       = 2, // high-priority temporal
        TH_LU       = 3, // load-only: last-use (NT and discard dirty if hit)
        TH_WB       = 3, // store-only: same encoding as LU, assembled as WB
        TH_NT_RT    = 4, // non-temporal near, regular far
        TH_RT_NT    = 5, // regular near, non-temporal far
        TH_NT_HT    = 6, // non-temporal near, high-priority far
        TH_RESERVED = 7, // load-only: reserved TH7 encoding, kept explicit for diagnostics/tests
        TH_NT_WB    = 7, // store-only: NT near, WB far
    };

    inline bool hasTemporalHint(TemporalHint th)
    {
        return th != TemporalHint::TH_NONE;
    }

    // Non-Volatile memory access modifier for gfx1250.
    // NV_NONE emits no modifier and preserves the default volatile behavior.
    // NV emits the ISA ``nv`` modifier when the assembler supports it.
    enum class NonVolatile : int
    {
        NV_NONE = 0,
        NV      = 1,
    };

    enum class HighBitSel : int
    {
        NONE = -1,
        LOW  = 0,
        HIGH = 1
    };

    enum class CvtType : int
    {
        CVT_F16_to_F32          = 1,
        CVT_F32_to_F16          = 2,
        CVT_U32_to_F32          = 3,
        CVT_F32_to_U32          = 4,
        CVT_I32_to_F32          = 5,
        CVT_F32_to_I32          = 6,
        CVT_FP8_to_F32          = 7,
        CVT_BF8_to_F32          = 8,
        CVT_PK_FP8_to_F32       = 9,
        CVT_PK_BF8_to_F32       = 10,
        CVT_PK_F32_to_FP8       = 11,
        CVT_PK_F32_to_BF8       = 12,
        CVT_SR_F32_to_FP8       = 13,
        CVT_SR_F32_to_BF8       = 14,
        CVT_SCALEF32_PK_F16_FP8 = 15,
        CVT_SCALEF32_PK_F16_BF8 = 16,
        CVT_SCALEF32_F16_FP8    = 17,
        CVT_SCALEF32_F16_BF8    = 18,
        CVT_SCALEF32_PK_FP8_F16 = 19,
        CVT_SCALEF32_PK_BF8_F16 = 20,
        CVT_SCALEF32_SR_FP8_F16 = 21,
        CVT_SCALEF32_SR_BF8_F16 = 22,
        CVT_BF16_to_F32         = 23,
        CVT_PK_F32_to_BF16      = 24,
        CVT_U32_to_F64          = 25,
        CVT_F64_to_U32          = 26,
        CVT_FP8_to_F16          = 27,
        CVT_PK_FP8_to_F16       = 28,
        CVT_PK_F32_to_F16       = 29,
        CVT_SCALEF32_PK8_FP8_F32 = 30,
        CVT_SCALEF32_PK8_BF8_F32 = 31,
        CVT_SCALEF32_SR_PK8_FP8_F32 = 32
    };

    enum class RoundType : int
    {
        ROUND_UP              = 0,
        ROUND_TO_NEAREST_EVEN = 1
    };

    enum class ArgType : int
    {
        DST  = 0,
        DST1 = 1,
        SRC0 = 2
    };

    inline std::string toString(SelectBit bit)
    {
        switch(bit)
        {
        case SelectBit::DWORD:
            return "DWORD";
        case SelectBit::BYTE_0:
            return "BYTE_0";
        case SelectBit::BYTE_1:
            return "BYTE_1";
        case SelectBit::BYTE_2:
            return "BYTE_2";
        case SelectBit::BYTE_3:
            return "BYTE_3";
        case SelectBit::WORD_0:
            return "WORD_0";
        case SelectBit::WORD_1:
            return "WORD_1";
        default:
            return "";
        }
    }

    inline std::string toString(UnusedBit bit)
    {
        switch(bit)
        {
        case UnusedBit::UNUSED_PAD:
            return "UNUSED_PAD";
        case UnusedBit::UNUSED_SEXT:
            return "UNUSED_SEXT";
        case UnusedBit::UNUSED_PRESERVE:
            return "UNUSED_PRESERVE";
        default:
            return "";
        }
    }

    inline std::string toString(CacheScope scope)
    {
        switch(scope)
        {
        case CacheScope::SCOPE_CU:
            return "SCOPE_CU";
        case CacheScope::SCOPE_SE:
            return "SCOPE_SE";
        case CacheScope::SCOPE_DEV:
            return "SCOPE_DEV";
        case CacheScope::SCOPE_SYS:
            return "SCOPE_SYS";
        default:
            return "";
        }
    }

    // Emits the optional "nv" mnemonic for Non-Volatile memory accesses.
    // NV_NONE returns an empty string so callers can skip the modifier.
    inline std::string toString(NonVolatile nv)
    {
        return nv == NonVolatile::NV ? "nv" : "";
    }

    // Emits the "TH_LOAD_*" / "TH_STORE_*" mnemonic for the given temporal hint.
    // Caller picks the prefix via isStore because LOAD and STORE share TH[2:0]
    // encodings but differ in the assembled name for TH3 and TH7.
    inline std::string toString(TemporalHint th, bool isStore)
    {
        const std::string prefix = isStore ? "TH_STORE_" : "TH_LOAD_";
        switch(th)
        {
        case TemporalHint::TH_RT:
            return prefix + "RT";
        case TemporalHint::TH_NT:
            return prefix + "NT";
        case TemporalHint::TH_HT:
            return prefix + "HT";
        case TemporalHint::TH_LU:
            return isStore ? prefix + "WB" : prefix + "LU";
        case TemporalHint::TH_NT_RT:
            return prefix + "NT_RT";
        case TemporalHint::TH_RT_NT:
            return prefix + "RT_NT";
        case TemporalHint::TH_NT_HT:
            return prefix + "NT_HT";
        case TemporalHint::TH_RESERVED:
            return isStore ? prefix + "NT_WB" : prefix + "RESERVED";
        default:
            return "";
        }
    }

    enum class SaturateCastType : int
    {
        NORMAL     = 1,
        DO_NOTHING = 2,
        UPPER      = 3,
        LOWER      = 4
    };

    enum class DelayALUType : int
    {
        VALU  = 0,
        TRANS = 1,
        SALU  = 2,
        OTHER = 3,
    };

    enum class DelayALUSkip : int
    {
        SAME   = 0,
        NEXT   = 1,
        SKIP_1 = 2,
        SKIP_2 = 3,
        SKIP_3 = 4,
        SKIP_4 = 5,
    };

    inline std::string toString(DelayALUSkip cnt)
    {
        switch(cnt)
        {
        case DelayALUSkip::SAME:
            return "SAME";
        case DelayALUSkip::NEXT:
            return "NEXT";
        case DelayALUSkip::SKIP_1:
            return "SKIP_1";
        case DelayALUSkip::SKIP_2:
            return "SKIP_2";
        case DelayALUSkip::SKIP_3:
            return "SKIP_3";
        case DelayALUSkip::SKIP_4:
            return "SKIP_4";
        default:
            return "";
        }
    }

    inline std::string toString(DelayALUType type)
    {
        switch(type)
        {
        case DelayALUType::VALU:
            return "VALU";
        case DelayALUType::TRANS:
            return "TRANS";
        case DelayALUType::SALU:
            return "SALU";
        default:
            return "";
        }
    }

    inline std::string toString(DelayALUType type, int cnt)
    {
        if(!cnt)
            return "NO_DEP";

        switch(type)
        {
        case DelayALUType::VALU:
        case DelayALUType::TRANS:
            return toString(type) + "_DEP_" + std::to_string(cnt);
        case DelayALUType::SALU:
            return "SALU_CYCLE_" + std::to_string(cnt);
        case DelayALUType::OTHER:
        default:
            return "";
        }
    }
}
