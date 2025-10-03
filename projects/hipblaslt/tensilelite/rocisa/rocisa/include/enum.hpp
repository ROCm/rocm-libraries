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
        Count,
        None = Count
    };

    inline int dataTypeToBytes(DataType type)
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
        INST_B256       = 22,
        INST_B512       = 23,
        INST_B8_HI_D16  = 24,
        INST_D16_U8     = 25,
        INST_D16_HI_U8  = 26,
        INST_D16_U16    = 27,
        INST_D16_HI_U16 = 28,
        INST_D16_B8     = 29,
        INST_D16_HI_B8  = 30,
        INST_D16_B16    = 31,
        INST_D16_HI_B16 = 32,
        INST_XF32       = 33,
        INST_BF8        = 34,
        INST_F8_BF8     = 35,
        INST_BF8_F8     = 36,
        INST_TR8_B64    = 37,
        INST_TR16_B128  = 38,
        INST_CVT        = 39,
        INST_MACRO      = 40,
        INST_NOTYPE     = 41
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
    };

    enum class RoundType : int
    {
        ROUND_UP              = 0,
        ROUND_TO_NEAREST_EVEN = 1
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
