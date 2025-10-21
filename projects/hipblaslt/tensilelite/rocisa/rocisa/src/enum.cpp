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
#include "enum.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

void init_enum(nb::module_ m)
{
    auto m_enum = m.def_submodule("enum", "rocIsa enum submodule.");

    nb::enum_<rocisa::RegisterType>(m_enum, "RegisterType")
        .value("Vgpr", rocisa::RegisterType::Vgpr)
        .value("Sgpr", rocisa::RegisterType::Sgpr)
        .value("Accvgpr", rocisa::RegisterType::Accvgpr)
        .value("mgpr", rocisa::RegisterType::mgpr)
        .export_values();

    // For Python only, Python already has a class named DataType
    nb::enum_<rocisa::DataType>(m_enum, "DataTypeEnum")
        .value("Float", rocisa::DataType::Float)
        .value("Double", rocisa::DataType::Double)
        .value("ComplexFloat", rocisa::DataType::ComplexFloat)
        .value("ComplexDouble", rocisa::DataType::ComplexDouble)
        .value("Half", rocisa::DataType::Half)
        .value("Int8x4", rocisa::DataType::Int8x4)
        .value("Int32", rocisa::DataType::Int32)
        .value("BFloat16", rocisa::DataType::BFloat16)
        .value("Int8", rocisa::DataType::Int8)
        .value("Int64", rocisa::DataType::Int64)
        .value("XFloat32", rocisa::DataType::XFloat32)
        .value("Float8_fnuz", rocisa::DataType::Float8_fnuz)
        .value("BFloat8_fnuz", rocisa::DataType::BFloat8_fnuz)
        .value("Float8BFloat8_fnuz", rocisa::DataType::Float8BFloat8_fnuz)
        .value("BFloat8Float8_fnuz", rocisa::DataType::BFloat8Float8_fnuz)
        .value("Float8", rocisa::DataType::Float8)
        .value("BFloat8", rocisa::DataType::BFloat8)
        .value("Float8BFloat8", rocisa::DataType::Float8BFloat8)
        .value("BFloat8Float8", rocisa::DataType::BFloat8Float8)
        .export_values();

    nb::enum_<rocisa::SignatureValueKind>(m_enum, "SignatureValueKind")
        .value("SIG_VALUE", rocisa::SignatureValueKind::SIG_VALUE)
        .value("SIG_GLOBALBUFFER", rocisa::SignatureValueKind::SIG_GLOBALBUFFER)
        .export_values();

    nb::enum_<rocisa::InstType>(m_enum, "InstType")
        .value("INST_F8", rocisa::InstType::INST_F8)
        .value("INST_F16", rocisa::InstType::INST_F16)
        .value("INST_F32", rocisa::InstType::INST_F32)
        .value("INST_F64", rocisa::InstType::INST_F64)
        .value("INST_I8", rocisa::InstType::INST_I8)
        .value("INST_I16", rocisa::InstType::INST_I16)
        .value("INST_I32", rocisa::InstType::INST_I32)
        .value("INST_U8", rocisa::InstType::INST_U8)
        .value("INST_U16", rocisa::InstType::INST_U16)
        .value("INST_U32", rocisa::InstType::INST_U32)
        .value("INST_U64", rocisa::InstType::INST_U64)
        .value("INST_LO_I32", rocisa::InstType::INST_LO_I32)
        .value("INST_HI_I32", rocisa::InstType::INST_HI_I32)
        .value("INST_LO_U32", rocisa::InstType::INST_LO_U32)
        .value("INST_HI_U32", rocisa::InstType::INST_HI_U32)
        .value("INST_BF16", rocisa::InstType::INST_BF16)
        .value("INST_B8", rocisa::InstType::INST_B8)
        .value("INST_B16", rocisa::InstType::INST_B16)
        .value("INST_B32", rocisa::InstType::INST_B32)
        .value("INST_B64", rocisa::InstType::INST_B64)
        .value("INST_B128", rocisa::InstType::INST_B128)
        .value("INST_B256", rocisa::InstType::INST_B256)
        .value("INST_B512", rocisa::InstType::INST_B512)
        .value("INST_B8_HI_D16", rocisa::InstType::INST_B8_HI_D16)
        .value("INST_D16_U8", rocisa::InstType::INST_D16_U8)
        .value("INST_D16_HI_U8", rocisa::InstType::INST_D16_HI_U8)
        .value("INST_D16_U16", rocisa::InstType::INST_D16_U16)
        .value("INST_D16_HI_U16", rocisa::InstType::INST_D16_HI_U16)
        .value("INST_D16_B8", rocisa::InstType::INST_D16_B8)
        .value("INST_D16_HI_B8", rocisa::InstType::INST_D16_HI_B8)
        .value("INST_D16_B16", rocisa::InstType::INST_D16_B16)
        .value("INST_D16_HI_B16", rocisa::InstType::INST_D16_HI_B16)
        .value("INST_XF32", rocisa::InstType::INST_XF32)
        .value("INST_BF8", rocisa::InstType::INST_BF8)
        .value("INST_F8_BF8", rocisa::InstType::INST_F8_BF8)
        .value("INST_BF8_F8", rocisa::InstType::INST_BF8_F8)
        .value("INST_TR8_B64", rocisa::InstType::INST_TR8_B64)
        .value("INST_TR16_B128", rocisa::InstType::INST_TR16_B128)
        .value("INST_CVT", rocisa::InstType::INST_CVT)
        .value("INST_MACRO", rocisa::InstType::INST_MACRO)
        .value("INST_NOTYPE", rocisa::InstType::INST_NOTYPE)
        .export_values();

    nb::enum_<rocisa::SelectBit>(m_enum, "SelectBit")
        .value("SEL_NONE", rocisa::SelectBit::SEL_NONE)
        .value("DWORD", rocisa::SelectBit::DWORD)
        .value("BYTE_0", rocisa::SelectBit::BYTE_0)
        .value("BYTE_1", rocisa::SelectBit::BYTE_1)
        .value("BYTE_2", rocisa::SelectBit::BYTE_2)
        .value("BYTE_3", rocisa::SelectBit::BYTE_3)
        .value("WORD_0", rocisa::SelectBit::WORD_0)
        .value("WORD_1", rocisa::SelectBit::WORD_1)
        .export_values();

    nb::enum_<rocisa::UnusedBit>(m_enum, "UnusedBit")
        .value("UNUSED_NONE", rocisa::UnusedBit::UNUSED_NONE)
        .value("UNUSED_PAD", rocisa::UnusedBit::UNUSED_PAD)
        .value("UNUSED_SEXT", rocisa::UnusedBit::UNUSED_SEXT)
        .value("UNUSED_PRESERVE", rocisa::UnusedBit::UNUSED_PRESERVE)
        .export_values();

    nb::enum_<rocisa::CvtType>(m_enum, "CvtType")
        .value("CVT_F16_to_F32", rocisa::CvtType::CVT_F16_to_F32)
        .value("CVT_F32_to_F16", rocisa::CvtType::CVT_F32_to_F16)
        .value("CVT_U32_to_F32", rocisa::CvtType::CVT_U32_to_F32)
        .value("CVT_F32_to_U32", rocisa::CvtType::CVT_F32_to_U32)
        .value("CVT_U32_to_F64", rocisa::CvtType::CVT_U32_to_F64)
        .value("CVT_F64_to_U32", rocisa::CvtType::CVT_F64_to_U32)
        .value("CVT_I32_to_F32", rocisa::CvtType::CVT_I32_to_F32)
        .value("CVT_F32_to_I32", rocisa::CvtType::CVT_F32_to_I32)
        .value("CVT_FP8_to_F32", rocisa::CvtType::CVT_FP8_to_F32)
        .value("CVT_BF8_to_F32", rocisa::CvtType::CVT_BF8_to_F32)
        .value("CVT_PK_FP8_to_F32", rocisa::CvtType::CVT_PK_FP8_to_F32)
        .value("CVT_PK_BF8_to_F32", rocisa::CvtType::CVT_PK_BF8_to_F32)
        .value("CVT_PK_F32_to_FP8", rocisa::CvtType::CVT_PK_F32_to_FP8)
        .value("CVT_PK_F32_to_BF8", rocisa::CvtType::CVT_PK_F32_to_BF8)
        .value("CVT_SR_F32_to_FP8", rocisa::CvtType::CVT_SR_F32_to_FP8)
        .value("CVT_SR_F32_to_BF8", rocisa::CvtType::CVT_SR_F32_to_BF8)
        .export_values();

    nb::enum_<rocisa::RoundType>(m_enum, "RoundType")
        .value("ROUND_UP", rocisa::RoundType::ROUND_UP)
        .value("ROUND_TO_NEAREST_EVEN", rocisa::RoundType::ROUND_TO_NEAREST_EVEN)
        .export_values();

    nb::enum_<rocisa::SaturateCastType>(m_enum, "SaturateCastType")
        .value("NORMAL", rocisa::SaturateCastType::NORMAL)
        .value("DO_NOTHING", rocisa::SaturateCastType::DO_NOTHING)
        .value("UPPER", rocisa::SaturateCastType::UPPER)
        .value("LOWER", rocisa::SaturateCastType::LOWER)
        .export_values();
}
