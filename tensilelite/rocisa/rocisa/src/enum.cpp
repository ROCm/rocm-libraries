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

    // For Python only, Python already has a class named DataType
    nb::enum_<TensileLite::DataType>(m_enum, "DataTypeEnum")
        .value("Float", TensileLite::DataType::Float)
        .value("Double", TensileLite::DataType::Double)
        .value("ComplexFloat", TensileLite::DataType::ComplexFloat)
        .value("ComplexDouble", TensileLite::DataType::ComplexDouble)
        .value("Half", TensileLite::DataType::Half)
        .value("Int8x4", TensileLite::DataType::Int8x4)
        .value("Int32", TensileLite::DataType::Int32)
        .value("BFloat16", TensileLite::DataType::BFloat16)
        .value("Int8", TensileLite::DataType::Int8)
        .value("Int64", TensileLite::DataType::Int64)
        .value("XFloat32", TensileLite::DataType::XFloat32)
        .value("Float8_fnuz", TensileLite::DataType::Float8_fnuz)
        .value("BFloat8_fnuz", TensileLite::DataType::BFloat8_fnuz)
        .value("Float8BFloat8_fnuz", TensileLite::DataType::Float8BFloat8_fnuz)
        .value("BFloat8Float8_fnuz", TensileLite::DataType::BFloat8Float8_fnuz)
        .value("Float8", TensileLite::DataType::Float8)
        .value("BFloat8", TensileLite::DataType::BFloat8)
        .value("Float8BFloat8", TensileLite::DataType::Float8BFloat8)
        .value("BFloat8Float8", TensileLite::DataType::BFloat8Float8)
        .export_values();

    nb::enum_<SignatureValueKind>(m_enum, "SignatureValueKind")
        .value("SIG_VALUE", SignatureValueKind::SIG_VALUE)
        .value("SIG_GLOBALBUFFER", SignatureValueKind::SIG_GLOBALBUFFER)
        .export_values();

    nb::enum_<InstType>(m_enum, "InstType")
        .value("INST_F8", InstType::INST_F8)
        .value("INST_F16", InstType::INST_F16)
        .value("INST_F32", InstType::INST_F32)
        .value("INST_F64", InstType::INST_F64)
        .value("INST_I8", InstType::INST_I8)
        .value("INST_I16", InstType::INST_I16)
        .value("INST_I32", InstType::INST_I32)
        .value("INST_U8", InstType::INST_U8)
        .value("INST_U16", InstType::INST_U16)
        .value("INST_U32", InstType::INST_U32)
        .value("INST_U64", InstType::INST_U64)
        .value("INST_LO_I32", InstType::INST_LO_I32)
        .value("INST_HI_I32", InstType::INST_HI_I32)
        .value("INST_LO_U32", InstType::INST_LO_U32)
        .value("INST_HI_U32", InstType::INST_HI_U32)
        .value("INST_BF16", InstType::INST_BF16)
        .value("INST_B8", InstType::INST_B8)
        .value("INST_B16", InstType::INST_B16)
        .value("INST_B32", InstType::INST_B32)
        .value("INST_B64", InstType::INST_B64)
        .value("INST_B128", InstType::INST_B128)
        .value("INST_B256", InstType::INST_B256)
        .value("INST_B512", InstType::INST_B512)
        .value("INST_B8_HI_D16", InstType::INST_B8_HI_D16)
        .value("INST_D16_U8", InstType::INST_D16_U8)
        .value("INST_D16_HI_U8", InstType::INST_D16_HI_U8)
        .value("INST_D16_U16", InstType::INST_D16_U16)
        .value("INST_D16_HI_U16", InstType::INST_D16_HI_U16)
        .value("INST_D16_B8", InstType::INST_D16_B8)
        .value("INST_D16_HI_B8", InstType::INST_D16_HI_B8)
        .value("INST_D16_B16", InstType::INST_D16_B16)
        .value("INST_D16_HI_B16", InstType::INST_D16_HI_B16)
        .value("INST_XF32", InstType::INST_XF32)
        .value("INST_BF8", InstType::INST_BF8)
        .value("INST_F8_BF8", InstType::INST_F8_BF8)
        .value("INST_BF8_F8", InstType::INST_BF8_F8)
        .value("INST_CVT", InstType::INST_CVT)
        .value("INST_MACRO", InstType::INST_MACRO)
        .value("INST_NOTYPE", InstType::INST_NOTYPE)
        .export_values();

    nb::enum_<SelectBit>(m_enum, "SelectBit")
        .value("SEL_NONE", SelectBit::SEL_NONE)
        .value("DWORD", SelectBit::DWORD)
        .value("BYTE_0", SelectBit::BYTE_0)
        .value("BYTE_1", SelectBit::BYTE_1)
        .value("BYTE_2", SelectBit::BYTE_2)
        .value("BYTE_3", SelectBit::BYTE_3)
        .value("WORD_0", SelectBit::WORD_0)
        .value("WORD_1", SelectBit::WORD_1)
        .export_values();

    nb::enum_<UnusedBit>(m_enum, "UnusedBit")
        .value("UNUSED_NONE", UnusedBit::UNUSED_NONE)
        .value("UNUSED_PAD", UnusedBit::UNUSED_PAD)
        .value("UNUSED_SEXT", UnusedBit::UNUSED_SEXT)
        .value("UNUSED_PRESERVE", UnusedBit::UNUSED_PRESERVE)
        .export_values();

    nb::enum_<CvtType>(m_enum, "CvtType")
        .value("CVT_F16_to_F32", CvtType::CVT_F16_to_F32)
        .value("CVT_F32_to_F16", CvtType::CVT_F32_to_F16)
        .value("CVT_U32_to_F32", CvtType::CVT_U32_to_F32)
        .value("CVT_F32_to_U32", CvtType::CVT_F32_to_U32)
        .value("CVT_I32_to_F32", CvtType::CVT_I32_to_F32)
        .value("CVT_F32_to_I32", CvtType::CVT_F32_to_I32)
        .value("CVT_FP8_to_F32", CvtType::CVT_FP8_to_F32)
        .value("CVT_BF8_to_F32", CvtType::CVT_BF8_to_F32)
        .value("CVT_PK_FP8_to_F32", CvtType::CVT_PK_FP8_to_F32)
        .value("CVT_PK_BF8_to_F32", CvtType::CVT_PK_BF8_to_F32)
        .value("CVT_PK_F32_to_FP8", CvtType::CVT_PK_F32_to_FP8)
        .value("CVT_PK_F32_to_BF8", CvtType::CVT_PK_F32_to_BF8)
        .value("CVT_SR_F32_to_FP8", CvtType::CVT_SR_F32_to_FP8)
        .value("CVT_SR_F32_to_BF8", CvtType::CVT_SR_F32_to_BF8)
        .export_values();

    nb::enum_<RoundType>(m_enum, "RoundType")
        .value("ROUND_UP", RoundType::ROUND_UP)
        .value("ROUND_TO_NEAREST_EVEN", RoundType::ROUND_TO_NEAREST_EVEN)
        .export_values();
}
