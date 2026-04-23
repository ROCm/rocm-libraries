# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.enum``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/enum.cpp``
(``init_enum``). Every enum here matches the nanobind binding 1:1 including
the member values so that ``from rocisa.enum import DWORD`` etc. keep
working exactly as before (nanobind's ``export_values()`` surfaces each
member as a top-level attribute of the submodule).

logicalIR correspondence: none of these enums has a 1:1 counterpart in
``shared/stinkytofu/src/ir/logical/`` at the time of writing - logicalIR
uses a different type system (see ``StinkyRegister::Type`` / ``RegType`` in
``python_module/src/python_bindings.cpp``). Left as pure dummies.
"""

from __future__ import annotations

from ._dummy import export_enum_values, make_dummy_enum

_P = "rocisa.enum"


_RegisterType_values = ["Vgpr", "Sgpr", "Accvgpr", "mgpr"]
RegisterType = make_dummy_enum(f"{_P}.RegisterType", _RegisterType_values)
export_enum_values(globals(), RegisterType, _RegisterType_values)


_DataTypeEnum_values = [
    "Float", "Double", "ComplexFloat", "ComplexDouble", "Half",
    "Int8x4", "Int32", "BFloat16", "Int8", "Int64", "XFloat32",
    "Float8_fnuz", "BFloat8_fnuz", "Float8BFloat8_fnuz", "BFloat8Float8_fnuz",
    "Float8", "BFloat8", "Float8BFloat8", "BFloat8Float8",
    "Float6", "BFloat6", "Float4", "E8", "E5M3",
]
DataTypeEnum = make_dummy_enum(f"{_P}.DataTypeEnum", _DataTypeEnum_values)
export_enum_values(globals(), DataTypeEnum, _DataTypeEnum_values)


_SignatureValueKind_values = ["SIG_VALUE", "SIG_GLOBALBUFFER"]
SignatureValueKind = make_dummy_enum(f"{_P}.SignatureValueKind", _SignatureValueKind_values)
export_enum_values(globals(), SignatureValueKind, _SignatureValueKind_values)


_InstType_values = [
    "INST_E5M3", "INST_E8", "INST_F4", "INST_F6", "INST_BF6",
    "INST_F6_B6", "INST_B6_F6", "INST_F8", "INST_F16", "INST_F32",
    "INST_F64", "INST_I8", "INST_I16", "INST_I32", "INST_U8",
    "INST_U16", "INST_U32", "INST_U64", "INST_LO_I32", "INST_HI_I32",
    "INST_LO_U32", "INST_HI_U32", "INST_BF16", "INST_B8", "INST_B16",
    "INST_B32", "INST_B64", "INST_B128", "INST_B256", "INST_B512",
    "INST_B8_HI_D16", "INST_D16_U8", "INST_D16_HI_U8", "INST_D16_U16",
    "INST_D16_HI_U16", "INST_D16_B8", "INST_D16_HI_B8", "INST_D16_B16",
    "INST_D16_HI_B16", "INST_XF32", "INST_BF8", "INST_F8_BF8",
    "INST_BF8_F8", "INST_TR8_B64", "INST_TR16_B128", "INST_F8_F4",
    "INST_F4_F8", "INST_F6_F4", "INST_F4_F6", "INST_F8_F6",
    "INST_F6_F8", "INST_F8_B6", "INST_B6_F8", "INST_B8_F4",
    "INST_F4_B8", "INST_B6_F4", "INST_F4_B6", "INST_B8_F6",
    "INST_F6_B8", "INST_B8_B6", "INST_B6_B8", "INST_CVT",
    "INST_MACRO", "INST_NOTYPE",
]
InstType = make_dummy_enum(f"{_P}.InstType", _InstType_values)
export_enum_values(globals(), InstType, _InstType_values)


_SelectBit_values = [
    "SEL_NONE", "DWORD", "BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3",
    "WORD_0", "WORD_1",
]
SelectBit = make_dummy_enum(f"{_P}.SelectBit", _SelectBit_values)
export_enum_values(globals(), SelectBit, _SelectBit_values)


_UnusedBit_values = ["UNUSED_NONE", "UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE"]
UnusedBit = make_dummy_enum(f"{_P}.UnusedBit", _UnusedBit_values)
export_enum_values(globals(), UnusedBit, _UnusedBit_values)


_CacheScope_values = ["SCOPE_NONE", "SCOPE_CU", "SCOPE_SE", "SCOPE_DEV", "SCOPE_SYS"]
CacheScope = make_dummy_enum(f"{_P}.CacheScope", _CacheScope_values)
export_enum_values(globals(), CacheScope, _CacheScope_values)


_CvtType_values = [
    "CVT_F16_to_F32", "CVT_F32_to_F16", "CVT_U32_to_F32", "CVT_F32_to_U32",
    "CVT_U32_to_F64", "CVT_F64_to_U32", "CVT_I32_to_F32", "CVT_F32_to_I32",
    "CVT_FP8_to_F32", "CVT_BF8_to_F32", "CVT_PK_FP8_to_F32", "CVT_PK_BF8_to_F32",
    "CVT_PK_F32_to_FP8", "CVT_PK_F32_to_BF8", "CVT_SR_F32_to_FP8",
    "CVT_SR_F32_to_BF8",
]
CvtType = make_dummy_enum(f"{_P}.CvtType", _CvtType_values)
export_enum_values(globals(), CvtType, _CvtType_values)


_ArgType_values = ["DST", "DST1", "SRC0"]
ArgType = make_dummy_enum(f"{_P}.ArgType", _ArgType_values)
export_enum_values(globals(), ArgType, _ArgType_values)


_HighBitSel_values = ["NONE", "LOW", "HIGH"]
HighBitSel = make_dummy_enum(f"{_P}.HighBitSel", _HighBitSel_values)
export_enum_values(globals(), HighBitSel, _HighBitSel_values)


_RoundType_values = ["ROUND_UP", "ROUND_TO_NEAREST_EVEN"]
RoundType = make_dummy_enum(f"{_P}.RoundType", _RoundType_values)
export_enum_values(globals(), RoundType, _RoundType_values)


_SaturateCastType_values = ["NORMAL", "DO_NOTHING", "UPPER", "LOWER"]
SaturateCastType = make_dummy_enum(f"{_P}.SaturateCastType", _SaturateCastType_values)
export_enum_values(globals(), SaturateCastType, _SaturateCastType_values)
