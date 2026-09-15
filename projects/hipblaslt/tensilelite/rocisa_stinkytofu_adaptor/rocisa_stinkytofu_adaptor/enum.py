# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real IntEnum definitions mirroring rocisa/src/enum.cpp 1:1.

All enums (RegisterType, DataTypeEnum, InstType, SelectBit, etc.) are fully implemented.
"""

from __future__ import annotations

from ._dummy import export_enum_values, make_bound_enum, make_dummy_enum

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


# Member order follows ``rocisa::enum.cpp`` ``InstType`` binding; values
# follow ``rocisa::InstType`` in ``enum.hpp``.
_InstType_members = [
    ("INST_E5M3", 65), ("INST_E8", 64), ("INST_F4", 44), ("INST_F6", 42),
    ("INST_BF6", 43), ("INST_F6_B6", 61), ("INST_B6_F6", 62), ("INST_F8", 1),
    ("INST_F16", 2), ("INST_F32", 3), ("INST_F64", 4), ("INST_I8", 5),
    ("INST_I16", 6), ("INST_I32", 7), ("INST_U8", 8), ("INST_U16", 9),
    ("INST_U32", 10), ("INST_U64", 11), ("INST_LO_I32", 12), ("INST_HI_I32", 13),
    ("INST_LO_U32", 14), ("INST_HI_U32", 15), ("INST_BF16", 16), ("INST_B8", 17),
    ("INST_B16", 18), ("INST_B32", 19), ("INST_B64", 20), ("INST_B128", 21),
    ("INST_B256", 23), ("INST_B512", 24), ("INST_B8_HI_D16", 25),
    ("INST_D16_U8", 26), ("INST_D16_HI_U8", 27), ("INST_D16_U16", 28),
    ("INST_D16_HI_U16", 29), ("INST_D16_B8", 30), ("INST_D16_HI_B8", 31),
    ("INST_D16_B16", 32), ("INST_D16_HI_B16", 33), ("INST_XF32", 34),
    ("INST_BF8", 35), ("INST_F8_BF8", 36), ("INST_BF8_F8", 37),
    ("INST_TR8_B64", 38), ("INST_TR16_B128", 39), ("INST_F8_F4", 45),
    ("INST_F4_F8", 46), ("INST_F6_F4", 47), ("INST_F4_F6", 48),
    ("INST_F8_F6", 49), ("INST_F6_F8", 50), ("INST_F8_B6", 51),
    ("INST_B6_F8", 52), ("INST_B8_F4", 53), ("INST_F4_B8", 54),
    ("INST_B6_F4", 55), ("INST_F4_B6", 56), ("INST_B8_F6", 57),
    ("INST_F6_B8", 58), ("INST_B8_B6", 59), ("INST_B6_B8", 60),
    ("INST_CVT", 40), ("INST_MACRO", 41), ("INST_B192", 22),
    ("INST_NOTYPE", 68),
]
_InstType_values = [name for name, _ in _InstType_members]
InstType = make_bound_enum(f"{_P}.InstType", _InstType_members)
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


_TemporalHint_members = [
    ("TH_NONE", -1),
    ("TH_RT", 0),
    ("TH_NT", 1),
    ("TH_HT", 2),
    ("TH_LU", 3),
    ("TH_NT_RT", 4),
    ("TH_RT_NT", 5),
    ("TH_NT_HT", 6),
    ("TH_RESERVED", 7),
    ("TH_WB", 3),
    ("TH_NT_WB", 7),
]
_TemporalHint_values = [name for name, _ in _TemporalHint_members]
TemporalHint = make_bound_enum(f"{_P}.TemporalHint", _TemporalHint_members)
export_enum_values(globals(), TemporalHint, _TemporalHint_values)


_NonVolatile_members = [
    ("NV_NONE", 0),
    ("NV", 1),
]
_NonVolatile_values = [name for name, _ in _NonVolatile_members]
NonVolatile = make_bound_enum(f"{_P}.NonVolatile", _NonVolatile_members)
export_enum_values(globals(), NonVolatile, _NonVolatile_values)


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
