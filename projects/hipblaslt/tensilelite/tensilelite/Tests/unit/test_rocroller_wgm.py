# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocRoller workgroup-mapping kernargs match libdivide branchfree magic.

The frozen table was captured from libdivide 5.4.0 (the header hipBLASLt
fetches for rocRoller) plus the divisor 0 and 1 special cases in
rocRoller Expression_evaluate_unary.cpp. Tensile magicNumber() is a
different algorithm and is not what these kernels consume.

tensilelite/tests/RocRollerWgmKernargs_test.cpp checks the C++ launch header
against the same tables; edit both together.
"""

from pathlib import Path

import pytest

from tensilelite.CustomKernels import _metadataArgToCustomArg, readCustomKernelConfig
from tensilelite.RocRollerWgm import (
    WGM_ROLE_ORDER,
    branchfree_divisor,
    classify_wgm_expression,
    evaluate_branchfree_tree,
    evaluate_wgm,
    magic_div_s32,
    magic_multiple_s32,
    magic_multiple_s64,
    magic_multiple_u32,
    magic_shift_and_sign_s32,
    magic_shift_and_sign_s64,
    magic_shifts_u32,
    num_tiles_s64,
    s32,
)

pytestmark = pytest.mark.unit

# (kind, divisor, magic, second). For u32, second is MagicShifts printed as a
# signed int, which is how rocRoller returns `1 << 31` for divisor 1.
# For s32/s64, second is MagicShiftAndSign (the zero-extended `more` byte).
_LIBDIVIDE = [
    ("s32", 0, 1073741823, 0),
    ("s32", 1, 0, 0),
    ("s32", -1, 0, 128),
    ("s32", 2, 0, 1),
    ("s32", 3, -1431655765, 65),
    ("s32", 5, -858993459, 66),
    ("s32", 6, -1431655765, 66),
    ("s32", 7, -1840700269, 66),
    ("s32", -5, -858993459, 194),
    ("s32", 15, -2004318071, 67),
    ("s32", 16, 0, 4),
    ("s32", 17, -252645135, 68),
    ("s32", 31, -2078209981, 68),
    ("s32", 32, 0, 5),
    ("s32", 100, -1546188226, 70),
    ("s32", 127, -2130574327, 70),
    ("s32", 128, 0, 7),
    ("s32", 255, -2139062143, 71),
    ("s32", 256, 0, 8),
    ("s32", 1000, -2095944040, 73),
    ("s32", 1024, 0, 10),
    ("s32", 2147483647, -2147483646, 94),
    ("s32", -2147483648, 0, 159),
    ("s32", -2147483647, -2147483646, 222),
    ("s32", 9, -477218588, 67),
    ("s32", 10, -858993459, 67),
    ("u32", 0, 2147483647, 0),
    ("u32", 1, 0, -2147483648),
    ("u32", 2, 0, 0),
    ("u32", 3, 1431655766, 1),
    ("u32", 4, 0, 1),
    ("u32", 5, 2576980378, 2),
    ("u32", 6, 1431655766, 2),
    ("u32", 7, 613566757, 2),
    ("u32", 16, 0, 3),
    ("u32", 17, 3789677026, 4),
    ("u32", 100, 1202590843, 6),
    ("u32", 2147483648, 0, 30),
    ("u32", 4294967295, 2, 31),
    ("s64", 0, 4611686018427387903, 0),
    ("s64", 1, 0, 0),
    ("s64", -1, 0, 128),
    ("s64", 2, 0, 1),
    ("s64", 3, -6148914691236517205, 65),
    ("s64", 7, -7905747460161236406, 66),
    ("s64", 16, 0, 4),
    ("s64", 100, -6640827866535438581, 70),
    ("s64", 4294967296, 0, 32),
    ("s64", 4294967297, -4294967295, 96),
    ("s64", -5, -3689348814741910323, 194),
    ("s64", -9223372036854775808, 0, 191),
    ("s64", 9223372036854775807, -9223372036854775806, 126),
]

# signed_magic_div results, including divisor 0 (not a truncating division).
_DIV = [
    (0, 0, 0),
    (1, 0, 1),
    (1, 1, 1),
    (5, 3, 1),
    (32, 2, 16),
    (-20, 6, -3),
    (7, 0, 8),
]

# evaluate_wgm(m, n, tile_m, tile_n, wgm) in WGM_ROLE_ORDER.
_WGM = [
    ((4096, 4096, 128, 256, 2), [16, 0, 0, 2, 1073741823, 1, 0, 4, 0, 9, 16]),
    ((0, 0, 128, 32, 2), [0, 1073741823, 0, 2, 1073741823, 1, 1073741823, 0, 0, 0, 0]),
    ((1, 1, 16, 256, 1), [1, 0, 0, 1, 1073741823, 0, 0, 0, 0, 0, 1]),
    ((2147483647, 1048576, 32, 64, 2), [0, 1073741823, 0, 2, 1073741823, 1, 0, 14, 0, 0, 33554432]),
    ((17, 19, 256, 256, 6), [0, 1073741823, -1431655765, 6, 0, 66, 0, 0, 0, 0, 0]),
    ((5, 0, 64, 64, 2), [0, 1073741823, 0, 2, 0, 1, 1073741823, 0, 0, 0, 0]),
]


def _python_magic(kind, divisor):
    if kind == "s32":
        return magic_multiple_s32(divisor), magic_shift_and_sign_s32(divisor)
    if kind == "u32":
        return magic_multiple_u32(divisor), s32(magic_shifts_u32(divisor))
    if kind == "s64":
        return magic_multiple_s64(divisor), magic_shift_and_sign_s64(divisor)
    raise AssertionError(kind)


def test_magic_matches_rocroller_libdivide_edges():
    """Divisor 0 and 1 take rocRoller's special cases, not a raw libdivide call."""
    assert magic_multiple_s32(0) == 1073741823
    assert magic_shift_and_sign_s32(0) == 0
    assert magic_multiple_s32(1) == 0
    assert magic_shift_and_sign_s32(1) == 0
    assert magic_multiple_u32(0) == 2147483647
    assert magic_multiple_u32(1) == 0
    assert magic_shifts_u32(0) == 0
    assert magic_shifts_u32(1) == 1 << 31
    assert magic_multiple_s64(0) == 4611686018427387903
    assert magic_shift_and_sign_s64(0) == 0
    assert magic_multiple_s64(1) == 0
    assert magic_shift_and_sign_s64(1) == 0


@pytest.mark.parametrize("kind,divisor,magic,second", _LIBDIVIDE)
def test_magic_matches_libdivide_branchfree(kind, divisor, magic, second):
    assert _python_magic(kind, divisor) == (magic, second)


@pytest.mark.parametrize("numer,denom,expected", _DIV)
def test_signed_magic_div_matches_rocroller_tree(numer, denom, expected):
    assert magic_div_s32(numer, denom) == expected


@pytest.mark.parametrize("sizes,expected", _WGM)
def test_wgm_kernargs_follow_the_hoisted_formula(sizes, expected):
    got = evaluate_wgm(*sizes)
    assert [got[name] for name in WGM_ROLE_ORDER] == expected


def _shipped_rocroller_custom_kernels():
    """(name, CustomKernel) for every shipped kernel of rocRoller origin."""
    root = Path(__file__).resolve().parents[2] / "CustomKernels" / "rocroller"
    for path in sorted(root.glob("*.s")):
        config = readCustomKernelConfig(path.stem, directory=str(root))
        if config.get("Source", {}).get("Origin") == "rocroller":
            yield path.stem, config["CustomKernel"]


def test_shipped_wgm_kernels_declare_the_eleven_roles():
    # A kernel maps workgroups when its custom.config declares the
    # RR_WorkgroupMapping kernarg: that is what the launch reads, so it is what
    # this checks, rather than the name or rocRoller's expression trees.
    wgmKernels = []
    namedWgm = []
    for name, customKernel in _shipped_rocroller_custom_kernels():
        if "_WGM_" in name:
            namedWgm.append(name)
        roles = [arg["semantic"] for arg in customKernel["args"]
                 if arg["semantic"] in WGM_ROLE_ORDER]
        if "RR_WorkgroupMapping" not in roles:
            assert roles == [], name
            continue
        wgmKernels.append(name)
        # Flattened workgroup X. These kernels do not read workgroup Y.
        assert customKernel["grid"] == ["TilesXY", "One", "Batch"], name
        # The launch fills each kernarg by its semantic, so the eleven roles may
        # come in any order, but each exactly once.
        assert sorted(roles) == sorted(WGM_ROLE_ORDER), name
    # Captures with workgroup mapping carry the _WGM_ name suffix, and only those.
    assert namedWgm, "no _WGM_ kernel found; the kernels or their naming moved"
    assert wgmKernels == namedWgm


def test_wgm_expression_classifier_reads_the_tree_not_the_suffix():
    wgm = {
        "type": "CommandArgument",
        "name": "WGM",
        ".size": 4,
        ".value_kind": "by_value",
    }
    assert classify_wgm_expression({"type": "CommandArgument", "name": "WGM"}) == "RR_WorkgroupMapping"
    meta = {
        ".name": "MagicMultiple_22",
        ".size": 4,
        ".value_kind": "by_value",
        ".variableType": {"dataType": "Int32"},
        ".expression": {"type": "MagicMultiple", "arg": wgm},
    }
    assert _metadataArgToCustomArg(meta) == {"type": "int32", "semantic": "RR_MagicMultipleWgm"}


def _lit(value, kind="Int64"):
    return {"type": "LiteralValue." + kind, "dataType": kind, "value": value}


def _hoisted_num_tiles(size_name, tile, magic=None, sign=0):
    """The branchfree s64 ceil(size / tile) tree a non-power-of-two macrotile hoists.

    Spelled as gfx950's 256x192 MXFP4 WGM kernel spells BitwiseXor_18: the
    size read through its kernel argument, the 1 and the shift as UInt32.
    """
    shift = magic_shift_and_sign_s64(tile) & 0x3F
    magic = magic_multiple_s64(tile) if magic is None else magic
    size = {
        "type": "Kernel Argument",
        "name": size_name + "_4",
        "expression": {"type": "CommandArgument", "name": size_name},
    }
    x = {
        "type": "Subtract",
        "lhs": {"type": "Add", "lhs": size, "rhs": _lit(tile)},
        "rhs": _lit(1, "UInt32"),
    }
    u = {"type": "Add", "lhs": {"type": "MultiplyHigh", "lhs": x, "rhs": _lit(magic)}, "rhs": x}
    corr = {
        "type": "BitwiseAnd",
        "lhs": {"type": "ArithmeticShiftR", "lhs": u, "rhs": _lit(63)},
        "rhs": _lit(1 << shift),
    }
    q = {
        "type": "ArithmeticShiftR",
        "lhs": {"type": "Add", "lhs": u, "rhs": corr},
        "rhs": _lit(shift, "UInt32"),
    }
    return {"type": "BitwiseXor", "lhs": q, "rhs": _lit(sign)}


@pytest.mark.parametrize("tile", [96, 128, 160, 192, 256, 320])
def test_num_tiles_s64_is_ceil_div_without_int32_truncation(tile):
    for size in (0, 1, tile - 1, tile, tile + 1, 4096, (1 << 31) + 5, (1 << 40) + 3):
        assert num_tiles_s64(size, tile) == (size + tile - 1) // tile
    with pytest.raises(ValueError):
        num_tiles_s64(128, 0)


@pytest.mark.parametrize("tile", [96, 160, 192, 320])
def test_branchfree_tree_with_libdivide_magic_is_the_ceil_div(tile):
    magic = magic_multiple_s64(tile)
    shift = magic_shift_and_sign_s64(tile) & 0x3F
    for size in list(range(0, 4 * tile)) + [1 << 20, (1 << 31) - 1, 1 << 31, (1 << 52) - 1]:
        assert evaluate_branchfree_tree(size, tile, magic, shift, 1 << shift) == num_tiles_s64(size, tile)


def test_hoisted_tile_counts_classify_by_the_size_they_read():
    # The magic that kernel carries for 192.
    assert magic_multiple_s64(192) == -6148914691236517205
    assert branchfree_divisor(_hoisted_num_tiles("Tensor_0_size_0", 192)) == (192, "Tensor_0_size_0")
    assert classify_wgm_expression(_hoisted_num_tiles("Tensor_0_size_0", 192)) == "RR_NumTilesM"
    assert classify_wgm_expression(_hoisted_num_tiles("Tensor_2_size_1", 192)) == "RR_NumTilesN"
    meta = {
        ".name": "BitwiseXor_7",
        ".size": 8,
        ".value_kind": "by_value",
        ".variableType": {"dataType": "Int64"},
        ".expression": _hoisted_num_tiles("Tensor_2_size_1", 192),
    }
    assert _metadataArgToCustomArg(meta) == {"type": "int64", "semantic": "RR_NumTilesN"}


def test_hoisted_tile_count_decode_rejects_other_divisions():
    # A K tile count is not a launch semantic.
    assert classify_wgm_expression(_hoisted_num_tiles("Tensor_0_size_1", 192)) is None
    # A negative divisor's sign term.
    assert branchfree_divisor(_hoisted_num_tiles("Tensor_0_size_0", 192, sign=-1)) is None
    # 160's magic does not divide by 192.
    wrong = _hoisted_num_tiles("Tensor_0_size_0", 192, magic=magic_multiple_s64(160))
    assert branchfree_divisor(wrong) is None
    assert classify_wgm_expression(wrong) is None


# The strides and sizes each layout's gfx950 MXFP8 64x64x128 kernel reads, as
# JIT-compiled by rocRoller, and the semantic each must launch with.
_NON_TN_COMMAND_ARGUMENTS = {
    "NN": {
        "Tensor_0_stride_1": "StrideA0",
        "Tensor_0_size_1": "SizeSum",
        "Tensor_2_stride_1": "StrideB0",
        "Tensor_2_size_1": "SizeFree1",
        "Tensor_4_stride_1": "StrideScaleA0",
        "Tensor_4_size_1": "SizeSumScaleA",
        "Tensor_7_stride_1": "StrideScaleB0",
        "Tensor_10_stride_1": "StrideC0",
        "Tensor_21_stride_1": "StrideD0",
    },
    "NT": {
        "Tensor_0_stride_1": "StrideA0",
        "Tensor_0_size_1": "SizeSum",
        "Tensor_2_stride_0": "StrideB0",
        "Tensor_2_size_1": "SizeFree1",
        "Tensor_4_stride_1": "StrideScaleA0",
        "Tensor_4_size_1": "SizeSumScaleA",
        "Tensor_7_stride_0": "StrideScaleB0",
        "Tensor_10_stride_1": "StrideC0",
        "Tensor_21_stride_1": "StrideD0",
    },
    "TT": {
        "Tensor_0_stride_0": "StrideA0",
        "Tensor_0_size_0": "SizeFree0",
        "Tensor_0_size_1": "SizeSum",
        "Tensor_2_stride_0": "StrideB0",
        "Tensor_2_size_1": "SizeFree1",
        "Tensor_4_stride_0": "StrideScaleA0",
        "Tensor_4_size_1": "SizeSumScaleA",
        "Tensor_7_stride_0": "StrideScaleB0",
        "Tensor_10_stride_1": "StrideC0",
        "Tensor_21_stride_1": "StrideD0",
    },
}


@pytest.mark.parametrize("layout", sorted(_NON_TN_COMMAND_ARGUMENTS))
def test_non_tn_kernels_command_arguments_map_to_semantics(layout):
    for index, (name, semantic) in enumerate(_NON_TN_COMMAND_ARGUMENTS[layout].items()):
        meta = {
            ".name": f"{name}_{index}",
            ".size": 8,
            ".value_kind": "by_value",
            ".expression": {
                "type": "CommandArgument",
                "name": name,
                "variableType": {"dataType": "Int64", "pointerType": "Value"},
            },
        }
        assert _metadataArgToCustomArg(meta) == {"type": "int64", "semantic": semantic}, name
