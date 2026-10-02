# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch values for rocRoller workgroup-mapping custom kernels.

rocRoller hoists libdivide branchfree MagicMultiple / MagicShiftAndSign
constants, the workgroup-mapping divisor, and two signed magic quotients
into kernel arguments. hipBLASLt passes WGM = 2 (DEFAULT_WGM). Tensile's
magicNumber() is a different algorithm and is not used here.

The integer formulas match rocRoller Expression_evaluate_unary.cpp,
including the divisor 0 and 1 special cases, and the signed fast-division
tree in FastDivision.cpp.
"""

from __future__ import annotations

_I32_MIN = -2147483648
_I32_MAX = 2147483647
_U32_MASK = 0xFFFFFFFF
_I64_MIN = -9223372036854775808
_I64_MAX = 9223372036854775807
_U64_MASK = 0xFFFFFFFFFFFFFFFF

# libdivide branchfree markers. Shift masks are 31 and 63.
_ADD_MARKER = 0x40
_NEGATIVE_DIVISOR = 0x80
_SHIFT_MASK_32 = 31
_SHIFT_MASK_64 = 63

# Command arguments these kernels read. Tensor_0 is A {M, K}; Tensor_2 is B {K, N}.
_CA_WGM = "WGM"
_CA_M = "Tensor_0_size_0"
_CA_N = "Tensor_2_size_1"

# Order of the eleven hoisted arguments, stable across the gfx950 WGM kernels.
WGM_ROLE_ORDER = (
    "QuotientTilesByBlock",
    "MagicMultipleWgmMainBlock",
    "MagicMultipleWgm",
    "WorkgroupMapping",
    "MagicMultipleWgmTail",
    "MagicShiftAndSignWgm",
    "MagicMultipleNumTilesN",
    "MagicShiftAndSignNumTilesN",
    "MagicShiftAndSignWgmTail",
    "MagicShiftAndSignWgmMainBlock",
    "QuotientTilesMByWgm",
)

# hipBLASLt rocRoller::DEFAULT_WGM. Every shipped expression reads the WGM
# command argument, and the library passes this constant.
DEFAULT_WGM = 2

# Direct command arguments already used by the non-WGM rocRoller kernels.
DIRECT_COMMAND_ARGUMENT = {
    "Tensor_0_stride_0": ("int64", "StrideA0"),
    "Tensor_0_pointer": ("address", "AddressA"),
    "Tensor_0_size_0": ("int64", "SizeFree0"),
    "Tensor_0_size_1": ("int64", "SizeSum"),
    "Tensor_2_stride_1": ("int64", "StrideB0"),
    "Tensor_2_pointer": ("address", "AddressB"),
    "Tensor_2_size_1": ("int64", "SizeFree1"),
    "Tensor_4_stride_0": ("int64", "StrideScaleA0"),
    "Tensor_4_pointer": ("address", "AddressMXScaleA"),
    "Tensor_7_stride_1": ("int64", "StrideScaleB0"),
    "Tensor_7_pointer": ("address", "AddressMXScaleB"),
    "Tensor_10_stride_1": ("int64", "StrideC0"),
    "Tensor_10_pointer": ("address", "AddressC"),
    "Tensor_21_stride_1": ("int64", "StrideD0"),
    "Tensor_21_pointer": ("address", "AddressD"),
    "user_Float_Value_12": ("float32", "Alpha"),
    "user_Float_Value_14": ("float32", "Beta"),
}


def s32(value):
    value &= _U32_MASK
    return value - 0x100000000 if value & 0x80000000 else value


def u32(value):
    return value & _U32_MASK


def s64(value):
    value &= _U64_MASK
    return value - 0x10000000000000000 if value & 0x8000000000000000 else value


def u64(value):
    return value & _U64_MASK


def _clz32(value):
    value = u32(value)
    if value == 0:
        return 32
    return 32 - value.bit_length()


def _clz64(value):
    value = u64(value)
    if value == 0:
        return 64
    return 64 - value.bit_length()


def _div64_32(hi, lo, den):
    """libdivide_64_div_32_to_32. Quotient fits in 32 bits."""
    n = (u32(hi) << 32) | u32(lo)
    q = n // u32(den)
    rem = n - q * u32(den)
    return u32(q), u32(rem)


def _div128_64(hi, lo, den):
    """libdivide_128_div_64_to_64 for a quotient that fits in 64 bits."""
    n = (u64(hi) << 64) | u64(lo)
    den = u64(den)
    q = n // den
    rem = n - q * den
    return u64(q), u64(rem)


def _s32_branchfree(divisor):
    """libdivide_s32_branchfree_gen, without the divisor-0 error."""
    d = s32(divisor)
    ud = u32(d)
    abs_d = u32(-ud) if d < 0 else ud
    floor_log = 31 - _clz32(abs_d)
    if (abs_d & u32(abs_d - 1)) == 0:
        more = floor_log | (_NEGATIVE_DIVISOR if d < 0 else 0)
        return 0, more
    proposed, rem = _div64_32(1 << (floor_log - 1), 0, abs_d)
    proposed = u32(proposed + proposed)
    twice_rem = u32(rem + rem)
    if twice_rem >= abs_d or twice_rem < rem:
        proposed = u32(proposed + 1)
    more = floor_log | _ADD_MARKER
    proposed = u32(proposed + 1)
    magic = s32(proposed)
    if d < 0:
        more |= _NEGATIVE_DIVISOR
    return magic, more


def _u32_branchfree(divisor):
    """libdivide_u32_branchfree_gen. Caller must not pass 0 or 1."""
    d = u32(divisor)
    floor_log = 31 - _clz32(d)
    if (d & u32(d - 1)) == 0:
        return 0, u32(floor_log - 1) & _SHIFT_MASK_32
    proposed, rem = _div64_32(1 << floor_log, 0, d)
    proposed = u32(proposed + proposed)
    twice_rem = u32(rem + rem)
    if twice_rem >= d or twice_rem < rem:
        proposed = u32(proposed + 1)
    magic = u32(1 + proposed)
    # branchfree_gen keeps only the shift, dropping LIBDIVIDE_ADD_MARKER.
    return magic, floor_log & _SHIFT_MASK_32


def _s64_branchfree(divisor):
    """libdivide_s64_branchfree_gen, without the divisor-0 error."""
    d = s64(divisor)
    ud = u64(d)
    abs_d = u64(-ud) if d < 0 else ud
    floor_log = 63 - _clz64(abs_d)
    if (abs_d & u64(abs_d - 1)) == 0:
        more = floor_log | (_NEGATIVE_DIVISOR if d < 0 else 0)
        return 0, more
    proposed, rem = _div128_64(1 << (floor_log - 1), 0, abs_d)
    proposed = u64(proposed + proposed)
    twice_rem = u64(rem + rem)
    if twice_rem >= abs_d or twice_rem < rem:
        proposed = u64(proposed + 1)
    more = floor_log | _ADD_MARKER
    proposed = u64(proposed + 1)
    magic = s64(proposed)
    if d < 0:
        more |= _NEGATIVE_DIVISOR
    return magic, more


def magic_multiple_u32(divisor):
    """rocRoller MagicMultiple for uint32, including 0 and 1."""
    arg = u32(divisor)
    if arg == 0:
        return u32(0xFFFFFFFF // 2)
    if arg == 1:
        return 0
    magic, _ = _u32_branchfree(arg)
    return magic


def magic_shifts_u32(divisor):
    """rocRoller MagicShifts for uint32, including 0 and 1."""
    arg = u32(divisor)
    if arg == 0:
        return 0
    if arg == 1:
        return u32(1 << 31)
    _, shifts = _u32_branchfree(arg)
    return shifts & _SHIFT_MASK_32


def magic_multiple_s32(divisor):
    """rocRoller MagicMultiple for int32. Divisor 0 is special-cased; 1 is not."""
    arg = s32(divisor)
    if arg == 0:
        return s32(_I32_MAX // 2)
    magic, _ = _s32_branchfree(arg)
    return magic


def magic_shift_and_sign_s32(divisor):
    """rocRoller MagicShiftAndSign for int32. The uint8 more field, zero-extended."""
    arg = s32(divisor)
    if arg == 0:
        return 0
    _, more = _s32_branchfree(arg)
    return u32(more)


def magic_multiple_s64(divisor):
    """rocRoller MagicMultiple for int64."""
    arg = s64(divisor)
    if arg == 0:
        return s64(_I64_MAX // 2)
    magic, _ = _s64_branchfree(arg)
    return magic


def magic_shift_and_sign_s64(divisor):
    """rocRoller MagicShiftAndSign for int64."""
    arg = s64(divisor)
    if arg == 0:
        return 0
    _, more = _s64_branchfree(arg)
    return u32(more)


def _mul_s32(lhs, rhs):
    return s32(s32(lhs) * s32(rhs))


def _mulhi_s32(lhs, rhs):
    return s32((s32(lhs) * s32(rhs)) >> 32)


def _asr_s32(value, count):
    count &= 31
    return s32(s32(value) >> count)


def _shl_s32(value, count):
    count &= 31
    return s32(s32(value) << count)


def magic_div_s32(numerator, divisor):
    """Signed division by a libdivide branchfree constant, matching rocRoller's tree.

    Divisor 0 uses MagicMultiple's special case rather than raising. The
    result is whatever that tree produces, not a truncating C division.
    """
    magic = magic_multiple_s32(divisor)
    bitfield = magic_shift_and_sign_s32(divisor)
    q = s32(_mulhi_s32(numerator, magic) + s32(numerator))
    sign_of_q = _asr_s32(q, 31)
    shifts = bitfield & _SHIFT_MASK_32
    one_shifted = _shl_s32(1, shifts)
    magic_is_pow2 = -1 if s32(magic) == 0 else 0
    handle = s32(q + s32(sign_of_q & s32(one_shifted + magic_is_pow2)))
    shifted = _asr_s32(handle, shifts)
    sign = _asr_s32(_shl_s32(s32(bitfield), 24), 31)
    return s32(s32(shifted ^ sign) - sign)


def num_tiles(size, tile):
    """int32 ceil-div of a tensor size by a power-of-two macrotile.

    rocRoller lowers unsigned division by a translate-time power of two to
    an arithmetic shift, then converts the int64 result to int32.
    """
    tile = u32(tile)
    if tile == 0 or (tile & (tile - 1)) != 0:
        raise ValueError(f"macrotile {tile} is not a positive power of two")
    log = tile.bit_length() - 1
    shifted = (int(size) + int(tile) - 1) >> log
    return s32(shifted)


def evaluate_wgm(m, n, tile_m, tile_n, wgm=DEFAULT_WGM):
    """The eleven hoisted kernargs, keyed by CustomArgSemantic name.

    Products are int32 wrapping products, matching rocRoller's int32
    workgroup-mapping arithmetic. ``tile_m * tile_n`` is not cancelled
    algebraically, so an overflowing ``ntM * ntN`` stays overflowing.
    """
    nt_m = num_tiles(m, tile_m)
    nt_n = num_tiles(n, tile_n)
    w = s32(wgm)
    total = _mul_s32(nt_m, nt_n)
    block = _mul_s32(w, nt_n)
    q_block = magic_div_s32(total, block)
    main = _mul_s32(q_block, block)
    q_m = magic_div_s32(nt_m, w)
    tail = s32(nt_m - _mul_s32(q_m, w))
    return {
        "QuotientTilesByBlock": q_block,
        "MagicMultipleWgmMainBlock": magic_multiple_s32(main),
        "MagicMultipleWgm": magic_multiple_s32(w),
        "WorkgroupMapping": w,
        "MagicMultipleWgmTail": magic_multiple_s32(tail),
        "MagicShiftAndSignWgm": magic_shift_and_sign_s32(w),
        "MagicMultipleNumTilesN": magic_multiple_s32(nt_n),
        "MagicShiftAndSignNumTilesN": magic_shift_and_sign_s32(nt_n),
        "MagicShiftAndSignWgmTail": magic_shift_and_sign_s32(tail),
        "MagicShiftAndSignWgmMainBlock": magic_shift_and_sign_s32(main),
        "QuotientTilesMByWgm": q_m,
    }


def _command_argument_names(node, found=None):
    if found is None:
        found = []
    if isinstance(node, dict):
        if node.get("type") == "CommandArgument":
            found.append(node.get("name"))
        for value in node.values():
            _command_argument_names(value, found)
    elif isinstance(node, list):
        for value in node:
            _command_argument_names(value, found)
    return found


def _role_suffix(names):
    names = frozenset(names)
    if names == frozenset((_CA_WGM,)):
        return "Wgm"
    if names == frozenset((_CA_N,)):
        return "NumTilesN"
    if names == frozenset((_CA_M, _CA_WGM)):
        return "WgmTail"
    if names == frozenset((_CA_M, _CA_N, _CA_WGM)):
        return "WgmMainBlock"
    return None


def classify_wgm_expression(expr):
    """Map one rocRoller ``.expression`` onto a WGM CustomArgSemantic, or None.

    The eleven roles are distinguished by the root opcode and the set of
    command arguments the tree reads. Numeric kernel-argument suffixes are
    not stable across kernels, so the name is not used.
    """
    if not isinstance(expr, dict):
        return None
    root = expr.get("type")
    names = _command_argument_names(expr)
    if root == "CommandArgument" and expr.get("name") == _CA_WGM:
        return "WorkgroupMapping"
    suffix = _role_suffix(names)
    if suffix is None:
        return None
    if root == "MagicMultiple":
        return "MagicMultiple" + suffix
    if root == "MagicShiftAndSign":
        return "MagicShiftAndSign" + suffix
    if root == "Subtract":
        if suffix == "WgmTail":
            return "QuotientTilesMByWgm"
        if suffix == "WgmMainBlock":
            return "QuotientTilesByBlock"
    return None


def _variable_arg_type(meta_arg):
    data_type = (meta_arg.get(".variableType") or {}).get("dataType")
    return {
        "Int32": "int32",
        "UInt32": "uint32",
        "Int64": "int64",
        "Float": "float32",
    }.get(data_type)


def custom_arg_for_metadata(meta_arg):
    """custom.config arg dict for one amdgpu ``.args`` entry, or None."""
    expr = meta_arg.get(".expression")
    if not isinstance(expr, dict):
        return None
    semantic = classify_wgm_expression(expr)
    if semantic is not None:
        arg_type = _variable_arg_type(meta_arg)
        if arg_type is None:
            arg_type = "uint32" if semantic.startswith("MagicShiftAndSign") else "int32"
        return {"type": arg_type, "semantic": semantic}
    if expr.get("type") == "CommandArgument":
        mapped = DIRECT_COMMAND_ARGUMENT.get(expr.get("name"))
        if mapped is not None:
            arg_type, semantic = mapped
            return {"type": arg_type, "semantic": semantic}
    return None
