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

# The eleven hoisted arguments. This tuple names the complete set and fixes a
# reference order for test vectors; it is NOT the kernarg layout. Each kernel's
# .amdgpu_metadata declares its own order (the pure-MXFP4 32x32x128 kernels
# permute it) and the launcher switches per argument on the semantic.
WGM_ROLE_ORDER = (
    "RR_QuotientTilesByBlock",
    "RR_MagicMultipleWgmMainBlock",
    "RR_MagicMultipleWgm",
    "RR_WorkgroupMapping",
    "RR_MagicMultipleWgmTail",
    "RR_MagicShiftAndSignWgm",
    "RR_MagicMultipleNumTilesN",
    "RR_MagicShiftAndSignNumTilesN",
    "RR_MagicShiftAndSignWgmTail",
    "RR_MagicShiftAndSignWgmMainBlock",
    "RR_QuotientTilesMByWgm",
)

# Two further hoisted arguments, present only when a macrotile is not a power
# of two (gfx950 ships 192-wide MXFP4 tiles). rocRoller cannot lower the tile
# count to a shift there, so it hoists the division result instead of
# recomputing it inline. They are deliberately NOT in WGM_ROLE_ORDER: that
# tuple names the eleven arguments every WGM kernel carries, and these two
# appear in addition to them, never instead of one.
NON_POW2_TILE_ROLES = (
    "RR_NumTilesM",
    "RR_NumTilesN",
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


def num_tiles_s64(size, tile):
    """int64 tile count. Both of rocRoller's lowerings evaluate to this.

    *Power-of-two macrotile.* The division becomes an arithmetic shift right
    by ``log2(tile)`` and nothing is hoisted -- the kernel recomputes the
    count inline from the size kernarg it already holds.

    *Any other macrotile* (gfx950 ships 192-wide tiles for MXFP4). There is
    no shift to lower to, so rocRoller emits a libdivide branchfree signed
    64-bit division and **hoists the result into its own kernarg**:

        x = size + tile - 1
        u = mulhi_s64(x, magic) + x
        q = (u + ((u >> 63) & (1 << shift))) >> shift
        q = q ^ sign              # sign is 0 for a positive divisor

    ``branchfree_divisor`` recognises exactly that tree and recovers ``tile``
    from it, so the classifier decodes the expression instead of matching a
    kernarg name. ``evaluate_branchfree_tree`` replays the tree verbatim;
    the unit tests assert it agrees with this ceil-div.
    """
    tile = int(tile)
    if tile <= 0:
        raise ValueError(f"macrotile {tile} is not positive")
    return -(-int(size) // tile)


def num_tiles(size, tile):
    """int32 tile count, as the hoisted int32 kernargs consume it."""
    return s32(num_tiles_s64(size, tile))


def evaluate_branchfree_tree(size, tile, magic, shift, add):
    """rocRoller's hoisted division, evaluated exactly as the tree spells it.

    Kept next to ``num_tiles_s64`` so the two can be compared in a test
    rather than one being asserted to imply the other.
    """
    x = s64(s64(size + tile) - 1)
    u = s64(_mulhi_s64(x, magic) + x)
    corr = s64(u >> 63) & add
    return s64(s64(u + corr) >> shift)


def _mulhi_s64(a, b):
    return s64((s64(a) * s64(b)) >> 64)


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
        "RR_QuotientTilesByBlock": q_block,
        "RR_MagicMultipleWgmMainBlock": magic_multiple_s32(main),
        "RR_MagicMultipleWgm": magic_multiple_s32(w),
        "RR_WorkgroupMapping": w,
        "RR_MagicMultipleWgmTail": magic_multiple_s32(tail),
        "RR_MagicShiftAndSignWgm": magic_shift_and_sign_s32(w),
        "RR_MagicMultipleNumTilesN": magic_multiple_s32(nt_n),
        "RR_MagicShiftAndSignNumTilesN": magic_shift_and_sign_s32(nt_n),
        "RR_MagicShiftAndSignWgmTail": magic_shift_and_sign_s32(tail),
        "RR_MagicShiftAndSignWgmMainBlock": magic_shift_and_sign_s32(main),
        "RR_QuotientTilesMByWgm": q_m,
        # Present in the kernarg list only when the corresponding macrotile is
        # not a power of two; harmless to compute either way.
        "RR_NumTilesM": num_tiles_s64(m, tile_m),
        "RR_NumTilesN": num_tiles_s64(n, tile_n),
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


def branchfree_divisor(expr):
    """Recover the divisor from a hoisted libdivide branchfree s64 division.

    Returns ``(divisor, command_argument_name)`` when ``expr`` is exactly the
    shape ``num_tiles_s64`` documents, else ``None``. Everything is read out
    of the tree: the divisor appears as the ``size + tile`` literal, the
    shift as the outer ``ArithmeticShiftR`` count, and the correction mask as
    ``1 << shift``. Shape and arithmetic are both checked, so a different
    division that happens to be rooted at ``BitwiseXor`` is rejected rather
    than silently mapped.
    """
    def lit(node, kinds=("LiteralValue.Int64", "LiteralValue.UInt32",
                         "LiteralValue.Int32")):
        if isinstance(node, dict) and node.get("type") in kinds:
            return node.get("value")
        return None

    if not isinstance(expr, dict) or expr.get("type") != "BitwiseXor":
        return None
    if lit(expr.get("rhs")) != 0:          # sign term; non-zero = negative divisor
        return None
    shr = expr.get("lhs")
    if not isinstance(shr, dict) or shr.get("type") != "ArithmeticShiftR":
        return None
    shift = lit(shr.get("rhs"))
    if not isinstance(shift, int) or not 0 < shift < 64:
        return None
    add = shr.get("lhs")
    if not isinstance(add, dict) or add.get("type") != "Add":
        return None
    mask = add.get("rhs")
    if not isinstance(mask, dict) or mask.get("type") != "BitwiseAnd":
        return None
    if lit(mask.get("rhs")) != 1 << shift:
        return None

    # x = size + tile - 1, reached through Subtract(Add(KernArg, tile), 1)
    inner = add.get("lhs")
    if not isinstance(inner, dict) or inner.get("type") != "Add":
        return None
    mulhi = inner.get("lhs")
    if not isinstance(mulhi, dict) or mulhi.get("type") != "MultiplyHigh":
        return None
    magic = lit(mulhi.get("rhs"))
    sub = mulhi.get("lhs")
    if not isinstance(sub, dict) or sub.get("type") != "Subtract":
        return None
    if lit(sub.get("rhs")) != 1:
        return None
    plus = sub.get("lhs")
    if not isinstance(plus, dict) or plus.get("type") != "Add":
        return None
    divisor = lit(plus.get("rhs"))
    if not isinstance(divisor, int) or divisor <= 0:
        return None
    if not isinstance(magic, int):
        return None

    names = set(_command_argument_names(expr))
    if len(names) != 1:
        return None

    # The decode is only trusted if replaying the tree agrees with the plain
    # ceil-div: exact and remainder cases, and sizes large enough that a magic
    # constant outside libdivide's rounding slack would show up. (Inside the
    # slack several magics compute the same function, and any of them is a
    # correct decode -- the divisor is what the classifier needs, not the
    # constant.)
    for size in (1, divisor - 1, divisor, divisor + 1, 7 * divisor,
                 7 * divisor + 1, 1 << 20, (1 << 31) - 1, 1 << 31,
                 (1 << 40) + divisor - 1, (1 << 52) - 1):
        if evaluate_branchfree_tree(size, divisor, magic, shift,
                                    1 << shift) != num_tiles_s64(size, divisor):
            return None
    return divisor, names.pop()


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
        return "RR_WorkgroupMapping"
    if root == "BitwiseXor":
        decoded = branchfree_divisor(expr)
        if decoded is None:
            return None
        _, read = decoded
        if read == _CA_M:
            return "RR_NumTilesM"
        if read == _CA_N:
            return "RR_NumTilesN"
        return None
    suffix = _role_suffix(names)
    if suffix is None:
        return None
    if root == "MagicMultiple":
        return "RR_MagicMultiple" + suffix
    if root == "MagicShiftAndSign":
        return "RR_MagicShiftAndSign" + suffix
    if root == "Subtract":
        if suffix == "WgmTail":
            return "RR_QuotientTilesMByWgm"
        if suffix == "WgmMainBlock":
            return "RR_QuotientTilesByBlock"
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
            arg_type = "uint32" if semantic.startswith("RR_MagicShiftAndSign") else "int32"
        return {"type": arg_type, "semantic": semantic}
    if expr.get("type") == "CommandArgument":
        mapped = DIRECT_COMMAND_ARGUMENT.get(expr.get("name"))
        if mapped is not None:
            arg_type, semantic = mapped
            return {"type": arg_type, "semantic": semantic}
    return None
