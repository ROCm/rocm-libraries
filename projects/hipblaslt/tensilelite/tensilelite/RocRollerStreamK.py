# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch values for rocRoller Stream-K custom kernels.

A rocRoller Stream-K kernel hoists its own work decomposition into kernel
arguments: how many tile-iterations go to the Stream-K phase, how many to
the data-parallel phase, and libdivide constants for the two divisors the
kernel divides by at runtime. Tensile's own Stream-K decomposition
(computeStreamKDecisions) is a *different* partitioning and must not be
substituted here -- the kernel's prologue recomputes nothing, so a value
that disagrees with the kernel's own arithmetic silently corrupts the
fixup pass and leaves the flag region dirty.

The formulas below are transcribed from the ``.expression`` trees that
rocRoller serialises into each kernel's ``.amdgpu_metadata``. Those trees
are the contract; Tests/unit/test_rocroller_streamk.py evaluates them
generically and asserts this module agrees, so a rocRoller change that
alters the decomposition is caught as a test failure rather than as
silently wrong numerics.

tensilelite/tests/RocRollerStreamKKernargs_test.cpp checks the C++ launch
header against the same vectors; edit both together.
"""

from __future__ import annotations

from .RocRollerWgm import magic_multiple_u32, magic_shifts_u32, u32

# Command arguments the Stream-K launch tree reads. Tensor_0 is A {M, K};
# Tensor_2 is B {K, N}. numWGs is supplied by the host, not derived.
_CA_M = "Tensor_0_size_0"
_CA_K = "Tensor_0_size_1"
_CA_N = "Tensor_2_size_1"
_CA_NUM_WGS = "numWGs"

# The ten hoisted arguments. This tuple names the complete set and fixes a
# reference order for test vectors; it is NOT the kernarg layout. Each
# kernel's .amdgpu_metadata declares its own order and the launcher
# switches per argument on the semantic.
SK_ROLE_ORDER = (
    "RR_SKNumTiles1",
    "RR_SKNumWGs",
    "RR_SKNumSKTiles",
    "RR_SKNumSKTilesPerWG",
    "RR_SKNumDPTiles",
    "RR_SKNumDPTilesPerWG",
    "RR_SKMagicMultipleIters",
    "RR_SKMagicShiftsIters",
    "RR_SKMagicMultipleNumTiles1",
    "RR_SKMagicShiftsNumTiles1",
)

# The two scratch pointers. Kept out of SK_ROLE_ORDER because they are
# addresses the launcher binds, not values it computes.
SK_POINTER_ROLES = (
    "RR_SKAddressPartials",
    "RR_SKAddressFlags",
)

# Stream-K kernarg names rocRoller emits without a numeric suffix. Unlike
# the MagicMultiple_/MagicShifts_ arguments these are stable across tiles
# and data types, so matching on the name is safe and is what the
# _METADATA_NAME_TO_SEMANTIC table in CustomKernels.py does.
#
# numWGs is listed here deliberately rather than by widening the
# CommandArgument fallback: its expression is wrapped in a Convert, so the
# fallback would not see a CommandArgument root, and the name is one
# character away from the unrelated Tensile "numWG" key.
STABLE_NAME_TO_SEMANTIC = {
    "numTiles1": "RR_SKNumTiles1",
    "numWGs": "RR_SKNumWGs",
    "numSKTiles": "RR_SKNumSKTiles",
    "numSKTilesPerWG": "RR_SKNumSKTilesPerWG",
    "numDPTiles": "RR_SKNumDPTiles",
    "numDPTilesPerWG": "RR_SKNumDPTilesPerWG",
    # rocRoller names its two scratch buffers by their zeroing contract, not
    # by role. SCRATCH_None is the partial-tile accumulator and
    # SCRATCH_ZeroedBeforeAndAfter is the Stream-K flag region: zero on
    # entry, and the kernel restores it to zero before it exits.
    #
    # These deliberately do NOT reuse Tensile's AddressWorkspace /
    # AddressFlags. AddressFlags appends a null pointer whenever the
    # resolved reduction is parallel, and the region it points at is only
    # the private per-stream Stream-K block when sizeMapping.isStreamK()
    # -- which is false for a handwritten custom kernel, so it would
    # silently hand over the GSU region that is shared across streams.
    # The dedicated semantics bind unconditionally and correctly.
    "SCRATCH_None": "RR_SKAddressPartials",
    "SCRATCH_ZeroedBeforeAndAfter": "RR_SKAddressFlags",
}


def _ceil_div_u32(size, tile):
    """``(size + tile - 1) / tile`` in int64, truncated to uint32.

    Matches the serialised Add/Subtract/Divide tree wrapped in a Convert to
    UInt32. The divisor is not required to be a power of two: MT_N is 192
    for several shipped tiles, and rocRoller emits a real Divide there.
    """
    tile = int(tile)
    if tile <= 0:
        raise ValueError(f"macrotile {tile} must be positive")
    return u32((int(size) + tile - 1) // tile)


def iters_per_tile(k, tile_k):
    """``K / MT_K``, truncating. rocRoller does not round this one up."""
    tile_k = int(tile_k)
    if tile_k <= 0:
        raise ValueError(f"macrotile K {tile_k} must be positive")
    return u32(int(k) // tile_k)


def evaluate_streamk(m, n, k, tile_m, tile_n, tile_k, num_wgs):
    """The ten hoisted kernargs, keyed by CustomArgSemantic name.

    Every arithmetic node downstream of the tile counts is UInt32 in the
    serialised tree, so products, differences and the ``numWGs - 1``
    comparison all wrap at 32 bits here too.
    """
    if int(num_wgs) <= 0:
        raise ValueError(f"Stream-K grid size {num_wgs} must be positive")

    nt_m = _ceil_div_u32(m, tile_m)
    nt_n = _ceil_div_u32(n, tile_n)
    tiles = u32(nt_m * nt_n)
    iters = iters_per_tile(k, tile_k)
    wgs = u32(num_wgs)

    # Fewer tiles than workgroups: every tile is split Stream-K and there is
    # no data-parallel phase at all.
    if tiles > u32(wgs - 1):
        sk_tiles = u32(u32(u32(tiles % wgs) + wgs) * iters)
        full_rounds = u32(u32(tiles // wgs) - 1)
        dp_tiles = u32(u32(full_rounds * wgs) * iters)
        dp_tiles_per_wg = u32(full_rounds * iters)
    else:
        sk_tiles = u32(tiles * iters)
        dp_tiles = 0
        dp_tiles_per_wg = 0

    sk_tiles_per_wg = u32(u32(sk_tiles + u32(wgs - 1)) // wgs)

    return {
        "RR_SKNumTiles1": nt_n,
        "RR_SKNumWGs": wgs,
        "RR_SKNumSKTiles": sk_tiles,
        "RR_SKNumSKTilesPerWG": sk_tiles_per_wg,
        "RR_SKNumDPTiles": dp_tiles,
        "RR_SKNumDPTilesPerWG": dp_tiles_per_wg,
        "RR_SKMagicMultipleIters": magic_multiple_u32(iters),
        "RR_SKMagicShiftsIters": magic_shifts_u32(iters),
        "RR_SKMagicMultipleNumTiles1": magic_multiple_u32(nt_n),
        "RR_SKMagicShiftsNumTiles1": magic_shifts_u32(nt_n),
    }


def _command_argument_names(node, found=None):
    if found is None:
        found = set()
    if isinstance(node, dict):
        if node.get("type") == "CommandArgument":
            found.add(node.get("name"))
        for value in node.values():
            _command_argument_names(value, found)
    elif isinstance(node, list):
        for value in node:
            _command_argument_names(value, found)
    return found


def classify_streamk_expression(expr, dataType=None):
    """Map one rocRoller ``.expression`` onto a Stream-K magic semantic.

    Only the four libdivide arguments are classified here. Their kernarg
    names carry a numeric suffix that shifts between kernels -- the same
    constant is MagicMultiple_28 on the 16x16x128 tile and
    MagicMultiple_29 on 256x192x128 -- so the name cannot be used. The
    root opcode gives MagicMultiple vs MagicShifts, and the set of command
    arguments underneath gives the divisor: numTiles1 is derived from N,
    the iteration count from K.

    Everything else Stream-K hoists has a stable name and is resolved by
    STABLE_NAME_TO_SEMANTIC before this is reached.
    """
    if not isinstance(expr, dict):
        return None
    root = expr.get("type")
    if root == "MagicMultiple":
        # MagicMultiple is the one root a workgroup-mapping kernel shares
        # with Stream-K, and over the same divisor: both take the N tile
        # count. They are not the same constant -- workgroup mapping uses
        # the signed libdivide form and Stream-K the unsigned one -- so
        # they must not be merged. rocRoller declares them with different
        # result types, which is the only stable way to tell them apart.
        # Stream-K's shift constant has its own root (MagicShifts, not
        # MagicShiftAndSign) and needs no such test.
        if dataType != "UInt32":
            return None
        prefix = "RR_SKMagicMultiple"
    elif root == "MagicShifts":
        prefix = "RR_SKMagicShifts"
    else:
        return None

    names = _command_argument_names(expr)
    if names == {_CA_N}:
        return prefix + "NumTiles1"
    if names == {_CA_K}:
        return prefix + "Iters"
    return None


_ARG_TYPES = {
    "Int32": "int32",
    "UInt32": "uint32",
    "Int64": "int64",
    "Float": "float32",
}


def custom_arg_for_metadata(meta_arg):
    """custom.config arg dict for one amdgpu ``.args`` entry, or None."""
    dataType = (meta_arg.get(".variableType") or {}).get("dataType")
    semantic = classify_streamk_expression(meta_arg.get(".expression"), dataType)
    if semantic is None:
        return None
    # Each constant is four bytes; take the declared type so the slot the
    # launcher writes matches the one the kernel reads. MagicShifts is
    # declared Int32 even though its value is an unsigned shift count.
    return {"type": _ARG_TYPES.get(dataType, "uint32"), "semantic": semantic}


def is_streamk_metadata(args):
    """True when a custom.config arg list carries the Stream-K kernargs.

    The kernel name is not consulted; presence of the decomposition
    arguments is the signal, the same way _hasRocRollerWgmArgs works.
    """
    roles = frozenset(SK_ROLE_ORDER) | frozenset(SK_POINTER_ROLES)
    return any(arg.get("semantic") in roles for arg in args)
