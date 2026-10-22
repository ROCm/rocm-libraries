# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The MMA catalog's two layout descriptions agree, and agree with the independent arbiters.

Every catalog row describes its register layout twice: as hardware-notation strings (``a_layout``,
``b_layout``, ``c_d_layout``) and as numeric layout parameters, which the tiling encoder turns into
lane/slot maps. These tests interpret the strings (``layout_descriptor.py``) and require them to equal
the real encoder's maps exactly, for every row -- so neither description can drift from the other. A
multi-block row is compared one block at a time against the encoder's single-block layout; a sparse
row's A operand is compared in the compressed K coordinates (``KP``) its string uses.

Where an independent arbiter exists, the tests also pin the rows to it: rocke core's gfx1250 scaled
operand layout, and point facts read from the public AMD Matrix Instruction Calculator.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from rocke.core.arch.wmma_scale import scaled_matrix_layout
from rocke.core.dtypes import dtype_info
from rocke.helpers.tiling.mma.warp_encoding import (
    a_warp_encoding,
    b_warp_encoding,
    c_warp_encoding,
)
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.helpers.tiling.traits import DEFAULT_TRAITS_PATH, MmaTraits, load_mma_traits

from .layout_descriptor import interpret_layout, operand_layout_strings

_CATALOG = load_mma_traits()
_ROWS = sorted(_CATALOG.by_op_id.values(), key=lambda traits: traits.op_id)
_IDS = [traits.op_id for traits in _ROWS]

Element = tuple[tuple[str, int], ...]
LaneMap = dict[tuple[int, int], Element]


def _encoder_map(encoding, row_dim: str, col_dim: str) -> LaneMap:
    """``(lane, slot) -> element`` of one encoding, ``row_dim`` / ``col_dim`` naming its two axes."""
    mapper = RegisterMapper(encoding)
    mapping = {}
    for lane in range(mapper.num_lanes):
        for slot in range(mapper.num_vector_items):
            row, col = mapper.matrix_coordinates(lane=lane, register=slot)
            mapping[(lane, slot)] = tuple(sorted(((row_dim, row), (col_dim, col))))
    return mapping


def _compressed(mapping: LaneMap) -> LaneMap:
    """A sparse A map in the compressed K coordinates its string uses: two adjacent K elements
    (one 2:4 pair) share a compressed slot, so slot ``s`` holding K ``k`` becomes slot ``s // 2``
    holding ``KP = k // 2``. Both elements of a pair must land on the same compressed element.
    """
    compressed: LaneMap = {}
    for (lane, slot), element in mapping.items():
        coords = dict(element)
        target = tuple(sorted((("M", coords["M"]), ("KP", coords["K"] // 2))))
        assert (
            compressed.setdefault((lane, slot // 2), target) == target
        ), f"lane {lane} slot {slot}: the 2:4 pair does not share one compressed element"
    return compressed


def _encoder_maps(traits: MmaTraits) -> dict[str, LaneMap]:
    """The encoder's A, B and C maps for one block of ``traits``."""
    one_block = dataclasses.replace(traits, b=1)
    a_map = _encoder_map(a_warp_encoding(one_block), "M", "K")
    if traits.family == "sparse":
        a_map = _compressed(a_map)
    return {
        "A": a_map,
        "B": _encoder_map(b_warp_encoding(one_block), "N", "K"),
        "C": _encoder_map(c_warp_encoding(one_block), "M", "N"),
    }


def _per_block(mapping: LaneMap, blocks: int) -> list[LaneMap]:
    """Split a whole-instruction map into one map per block, each renumbered to the lanes and
    slots that block uses (in order), so it compares with a single-block layout."""
    by_block: dict[int, LaneMap] = {}
    for key, element in mapping.items():
        coords = dict(element)
        block = coords.pop("B", 0)
        by_block.setdefault(block, {})[key] = tuple(sorted(coords.items()))
    split = []
    for block in range(blocks):
        part = by_block[block]
        lanes = {
            lane: rank for rank, lane in enumerate(sorted({lane for lane, _ in part}))
        }
        slots: dict[int, list[int]] = {}
        for lane, slot in sorted(part):
            slots.setdefault(lane, []).append(slot)
        split.append(
            {
                (lanes[lane], slots[lane].index(slot)): element
                for (lane, slot), element in part.items()
            }
        )
    return split


def _string_maps(traits: MmaTraits) -> dict[str, list[LaneMap]]:
    """The A, B and C maps the row's strings state, one per block."""
    sizes = {
        "M": traits.m,
        "N": traits.n,
        "K": traits.k,
        "KP": traits.k // 2,
        "B": traits.b,
        "R": traits.r,
    }
    a_k = "KP" if traits.family == "sparse" else "K"
    dims = {"A": ("M", a_k), "B": ("N", "K"), "C": ("M", "N")}
    strings = operand_layout_strings(traits)
    return {
        role: _per_block(
            interpret_layout(strings[role], sizes, traits.wave_size, dims[role]),
            traits.b,
        )
        for role in ("A", "B", "C")
    }


@pytest.mark.parametrize("traits", _ROWS, ids=_IDS)
def test_layout_strings_equal_the_encoder_layout(traits: MmaTraits) -> None:
    # The strings and the parameters are two descriptions of one hardware layout; they must agree
    # element for element, with no K reordering allowed (the layout-exactness rule).
    encoded = _encoder_maps(traits)
    for role, blocks in _string_maps(traits).items():
        for block, stated in enumerate(blocks):
            assert stated == encoded[role], (
                f"{traits.op_id} {role} (block {block}): the layout string and the layout "
                f"parameters disagree"
            )


def _k_per_lane(encoding) -> dict[int, set[int]]:
    """The K positions each lane holds under one A or B encoding (a lane may hold one K for several
    rows, as A of a two-block row does)."""
    mapper = RegisterMapper(encoding)
    return {
        lane: {
            mapper.matrix_coordinates(lane=lane, register=slot)[1]
            for slot in range(mapper.num_vector_items)
        }
        for lane in range(mapper.num_lanes)
    }


_EQUAL_WIDTH = [
    traits
    for traits in _ROWS
    if dtype_info(traits.a_dtype).encoded_bits
    == dtype_info(traits.b_dtype).encoded_bits
]


@pytest.mark.parametrize("traits", _EQUAL_WIDTH, ids=[t.op_id for t in _EQUAL_WIDTH])
def test_a_and_b_hold_the_same_k_in_every_lane(traits: MmaTraits) -> None:
    # An instruction multiplies the A and B elements of one K together, so when both operands have
    # the same width each lane must hold the same K positions of A (dense K for sparse A) as of B.
    # A layout change on one operand only would describe a pairing that is no matrix product.
    one_block = dataclasses.replace(traits, b=1)
    assert _k_per_lane(a_warp_encoding(one_block)) == _k_per_lane(
        b_warp_encoding(one_block)
    ), f"{traits.op_id}: A and B hold different K in some lane"


_SPARSE_WITH_INDEX = [t for t in _ROWS if t.index_layout is not None]


@pytest.mark.parametrize(
    "traits", _SPARSE_WITH_INDEX, ids=[t.op_id for t in _SPARSE_WITH_INDEX]
)
def test_sparse_index_operand_follows_the_a_layout(traits: MmaTraits) -> None:
    # The sparsity index operand describes the compressed A elements a lane holds, so its string is
    # A's string with the selector index S added as the most significant slot level.
    assert traits.index_layout == traits.a_layout.replace("V{", "V{S")


_GFX1250_SCALED = [
    traits
    for traits in _ROWS
    if traits.family == "scaled" and "gfx1250" in traits.supported_targets
]


@pytest.mark.parametrize(
    "traits", _GFX1250_SCALED, ids=[t.op_id for t in _GFX1250_SCALED]
)
def test_gfx1250_scaled_operands_equal_core_scaled_layout(traits: MmaTraits) -> None:
    # rocke core's scaled operand layout (runs of 16 K for 8-bit formats, 32 for 6- and 4-bit) is
    # the arbiter for the gfx1250 scaled instructions. Core describes a 16-row operand, so the
    # check covers every operand that is 16 rows wide: A and B of the 16x16x128 rows, B of the
    # 32x16x128 rows.
    operands = [("B", traits.b_dtype, b_warp_encoding(traits), "N")]
    if traits.m == 16:
        operands.append(("A", traits.a_dtype, a_warp_encoding(traits), "M"))
    for role, dtype, encoding, row_dim in operands:
        core = scaled_matrix_layout(dtype, 16)
        expected = {
            (lane, slot): tuple(sorted(((row_dim, row), ("K", k))))
            for lane in range(traits.wave_size)
            for slot in range(core.fragment.count)
            for row, k in [core.coord(lane, slot)]
        }
        assert (
            _encoder_map(encoding, row_dim, "K") == expected
        ), f"{traits.op_id} {role}"


def _selection_twins() -> list[tuple[MmaTraits, MmaTraits]]:
    """Each pin-only row with the selectable row that answers its selection on a shared target."""

    def key(traits: MmaTraits, target: str) -> tuple:
        block_k = traits.scale.block_k if traits.scale is not None else None
        return (
            target,
            traits.family,
            traits.a_dtype,
            traits.b_dtype,
            traits.c_dtype,
            traits.d_dtype,
            traits.m,
            traits.n,
            traits.k,
            block_k,
        )

    selectable = {
        key(traits, target): traits
        for traits in _ROWS
        if traits.auto_select and traits.b == 1
        for target in traits.supported_targets
    }
    twins = {
        (pinned.op_id, selectable[key(pinned, target)].op_id)
        for pinned in _ROWS
        if not pinned.auto_select
        for target in pinned.supported_targets
    }
    return [
        (_CATALOG.get(pinned), _CATALOG.get(twin)) for pinned, twin in sorted(twins)
    ]


# The f8f6f4 instruction's 8-bit x 8-bit rows: their K order comes from core's scaled layout, their
# dedicated fp8 twins' from CK (README, Selection).
_GFX1250_F8F6F4_8BIT_ROWS = {
    "wmma_f32_16x16x128_f8f6f4_fp8e4m3_fp8e4m3",
    "wmma_f32_16x16x128_f8f6f4_fp8e4m3_bf8e5m2",
    "wmma_f32_16x16x128_f8f6f4_bf8e5m2_fp8e4m3",
    "wmma_f32_16x16x128_f8f6f4_bf8e5m2_bf8e5m2",
}


def _k_relabelling(pinned: LaneMap, twin: LaneMap) -> dict[int, int]:
    """The twin's K -> the pinned row's K at the same (lane, slot); the other coordinate must be
    equal there, and each twin K must map to one pinned K."""
    relabelling: dict[int, int] = {}
    for position, element in twin.items():
        twin_coords, pinned_coords = dict(element), dict(pinned[position])
        twin_k, pinned_k = twin_coords.pop("K"), pinned_coords.pop("K")
        assert pinned_coords == twin_coords, f"{position}: rows or columns differ"
        assert (
            relabelling.setdefault(twin_k, pinned_k) == pinned_k
        ), f"{position}: K {twin_k} maps to two K values"
    return relabelling


_TWINS = _selection_twins()


@pytest.mark.parametrize(
    "pinned, twin", _TWINS, ids=[pinned.op_id for pinned, _ in _TWINS]
)
def test_pin_only_rows_lay_out_like_their_selectable_twin(
    pinned: MmaTraits, twin: MmaTraits
) -> None:
    # A pin-only row answers the same selection as its twin, so a kernel may hold either. Their A, B
    # and C maps are equal, except on the gfx1250 f8f6f4 8-bit rows, another instruction with its
    # own arbiter: there A and B may differ from the twin only by one K permutation shared by both,
    # which forms the same products.
    pinned_maps, twin_maps = _encoder_maps(pinned), _encoder_maps(twin)
    if pinned.op_id not in _GFX1250_F8F6F4_8BIT_ROWS:
        assert pinned_maps == twin_maps, f"{pinned.op_id} vs {twin.op_id}"
        return
    assert pinned_maps["C"] == twin_maps["C"]
    a_relabelling = _k_relabelling(pinned_maps["A"], twin_maps["A"])
    assert a_relabelling == _k_relabelling(pinned_maps["B"], twin_maps["B"])
    assert sorted(a_relabelling.values()) == list(range(pinned.k))


def _a_map(op_id: str) -> LaneMap:
    return _encoder_map(a_warp_encoding(_CATALOG.get(op_id)), "M", "K")


def test_gfx12_16bit_a_matches_the_calculator() -> None:
    # Public Matrix Instruction Calculator, v_wmma_f32_16x16x16_f16 on gfx12 (wave32): each lane
    # holds two runs of 4 K, so A[0][8] is in lane 0 slot 4 and A[0][4] is in lane 16 slot 0.
    a_map = _a_map("wmma_f32_16x16x16_f16_w32_gfx12")
    assert a_map[(0, 4)] == (("K", 8), ("M", 0))
    assert a_map[(16, 0)] == (("K", 4), ("M", 0))


def test_gfx11_16bit_result_matches_the_calculator() -> None:
    # Public Matrix Instruction Calculator, v_wmma_f16_16x16x16_f16 on gfx11 (wave32): a lane holds
    # 8 logical results, rows 2s + lane / 16 of column lane % 16, so logical slot 3 of lane 17 is
    # C(7, 1). How those 8 values sit in the registers (the even halves) is core's business.
    traits = _CATALOG.get("wmma_f16_16x16x16_f16_w32")
    c_map = _encoder_map(c_warp_encoding(traits), "M", "N")
    assert c_map[(17, 3)] == (("M", 7), ("N", 1))
    assert {slot for _, slot in c_map} == set(range(8))


def _unverified_section() -> str:
    readme = (DEFAULT_TRAITS_PATH.parent.parent / "README.md").read_text()
    match = re.search(
        r"^## Unverified layouts\n(.*?)(?=^## )", readme, re.DOTALL | re.MULTILINE
    )
    assert match, "traits/README.md has no 'Unverified layouts' section"
    return match.group(1)


def test_readme_lists_every_row_without_an_independent_arbiter() -> None:
    # Rows whose layout comes from CK or from the earlier table have no independent arbiter; the
    # README names each of them, so the list cannot fall behind the data.
    listed = set(re.findall(r"`([a-z0-9_]+)`", _unverified_section()))
    unverified = {
        traits.op_id
        for traits in _ROWS
        if traits.layout_source in ("ck", "legacy_table")
    }
    assert listed & set(_CATALOG.by_op_id) == unverified
