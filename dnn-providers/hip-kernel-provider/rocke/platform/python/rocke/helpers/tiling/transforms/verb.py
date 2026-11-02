# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``transform_fragment`` verb -- the front-door IR verb that REALISES a classified reorder.

The solving is done in ``observers.classify_transform`` / ``_core._classify_maps``; this module only
emits the register permutation a ``reorder`` plan describes. It lives apart from ``emit`` so the memory
verbs never import the solver.
"""

from __future__ import annotations

from typing import Any

from ....core.storage import FragmentPacking, register_packing
from ..emit import _move_registers
from ..fragments import Fragment, TileDesc
from .observers import classify_transform


def transform_fragment(b: Any, fragment: Fragment, target_desc: TileDesc) -> Fragment:
    """Retarget `fragment` to `target_desc`'s layout via the cheapest op sequence -- the verb that
    REALISES what :func:`classify_transform` plans.

    The delta between the two layouts is solved from their encodings and classified: a `reorder`
    (same element set per lane, one lane-uniform register permutation) is emitted as a compile-time
    register shuffle (`vec_extract`/`vec_insert`) of the fragment's storage registers
    (:attr:`Fragment.storage_type`): the logical elements themselves for a byte-or-wider dtype (``i32``
    words for tf32), the 32-bit words for a packed sub-byte dtype. A packed reorder must move whole
    word groups (:func:`_packed_word_moves`); one that moves elements within a word raises
    `NotImplementedError`. A `cross_lane` delta (an element changes lanes) needs cross-lane movement
    (DPP / ds_bpermute / LDS) and raises `NotImplementedError` -- the reserved seam a future cross-lane
    realisation fills. Same element set required (else `ValueError`). Dtype is carried through unchanged.
    """
    plan = classify_transform(fragment.tile_desc.layout, target_desc.layout)
    if plan.tier != "reorder":
        raise NotImplementedError(
            f"cross-lane fragment transform is not supported yet -- {plan.reason}"
        )
    packing = register_packing(fragment.dtype.name, target_desc.register_count)
    if packing is None:
        moves = list(enumerate(plan.permutation))
    else:
        moves = _packed_word_moves(fragment.dtype.name, packing, plan.permutation)
    out = _move_registers(b, fragment.value, moves, fragment.storage_type.count)
    return Fragment(target_desc, fragment.dtype, out)


def _packed_word_moves(
    dtype: str, packing: FragmentPacking, permutation: tuple[int, ...]
) -> list[tuple[int, int]]:
    """The (source word, destination word) moves that realise `permutation` (source register ->
    destination register) on the 32-bit words of a `dtype` fragment packed as `packing`.

    Elements pack low-bit-first into words, so the smallest unit that moves without repacking bits is
    a word group: the fewest elements that fill whole words (8 elements in one word at 4 bits, 16 in
    three words at 6 bits). The permutation must carry every word group, in order, onto a word group;
    otherwise it raises `NotImplementedError`, since repacking the bits of a packed fragment is not
    implemented.
    """
    group, words = packing.packing.group(packing.carrier_bits)
    moves = []
    for source in range(0, len(permutation), group):
        destination = permutation[source]
        run = tuple(permutation[source : source + group])
        if destination % group or run != tuple(range(destination, destination + group)):
            raise NotImplementedError(
                f"transform_fragment cannot move {dtype} registers {source}..{source + group - 1} "
                f"to {list(run)}: {dtype} registers are held packed, so a reorder must move each "
                f"run of {group} registers ({words} i32 {'word' if words == 1 else 'words'}) whole "
                "and in order onto another such run -- TODO: not implemented yet; repacking the "
                "bits of a packed fragment"
            )
        moves.extend(
            (source // group * words + w, destination // group * words + w)
            for w in range(words)
        )
    return moves
