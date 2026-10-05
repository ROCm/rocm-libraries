# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``transform_fragment`` verb -- the front-door IR verb that REALISES a classified reorder.

The solving is done in ``observers.classify_transform`` / ``_core._classify_maps``; this module only
emits the register permutation a ``reorder`` plan describes. It lives apart from ``emit`` so the memory
verbs never import the solver; the IRBuilder ``b`` is duck-typed, so this module takes NO IR import (the
tests drive it with a plain-int builder, exactly like the address-map replay).
"""

from __future__ import annotations

from typing import Any

from ..fragments import Fragment, TileDesc
from .observers import classify_transform


def transform_fragment(b: Any, fragment: Fragment, target_desc: TileDesc) -> Fragment:
    """Retarget `fragment` to `target_desc`'s layout via the cheapest op sequence -- the verb that
    REALISES what :func:`classify_transform` plans.

    The delta between the two layouts is solved from their encodings and classified: a `reorder`
    (same element set per lane, one lane-uniform register permutation) is emitted as a compile-time
    register shuffle (`vec_extract`/`vec_insert`, element-granular so it is correct for any dtype
    packing). A `cross_lane` delta (an element changes lanes) needs cross-lane movement (DPP /
    ds_bpermute / LDS) and raises `NotImplementedError` -- the reserved seam a future cross-lane
    realisation fills. Same element set required (else `ValueError`). Dtype is carried through unchanged.
    """
    plan = classify_transform(fragment.tile_desc.layout, target_desc.layout)
    if plan.tier != "reorder":
        raise NotImplementedError(
            f"cross-lane fragment transform is not supported yet -- {plan.reason}"
        )
    n = target_desc.register_count
    out = b.zero_vec(fragment.dtype, n)
    for src_reg, dst_reg in enumerate(plan.permutation):
        out = b.vec_insert(out, b.vec_extract(fragment.value, src_reg), dst_reg)
    return Fragment(target_desc, fragment.dtype, out)
