# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Differential of the gfx950 dense tile decision, across three selectors.

Tile selection is a pure function of a few scalars, so it can be evaluated over
its whole input domain on a CPU rather than inferred from a diff. This tabulates
where the three selectors disagree:

``shape-blind``
    What ``develop`` shipped: one constant preference (256/64) for every problem,
    then ascending ``(block_m, block_n)`` for the rest. Restated here because it
    no longer exists in the tree to call.
``preferred``
    What this change ships: ``preferredTile`` / ``tilePreference`` in
    ``Gfx950AttentionDenseNative.cpp``, restated in Python.
``dispatcher``
    rocKE's own ``library/dispatch/attention/gfx950_dense.py::_base_spec``, called
    directly rather than restated, so a drift between the engine and the
    dispatcher shows up here instead of being asserted.

The C++ is the shipping selector; the first two entries above are mirrors of it
and of its predecessor. ``--check`` re-derives the mirror against the thresholds
named in the C++ and fails if they have moved, so this file cannot quietly
describe a rule the engine no longer implements.

Runs anywhere: no GPU, no measurement data, no build tree.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

from kernels.gfx950.attention_dense import (  # noqa: E402
    gfx950_dense_tile_legality,
)

# Mirrors the constants in Gfx950AttentionDenseNative.cpp. --check re-reads them
# from that file, so this table cannot drift from the engine unnoticed.
BLOCK_M_256_FROM = {
    (64, True): 8192,
    (64, False): 1024,
    (128, True): 4096,
    (128, False): 0,
}
PREFERRED_BLOCK_N_D128_CAUSAL = 64
PREFERRED_BLOCK_N_OTHERWISE = 128

# The one tile develop preferred, for every shape.
SHAPE_BLIND_TILE = (256, 64)

_NATIVE_CPP = Path(_RK).parents[0] / (
    "src/engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseNative.cpp"
)


def divides(seqlen_q: int, seqlen_kv: int, block_m: int, block_n: int) -> bool:
    """Whether a tile can serve the shape at all.

    ``kernelMatches`` drops a tile that does not divide both lengths before
    anything is ranked, so an undivided tile is never selected by any of the
    three selectors and must not appear in the differential.
    """
    return seqlen_q % block_m == 0 and seqlen_kv % block_n == 0


def servable(head_size: int, seqlen_q: int, seqlen_kv: int):
    return [
        t
        for t in gfx950_dense_tile_legality(head_size=head_size)
        if divides(seqlen_q, seqlen_kv, *t)
    ]


def _rank(candidates, want):
    """The candidate closest to ``want``: block_m first, then block_n, then the
    ascending order as the remainder. Mirrors tilePreference's ordering."""

    def key(tile):
        return (tile[0] == want[0], tile[1] == want[1], -(tile[0] * 1024 + tile[1]))

    return max(candidates, key=key)


def shape_blind(head_size, causal, seqlen_q, seqlen_kv):
    """develop's selector: 256/64 wherever it serves, ascending otherwise.

    Note the fallback is a plain ascending ``(block_m, block_n)``, with no credit
    for matching one axis of 256/64: develop scored the baseline tile alone and
    put every alternative in a band below it ordered on the tile key. The partial
    match is this change's idea, so crediting it here would understate the diff.
    """
    candidates = servable(head_size, seqlen_q, seqlen_kv)
    if not candidates:
        return None
    if SHAPE_BLIND_TILE in candidates:
        return SHAPE_BLIND_TILE
    return min(candidates)


def preferred_tile(head_size, causal, seqlen_q):
    """The tile this change prefers, before divisibility is applied."""
    block_m = 256 if seqlen_q >= BLOCK_M_256_FROM[(head_size, causal)] else 128
    block_n = (
        PREFERRED_BLOCK_N_D128_CAUSAL
        if head_size == 128 and causal
        else PREFERRED_BLOCK_N_OTHERWISE
    )
    return block_m, block_n


def preferred(head_size, causal, seqlen_q, seqlen_kv):
    """This change's selector."""
    candidates = servable(head_size, seqlen_q, seqlen_kv)
    if not candidates:
        return None
    return _rank(candidates, preferred_tile(head_size, causal, seqlen_q))


def dispatcher(head_size, causal, seqlen_q, seqlen_kv, *, batch=1, heads=8):
    """rocKE's own selector, called rather than restated.

    Returns ``None`` when the dispatcher declines, which is not a disagreement:
    the registry is a different code path with its own support rules.
    """
    from dispatch.attention.common import AttentionMaskType, AttentionRequest
    from dispatch.attention.gfx950_dense import GFX950_DENSE_VARIANTS, _base_spec

    request = AttentionRequest(
        batch=batch,
        nhead_q=heads,
        nhead_k=heads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_kv,
        hdim_q=head_size,
        hdim_v=head_size,
        arch="gfx950",
        mask_type=(
            AttentionMaskType.TOP_LEFT_CAUSAL if causal else AttentionMaskType.NO_MASK
        ),
    )
    grid = next(v for v in GFX950_DENSE_VARIANTS if not v.persistent)
    try:
        spec = _base_spec(request, grid)
    except Exception:  # noqa: BLE001 - a decline is data, not a failure
        return None
    return int(spec.block_m), int(spec.block_n)


def domain(lengths):
    """Every (head size, mask, length pair) the catalog can be asked for."""
    for head_size, causal in itertools.product((64, 128), (True, False)):
        for seqlen_q, seqlen_kv in itertools.product(lengths, lengths):
            # The kernels build a top-left causal mask, so a causal graph with
            # different lengths is declined before selection and is not a shape
            # any selector sees.
            if causal and seqlen_q != seqlen_kv:
                continue
            if not servable(head_size, seqlen_q, seqlen_kv):
                continue
            yield head_size, causal, seqlen_q, seqlen_kv


def check_against_native():
    """Fail if the mirrored thresholds no longer match the shipping C++."""
    if not _NATIVE_CPP.exists():
        raise SystemExit(f"cannot read the engine source at {_NATIVE_CPP}")
    text = _NATIVE_CPP.read_text()

    def constant(name):
        match = re.search(rf"{name}\s*=\s*(\d+)\s*;", text)
        if match is None:
            raise SystemExit(f"{name} is gone from {_NATIVE_CPP.name}")
        return int(match.group(1))

    expected = {
        (64, True): constant("BLOCK_M_256_FROM_D64_CAUSAL"),
        (64, False): constant("BLOCK_M_256_FROM_D64_NONCAUSAL"),
        (128, True): constant("BLOCK_M_256_FROM_D128_CAUSAL"),
        (128, False): constant("BLOCK_M_256_FROM_D128_NONCAUSAL"),
    }
    problems = [
        f"  {key}: this file says {BLOCK_M_256_FROM[key]}, the engine says {value}"
        for key, value in expected.items()
        if BLOCK_M_256_FROM[key] != value
    ]
    for name, mine in (
        ("PREFERRED_BLOCK_N_D128_CAUSAL", PREFERRED_BLOCK_N_D128_CAUSAL),
        ("PREFERRED_BLOCK_N_OTHERWISE", PREFERRED_BLOCK_N_OTHERWISE),
    ):
        if constant(name) != mine:
            problems.append(
                f"  {name}: this file says {mine}, the engine says {constant(name)}"
            )
    if problems:
        raise SystemExit(
            "the mirrored rule no longer matches the engine:\n" + "\n".join(problems)
        )
    print(f"mirror agrees with {_NATIVE_CPP.name} on every threshold")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        default="128,256,384,512,1024,2048,4096,8192,16384",
        help="comma-separated sequence lengths to walk",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the mirrored thresholds still match the engine, then exit",
    )
    parser.add_argument(
        "--all", action="store_true", help="print agreeing rows as well as divergences"
    )
    args = parser.parse_args(argv)

    if args.check:
        check_against_native()
        return 0

    check_against_native()
    lengths = [int(v) for v in args.lengths.split(",")]

    rows, diverge_new, diverge_py = [], 0, 0
    for head_size, causal, seqlen_q, seqlen_kv in domain(lengths):
        old = shape_blind(head_size, causal, seqlen_q, seqlen_kv)
        new = preferred(head_size, causal, seqlen_q, seqlen_kv)
        py = dispatcher(head_size, causal, seqlen_q, seqlen_kv)
        changed = old != new
        py_differs = py is not None and py != new
        diverge_new += changed
        diverge_py += py_differs
        if changed or py_differs or args.all:
            rows.append((head_size, causal, seqlen_q, seqlen_kv, old, new, py, changed))

    total = sum(1 for _ in domain(lengths))
    print(f"shapes walked: {total}")
    print(f"  selection changed by this PR: {diverge_new}")
    print(f"  dispatcher disagrees with the engine: {diverge_py}\n")
    print(
        f"{'D':>4} {'mask':>10} {'Sq':>6} {'Skv':>6} "
        f"{'develop':>10} {'this PR':>10} {'dispatcher':>10}"
    )
    for head_size, causal, seqlen_q, seqlen_kv, old, new, py, changed in rows:
        mark = "*" if changed else " "
        fmt = lambda t: "-" if t is None else f"{t[0]}/{t[1]}"  # noqa: E731
        print(
            f"{mark}{head_size:>3} {'causal' if causal else 'full':>10} "
            f"{seqlen_q:>6} {seqlen_kv:>6} "
            f"{fmt(old):>10} {fmt(new):>10} {fmt(py):>10}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
