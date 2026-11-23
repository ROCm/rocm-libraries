# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The SDPA backward row set of a CDNA-9 arch (gfx90a, gfx942): the canonical Spec keys ``sdpa_bwd_pin_build.py`` pins.

    python tools/sdpa_bwd_rowset.py ARCH [--knobs KNOBS.csv] > rows.txt   (default overlay: tools/sdpa_bwd_rowset_knobs_ARCH.csv when it exists)

One padded ``g1`` object per (rung, class, dropout, dtype), one schedule per cell (the standing instance rule):

  16-bit (f16, bf16)  dkdv and dq: 12 rungs (``CDNA9_LADDER``) x 6 (plain/causal/bias x dropout 0/1) x 2 dtypes = 144 each
                      prep: 12 rungs x 2 dtypes = 24; ``dbias_reduce``: 2 (f16, bf16)
  fp32                dkdv and dq: 8 rungs + the half-head 512 rung x 6 = 54 each; prep: 9
                      (fp32 bias dQ ships combined; the one fp32 ``dbias_reduce`` row is the f32 dB reduce of the 16-bit rows)

431 rows: 288 + 24 + 2 + 104 + 13. On top of them come the 72 feature rows (``feature_rows``) and the fp32 ``dbias_reduce`` (the gfx11 kernel,
wave64): 504. The rows carry the arch's policy (no perf knob) except where a knob overlay spells one. The
overlay (``--knobs``) is a CSV with a header ``kernel,dtype,tile,cls,dropout,knobs`` and one row per overridden cell, ``knobs`` being
``;``-joined ``name=value`` pairs spelled as in a Spec key; ``*`` in ``tile``, ``cls`` or ``dropout`` matches every value. A later row
overrides an earlier one on the same knob. The gfx90a overlay holds the pins that are not arch-neutral policy: dK/dV head-fastest grid order (the arch default is AOTriton's KV-fastest) and the fp32 geometry that differs from the 16-row default. fp32 rows spell the geometry their family needs (``block_q=16`` / ``block_m=16``: one
schedule for every fp32 cell, the fp32 family's default is the 32-row tile that does not fit the hd 512 half-head rung).
"""

from __future__ import annotations

import csv
import sys
from itertools import product
from pathlib import Path

FLYDSL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FLYDSL))

from generators._sdpa_bwd_spec import (  # noqa: E402
    CDNA9_ARCHES,
    CLASSES,
    DTYPES,
    FP32,
    Spec,
    dbias_reduce_spec,
    ladder,
)
from generators._instances import SDPA_BWD_GFX11_TARGET  # noqa: E402
from generators._sdpa_bwd_rdna_build import specs_for  # noqa: E402

FP32_PINS = {"dkdv": {"block_q": 16}, "dq": {"block_m": 16}}


def load_overlay(path: Path | None) -> list[tuple]:
    if path is None:
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pins = {}
            for item in r["knobs"].split(";"):
                n, v = item.split("=")
                pins[n] = v if not v.lstrip("-").isdigit() else int(v)
            out.append(
                (r["kernel"], r["dtype"], r["tile"], r["cls"], r["dropout"], pins)
            )
    return out


def _matches(want: str, have) -> bool:
    return want == "*" or str(have) == want


def feature_rows(arch: str, specs: list[Spec]) -> list[Spec]:
    """The feature rows of @p arch: one per feature row of gfx11-generic (a dK/dV or dQ row with ``bias_f32`` or ``u2``), in gfx11's table order.

    A row is the arch's own sibling (the row of @p specs with the same kernel and the same dtype, tile, class, dform, dropout, padded,
    floor, V head dim and hints: its perf pins after the policy, the overlay and any re-knob) plus gfx11's feature pins (``bias_f32``,
    ``u2``, and ``store_db``, which gfx11 pins to 0 on the dQ objects that store no dB). ``ValueError`` lists the feature rows without sibling.
    """
    siblings = {(s.kernel, *s.functional[1:10]): s for s in specs}
    rows, orphans = [], []
    for kernel in ("dkdv", "dq"):
        for feature in specs_for(kernel, SDPA_BWD_GFX11_TARGET):
            if not (feature.bias_f32 or feature.u2):
                continue
            sibling = siblings.get((kernel, *feature.functional[1:10]))
            if sibling is None:
                orphans.append(feature.key())
                continue
            pins = {k: v for k, v in sibling.knobs.items() if k != "store_db"}
            pins.update(
                {k: getattr(feature, k) for k in ("bias_f32", "u2", "store_db")}
            )
            rows.append(
                Spec(
                    kernel,
                    sibling.dtype,
                    sibling.tile,
                    sibling.cls,
                    sibling.dform,
                    sibling.dropout,
                    sibling.padded,
                    sibling.floor,
                    sibling.hdv,
                    sibling.hints,
                    arch=arch,
                    **{k: v for k, v in pins.items() if v is not None},
                )
            )
    if orphans:
        raise ValueError(f"{arch} has no sibling row for the feature rows {orphans}")
    return rows


def rowset(arch: str, overlay: list[tuple] = ()) -> list[Spec]:
    """The Specs of @p arch, in kernel, dtype, tile, class, dropout order."""
    if arch not in CDNA9_ARCHES:
        raise ValueError(f"{arch} has no CDNA-9 row set")

    def spec(kernel, dtype, tile, cls, dropout):
        pins = dict(FP32_PINS.get(kernel, {})) if dtype == FP32 else {}
        for k, dt, t, c, d, o in overlay:
            if (
                k == kernel
                and dt == dtype
                and _matches(t, tile)
                and _matches(c, cls)
                and _matches(d, dropout)
            ):
                pins.update(o)
        if kernel == "prep":
            return Spec(
                "prep",
                dtype,
                tile,
                "plain",
                "bk",
                0,
                1,
                "tail",
                0,
                1,
                arch=arch,
                **pins,
            )
        return Spec(
            kernel, dtype, tile, cls, "bk", dropout, 1, "g1", 0, 1, arch=arch, **pins
        )

    specs = []
    for kernel in ("prep", "dkdv", "dq"):
        for dtype in (*DTYPES, FP32):
            for tile in ladder(arch, dtype):
                if kernel == "prep":
                    specs.append(spec(kernel, dtype, tile, "plain", 0))
                    continue
                for cls, dropout in product(CLASSES, (0, 1)):
                    specs.append(spec(kernel, dtype, tile, cls, dropout))
    specs += feature_rows(arch, specs)
    specs += [dbias_reduce_spec(dtype, arch) for dtype in (*DTYPES, FP32)]
    return specs


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    knobs = None
    if "--knobs" in args:
        i = args.index("--knobs")
        knobs = Path(args[i + 1])
        del args[i : i + 2]
    (arch,) = args
    if (
        knobs is None
        and (
            default := Path(__file__).with_name(f"sdpa_bwd_rowset_knobs_{arch}.csv")
        ).is_file()
    ):
        knobs = default
    keys = [s.key() for s in rowset(arch, load_overlay(knobs))]
    assert len(set(keys)) == len(keys)
    sys.stdout.write("\n".join(keys) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
