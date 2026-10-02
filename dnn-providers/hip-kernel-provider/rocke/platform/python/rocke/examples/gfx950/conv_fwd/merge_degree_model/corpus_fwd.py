#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Load the measured depthwise-forward degree corpus from measure_degrees.py CSVs.

Every row the fitter ever sees passes through here. The job of this module is to
turn a sweep CSV into ``Config`` records that carry *exactly* the features a
dispatch policy is allowed to look at -- N, Hi, Wi, G, Y, X, stride, dilation,
dtype -- plus the measured per-degree curve, normalised so no absolute figure
survives.

Two invariants make the corpus usable as a fitting target:

  * **Only verified rows count.** A merged degree that runs fast because its
    diagonal B mask is wrong must never be allowed to set an oracle, so rows
    without ``passed == True`` are dropped before anything else happens.

  * **Curves are normalised against the shape's own Gm=1.** A shape with no
    measured unmerged baseline is dropped entirely rather than normalised
    against its own best, because "fraction of the best degree" and "speedup
    over not merging" are different quantities and mixing them would corrupt the
    objective. The normalisation is also what keeps this corpus free of absolute
    performance figures: every number downstream is a ratio within one shape.

Shapes are keyed on the *full request identity* -- shape string plus stride,
dilation, padding and dtype -- not just the shape string. The corpus
deliberately contains the same spatial shape at several strides and dilations,
and collapsing those would average away the exact mechanism the model is being
fitted to separate.

The CSVs themselves are not in the repository; regenerate them locally, see
README.md. With no CSV present ``measured()`` returns an empty list and every
consumer degrades to a no-op rather than failing.

Nothing here touches a GPU.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent


@dataclass
class Config:
    """One measured shape and its degree curve.

    The fields above ``curve`` are precisely the request-side features dispatch
    has at selection time. Anything the model is allowed to use must be derivable
    from them; if a feature cannot be computed here, it cannot be used in
    model.py either.
    """

    name: str
    n: int
    hi: int
    wi: int
    groups: int
    y: int
    x: int
    stride: int
    dtype: str
    curve: dict  # degree -> relative score (higher is better); Gm=1 == 1.0
    source: str
    # Exact output geometry, when the source records enough to compute it. The
    # measured corpus carries dilation and non-square inputs, where the
    # SAME-padding shortcut in m() is simply wrong.
    mo: int = None
    dilation: int = 1
    # Output width. The GEMM M index is n*Ho*Wo + ho*Wo + wo, so Wo is what
    # decides whether a tile_m run of M stays inside one output row (and reads a
    # contiguous input span) or wraps across rows.
    wo_: int = None

    @property
    def wo(self):
        return self.wo_ if self.wo_ is not None else -(-self.wi // self.stride)

    @property
    def ho(self):
        return (self.hi - self.y) // self.stride + 1 if self.hi >= self.y else 1

    def m(self, pad_same=True):
        """GEMM M extent = N*Ho*Wo -- the dimension the degree does *not* move."""
        if self.mo is not None:
            return self.mo
        if pad_same:
            ho = -(-self.hi // self.stride)
            wo = -(-self.wi // self.stride)
        else:
            ho, wo = self.ho, self.ho
        return self.n * ho * wo

    @property
    def esize(self):
        return 4 if self.dtype == "fp32" else 2

    @property
    def oracle(self):
        """The degree that actually won. The thing the model is trying to guess."""
        return max(self.curve, key=self.curve.get)

    def realised(self, gm):
        """Fraction of this shape's measured best that degree ``gm`` achieves.

        This is the per-shape objective: 1.0 means the model picked the oracle,
        0.8 means it left a fifth of the available gain on the table. A degree
        the sweep never built falls back to unmerged, which is what dispatch
        would do anyway if it named an unbuildable degree.
        """
        best = self.curve[self.oracle]
        got = self.curve.get(gm)
        if got is None:
            got = self.curve.get(1, 0.0)
        return got / best if best > 0 else float("nan")


_SHAPE_RE = re.compile(r"N(\d+)H(\d+)W(\d+)C(\d+)_K(\d+)Y(\d+)X(\d+)G(\d+)")

# Both files are produced by measure_degrees.py at the same pinned tile with the
# degree window disabled, so their rows are commensurate and union cleanly: the
# generated corpus (gen_shapes.py) and the case-study shapes re-measured under
# the same pinning.
MEASURED_CSVS = ("degrees.csv", "degrees_tuning.csv")


def _measured(path=None, tile_m=64, tile_n=64):
    """Ingest the measure_degrees.py CSVs into normalised Config records."""
    paths = [Path(path)] if path else [HERE / n for n in MEASURED_CSVS]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return []
    groups = {}
    for p in paths:
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                # Correctness gate first: an unverified row cannot set an oracle.
                if row.get("passed", "").strip() != "True":
                    continue
                if row.get("direction", "fwd") != "fwd":
                    continue
                try:
                    tf = float(row["rocke_tflops"])
                except (TypeError, ValueError):
                    continue
                if tf <= 0.0:
                    continue
                # The tile is pinned by the driver, but a concatenated corpus can
                # still carry foreign rows; keep only the tile asked for.
                if int(row["tile_m"]) != tile_m or int(row["tile_n"]) != tile_n:
                    continue
                mm = _SHAPE_RE.match(row["shape"])
                if not mm:
                    continue
                key = (
                    row["shape"], row["dtype"], row["sH"], row["sW"],
                    row["pH"], row["pW"], row["dH"], row["dW"],
                )
                gm = int(row["group_merge"] or 1)
                slot = groups.setdefault(key, {})
                # Best observed per degree: the sweep may report a degree more
                # than once across pipelines/epilogues, and the degree's merit is
                # its best achievable, not its average.
                slot[gm] = max(slot.get(gm, 0.0), tf)

    out = []
    for key, best in sorted(groups.items()):
        if 1 not in best or best[1] <= 0.0:
            continue  # no unmerged baseline measured -> no normalisable curve
        shape, dtype, sh, sw, ph, pw, dh, dw = key
        n, hi, wi, _c, _k, y, x, g = (
            int(v) for v in _SHAPE_RE.match(shape).groups()
        )
        sh, sw, ph, pw, dh, dw = (int(v) for v in (sh, sw, ph, pw, dh, dw))
        ho = (hi + 2 * ph - dh * (y - 1) - 1) // sh + 1
        wo = (wi + 2 * pw - dw * (x - 1) - 1) // sw + 1
        base = best[1]
        name = f"N{n}_H{hi}W{wi}_G{g}_Y{y}X{x}"
        if sh != 1:
            name += f"_s{sh}"
        if dh != 1:
            name += f"_d{dh}"
        if dtype != "bf16":
            name += f"_{dtype}"
        out.append(
            Config(
                name, n, hi, wi, g, y, x, sh, dtype,
                # The normalisation. Everything downstream is a ratio.
                {deg: v / base for deg, v in best.items()},
                "measured", mo=n * ho * wo, dilation=dh, wo_=wo,
            )
        )
    return out


def measured(**kw):
    """The full pinned-tile measured corpus, or [] if it has not been generated."""
    return _measured(**kw)


if __name__ == "__main__":
    cfgs = measured()
    if not cfgs:
        raise SystemExit("no measured corpus present — see README.md")
    for c in cfgs:
        print(
            f"{c.name:<34} {c.source:<8} s={c.stride} M={c.m():>8} "
            f"CTAs={-(-c.m()//64)*c.groups:>9} oracle={c.oracle:>3}"
        )
    print(f"{len(cfgs)} configs")
