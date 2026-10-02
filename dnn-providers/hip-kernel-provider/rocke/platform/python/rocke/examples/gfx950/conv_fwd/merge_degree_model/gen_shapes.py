#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate a wide depthwise-forward shape corpus as MIOpenDriver lines.

Two sources, tagged in a comment on every line so the scorer can split them:

``model``  depthwise stages from published convolution architectures -- the
           (resolution, channels, filter, stride) tuples that real networks
           actually run. These are what the heuristic has to get right; they
           cluster hard (3x3 and 5x5 at a handful of resolutions) and on their
           own leave most of the parameter space untouched.

``grid``   a systematic sweep that fills in what the models never ask for:
           non-power-of-two group counts, asymmetric and very large filters,
           dilation, tiny and huge spatial extents, and the batch sizes that
           move the occupancy term. Without these the fit sees five mechanisms
           through one keyhole.

Every emitted case is depthwise (C == K == G, so cpg == kpg == 1), is SAME- or
VALID-padded consistently, and clears the benchmark's int32 tensor-byte guard.

Usage:
    python3 gen_shapes.py --count 400 -o shapes_fwd.txt
"""

import argparse
import random

# The benchmark skips any case whose largest tensor exceeds int32 bytes; stay
# well under so a case is never silently dropped mid-run.
_MAX_TENSOR_BYTES = 1_900_000_000
# Keeps one timed launch short enough that 400 shapes x 7 degrees is a
# measurement campaign rather than an overnight job.
_MAX_MACS = 40e9


class Shape:
    __slots__ = ("n", "hi", "wi", "g", "y", "x", "sh", "sw", "dh", "dw", "dtype",
                 "source", "tag")

    def __init__(self, n, hi, wi, g, y, x, sh=1, sw=None, dh=1, dw=None,
                 dtype="bf16", source="grid", tag=""):
        self.n, self.hi, self.wi, self.g = n, hi, wi, g
        self.y, self.x = y, x
        self.sh, self.sw = sh, sh if sw is None else sw
        self.dh, self.dw = dh, dh if dw is None else dw
        self.dtype, self.source, self.tag = dtype, source, tag

    # -- derived geometry ---------------------------------------------------
    @property
    def ph(self):
        return (self.dh * (self.y - 1)) // 2

    @property
    def pw(self):
        return (self.dw * (self.x - 1)) // 2

    @property
    def ho(self):
        return (self.hi + 2 * self.ph - self.dh * (self.y - 1) - 1) // self.sh + 1

    @property
    def wo(self):
        return (self.wi + 2 * self.pw - self.dw * (self.x - 1) - 1) // self.sw + 1

    def key(self):
        return (self.n, self.hi, self.wi, self.g, self.y, self.x,
                self.sh, self.sw, self.dh, self.dw, self.dtype)

    def valid(self):
        if min(self.ho, self.wo) < 1 or self.g < 2:
            return False
        # Depthwise forward merging needs a filter; the gate rejects pointwise.
        if self.y * self.x <= 1:
            return False
        if self.dh * (self.y - 1) + 1 > self.hi + 2 * self.ph:
            return False
        if self.dw * (self.x - 1) + 1 > self.wi + 2 * self.pw:
            return False
        esz = 4 if self.dtype == "fp32" else 2
        a = self.n * self.hi * self.wi * self.g * esz
        d = self.n * self.ho * self.wo * self.g * esz
        if max(a, d) > _MAX_TENSOR_BYTES:
            return False
        if self.n * self.ho * self.wo * self.g * self.y * self.x > _MAX_MACS:
            return False
        return True

    def line(self):
        drv = {"fp16": "convfp16", "bf16": "convbfp16", "fp32": "conv"}[self.dtype]
        return (
            f"MIOpenDriver {drv} -n {self.n} -c {self.g} -H {self.hi} -W {self.wi} "
            f"-k {self.g} -y {self.y} -x {self.x} -p {self.ph} -q {self.pw} "
            f"-u {self.sh} -v {self.sw} -l {self.dh} -j {self.dw} "
            f"-m conv -g {self.g} -F 1 -t 1 -in_layout=NHWC"
            f"  # {self.source}:{self.tag}"
        )


# ---------------------------------------------------------------------------
# Source 1: depthwise stages from published architectures
#
# Each entry is (resolution, channels, Y, X, stride). Channel counts and
# resolutions are the per-stage values the architectures are defined with; the
# depthwise layer in an inverted-residual block runs on the *expanded* channel
# count, which is what is listed.
# ---------------------------------------------------------------------------

_MODELS = {
    # (res, ch, y, x, stride)
    "mobilenetv1": [
        (112, 32, 3, 3, 1), (112, 64, 3, 3, 2), (56, 128, 3, 3, 1),
        (56, 128, 3, 3, 2), (28, 256, 3, 3, 1), (28, 256, 3, 3, 2),
        (14, 512, 3, 3, 1), (14, 512, 3, 3, 2), (7, 1024, 3, 3, 1),
    ],
    "mobilenetv1_075": [
        (112, 24, 3, 3, 1), (112, 48, 3, 3, 2), (56, 96, 3, 3, 1),
        (56, 96, 3, 3, 2), (28, 192, 3, 3, 1), (28, 192, 3, 3, 2),
        (14, 384, 3, 3, 1), (14, 384, 3, 3, 2), (7, 768, 3, 3, 1),
    ],
    "mobilenetv2": [
        (112, 32, 3, 3, 1), (112, 96, 3, 3, 2), (56, 144, 3, 3, 1),
        (56, 144, 3, 3, 2), (28, 192, 3, 3, 1), (28, 192, 3, 3, 2),
        (14, 384, 3, 3, 1), (14, 576, 3, 3, 1), (14, 576, 3, 3, 2),
        (7, 960, 3, 3, 1),
    ],
    "mobilenetv2_14": [
        (112, 48, 3, 3, 1), (112, 144, 3, 3, 2), (56, 192, 3, 3, 1),
        (56, 192, 3, 3, 2), (28, 288, 3, 3, 1), (28, 288, 3, 3, 2),
        (14, 528, 3, 3, 1), (14, 816, 3, 3, 1), (14, 816, 3, 3, 2),
        (7, 1344, 3, 3, 1),
    ],
    "mobilenetv3_large": [
        (112, 16, 3, 3, 1), (112, 64, 3, 3, 2), (56, 72, 3, 3, 1),
        (56, 72, 5, 5, 2), (28, 120, 5, 5, 1), (28, 240, 3, 3, 2),
        (14, 200, 3, 3, 1), (14, 184, 3, 3, 1), (14, 480, 3, 3, 1),
        (14, 672, 3, 3, 1), (14, 672, 5, 5, 2), (7, 960, 5, 5, 1),
    ],
    "mobilenetv3_small": [
        (112, 16, 3, 3, 2), (56, 72, 3, 3, 2), (28, 88, 3, 3, 1),
        (28, 96, 5, 5, 2), (14, 240, 5, 5, 1), (14, 120, 5, 5, 1),
        (14, 144, 5, 5, 1), (14, 288, 5, 5, 2), (7, 576, 5, 5, 1),
    ],
    "efficientnet_b0": [
        (112, 32, 3, 3, 1), (112, 96, 3, 3, 2), (56, 144, 3, 3, 1),
        (56, 144, 5, 5, 2), (28, 240, 5, 5, 1), (28, 240, 3, 3, 2),
        (14, 480, 3, 3, 1), (14, 480, 5, 5, 1), (14, 672, 5, 5, 1),
        (14, 672, 5, 5, 2), (7, 1152, 5, 5, 1), (7, 1152, 3, 3, 1),
    ],
    "efficientnet_b2": [
        (130, 32, 3, 3, 1), (130, 96, 3, 3, 2), (65, 144, 3, 3, 1),
        (65, 144, 5, 5, 2), (33, 288, 5, 5, 1), (33, 288, 3, 3, 2),
        (17, 528, 3, 3, 1), (17, 528, 5, 5, 1), (17, 720, 5, 5, 1),
        (17, 720, 5, 5, 2), (9, 1248, 5, 5, 1),
    ],
    "efficientnet_b4": [
        (190, 48, 3, 3, 1), (190, 144, 3, 3, 2), (95, 192, 3, 3, 1),
        (95, 192, 5, 5, 2), (48, 336, 5, 5, 1), (48, 336, 3, 3, 2),
        (24, 672, 3, 3, 1), (24, 672, 5, 5, 1), (24, 960, 5, 5, 1),
        (24, 960, 5, 5, 2), (12, 1632, 5, 5, 1), (12, 2688, 3, 3, 1),
    ],
    "efficientnet_b6": [
        (264, 56, 3, 3, 1), (264, 192, 3, 3, 2), (132, 240, 3, 3, 1),
        (132, 240, 5, 5, 2), (66, 432, 5, 5, 1), (66, 432, 3, 3, 2),
        (33, 864, 3, 3, 1), (33, 864, 5, 5, 1), (33, 1200, 5, 5, 1),
        (33, 1200, 5, 5, 2), (17, 2064, 5, 5, 1),
    ],
    "efficientnetv2_s": [
        (48, 256, 3, 3, 2), (24, 512, 3, 3, 1), (24, 640, 3, 3, 2),
        (12, 960, 3, 3, 1), (12, 1536, 3, 3, 1),
    ],
    "efficientnetv2_m": [
        (60, 320, 3, 3, 2), (30, 640, 3, 3, 1), (30, 960, 3, 3, 2),
        (15, 1056, 3, 3, 1), (15, 1824, 3, 3, 1),
    ],
    "mobilenetv4_conv_s": [
        (64, 128, 3, 3, 2), (32, 256, 5, 5, 1), (32, 384, 3, 3, 2),
        (16, 480, 5, 5, 1), (16, 960, 3, 3, 1), (8, 1280, 5, 5, 1),
    ],
    "mobilenetv4_conv_l": [
        (96, 192, 3, 3, 2), (48, 384, 5, 5, 1), (48, 576, 3, 3, 2),
        (24, 768, 5, 5, 1), (24, 1152, 3, 3, 1), (12, 1920, 5, 5, 1),
        (12, 1920, 3, 3, 1),
    ],
    "mnasnet_a1": [
        (112, 48, 3, 3, 2), (56, 72, 5, 5, 2), (28, 240, 5, 5, 1),
        (28, 240, 3, 3, 2), (14, 480, 3, 3, 1), (14, 576, 3, 3, 1),
        (14, 576, 5, 5, 2), (7, 1152, 5, 5, 1),
    ],
    "shufflenetv2_1x": [
        (56, 116, 3, 3, 2), (28, 116, 3, 3, 1), (28, 232, 3, 3, 2),
        (14, 232, 3, 3, 1), (14, 464, 3, 3, 2), (7, 464, 3, 3, 1),
    ],
    "shufflenetv2_2x": [
        (56, 244, 3, 3, 2), (28, 244, 3, 3, 1), (28, 488, 3, 3, 2),
        (14, 488, 3, 3, 1), (14, 976, 3, 3, 2), (7, 976, 3, 3, 1),
    ],
    "ghostnet": [
        (112, 16, 3, 3, 2), (56, 48, 3, 3, 1), (56, 72, 3, 3, 2),
        (28, 72, 5, 5, 1), (28, 120, 5, 5, 2), (14, 240, 3, 3, 1),
        (14, 336, 3, 3, 1), (14, 672, 5, 5, 2), (7, 960, 5, 5, 1),
    ],
    "mixnet": [
        (112, 32, 3, 3, 1), (56, 144, 3, 3, 2), (56, 144, 5, 5, 2),
        (28, 240, 5, 5, 1), (28, 240, 7, 7, 1), (14, 480, 3, 3, 1),
        (14, 480, 9, 9, 1), (14, 720, 5, 5, 1), (7, 1200, 9, 9, 1),
        (7, 1200, 11, 11, 1),
    ],
    "xception": [
        (147, 128, 3, 3, 1), (74, 256, 3, 3, 1), (37, 728, 3, 3, 1),
        (19, 728, 3, 3, 1), (10, 1024, 3, 3, 1), (10, 1536, 3, 3, 1),
        (10, 2048, 3, 3, 1),
    ],
    "convnext_t": [
        (56, 96, 7, 7, 1), (28, 192, 7, 7, 1), (14, 384, 7, 7, 1),
        (7, 768, 7, 7, 1),
    ],
    "convnext_b": [
        (56, 128, 7, 7, 1), (28, 256, 7, 7, 1), (14, 512, 7, 7, 1),
        (7, 1024, 7, 7, 1),
    ],
    "convnext_l": [
        (56, 192, 7, 7, 1), (28, 384, 7, 7, 1), (14, 768, 7, 7, 1),
        (7, 1536, 7, 7, 1),
    ],
    "convnextv2_atto": [
        (56, 40, 7, 7, 1), (28, 80, 7, 7, 1), (14, 160, 7, 7, 1),
        (7, 320, 7, 7, 1),
    ],
    "convnextv2_pico": [
        (56, 64, 7, 7, 1), (28, 128, 7, 7, 1), (14, 256, 7, 7, 1),
        (7, 512, 7, 7, 1),
    ],
    "convnextv2_huge": [
        (56, 352, 7, 7, 1), (28, 704, 7, 7, 1), (14, 1408, 7, 7, 1),
        (7, 2816, 7, 7, 1),
    ],
    "replknet_31b": [
        (56, 128, 31, 31, 1), (28, 256, 31, 31, 1), (14, 512, 31, 31, 1),
        (7, 1024, 31, 31, 1), (56, 128, 5, 5, 1), (28, 256, 5, 5, 1),
    ],
    "replknet_31l": [
        (56, 192, 31, 31, 1), (28, 384, 31, 31, 1), (14, 768, 31, 31, 1),
        (7, 1536, 31, 31, 1),
    ],
    "slak_t": [
        (56, 96, 51, 5, 1), (56, 96, 5, 51, 1), (28, 192, 51, 5, 1),
        (28, 192, 5, 51, 1), (14, 384, 51, 5, 1), (14, 384, 5, 51, 1),
        (7, 768, 51, 5, 1), (7, 768, 5, 51, 1),
    ],
    "unireplknet_n": [
        (56, 80, 13, 13, 1), (28, 160, 13, 13, 1), (14, 320, 13, 13, 1),
        (7, 640, 13, 13, 1), (28, 160, 3, 3, 1), (14, 320, 3, 3, 1),
    ],
    "segnext_b": [
        (56, 64, 5, 5, 1), (56, 64, 1, 7, 1), (56, 64, 7, 1, 1),
        (28, 128, 1, 11, 1), (28, 128, 11, 1, 1), (14, 320, 1, 21, 1),
        (14, 320, 21, 1, 1), (7, 512, 1, 21, 1), (7, 512, 21, 1, 1),
    ],
    "edgenext_s": [
        (56, 48, 7, 7, 1), (28, 96, 7, 7, 1), (14, 160, 9, 9, 1),
        (7, 304, 9, 9, 1),
    ],
    "mobilevit_s": [
        (112, 32, 3, 3, 1), (56, 64, 3, 3, 2), (28, 128, 3, 3, 2),
        (14, 256, 3, 3, 2),
    ],
    "yolo11": [
        (80, 128, 3, 3, 1), (40, 256, 3, 3, 1), (20, 512, 3, 3, 1),
        (80, 64, 3, 3, 2), (40, 128, 3, 3, 2), (20, 256, 3, 3, 2),
        (20, 1024, 3, 3, 1),
    ],
    "yolo12": [
        (80, 128, 7, 7, 1), (40, 256, 7, 7, 1), (20, 512, 7, 7, 1),
        (160, 64, 3, 3, 2), (80, 256, 3, 3, 1), (20, 512, 5, 5, 1),
    ],
}

# Batch sizes the model stages are replicated over. N is the single strongest
# lever on the occupancy term -- a shape that wants Gm=4 at N=1 can want 32 at
# N=128 -- so the corpus has to carry it.
_MODEL_BATCHES = (1, 8, 32, 128)


def model_shapes():
    out = []
    for name, stages in _MODELS.items():
        for i, (res, ch, y, x, s) in enumerate(stages):
            for n in _MODEL_BATCHES:
                sh = Shape(n, res, res, ch, y, x, s, dtype="bf16",
                           source="model", tag=f"{name}.{i}")
                if sh.valid():
                    out.append(sh)
    return out


# ---------------------------------------------------------------------------
# Source 2: systematic grid over the axes the models leave thin
# ---------------------------------------------------------------------------

_G_AXIS = (8, 16, 24, 32, 48, 64, 72, 96, 128, 160, 192, 256, 320, 384,
           512, 576, 640, 768, 960, 1024, 1280, 1536, 2048)
_SPATIAL = (4, 6, 7, 10, 13, 16, 20, 26, 32, 40, 52, 64, 80, 104, 128, 160, 224)
_FILTERS = ((3, 3), (5, 5), (7, 7), (9, 9), (11, 11), (13, 13), (15, 15),
            (21, 21), (31, 31), (1, 3), (3, 1), (1, 7), (7, 1), (1, 11),
            (11, 1), (3, 5), (5, 3), (7, 3), (5, 51), (51, 5))
_N_AXIS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
_DTYPES = ("bf16", "fp16")


def grid_shapes(count, seed):
    """Random but stratified: every draw is rejected unless it lands in a
    (G, filter-area, N, stride) cell that is not already over-represented, so
    the grid spreads instead of clumping where the valid region is widest."""
    rng = random.Random(seed)
    out, seen, cells = [], set(), {}
    tries = 0
    while len(out) < count and tries < count * 400:
        tries += 1
        n = rng.choice(_N_AXIS)
        g = rng.choice(_G_AXIS)
        y, x = rng.choice(_FILTERS)
        s = rng.choice((1, 1, 2, 2, 4))
        hi = rng.choice(_SPATIAL)
        # Non-square inputs on purpose: detection backbones are 4:3 or 16:9 and
        # Ho*Wo, not Hi, is what the M axis sees.
        wi = hi if rng.random() < 0.65 else rng.choice(_SPATIAL)
        d = 2 if rng.random() < 0.08 else 1
        dtype = _DTYPES[0] if rng.random() < 0.75 else _DTYPES[1]

        sh = Shape(n, hi, wi, g, y, x, s, dh=d, dtype=dtype,
                   source="grid", tag="")
        if not sh.valid() or sh.key() in seen:
            continue
        cell = (g, y * x, n, s)
        if cells.get(cell, 0) >= 2:
            continue
        cells[cell] = cells.get(cell, 0) + 1
        seen.add(sh.key())
        sh.tag = f"g{g}_{y}x{x}_n{n}_s{s}"
        out.append(sh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=400,
                    help="total shapes to emit (model stages first, grid fills)")
    ap.add_argument("--seed", type=int, default=20251001)
    ap.add_argument("-o", "--out", default="shapes_fwd.txt")
    ap.add_argument("--model-cap", type=int, default=200,
                    help="max model-derived cases before the grid takes over")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    models = model_shapes()
    rng.shuffle(models)
    # Deduplicate: architectures share stages (many are 3x3 @ 14x14 x 512).
    seen, uniq = set(), []
    for s in models:
        if s.key() in seen:
            continue
        seen.add(s.key())
        uniq.append(s)
    models = uniq[: args.model_cap]

    grid = [s for s in grid_shapes(args.count * 2, args.seed) if s.key() not in seen]
    grid = grid[: max(0, args.count - len(models))]

    allsh = models + grid
    with open(args.out, "w") as fh:
        fh.write("# Copyright (c) Advanced Micro Devices, Inc., or its"
                 " affiliates.\n# SPDX-License-Identifier: MIT\n")
        fh.write("# depthwise forward shape corpus -- generated by gen_shapes.py\n")
        fh.write(f"# {len(models)} model stages + {len(grid)} grid points "
                 f"= {len(allsh)} cases\n")
        for s in allsh:
            fh.write(s.line() + "\n")

    print(f"wrote {len(allsh)} cases to {args.out} "
          f"({len(models)} model, {len(grid)} grid)")
    gs = sorted({s.g for s in allsh})
    print(f"  G       : {len(gs)} distinct, {gs[0]}..{gs[-1]}")
    print(f"  filters : {len(sorted({(s.y, s.x) for s in allsh}))} distinct")
    print(f"  N       : {sorted({s.n for s in allsh})}")
    print(f"  strides : {sorted({s.sh for s in allsh})}")
    print(f"  dilation: {sorted({s.dh for s in allsh})}")
    print(f"  dtypes  : {sorted({s.dtype for s in allsh})}")
    mt = sorted(s.n * s.ho * s.wo for s in allsh)
    print(f"  M       : {mt[0]} .. {mt[-1]} (median {mt[len(mt)//2]})")


if __name__ == "__main__":
    main()
