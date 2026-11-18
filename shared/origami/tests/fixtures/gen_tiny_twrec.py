#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
r"""Generate the tiny TWREC fixture used by the Origami NN tests.

Writes, next to this script under tilewright/:
    tiny_gfx950.tilewright.yaml       manifest
    tiny_gfx950.tilewright.wts.yaml   weights sidecar
    origami_nn_index                  maps the stem "tiny_gfx950_stem" to the manifest

The files are checked in. Run this script only to change the fixture, then re-run the tests:

    python3 gen_tiny_twrec.py

The model is built so that its score can be written down without running it. Every weight is
either zero or one, and the whitening is the identity (mean 0, std 1), so for a config c in a
cell with preferred axis p, offset o and temperature t:

    score(c) = log2(c.mt[p]) / max(|t|, 0.1) + o

  * The query network ends in q_w4 = 0, so the query embedding is just the bias q_b4, a one-hot
    vector that selects the preferred axis p.
  * The item network copies log2(mt.m), log2(mt.n) and log2(mt.k) (item features 0 to 2) into
    the first three embedding dimensions: hidden unit r reads item feature r, and embedding
    dimension r reads hidden unit r.
  * The interaction network has x_w2 = 0, so it contributes only its output bias x_b2[0] = o.

The tests use o to tell which cell scored a problem, and use the log2 term for ordering. All
three terms are exact in float for the power-of-two tiles the tests use.

Cells and routing (base label "Large|Large|LargeK|Bnone", that is M and N above 512, K above
512, batch 1):

    base --M<=2048--> A  "base#M<=2048"            prefers mt.m, t=1, o=0,   smart-K whitelist
         \--M>2048--> B  "base#M>2048"             prefers mt.n, t=2, o=100, no whitelist
                       --N<=2048--> "base#M>2048#N<=2048" has no cell, so it resolves to B
                       \-N>2048--> C "base#M>2048#N>2048" prefers mt.k, t=1, o=200,
                                                  a whitelist that matches no test config,
                                                  and a larger network (embed 24)

A problem in any other base cell, such as M <= 128, has no model cell and is left unscored.
"""

import base64
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tilewright")

STEM = "tiny_gfx950"
LOGIC_STEM = "tiny_gfx950_stem"

# gemm_tilewright feature layout, feature_names_hash e7fe4b524851e895.
ARCH = "gfx950"
FEATURE_HASH = "e7fe4b524851e895"
QD, ID, XD = 55, 12, 37

BASE = "Large|Large|LargeK|Bnone"
LABEL_A = BASE + "#M<=2048"
LABEL_B = BASE + "#M>2048"
LABEL_B_LO_N = BASE + "#M>2048#N<=2048"  # deliberately has no cell
LABEL_C = BASE + "#M>2048#N>2048"

# (embed, hidden, inter) must be one of the tiers the loader accepts.
TIERS = {16: (48, 24), 24: (80, 40)}

CELLS = [
    {
        "label": LABEL_A,
        "embed": 16,
        "temperature": 1.0,
        "axis": 0,
        "offset": 0.0,
        # mt.m, mt.n, mt.k, mi.m, mi.n, mi.k, cache_hints_a, cache_hints_b
        "smart_k": [
            (256, 256, 32, 16, 16, 32, 0, 0),
            (128, 128, 64, 16, 16, 32, 0, 0),
            (256, 128, 64, 16, 16, 32, 0, 0),
        ],
    },
    {
        "label": LABEL_B,
        "embed": 16,
        "temperature": 2.0,
        "axis": 1,
        "offset": 100.0,
        "smart_k": [],
    },
    {
        "label": LABEL_C,
        "embed": 24,
        "temperature": 1.0,
        "axis": 2,
        "offset": 200.0,
        "smart_k": [(64, 64, 64, 16, 16, 16, 0, 0)],
    },
]

SPLITS = [
    (BASE, "M", 2048, LABEL_A, LABEL_B),
    (LABEL_B, "N", 2048, LABEL_B_LO_N, LABEL_C),
]


def int4_b64(values):
    """TWREC int4 tensor: float32 scale, then nibbles, low nibble first. value = scale*(n-8)."""
    scale = 1.0
    nibbles = [int(v) + 8 for v in values]
    assert all(0 <= n <= 15 for n in nibbles)
    if len(nibbles) % 2:
        nibbles.append(8)
    packed = bytes(nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2))
    return base64.b64encode(struct.pack("<f", scale) + packed).decode("ascii")


def fp32_b64(values):
    return base64.b64encode(struct.pack("<%df" % len(values), *values)).decode("ascii")


def cell_tensors(cell):
    e = cell["embed"]
    h, ih = TIERS[e]
    zeros = lambda n: [0] * n

    q_b4 = [0.0] * e
    q_b4[cell["axis"]] = 1.0

    i_w0 = zeros(h * ID)  # [h][ID]
    for r in range(3):
        i_w0[r * ID + r] = 1
    i_w2 = zeros(e * h)  # [e][h]
    for j in range(3):
        i_w2[j * h + j] = 1

    x_b2 = [cell["offset"]]

    tensors = [
        ("q_w0", "int4", zeros(h * QD)),
        ("q_b0", "fp32", [0.0] * h),
        ("q_w2", "int4", zeros(h * h)),
        ("q_b2", "fp32", [0.0] * h),
        ("q_w4", "int4", zeros(e * h)),
        ("q_b4", "fp32", q_b4),
        ("i_w0", "int4", i_w0),
        ("i_b0", "fp32", [0.0] * h),
        ("i_w2", "int4", i_w2),
        ("i_b2", "fp32", [0.0] * e),
        ("x_w0", "int4", zeros(ih * XD)),
        ("x_b0", "fp32", [0.0] * ih),
        ("x_w2", "int4", zeros(ih)),
        ("x_b2", "fp32", x_b2),
    ]
    return tensors


def flow(values):
    return "[" + ", ".join(str(v) for v in values) + "]"


def manifest_text():
    out = []
    out.append("# Tiny TWREC manifest for the Origami NN tests. Generated by gen_tiny_twrec.py.")
    out.append("schema_version: 1")
    out.append("format: TWREC_v1")
    out.append("metadata:")
    out.append("  arch: %s" % ARCH)
    out.append("  feature_names_hash: %s" % FEATURE_HASH)
    out.append("features:")
    out.append("  query_dim: %d" % QD)
    out.append("  item_dim: %d" % ID)
    out.append("  interaction_dim: %d" % XD)
    out.append("weights_sidecar: %s.tilewright.wts.yaml" % STEM)
    out.append("routing:")
    out.append("  splits:")
    for parent, axis, thr, lo, hi in SPLITS:
        out.append('  - parent: "%s"' % parent)
        out.append("    axis: %s" % axis)
        out.append("    threshold: %d" % thr)
        out.append('    lo: "%s"' % lo)
        out.append('    hi: "%s"' % hi)
    out.append("cells:")
    for cell in CELLS:
        e = cell["embed"]
        h, ih = TIERS[e]
        out.append('- label: "%s"' % cell["label"])
        out.append("  embed_dim: %d" % e)
        out.append("  hidden_dim: %d" % h)
        out.append("  inter_hidden: %d" % ih)
        out.append("  temperature: %s" % cell["temperature"])
        out.append("  whitening:")
        for name, dim in (("query", QD), ("item", ID), ("interaction", XD)):
            out.append("    %s:" % name)
            out.append("      mean: %s" % flow([0] * dim))
            out.append("      std: %s" % flow([1] * dim))
        if cell["smart_k"]:
            out.append("  smart_k:")
            for sig in cell["smart_k"]:
                out.append("  - %s" % flow(sig))
        else:
            out.append("  smart_k: []")
    return "\n".join(out) + "\n"


def sidecar_text():
    out = []
    out.append("# Tiny TWREC weights for the Origami NN tests. Generated by gen_tiny_twrec.py.")
    out.append("schema_version: 1")
    out.append("format: TWREC_WTS_v1")
    out.append("encoding: int4_b64_fp32_b64")
    out.append("cells:")
    for cell in CELLS:
        out.append('- label: "%s"' % cell["label"])
        out.append("  tensors:")
        for name, dtype, values in cell_tensors(cell):
            out.append("    %s:" % name)
            out.append("      dtype: %s" % dtype)
            enc = int4_b64(values) if dtype == "int4" else fp32_b64(values)
            out.append("      data_b64: %s" % enc)
    return "\n".join(out) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    files = {
        "%s.tilewright.yaml" % STEM: manifest_text(),
        "%s.tilewright.wts.yaml" % STEM: sidecar_text(),
        "origami_nn_index": "# Fixture index: logic stem, backend, manifest.\n"
        "%s tilewright %s.tilewright.yaml\n" % (LOGIC_STEM, STEM),
    }
    for name, text in files.items():
        with open(os.path.join(OUT, name), "w", newline="\n") as f:
            f.write(text)
        print("wrote %s (%d bytes)" % (os.path.join(OUT, name), len(text)))


if __name__ == "__main__":
    main()
