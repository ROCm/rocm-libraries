# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device knob sets of the small-tile / occupancy region of the backward.

Each set is a tile (``kM0 x kN0`` on ``W`` waves), its warp grids (G0 / G2,
G1 / G3, G4; see :mod:`kernels.common._attention_bwd_tiles`), the operand
route, the K / V and K^T residency, an occupancy hint and the head sizes it
runs at. ``test_attention_bwd_tiles.py`` checks on the CPU that every set is
legal and built on gfx942 and gfx950 at each of its head sizes;
``test_attention_bwd_tiles_numeric.py`` runs them on the device.

``BITWISE_GROUPS``: sets of one tile whose grids differ only in how the S /
dP fragments (or the dK / dV fragments) are spread over the waves. Every
fragment is computed by the same matrix-op chain whichever wave owns it, so
direct-mode dK / dV are bit for bit equal across a group.
"""

from __future__ import annotations

__all__ = [
    "BITWISE_GROUPS",
    "REGION_CASES",
    "REGION_SETS",
    "region_knobs",
]

# name -> (kM0, kN0, W, G0/G2 grid, G1/G3 grid, G4 grid, pt_route,
#          (kv_residency, kt_source), waves_per_eu, head sizes)
REGION_SETS = {
    # replicated G0 / G2 grids: fewer 16 x 16 S fragments than waves
    "rep16x16_w2": (16, 16, 2, (1, 1), (1, 2), (1, 2), "lds", ("reg", "reg"), None, (64, 128)),
    "rep16x16_w4": (16, 16, 4, (1, 1), (1, 4), (1, 4), "lds", ("reg", "reg"), None, (64, 128)),
    "rep32x16_w4": (32, 16, 4, (2, 1), (1, 4), (1, 4), "lds", ("reg", "reg"), None, (64, 128)),
    "rep16x32_w4": (16, 32, 4, (1, 2), (1, 4), (1, 4), "lds", ("reg", "reg"), None, (64, 128)),
    # transposed grids: waves split the q rows of S and the D columns of dK / dV
    "tr32x16_w2": (32, 16, 2, (2, 1), (1, 2), (1, 2), "lds", ("reg", "reg"), None, (64, 128)),
    "tr64x16_w4": (64, 16, 4, (4, 1), (1, 4), (2, 2), "lds", ("reg", "reg"), None, (64, 128)),
    "tr64x32_w4": (64, 32, 4, (4, 1), (1, 4), (1, 4), "lds", ("reg", "reg"), None, (64, 128)),
    # (2, W / 2) on a tile of fewer than 16 * W keys
    "alt32x32_w4": (32, 32, 4, (2, 2), (2, 2), (2, 2), "lds", ("reg", "reg"), None, (64, 128)),
    # the same tile, replicated and transposed variants (BITWISE_GROUPS)
    "alt32x32_w4_g02_12": (32, 32, 4, (1, 2), (2, 2), (2, 2), "lds", ("reg", "reg"), None, (64, 128)),
    "alt32x32_w4_g02_21": (32, 32, 4, (2, 1), (2, 2), (2, 2), "lds", ("reg", "reg"), None, (64, 128)),
    "alt32x32_w4_g02_11": (32, 32, 4, (1, 1), (2, 2), (2, 2), "lds", ("reg", "reg"), None, (64, 128)),
    "alt32x32_w4_g13_14": (32, 32, 4, (2, 2), (1, 4), (2, 2), "lds", ("reg", "reg"), None, (64, 128)),
    "tr64x16_w4_g02_21": (64, 16, 4, (2, 1), (1, 4), (2, 2), "lds", ("reg", "reg"), None, (64, 128)),
    "tr64x16_w4_g02_11": (64, 16, 4, (1, 1), (1, 4), (2, 2), "lds", ("reg", "reg"), None, (64, 128)),
    # occupancy hints above 1: the register estimate fits 512 / n and the LDS
    # plan holds n waves per SIMD (attn_bwd_occupancy) on both targets
    "occ16x16_w2_wpe4": (16, 16, 2, (1, 1), (1, 2), (1, 2), "lds", ("reg", "reg"), 4, (64,)),
    "occ16x16_w2_wpe3": (16, 16, 2, (1, 1), (1, 2), (1, 2), "lds", ("reg", "reg"), 3, (64, 128)),
    "occ16x16_w4_wpe4": (16, 16, 4, (1, 1), (1, 4), (1, 4), "lds", ("reg", "lds"), 4, (64, 128)),
    "occ32x16_w4_wpe4": (32, 16, 4, (2, 1), (1, 4), (1, 4), "lds", ("reg", "reg"), 4, (64,)),
    "occ16x32_w4_wpe3": (16, 32, 4, (1, 2), (1, 4), (1, 4), "lds", ("reg", "reg"), 3, (64, 128)),
    "occ16x32_w2_wpe3": (16, 32, 2, (1, 2), (2, 1), (1, 2), "relabel", ("reg", "reg"), 3, (64,)),
    "occ16x64_w4_wpe4": (16, 64, 4, (1, 4), (4, 1), (1, 4), "relabel", ("reg", "reg"), 4, (64,)),
    "occ16x64_w4_wpe2": (16, 64, 4, (1, 4), (4, 1), (1, 4), "relabel", ("reg", "reg"), 2, (64, 128)),
}  # fmt: skip

BITWISE_GROUPS = (
    (
        "alt32x32_w4",
        "alt32x32_w4_g02_12",
        "alt32x32_w4_g02_21",
        "alt32x32_w4_g02_11",
        "alt32x32_w4_g13_14",
    ),
    ("tr64x16_w4", "tr64x16_w4_g02_21", "tr64x16_w4_g02_11"),
)

# Cases the sets run on, by head size: an MHA band with edge lengths, a GQA
# causal group, a bf16 causal case, h_k != h_v (atomic dK / dV) and THD.
REGION_CASES = {
    64: ("bwd_seqlen_33x72_br", "bwd_kern_hkv_842_tl_fp16_d64"),
    128: ("bwd_gqa_gqa4_causal_tl", "bwd_dtype_bf16_d128_causal_tl"),
}


def region_knobs(name: str) -> dict:
    """Spec knobs of set ``name`` (``block_k4`` and the G4 atom follow kN0)."""
    km, kn, w, g02, g13, g4, pt, (kv, kt), wpe, _heads = REGION_SETS[name]
    knobs = dict(
        waves=w,
        block_m=km,
        block_n=kn,
        block_k4=32 if kn % 32 == 0 else 16,
        warp_grid_g02=g02,
        warp_grid_g13=g13,
        warp_grid_g4=g4,
        pt_route=pt,
        kv_residency=kv,
        kt_source=kt,
        waves_per_eu=wpe,
    )
    if kn % 32:
        knobs["atom_g4"] = "16x16x16"
    return knobs
