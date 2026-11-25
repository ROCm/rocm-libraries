"""Generate the SdpaBackward spaces of hipkernel:flydsl_sdpa_bwd_split.

    python gauntlet/generators/flydsl_sdpa_bwd_spaces.py [OUT_DIR]

Writes SdpaBackward.FlydslBwdSplit{,Bias,Padded,Dropout,PaddedDropout,Long,Ragged,Stride,Probe}.space.json to OUT_DIR
(default: gauntlet/spaces). The shipped files are this script's output: change the script, regenerate, then re-pin
(pairwise_pins.json). The probe space starts from SdpaBackward.AsmProbe.space.json in gauntlet/spaces.
"""

import json
import sys
from pathlib import Path

SPACES_DIR = Path(__file__).resolve().parent.parent / "spaces"
ENGINE = "hipkernel:flydsl_sdpa_bwd_split"
TEMPLATE = (
    "../../integration-test-bundles/quick/SdpaBackward/Default/graph.template.json"
)

COST = {
    "rule": "b*hq*sq*skv*(dqk+dv) <= 2**29",
    "reason": "GPU reference cost ceiling, as SdpaBackward.space.json: the case time is linear above 2**26 and the CPU backward reference is serial, so 2**29 keeps the worst drawn case near 24 s (measured on a gfx942 host for the ASM region space)",
}
ROWS_SEE = {
    "rule": "rows_see_a_key",
    "reason": "a query row with no key left under the mask has an LSE of -inf in the derived stats (CpuFpReferenceSdpa.hpp forward, 'fully masked rows'), which is not an input the backward kernels define; SdpaBackward/Default excludes the same samples. A key that no row sees is fine: its dK and dV are zero",
}

DOMAIN = {
    "dom_kind": "'fixed' if domain == 'constant' else ('free' if domain in ['default', 'narrow', 'wide', 'positive'] else domain)",
    "dom_lo": "-0.01 if domain == 'narrow' else (-8 if domain == 'wide' else (0.25 if domain == 'positive' else (0.5 if domain == 'constant' else -1)))",
    "dom_hi": "0.01 if domain == 'narrow' else (8 if domain == 'wide' else (4 if domain == 'positive' else (0.5 if domain == 'constant' else 1)))",
    "dom_value": "0.5 if domain == 'constant' else 0",
    "dom_scale": "8 if domain == 'large' else 1",
    "dom_g_lo": "-1 if domain == 'wide' else dom_lo",
    "dom_g_hi": "1 if domain in ['wide', 'positive'] else dom_hi",
    "eff_placement": "'exact' if placement == 'under' and alignment < 2 else placement",
}
MASK = {
    "bottom": "mask == 'causal_bottom_right' or (mask == 'window' and walign == 'bottom_right')",
    "align": "'BOTTOM_RIGHT' if bottom else 'TOP_LEFT'",
    "left": "wl if mask == 'window' else None",
    "right": "0 if mask in ['causal_top_left', 'causal_bottom_right'] else (wr if mask == 'window' else None)",
    "diag": "skv - sq if bottom else 0",
    "rows_see_a_key": "(right == None or diag + right >= 0) and (left == None or sq + diag - left <= skv - 1)",
}


def row(layout_expr_bshd, layout_expr_bhsd):
    return f"{layout_expr_bshd} if layout == 'bshd' else {layout_expr_bhsd}"


def shape_derive(pad=False):
    """dims and strides of every tensor. hq = hkv * hvmul * group; K has hkv heads, V has hv."""
    d = {
        "hv": "hkv * hvmul",
        "hq": "hkv * hvmul * group",
        "dqk": "hd[0]",
        "dv": "hd[1]",
        "dims_q": "[b, hq, sq, dqk]",
        "dims_k": "[b, hkv, skv, dqk]",
        "dims_v": "[b, hv, skv, dv]",
        "dims_o": "[b, hq, sq, dv]",
        "dims_lse": "[b, hq, sq, 1]",
    }
    p = {t: (f"pad_{t}" if pad else "0") for t in "qkvo"}
    for t, h, s, dd in (
        ("q", "hq", "sq", "dqk"),
        ("k", "hkv", "skv", "dqk"),
        ("v", "hv", "skv", "dv"),
        ("o", "hq", "sq", "dv"),
    ):
        d[f"row_{t}"] = f"({h}*{dd} if layout == 'bshd' else {dd}) + {p[t]}"
        d[f"stride_{t}"] = (
            f"[{s}*row_{t}, {dd}, row_{t}, 1] if layout == 'bshd' else [{h}*{s}*row_{t}, {s}*row_{t}, row_{t}, 1]"
        )
    d["stride_lse"] = (
        "[hq*(sq + pad_stats), sq + pad_stats, 1, 1]" if pad else "[hq*sq, sq, 1, 1]"
    )
    return d


def tensors():
    ts = [
        (0, "dims_k", "stride_k", "dtype"),
        (1, "dims_o", "stride_o", "dtype"),
        (2, "dims_q", "stride_q", "dtype"),
        (3, "dims_k", "stride_k", "dtype"),
        (4, "dims_q", "stride_q", "dtype"),
        (5, "dims_v", "stride_v", "dtype"),
        (6, "dims_v", "stride_v", "dtype"),
        (7, "dims_o", "stride_o", "dtype"),
        (8, "dims_lse", "stride_lse", None),
    ]
    out = []
    for uid, dims, strides, dt in ts:
        out.append(
            {
                "uid": uid,
                "dims": "${%s}" % dims,
                "strides": "${%s}" % strides,
                "data_type": "${dtype}" if dt else "float",
                "alignment": 16,
            }
        )
    return out


def inputs(dom=True):
    def rec(g):
        lo, hi = ("dom_g_lo", "dom_g_hi") if g else ("dom_lo", "dom_hi")
        return {
            "kind": "${dom_kind}",
            "lo": "${%s}" % lo,
            "hi": "${%s}" % hi,
            "value": "${dom_value}",
            "scale": 1 if g else "${dom_scale}",
        }

    return {"2": rec(False), "0": rec(False), "6": rec(True), "7": rec(True)}


def attrs():
    return {
        "compute_data_type": "float",
        "parameters__alibi_mask": False,
        "parameters__attn_scale_value": "${scale}",
        "parameters__causal_mask": False,
        "parameters__causal_mask_bottom_right": False,
        "parameters__dropout_probability": None,
        "parameters__diagonal_alignment": "${align}",
        "parameters__left_bound": "${left}",
        "parameters__right_bound": "${right}",
    }


def align_patches():
    return [{"uid": u, "set": {"alignment": "${alignment}"}} for u in range(9)]


# 0.0001 is a tiny scale. The largest the matcher accepts, 2^20 (SC-5), is drawn by the probe (scale_limit, built only):
# its scores reach 1e9, where no backward that consumes an fp32 LSE it did not compute can be graded (see SCORE)
SCALES = [
    0.0883883461356163,
    0.07216878235340118,
    0.125,
    0.25,
    0.5,
    1.0,
    4.0,
    0.0,
    -0.125,
    0.0001,
]

# The backward computes P = exp(S - LSE) from a given fp32 LSE. Two correct fp32 computations of a score S differ by about
# |S| * 2^-24, and exp() turns that into the same relative error in P and in the gradients. So the oracle can only judge a
# case whose largest score keeps that below a tenth of the tolerance: |S| * 2^-24 <= atol / 10, i.e. |S| <= 168 (fp32,
# atol 1e-4), 8388 (fp16, 5e-3), 16777 (bf16, 1e-2). |S| is bounded by |scale| * dqk * mag^2 (+ |bias|), with mag the largest
# magnitude the input domain draws for Q and K.
DOM_MAG = (
    "0.5 if domain == 'constant' else (8 if domain in ['large', 'wide'] else (4 if domain == 'positive' else "
    "(0.01 if domain == 'narrow' else (0 if domain in ['zeros', 'subnormal'] else 1))))"
)
S_LIMIT = "(168 if dtype == 'float' else (8388 if dtype == 'half' else 16777))"
SCORE = {
    "rule": "s_max <= " + S_LIMIT,
    "reason": "the largest score |S| (bounded by |scale| * dqk * mag^2, plus |bias|) must keep |S| * 2^-24 below a tenth of the dtype's tolerance: the backward computes P = exp(S - LSE) from a given fp32 LSE, and two correct fp32 computations of S differ by |S| * 2^-24, which exp() passes on to P and the gradients. Beyond that bound the oracle cannot tell a correct engine from a wrong one (scale 2^20 with scores near 1e9 gave NaN in every engine that did not produce the LSE itself)",
}


def add_score_bound(sp, bias_mag="0", dqk="dqk"):
    """dqk: the head-dim term; the ragged space passes its largest head dim (192), a looser bound over fewer axes that
    the pairwise search settles (with dqk itself two (b, gaps) pairs stay unproven at the search cap).
    """
    sp["derive"]["dom_mag"] = DOM_MAG
    sp["derive"]["s_max"] = f"abs(scale) * {dqk} * dom_mag * dom_mag + {bias_mag}"
    sp["constraints"].append(dict(SCORE))


HD_EQ = [
    [8, 8],
    [16, 16],
    [24, 24],
    [32, 32],
    [40, 40],
    [48, 48],
    [56, 56],
    [64, 64],
    [64, 64],
    [72, 72],
    [80, 80],
    [96, 96],
    [104, 104],
    [112, 112],
    [120, 120],
    [128, 128],
    [128, 128],
    [136, 136],
    [144, 144],
    [160, 160],
    [192, 192],
    [224, 224],
    [256, 256],
    [256, 256],
    [264, 264],
    [320, 320],
    [384, 384],
    [448, 448],
    [512, 512],
]
HD_ASYM = [
    [192, 128],
    [128, 64],
    [64, 128],
    [256, 128],
    [128, 192],
    [136, 128],
    [512, 256],
    [256, 512],
    # a tiny dim beside one served only by the padded tile-512 row (ladder has no 16-bit 384 rung on gfx90a), and
    # the prep boundaries of dv at 224/232 and 256/264 against a small and a large dqk (scout findings)
    [8, 264],
    [264, 8],
    [16, 320],
    [320, 16],
    [40, 384],
    [104, 448],
    [320, 256],
    [256, 320],
    [8, 512],
    [512, 8],
    [8, 224],
    [8, 232],
    [512, 224],
    [512, 232],
    [512, 264],
    [264, 512],
]


def region():
    axes = {
        "dtype": {"values": ["half", "bfloat16", "float"]},
        "layout": {"values": ["bshd", "bhsd"]},
        "b": {"values": [1, 2, 3]},
        "hkv": {"values": [1, 2, 4]},
        "group": {"values": [1, 1, 1, 2, 4, 8, 3, 5, 6, 16]},
        "hvmul": {"values": [1, 1, 1, 1, 2]},
        "hd": {"values": HD_EQ + HD_ASYM},
        "sq": {
            "range": [1, 512],
            "boundaries": [
                1,
                2,
                15,
                16,
                17,
                63,
                64,
                65,
                127,
                128,
                129,
                192,
                256,
                257,
                512,
            ],
        },
        "skv": {
            "range": [1, 1024],
            "boundaries": [
                1,
                2,
                63,
                64,
                65,
                127,
                128,
                129,
                191,
                192,
                193,
                256,
                257,
                512,
                1024,
            ],
        },
        "mask": {
            "values": [
                "none",
                "none",
                "causal_top_left",
                "causal_bottom_right",
                "window",
            ]
        },
        "wl": {"range": [1, 512], "boundaries": [1, 2, 16, 17, 64, 65, 128, 192, 256]},
        "wr": {"values": [None, 0, 1, 16, 64, 128]},
        "walign": {"values": ["top_left", "bottom_right"]},
        "scale": {"values": SCALES},
        "domain": {
            "values": [
                "default",
                "narrow",
                "wide",
                "positive",
                "constant",
                "nonzero",
                "zeros",
                "large",
                "subnormal",
            ]
        },
        "alignment": {"values": [1, 2, 4, 8, 16, 32, 64, 128, 256]},
        "placement": {"values": ["default", "exact", "under"]},
    }
    derive = {}
    derive.update(shape_derive())
    derive.update(MASK)
    derive.update(DOMAIN)
    constraints = [
        ROWS_SEE,
        {
            "rule": "hvmul == 1 or (dtype != 'float' and dqk == dv and dqk in [64, 128])",
            "reason": "Hk != Hv is served on the split engine for 16-bit data at head dims 64 and 128 only, with no bias and no ragged tensor (COVERAGE.md 9.1 SH-1, table 9.2); SdpaBackward/FlydslBwdSplitProbe draws v_heads, v_heads_f32 and v_heads_bias",
        },
        COST,
    ]
    return {
        "version": 1,
        "operation": "SdpaBackward",
        "description": "SDPA backward in the region hipkernel:flydsl_sdpa_bwd_split serves on gfx942 and gfx90a (FlydslSdpaBwdCommon.hpp, FlydslSdpaBwdSplitNative.cpp; COVERAGE.md section 9). Data half, bfloat16 or float (f32 is a native f32 MFMA family on the split engine); Q, K, V, O, dO and the three gradients are each BSHD or BHSD (one layout axis); head dims dqk and dv in the `hd` axis, both up to 512 and a multiple of 8, equal or different (DT-4, DT-5; a dim that is not a multiple of 8 is declined at a compact pitch, LY-3, and drawn by SdpaBackward/FlydslBwdSplitProbe); any group size (DT-6); V with its own head count (hvmul 2 over Hk, no bias, 16-bit, head dim 64 or 128, SH-1); causal top-left and bottom-right and windows in cuDNN's convention (a left bound of 0 is declined, MW-9; the `rows_see_a_key` constraint drops draws where a query row sees no key); scale zero and negative included (SC-1); declared alignment (below 16 bytes the engine declines, LY-1), placed default, exact or below it (expects BAD_PARAM). The O and the LSE are derived from the forward by the harness. Bias, dBias, dropout, ragged tensors and padding are in the companion spaces: SdpaBackward/FlydslBwdSplitBias, and the probe. Written for gfx942 and gfx90a (the arches the PR ports the engine to); the engine also runs on gfx950 and gfx11. Pairwise coverage proves every value pair; random mode is the like-for-like comparison across builds.",
        "engine": ENGINE,
        "template": TEMPLATE,
        "axes": axes,
        "derive": derive,
        "constraints": [
            c if c is not COST else {"rule": COST["rule"], "reason": COST["reason"]}
            for c in constraints
        ],
        "case": {
            "values": {
                "io_data_type": "${dtype}",
                "tensors": tensors(),
                "attributes": attrs(),
            },
            "metadata": {
                "purpose": "probe",
                "validator": "default",
                "inputs": inputs(),
                "placement": "${eff_placement}",
                "placement_uid": 2,
            },
            "node_patches": [],
            "tensor_patches": align_patches(),
        },
    }


def padded():
    """Padding mask with both seq_len tensors (PV-1): per-batch valid lengths from pad_q / pad_kv."""
    r = region()
    a = r["axes"]
    r["axes"] = {
        "dtype": {"values": ["half", "bfloat16", "float"]},
        "layout": a["layout"],
        "b": a["b"],
        "hkv": a["hkv"],
        "group": a["group"],
        "hvmul": {"values": [1]},
        "hd": {
            "values": [
                [16, 16],
                [32, 32],
                [64, 64],
                [64, 64],
                [96, 96],
                [128, 128],
                [128, 128],
                [136, 128],
                [192, 192],
                [256, 256],
                [320, 320],
                [512, 512],
                [192, 128],
            ]
        },
        "sq": {
            "range": [1, 512],
            "boundaries": [1, 2, 16, 17, 63, 64, 65, 128, 129, 256, 512],
        },
        "skv": {
            "range": [1, 1024],
            "boundaries": [1, 2, 63, 64, 65, 128, 129, 192, 256, 512, 1024],
        },
        "mask": {"values": ["none", "none", "causal_top_left", "causal_bottom_right"]},
        "pad_q": {"values": [0, 1, 3, 17, 64]},
        "pad_kv": {"values": [0, 1, 3, 17, 64]},
        "scale": a["scale"],
        "domain": a["domain"],
        "alignment": a["alignment"],
        "placement": a["placement"],
    }
    d = r["derive"]
    d.update(
        {
            "lq0": "sq",
            "lq1": "max(1, sq - pad_q)",
            "lq2": "max(1, sq - 2 * pad_q)",
            "lk0": "skv",
            "lk1": "max(1, skv - pad_kv)",
            "lk2": "max(1, skv - 2 * pad_kv)",
            "lens_q": "[lq0] if b == 1 else ([lq0, lq1] if b == 2 else [lq0, lq1, lq2])",
            "lens_kv": "[lk0] if b == 1 else ([lk0, lk1] if b == 2 else [lk0, lk1, lk2])",
            "pad_ok": "(mask == 'none') or ((lq0 <= lk0) and (b < 2 or lq1 <= lk1) and (b < 3 or lq2 <= lk2))",
            "bottom": "mask == 'causal_bottom_right'",
            "left": "None",
            "right": "0 if mask in ['causal_top_left', 'causal_bottom_right'] else None",
        }
    )
    d.pop("diag", None)
    d.pop("rows_see_a_key", None)
    # keep the key-seen rule only where the diagonal is tied to the valid lengths
    d["rows_see_a_key"] = "pad_ok"
    r["constraints"] = [
        {
            "rule": "rows_see_a_key",
            "reason": "a causal mask needs every valid query row to see a valid key: top-left needs the valid query length to be at most the valid key length, and bottom-right as well (a query row with no key has an LSE of -inf, which the backward kernels do not define); the lengths of a batch entry are the full extents less pad_q or pad_kv per batch index (sequence index 1 and 2 get one and two pads)",
        },
        COST,
    ]
    c = r["case"]
    c["values"]["tensors"] = c["values"]["tensors"]
    c["values"]["attributes"]["parameters__left_bound"] = "${left}"
    c["metadata"]["inputs"]["9"] = {
        "kind": "values",
        "values": "${lens_kv}",
        "repeat": False,
    }
    c["metadata"]["inputs"]["10"] = {
        "kind": "values",
        "values": "${lens_q}",
        "repeat": False,
    }
    c["add_tensors"] = [
        {
            "uid": 9,
            "name": "z_seq_len_kv",
            "data_type": "int32",
            "dims": "${[b, 1, 1, 1]}",
            "strides": [1, 1, 1, 1],
            "alignment": 16,
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
        {
            "uid": 10,
            "name": "z_seq_len_q",
            "data_type": "int32",
            "dims": "${[b, 1, 1, 1]}",
            "strides": [1, 1, 1, 1],
            "alignment": 16,
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
    ]
    c["node_patches"] = [
        {"node": "SdpaBwd_0", "section": "parameters", "set": {"padding_mask": True}},
        {
            "node": "SdpaBwd_0",
            "section": "inputs",
            "set": {"seq_len_q_tensor_uid": 10, "seq_len_kv_tensor_uid": 9},
        },
    ]
    r["description"] = (
        "SDPA backward with padding_mask and both seq_len tensors over hipkernel:flydsl_sdpa_bwd_split (COVERAGE.md 9.1 PV-1; 9.3 padded graphs): batch entry i has valid lengths max(1, Sq - i * pad_q) and max(1, Skv - i * pad_kv) (INT32 [B, 1, 1, 1], exact `values` fills, so metadata format_version 2); the engine zeroes dQ rows at or past seq_len_q and dK/dV rows at or past seq_len_kv, as the references do. Masks none, top-left and bottom-right causal (a window with bottom-right alignment under padding is declined, MW-13, and bias or dBias with padding, PV-10: not drawn). O and the LSE come from the harness's forward derivation, which models padding. The seq_len tensors are named z_seq_len_q and z_seq_len_kv so their canonical uids (sorted name, bundle_utils.canonical_uid_map_by_name) are 10 and 9."
    )
    return r


def dropout():
    """Philox dropout (probability, seed, offset): the harness draws the keep mask (SEM-015)."""
    r = region()
    a = r["axes"]
    ax = {k: a[k] for k in ("dtype", "layout", "b", "hkv", "group")}
    ax["hvmul"] = {"values": [1]}
    ax["hd"] = {
        "values": [
            [16, 16],
            [32, 32],
            [64, 64],
            [64, 64],
            [96, 96],
            [128, 128],
            [128, 128],
            [136, 128],
            [192, 192],
            [256, 256],
            [320, 320],
            [512, 512],
            [192, 128],
        ]
    }
    ax["sq"] = {
        "range": [1, 512],
        "boundaries": [1, 2, 3, 4, 5, 15, 16, 17, 63, 64, 65, 129, 256, 512],
    }
    ax["skv"] = {
        "range": [1, 1024],
        "boundaries": [1, 2, 3, 4, 5, 7, 8, 9, 63, 64, 65, 129, 256, 1024],
    }
    for k in ("mask", "wl", "wr", "walign"):
        ax[k] = a[k]
    ax["p"] = {"values": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.0, 0.999, 0.9999]}
    ax["seed"] = {
        "values": [0, 1, 12345, 305419896, 81985529216486895, 9223372036854775807, -7]
    }
    ax["offset"] = {
        "values": [0, 1, 7, 1000, 4294967293, 1099511627776, 9223372036854775807, -1]
    }
    for k in ("scale", "domain", "alignment", "placement"):
        ax[k] = a[k]
    r["axes"] = ax
    r["constraints"] = [
        c
        for c in r["constraints"]
        if c.get("rule")
        != "hvmul == 1 or (dtype != 'float' and dqk == dv and dqk in [64, 128])"
    ]
    c = r["case"]
    c["metadata"]["inputs"]["9"] = {
        "kind": "values",
        "values": "${[offset]}",
        "repeat": False,
    }
    c["metadata"]["inputs"]["10"] = {
        "kind": "values",
        "values": "${[seed]}",
        "repeat": False,
    }
    c["add_tensors"] = [
        {
            "uid": 9,
            "name": "z_offset",
            "data_type": "int64",
            "dims": [1, 1, 1, 1],
            "strides": [1, 1, 1, 1],
            "alignment": 16,
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
        {
            "uid": 10,
            "name": "z_seed",
            "data_type": "int64",
            "dims": [1, 1, 1, 1],
            "strides": [1, 1, 1, 1],
            "alignment": 16,
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
    ]
    c["node_patches"] = [
        {
            "node": "SdpaBwd_0",
            "section": "parameters",
            "set": {"dropout_probability": "${p}"},
        },
        {
            "node": "SdpaBwd_0",
            "section": "inputs",
            "set": {"seed_tensor_uid": 10, "offset_tensor_uid": 9},
        },
    ]
    r["description"] = (
        "SDPA backward with Philox dropout (a probability, an INT64 seed and an INT64 offset as device tensors with exact `values` fills, so metadata format_version 2) over hipkernel:flydsl_sdpa_bwd_split (COVERAGE.md 9.1 DR-1 to DR-6). No reference reproduces an engine's random stream, but SEM-015 pins the mask bits (Philox4x32-10 keyed by the seed, counter offset + ((b * Hq + h) * Sq + i) * ceil(Skv / 4) + j / 4, kept iff the word at j % 4 read as int32 exceeds int32(f32(p - 0.5) * 2^32)), so the harness draws the keep mask itself (harness/input-init/PhiloxDropout.hpp): it derives O and the LSE from the forward over that mask with the scale 1 / (1 - p), and grades the engine against the explicit-mask backward gradient (scale 1 / (1 - p), scale inverse 1 - p), which is the exact gradient. The engine's gradient is correct only when its regenerated mask is that mask, so a mismatch in the mask bits shows as a WRONG over every probability. Seeds include negative and large values, offsets past 2^32 (the counter's high word). Masks none, causal (both alignments, MW-11 accepts dropout with bottom-right) and windows. The seed and offset tensors are named z_seed and z_offset so their canonical uids (sorted name, bundle_utils.canonical_uid_map_by_name) are 10 and 9."
    )
    return r


RAGGED_TEMPLATE = "../../integration-test-bundles/quick/SdpaBackward/Golden/Ragged/graph.template.json"


def ragged():
    """Packed (THD) ragged Q, K, V, O, dO, dQ, dK, dV and stats with seq_len tensors (PV-3, SS-4, SS-5)."""
    r = region()
    a = r["axes"]
    r["template"] = RAGGED_TEMPLATE
    r["axes"] = {
        "dtype": {"values": ["half", "bfloat16", "float"]},
        "b": a["b"],
        "hkv": a["hkv"],
        "group": {"values": [1, 1, 2, 4]},
        "hd": {
            "values": [
                [16, 16],
                [32, 32],
                [64, 64],
                [64, 64],
                [96, 96],
                [128, 128],
                [128, 128],
                [192, 192],
                [128, 64],
                [64, 128],
            ]
        },
        "sq": {
            "range": [1, 256],
            "boundaries": [1, 2, 16, 17, 63, 64, 65, 128, 129, 256],
        },
        "skv": {
            "range": [1, 256],
            "boundaries": [1, 2, 16, 17, 63, 64, 65, 128, 129, 256],
        },
        "mask": {"values": ["none", "none", "causal_top_left", "causal_bottom_right"]},
        "pad_q": {"values": [0, 1, 3, 17, 64]},
        "pad_kv": {"values": [0, 1, 3, 17, 64]},
        "gaps": {"values": ["compact", "gapped"]},
        "scale": a["scale"],
        "domain": a["domain"],
        "alignment": a["alignment"],
        "placement": a["placement"],
    }
    d = {"hvmul": "1", "layout": "'bshd'"}
    d.update(shape_derive())
    d.update(DOMAIN)
    d.update(
        {
            "bottom": "mask == 'causal_bottom_right'",
            "left": "None",
            "right": "0 if mask in ['causal_top_left', 'causal_bottom_right'] else None",
            "align": "'BOTTOM_RIGHT' if bottom else 'TOP_LEFT'",
            "lq0": "sq",
            "lq1": "max(1, sq - pad_q)",
            "lq2": "max(1, sq - 2 * pad_q)",
            "lk0": "skv",
            "lk1": "max(1, skv - pad_kv)",
            "lk2": "max(1, skv - 2 * pad_kv)",
            "lens_q": "[lq0] if b == 1 else ([lq0, lq1] if b == 2 else [lq0, lq1, lq2])",
            "lens_kv": "[lk0] if b == 1 else ([lk0, lk1] if b == 2 else [lk0, lk1, lk2])",
            "pad_ok": "(mask == 'none') or ((lq0 <= lk0) and (b < 2 or lq1 <= lk1) and (b < 3 or lq2 <= lk2))",
            # token starts of batch i: compact packs the sequences, gapped leaves each at its padded slot
            "cq1": "lq0 if gaps == 'compact' else sq",
            "cq2": "lq0 + lq1 if gaps == 'compact' else 2 * sq",
            "cq3": "lq0 + lq1 + lq2 if gaps == 'compact' else 3 * sq",
            "ck1": "lk0 if gaps == 'compact' else skv",
            "ck2": "lk0 + lk1 if gaps == 'compact' else 2 * skv",
            "ck3": "lk0 + lk1 + lk2 if gaps == 'compact' else 3 * skv",
            "f_q": "hq * dqk",
            "f_k": "hkv * dqk",
            "f_v": "hv * dv",
            "f_o": "hq * dv",
            "f_s": "hq",
            "stride_lse": "[sq * hq, 1, hq, 1]",
        }
    )

    def offs(c, f):
        return f"[0, {c}1 * {f}] if b == 1 else ([0, {c}1 * {f}, {c}2 * {f}] if b == 2 else [0, {c}1 * {f}, {c}2 * {f}, {c}3 * {f}])"

    for name, c, f in (
        ("off_q", "cq", "f_q"),
        ("off_k", "ck", "f_k"),
        ("off_v", "ck", "f_v"),
        ("off_o", "cq", "f_o"),
        ("off_do", "cq", "f_o"),
        ("off_dq", "cq", "f_q"),
        ("off_dk", "ck", "f_k"),
        ("off_dv", "ck", "f_v"),
        ("off_s", "cq", "f_s"),
    ):
        d[name] = offs(c, f)
    d.pop("rows_see_a_key", None)
    d["rows_see_a_key"] = "pad_ok"
    for k in (
        "row_q",
        "row_k",
        "row_v",
        "row_o",
        "stride_q",
        "stride_k",
        "stride_v",
        "stride_o",
    ):
        pass
    r["derive"] = d
    r["constraints"] = [
        {
            "rule": "rows_see_a_key",
            "reason": "a causal mask needs every valid query row to see a valid key (the valid lengths are the padded extents less pad_q or pad_kv per batch index); a query row with no key has an LSE of -inf, which the backward kernels do not define",
        },
        COST,
    ]
    c = r["case"]
    c["values"]["attributes"] = {
        "parameters__diagonal_alignment": "${align}",
        "parameters__left_bound": "${left}",
        "parameters__right_bound": "${right}",
    }
    ts = tensors()
    for t in ts:
        t.pop("alignment", None)
        if t["uid"] <= 8:
            t["ragged_offset_multiplier"] = 1
    for uid in (10, 11):  # seq_len_q, seq_len_kv
        ts.append(
            {
                "uid": uid,
                "dims": "${[b, 1, 1, 1]}",
                "strides": [1, 1, 1, 1],
                "data_type": "int32",
            }
        )
    for uid in range(16, 25):  # the nine element-offset tensors
        ts.append(
            {
                "uid": uid,
                "dims": "${[b + 1, 1, 1, 1]}",
                "strides": [1, 1, 1, 1],
                "data_type": "int32",
            }
        )
    c["values"]["tensors"] = ts
    inp = c["metadata"]["inputs"]
    inp["10"] = {"kind": "values", "values": "${lens_q}", "repeat": False}
    inp["11"] = {"kind": "values", "values": "${lens_kv}", "repeat": False}
    for uid, name in (
        (16, "off_q"),
        (17, "off_k"),
        (18, "off_v"),
        (19, "off_o"),
        (20, "off_do"),
        (21, "off_dq"),
        (22, "off_dk"),
        (23, "off_dv"),
        (24, "off_s"),
    ):
        inp[str(uid)] = {"kind": "values", "values": "${%s}" % name, "repeat": False}
    c["tensor_patches"] = [
        {"uid": u, "set": {"alignment": "${alignment}"}}
        for u in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    ]
    c["node_patches"] = [
        {
            "node": "SdpaBwd_0",
            "section": "parameters",
            "set": {"attn_scale_value": "${scale}"},
        }
    ]
    c.pop("add_tensors", None)
    r["description"] = (
        "SDPA backward on packed (THD) ragged tensors over hipkernel:flydsl_sdpa_bwd_split (COVERAGE.md 9.1 PV-3, SS-4, SS-5; 9.3 gap rows): Q, K, V, O, dO, dQ, dK, dV and the LSE are ragged, each with its own INT32 [B + 1, 1, 1, 1] element-offset tensor and the seq_len_q / seq_len_kv tensors, as the SdpaBackward/Golden/Ragged bundles declare them (the template is theirs: its tensors are not numbered canonically, so a failure of this space cannot be frozen into a regression bundle by import_graph.py; BSHD strides, LSE packed (T, H)). Batch entry i has valid lengths max(1, Sq - i * pad_q) and max(1, Skv - i * pad_kv); `gaps` packs the sequences back to back (compact) or leaves each in its padded slot (gapped, so gap rows exist between sequences). All offsets and lengths are exact `values` fills (metadata format 2). O and the LSE come from the harness's ragged forward derivation; the references run the dense padded rewrite. Masks none and causal; no bias, dropout or window with ragged (declined by the matcher or not modelled)."
    )
    return r


def bias():
    r = region()
    a = r["axes"]
    ax = {
        "dtype": {"values": ["half", "bfloat16", "float", "half", "bfloat16"]},
        "layout": a["layout"],
        "b": a["b"],
        "hkv": a["hkv"],
        "group": a["group"],
        "hd": {
            "values": [
                [16, 16],
                [32, 32],
                [64, 64],
                [64, 64],
                [64, 64],
                [72, 72],
                [96, 96],
                [128, 128],
                [128, 128],
                [128, 128],
                [160, 160],
                [192, 192],
                [256, 256],
                [320, 320],
                [512, 512],
                [128, 64],
                [64, 128],
                [192, 128],
            ]
        },
        "sq": a["sq"],
        "skv": {
            "range": [1, 1024],
            "boundaries": [
                1,
                7,
                8,
                9,
                15,
                16,
                17,
                63,
                64,
                65,
                127,
                128,
                129,
                192,
                255,
                256,
                512,
                1024,
            ],
        },
        "bias_dt": {"values": ["same", "same", "f32"]},
        "dbias": {"values": [False, True, True]},
        "bcast_b": {"values": [False, True]},
        "bcast_h": {"values": [False, True]},
        "bcast_q": {"values": [False, False, True]},
        # the rank of the bias: 4, or 3 / 2 (right-aligned to [B, H, Sq, Skv], batch and head broadcast); the bias value
        # range. A padded or overlapping row pitch is drawn by SdpaBackward/FlydslBwdSplitProbe (bias_pitch_pad, bias_overlap):
        # the shipped region spaces carry packed strides only
        "bias_rank": {"values": [4, 4, 4, 3, 2]},
        "bias_dom": {"values": ["default", "default", "wide", "big"]},
        "scale": a["scale"],
        "domain": a["domain"],
        "alignment": a["alignment"],
        "placement": a["placement"],
    }
    ax["hd"]["values"].append([80, 80])
    r["axes"] = ax
    d = {"hvmul": "1"}
    d.update(shape_derive())
    d.update(DOMAIN)
    d.update(
        {
            "right": "None",
            "left": "None",
            "align": "'TOP_LEFT'",
            "bias_dtype": "dtype if bias_dt == 'same' else 'float'",
            "bias_b": "1 if bcast_b else b",
            "bias_h": "1 if bcast_h else hq",
            "bias_q": "1 if bcast_q else sq",
            "pitch": "skv",
            "dims_bias": "[bias_b, bias_h, bias_q, skv] if bias_rank == 4 else ([bias_h, bias_q, skv] if bias_rank == 3 else [bias_q, skv])",
            "stride_bias": "[bias_h*bias_q*pitch, bias_q*pitch, pitch, 1] if bias_rank == 4 else ([bias_q*pitch, pitch, 1] if bias_rank == 3 else [pitch, 1])",
            "dims_dbias": "[bias_b, bias_h, bias_q, skv] if bias_rank == 4 else ([bias_h, bias_q, skv] if bias_rank == 3 else [bias_q, skv])",
            "bias_lo": "-1 if bias_dom == 'default' else (-8 if bias_dom == 'wide' else -1000)",
            "bias_hi": "1 if bias_dom == 'default' else (8 if bias_dom == 'wide' else 1000)",
            "bias_mag": "1 if bias_dom == 'default' else (8 if bias_dom == 'wide' else 1000)",
        }
    )
    r["derive"] = d
    r["constraints"] = [
        {
            "rule": "not (dbias and bcast_q)",
            "reason": "dBias with a bias broadcast over query rows is declined by every engine (BI-4); SdpaBackward/FlydslBwdSplitProbe draws dbias_rowbcast",
        },
        {
            "rule": "not dbias or skv % (8 if dtype != 'float' and bias_dt == 'same' else 4) == 0",
            "reason": "a dBias stride that is not a multiple of 16 bytes is declined (BI-9, [layout]); the bias itself may have any row pitch on the split engine (9.2), so skv is free without dBias. SdpaBackward/FlydslBwdSplitProbe draws dbias_unaligned",
        },
        {
            "rule": "bias_dt == 'same' or (dbias and dtype != 'float' and dqk == dv and dqk in [64, 80, 96, 128])",
            "reason": "an f32 bias beside 16-bit Q/K/V is served with an f32 dBias, at head dims 64 and 128 on the split engine (BI-2, 9.2); 80 and 96 are drawn too, because the descriptor listing shows f32-bias dQ rows only at d64, so a decline there is information and an accept must grade",
        },
        COST,
    ]
    r["description"] = (
        "SDPA backward with an additive bias and, on most draws, dBias over hipkernel:flydsl_sdpa_bwd_split on gfx942 and gfx90a (COVERAGE.md 9.1 and 9.2): a rank-4 bias of the data dtype or, beside 16-bit data, f32 with an f32 dBias (BI-2), broadcast over batch and head (BI-1, BI-3) or, without dBias, over query rows; any key length, so the bias row pitch is aligned or not (a 16-bit bias with an unaligned pitch is served by dQ rows that read it directly; with dBias the dBias stride must be a multiple of 16 bytes, BI-9, which the constraint keeps). No mask: a bias with a causal or window mask is declined (MW-10) and drawn by SdpaBackward/FlydslBwdSplitProbe. Head dims in the `hd` axis include 160 and the 256 and 512 tiles. The bias and dBias tensors are named z_attn_mask and z_dbias so that their canonical uids (by sorted name, bundle_utils.canonical_uid_map_by_name) are 9 and 10, after the template's stats. The O and the LSE come from the forward with the graph's own bias (CpuFpReferenceSdpa)."
    )
    r["case"]["values"]["attributes"]["parameters__left_bound"] = None
    r["case"]["values"]["attributes"]["parameters__right_bound"] = None
    r["case"]["values"]["attributes"]["parameters__diagonal_alignment"] = "TOP_LEFT"
    r["case"]["add_tensors"] = [
        {
            "uid": 9,
            "name": "z_attn_mask",
            "data_type": "${bias_dtype}",
            "dims": "${dims_bias}",
            "strides": "${stride_bias}",
            "alignment": "${alignment}",
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
        {
            "when": "${dbias}",
            "uid": 10,
            "name": "z_dbias",
            "data_type": "${bias_dtype}",
            "dims": "${dims_dbias}",
            "strides": "${stride_bias}",
            "alignment": "${alignment}",
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
    ]
    r["case"]["node_patches"] = [
        {"node": "SdpaBwd_0", "section": "inputs", "set": {"attn_mask_tensor_uid": 9}},
        {
            "when": "${dbias}",
            "node": "SdpaBwd_0",
            "section": "outputs",
            "set": {"dbias_tensor_uid": 10},
        },
    ]
    r["case"]["metadata"]["inputs"]["9"] = {
        "kind": "free",
        "lo": "${bias_lo}",
        "hi": "${bias_hi}",
        "value": 0,
        "scale": 1,
    }
    return r


def padded_dropout():
    """Philox dropout over a padded graph: the mask uses the declared extents (SEM-015, COVERAGE 9.3), the kernels' plane
    offsets may use the live ones (scout finding)."""
    r = padded()
    d = dropout()
    late = {k: r["axes"].pop(k) for k in ("scale", "domain", "alignment", "placement")}
    r["axes"]["p"] = {"values": [0.1, 0.25, 0.5, 0.9]}
    r["axes"]["seed"] = d["axes"]["seed"]
    r["axes"]["offset"] = {"values": [0, 1, 1000, 4294967293]}
    r["axes"].update(late)
    c = r["case"]
    # names sort after z_seq_len_*, so the canonical (name-sorted) uids stay 9, 10 for the lengths and 11, 12 here
    c["metadata"]["inputs"]["11"] = {
        "kind": "values",
        "values": "${[offset]}",
        "repeat": False,
    }
    c["metadata"]["inputs"]["12"] = {
        "kind": "values",
        "values": "${[seed]}",
        "repeat": False,
    }
    c["add_tensors"] += [
        {
            "uid": 11,
            "name": "z_zz_offset",
            "data_type": "int64",
            "dims": [1, 1, 1, 1],
            "strides": [1, 1, 1, 1],
            "alignment": 16,
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
        {
            "uid": 12,
            "name": "z_zz_seed",
            "data_type": "int64",
            "dims": [1, 1, 1, 1],
            "strides": [1, 1, 1, 1],
            "alignment": 16,
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
    ]
    c["node_patches"][0]["set"]["dropout_probability"] = "${p}"
    c["node_patches"][1]["set"].update({"seed_tensor_uid": 12, "offset_tensor_uid": 11})
    r["description"] = (
        "SDPA backward with Philox dropout (probability, INT64 seed and offset device tensors) AND padding (both seq_len "
        "tensors) over hipkernel:flydsl_sdpa_bwd_split. The harness draws the keep mask over the declared extents "
        "(SEM-015; padded and packed graphs follow the declared-extent layout, COVERAGE.md 9.3) and grades the "
        "engine against the explicit-mask gradient; a kernel that derives its Philox plane offsets from the live "
        "lengths instead disagrees with the oracle for every batch entry shorter than the declared extent. "
        "Otherwise as SdpaBackward/FlydslBwdSplitPadded (valid lengths max(1, Sq - i * pad_q) and max(1, Skv - i * pad_kv) "
        "for batch index i; masks none, top-left and bottom-right causal) and SdpaBackward/FlydslBwdSplitDropout."
    )
    return r


def long_kv():
    """Long key sequences with few query rows: softmax probabilities near 1/Skv fall below 2**-14, where the gfx90a MFMA flushes
    fp16 subnormals (COVERAGE.md 9.x, scout finding); also window and bottom-right bands far from the tile edges.
    """
    r = region()
    a = r["axes"]
    r["axes"] = {
        "dtype": {"values": ["half", "bfloat16", "float", "half"]},
        "layout": a["layout"],
        "b": {"values": [1, 2]},
        "hkv": {"values": [1, 2]},
        "group": {"values": [1, 2, 4]},
        "hvmul": {"values": [1]},
        "hd": {
            "values": [[16, 16], [32, 32], [64, 64], [128, 128], [64, 128], [128, 64]]
        },
        "sq": {"values": [1, 2, 3, 16, 17, 64, 128]},
        "skv": {"values": [1025, 2048, 4096, 8191, 8192, 16384, 16385, 32768, 65536]},
        "mask": {"values": ["none", "none", "causal_bottom_right", "window"]},
        "wl": {"values": [1, 16, 64, 128, 1024, 16384]},
        "wr": {"values": [None, 0, 1, 64]},
        "walign": a["walign"],
        "scale": a["scale"],
        "domain": a["domain"],
        "alignment": a["alignment"],
        "placement": a["placement"],
    }
    r["constraints"] = [
        ROWS_SEE,
        {
            "rule": "b*hq*sq*skv*(dqk+dv) <= 2**27",
            "reason": "reference cost ceiling, a quarter of the other spaces' 2**29: the long cases are the slowest per element",
        },
    ]
    r["description"] = (
        "SDPA backward with a long key sequence (1025 to 65536) and few query rows over hipkernel:flydsl_sdpa_bwd_split on gfx942 and gfx90a: "
        "a softmax row over Skv keys has probabilities near 1/Skv, below 2**-14 (6.1e-5) from Skv = 16384, where the MFMA of gfx90a "
        "flushes fp16 subnormal operands (COVERAGE.md: fp16 divergence). Data half, bfloat16 or float; masks none, bottom-right causal and "
        "windows with bounds far from the tile edges; domains as SdpaBackward/FlydslBwdSplit. The cost ceiling is 2**27."
    )
    return r


def stride():
    axes = {
        "dtype": {"values": ["bfloat16", "half"]},
        "layout": {"values": ["bshd", "bhsd"]},
        "b": {"values": [1, 2]},
        "hkv": {"values": [1, 2]},
        "group": {"values": [1, 4]},
        "sq": {"values": [16, 64, 128]},
        "skv": {"values": [16, 64, 128]},
        "d": {"values": [64, 96, 128, 256]},
        "pad": {"values": [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31]},
        "pad_on": {"values": ["q", "k", "v", "o", "stats", "all"]},
        "alignment": {"values": [1, 2, 4, 8, 16, 32, 64, 128, 256]},
        "placement": {"values": ["default", "exact", "under"]},
    }
    derive = {
        "hvmul": "1",
        "hd": "[d, d]",
        "pad_q": "pad if pad_on in ['q', 'all'] else 0",
        "pad_k": "pad if pad_on in ['k', 'all'] else 0",
        "pad_v": "pad if pad_on in ['v', 'all'] else 0",
        "pad_o": "pad if pad_on in ['o', 'all'] else 0",
        "pad_stats": "pad if pad_on in ['stats', 'all'] else 0",
    }
    derive.update(shape_derive(pad=True))
    derive.update(
        {
            "left": "None",
            "right": "None",
            "align": "'TOP_LEFT'",
            "scale": "0.0883883461356163",
            "eff_placement": DOMAIN["eff_placement"],
        }
    )
    inp = {
        k: {"kind": "free", "lo": -1, "hi": 1, "value": 0, "scale": 1}
        for k in ("2", "0", "6", "7")
    }
    return {
        "version": 1,
        "operation": "SdpaBackward",
        "description": "Padded leading strides on hipkernel:flydsl_sdpa_bwd_split, crossed with the declared alignment: BSHD or BHSD (layout), a small shape set, and a `pad` (0 to 31 elements) added to the row stride (the stride between two consecutive sequence positions) of the tensors `pad_on` names: q (Q and dQ), k (K and dK), v (V and dV), o (O and dO), or only the LSE (stats: padded dense strides are served, SS-1, where AOTriton's are not), or all nine tensors. A row stride that is not a multiple of 8 elements is declined (LY-3), as is a declared alignment below 16 bytes (LY-1, SS-3). The pad elements are gap elements that belong to no tensor element; the harness poisons them where the type allows, so a kernel that reads or writes a gap shows as a mismatch. A probe: expect is declined_or_pass on a multiple of 8 and declined otherwise.",
        "engine": ENGINE,
        "template": TEMPLATE,
        "axes": axes,
        "derive": derive,
        "constraints": [
            {
                "rule": "pad > 0 or pad_on == 'all'",
                "reason": "pad_on names no tensor when pad is 0: every pad_on would render the same packed graph",
            }
        ],
        "expect": "${'declined' if ((pad % 8 != 0 and pad_on != 'stats') or alignment < 16) else 'declined_or_pass'}",
        "case": {
            "values": {
                "io_data_type": "${dtype}",
                "tensors": tensors(),
                "attributes": attrs(),
            },
            "metadata": {
                "purpose": "probe",
                "validator": "default",
                "inputs": inp,
                "placement": "${eff_placement}",
                "placement_uid": 2,
            },
            "tensor_patches": align_patches(),
        },
    }


def probe():
    d = json.loads(
        (SPACES_DIR / "SdpaBackward.AsmProbe.space.json").read_text(encoding="utf8")
    )
    d["engine"] = ENGINE
    fam = d["axes"]["family"]["values"]
    # overflow_kernarg is ASM's kernarg/tile limits (stats strides, tile sizes, workspace): no rule of the FlyDSL
    # matcher behind it, so no expectation can be written from one
    fam[:] = [f for f in fam if f != "overflow_kernarg"]
    new = [
        "hd_over",
        "bias_causal",
        "bias_window",
        "dbias_rowbcast",
        "bias_f32_dbias",
        "bias_unaligned_pitch",
        "dbias_unaligned",
        "window_left0",
        "scale_huge",
        "padding_noseq",
        "v_heads_bias",
        "v_heads_f32",
        "hd_asym_over",
        "bias_pitch_pad",
        "bias_overlap",
        "scale_limit",
    ]
    fam += [n for n in new if n not in fam]
    for f in ("hd64", "window_hd192"):
        pass
    dv = d["derive"]
    # f32 variants share the all-float switch
    for k, v in list(dv.items()):
        if "family == 'dt_all_float'" in v:
            dv[k] = v.replace(
                "family == 'dt_all_float'", "family in ['dt_all_float', 'v_heads_f32']"
            )
    dv["dtype"] = "shape[7]"
    dv["hkv_e"] = dv["hkv_e"].replace(
        "family == 'v_heads'", "family in ['v_heads', 'v_heads_bias', 'v_heads_f32']"
    )
    dv["hv"] = "1 if family in ['v_heads', 'v_heads_bias', 'v_heads_f32'] else hkv_e"
    dv["dqk_e"] = (
        "520 if family in ['hd_over', 'hd_asym_over'] else (128 if family in ['v_heads_bias', 'v_heads_f32', 'bias_f32_dbias'] else ("
        + dv["dqk_e"]
        + "))"
    )
    dv["dv_e"] = (
        "64 if family == 'hd_asym_over' else (128 if family == 'asym_hd' else dqk_e)"
    )
    dv["mk"] = (
        "(band[0] if band[0] != 'window' else 'none') if family == 'none' else ('causal_top_left' if family == 'bias_causal' else 'none')"
    )
    dv["left"] = (
        "wl if family in ['window_hd192', 'bias_window'] else (0 if family == 'window_left0' else None)"
    )
    dv["right"] = (
        "0 if family == 'alibi' or mk in ['causal_top_left', 'causal_bottom_right'] else (wr if family in ['window_hd192', 'bias_window', 'window_left0'] else None)"
    )
    dv["bias_on"] = (
        "family in ['attn_mask', 'attn_mask_dbias', 'bias_causal', 'bias_window', 'dbias_rowbcast', 'bias_f32_dbias', 'bias_unaligned_pitch', 'dbias_unaligned', 'v_heads_bias', 'bias_pitch_pad', 'bias_overlap']"
    )
    dv["dbias_on"] = (
        "family in ['attn_mask_dbias', 'dbias_rowbcast', 'bias_f32_dbias', 'dbias_unaligned']"
    )
    dv["bias_dtype"] = "'float' if family == 'bias_f32_dbias' else dtype"
    dv["bias_q"] = "1 if family == 'dbias_rowbcast' else sq_e"
    dv["skv_e"] = (
        "(skv + 1 if skv % 8 == 0 else skv) if family in ['bias_unaligned_pitch', 'dbias_unaligned'] else ("
        + dv["skv_e"]
        + ")"
    )
    dv["bias_skv"] = "skv_e"
    # a padded row pitch (skv + 1..8 elements, var picks it) and an overlapping one (skv / 2: a read-only view whose rows share
    # elements; the matcher checks overlap for dBias only)
    dv["bias_pitch"] = (
        "bias_skv + [1, 2, 3, 4, 5, 6, 7, 8][var] if family == 'bias_pitch_pad' else (max(1, bias_skv // 2) if family == 'bias_overlap' else bias_skv)"
    )
    dv["scale_on"] = "family == 'scale_const'"
    # overflow_stride: the matcher declines a per-head slab beyond 32-bit byte offsets (DT-9, SS-1) and nothing else
    # about a stride: the batch strides and some head strides of 2**31 elements are served and computed correctly
    # (q_batch, o_batch, dk_batch, dv_head: var 0, 3, 6, 7), the v_row and dq_row strides are declined (var 2, 5).
    # k_head and do_head (var 1, 4) are served too, but the harness's device input fill fails on a 2**31-element
    # stride: they are left out (constraint)
    dv["ovf_served"] = "family == 'overflow_stride' and var in [0, 3, 6, 7]"
    # scale_limit: 2^20, the largest scale the matcher accepts (SC-4/SC-5). Q and K have magnitudes in [5e-4, 1e-3] with a
    # random sign (fill kind `large`, scale 1e-3), so the scores stay at most 2^20 x 192 x 1e-6 = 201 and the case is graded.
    # With the usual +-1 inputs the scores reach 1e8 and no backward that consumes an fp32 LSE it did not compute can be judged;
    # a uniform draw near 0 would put fp16 values below 2^-14, which the gfx90a MFMA flushes; and at buildable enforcement an
    # accept would end UNEXERCISED, which no expectation allows
    dv["scale_v"] = (
        "2**21 if family == 'scale_huge' else (2**20 if family == 'scale_limit' else 0.08838834764831843)"
    )
    dv["qk_kind"] = "'large' if family == 'scale_limit' else 'free'"
    dv["qk_scale"] = "0.001 if family == 'scale_limit' else 1"
    dv["nseq"] = "family in ['padding', 'seq_len_group']"
    dv["ovf_skv"] = "skv_d"
    d["axes"]["family"]["values"] = fam
    d["constraints"] = [
        {
            "rule": "ov or b_e * hq * sq_e * skv_e * (dqk_e + dv_e) <= 2**29",
            "reason": "GPU reference cost ceiling (as SdpaBackward.space.json); the overflow families that are declined are built, never run",
        },
        {
            "rule": "family != 'overflow_stride' or var not in [1, 4]",
            "reason": "k_head and do_head strides of 2**31 elements are served and correct (checked with HIPDNN_TEST_HOST_INPUT_FILL=1) but the device input fill fails on them (hipModuleLaunchKernel invalid argument), so they would end UNEXERCISED",
        },
    ]
    d["expect"] = (
        "${'declined_or_pass' if family in ['none', 'odd_dims', 'odd_head_size', 'dropout_p0', 'dropout_philox', "
        "'scale_const', 'stats_rank3', 'stats_rank1', 'dt_all_float', 'gqa', 'asym_hd', 'hd64', 'window_hd192', "
        "'padding', 'attn_mask', 'attn_mask_dbias', 'bias_f32_dbias', 'bias_unaligned_pitch', 'v_heads', 'bias_pitch_pad', 'bias_overlap', 'scale_limit'] or ovf_served else 'declined'}"
    )
    cs = d["case"]
    cs["metadata"][
        "enforcement_level"
    ] = "${'full' if ovf_served else ('buildable' if ov else 'full')}"
    for uid in ("2", "0"):
        cs["metadata"]["inputs"][uid] = {
            "kind": "${qk_kind}",
            "lo": -1,
            "hi": 1,
            "value": 0,
            "scale": "${qk_scale}",
        }
    cs["values"]["attributes"]["parameters__attn_scale_value"] = "${scale_v}"
    # bias tensors: replace the asm ones with the shared bias / dbias builders (no family conflict: ids 9, 10)
    add = [t for t in cs["add_tensors"] if t["name"] not in ("attn_mask", "dbias")]
    add += [
        {
            "when": "${bias_on}",
            "uid": 9,
            "name": "attn_mask",
            "data_type": "${bias_dtype}",
            "dims": "${[1, 1, bias_q, bias_skv]}",
            "strides": "${[bias_q*bias_pitch, bias_q*bias_pitch, bias_pitch, 1]}",
            "alignment": 16,
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
        {
            "when": "${dbias_on}",
            "uid": 10,
            "name": "dbias",
            "data_type": "${bias_dtype}",
            "dims": "${[1, 1, bias_q, bias_skv]}",
            "strides": "${[bias_q*bias_skv, bias_q*bias_skv, bias_skv, 1]}",
            "alignment": 16,
            "is_runtime_pass_by_value": False,
            "ragged_offset_multiplier": 1,
            "virtual": False,
        },
    ]
    cs["add_tensors"] = add
    nps = [
        p
        for p in cs["node_patches"]
        if not (
            p["set"].get("attn_mask_tensor_uid") or p["set"].get("dbias_tensor_uid")
        )
    ]
    nps += [
        {
            "when": "${bias_on}",
            "node": "SdpaBwd_0",
            "section": "inputs",
            "set": {"attn_mask_tensor_uid": 9},
        },
        {
            "when": "${dbias_on}",
            "node": "SdpaBwd_0",
            "section": "outputs",
            "set": {"dbias_tensor_uid": 10},
        },
        {
            "when": "${family == 'padding_noseq'}",
            "node": "SdpaBwd_0",
            "section": "parameters",
            "set": {"padding_mask": True},
        },
    ]
    cs["node_patches"] = nps
    # window_left0 needs its left bound in the attributes too (the shared template carries left_bound as a value)
    d["description"] = (
        "Probe-only rejection library for hipkernel:flydsl_sdpa_bwd_split on gfx942 and gfx90a, written from the matcher's decline table (COVERAGE.md 9.1 and 9.2; FlydslSdpaBwdCommon.hpp): a legal SDPA backward (bfloat16 or half, head size 128 or 192, BHSD or BSHD, FP32 stats [B,H,Sq,1] as an INPUT, declared alignment 16) with ONE feature per case, named by the `family` axis; every family is a graph the frontend accepts. Families the engine serves expect declined_or_pass: the plain graph, odd dims and head sizes (declined only at a compact pitch, LY-3), dropout probability 0 and Philox dropout, a baked scale tensor, stats rank 1 and 3, f32 data, GQA, asymmetric head dims, head dim 64, a window, padding with both sequence lengths, a bias with and without dBias, an f32 bias with an f32 dBias at head dim 128 (BI-2), a 16-bit bias with an unaligned pitch, V with a head count of its own at head dim 128 (SH-1; at head dim 192 it declines). Families it declines expect declined: a dropout mask or scale tensor (DR-7), ALiBi (MW-12), sequence lengths without padding_mask (PV-2), mixed dtypes and a non-f32 stats tensor (DT-1, DT-3), head dims above 512 (DT-4), a bias with a causal or window mask (MW-10), dBias with a row-broadcast bias (BI-4), an unaligned dBias stride (BI-9), a window left bound of 0 (MW-9), a scale above 2**20 (SC-5), padding_mask without its tensors (PV-2), Hk != Hv with a bias or f32 data (SH-1), and the overflow families: overflow_dim and the v_row and dq_row strides are declined and built, never run; the q_batch, o_batch, dk_batch and dv_head strides of 2**31 elements are served (DT-9 declines only a per-head slab beyond 32-bit byte offsets), so they run at full enforcement and expect declined_or_pass (allocations of 4 to 8 GiB per tensor); ASM's kernarg families are not drawn (no matcher rule behind them). Dropout is served but its gradient needs the forward's keep mask: the harness cannot supply it for a backward graph, so an accepted dropout case ends UNEXERCISED and shows as a violation: the list of those is the known oracle gap, not an engine defect."
    )
    return d


NAMES = (
    "FlydslBwdSplit",
    "FlydslBwdSplitBias",
    "FlydslBwdSplitPadded",
    "FlydslBwdSplitDropout",
    "FlydslBwdSplitPaddedDropout",
    "FlydslBwdSplitLong",
    "FlydslBwdSplitRagged",
    "FlydslBwdSplitStride",
    "FlydslBwdSplitProbe",
)


def generate():
    """{space name: space dict} for every space this script owns."""
    builders = {
        "FlydslBwdSplit": region,
        "FlydslBwdSplitBias": bias,
        "FlydslBwdSplitPadded": padded,
        "FlydslBwdSplitDropout": dropout,
        "FlydslBwdSplitPaddedDropout": padded_dropout,
        "FlydslBwdSplitLong": long_kv,
        "FlydslBwdSplitRagged": ragged,
        "FlydslBwdSplitStride": stride,
        "FlydslBwdSplitProbe": probe,
    }
    out = {}
    for name in NAMES:
        sp = builders[name]()
        if name in (
            "FlydslBwdSplit",
            "FlydslBwdSplitPadded",
            "FlydslBwdSplitDropout",
            "FlydslBwdSplitPaddedDropout",
            "FlydslBwdSplitLong",
        ):
            add_score_bound(sp)
        elif name == "FlydslBwdSplitRagged":
            add_score_bound(sp, dqk="192")
        elif name == "FlydslBwdSplitBias":
            add_score_bound(sp, "bias_mag")
        out[name] = sp
    return out


def main(argv):
    out_dir = Path(argv[1]) if len(argv) > 1 else SPACES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sp in generate().items():
        with open(
            out_dir / f"SdpaBackward.{name}.space.json",
            "w",
            newline="\n",
            encoding="utf8",
        ) as f:
            json.dump(sp, f, indent=2)
            f.write("\n")
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv)
