"""Plan-as-code for the BatchnormInference test bundles (ALMIOPEN-2903).

Pure functions that turn a case spec (dims, dtype, layout, clip, upcast) into the
case ``id`` and ``values`` block used by sweep.json. Tier membership and claim rules
live in generate_batchnorm_inference_bundles.py.
"""

import math
import re

DTYPES = {"fp32": "float", "fp16": "half", "bfp16": "bfloat16"}
FP = {v: k for k, v in DTYPES.items()}

# layout -> (spatial rank, channel-last?)
LAYOUTS = {
    "ncl": (1, False),
    "nlc": (1, True),
    "nchw": (2, False),
    "nhwc": (2, True),
    "ncdhw": (3, False),
    "ndhwc": (3, True),
}

# clip suffix -> (relu_lower_clip, relu_lower_clip_slope, relu_upper_clip)
# 0.1 and 0.01 are stored as float32 in the existing bundles.
CLIPS = {
    "": None,
    "_rlo0p1_norslope_rhi0p5": (0.10000000149011612, None, 0.5),
    "_rlo0_norslope_norhi": (0.0, None, None),
    "_norlo_norslope_rhi6": (None, None, 6.0),
    "_norlo_rslope0p01_norhi": (None, 0.009999999776482582, None),
}

ID_RE = re.compile(r"^(?P<dims>\d+(?:_\d+)*)_(?P<dt>fp32|fp16|bfp16)_(?P<layout>[a-z]+)(?P<clip>_.*)?$")


def parse_id(case_id):
    upcast = case_id.endswith("_upcast")
    m = ID_RE.match(case_id[: -len("_upcast")] if upcast else case_id)
    if not m or m["layout"] not in LAYOUTS:
        raise ValueError(f"unparseable case id: {case_id}")
    clip = m["clip"] or ""
    if clip not in CLIPS:
        raise ValueError(f"unknown clip suffix {clip!r} in {case_id}")
    dims = [int(d) for d in m["dims"].split("_")]
    return {"dims": dims, "dt": m["dt"], "layout": m["layout"], "clip": clip, "upcast": upcast}


def make_id(dims, dt, layout, clip="", upcast=False):
    return "_".join(str(d) for d in dims) + f"_{dt}_{layout}{clip}" + ("_upcast" if upcast else "")


def strides(dims, layout):
    """Dense strides for the layout; dims are always listed N, C, spatial..."""
    if LAYOUTS[layout][1]:
        order = [0] + list(range(2, len(dims))) + [1]  # outermost -> innermost
    else:
        order = list(range(len(dims)))
    out = [0] * len(dims)
    acc = 1
    for ax in reversed(order):
        out[ax] = acc
        acc *= dims[ax]
    return out


def stats_dims(dims):
    return [1, dims[1]] + [1] * (len(dims) - 2)


def stats_strides(dims):
    return [dims[1]] + [1] * (len(dims) - 1)


def tensor(uid, dims, strd, dtype):
    return {"uid": uid, "dims": list(dims), "strides": list(strd), "data_type": dtype}


def values(spec, pointwise):
    """Build the ``values`` block (io_data_type, tensors[, attributes]) for one case."""
    dims, layout, up = spec["dims"], spec["layout"], spec["upcast"]
    io = DTYPES[spec["dt"]]
    full = (dims, strides(dims, layout))
    stats = [tensor(u, stats_dims(dims), stats_strides(dims), "float") for u in (2, 3, 4, 5)]
    if not pointwise:
        # BatchnormInference/Default: uid0 Y, uid1 X
        tens = [tensor(0, *full, "float" if up else io), tensor(1, *full, io)] + stats
        return {"io_data_type": io, "tensors": tens}
    # Pointwise: uid0 virtual BN output, uid1 BN input, uid6 final output
    tens = [tensor(0, *full, "float"), tensor(1, *full, "float" if up else io)] + stats + [tensor(6, *full, io)]
    c = CLIPS[spec["clip"]] or (None, None, None)
    attrs = {"relu_lower_clip": c[0], "relu_lower_clip_slope": c[1], "relu_upper_clip": c[2]}
    return {"io_data_type": io, "tensors": tens, "attributes": attrs}


def elements(dims):
    return math.prod(dims)
