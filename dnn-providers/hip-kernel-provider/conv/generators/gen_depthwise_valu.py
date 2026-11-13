# Generates the CONV_DEPTHWISE_VALU descriptors (FWD, BWD) for the HKP conv group.
# Configs are the PR #10945 candidates minus their gfx1151 routing guards; selection among them
# is autotune's (and later UHD's). Floors first: with no ranker, declared order is the default.
import json, uuid, pathlib, sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("0a6f3c1e-9b52-4d7e-8e0f-5c2d1b7a4e90")
U = lambda name: str(uuid.uuid5(NS, name))


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


FIELDS = [
    "dtype",
    "direction",
    "layout",
    "ndims",
    "variant",
    "kd",
    "kh",
    "kw",
    "wstrip",
    "th",
    "tw",
    "bd",
    "bh",
    "bw",
    "bk",
    "rd",
    "rh",
    "rw",
    "stride",
    "vec",
]
STRINGS = {"dtype", "direction", "layout", "variant"}
BYTES = {"FLOAT": 4, "HALF": 2, "BFLOAT16": 2}


def cfg(layout, ndims, variant, **t):
    c = {"layout": layout, "ndims": ndims, "variant": variant}
    for f in FIELDS[5:]:
        c[f] = t.get(f, 0)
    return c


CONFIGS = [
    # Floors: any filter, stride, dilation, padding.
    cfg("NHWC", 2, "floor", wstrip=4),
    cfg("NCHW", 2, "floor"),
    cfg("NHWC", 3, "floor", wstrip=4),
    cfg("NCHW", 3, "floor"),
    # Halo variants: channels-last, unit stride and dilation, compile-time filter.
    cfg("NHWC", 2, "microtile", kh=3, kw=3, th=2, tw=4),
    cfg("NHWC", 2, "microtile", kh=3, kw=3, th=2, tw=8),
    cfg("NHWC", 2, "microtile", kh=3, kw=3, th=4, tw=4),
    cfg("NHWC", 2, "fused", kh=5, kw=5, bh=16, bw=16, bk=32, rh=4, rw=4),
    cfg("NHWC", 2, "fused", kh=5, kw=5, bh=8, bw=8, bk=32, rh=2, rw=2),
    cfg("NHWC", 2, "fused", kh=7, kw=7, bh=16, bw=16, bk=32, rh=4, rw=4),
    cfg("NHWC", 2, "fused", kh=7, kw=7, bh=8, bw=8, bk=32, rh=2, rw=2),
    cfg("NHWC", 2, "lds", kh=9, kw=9, bh=8, bw=8, bk=16, rh=2, rw=2),
    cfg("NHWC", 3, "lds", kd=3, kh=3, kw=3, bd=4, bh=8, bw=8, bk=8, rd=2, rh=2, rw=2),
    # Channels-first plane tiles (v3b_lds_core_nchw): a wide tile for large planes, a small
    # one holding several channels for the 7x7 and 14x14 late layers.
    cfg("NCHW", 2, "lds", kh=3, kw=3, bh=16, bw=64, bk=1, rh=2, rw=4),
    cfg("NCHW", 2, "lds", kh=3, kw=3, bh=16, bw=32, bk=2, rh=2, rw=4),
    cfg("NCHW", 2, "lds", kh=3, kw=3, bh=8, bw=16, bk=8, rh=2, rw=2),
    # Whole small planes (<= 16 x 16: efficientnet 12-15) per block, 8 channels.
    cfg("NCHW", 2, "lds", kh=3, kw=3, bh=16, bw=16, bk=8, rh=2, rw=4),
    cfg("NCHW", 2, "lds", kh=5, kw=5, bh=16, bw=32, bk=2, rh=2, rw=4),
    cfg("NCHW", 2, "lds", kh=5, kw=5, bh=8, bw=8, bk=8, rh=2, rw=2),
    cfg("NCHW", 2, "lds", kh=7, kw=7, bh=16, bw=32, bk=2, rh=2, rw=4),
    cfg("NCHW", 2, "lds", kh=7, kw=7, bh=8, bw=8, bk=8, rh=2, rw=2),
    # Whole 14 x 14 / 16 x 16 planes (convnext) per block, 4 channels, 4 x 4 outputs per thread.
    cfg("NCHW", 2, "lds", kh=7, kw=7, bh=16, bw=16, bk=4, rh=4, rw=4),
]

# Backward data only: the gather over the taps (v5_gather_bwd), the one form a strided backward
# data runs in.
BWD_ONLY = [
    cfg("NHWC", 2, "gather"),
    cfg("NCHW", 2, "gather"),
    cfg("NHWC", 3, "gather"),
    cfg("NCHW", 3, "gather"),
]
# Backward data by stride phase blocks (v6_phase_block_*): 2-D, compile-time filter and stride,
# unit dilation. Channels-last: vec channels per thread (16 bytes: set per dtype below), bk channel
# vectors per block.
for k in (3, 5, 7):
    for s in (2,):
        for bk in (8, 16, 32):
            BWD_ONLY.append(
                cfg("NHWC", 2, "phase", kh=k, kw=k, stride=s, bk=bk, vec=-1)
            )
        BWD_ONLY.append(cfg("NCHW", 2, "phase", kh=k, kw=k, stride=s))


def lds_bytes(c, dtype):
    if c["variant"] not in ("fused", "lds"):
        return 0
    if c["layout"] == "NCHW":
        # fp32 staging with one column of padding per row.
        return c["bk"] * (c["bh"] + c["kh"] - 1) * (c["bw"] + c["kw"]) * 4
    depth = (c["bd"] + c["kd"] - 1) if c["ndims"] == 3 else 1
    return (
        depth
        * (c["bh"] + c["kh"] - 1)
        * (c["bw"] + c["kw"] - 1)
        * c["bk"]
        * BYTES[dtype]
    )


def tag(c):
    dims = f"{c['layout'].lower()}{c['ndims']}d"
    if c["variant"] == "floor":
        return f"{dims}.floor" + (f"_w{c['wstrip']}" if c["wstrip"] else "")
    if c["variant"] == "gather":
        return f"{dims}.gather"
    if c["variant"] == "phase":
        return f"{dims}.phase_k{c['kh']}x{c['kw']}_s{c['stride']}" + (
            f"_b{c['bk']}" if c["bk"] else ""
        )
    k = "x".join(
        str(c[a]) for a in (("kd", "kh", "kw") if c["ndims"] == 3 else ("kh", "kw"))
    )
    if c["variant"] == "microtile":
        return f"{dims}.microtile_k{k}_t{c['th']}x{c['tw']}"
    blk = "x".join(
        str(c[a])
        for a in (("bd", "bh", "bw", "bk") if c["ndims"] == 3 else ("bh", "bw", "bk"))
    )
    reg = "x".join(
        str(c[a]) for a in (("rd", "rh", "rw") if c["ndims"] == 3 else ("rh", "rw"))
    )
    return f"{dims}.{c['variant']}_k{k}_b{blk}_r{reg}"


kmd, umd, udd = U("kmd"), U("umd"), U("udd")
w(
    "conv_depthwise_valu.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_depthwise_valu variant fields",
        "fields": [
            {"name": f, "type": "string" if f in STRINGS else "int"} for f in FIELDS
        ],
    },
)
w(
    "kernel_matches_graph.umd.json",
    {
        "version": "1.0",
        "id": umd,
        "name": "kernel dtype, direction, layout, rank and variant constraints match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_depthwise_valu.kernel_match",
    },
)
w(
    "conv_depthwise_valu.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_depthwise_valu dispatch",
        "dispatch_symbol": "hipkernel.conv_depthwise_valu.dispatch",
    },
)
for d in ("fwd", "bwd"):
    ued = U(f"ued.{d}")
    w(
        f"conv_depthwise_valu_{d}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_DEPTHWISE_VALU_{d.upper()}",
            "graph_match": {"native": f"hipkernel.conv_depthwise_valu_{d}.graph_match"},
            "metadata": kmd,
            "knobs": ["variant"] + FIELDS[5:],
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    for c in CONFIGS + (BWD_ONLY if d == "bwd" else []):
        for dtype, dt in (("FLOAT", "f32"), ("HALF", "f16"), ("BFLOAT16", "bf16")):
            assert lds_bytes(c, dtype) <= 65536, (tag(c), dtype)
            name = f"conv_depthwise_valu_{d}.{dt}.{tag(c)}"
            meta = {"dtype": dtype, "direction": d.upper(), **c}
            if meta["vec"] == -1:
                meta["vec"] = 16 // BYTES[dtype]
            kernels.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvDepthwiseValu.cpp",
                        "entry_point": "ConvDepthwiseValu",
                    },
                    "metadata": meta,
                    "priority": 0,
                }
            )
    w(
        f"conv_depthwise_valu_{d}.kdp.json",
        {
            "version": "1.0",
            "id": U(f"kdp.{d}"),
            "name": f"hipkernel:conv_depthwise_valu_{d}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
print(len(CONFIGS), "configs x 3 dtypes per direction")
