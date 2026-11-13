# Generates the CONV_DEPTHWISE_ROWS descriptors (FWD, BWD) for the HKP conv group.
#   python3 gen_rows.py <descriptors/conv/depthwise_rows>
# Every config is a catalog entry admitted on correctness alone; autotune (UHD later) picks.
# Configs come from the K6 prototype sweep on gfx1100 / gfx1201 / gfx942
# (~/old-asic-hipdnn/k6/STATUS.md): each one won at least one target shape somewhere.
import json, uuid, pathlib, sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("5b0e9d3a-6c1f-4e27-9a84-3d7c2f1e8b65")
U = lambda name: str(uuid.uuid5(NS, name))


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


STRINGS = ["dtype", "direction", "layout"]
INTS = [
    "variant",
    "kh",
    "kw",
    "tx",
    "ty",
    "rw",
    "rh",
    "bh",
    "threads",
    "stride",
    "q",
    "lds",
    "mmax",
]
TILE, BAND, PHASE, LANES = 0, 1, 2, 3
LDS_1D, LDS_2D = 4096, 8192
MMAX = 32


def cfg(layout, variant, **f):
    c = {"layout": layout, "variant": variant}
    for k in INTS[1:]:
        c[k] = f.get(k, 0)
    return c


def tile(kh, kw, tx, ty, rh, rw, lds):
    return cfg(
        "NCHW", TILE, kh=kh, kw=kw, tx=tx, ty=ty, rh=rh, rw=rw, threads=tx * ty, lds=lds
    )


def band(kh, kw, bh, rh, threads):
    return cfg("NCHW", BAND, kh=kh, kw=kw, bh=bh, rh=rh, threads=threads, lds=LDS_2D)


def phase(layout, k, s, q):
    return cfg(
        layout,
        PHASE,
        kw=k,
        stride=s,
        q=q,
        threads=256,
        mmax=(k + s - 1) // s if k else MMAX,
    )


def lanes(k, s):
    return cfg("NHWC", LANES, kw=k, stride=s, threads=256)


# Forward form: forward, and unit-stride backward data (the forward convolution of dy).
FORWARD = [
    # 1-D rows (height 1): 256 threads x rw outputs.
    tile(1, 3, 256, 1, 1, 2, LDS_1D),
    tile(1, 3, 256, 1, 1, 4, LDS_1D),
    tile(1, 7, 256, 1, 1, 4, LDS_1D),
    tile(1, 9, 256, 1, 1, 4, LDS_1D),
    tile(1, 31, 256, 1, 1, 2, LDS_1D),
    tile(1, 31, 256, 1, 1, 4, LDS_1D),
    tile(0, 0, 256, 1, 1, 2, LDS_1D),
    tile(0, 0, 256, 1, 1, 4, LDS_1D),
    # 2-D plane tiles.
    tile(3, 3, 32, 8, 4, 2, LDS_2D),
    tile(3, 3, 64, 4, 4, 1, LDS_2D),
    tile(3, 3, 16, 16, 2, 5, LDS_2D),
    tile(0, 0, 32, 8, 2, 2, LDS_2D),
    # Bands of whole rows.
    band(3, 3, 72, 8, 1024),
    band(3, 3, 48, 8, 1024),
    band(3, 3, 32, 8, 512),
    band(3, 3, 32, 4, 512),
    band(3, 3, 24, 8, 512),
    band(5, 5, 36, 4, 1024),
    band(5, 5, 32, 8, 512),
    band(5, 5, 16, 2, 512),
    band(9, 9, 32, 8, 1024),
    band(9, 9, 16, 4, 512),
    band(9, 9, 16, 8, 512),
    band(0, 0, 16, 4, 512),
    band(0, 0, 8, 2, 256),
]

# Backward data only: strided 1-D (ConvTranspose1d) by stride phases.
PHASES = [
    phase("NCHW", 12, 2, 1),
    phase("NCHW", 12, 2, 2),
    phase("NCHW", 29, 2, 1),
    phase("NCHW", 29, 2, 2),
    phase("NCHW", 43, 3, 1),
    phase("NCHW", 0, 2, 1),
    phase("NCHW", 0, 2, 2),
    phase("NCHW", 0, 3, 1),
    phase("NCHW", 0, 4, 1),
    phase("NCHW", 0, 5, 1),
    phase("NCHW", 0, 8, 1),
    phase("NHWC", 12, 2, 1),
    phase("NHWC", 29, 2, 1),
    phase("NHWC", 43, 3, 1),
    phase("NHWC", 0, 2, 1),
    phase("NHWC", 0, 3, 1),
    lanes(12, 2),
    lanes(29, 2),
    lanes(43, 3),
    lanes(0, 2),
    lanes(0, 3),
    lanes(0, 4),
    lanes(0, 5),
    lanes(0, 8),
]


def tag(c):
    k = f"k{c['kh']}x{c['kw']}" if c["kh"] or c["kw"] else "kany"
    lay = c["layout"].lower()
    if c["variant"] == TILE:
        return f"{lay}.tile_{k}_t{c['tx']}x{c['ty']}_r{c['rh']}x{c['rw']}_l{c['lds']}"
    if c["variant"] == BAND:
        return f"{lay}.band_{k}_b{c['bh']}_r{c['rh']}_t{c['threads']}"
    kk = f"k{c['kw']}" if c["kw"] else "kany"
    if c["variant"] == PHASE:
        return f"{lay}.phase_{kk}_s{c['stride']}_q{c['q']}_m{c['mmax']}"
    return f"{lay}.lanes_{kk}_s{c['stride']}"


def lds_bytes(c):
    if c["variant"] in (TILE, BAND):
        return 4 * c["lds"]
    if c["variant"] == PHASE:
        return 4 * (c["threads"] * c["q"] + c["mmax"] - 1)
    return 0


kmd, umd, udd = U("kmd"), U("umd"), U("udd")
w(
    "conv_depthwise_rows.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_depthwise_rows config fields",
        "fields": [{"name": f, "type": "string"} for f in STRINGS]
        + [{"name": f, "type": "int"} for f in INTS],
    },
)
w(
    "kernel_matches_graph.umd.json",
    {
        "version": "1.0",
        "id": umd,
        "name": "kernel dtype, direction, layout, variant geometry, filter, stride and LDS halo match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_depthwise_rows.kernel_match",
    },
)
w(
    "conv_depthwise_rows.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_depthwise_rows dispatch",
        "dispatch_symbol": "hipkernel.conv_depthwise_rows.dispatch",
    },
)
total = 0
for d in ("fwd", "bwd"):
    ued = U(f"ued.{d}")
    w(
        f"conv_depthwise_rows_{d}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_DEPTHWISE_ROWS_{d.upper()}",
            "graph_match": {"native": f"hipkernel.conv_depthwise_rows_{d}.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    seen = set()
    for c in FORWARD + (PHASES if d == "bwd" else []):
        assert lds_bytes(c) <= 65536, tag(c)
        key = tuple(c[k] for k in INTS) + (c["layout"],)
        assert key not in seen, tag(c)
        seen.add(key)
        for dtype, dt in (("FLOAT", "f32"), ("HALF", "f16"), ("BFLOAT16", "bf16")):
            name = f"conv_depthwise_rows_{d}.{dt}.{tag(c)}"
            meta = {"dtype": dtype, "direction": d.upper(), **c}
            kernels.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvDepthwiseRows.cpp",
                        "entry_point": "ConvDepthwiseRows",
                    },
                    "metadata": meta,
                    "priority": 0,
                }
            )
    total += len(kernels)
    w(
        f"conv_depthwise_rows_{d}.kdp.json",
        {
            "version": "1.0",
            "id": U(f"kdp.{d}"),
            "name": f"hipkernel:conv_depthwise_rows_{d}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
print(
    len(FORWARD),
    "forward-form configs,",
    len(PHASES),
    "phase configs;",
    total,
    "kernels",
)
