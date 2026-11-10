# Generates the CONV_DEPTHWISE_MFMA_FWD / CONV_DEPTHWISE_MFMA_BWD descriptors: unit-stride 2-D depthwise,
# channels-last, fp16 / bf16, on CDNA matrix cores. Each config is admitted on correctness alone; autotune
# (UHD later) ranks them.
#   python3 gen_dwm.py <descriptors/conv/depthwise_mfma>
import json, uuid, pathlib, sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("5d1e9a77-3c2b-4f60-9e1d-8a4b2c6f0d35")
U = lambda name: str(uuid.uuid5(NS, "depthwise_mfma." + name))


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


FIELDS = [
    "dtype",
    "direction",
    "layout",
    "atom",
    "kh",
    "kw",
    "stride",
    "waves",
    "depth",
    "fold",
    "rows",
]
STRINGS = {"dtype", "direction", "layout"}
KNOBS = ["atom", "kh", "kw", "stride", "waves", "depth", "fold", "rows"]

# (waves, depth, fold, rows) per filter class. Waves own 8 channels each (16 B per pixel); fold puts 2 or 4
# narrow images side by side in the 32 MFMA columns; rows is the output rows a block streams (the filter
# height - 1 rows before them are re-read).
SMALL = [
    (8, 3, 1, 32),
    (8, 3, 2, 16),
    (16, 2, 1, 32),
    (4, 3, 2, 16),
    (8, 2, 1, 16),
    (4, 3, 4, 16),
    (2, 2, 1, 32),
    (1, 2, 1, 32),
]
LARGE = [
    (8, 2, 1, 64),
    (16, 2, 1, 64),
    (8, 2, 1, 32),
    (4, 2, 2, 16),
    (4, 2, 2, 32),
    (4, 2, 4, 8),
    (2, 2, 1, 32),
    (1, 2, 1, 32),
]
FILTERS = {3: SMALL, 5: LARGE, 7: LARGE, 9: LARGE}
# Stride 2 (forward only: strided backward data is a gather): 3x3 and 5x5, 64 + halo staged columns.
STRIDED = [
    (8, 3, 2, 32),
    (8, 3, 2, 16),
    (16, 2, 1, 32),
    (16, 2, 1, 16),
    (8, 2, 1, 32),
    (4, 3, 4, 8),
    (2, 2, 1, 16),
    (1, 2, 1, 16),
]
STRIDED_FILTERS = {3: STRIDED, 5: STRIDED}

# Channels-first (a wave per channel pair, no LDS; fold up to 8 images over the 128 columns), every filter.
NCHW = [
    (4, 2, 1, 64),
    (4, 2, 1, 32),
    (1, 3, 2, 16),
    (4, 3, 2, 64),
    (4, 2, 4, 32),
    (1, 2, 4, 16),
    (1, 2, 8, 16),
    (4, 2, 4, 8),
]
NCHW_FILTERS = {3: NCHW, 5: NCHW, 7: NCHW, 9: NCHW}
NCHW_STRIDED = {3: NCHW, 5: NCHW}


def configs(direction):
    for layout, plans in (
        ("NHWC", ((1, FILTERS), (2, STRIDED_FILTERS))),
        ("NCHW", ((1, NCHW_FILTERS), (2, NCHW_STRIDED))),
    ):
        for atom in (32, 16):
            for stride, filters in plans:
                if stride == 2 and direction == "BWD":
                    continue
                for k, blocks in filters.items():
                    for waves, depth, fold, rows in blocks:
                        yield dict(
                            layout=layout,
                            atom=atom,
                            kh=k,
                            kw=k,
                            stride=stride,
                            waves=waves,
                            depth=depth,
                            fold=fold,
                            rows=rows,
                        )


def label(c):
    return (
        f"{c['layout'].lower()}.x{c['atom']}.k{c['kh']}x{c['kw']}s{c['stride']}.w{c['waves']}d{c['depth']}f{c['fold']}"
        f"r{c['rows']}"
    )


kmd, umd, udd = U("kmd"), U("umd"), U("udd")
w(
    "conv_depthwise_mfma.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_depthwise_mfma config fields",
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
        "name": "kernel dtype, direction, layout, MFMA (device), filter, waves, depth, fold, rows and LDS match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_depthwise_mfma.kernel_match",
    },
)
w(
    "conv_depthwise_mfma.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_depthwise_mfma dispatch",
        "dispatch_symbol": "hipkernel.conv_depthwise_mfma.dispatch",
    },
)
total = 0
for direction, tag in (("FWD", "fwd"), ("BWD", "bwd")):
    ued = U("ued." + tag)
    w(
        f"conv_depthwise_mfma_{tag}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_DEPTHWISE_MFMA_{direction}",
            "graph_match": {
                "native": f"hipkernel.conv_depthwise_mfma_{tag}.graph_match"
            },
            "metadata": kmd,
            "knobs": KNOBS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    for dtype, dt in (("HALF", "f16"), ("BFLOAT16", "bf16")):
        for i, c in enumerate(configs(direction)):
            name = f"conv_depthwise_mfma.{tag}.{dt}.{label(c)}"
            kernels.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvDepthwiseMfma.cpp",
                        "entry_point": "ConvDepthwiseMfma",
                    },
                    "metadata": {"dtype": dtype, "direction": direction, **c},
                    "priority": 1000 - i,
                }
            )
    total += len(kernels)
    w(
        f"conv_depthwise_mfma_{tag}.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp." + tag),
            "name": f"hipkernel:conv_depthwise_mfma_{tag}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
print(total, "kernel configs")
