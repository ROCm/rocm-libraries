# Generates the CONV_DEPTHWISE_VALU_WRW descriptors (backward weights, partial + fold kernels).
# Every config is admitted on correctness alone; autotune (UHD later) ranks them. Floors first:
# with no ranker, declared order is the default.
import json, uuid, pathlib, sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("0a6f3c1e-9b52-4d7e-8e0f-5c2d1b7a4e90")
U = lambda name: str(uuid.uuid5(NS, "wrw." + name))


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


FIELDS = ["dtype", "layout", "variant", "kd", "kh", "kw"]
STRINGS = {"dtype", "layout", "variant"}
CONFIGS = [dict(variant="floor", kd=0, kh=0, kw=0)] + [
    dict(variant="tile", kd=kd, kh=kh, kw=kw)
    for kd, kh, kw in (
        (1, 3, 3),
        (1, 5, 5),
        (1, 7, 7),
        (1, 1, 4),
        (1, 1, 31),
        (3, 3, 3),
    )
]

kmd, umd, udd, ued = U("kmd"), U("umd"), U("udd"), U("ued")
w(
    "conv_depthwise_valu_wrw.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_depthwise_valu_wrw variant fields",
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
        "name": "kernel dtype, layout and variant constraints match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_depthwise_valu_wrw.kernel_match",
    },
)
w(
    "conv_depthwise_valu_wrw.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_depthwise_valu_wrw dispatch",
        "dispatch_symbol": "hipkernel.conv_depthwise_valu_wrw.dispatch",
    },
)
w(
    "conv_depthwise_valu_wrw.ued.json",
    {
        "version": "1.0",
        "id": ued,
        "name": "hipkernel:CONV_DEPTHWISE_VALU_WRW",
        "graph_match": {"native": "hipkernel.conv_depthwise_valu_wrw.graph_match"},
        "metadata": kmd,
        "knobs": ["variant"],
        "behavior_notes": ["runtime_compilation"],
    },
)
kernels = []
for layout in ("NHWC", "NCHW"):
    for c in CONFIGS:
        for dtype, dt in (("FLOAT", "f32"), ("HALF", "f16"), ("BFLOAT16", "bf16")):
            k = (
                f"{c['kh']}x{c['kw']}"
                if c["kd"] == 1
                else f"{c['kd']}x{c['kh']}x{c['kw']}"
            )
            tag = "floor" if c["variant"] == "floor" else f"tile_k{k}"
            name = f"conv_depthwise_valu_wrw.{dt}.{layout.lower()}.{tag}"
            kernels.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvDepthwiseValuWrw.cpp",
                        "entry_point": "ConvDepthwiseValuWrwPartial",
                    },
                    "metadata": {"dtype": dtype, "layout": layout, **c},
                    "priority": 0,
                }
            )
# Floors first across both layouts.
kernels.sort(key=lambda k: k["metadata"]["variant"] != "floor")
w(
    "conv_depthwise_valu_wrw.kdp.json",
    {
        "version": "1.0",
        "id": U("kdp"),
        "name": "hipkernel:conv_depthwise_valu_wrw",
        "arch": [],
        "matchers": [umd],
        "engine": ued,
        "dispatch": udd,
        "kernelDescriptors": kernels,
    },
)
print(len(kernels), "kernel configs")
