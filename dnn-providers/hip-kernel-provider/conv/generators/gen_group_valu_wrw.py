# Generates the CONV_GROUP_VALU_WRW descriptors: backward weights on vector ALU, the design of
# MIOpen's hand-written ConvAsmBwdWrW3x3 in HIP (descriptors/conv/group_valu_wrw/kernels/
# ConvGroupValuWrw.cpp). A config fixes the filter class (rows x columns, width stride, width
# dilation, width pre-padding), the tile (cb input by kb output channels of one group), the
# output columns a lane (p), the waves a workgroup (nw) and, for fp16 channels-first, whether
# pairs of columns multiply on v_dot2_f32_f16 (dot2). Every extent, channel count, group count,
# height stride, height dilation and height padding is a runtime argument.
#
# Every config is admitted on correctness alone; autotune picks. Packs are per architecture group
# (`arch`), each listing the configs that won somewhere in that group's sweeps (see TILES).
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("5d3a9e27-1c64-4f0b-9a8e-2b7f61c04d95")
U = lambda name: str(uuid.uuid5(NS, "gvw." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


INTS = ["cb", "kb", "p", "fr", "fs", "sw", "dw", "pw", "nw", "dot2", "ring"]
STRINGS = ["dtype", "layout"]
KNOBS = ["cb", "kb", "p", "nw", "dot2", "ring"]

# Filter classes: (rows, columns, width stride, width dilation, width pre-padding).
F3 = (3, 3, 1, 1, 1)
F3S2 = (3, 3, 2, 1, 1)
F5 = (5, 5, 1, 1, 2)

# Tiles per filter class and form: (cb, kb, p, nw). "pass": a wave's lanes take a pass of rows
# side by side, each reading its filter rows' inputs (any height stride and dilation, either
# layout); "ring": a lane group walks consecutive rows of one image with the input rows in a
# register ring (channels-first, unit height stride and dilation). No 1x1 class: a pointwise
# backward weights is a GEMM whose operands every tile re-reads from memory here (3.5 TFLOP/s on
# gfx906 against MIOpen's 6), the BLAS and K3 engines' work. The tiles are the ones that
# won somewhere in the gfx906 sweeps of ResNeXt / RegNet / HRNet / Real-ESRGAN / RetinaNet layers
# (~/old-asic-hipdnn/oldcdna/REPORT.md).
TILES = {
    F3: {
        "pass": [
            (4, 4, 8, 2),
            (2, 8, 8, 2),
            (1, 4, 8, 2),
            (1, 4, 4, 2),
            (1, 4, 2, 2),
            (4, 4, 8, 4),
            (1, 8, 4, 2),
            (2, 8, 4, 2),
        ],
        "ring": [
            (4, 4, 2, 2),
            (4, 4, 4, 2),
            (2, 4, 4, 2),
            (2, 4, 2, 2),
            (2, 4, 8, 2),
            (4, 2, 4, 2),
            (4, 2, 4, 4),
        ],
    },
    F3S2: {"pass": [(4, 4, 4, 2), (2, 4, 4, 2), (1, 4, 4, 2), (1, 4, 2, 2)]},
    F5: {"pass": [(2, 4, 4, 2), (1, 4, 4, 2), (2, 2, 8, 2)], "ring": [(2, 2, 4, 2)]},
}

ARCHS = {
    # Every shipped arch for now; narrowed per group from the sweeps.
    "all": [],
}

kmd, umd, udd, ued = U("kmd"), U("umd"), U("udd"), U("ued")
write(
    "conv_group_valu_wrw.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_group_valu_wrw config fields",
        "fields": [{"name": f, "type": "string"} for f in STRINGS]
        + [{"name": f, "type": "int"} for f in INTS],
    },
)
write(
    "kernel_matches_graph.umd.json",
    {
        "version": "1.0",
        "id": umd,
        "name": "kernel dtype, layout and filter class match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_group_valu_wrw.kernel_match",
    },
)
write(
    "conv_group_valu_wrw.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_group_valu_wrw dispatch",
        "dispatch_symbol": "hipkernel.conv_group_valu_wrw.dispatch",
    },
)
write(
    "conv_group_valu_wrw.ued.json",
    {
        "version": "1.0",
        "id": ued,
        "name": "hipkernel:CONV_GROUP_VALU_WRW",
        "graph_match": {"native": "hipkernel.conv_group_valu_wrw.graph_match"},
        "metadata": kmd,
        "knobs": KNOBS,
        "behavior_notes": ["runtime_compilation"],
    },
)

DTYPES = (("FLOAT", "f32"), ("HALF", "f16"), ("BFLOAT16", "bf16"))
total = 0
for group, archs in ARCHS.items():
    kernels = []
    for fil, forms in TILES.items():
        fr, fs, sw, dw, pw = fil
        for layout in ("nchw", "nhwc"):
            for form, tiles in forms.items():
                ring = 1 if form == "ring" else 0
                if ring and layout != "nchw":
                    continue
                for dtype, dt in DTYPES:
                    for cb, kb, p, nw in tiles:
                        dots = [0]
                        if (
                            dtype == "HALF"
                            and layout == "nchw"
                            and sw == 1
                            and p % 2 == 0
                        ):
                            dots = [1, 0]
                        for dot2 in dots:
                            tag = f"f{fr}x{fs}s{sw}d{dw}p{pw}.c{cb}k{kb}p{p}w{nw}"
                            tag += ".ring" if ring else ""
                            tag += ".dot2" if dot2 else ""
                            name = f"conv_group_valu_wrw.{dt}.{layout}.{tag}"
                            kernels.append(
                                {
                                    "version": "1.0",
                                    "id": U(f"{group}.{name}"),
                                    "name": name,
                                    "kernel_source": {
                                        "kind": "embedded_source",
                                        "source_file": "kernels/ConvGroupValuWrw.cpp",
                                        "entry_point": "ConvGroupValuWrw",
                                    },
                                    "metadata": {
                                        "dtype": dtype,
                                        "layout": layout,
                                        "cb": cb,
                                        "kb": kb,
                                        "p": p,
                                        "fr": fr,
                                        "fs": fs,
                                        "sw": sw,
                                        "dw": dw,
                                        "pw": pw,
                                        "nw": nw,
                                        "dot2": dot2,
                                        "ring": ring,
                                    },
                                    "priority": 0,
                                }
                            )
    write(
        f"conv_group_valu_wrw_{group}.kdp.json",
        {
            "version": "1.0",
            "id": U(f"kdp.{group}"),
            "name": f"hipkernel:conv_group_valu_wrw_{group}",
            "arch": archs,
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
    total += len(kernels)
print(total, "kernel configs")
