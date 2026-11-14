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
F1 = (1, 1, 1, 1, 0)
F5 = (5, 5, 1, 1, 2)

# Tiles per filter class and form: (cb, kb, p, nw). "pass": a wave's lanes take a pass of rows
# side by side, each reading its filter rows' inputs (any height stride and dilation, either
# layout); "ring": a lane group walks consecutive rows of one image with the input rows in a
# register ring (channels-first, unit height stride and dilation). The 1x1 tiles serve narrow
# pointwise layers (16 to 32 channels, or few rows): a large pointwise backward weights is a GEMM
# whose operands every tile re-reads from memory here, the BLAS and K3 engines' work. The tiles
# are the ones that
# won somewhere in the gfx906 sweeps of ResNeXt / RegNet / HRNet / Real-ESRGAN / RetinaNet layers.
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
            # 3-D grouped (resnext3d, a tile per depth tap: short rows, many of them).
            (1, 4, 4, 4),
            (1, 8, 2, 4),
            (1, 8, 2, 2),
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
    F3S2: {
        "pass": [
            (4, 4, 4, 2),
            (2, 4, 4, 2),
            (1, 4, 4, 2),
            (1, 4, 2, 2),
            (1, 8, 2, 4),
            (1, 8, 2, 2),
        ]
    },
    F1: {"pass": [(4, 4, 8, 2), (8, 8, 8, 2), (8, 8, 4, 2), (8, 16, 4, 2)]},
    F5: {"pass": [(1, 4, 4, 2), (2, 2, 8, 2)], "ring": [(2, 2, 4, 2)]},
}

# The matrix-core architectures take a small pack: there the matrix-core engines serve most
# layers, and this engine wins only narrow ones (groups of 4 to 8 channels, few-channel dense
# and pointwise layers) by 1.1-2.7x, channels-first only (no channels-last config won on any of
# gfx90a, gfx942, gfx950, gfx1201, gfx1101).
TILES_MMA = {
    F3: {
        "pass": [(4, 4, 8, 4), (1, 4, 8, 2), (1, 4, 4, 2), (1, 4, 2, 2), (1, 8, 4, 2)],
        "ring": [
            (4, 4, 2, 2),
            (2, 4, 2, 2),
            (2, 4, 4, 2),
            (2, 4, 8, 2),
            (4, 2, 4, 4),
            (4, 2, 4, 2),
        ],
    },
    F3S2: {"pass": [(2, 4, 4, 2), (1, 4, 4, 2)]},
    F1: {"pass": [(4, 4, 8, 2), (8, 8, 4, 2), (8, 8, 8, 2), (8, 16, 4, 2)]},
    F5: {"pass": [(1, 4, 4, 2)]},
}
# Depthwise (one input and one output channel a group, cb = kb = 1), channels-first, every
# architecture: a wave owns one channel's filter, its lanes along the output row. At least two
# output columns a lane (the kernel's 16-bit windows are whole dwords).
F5S2 = (5, 5, 2, 1, 2)
F7 = (7, 7, 1, 1, 3)
DW_TILES = {
    F3: {
        "pass": [(1, 1, 2, 2), (1, 1, 4, 2), (1, 1, 8, 2), (1, 1, 2, 4)],
        "ring": [(1, 1, 2, 2), (1, 1, 4, 2)],
    },
    F3S2: {"pass": [(1, 1, 2, 2), (1, 1, 4, 2), (1, 1, 2, 4)]},
    F5: {
        "pass": [(1, 1, 2, 2), (1, 1, 4, 2)],
        "ring": [(1, 1, 2, 2), (1, 1, 4, 2)],
    },
    F5S2: {"pass": [(1, 1, 2, 2), (1, 1, 4, 2)]},
    F7: {"pass": [(1, 1, 4, 2), (1, 1, 8, 2)], "ring": [(1, 1, 4, 2)]},
}
RDNA = ["gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152"]
RDNA += ["gfx1200", "gfx1201"]
# group -> (archs, tiles, layouts)
PACKS = {
    # No matrix core for the type (gfx906), or a vector unit fast enough to matter (gfx908).
    "valu": (["gfx906", "gfx908"], TILES, ("nchw", "nhwc")),
    "mma": (["gfx90a", "gfx942", "gfx950"] + RDNA, TILES_MMA, ("nchw",)),
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
# The kernels hold cb kb fr fs fp32 sums a lane (ConvGroupValuWrwProblem.hpp: at most 160).
for tiles_of in (TILES, TILES_MMA, DW_TILES):
    for (fr, fs, _, _, _), forms in tiles_of.items():
        for tiles in forms.values():
            for cb, kb, _, _ in tiles:
                assert cb * kb * fr * fs <= 160, (fr, fs, cb, kb)

for group, (archs, tiles_of, layouts) in PACKS.items():
    kernels = []
    sets = [(tiles_of, layouts), (DW_TILES, ("nchw",))]
    for fil, forms, set_layouts in (
        (f, fo, ls) for t, ls in sets for f, fo in t.items()
    ):
        fr, fs, sw, dw, pw = fil
        for layout in set_layouts:
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
    print(group, len(kernels), "kernel configs")
print(total, "kernel configs")
