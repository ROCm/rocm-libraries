# Generates the CONV_GROUP_VALU_FWD / CONV_GROUP_VALU_BWD descriptors: grouped forward and backward
# data on vector ALU (descriptors/conv/group_valu/kernels/ConvGroupValu.cpp), the forward
# counterpart of CONV_GROUP_VALU_WRW. A config fixes the direction, the filter class (depth x rows
# x columns, width stride, width dilation, the graph's width pre-padding), the output channels a
# wave (kb), input channels a step (cb), output columns a lane (p) and the waves a workgroup (nw).
# Every extent, channel count, group count, depth / height stride, dilation and padding is a
# runtime argument. Backward data reads dy upsampled by the graph's stride (the width's is the
# config's sw; p a multiple of it).
#
# Every config is admitted on correctness alone; autotune picks. Packs are per direction and
# architecture group (`arch`), each listing the tiles that won somewhere in that group's sweeps.
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("8b4f2c19-6d3e-4a7a-b5c0-91e2f73d6a48")
U = lambda name: str(uuid.uuid5(NS, "gv." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


INTS = ["kb", "cb", "p", "fd", "fr", "fs", "sw", "dw", "pw", "nw"]
STRINGS = ["direction", "dtype", "layout"]
KNOBS = ["kb", "cb", "p", "nw"]

# Filter classes: (depth, rows, columns, width stride, width dilation, width pre-padding).
F3 = (1, 3, 3, 1, 1, 1)
F3S2 = (1, 3, 3, 2, 1, 1)
F333 = (3, 3, 3, 1, 1, 1)
F333S2 = (3, 3, 3, 2, 1, 1)

# Tiles: (kb, cb, p, nw); kb x p sums a lane (at most 64). The valu tiles cover, within 5%, the
# best config of every graph of the gfx906 sweeps of ResNeXt / ResNeXt3D / RegNet / StyleGAN2
# grouped layers, plus one kb = cb = 1 tile a class for channel counts
# no wider tile divides.
F3_TILES = [
    (8, 1, 8, 4),
    (4, 1, 8, 2),
    (4, 1, 2, 2),
    (2, 4, 2, 2),
    (4, 1, 8, 4),
    (16, 1, 4, 2),
    (8, 1, 4, 4),
    (16, 1, 2, 4),
    (4, 4, 4, 2),
    (2, 1, 8, 2),
    (1, 1, 4, 2),
]
F333_TILES = [
    (8, 1, 2, 4),
    (4, 1, 4, 4),
    (4, 1, 2, 2),
    (2, 1, 2, 4),
    (8, 1, 4, 2),
    (4, 2, 2, 2),
    (8, 1, 2, 2),
    (1, 1, 4, 2),
]
TILES_VALU = {
    "FWD": {
        F3: F3_TILES,
        F3S2: [
            (8, 1, 8, 4),
            (16, 1, 2, 4),
            (4, 1, 4, 4),
            (4, 1, 2, 4),
            (2, 1, 2, 4),
            (1, 1, 4, 2),
        ],
        F333: F333_TILES,
        F333S2: [(4, 1, 2, 2), (8, 1, 2, 4), (4, 1, 2, 4), (2, 1, 2, 2), (1, 1, 4, 2)],
    },
    "BWD": {
        F3: F3_TILES,
        F3S2: [(4, 4, 8, 4), (8, 2, 2, 4), (16, 1, 2, 4), (8, 4, 2, 4), (1, 1, 4, 2)],
        F333: F333_TILES,
        F333S2: [(8, 4, 4, 4), (4, 4, 4, 4), (4, 4, 2, 4), (1, 1, 4, 2)],
    },
}
TILES_MMA = {
    "FWD": {
        F3: [(4, 2, 8, 2), (8, 2, 4, 2)],
        F3S2: [(4, 2, 4, 2)],
        F333: [(4, 2, 4, 2)],
        F333S2: [(4, 2, 4, 2)],
    },
    "BWD": {
        F3: [(4, 2, 8, 2), (8, 2, 4, 2)],
        F3S2: [(4, 2, 4, 2)],
        F333: [(4, 2, 4, 2)],
        F333S2: [(4, 2, 4, 2)],
    },
}
RDNA = ["gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152"]
RDNA += ["gfx1200", "gfx1201"]
# group -> (archs, tiles, layouts)
PACKS = {
    "valu": (["gfx906", "gfx908"], TILES_VALU, ("nchw", "nhwc")),
    "mma": (["gfx90a", "gfx942", "gfx950"] + RDNA, TILES_MMA, ("nchw",)),
}
for _, tiles_of, _ in PACKS.values():
    for direction, classes in tiles_of.items():
        for fil, tiles in classes.items():
            for kb, cb, p, nw in tiles:
                assert kb * p <= 64, (fil, kb, p)
                assert direction == "FWD" or p % fil[3] == 0, "whole dy columns a lane"

kmd, umd, udd = U("kmd"), U("umd"), U("udd")
write(
    "conv_group_valu.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_group_valu config fields",
        "fields": [{"name": f, "type": "string"} for f in STRINGS]
        + [{"name": f, "type": "int"} for f in INTS],
    },
)
write(
    "kernel_matches_graph.umd.json",
    {
        "version": "1.0",
        "id": umd,
        "name": "kernel direction, dtype, layout and filter class match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_group_valu.kernel_match",
    },
)
write(
    "conv_group_valu.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_group_valu dispatch",
        "dispatch_symbol": "hipkernel.conv_group_valu.dispatch",
    },
)
for direction, low in (("FWD", "fwd"), ("BWD", "bwd")):
    write(
        f"conv_group_valu_{low}.ued.json",
        {
            "version": "1.0",
            "id": U(f"ued.{low}"),
            "name": f"hipkernel:CONV_GROUP_VALU_{direction}",
            "graph_match": {"native": f"hipkernel.conv_group_valu_{low}.graph_match"},
            "metadata": kmd,
            "knobs": KNOBS,
            "behavior_notes": ["runtime_compilation"],
        },
    )

DTYPES = (("FLOAT", "f32"), ("HALF", "f16"), ("BFLOAT16", "bf16"))
total = 0
for group, (archs, tiles_of, layouts) in PACKS.items():
    for direction, low in (("FWD", "fwd"), ("BWD", "bwd")):
        kernels = []
        for fil, tiles in tiles_of[direction].items():
            fd, fr, fs, sw, dw, pw = fil
            for layout in layouts:
                for dtype, dt in DTYPES:
                    for kb, cb, p, nw in tiles:
                        tag = f"f{fd}x{fr}x{fs}s{sw}d{dw}p{pw}.k{kb}c{cb}p{p}w{nw}"
                        name = f"conv_group_valu.{low}.{dt}.{layout}.{tag}"
                        kernels.append(
                            {
                                "version": "1.0",
                                "id": U(f"{group}.{name}"),
                                "name": name,
                                "kernel_source": {
                                    "kind": "embedded_source",
                                    "source_file": "kernels/ConvGroupValu.cpp",
                                    "entry_point": "ConvGroupValu",
                                },
                                "metadata": {
                                    "direction": direction,
                                    "dtype": dtype,
                                    "layout": layout,
                                    "kb": kb,
                                    "cb": cb,
                                    "p": p,
                                    "fd": fd,
                                    "fr": fr,
                                    "fs": fs,
                                    "sw": sw,
                                    "dw": dw,
                                    "pw": pw,
                                    "nw": nw,
                                },
                                "priority": 0,
                            }
                        )
        write(
            f"conv_group_valu_{low}_{group}.kdp.json",
            {
                "version": "1.0",
                "id": U(f"kdp.{low}.{group}"),
                "name": f"hipkernel:conv_group_valu_{low}_{group}",
                "arch": archs,
                "matchers": [umd],
                "engine": U(f"ued.{low}"),
                "dispatch": udd,
                "kernelDescriptors": kernels,
            },
        )
        total += len(kernels)
        print(group, low, len(kernels), "kernel configs")
print(total, "kernel configs")
