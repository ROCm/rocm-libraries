# Generates the CONV_GROUP_FWD / CONV_GROUP_BWD descriptors (small-group convolution, K4).
# Each config is admitted on correctness alone (its atom on the device and the graph's element
# type, its filter and stride the graph's, its channel tile dividing the channels, its halo in
# 64 KB of LDS); autotune ranks them.
import json, uuid, pathlib, sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("7d4b1e2c-3a5f-4c8e-b1d2-9e0f6a7b8c91")
U = lambda name: str(uuid.uuid5(NS, "group." + name))


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


FIELDS = [
    "direction",
    "dtype",
    "atom",
    "t",
    "r",
    "s",
    "stride",
    "th",
    "wp",
    "cb",
    "cf",
    "cg",
    "oc",
    "px",
    "pf",
    "nbi",
    "nbo",
]
KNOBS = [f for f in FIELDS if f not in ("direction", "dtype")]
STRINGS = {"direction", "dtype", "atom"}

# The 16x16 atoms whose k-step divides 16, per element type.
ATOMS = {
    "HALF": ["mfma16_f16", "wmma11_f16", "wmma12_f16"],
    "BFLOAT16": ["mfma16_bf16", "mfma16_bf16_908", "wmma11_bf16", "wmma12_bf16"],
    "FLOAT": ["mfma16_f32"],
}
EB = {"HALF": 2, "BFLOAT16": 2, "FLOAT": 4}
# (th, wp, cb): output rows, 16-column fragments and 16-channel blocks a workgroup.
# One-block tiles (cb 1) are RDNA's best for 16 channels a group (gfx1201 1.2x, stride-2
# forward 1.5x over the cb 2 tiles).
TILES = [
    (4, 2, 2),
    (4, 2, 4),
    (4, 4, 2),
    (8, 2, 2),
    (8, 1, 4),
    (2, 4, 4),
    (4, 2, 1),
    (8, 2, 1),
]
# Tiles that also ship channels-first in place (cf 1: the graph's packed N C D H W activations
# read and written without staging): the bench winners of that form on every target.
CF_TILES = [(4, 2, 2), (4, 4, 2), (8, 2, 2)]
# Phase-fused backward data at stride 2 (pf 1: one launch writes all four row and column
# phases through an LDS output tile, whole dx lines at once): (th, wp, cb) per phase, the
# workgroup's dx tile 2 th x 32 wp; cf twins for the first two.
PF_TILES = [(2, 1, 2), (4, 1, 1), (2, 2, 2)]
PF_CF_TILES = [(2, 1, 2), (4, 1, 1)]


def lds_pf(eb, r, s, th, tw, cb, cf):
    vec = 16 // eb
    pitch = 16 * cb + vec
    halo = (th + r) * (tw + s) * pitch
    tile = 2 * th * 16 * cb * (2 * tw + 1) if cf else 4 * th * tw * pitch
    return (halo + tile) * eb


# Wide groups (16 < cg <= 64, k4ConvMmaWide): nbi 16-channel blocks a group (ceil(cg / 16)),
# (th, wp, nbo) output rows, 16-column fragments and output blocks a workgroup; every tile also
# ships channels-first in place.
WIDE_TILES = {
    2: [(8, 1, 2), (4, 1, 2), (4, 2, 2)],
    3: [(8, 1, 3), (4, 1, 3), (4, 2, 3)],
    4: [(4, 2, 4), (8, 2, 4), (8, 1, 4), (4, 1, 4)],
}


# (t, r, s): filter depth, rows and columns (depth 1 serves 2-D graphs too).
FILTERS = [(1, 3, 3), (3, 3, 3)]


# FMA kernel (atom "fma", fp32: RDNA has no fp32 matrix core), compiled per channels a group cg:
# (th, px, cb, oc) output rows, columns a thread, 16-channel blocks and output channels a thread.
# 256 threads; a chunk of oc channels takes 256 oc / (16 cb) threads (whole waves), so a
# workgroup is th x (256 oc / (16 cb) / th) px columns.
FMA_CGS = [2, 4, 8, 16]
FMA_TILES = {
    4: [(4, 2, 1), (8, 2, 1), (4, 4, 1), (4, 1, 1)],
    8: [(4, 2, 1), (8, 2, 1), (4, 1, 1), (4, 2, 2)],
    16: [(4, 1, 2), (8, 1, 1)],
}
FMA_CF_TILES = [(4, 2, 1)]


def fma_tiles(cg):
    for oc, tiles in FMA_TILES.items():
        if oc != 4 and oc > cg:
            continue
        for th, px, cb in tiles:
            yield th, px, cb, oc


def fma_tw(th, px, cb, oc):
    return 256 * oc // (16 * cb) // th * px


def lds(eb, r, s, st, th, tw, cb):
    vec = 16 // eb
    return ((th - 1) * st + r) * ((tw - 1) * st + s) * (16 * cb + vec) * eb


kmd, umd, udd = U("kmd"), U("umd"), U("udd")
w(
    "conv_group.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_group config fields",
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
        "name": "kernel direction, dtype, atom (device), filter (depth, rows, columns), stride, channel tile and LDS match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_group.kernel_match",
    },
)
w(
    "conv_group.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_group dispatch",
        "dispatch_symbol": "hipkernel.conv_group.dispatch",
    },
)
total = 0
for direction, tag in (("FWD", "fwd"), ("BWD", "bwd")):
    ued = U("ued." + tag)
    w(
        f"conv_group_{tag}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_GROUP_{direction}",
            "graph_match": {"native": f"hipkernel.conv_group_{tag}.graph_match"},
            "metadata": kmd,
            "knobs": KNOBS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    strides = (1, 2)
    for dtype, atoms in ATOMS.items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        i = 0
        for atom in atoms:
            for t, r, s in FILTERS:
                for st in strides:
                    pfs = [
                        tile + (cf, 1)
                        for cf in (0, 1)
                        for tile in PF_TILES
                        if cf == 0 or tile in PF_CF_TILES
                    ]
                    for th, wp, cb, cf, pf in [
                        tile + (cf, 0)
                        for cf in (0, 1)
                        for tile in TILES
                        if cf == 0 or tile in CF_TILES
                    ] + (pfs if tag == "bwd" and st == 2 else []):
                        # Backward data runs stride phases: ceil(R / st) taps at unit stride.
                        kr, ks, kst = (
                            (r, s, st)
                            if tag == "fwd"
                            else (-(-r // st), -(-s // st), 1)
                        )
                        if pf:
                            if (
                                lds_pf(EB[dtype], kr, ks, th, 16 * wp, cb, cf)
                                > 64 * 1024
                            ):
                                continue
                        elif lds(EB[dtype], kr, ks, kst, th, 16 * wp, cb) > 64 * 1024:
                            continue
                        c = dict(
                            atom=atom,
                            t=t,
                            r=r,
                            s=s,
                            stride=st,
                            th=th,
                            wp=wp,
                            cb=cb,
                            cf=cf,
                            cg=0,
                            oc=0,
                            px=0,
                            pf=pf,
                            nbi=0,
                            nbo=0,
                        )
                        f = f"{r}x{s}" if t == 1 else f"{t}x{r}x{s}"
                        name = f"conv_group.{tag}.{dt}.{atom}.{f}.s{st}.t{th}x{wp}x{cb}"
                        name += (".cf" if cf else "") + (".pf" if pf else "")
                        kernels.append(
                            {
                                "version": "1.0",
                                "id": U(name),
                                "name": name,
                                "kernel_source": {
                                    "kind": "embedded_source",
                                    "source_file": "kernels/ConvGroup.cpp",
                                    "entry_point": "ConvGroup",
                                },
                                "metadata": {
                                    "direction": direction,
                                    "dtype": dtype,
                                    **c,
                                },
                                "priority": 1000 - i,
                            }
                        )
                        i += 1
    # Wide-group configs: every matrix-core atom.
    for dtype, atoms in ATOMS.items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        i = 0
        for atom in atoms:
            for t, r, s in FILTERS:
                for st in strides:
                    for nbi, tiles in WIDE_TILES.items():
                        for th, wp, nbo in tiles:
                            for cf in (0, 1):
                                kr, ks, kst = (
                                    (r, s, st)
                                    if tag == "fwd"
                                    else (-(-r // st), -(-s // st), 1)
                                )
                                halo = lds(EB[dtype], kr, ks, kst, th, 16 * wp, nbi)
                                if halo > 64 * 1024:
                                    continue
                                if (
                                    cf
                                    and th * 16 * nbo * (16 * wp + 1) * EB[dtype] > halo
                                ):
                                    continue
                                c = dict(
                                    atom=atom,
                                    t=t,
                                    r=r,
                                    s=s,
                                    stride=st,
                                    th=th,
                                    wp=wp,
                                    cb=0,
                                    cf=cf,
                                    cg=0,
                                    oc=0,
                                    px=0,
                                    pf=0,
                                    nbi=nbi,
                                    nbo=nbo,
                                )
                                f = f"{r}x{s}" if t == 1 else f"{t}x{r}x{s}"
                                name = (
                                    f"conv_group.{tag}.{dt}.{atom}.{f}.s{st}.w{nbi}"
                                    f".t{th}x{wp}x{nbo}" + (".cf" if cf else "")
                                )
                                kernels.append(
                                    {
                                        "version": "1.0",
                                        "id": U(name),
                                        "name": name,
                                        "kernel_source": {
                                            "kind": "embedded_source",
                                            "source_file": "kernels/ConvGroup.cpp",
                                            "entry_point": "ConvGroup",
                                        },
                                        "metadata": {
                                            "direction": direction,
                                            "dtype": dtype,
                                            **c,
                                        },
                                        "priority": 700 - i,
                                    }
                                )
                                i += 1
    # FMA configs, fp32.
    i = 0
    for t, r, s in FILTERS:
        for st in strides:
            for cg in FMA_CGS:
                for th, px, cb, oc in fma_tiles(cg):
                    for cf in (0, 1):
                        if cf and (th, px, cb) not in FMA_CF_TILES:
                            continue
                        kr, ks, kst = (
                            (r, s, st)
                            if tag == "fwd"
                            else (-(-r // st), -(-s // st), 1)
                        )
                        tw = fma_tw(th, px, cb, oc)
                        if lds(4, kr, ks, kst, th, tw, cb) > 64 * 1024:
                            continue
                        c = dict(
                            atom="fma",
                            t=t,
                            r=r,
                            s=s,
                            stride=st,
                            th=th,
                            wp=0,
                            cb=cb,
                            cf=cf,
                            cg=cg,
                            oc=oc,
                            px=px,
                            pf=0,
                            nbi=0,
                            nbo=0,
                        )
                        f = f"{r}x{s}" if t == 1 else f"{t}x{r}x{s}"
                        name = (
                            f"conv_group.{tag}.f32.fma.{f}.s{st}.g{cg}.o{oc}"
                            f".t{th}x{px}x{cb}" + (".cf" if cf else "")
                        )
                        kernels.append(
                            {
                                "version": "1.0",
                                "id": U(name),
                                "name": name,
                                "kernel_source": {
                                    "kind": "embedded_source",
                                    "source_file": "kernels/ConvGroup.cpp",
                                    "entry_point": "ConvGroup",
                                },
                                "metadata": {
                                    "direction": direction,
                                    "dtype": "FLOAT",
                                    **c,
                                },
                                "priority": 500 - i,
                            }
                        )
                        i += 1
    total += len(kernels)
    w(
        f"conv_group_{tag}.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp." + tag),
            "name": f"hipkernel:conv_group_{tag}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
print(total, "kernel configs")
