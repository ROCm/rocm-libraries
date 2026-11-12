# Generates the CONV_GROUP_FWD / CONV_GROUP_BWD / CONV_GROUP_WRW descriptors (small-group
# convolution, K4).
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
    "tw",
    "bpu",
    "math",
]
KNOBS = [f for f in FIELDS if f not in ("direction", "dtype")]
STRINGS = {"direction", "dtype", "atom"}

# The 16x16 atoms whose k-step divides 16, per element type.
ATOMS = {
    "HALF": ["mfma16_f16", "wmma11_f16", "wmma12_f16"],
    "BFLOAT16": ["mfma16_bf16", "mfma16_bf16_908", "wmma11_bf16", "wmma12_bf16"],
    "FLOAT": ["mfma16_f32"],
    # Three-bf16 (math 1, TF32 permission only; wide groups, forward and backward data): fp32
    # graphs on the bf16 atoms, the kernel multiplying three bf16 parts of each operand into fp32
    # accumulators and writing fp32. gfx908 (mfma16_bf16_908) gets none. Small groups and
    # backward weights measured slower than exact (gfx950, gfx1201) and get none.
    "F32X3": ["mfma16_bf16", "wmma11_bf16", "wmma12_bf16"],
}
EB = {"HALF": 2, "BFLOAT16": 2, "FLOAT": 4, "F32X3": 2}
# The element type in the metadata, the name tag, and the output element bytes.
DTYPE = {"HALF": "HALF", "BFLOAT16": "BFLOAT16", "FLOAT": "FLOAT", "F32X3": "FLOAT"}
DT = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16", "F32X3": "f32x3"}
OUT_EB = {"HALF": 2, "BFLOAT16": 2, "FLOAT": 4, "F32X3": 4}
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


def lds_pf(eb, r, s, th, tw, cb, cf, ob=0):
    vec = 16 // eb
    pitch = 16 * cb + vec
    out = ob or cb
    halo = (th + r) * (tw + s) * pitch
    tile = 2 * th * 16 * out * (2 * tw + 1) if cf else 4 * th * tw * (16 * out + vec)
    return (halo + tile) * eb


# Wide groups (16 < cg <= 64, k4ConvMmaWide): nbi 16-channel blocks a group (ceil(cg / 16)),
# (th, wp, nbo) output rows, 16-column fragments and output blocks a workgroup; every tile also
# ships channels-first in place.
# Phase-fused stride-2 backward data for wide groups: (th, wp, nbo) per nbi (both forms).
WIDE_PF_TILES = {
    2: [(4, 1, 2), (4, 1, 1)],
    3: [(4, 1, 3), (4, 1, 1)],
    4: [(4, 1, 4), (4, 1, 2)],
}
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
        dt = DT[dtype]
        if dtype == "F32X3":
            continue
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
                        # A channels-first tile parks its output in the halo.
                        if (
                            cf
                            and not pf
                            and th * 16 * cb * (16 * wp + 1) * OUT_EB[dtype]
                            > lds(EB[dtype], kr, ks, kst, th, 16 * wp, cb)
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
                            cb=cb,
                            cf=cf,
                            cg=0,
                            oc=0,
                            px=0,
                            pf=pf,
                            nbi=0,
                            nbo=0,
                            tw=0,
                            bpu=0,
                            math=0,
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
                                    "dtype": DTYPE[dtype],
                                    **c,
                                },
                                "priority": 1000 - i,
                            }
                        )
                        i += 1
    # Wide-group configs: every matrix-core atom.
    for dtype, atoms in ATOMS.items():
        dt = DT[dtype]
        x3 = dtype == "F32X3"
        i = 0
        for atom in atoms:
            for t, r, s in FILTERS:
                for st in strides:
                    for nbi, tiles in WIDE_TILES.items():
                        pft = (
                            WIDE_PF_TILES[nbi]
                            if tag == "bwd" and st == 2 and not x3
                            else []
                        )
                        for th, wp, nbo, pf in [x + (0,) for x in tiles] + [
                            x + (1,) for x in pft
                        ]:
                            for cf in (0, 1):
                                kr, ks, kst = (
                                    (r, s, st)
                                    if tag == "fwd"
                                    else (-(-r // st), -(-s // st), 1)
                                )
                                if pf:
                                    if (
                                        lds_pf(
                                            EB[dtype], kr, ks, th, 16 * wp, nbi, cf, nbo
                                        )
                                        > 64 * 1024
                                    ):
                                        continue
                                else:
                                    halo = lds(EB[dtype], kr, ks, kst, th, 16 * wp, nbi)
                                    if halo > 64 * 1024:
                                        continue
                                    if (
                                        cf
                                        and th
                                        * 16
                                        * nbo
                                        * (16 * wp + 1)
                                        * OUT_EB[dtype]
                                        > halo
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
                                    pf=pf,
                                    nbi=nbi,
                                    nbo=nbo,
                                    tw=0,
                                    bpu=0,
                                    math=1 if x3 else 0,
                                )
                                f = f"{r}x{s}" if t == 1 else f"{t}x{r}x{s}"
                                name = (
                                    f"conv_group.{tag}.{dt}.{atom}.{f}.s{st}.w{nbi}"
                                    f".t{th}x{wp}x{nbo}"
                                    + (".cf" if cf else "")
                                    + (".pf" if pf else "")
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
                                            "dtype": DTYPE[dtype],
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
                            tw=0,
                            bpu=0,
                            math=0,
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
    # The three-bf16 configs in a pack of their own.
    for suffix, members in (
        ("", [k for k in kernels if k["metadata"]["math"] == 0]),
        ("_f32x3", [k for k in kernels if k["metadata"]["math"] == 1]),
    ):
        w(
            f"conv_group_{tag}{suffix}.kdp.json",
            {
                "version": "1.0",
                "id": U("kdp." + tag + suffix),
                "name": f"hipkernel:conv_group_{tag}{suffix}",
                "arch": [],
                "matchers": [umd],
                "engine": ued,
                "dispatch": udd,
                "kernelDescriptors": members,
            },
        )

# Backward weights (CONV_GROUP_WRW, k4WrwMma): (th, tw, cb) dy tile rows, columns and 16-channel
# blocks, each at bpu workgroups a dispatch unit (the split planner's target; host-side only, so
# the bpu twins share one compiled kernel); both forms (cf 1:
# packed N C D H W, cf 0: packed N D H W C). The tile's positions are flattened into the atom's
# k-steps, so tw need not be a multiple of anything: 14, 28 and 7 serve ResNeXt's 56/28/14/7
# planes exactly.
WRW_ATOMS = {
    "HALF": ["mfma16_f16", "wmma11_f16", "wmma12_f16"],
    "BFLOAT16": ["mfma16_bf16", "mfma16_bf16_908", "wmma11_bf16", "wmma12_bf16"],
    "FLOAT": ["mfma16_f32"],
}
WRW_TILES = [
    (8, 16, 1),
    (4, 16, 1),
    (8, 14, 1),
    (4, 28, 1),
    (2, 32, 1),
    (4, 32, 1),
    (2, 56, 1),
    (4, 16, 2),
    (4, 28, 2),
    (4, 14, 4),
    (4, 7, 4),
]
WRW_BPU = [4, 8, 16]
# Wide groups (17 to 32 channels, k4WrwMmaWide): nbi 16-channel blocks a group (2), (th, tw, nbo)
# dy tile rows, columns and dy blocks a workgroup. MFMA only: on WMMA (gfx11, gfx12) the wide
# kernel measured behind K3 on every graph. Groups past 32 channels are K3's (its per-group GEMM
# is dense there; nbi 3 and 4 measured behind it on every arch).
WRW_WIDE_NBI = [2]
WRW_WIDE_ATOMS = {
    "HALF": ["mfma16_f16"],
    "BFLOAT16": ["mfma16_bf16", "mfma16_bf16_908"],
    "FLOAT": ["mfma16_f32"],
}
WRW_WIDE_TILES = [
    (8, 16, 1),
    (4, 16, 1),
    (4, 14, 1),
    (4, 16, 2),
    (8, 8, 2),
    (4, 7, 2),
]


def lds_wrw_wide(eb, r, s, st, th, tw, nbi, nbo):
    vec = 16 // eb
    halo = ((th - 1) * st + r) * ((tw - 1) * st + s) + 1
    stage = (halo * (16 * nbi + vec) + th * tw * (16 * nbo + vec)) * eb
    red = nbi * nbo * 256 * 4 if nbi == 2 else 0
    return max(stage, red)


# FMA (atom "fma", fp32: RDNA has no fp32 matrix core), compiled per channels a group cg: (th, tw,
# cb, px) dy tile rows, columns, 16-channel blocks and positions a thread's row segment.
WRW_FMA_CGS = [2, 4, 8, 16]
WRW_FMA_TILES = [
    (8, 16, 1, 4),
    (4, 16, 1, 4),
    (4, 28, 1, 4),
    (8, 14, 1, 7),
    (4, 16, 1, 8),
]


def lds_wrw(eb, r, s, st, th, tw, cb):
    vec = 16 // eb
    pixels = ((th - 1) * st + r) * ((tw - 1) * st + s) + th * tw + 1
    return pixels * (16 * cb + vec) * eb


ued = U("ued.wrw")
w(
    "conv_group_wrw.ued.json",
    {
        "version": "1.0",
        "id": ued,
        "name": "hipkernel:CONV_GROUP_WRW",
        "graph_match": {"native": "hipkernel.conv_group_wrw.graph_match"},
        "metadata": kmd,
        "knobs": KNOBS,
        "behavior_notes": ["runtime_compilation"],
    },
)
kernels = []
for dtype, atoms in WRW_ATOMS.items():
    dt = DT[dtype]
    i = 0
    for atom in atoms:
        for t, r, s in FILTERS:
            for st in (1, 2):
                for th, tw, cb in WRW_TILES:
                    if lds_wrw(EB[dtype], r, s, st, th, tw, cb) > 64 * 1024:
                        continue
                    for bpu in WRW_BPU:
                        for cf in (1, 0):
                            c = dict(
                                atom=atom,
                                t=t,
                                r=r,
                                s=s,
                                stride=st,
                                th=th,
                                wp=0,
                                cb=cb,
                                cf=cf,
                                cg=0,
                                oc=0,
                                px=0,
                                pf=0,
                                nbi=0,
                                nbo=0,
                                tw=tw,
                                bpu=bpu,
                                math=0,
                            )
                            f = f"{r}x{s}" if t == 1 else f"{t}x{r}x{s}"
                            name = (
                                f"conv_group.wrw.{dt}.{atom}.{f}.s{st}.t{th}x{tw}x{cb}.u{bpu}"
                                + (".cf" if cf else "")
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
                                        "direction": "WRW",
                                        "dtype": DTYPE[dtype],
                                        **c,
                                    },
                                    "priority": 1000 - i,
                                }
                            )
                            i += 1
for dtype, atoms in WRW_WIDE_ATOMS.items():
    dt = DT[dtype]
    i = 0
    for atom in atoms:
        for t, r, s in FILTERS:
            for st in (1, 2):
                for nbi in WRW_WIDE_NBI:
                    for th, tw, nbo in WRW_WIDE_TILES:
                        if (
                            lds_wrw_wide(EB[dtype], r, s, st, th, tw, nbi, nbo)
                            > 64 * 1024
                        ):
                            continue
                        for bpu in WRW_BPU:
                            for cf in (1, 0):
                                c = dict(
                                    atom=atom,
                                    t=t,
                                    r=r,
                                    s=s,
                                    stride=st,
                                    th=th,
                                    wp=0,
                                    cb=0,
                                    cf=cf,
                                    cg=0,
                                    oc=0,
                                    px=0,
                                    pf=0,
                                    nbi=nbi,
                                    nbo=nbo,
                                    tw=tw,
                                    bpu=bpu,
                                    math=0,
                                )
                                f = f"{r}x{s}" if t == 1 else f"{t}x{r}x{s}"
                                name = (
                                    f"conv_group.wrw.{dt}.{atom}.{f}.s{st}.w{nbi}"
                                    f".t{th}x{tw}x{nbo}.u{bpu}" + (".cf" if cf else "")
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
                                            "direction": "WRW",
                                            "dtype": DTYPE[dtype],
                                            **c,
                                        },
                                        "priority": 700 - i,
                                    }
                                )
                                i += 1
i = 0
for t, r, s in FILTERS:
    for st in (1, 2):
        for cg in WRW_FMA_CGS:
            for th, tw, cb, px in WRW_FMA_TILES:
                if lds_wrw(4, r, s, st, th, tw, cb) > 64 * 1024 or 256 % (16 * cb * cg):
                    continue
                for bpu in WRW_BPU:
                    for cf in (1, 0):
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
                            oc=0,
                            px=px,
                            pf=0,
                            nbi=0,
                            nbo=0,
                            tw=tw,
                            bpu=bpu,
                            math=0,
                        )
                        f = f"{r}x{s}" if t == 1 else f"{t}x{r}x{s}"
                        name = (
                            f"conv_group.wrw.f32.fma.{f}.s{st}.g{cg}.t{th}x{tw}x{cb}.p{px}"
                            f".u{bpu}" + (".cf" if cf else "")
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
                                    "direction": "WRW",
                                    "dtype": "FLOAT",
                                    **c,
                                },
                                "priority": 500 - i,
                            }
                        )
                        i += 1


def wrw_file(kernel):
    # One pack a dtype and atom family (each under the repo's 500 KB file limit).
    m = kernel["metadata"]
    dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[m["dtype"]]
    atom = m["atom"]
    family = (
        "fma"
        if atom == "fma"
        else (
            "rdna"
            if atom.startswith("wmma")
            else "cdna908" if atom.endswith("_908") else "cdna"
        )
    )
    return f"{dt}_{family}"


total += len(kernels)
packs = {}
for kernel in kernels:
    packs.setdefault(wrw_file(kernel), []).append(kernel)
for suffix, members in sorted(packs.items()):
    w(
        f"conv_group_wrw_{suffix}.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp.wrw." + suffix),
            "name": f"hipkernel:conv_group_wrw_{suffix}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": members,
        },
    )
print(total, "kernel configs")
