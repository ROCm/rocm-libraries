# Generates the CONV_IGEMM_NCHW_BWD descriptors (channels-first backward data as an implicit GEMM
# with the col2im fused, K7). Each config is admitted on correctness alone (its atom on the device,
# its stride and taps per phase equal to the problem's, its tile within the launch limits);
# autotune (UHD later) ranks them. Priority orders the declared fallback.
#
# Every config-distinguishing field is an integer knob (hipDNN's candidate enumeration tells
# configs apart by their integer knobs only); the strings (direction, dtype, atom) are
# descriptive and checked by the kernel match (atom_id must name the same atom as atom).
#   python3 gen_igemm_nchw.py <descriptors/conv/igemm_nchw>
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("8a3f0c2e-5d71-4b9e-a6c4-2e9d1f7b3c05")
U = lambda name: str(uuid.uuid5(NS, "igemm_nchw." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


# atom_id order: the pack's conv_igemm_nchw::ATOMS.
ATOMS = [
    "fma",
    "mfma16_f16",
    "mfma16_bf16",
    "mfma16_f32",
    "mfma32_f16",
    "mfma32_bf16",
    "mfma32_f32",
    "wmma11_f16",
    "wmma11_bf16",
    "wmma12_f16",
    "wmma12_bf16",
]
INTS = [
    "atom_id",
    "bmq",
    "bn",
    "bk",
    "wm",
    "wn",
    "wp",
    "tq",
    "tc",
    "vec",
    "stride",
    "taps",
    "stages",
    "flat",
    "halo",
    "math",
]
STRINGS = ["direction", "dtype", "atom"]
FIELDS = ["direction", "dtype", "atom"] + INTS
WAVE = {"mfma": 64, "wmma": 32}
LDS_LIMIT = 65536

# (stride, taps): every phase count the kernels unroll, with the taps of S = stride * taps
# (ConvTranspose1d's k = 2 s) and the odd vocoder filters (S 7 at stride 3, S 9 at stride 5).
STRIDES = [(2, 2), (3, 3), (4, 2), (5, 2), (6, 2), (8, 2)]


def eb(dtype):
    return 4 if dtype == "FLOAT" else 2


def lds_mma(c, e):
    pad = 16 // e
    if c["atom"].endswith("f32"):
        # One k per lane: k-major tiles as the FMA kernel's.
        smax = c["stride"] * c["taps"]
        chunks = -(-smax // pad)
        cp = (chunks + 1 if chunks % 2 == 0 else chunks) * pad
        o = c["bn"] * (c["bmq"] * c["stride"] + pad)
        return max(c["bk"] * ((c["bmq"] + c["halo"] + pad) + c["bn"] * cp + pad), o) * e
    wq = c["bmq"] + c["halo"]
    ldk = c["bk"] + pad
    a = wq * ldk + wq // c["vec"] * pad
    b = c["stride"] * c["taps"] * c["bn"] * ldk
    o = c["bn"] * (c["bmq"] * c["stride"] + pad)
    return max(a + b, o) * e


def lds_fma(c, e):
    pad = 16 // e
    smax = c["stride"] * c["taps"]
    chunks = -(-smax // pad)
    cp = (chunks + 1 if chunks % 2 == 0 else chunks) * pad  # k7FmaPitch
    return (
        c["stages"] * c["bk"] * ((c["bmq"] + c["halo"] + pad) + c["bn"] * cp + pad) * e
    )


def M(atom, bmq, bn, bk, wm, wn, wp, vec, stride, taps, flat=0, halo=0):
    return dict(
        atom=atom,
        atom_id=ATOMS.index(atom),
        bmq=bmq,
        bn=bn,
        bk=bk,
        wm=wm,
        wn=wn,
        wp=wp,
        tq=0,
        tc=0,
        vec=vec,
        stride=stride,
        taps=taps,
        stages=1,
        flat=flat,
        halo=halo or vec,
        math=0,
    )


def F(bmq, bn, bk, tq, tc, vec, stride, taps, stages, flat=0, halo=0):
    return dict(
        atom="fma",
        atom_id=0,
        bmq=bmq,
        bn=bn,
        bk=bk,
        wm=0,
        wn=0,
        wp=0,
        tq=tq,
        tc=tc,
        vec=vec,
        stride=stride,
        taps=taps,
        stages=stages,
        flat=flat,
        halo=halo or vec,
        math=0,
    )


def phases(c):
    return c["flat"] if c["flat"] > 0 else c["stride"]


def valid_mma(c, dtype):
    dim = 32 if c["atom"].startswith("mfma32") else 16
    wave = WAVE[c["atom"][:4]]
    pw = phases(c) // c["wp"] if phases(c) % c["wp"] == 0 else 0
    fq = c["bmq"] // (c["wm"] * dim) if c["bmq"] % (c["wm"] * dim) == 0 else 0
    fn = c["bn"] // (c["wn"] * dim) if c["bn"] % (c["wn"] * dim) == 0 else 0
    frags = fq * fn * pw
    maxf = 32 if wave == 64 else 16
    waves = c["wm"] * c["wn"] * c["wp"]
    return (
        pw > 0
        and fq > 0
        and fn > 0
        and 2 <= frags <= maxf
        and lds_mma(c, eb(dtype)) <= LDS_LIMIT
        and waves * wave <= 512
        and (c["taps"] - 1 + (c["stride"] > 1)) <= c["halo"]
        and c["halo"] % c["vec"] == 0
        and (c["bmq"] * c["stride"]) % c["vec"] == 0
    )


def valid_fma(c, dtype):
    threads = (c["bn"] // c["tc"]) * (c["bmq"] // c["tq"])
    return (
        lds_fma(c, eb(dtype)) <= LDS_LIMIT
        and c["tc"] * c["tq"] * phases(c) <= 64
        and threads <= 1024
        and c["halo"] % c["vec"] == 0
        and (c["taps"] <= c["vec"] if c["halo"] == c["vec"] else c["stride"] == 1)
    )


# The catalog, per stride: the tiles that won (or came within 3% of the winner on) a measured
# shape on any GPU of the family (k7_bench over the rdna_av vocoder shapes with MIOpen >= 200 us
# plus small ones; ~/scratch/k7, ~/old-asic-hipdnn/k7). A tile is (bmq, bn, bk, wm, wn, wp) for
# matrix cores and (bmq, bn, bk, tq, tc, stages) for FMA.
TAPS = {2: 2, 3: 3, 4: 2, 5: 2, 6: 2, 8: 2}
WMMA = {  # fp16 and bf16, gfx11 and gfx12
    2: [(64, 32, 32, 2, 2, 2), (128, 32, 32, 4, 1, 2)],
    3: [(64, 16, 32, 4, 1, 1), (128, 16, 32, 4, 1, 1)],
    4: [
        (64, 16, 32, 2, 1, 4),
        (64, 32, 32, 2, 1, 4),
        (256, 16, 32, 4, 1, 2),
        (128, 32, 32, 2, 1, 4),
    ],
    5: [(128, 32, 32, 2, 1, 5), (64, 16, 32, 1, 1, 5)],
    6: [(128, 16, 32, 4, 1, 2), (64, 16, 32, 4, 1, 1)],
    8: [
        (128, 16, 32, 2, 1, 4),
        (64, 16, 32, 2, 1, 4),
        (128, 16, 64, 2, 1, 4),
        (128, 16, 32, 4, 1, 2),
        (64, 32, 32, 1, 1, 8),
    ],
}
MFMA16 = {  # fp16 and bf16 (16x16 atoms; bf16 from gfx90a)
    2: [(32, 32, 32, 2, 1, 2), (128, 32, 32, 4, 1, 2)],
    3: [(64, 16, 32, 4, 1, 1)],
    4: [(64, 16, 32, 2, 1, 4), (128, 16, 32, 4, 1, 2)],
    5: [(64, 16, 32, 1, 1, 5), (128, 16, 32, 4, 1, 1), (64, 16, 32, 4, 1, 1)],
    6: [(128, 16, 32, 4, 1, 2)],
    8: [(64, 16, 32, 2, 1, 4), (64, 16, 32, 1, 1, 8), (128, 16, 32, 2, 1, 4)],
}
MFMA32F = {  # fp32 (v_mfma_f32_16x16x4f32)
    2: [(64, 32, 16, 2, 2, 2), (128, 32, 16, 2, 2, 2)],
    3: [(64, 16, 16, 4, 1, 1)],
    4: [(64, 16, 16, 2, 1, 4), (128, 16, 16, 2, 1, 4)],
    5: [(64, 16, 16, 4, 1, 1), (64, 16, 16, 1, 1, 5)],
    6: [(128, 16, 16, 4, 1, 2), (64, 32, 16, 2, 2, 2)],
    8: [(64, 16, 16, 2, 1, 4)],
}
FMA32 = {  # fp32 everywhere (RDNA has no fp32 matrix core)
    2: [
        (64, 16, 8, 4, 1, 1),
        (128, 32, 8, 8, 2, 2),
        (128, 64, 8, 8, 4, 2),
        (128, 32, 8, 8, 2, 1),
        (64, 32, 8, 4, 2, 1),
    ],
    3: [(160, 16, 8, 10, 1, 2), (64, 16, 8, 4, 1, 1), (160, 16, 8, 10, 1, 1)],
    4: [
        (64, 16, 8, 4, 1, 1),
        (128, 16, 8, 8, 1, 2),
        (256, 16, 8, 16, 1, 1),
        (192, 16, 8, 12, 1, 2),
        (160, 16, 8, 10, 1, 1),
    ],
    5: [
        (160, 16, 8, 10, 1, 1),
        (64, 16, 8, 4, 1, 1),
        (128, 16, 8, 8, 1, 2),
        (192, 16, 8, 12, 1, 1),
    ],
    6: [(64, 16, 8, 4, 1, 1), (128, 16, 8, 8, 1, 1), (160, 16, 8, 10, 1, 1)],
    8: [(128, 16, 8, 8, 1, 1), (64, 16, 8, 4, 1, 1), (96, 16, 8, 6, 1, 1)],
}
FMA16 = (
    128,
    16,
    8,
    8,
    1,
    1,
)  # the 16-bit floor on archs without a matrix core for the dtype

CATALOG = {"FLOAT": [], "HALF": [], "BFLOAT16": []}


def build():
    for stride, taps in TAPS.items():
        for dtype in CATALOG:
            cfgs = []
            if dtype == "FLOAT":
                mma = [("mfma16_f32", t) for t in MFMA32F[stride]]
                fma = FMA32[stride]
            else:
                x = "f16" if dtype == "HALF" else "bf16"
                mma = (
                    [("wmma11_" + x, t) for t in WMMA[stride]]
                    + [("wmma12_" + x, t) for t in WMMA[stride]]
                    + [("mfma16_" + x, t) for t in MFMA16[stride]]
                )
                fma = [FMA16]
            for atom, (bmq, bn, bk, wm, wn, wp) in mma:
                c = M(
                    atom,
                    bmq,
                    bn,
                    bk,
                    wm,
                    wn,
                    wp,
                    4 if dtype == "FLOAT" else 8,
                    stride,
                    taps,
                )
                assert valid_mma(c, dtype), c
                cfgs.append(c)
            for bmq, bn, bk, tq, tc, stages in fma:
                c = F(
                    bmq,
                    bn,
                    bk,
                    tq,
                    tc,
                    4 if dtype == "FLOAT" else 8,
                    stride,
                    taps,
                    stages,
                )
                assert valid_fma(c, dtype), c
                cfgs.append(c)
            CATALOG[dtype] += cfgs


# One tap per phase (S <= stride): strided 1x1, patchify (S = stride) and 2x2x2 stride-2 backward
# data, per row (flat 0) or over the flattened dy plane (flat 1: one computed phase, the 1x1 case;
# flat = stride: every phase). A systematic grid of tiles, wave layouts and vector widths (the
# narrow vectors serve dy planes the wide ones do not divide); validity filters it.
T1_STRIDES = [2, 4]
# Per-row tiles span a row's phase indices (rows are short), flat tiles a whole dy plane.
T1_ROW_TILES = [(64, 16), (128, 16), (32, 32), (64, 32), (64, 64), (128, 32), (128, 64)]
T1_FLAT_TILES = [(64, 64), (128, 32), (128, 64), (128, 128)]
T1_WAVES = [
    (2, 1),
    (2, 2),
    (1, 2),
    (4, 1),
    (1, 4),
    (4, 2),
    (2, 4),
]  # (wm, wn) before the phase groups
T1_FMA_TILES = [  # (bmq, bn, bk, tq, tc, stages)
    (64, 32, 8, 4, 2, 1),
    (64, 64, 8, 4, 4, 1),
    (128, 32, 8, 8, 2, 2),
    (128, 64, 8, 8, 4, 1),
    (128, 64, 8, 8, 4, 2),
    (128, 128, 8, 8, 8, 2),
    (256, 64, 8, 16, 4, 2),
    (64, 128, 8, 4, 8, 2),
]
T1_FMA16 = [(128, 16, 8, 8, 1, 1), (128, 64, 8, 8, 4, 1)]


def t1_mma(atom, dtype, stride, flat, taps=1):
    vec = 4 if dtype == "FLOAT" else 8
    bk = 16 if dtype == "FLOAT" else 32
    np_ = flat if flat > 0 else stride
    out = []
    # Wave layouts (wm, wn, wp): the 4-wave splits (phase groups of 1 or 2), and 8 waves from 64x64.
    four = [(4, 1, 1), (2, 2, 1), (2, 1, 2), (1, 2, 2)]
    eight = [(4, 2, 1), (2, 2, 2)]
    for v in ([vec, 1] if flat > 0 else [vec]):
        for bmq, bn in (T1_FLAT_TILES if flat > 0 else T1_ROW_TILES):
            if v == 1 and (bmq, bn) not in [(64, 64), (128, 64)]:
                continue
            layouts = four + (eight if bmq * bn >= 64 * 64 else [])
            for wm, wn, wp in layouts if v > 1 else [(2, 2, 1)]:
                if np_ % wp != 0:
                    continue
                c = M(atom, bmq, bn, bk, wm, wn, wp, v, stride, taps, flat)
                if valid_mma(c, dtype):
                    out.append(c)
    return out


def t1_fma(dtype, stride, flat, taps=1):
    out = []
    if dtype == "FLOAT":
        for v in ([4, 1] if flat > 0 else [4]):
            for bmq, bn, bk, tq, tc, stages in T1_FMA_TILES:
                if v == 1 and (bmq, bn) not in [(128, 64), (128, 128), (64, 64)]:
                    continue
                c = F(bmq, bn, bk, tq, tc, v, stride, taps, stages, flat)
                if valid_fma(c, dtype):
                    out.append(c)
    else:
        for bmq, bn, bk, tq, tc, stages in T1_FMA16:
            c = F(bmq, bn, bk, tq, tc, 8, stride, taps, stages, flat)
            if valid_fma(c, dtype):
                out.append(c)
    return out


# Stride 2 with two taps per phase (S 3 and 4: 3x3 / 3x3x3 stride-2 backward data, ConvTranspose
# k = 4): the same per-row grid, beside the measured vocoder tiles.
T2_FMA_TILES = [
    (64, 16, 8, 8, 1, 1),
    (64, 16, 8, 4, 1, 1),
    (64, 32, 8, 4, 2, 1),
    (64, 64, 8, 4, 4, 1),
    (128, 32, 8, 8, 2, 2),
    (128, 64, 8, 8, 4, 2),
    (128, 16, 8, 8, 1, 2),
]


# Stride 1 along the width (1-D convolutions: vocoder / HiFi-GAN residual blocks, dilated): one
# phase, S taps shifted by the dilation. Two halos per filter: the unit-dilation reach and the reach
# at dilation 5 (HiFi-GAN's dilations are 1, 3 and 5); larger dilations fall to the other engines.
ST1_TAPS = [3, 5, 7, 9, 11]
ST1_TILES = [(64, 32), (128, 32), (64, 64), (128, 64), (128, 128)]
ST1_FMA = [(128, 32, 8, 8, 2, 2), (128, 64, 8, 8, 4, 2), (64, 64, 8, 4, 4, 1)]


def build_st1():
    for dtype in CATALOG:
        vec = 4 if dtype == "FLOAT" else 8
        bk = 16 if dtype == "FLOAT" else 32
        atoms = (
            ["mfma16_f32"]
            if dtype == "FLOAT"
            else [
                p + ("f16" if dtype == "HALF" else "bf16")
                for p in ("wmma11_", "wmma12_", "mfma16_")
            ]
        )
        for taps in ST1_TAPS:
            small = max(vec, -(-(taps - 1) // vec) * vec)
            for halo in sorted({small, -(-(taps - 1) * 5 // vec) * vec}):
                for atom in atoms:
                    for bmq, bn in ST1_TILES:
                        layouts = [(4, 1, 1), (2, 2, 1)] + (
                            [(4, 2, 1), (2, 4, 1)] if bmq * bn >= 128 * 64 else []
                        )
                        for wm, wn, wp in layouts:
                            c = M(atom, bmq, bn, bk, wm, wn, wp, vec, 1, taps, 0, halo)
                            if valid_mma(c, dtype):
                                CATALOG[dtype].append(c)
            # FMA: a one-vector halo is the register window (taps within the vector, unit
            # dilation); a wider one reads each tap's window from LDS (not _Float16: HIPRTC 7.18
            # fails to select its form on gfx942, "Cannot select f32 = Constant<-1>"; reproducer in
            # ~/old-asic-hipdnn/k7/ice. bf16, converted to fp32 explicitly, compiles: it is the
            # only 16-bit path for these shapes where no matrix core takes bf16, gfx908).
            for halo in sorted({small, -(-(taps - 1) * 5 // vec) * vec}):
                if (halo == vec and taps > vec) or (halo > vec and dtype == "HALF"):
                    continue
                # HIPRTC 7.18 also fails the fp16 register window at S 5 and 7 (k7/ice).
                if dtype == "HALF" and taps != 3:
                    continue
                for bmq, bn, bk_, tq, tc, stages in (
                    ST1_FMA if dtype == "FLOAT" else [T1_FMA16[0]]
                ):
                    c = F(bmq, bn, bk_, tq, tc, vec, 1, taps, stages, 0, halo)
                    if valid_fma(c, dtype):
                        CATALOG[dtype].append(c)


def build_t1():
    for dtype in CATALOG:
        atoms = (
            ["mfma16_f32"]
            if dtype == "FLOAT"
            else [
                p + ("f16" if dtype == "HALF" else "bf16")
                for p in ("wmma11_", "wmma12_", "mfma16_")
            ]
        )
        have = {tuple(sorted(c.items())) for c in CATALOG[dtype]}
        extra = [c for atom in atoms for c in t1_mma(atom, dtype, 2, 0, taps=2)]
        if dtype == "FLOAT":
            for bmq, bn, bk, tq, tc, stages in T2_FMA_TILES:
                c = F(bmq, bn, bk, tq, tc, 4, 2, 2, stages)
                if valid_fma(c, dtype):
                    extra.append(c)
        for c in extra:
            if tuple(sorted(c.items())) not in have:
                have.add(tuple(sorted(c.items())))
                CATALOG[dtype].append(c)
    for stride in T1_STRIDES:
        for dtype in CATALOG:
            atoms = (
                ["mfma16_f32"]
                if dtype == "FLOAT"
                else [
                    p + ("f16" if dtype == "HALF" else "bf16")
                    for p in ("wmma11_", "wmma12_", "mfma16_")
                ]
            )
            for flat in (0, 1, stride):
                for atom in atoms:
                    CATALOG[dtype] += t1_mma(atom, dtype, stride, flat)
                CATALOG[dtype] += t1_fma(dtype, stride, flat)


# Three-bf16 (math 1): fp32 graphs under TF32 permission run a bf16 matrix-core config on dy and w
# split along K into 3K channels (the kernel's ConvTf32Split), dx written in fp32. Per (atom,
# stride, taps, flat) the bf16 catalog's first X3_TILES tiles whose LDS (fp32 output tile) fits.
X3_ATOMS = {"mfma16_bf16", "wmma11_bf16", "wmma12_bf16"}
X3_TILES = 2
CATALOGX3 = []


def lds_x3(c):
    wq = c["bmq"] + c["halo"]
    ldk = c["bk"] + 8
    a = wq * ldk + wq // c["vec"] * 8
    b = c["stride"] * c["taps"] * c["bn"] * ldk
    o = c["bn"] * (c["bmq"] * c["stride"] + 4) * 4
    return max((a + b) * 2, o)


def build_x3():
    groups = {}
    for c in CATALOG["BFLOAT16"]:
        if c["atom"] not in X3_ATOMS:
            continue
        g = (c["atom"], c["stride"], c["taps"], c["flat"])
        tiles = groups.setdefault(g, [])
        if len(tiles) >= X3_TILES or lds_x3(c) > LDS_LIMIT:
            continue
        tiles.append(c)
        CATALOGX3.append(dict(c, math=1))


def emit():
    kmd, umd, udd = U("kmd"), U("umd"), U("udd")
    write(
        "conv_igemm_nchw.kmd.json",
        {
            "version": "1.0",
            "id": kmd,
            "name": "conv_igemm_nchw config fields",
            "fields": [
                {"name": f, "type": "string" if f in STRINGS else "int"} for f in FIELDS
            ],
        },
    )
    write(
        "kernel_matches_graph.umd.json",
        {
            "version": "1.0",
            "id": umd,
            "name": "kernel dtype, atom (device), stride, taps and tile match the graph",
            "scope": "kernel",
            "match_symbol": "hipkernel.conv_igemm_nchw.kernel_match",
        },
    )
    write(
        "conv_igemm_nchw.udd.json",
        {
            "version": "1.0",
            "id": udd,
            "name": "conv_igemm_nchw dispatch",
            "dispatch_symbol": "hipkernel.conv_igemm_nchw.dispatch",
        },
    )
    ued = U("ued.bwd")
    write(
        "conv_igemm_nchw_bwd.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": "hipkernel:CONV_IGEMM_NCHW_BWD",
            "graph_match": {"native": "hipkernel.conv_igemm_nchw_bwd.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    x3 = []
    for dtype, cfgs in list(CATALOG.items()) + [("FLOATX3", CATALOGX3)]:
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16", "FLOATX3": "f32x3"}[
            dtype
        ]
        dtype = "FLOAT" if dtype == "FLOATX3" else dtype
        for i, c in enumerate(cfgs):
            kind = (
                f"w{c['wm']}x{c['wn']}x{c['wp']}"
                if c["atom"] != "fma"
                else f"t{c['tq']}x{c['tc']}.s{c['stages']}"
            )
            name = f"conv_igemm_nchw.bwd.{dt}.{c['atom']}.{c['bmq']}x{c['bn']}x{c['bk']}.{kind}" f".v{c['vec']}.s{c['stride']}t{c['taps']}" + (
                f".f{c['flat']}" if c["flat"] else ""
            ) + (
                f".h{c['halo']}" if c["halo"] != c["vec"] else ""
            )
            (x3 if c["math"] else kernels).append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvIgemmNchw.cpp",
                        "entry_point": "ConvIgemmNchw",
                    },
                    "metadata": {"direction": "BWD", "dtype": dtype, **c},
                    "priority": 1000 - i - (len(CATALOG["FLOAT"]) if c["math"] else 0),
                }
            )
    write(
        "conv_igemm_nchw_bwd.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp.bwd"),
            "name": "hipkernel:conv_igemm_nchw_bwd",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
    write(
        "conv_igemm_nchw_bwd_f32x3.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp.bwd.f32x3"),
            "name": "hipkernel:conv_igemm_nchw_bwd_f32x3",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": x3,
        },
    )
    kernels += x3
    names = [k["name"] for k in kernels]
    assert len(names) == len(set(names)), "duplicate config names"
    tuples = [
        (k["metadata"]["dtype"],) + tuple(k["metadata"][f] for f in INTS)
        for k in kernels
    ]
    assert len(tuples) == len(set(tuples)), "duplicate knob tuples"
    print(len(kernels), "kernel configs")


if __name__ == "__main__":
    build()
    build_t1()
    build_st1()
    build_x3()
    emit()
    print(len(CATALOGX3), "three-bf16 configs (TF32 permission)")
