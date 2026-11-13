# Generates the CONV_IGEMM_FWD / CONV_IGEMM_BWD descriptors (implicit-GEMM convolution, K1).
# Each config is admitted on correctness alone (its atom on the device, its channel vector
# dividing the reduction channels); autotune (UHD later) ranks them. Priority orders the declared
# fallback: larger matrix-core tiles first, FMA last.
import json, uuid, pathlib, sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("5c1e2b7a-8d34-4f6a-9e21-0b7c3d9a1f58")
U = lambda name: str(uuid.uuid5(NS, "igemm." + name))


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


FIELDS = [
    "direction",
    "dtype",
    "atom",
    "math",
    "bm",
    "bn",
    "bk",
    "wm",
    "wn",
    "threads",
    "vec",
    "stages",
    "xcd",
    "halo",
    "wdirect",
    "pack",
    "cf",
    "swz",
    "dma",
    "fuse",
    "large",
]
# Every field that tells two configs of one graph apart is a knob (the candidate enumeration
# identifies a config by its knob tuple).
KNOBS = [f for f in FIELDS if f not in ("direction", "dtype")]
STRINGS = {"direction", "dtype", "atom", "math"}


# xcd 1: XCD-contiguous block order (served on gfx942/gfx950 only). halo 1: the halo A loop
# (stride-1 "same" convolutions; served where the problem fits, see the pack's kernel match).
# wdirect 1: backward data reads the graph's K T R S C/g filter directly, no transposed copy
# (backward-data catalog only; served for packed K T R S C/g filters). pack 1: k-steps of whole
# taps when the reduction channels per tap divide BK (small-channel grouped convolutions).
# cf 1: the output is written packed channels-first from the epilogue (no staging transpose;
# served for packed channels-first outputs). Every matrix-core config but tap packing has a cf twin.
def C(
    atom,
    bm,
    bn,
    bk,
    wm=2,
    wn=2,
    vec=None,
    stages=1,
    math="native",
    threads=0,
    xcd=0,
    halo=0,
    wdirect=0,
    pack=0,
    cf=0,
    swz=0,
    dma=0,
    fuse=0,
    large=0,
):
    return dict(
        atom=atom,
        math=math,
        bm=bm,
        bn=bn,
        bk=bk,
        wm=wm,
        wn=wn,
        threads=threads,
        vec=vec,
        stages=stages,
        xcd=xcd,
        halo=halo,
        wdirect=wdirect,
        pack=pack,
        cf=cf,
        swz=swz,
        dma=dma,
        fuse=fuse,
        large=large,
    )


def withCf(cfgs):
    return cfgs + [dict(c, cf=1) for c in cfgs if c["atom"] != "fma" and not c["pack"]]


# swz 1: XOR-swizzled LDS tiles read in 16-byte chunks (no bank conflicts on 8- and 32-byte
# fragments: the 32x32x8 / 16x16x16 MFMA atoms, WMMA). Twins for the 16-bit native configs without
# halo, direct filter or tap packing; gfx950's 16-byte-fragment atoms are conflict-free already.
# The k-step a chunked read spans (8-byte fragments: two atom k-steps); BK must be a multiple.
SWZ_ATOMS = {
    "mfma16_f16": 32,
    "mfma32_f16": 16,
    "mfma16_bf16": 32,
    "mfma32_bf16": 16,
    "wmma11_f16": 16,
    "wmma11_bf16": 16,
    "wmma12_f16": 16,
    "wmma12_bf16": 16,
}


# pack 2 (flat reduction): the taps' channels end to end in k-steps of BK, so a tap whose channels
# BK does not divide pads nothing (24, 40, 56, 80 ... channels). Twins of the configs without
# halo, direct filter or tap packing; served only where BK does not divide the channels.
def withFlat(cfgs):
    return cfgs + [
        dict(c, pack=2)
        for c in cfgs
        if not (c["halo"] or c["wdirect"] or c["pack"] or c["dma"] or c["fuse"])
        and c["math"] == "native"
    ]


# fuse F: F consecutive groups run as one block-diagonal group (the filter expanded in the
# workspace), so small groups (ResNeXt's 4 to 16 channels) fill whole tiles and read whole lines.
# Forward reads the expanded filter as given, backward data through the direct-filter configs.
# Served where the fused groups stay within 64 channels each way.
FUSE_TILES = {
    (128, 64, 32, 2, 2),
    (128, 128, 32, 2, 2),
    (128, 128, 64, 2, 2),
    (128, 64, 32, 2, 4),
    (64, 64, 32, 2, 2),
    (128, 128, 32, 2, 4),
    (64, 64, 16, 0, 0),
}


def withFuse(cfgs, tag):
    return cfgs + [
        dict(c, fuse=f)
        for c in cfgs
        for f in (2, 4, 8, 16)
        if (c["bm"], c["bn"], c["bk"], c["wm"], c["wn"]) in FUSE_TILES
        and c["math"] == "native"
        and not (c["halo"] or c["pack"] or c["swz"] or c["dma"] or c["xcd"])
        and bool(c["wdirect"]) == (tag == "bwd")
    ]


# large 1: 64-bit positions within an image (K1_LARGE), the generic tier's variant for images past
# 2^31 - 1 pixels, served only there (and the other configs only below). Twins of the plain
# configs of a few tiles per atom, the 1-wide-vector ones included (one-channel images).
LARGE_TILES = {(128, 128, 32), (128, 64, 32), (64, 64, 16), (128, 128, 16)}


def withLarge(cfgs):
    return cfgs + [
        dict(c, large=1)
        for c in cfgs
        if (c["bm"], c["bn"], c["bk"]) in LARGE_TILES
        and c["math"] == "native"
        and not (
            c["xcd"]
            or c["halo"]
            or c["wdirect"]
            or c["pack"]
            or c["cf"]
            or c["swz"]
            or c["dma"]
            or c["fuse"]
            or c["large"]
        )
    ]


def withSwz(cfgs):
    return cfgs + [
        dict(c, swz=1)
        for c in cfgs
        if c["atom"] in SWZ_ATOMS
        and c["math"] == "native"
        and c["vec"] == 8
        and not c["swz"]
        and not c["fuse"]
        and not (c["halo"] or c["wdirect"] or c["pack"])
        and c["bk"] in (16, 32, 64)
        and c["bk"] % SWZ_ATOMS[c["atom"]] == 0
    ]


def mfma(dt, atoms32, atoms16, wide32=None, wide16=None):
    v = 8
    cfgs = []
    if atoms32:
        cfgs += [
            C(atoms32, 128, 128, 32, vec=v),
            C(atoms32, 256, 128, 32, vec=v),
            C(atoms32, 128, 64, 32, vec=v),
            C(atoms32, 64, 128, 64, vec=v),
            C(atoms32, 128, 128, 64, vec=v),
        ]
    if atoms16:
        cfgs += [
            C(atoms16, 128, 64, 32, vec=v),
            C(atoms16, 64, 64, 32, vec=v),
            C(atoms16, 128, 32, 32, 4, 1, vec=v),
            C(atoms16, 256, 16, 16, 4, 1, vec=v),
            C(atoms16, 64, 64, 16, vec=1),
        ]
    if wide32:
        cfgs += [C(wide32, 128, 128, 64, vec=v), C(wide32, 256, 128, 64, vec=v)]
    if wide16:
        cfgs += [C(wide16, 128, 128, 64, vec=v)]
    # XCD order on the dense tiles, and the halo loop with 4x1 waves (one B buffer).
    if atoms32:
        cfgs += [
            C(atoms32, 128, 128, 32, vec=v, xcd=1),
            C(atoms32, 64, 128, 64, vec=v, xcd=1),
        ]
    # Larger XCD-ordered tiles for compute-bound dense layers (L2 reuse; 256x256 with 2x4 waves),
    # and the 256x256 tile in plain order for the archs without XCDs (gfx908, gfx90a: 12-14% over
    # the 128-wide tiles on deeplab's dilated c2048 / c512 3x3).
    if atoms32:
        cfgs += [
            C(atoms32, 256, 128, 32, vec=v, xcd=1),
            C(atoms32, 256, 256, 32, 2, 4, vec=v, xcd=1),
            C(atoms32, 256, 256, 32, 2, 4, vec=v),
        ]
    if wide32:
        cfgs += [
            C(wide32, 128, 128, 64, vec=v, xcd=1),
            C(wide32, 256, 256, 64, 2, 4, vec=v, xcd=1),
            C(wide32, 256, 256, 64, 4, 2, vec=v, xcd=1),
            C(wide32, 256, 256, 32, 2, 4, vec=v, xcd=1),
        ]
    if wide16:
        cfgs += [C(wide16, 128, 128, 64, vec=v, xcd=1)]
    # gfx950 DMA: buffer loads straight into a ring of swizzled LDS stages (served on gfx950 only).
    if wide32:
        cfgs += [
            C(wide32, 256, 256, 32, 2, 4, vec=v, xcd=1, swz=1, dma=2),
            C(wide32, 256, 256, 32, 2, 4, vec=v, xcd=1, swz=1, dma=3),
            C(wide32, 256, 128, 64, 4, 2, vec=v, xcd=1, swz=1, dma=2),
            C(wide32, 256, 256, 64, 4, 2, vec=v, xcd=1, swz=1, dma=2),
            C(wide32, 128, 128, 64, vec=v, xcd=1, swz=1, dma=2),
            C(wide32, 128, 128, 32, vec=v, xcd=1, swz=1, dma=3),
        ]
    if wide16:
        cfgs += [C(wide16, 128, 128, 64, vec=v, xcd=1, swz=1, dma=2)]
    if atoms32 and wide32:
        cfgs += [C(atoms32, 128, 128, 32, vec=v, xcd=1, swz=1, dma=2)]
    if atoms16:
        cfgs += [
            C(atoms16, 128, 64, 32, vec=v, xcd=1),
            C(atoms16, 256, 64, 32, 4, 1, vec=v, halo=1),
            C(atoms16, 256, 64, 32, 4, 1, vec=v, xcd=1, halo=1),
            C(atoms16, 128, 64, 32, 4, 1, vec=v, halo=1),
        ]
    # Backward data straight from the K T R S C filter (no per-call transposed copy), and tap
    # packing for groups of fewer channels than the k-step.
    if atoms32:
        cfgs += [C(atoms32, 128, 128, 32, vec=v, wdirect=1)]
    if atoms16:
        cfgs += [
            C(atoms16, 128, 64, 32, vec=v, wdirect=1),
            C(atoms16, 256, 64, 32, 4, 1, vec=v, halo=1, wdirect=1),
            C(atoms16, 64, 16, 32, 4, 1, vec=v, pack=1),
            C(atoms16, 128, 32, 32, 4, 1, vec=v, pack=1),
        ]
    return cfgs


# No 2-stage LDS tiles: on 64 KB of LDS the second buffer halves occupancy and lost on every
# measured shape (gfx1201, gfx942).
def wmma(atom):
    return [
        C(atom, 128, 128, 32, 2, 4, vec=8),
        C(atom, 128, 128, 64, 2, 4, vec=8),
        C(atom, 64, 64, 32, 2, 2, vec=8),
        C(atom, 128, 64, 32, 2, 4, vec=8),
        C(atom, 128, 16, 16, 8, 1, vec=8),
        C(atom, 128, 32, 32, 4, 2, vec=8),
        C(atom, 64, 64, 16, 2, 2, vec=1),
        C(atom, 256, 64, 32, 8, 1, vec=8, halo=1),
        C(atom, 128, 64, 32, 2, 4, vec=8, halo=1),
        # 32x64 and 64x64 wave tiles (dense forward on gfx11 / gfx12).
        C(atom, 128, 128, 64, 4, 2, vec=8),
        C(atom, 128, 256, 32, 2, 4, vec=8),
        # 4-wide vectors: channel counts that 8 does not divide.
        C(atom, 128, 32, 32, 4, 2, vec=4),
        C(atom, 64, 64, 32, 2, 2, vec=4),
        # Backward data from the K T R S C filter directly (the per-call transposed copy was a third
        # to a half of the time on RDNA).
        C(atom, 128, 64, 32, 2, 4, vec=8, wdirect=1),
        C(atom, 64, 64, 32, 2, 2, vec=8, wdirect=1),
        C(atom, 128, 32, 32, 4, 2, vec=8, wdirect=1),
        C(atom, 128, 128, 32, 2, 4, vec=8, wdirect=1),
        C(atom, 128, 128, 64, 2, 4, vec=8, wdirect=1),
        C(atom, 256, 128, 32, 4, 2, vec=8, wdirect=1),
        # Tap packing: groups of 8 or 16 channels (k-steps of several taps).
        C(atom, 128, 16, 32, 8, 1, vec=8, pack=1),
        C(atom, 64, 16, 32, 4, 1, vec=8, pack=1),
        C(atom, 128, 32, 32, 4, 2, vec=8, pack=1),
        C(atom, 256, 16, 32, 8, 1, vec=8, pack=1),
        C(atom, 128, 16, 64, 8, 1, vec=8, pack=1),
    ]


def fma(dt):
    if dt == "FLOAT":
        return [
            C("fma", 128, 128, 16, 0, 0, vec=4, threads=256),
            C("fma", 64, 64, 16, 0, 0, vec=4, threads=256),
            C("fma", 64, 64, 16, 0, 0, vec=1, threads=256),
        ]
    return [
        C("fma", 128, 128, 32, 0, 0, vec=8, threads=256),
        C("fma", 64, 64, 32, 0, 0, vec=8, threads=256),
        C("fma", 64, 64, 16, 0, 0, vec=1, threads=256),
    ]


CATALOG = {
    "HALF": mfma("HALF", "mfma32_f16", "mfma16_f16", "mfma32x16_f16", "mfma16x32_f16")
    + wmma("wmma11_f16")
    + wmma("wmma12_f16")
    + fma("HALF"),
    "BFLOAT16": mfma(
        "BFLOAT16", "mfma32_bf16", "mfma16_bf16", "mfma32x16_bf16", "mfma16x32_bf16"
    )
    + [
        C("mfma32_bf16_908", 128, 128, 32, vec=8),
        C("mfma32_bf16_908", 128, 64, 32, vec=8),
        C("mfma16_bf16_908", 128, 64, 32, vec=8),
        C("mfma16_bf16_908", 64, 64, 32, vec=8),
        C("mfma16_bf16_908", 256, 16, 16, 4, 1, vec=8),
        C("mfma16_bf16_908", 64, 64, 16, vec=1),
        C("mfma16_bf16_908", 128, 64, 32, vec=8, wdirect=1),
        C("mfma16_bf16_908", 64, 16, 32, 4, 1, vec=8, pack=1),
    ]
    + wmma("wmma11_bf16")
    + wmma("wmma12_bf16")
    + fma("BFLOAT16"),
    "FLOAT": [
        C("mfma32_f32", 128, 128, 16, vec=4),
        C("mfma32_f32", 128, 64, 16, vec=4),
        C("mfma16_f32", 128, 128, 16, vec=4),
        C("mfma16_f32", 64, 64, 16, vec=4),
        C("mfma16_f32", 256, 16, 16, 4, 1, vec=4),
        C("mfma16_f32", 64, 64, 16, vec=1),
        C("mfma16_f32", 128, 128, 16, vec=4, xcd=1),
        C("mfma16_f32", 256, 64, 16, 4, 1, vec=4, halo=1),
        C("mfma16_f32", 128, 64, 16, vec=4, halo=1),
        C("mfma16_f32", 128, 64, 16, vec=4, wdirect=1),
        C("mfma16_f32", 64, 16, 16, 4, 1, vec=4, pack=1),
    ]
    + fma("FLOAT")
    # TF32 permission only (three-bf16 on every matrix-core arch; xf32 on gfx942).
    + [
        C("mfma32_bf16", 128, 128, 16, vec=4, math="bf16x3"),
        C("mfma32_bf16", 128, 64, 32, vec=4, math="bf16x3"),
        C("mfma32_bf16_908", 128, 128, 16, vec=4, math="bf16x3"),
        C("mfma32x16_bf16", 128, 128, 32, vec=4, math="bf16x3"),
        C("wmma11_bf16", 128, 128, 16, 2, 4, vec=4, math="bf16x3"),
        C("wmma11_bf16", 128, 64, 32, 2, 4, vec=4, math="bf16x3"),
        C("wmma12_bf16", 128, 128, 16, 2, 4, vec=4, math="bf16x3"),
        C("wmma12_bf16", 128, 64, 32, 2, 4, vec=4, math="bf16x3"),
        C("mfma32_xf32", 128, 128, 16, vec=4),
        C("mfma16_xf32", 128, 128, 16, vec=4),
        C("mfma32_bf16", 128, 128, 16, vec=4, math="bf16x3", xcd=1),
        C("mfma16_bf16", 128, 64, 16, vec=4, math="bf16x3", halo=1),
        C("wmma11_bf16", 128, 64, 16, 2, 4, vec=4, math="bf16x3", halo=1),
        C("wmma12_bf16", 128, 64, 16, 2, 4, vec=4, math="bf16x3", halo=1),
        C("mfma16_xf32", 128, 128, 16, vec=4, xcd=1),
        C("mfma16_xf32", 256, 64, 16, 4, 1, vec=4, halo=1),
        C("mfma16_xf32", 256, 64, 16, 4, 1, vec=4, xcd=1, halo=1),
    ],
}

kmd, umd, udd = U("kmd"), U("umd"), U("udd")
w(
    "conv_igemm.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_igemm config fields",
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
        "name": "kernel direction, dtype, atom (device) and channel vector match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_igemm.kernel_match",
    },
)
w(
    "conv_igemm.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_igemm dispatch",
        "dispatch_symbol": "hipkernel.conv_igemm.dispatch",
    },
)
total = 0
for direction, tag in (("FWD", "fwd"), ("BWD", "bwd")):
    ued = U("ued." + tag)
    w(
        f"conv_igemm_{tag}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_IGEMM_{direction}",
            "graph_match": {"native": f"hipkernel.conv_igemm_{tag}.graph_match"},
            "metadata": kmd,
            "knobs": KNOBS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    for dtype, cfgs in CATALOG.items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        # The direct filter is a backward-data loader: forward has no use for it.
        cfgs = withLarge(
            withCf(
                withSwz(
                    withFlat(
                        withFuse(
                            [c for c in cfgs if tag == "bwd" or not c["wdirect"]], tag
                        )
                    )
                )
            )
        )
        for i, c in enumerate(cfgs):
            name = f"conv_igemm.{tag}.{dt}.{c['atom']}.{c['math']}.{c['bm']}x{c['bn']}x{c['bk']}" f".w{c['wm']}x{c['wn']}.t{c['threads']}.v{c['vec']}.s{c['stages']}" + (
                ".xcd" if c["xcd"] else ""
            ) + (
                ".halo" if c["halo"] else ""
            ) + (
                ".wd" if c["wdirect"] else ""
            ) + (
                ".pk" if c["pack"] == 1 else ""
            ) + (
                ".flat" if c["pack"] == 2 else ""
            ) + (
                ".cf" if c["cf"] else ""
            ) + (
                ".swz" if c["swz"] else ""
            ) + (
                f".dma{c['dma']}" if c["dma"] else ""
            ) + (
                f".fuse{c['fuse']}" if c["fuse"] else ""
            ) + (
                ".large" if c["large"] else ""
            )
            kernels.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvIgemm.cpp",
                        "entry_point": "ConvIgemm",
                    },
                    "metadata": {"direction": direction, "dtype": dtype, **c},
                    "priority": 1000 - i,
                }
            )
    total += len(kernels)
    w(
        f"conv_igemm_{tag}.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp." + tag),
            "name": f"hipkernel:conv_igemm_{tag}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
print(total, "kernel configs")
