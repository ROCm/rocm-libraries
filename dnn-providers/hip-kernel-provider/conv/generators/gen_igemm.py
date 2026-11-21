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
    "patch",
    "split_policy",
    "cfx",
    "flex",
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
# served for packed channels-first outputs). Legacy matrix-core configs but tap packing have a cf twin.
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
    patch=0,
    split_policy=0,
    cfx=0,
    flex=0,
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
        patch=patch,
        split_policy=split_policy,
        cfx=cfx,
        flex=flex,
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
    "wmma125_f16": 32,
    "wmma125_bf16": 32,
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


# patch 1: patch convolutions (filter equal to the stride, no padding) run as one GEMM. Forward
# reads each filter row's taps as one contiguous run of the reduction (W / S pixels of S C
# channels), backward data runs one GEMM over dy's positions whose T R S C columns scatter to dx
# (K1_PATCH), with no stride phases. Twins of the plain configs without halo, tap packing, group
# fusion or large images (backward data: channels-last output only), plus flat-reduction twins
# with swizzled tiles for runs that the k-step does not divide (ViT's 16 x 3 = 48).
def withPatch(cfgs, tag):
    base = [
        c
        for c in cfgs
        if c["math"] == "native"
        and not (c["halo"] or c["pack"] == 1 or c["fuse"] or c["large"] or c["patch"])
        and not (tag == "bwd" and c["cf"])
    ]
    twins = [dict(c, patch=1) for c in base]
    twins += [
        dict(c, patch=1, pack=2)
        for c in base
        if c["swz"] and not (c["pack"] or c["cf"]) and tag == "fwd"
    ]
    # Patch-only tiles of the 16-bit matrix-core atoms: 4-wide vectors for RGB runs of 4 x 3
    # channels (ConvNeXt's 4x4 stem), which no 8-wide vector divides, and 64-deep k-steps on the
    # 64 x 64 tile (short GEMMs of a few hundred rows a block, ViT-B/16 at batch 16).
    atoms = {c["atom"] for c in cfgs}
    extra = []
    for atom, tiles in PATCH_TILES.items():
        if atom in atoms:
            extra += [
                C(atom, bm, bn, bk, wm, wn, vec=v) for bm, bn, bk, wm, wn, v in tiles
            ]
    extra = withSwz(withFlat(extra))
    # gfx950: the same short GEMMs on its 16x16x32 / 32x32x16 atoms with DMA rings (DMA serves
    # the flat reduction too, so ViT's 48-channel runs keep a 2- to 3-deep load ring).
    for atom in [a for a in PATCH_DMA_ATOMS if a in atoms]:
        extra += [
            C(atom, bm, bn, 64, wm, wn, vec=8, xcd=1, swz=1, dma=d)
            for bm, bn, wm, wn in ((64, 64, 2, 2), (64, 128, 2, 2), (128, 128, 2, 2))
            for d in (2, 3)
        ]
    if tag == "fwd":
        extra = withCf(extra)
        extra += [
            dict(c, pack=2) for c in extra if c["swz"] and not (c["pack"] or c["cf"])
        ]
    out = cfgs + twins + patchTile(cfgs, tag)
    seen = {tuple(sorted(c.items())) for c in out}
    for c in extra:
        c = dict(c, patch=1)
        if tuple(sorted(c.items())) not in seen:
            seen.add(tuple(sorted(c.items())))
            out.append(c)
    return out


# cfx 1: 1-D graphs whose activations are packed channels-first (N C L, PyTorch's only conv1d
# layout) read in place: no staging transpose of the input, the output written channels-first.
# Matrix-core configs run the channels-first halo loop (K1HaloCf), halo h staging up to h input
# pixels per output row (h 1: stride 1; h 2: forward stride 2, row tiles of at most 128; backward
# data gathers dy at stride 1 in every stride phase); twins of the
# plain channels-first-output configs without swizzled tiles, DMA, tap packing, direct filter,
# group fusion, large images or patch view. The FMA kernel reads in place without a halo
# (K1TilesCf) and writes the output channels-first itself: twins of every FMA config.
def withCfx(cfgs, tag):
    out = list(cfgs)
    seen = {tuple(sorted(c.items())) for c in out}

    def add(c):
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)

    for c in cfgs:
        if (
            c["math"] != "native"
            or c["atom"].endswith("_xf32")
            or c["swz"]
            or c["dma"]
            or c["pack"]
            or c["wdirect"]
            or c["fuse"]
            or c["large"]
            or c["patch"]
        ):
            continue
        if c["atom"] == "fma":
            add(dict(c, cfx=1))
        elif c["cf"]:
            add(dict(c, cfx=1, halo=1))
            if c["bm"] <= 128 and tag == "fwd":
                add(dict(c, cfx=1, halo=2))
    return out


PATCH_TILES = {
    atom: [
        (64, 64, 16, 2, 2, 4),
        (64, 64, 32, 2, 2, 4),
        (128, 64, 32, 2, 2, 4),
        (64, 64, 64, 2, 2, 8),
    ]
    for atom in ("mfma16_f16", "mfma16_bf16")
}
PATCH_TILES.update(
    {
        atom: [(64, 64, 32, 2, 2, 4), (128, 64, 32, 2, 2, 4), (64, 64, 64, 2, 2, 8)]
        for atom in ("mfma16x32_f16", "mfma16x32_bf16")
    }
)
# patch 2: the patch tile (k1PatchMma, ConvIgemmPatch.h), a dedicated GEMM for patch
# convolutions: two LDS stages with one barrier a k-step, swizzled 16-byte chunks, the forward
# filter-row cursor, backward data's filter staged [(t, r, s, c)][k] or (wdirect) read K-major in
# place and transposed (ds_read_b64_tr_b16 on gfx950), no split. Tiles per 16-bit 16x16 atom (bm, bn, bk, wm, wn, vec); 4-wide vectors for
# RGB runs of 4 x 3 channels.
PATCH_TILE = {
    "mfma16x32": [
        (64, 64, 64, 2, 2, 8),
        (64, 128, 64, 2, 2, 8),
        (128, 64, 64, 2, 2, 8),
        (128, 128, 64, 2, 2, 8),
        (128, 128, 64, 1, 4, 8),
        (32, 128, 64, 1, 4, 8),
        (256, 128, 64, 2, 4, 8),
        (256, 256, 64, 2, 4, 8),
        (64, 64, 32, 2, 2, 4),
        (128, 64, 32, 2, 2, 4),
    ],
    "mfma16": [
        (64, 64, 32, 2, 2, 8),
        (128, 64, 32, 2, 2, 8),
        (128, 128, 32, 2, 2, 8),
        (64, 64, 32, 2, 2, 4),
    ],
    "wmma11": [
        (64, 64, 32, 2, 2, 8),
        (128, 64, 32, 2, 4, 8),
        (128, 128, 32, 2, 4, 8),
        (64, 64, 32, 2, 2, 4),
    ],
}
PATCH_TILE["wmma12"] = PATCH_TILE["wmma11"]


# split_policy twins (splitPlanFor): 1 runs unsplit, N >= 2 exactly N splits. Every
# matrix-core gfx9 config that can split gets a no-split twin (an under-filled small problem may
# run faster unsplit than split plus the reduce launch), and the widest tiles a 32-split twin
# (past the default's 16, for GEMM-shaped layers with long reductions). The kernel match admits a
# twin only where its plan differs from the default's, so autotune times distinct plans only.
SPLIT_POLICY_WIDE = 32


def withSplitPolicy(cfgs):
    out = list(cfgs)
    for c in cfgs:
        if (
            not c["atom"].startswith("mfma")
            or c["atom"].endswith("_xf32")
            or c["math"] != "native"
            or c["patch"]
            or c["large"]
            or c["split_policy"]
        ):
            continue
        out.append(dict(c, split_policy=1))
        if c["bm"] * c["bn"] >= 256 * 128:
            out.append(dict(c, split_policy=SPLIT_POLICY_WIDE))
    return out


def patchTile(cfgs, tag):
    # The catalog's own 16-bit atoms (its fp32 catalog runs its three-bf16 configs on 16-bit
    # atoms, which the patch tile does not serve).
    atoms = {
        c["atom"] for c in cfgs if c["math"] == "native" and "f32" not in c["atom"]
    }
    out = []
    for family, tiles in PATCH_TILE.items():
        for atom in sorted(atoms):
            if atom.rsplit("_", 1)[0] != family or atom.endswith("_908"):
                continue
            out += [
                C(atom, bm, bn, bk, wm, wn, vec=v, patch=2, wdirect=wd)
                for bm, bn, bk, wm, wn, v in tiles
                for wd in ((0, 1) if tag == "bwd" else (0,))
            ]
    return out


# In emission order (a tuple, not a set: a set's iteration order follows string hashing, which
# PYTHONHASHSEED and unrelated edits change). The order is the one the catalogs carry.
PATCH_DMA_ATOMS = ("mfma32x16_f16", "mfma16x32_f16", "mfma32x16_bf16", "mfma16x32_bf16")
PATCH_TILES["mfma16_bf16_908"] = [(64, 64, 16, 2, 2, 4), (128, 64, 32, 2, 2, 4)]


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
        # Backward data from the K T R S C filter directly (the per-call transposed copy is a third
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


def wmma125(atom, fp32=False):
    """Native gfx1250 atoms: wave32, K=32 for 16-bit, K=4 for fp32."""
    bk = 16 if fp32 else 32
    vec = 4 if fp32 else 8
    tiles = [
        (64, 64, bk, 2, 2),
        (128, 64, bk, 2, 2),
        (128, 128, bk, 2, 4),
        (128, 128, 2 * bk, 2, 4),
        (128, 256, bk, 2, 4),
    ]
    configs = [C(atom, bm, bn, k, wm, wn, vec=vec) for bm, bn, k, wm, wn in tiles]
    configs += [
        C(atom, 64, 64, bk, 2, 2, vec=1),
        C(atom, 64, 64, bk, 2, 2, vec=vec, stages=2),
        C(atom, 128, 128, bk, 2, 4, vec=vec, stages=2),
        C(atom, 128, 64, bk, 2, 2, vec=vec, wdirect=1),
        C(atom, 64, 64, bk, 2, 2, vec=vec, wdirect=1),
        C(atom, 64, 16, bk, 4, 1, vec=vec, pack=1),
    ]
    if not fp32:
        configs.append(C(atom, 128, 64, bk, 2, 2, vec=4))
    return configs


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
    # TF32 configs (three-bf16 on every matrix-core arch; xf32 on gfx942): the CONV_IGEMM_*_TF32
    # engines' catalogs (isTf32), not the exact engines'.
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


def isTf32(c):
    """A TF32 config: fp32 through rounded operands (three-bf16, or gfx942's xf32 atoms)."""
    return c["math"] == "bf16x3" or c["atom"].endswith("_xf32")


CATALOG125 = {
    "HALF": wmma125("wmma125_f16"),
    "BFLOAT16": wmma125("wmma125_bf16"),
    "FLOAT": wmma125("wmma125_f32", fp32=True),
}

# The same three-product algorithm as the legacy TF32 tier, on gfx1250's native K32
# BF16 fragments. Keep its candidates separate from the exact native tier.
CATALOG125TF32 = {
    "FLOAT": [
        C("wmma125_bf16", 128, 128, 32, 2, 4, vec=4, math="bf16x3"),
        C("wmma125_bf16", 128, 64, 32, 2, 2, vec=4, math="bf16x3"),
    ],
}

FLEX_ATOMS = {"HALF": "mfma16x32_f16", "BFLOAT16": "mfma16x32_bf16"}
# Tall 1-wide-vector tiles for reductions of a few channels (backward data into a wide layer
# from 2 to 7 output channels, forward from a few input channels): a block's fixed setup and
# its single k-step spread over 256 rows, 8 waves. 16x16 atoms of the CDNA catalogs.
NARROW_ATOMS = {
    "HALF": "mfma16_f16",
    "BFLOAT16": "mfma16_bf16",
    "FLOAT": "mfma16_f32",
}
# gfx950's double-rate 32 x 32 atoms for the col2im configs (patch 3).
COL2IM_WIDE = {"HALF": "mfma32x16_f16", "BFLOAT16": "mfma32x16_bf16"}
# Col2im tiles beside the direct-filter twins: a 64-deep k-step and 256 dy positions a block on
# the CDNA 32 x 32 atoms (more of the filter's rows in flight a block).
COL2IM_TILES = {
    dt: [C(atom, 128, 128, 64, vec=8), C(atom, 256, 128, 32, vec=8)]
    for dt, atom in (("HALF", "mfma32_f16"), ("BFLOAT16", "mfma32_bf16"))
}
# The flex tile's register-staged form (gfx942: 64 KiB LDS, dword LDS DMA): 256 x 128, 128-byte
# reduction steps (64 16-bit or 32 fp32 channels).
FLEX_REG_ATOMS = {
    "HALF": ("mfma16_f16", 2),
    "BFLOAT16": ("mfma16_bf16", 2),
    "FLOAT": ("mfma16_f32", 4),
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


def kernels_for(catalog, direction, tag, priority=1000):
    kernels = []
    for dtype, cfgs in catalog.items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        # The direct filter is a backward-data loader: forward has no use for it.
        cfgs = withPatch(
            withLarge(
                withCf(
                    withSwz(
                        withFlat(
                            withFuse(
                                [c for c in cfgs if tag == "bwd" or not c["wdirect"]],
                                tag,
                            )
                        )
                    )
                )
            ),
            tag,
        )
        # Channels-first-input twins (cfx) are in the legacy catalogs, not the gfx1250 ones.
        if catalog is CATALOG:
            cfgs = withSplitPolicy(withCfx(cfgs, tag))
        # The measured explicit no-split candidate is appended after all legacy twins,
        # preserving their names, enumeration order and priorities.
        if tag == "bwd" and dtype == "HALF" and catalog is CATALOG125:
            cfgs += [
                C("wmma125_f16", 64, 64, 32, 2, 2, vec=8, cf=1, split_policy=1),
            ]
        if tag == "fwd" and dtype == "HALF" and catalog is CATALOG125:
            cfgs += [
                C("wmma125_f16", 64, 64, 32, 2, 2, vec=8, cf=2),
            ]
        # flex 1: the flex tile (ConvIgemmFlex.h, gfx950): 256 flattened output positions x 256,
        # 192 (whole tiles of Wan VAE's 192 and 384 channels) or 128 channels, both operands by
        # LDS DMA in 64-channel k-steps, eight waves in two ping-pong groups, its default split
        # the auto plan's tail split. Appended after every other config, with its split_policy
        # twins.
        if catalog is CATALOG and dtype in FLEX_ATOMS:
            cfgs += withSplitPolicy(
                [
                    C(FLEX_ATOMS[dtype], 256, bn, 64, 4, 2, vec=8, xcd=1, flex=1)
                    for bn in (256, 128, 192)
                ]
            )
        # flex 1 on 16x16x16 / 16x16x4 atoms: the register-staged form (ConvIgemmFlexReg.h,
        # gfx942), its default split the auto plan's tail split, with its split_policy twins.
        if catalog is CATALOG and dtype in FLEX_REG_ATOMS:
            atom, eb = FLEX_REG_ATOMS[dtype]
            cfgs += withSplitPolicy(
                [C(atom, 256, 128, 128 // eb, 4, 2, vec=16 // eb, xcd=1, flex=1)]
            )
        # Forward's 16-bit 256 x 256 register-staged tile on 64-byte steps (32 channels): a
        # third fewer operand bytes per MMA than 256 x 128, for wide-channel layers over few
        # positions. (fp32's 16x16x4 form of it spills.)
        if catalog is CATALOG and dtype in ("HALF", "BFLOAT16") and tag == "fwd":
            atom, eb = FLEX_REG_ATOMS[dtype]
            cfgs += withSplitPolicy(
                [C(atom, 256, 256, 64 // eb, 4, 2, vec=16 // eb, xcd=1, flex=1)]
            )
        # Backward data's register-staged flex tile reading the graph's K T R S C filter in
        # place (wdirect): no per-call transposed copy of the filter, which costs more than the
        # GEMM on wide-channel layers over few positions. Appended after the other flex configs.
        if catalog is CATALOG and dtype in FLEX_REG_ATOMS and tag == "bwd":
            atom, eb = FLEX_REG_ATOMS[dtype]
            cfgs += withSplitPolicy(
                [
                    C(
                        atom,
                        256,
                        128,
                        128 // eb,
                        4,
                        2,
                        vec=16 // eb,
                        xcd=1,
                        wdirect=1,
                        flex=1,
                    )
                ]
            )
        # patch 3: col2im backward data (withCol2imView), one GEMM over dy's positions whose
        # T R S C columns a second launch sums into dx, for layers whose dx rows mostly multiply
        # padding ("valid" small 3-D maps). Twins of the plain direct-filter configs (the filter
        # read in place: its K-major rows are the GEMM's B), and on gfx950 its 32 x 32 x 16 atom.
        # Appended after every other config.
        # Tall 1-wide-vector tiles with their flat, channels-first-output, channels-first-input and
        # split-policy twins, appended after every other legacy config.
        if catalog is CATALOG and dtype in NARROW_ATOMS:
            atom = NARROW_ATOMS[dtype]
            cfgs += withSplitPolicy(
                withCfx(
                    withCf(
                        withFlat(
                            [
                                C(atom, 256, 64, 16, 4, 2, vec=1),
                                C(atom, 256, 128, 16, 4, 2, vec=1),
                            ]
                        )
                    ),
                    tag,
                )
            )
        if catalog is CATALOG and tag == "bwd":
            plain = [
                c
                for c in catalog[dtype]
                if c["wdirect"]
                and c["atom"] != "fma"
                and c["math"] == "native"
                and not (c["halo"] or c["pack"] or c["fuse"] or c["cf"] or c["swz"])
            ]
            cfgs += [dict(c, patch=3) for c in plain]
            cfgs += [dict(c, wdirect=1, patch=3) for c in COL2IM_TILES.get(dtype, [])]
            if dtype in COL2IM_WIDE:
                cfgs += [
                    C(COL2IM_WIDE[dtype], 128, 128, 64, vec=8, wdirect=1, patch=3),
                    C(
                        COL2IM_WIDE[dtype],
                        256,
                        128,
                        64,
                        4,
                        2,
                        vec=8,
                        wdirect=1,
                        patch=3,
                    ),
                ]
        for i, c in enumerate(cfgs):
            name = f"conv_igemm.{tag}.{dt}.{c['atom']}.{c['math']}.{c['bm']}x{c['bn']}x{c['bk']}" f".w{c['wm']}x{c['wn']}.t{c['threads']}.v{c['vec']}.s{c['stages']}" + (
                ".xcd" if c["xcd"] else ""
            ) + (
                (".halo" if c["halo"] == 1 else f".halo{c['halo']}")
                if c["halo"]
                else ""
            ) + (
                ".wd" if c["wdirect"] else ""
            ) + (
                ".pk" if c["pack"] == 1 else ""
            ) + (
                ".flat" if c["pack"] == 2 else ""
            ) + (
                ".cfio" if c["cf"] == 2 else ".cf" if c["cf"] else ""
            ) + (
                ".swz" if c["swz"] else ""
            ) + (
                f".dma{c['dma']}" if c["dma"] else ""
            ) + (
                f".fuse{c['fuse']}" if c["fuse"] else ""
            ) + (
                ".large" if c["large"] else ""
            ) + (
                ".patch" if c["patch"] == 1 else ""
            ) + (
                ".patchtile" if c["patch"] == 2 else ""
            ) + (
                ".col2im" if c["patch"] == 3 else ""
            ) + (
                ".cfx" if c["cfx"] else ""
            ) + (
                ".nosplit"
                if c["split_policy"] == 1
                else f".split{c['split_policy']}" if c["split_policy"] else ""
            ) + (
                ".flex" if c["flex"] else ""
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
                    "priority": priority - i,
                }
            )
    return kernels


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
            # RFC 0025 numerical notes. No "nondeterministic": every config is
            # bitwise reproducible (fixed-order sums, no atomics), which the
            # family's device test checks by relaunching each config.
            "numerical_notes": [],
        },
    )
    kernels = kernels_for(CATALOG, direction, tag)
    total += len(kernels)
    # The TF32 configs make an engine of their own, which reports the TENSOR_CORE numerical
    # note (cuDNN's note for TF32 engines), so a caller can exclude them by note.
    ued_tf32 = U("ued." + tag + ".tf32")
    w(
        f"conv_igemm_{tag}_tf32.ued.json",
        {
            "version": "1.0",
            "id": ued_tf32,
            "name": f"hipkernel:CONV_IGEMM_{direction}_TF32",
            "graph_match": {"native": f"hipkernel.conv_igemm_{tag}.graph_match"},
            "metadata": kmd,
            "knobs": KNOBS,
            "behavior_notes": ["runtime_compilation"],
            # RFC 0025 numerical notes: TENSOR_CORE (fp32 through TF32-class rounded
            # operands). Deterministic as the exact engine.
            "numerical_notes": ["tensor_core"],
        },
    )
    for suffix, engine, members in (
        ("", ued, [k for k in kernels if not isTf32(k["metadata"])]),
        ("_tf32", ued_tf32, [k for k in kernels if isTf32(k["metadata"])]),
    ):
        w(
            f"conv_igemm_{tag}{suffix}.kdp.json",
            {
                "version": "1.0",
                "id": U("kdp." + tag + suffix.replace("_", ".")),
                "name": f"hipkernel:conv_igemm_{tag}{suffix}",
                "arch": [],
                "matchers": [umd],
                "engine": engine,
                "dispatch": udd,
                "kernelDescriptors": members,
            },
        )
    for suffix, catalog, engine in (
        ("_gfx1250", CATALOG125, ued),
        ("_tf32_gfx1250", CATALOG125TF32, ued_tf32),
    ):
        native_kernels = kernels_for(catalog, direction, tag, priority=1100)
        total += len(native_kernels)
        assert len({k["id"] for k in native_kernels}) == len(native_kernels)
        native_pack = {
            "version": "1.0",
            "id": U("kdp." + tag + suffix.replace("_", ".")),
            "name": f"hipkernel:conv_igemm_{tag}{suffix}",
            "arch": ["gfx1250"],
            "matchers": [umd],
            "engine": engine,
            "dispatch": udd,
            "kernelDescriptors": native_kernels,
        }
        assert len(json.dumps(native_pack, indent=2).encode()) < 500 * 1024
        w(f"conv_igemm_{tag}{suffix}.kdp.json", native_pack)
print(total, "kernel configs")
