# Generates the CONV_SMALLC_FWD descriptors (small-input-channel forward convolution, K8:
# stems) and the CONV_SMALLC_BWD ones (stride-1 backward data with few output-gradient
# channels, run on the same kernels as the forward conv over dy). Each config is a compile-time
# filter geometry (kd, r, s, stride, dilation), a channel bound cmax (the kernel's input
# channels: C forward, K backward), a tile and an epilogue (ep 0: LDS slots; ep 1: from
# registers, channels-fastest output only); it is admitted on correctness alone (its atom on the
# device, its ship list, the channel bound and its exact geometry). Autotune (UHD later) ranks
# them.
#
# Every config-distinguishing field is an integer knob; strings (direction, dtype, atom, archs)
# are descriptive and checked by the kernel match.
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("7b1d5e2a-0c4f-4a8e-9d36-2f5e8a1c4b70")
U = lambda name: str(uuid.uuid5(NS, "smallc." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


ATOMS = {  # name: (id, wave, ka, lds bytes per block available, element)
    "mfma16_f16": (1, 64, 16, 65536, "HALF"),
    "mfma16_bf16": (2, 64, 16, 65536, "BFLOAT16"),
    "mfma16_f32": (3, 64, 4, 65536, "FLOAT"),
    "mfma16x32_f16": (4, 64, 32, 163840, "HALF"),
    "mfma16x32_bf16": (5, 64, 32, 163840, "BFLOAT16"),
    "wmma11_f16": (6, 32, 16, 65536, "HALF"),
    "wmma11_bf16": (7, 32, 16, 65536, "BFLOAT16"),
    "wmma12_f16": (8, 32, 16, 65536, "HALF"),
    "wmma12_bf16": (9, 32, 16, 65536, "BFLOAT16"),
}
INTS = [
    "atom_id",
    "wave",
    "ka",
    "cmax",
    "kd",
    "r",
    "s",
    "sd",
    "sh",
    "sw",
    "dd",
    "dh",
    "dw",
    "fm",
    "fn",
    "wm",
    "wq",
    "wn",
    "ep",
]
STRINGS = ["direction", "dtype", "atom", "archs"]

# Filter geometries: (kd, r, s, sd, sh, sw); dilation 1.
GEOMS = [
    (1, 7, 7, 1, 2, 2),  # ResNet / Mask R-CNN / DeepLab stems
    (1, 3, 3, 1, 2, 2),  # EfficientNet / Inception / MobileNet stems
    (1, 3, 3, 1, 1, 1),  # super-resolution / U-Net first layers
    (1, 4, 4, 1, 4, 4),  # ConvNeXt patchify
    (3, 7, 7, 1, 2, 2),  # ResNeXt-3D / I3D stems
    (1, 7, 7, 1, 2, 2),  # R(2+1)D stems
    (3, 3, 3, 1, 1, 1),  # 3-D U-Nets
    (3, 3, 3, 2, 2, 2),
]
# Channel bounds: 3 (RGB) and 4 (RGBA, padded) everywhere; 1 (grayscale / medical) where
# such inputs appear.
CMAX_OF = {g: [3, 4] for g in GEOMS}
for g in [(1, 3, 3, 1, 1, 1), (3, 3, 3, 1, 1, 1), (3, 3, 3, 2, 2, 2)]:
    CMAX_OF[g] = [1, 3, 4]
# Tiles: (fm, fn, wm, wq, wn).
TILES = [(4, 2, 2, 2, 2), (4, 2, 4, 2, 1), (2, 4, 4, 1, 1), (2, 2, 4, 1, 1)]
# Per-arch extras by filter height: gfx90a's 16x16x16 MFMA prefers the 4 x 4 fragment tile (fm 4,
# fn 4, 2 x 2 waves) on 7x7 stems: s2 c3 5-11% (deeplab n16 1024²: 4.53 -> 4.02 ms); and the
# 8-wave 2 x 2 fragment tile on the widest ones (deeplab 4.08 -> 3.81 ms).
ARCH_TILES = {"gfx90a": {7: [(4, 4, 2, 2, 1), (2, 2, 4, 2, 1)]}}
# The register epilogue (ep 1): only where its channels-fastest output occurs (NHWC stems: 3x3 s1,
# 3x3 s2, 4x4 s4; 3 or 4 channels) and on the tiles that measured best per arch (kernel bench,
# leads/sk: K 320 / 512 stems, effnet c3 k24 s2, dx of c128 / c320 / c96 from 3-4 dy channels).
# The wide 2 x 8 fragment tile serves K 512 (gfx942) and the 3-D dx (gfx950).
EP1_GEOMS = [(1, 3, 3, 1, 1, 1), (1, 3, 3, 1, 2, 2), (1, 4, 4, 1, 4, 4)]
EP1_CMAX = [3, 4]
CDNA_EP1 = [(2, 4, 2, 2, 2), (2, 4, 4, 1, 1), (4, 2, 4, 2, 1), (2, 8, 2, 2, 1)]
EP1_TILES_OF = {
    "gfx942": CDNA_EP1,
    "gfx950": CDNA_EP1,
    "gfx90a": CDNA_EP1,
    "gfx908": CDNA_EP1,
    "gfx11": [(2, 4, 2, 2, 2), (2, 4, 4, 1, 1), (2, 8, 2, 2, 1), (2, 2, 4, 1, 1)],
    "gfx12": [(2, 4, 2, 2, 2), (2, 4, 4, 1, 1), (4, 2, 4, 2, 1)],
}
# Backward data runs only at unit stride: the corpus's output layers (3x3 and 3x3x3, dy of up to
# 4 channels). 2-D packs 3 and 4 channels into the same MMA steps, so one bound; 3-D keeps both.
# The slot epilogue only for channels-first dx, on the shared tiles that won there.
BWD_GEOMS = [(1, 3, 3, 1, 1, 1), (3, 3, 3, 1, 1, 1)]
BWD_CMAX = {(1, 3, 3, 1, 1, 1): [4], (3, 3, 3, 1, 1, 1): [3, 4]}
BWD_EP0_TILES = [(4, 2, 4, 2, 1), (2, 4, 4, 1, 1), (2, 2, 4, 1, 1)]


def pow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


def lds_bytes(atom, cmax, g, t, elem, ep=0):
    _, wave, ka, _, _ = ATOMS[atom]
    kd, r, s, sd, sh, sw = g
    fm, fn, wm, wq, wn = t
    sp = pow2(s)
    kp = (cmax * kd * r * sp + ka - 1) // ka * ka
    th, tw, bn = wm * fm, wq * 16, wn * fn * 16
    wrows = (th - 1) * sh + (r - 1) + 1
    wcols = ((tw - 1) * sw + (sp - 1) + 2) // 2 * 2
    win = (cmax * kd * wrows * wcols * elem + 15) // 16 * 16
    w = bn * (kp + 16 // elem) * elem
    slots = wm * wq * wn * 16 * fn * 16 * elem if ep == 0 else 0
    return win + w + slots


def lane_ok(atom, s):
    _, wave, ka, _, _ = ATOMS[atom]
    kpl = ka // 4 if wave == 64 else (ka // 2 if atom.startswith("wmma12") else ka)
    sp = pow2(s)
    return kpl % sp == 0 if kpl >= sp else sp % kpl == 0


ARCH_ATOMS = {
    "gfx950": ["mfma16x32_f16", "mfma16x32_bf16", "mfma16_f32"],
    "gfx942": ["mfma16_f16", "mfma16_bf16", "mfma16_f32"],
    "gfx90a": ["mfma16_f16", "mfma16_bf16", "mfma16_f32"],
    "gfx908": ["mfma16_f16", "mfma16_f32"],
    "gfx11": ["wmma11_f16", "wmma11_bf16"],
    "gfx12": ["wmma12_f16", "wmma12_bf16"],
}
ORDER = [
    "gfx906",
    "gfx908",
    "gfx90a",
    "gfx942",
    "gfx950",
    "gfx1030",
    "gfx11",
    "gfx12",
    "any",
]


def mma_configs(catalog, geoms, ep, cmax_of, tiles_of, sixteen_only=False):
    """MMA configs: per arch atom, geometry, channel bound (cmax_of(g)) and tile (tiles_of(arch,
    g)), those whose LDS fits."""
    for arch, atoms in ARCH_ATOMS.items():
        for atom in atoms:
            aid, wave, ka, ldsmax, dtype = ATOMS[atom]
            if sixteen_only and dtype == "FLOAT":
                continue
            elem = 4 if dtype == "FLOAT" else 2
            for g in geoms:
                if not lane_ok(atom, g[2]):
                    continue
                for cmax in cmax_of(g):
                    for t in tiles_of(arch, g):
                        if lds_bytes(atom, cmax, g, t, elem, ep) > ldsmax:
                            continue
                        c = dict(
                            dtype=dtype,
                            atom=atom,
                            atom_id=aid,
                            wave=wave,
                            ka=ka,
                            cmax=cmax,
                            kd=g[0],
                            r=g[1],
                            s=g[2],
                            sd=g[3],
                            sh=g[4],
                            sw=g[5],
                            dd=1,
                            dh=1,
                            dw=1,
                            fm=t[0],
                            fn=t[1],
                            wm=t[2],
                            wq=t[3],
                            wn=t[4],
                            ep=ep,
                        )
                        key = tuple(c[k] for k in INTS) + (dtype,)
                        catalog.setdefault(key, [set(), c])[0].add(arch)


# The FMA kernel (256 threads; fp32 where no fp32 MMA exists, and any dtype anywhere):
# tiles (fm, fn, wm, wq, wn) with 256 / (16 wn / fn) * fm == wm * 16 wq.
FMA_TILES = [
    (8, 8, 8, 2, 4),
    (4, 16, 8, 2, 4),
    (4, 8, 4, 2, 4),
    (4, 4, 4, 2, 2),
    (2, 2, 4, 1, 1),
]


def fma_lds(cmax, g, t, elem):
    kd, r, s, sd, sh, sw = g
    fm, fn, wm, wq, wn = t
    sp = pow2(s)
    th, tw, bn = wm, wq * 16, wn * 16
    wrows = (th - 1) * sh + (r - 1) + 1
    wcols = ((tw - 1) * sw + (sp - 1) + 2) // 2 * 2
    win = (cmax * kd * wrows * wcols * elem + 15) // 16 * 16
    return win + cmax * kd * r * sp * (bn + 4) * 4


def fma_configs(catalog, geoms, cmax_of, archs):
    # fp32 only: 16-bit has MMA wherever stems matter; fp32 lacks it on RDNA.
    dtype, elem = "FLOAT", 4
    for g in geoms:
        for cmax in cmax_of(g):
            for t in FMA_TILES:
                if fma_lds(cmax, g, t, elem) > 65536:
                    continue
                c = dict(
                    dtype=dtype,
                    atom="fma",
                    atom_id=0,
                    wave=0,
                    ka=1,
                    cmax=cmax,
                    kd=g[0],
                    r=g[1],
                    s=g[2],
                    sd=g[3],
                    sh=g[4],
                    sw=g[5],
                    dd=1,
                    dh=1,
                    dw=1,
                    fm=t[0],
                    fn=t[1],
                    wm=t[2],
                    wq=t[3],
                    wn=t[4],
                    ep=0,
                )
                key = tuple(c[k] for k in INTS) + (dtype,)
                catalog.setdefault(key, [set(), c])[0].update(archs)


# Forward: the original configs first (names, ids and priorities unchanged), then the ep 1
# twins. Backward: the stride-1 geometries, both epilogues.
catalog = {}  # key -> (archs set, config)
mma_configs(
    catalog,
    GEOMS,
    0,
    lambda g: CMAX_OF[g],
    lambda a, g: TILES + ARCH_TILES.get(a, {}).get(g[1], []),
)
# fp32 everywhere for the forward FMA kernel (only CDNA has fp32 MMA; it is the fallback).
fma_configs(catalog, GEOMS, lambda g: CMAX_OF[g], ["any"])
mma_configs(
    catalog,
    EP1_GEOMS,
    1,
    lambda g: EP1_CMAX,
    lambda a, g: EP1_TILES_OF[a],
    sixteen_only=True,
)
bwd_catalog = {}
mma_configs(
    bwd_catalog, BWD_GEOMS, 0, lambda g: BWD_CMAX[g], lambda a, g: BWD_EP0_TILES
)
mma_configs(
    bwd_catalog, BWD_GEOMS, 1, lambda g: BWD_CMAX[g], lambda a, g: EP1_TILES_OF[a]
)
# fp32 FMA only where no fp32 matrix core exists and the backward kernels are validated (RDNA3/4).
fma_configs(bwd_catalog, BWD_GEOMS, lambda g: BWD_CMAX[g], ["gfx11", "gfx12"])

kmd, umd, udd = U("kmd"), U("umd"), U("udd")
write(
    "conv_smallc.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_smallc config fields",
        "fields": [{"name": f, "type": "string"} for f in STRINGS]
        + [{"name": f, "type": "int"} for f in INTS],
    },
)
write(
    "kernel_matches_graph.umd.json",
    {
        "version": "1.0",
        "id": umd,
        "name": "kernel dtype, channel bound, filter, stride, dilation, atom (device, ship list) match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_smallc.kernel_match",
    },
)
write(
    "conv_smallc.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_smallc dispatch",
        "dispatch_symbol": "hipkernel.conv_smallc.dispatch",
    },
)
dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}
total = 0
for direction, cat in (("fwd", catalog), ("bwd", bwd_catalog)):
    ued = U("ued." + direction)
    write(
        f"conv_smallc_{direction}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_SMALLC_{direction.upper()}",
            "graph_match": {"native": f"hipkernel.conv_smallc_{direction}.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    for i, (archs, c) in enumerate(cat.values()):
        groups = ",".join(a for a in ORDER if a in archs)
        name = (
            f"conv_smallc.{direction}.{dt[c['dtype']]}.{c['atom']}.c{c['cmax']}.{c['kd']}x{c['r']}x{c['s']}"
            f".s{c['sd']}x{c['sh']}x{c['sw']}.d{c['dd']}x{c['dh']}x{c['dw']}"
            f".m{c['fm']}n{c['fn']}.w{c['wm']}x{c['wq']}x{c['wn']}"
            + (f".e{c['ep']}" if c["ep"] else "")
        )
        kernels.append(
            {
                "version": "1.0",
                "id": U(name),
                "name": name,
                "kernel_source": {
                    "kind": "embedded_source",
                    "source_file": "kernels/ConvSmallc.cpp",
                    "entry_point": "ConvSmallc",
                },
                "metadata": {
                    "direction": direction.upper(),
                    "dtype": c["dtype"],
                    "atom": c["atom"],
                    "archs": groups,
                    **{k: c[k] for k in INTS},
                },
                "priority": 1000 - i,
            }
        )
    write(
        f"conv_smallc_{direction}.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp." + direction),
            "name": f"hipkernel:conv_smallc_{direction}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
    print(direction, len(kernels), "kernel configs")
    total += len(kernels)
