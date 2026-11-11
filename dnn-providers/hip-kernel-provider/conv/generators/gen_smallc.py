# Generates the CONV_SMALLC_FWD descriptors (small-input-channel forward convolution, K8:
# stems). Each config is a compile-time filter geometry (kd, r, s, stride, dilation), a channel
# bound cmax and a tile; it is admitted on correctness alone (its atom on the device, its ship
# list, the graph's C <= cmax and its exact geometry). Autotune (UHD later) ranks them.
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
# fn 4, 2 x 2 waves) on 7x7 stems: s2 c3 5-11% (deeplab n16 1024²: 4.53 -> 4.02 ms).
ARCH_TILES = {"gfx90a": {7: [(4, 4, 2, 2, 1)]}}


def pow2(v):
    p = 1
    while p < v:
        p *= 2
    return p


def lds_bytes(atom, cmax, g, t, elem):
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
    slots = wm * wq * wn * 16 * fn * 16 * elem
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
catalog = {}  # key -> (archs set, config)
for arch, atoms in ARCH_ATOMS.items():
    for atom in atoms:
        aid, wave, ka, ldsmax, dtype = ATOMS[atom]
        elem = 4 if dtype == "FLOAT" else 2
        for g in GEOMS:
            if not lane_ok(atom, g[2]):
                continue
            for cmax in CMAX_OF[g]:
                for t in TILES + ARCH_TILES.get(arch, {}).get(g[1], []):
                    if lds_bytes(atom, cmax, g, t, elem) > ldsmax:
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


for dtype in ("FLOAT",):  # 16-bit has MMA wherever stems matter; fp32 lacks it on RDNA
    elem = 4
    for g in GEOMS:
        for cmax in CMAX_OF[g]:
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
                )
                key = tuple(c[k] for k in INTS) + (dtype,)
                # fp32 everywhere (only CDNA has fp32 MMA); 16-bit only where no MMA exists.
                catalog.setdefault(key, [set(), c])[0].add("any")

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
ued = U("ued.fwd")
write(
    "conv_smallc_fwd.ued.json",
    {
        "version": "1.0",
        "id": ued,
        "name": "hipkernel:CONV_SMALLC_FWD",
        "graph_match": {"native": "hipkernel.conv_smallc_fwd.graph_match"},
        "metadata": kmd,
        "knobs": INTS,
        "behavior_notes": ["runtime_compilation"],
    },
)
kernels = []
dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}
for i, (archs, c) in enumerate(catalog.values()):
    groups = ",".join(a for a in ORDER if a in archs)
    name = (
        f"conv_smallc.fwd.{dt[c['dtype']]}.{c['atom']}.c{c['cmax']}.{c['kd']}x{c['r']}x{c['s']}"
        f".s{c['sd']}x{c['sh']}x{c['sw']}.d{c['dd']}x{c['dh']}x{c['dw']}"
        f".m{c['fm']}n{c['fn']}.w{c['wm']}x{c['wq']}x{c['wn']}"
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
                "direction": "FWD",
                "dtype": c["dtype"],
                "atom": c["atom"],
                "archs": groups,
                **{k: c[k] for k in INTS},
            },
            "priority": 1000 - i,
        }
    )
write(
    "conv_smallc_fwd.kdp.json",
    {
        "version": "1.0",
        "id": U("kdp.fwd"),
        "name": "hipkernel:conv_smallc_fwd",
        "arch": [],
        "matchers": [umd],
        "engine": ued,
        "dispatch": udd,
        "kernelDescriptors": kernels,
    },
)
print(len(kernels), "kernel configs")
