# Generates the CONV_STEM_BWD descriptors (backward data for at most 4 input channels: stems).
# Each config is admitted on correctness alone (its atom on the device and taking the graph's
# dtype, its layout dy's, its strides and window the problem's, its GEMM columns at least the
# channels times the phases, its LDS within the device's); autotune (UHD later) ranks them.
#   python3 gen_conv_stem_bwd.py <descriptors/conv/stem_bwd> [--universe]
#
# build() enumerates the config space (the universe). The catalog ships the configs listed in
# conv_stem_bwd_kept.txt beside this script; --universe writes every config instead, the input
# of a per-config bench that re-selects the list (the list's header says how).
#
# The window of one axis: offset floor((s - 1 + p) / s), extent offset - ceil((p - f + 1) / s) + 1
# (ConvStemBwdProblem.cpp windowOf). GEOMETRIES lists the (stride, extent) pairs the catalog
# covers, square: 7x7 s2 p3 -> (2, 4); 3x3 s2 p1 -> (2, 2); 6x6 s2 p2 / 5x5 s2 p2 -> (2, 3);
# 2x2 s2 -> (2, 1); 3x3 / 5x5 / 7x7 s1 "same" -> (1, 3 / 5 / 7); 1x1 -> (1, 1); 4x4 s4 patchify
# -> (4, 1); 11x11 s4 p2 -> (4, 4).
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
UNIVERSE = sys.argv[2:] == ["--universe"]
if sys.argv[2:] not in ([], ["--universe"]):
    sys.exit("usage: gen_conv_stem_bwd.py <dir> [--universe]")
out.mkdir(parents=True, exist_ok=True)
KEPT_FILE = pathlib.Path(__file__).with_name("conv_stem_bwd_kept.txt")
NS_UUID = uuid.UUID("8f0c6a52-1d3e-4b7a-9c25-e4f1a0b3d7c9")
U = lambda name: str(uuid.uuid5(NS_UUID, "conv_stem_bwd." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


# atom_id order: the pack's conv_stem_bwd::ATOMS; (DIM, KA, WAVE).
ATOMS = [
    "mfma16_f16",
    "mfma16_bf16",
    "mfma16_bf16_908",
    "mfma16_f32",
    "mfma32_f16",
    "mfma32_bf16",
    "mfma32_bf16_908",
    "mfma32_f32",
    "mfma16x32_f16",
    "mfma16x32_bf16",
    "mfma32x16_f16",
    "mfma32x16_bf16",
    "wmma11_f16",
    "wmma11_bf16",
    "wmma12_f16",
    "wmma12_bf16",
    "fma",
]
SHAPE = {
    "mfma16_f16": (16, 16, 64),
    "mfma16_bf16": (16, 16, 64),
    "mfma16_bf16_908": (16, 8, 64),
    "mfma16_f32": (16, 4, 64),
    "mfma16x32_f16": (16, 32, 64),
    "mfma16x32_bf16": (16, 32, 64),
    "wmma11_f16": (16, 16, 32),
    "wmma11_bf16": (16, 16, 32),
    "wmma12_f16": (16, 16, 32),
    "wmma12_bf16": (16, 16, 32),
    "fma": (16, 4, 16),
}
# Per dtype, the atoms the catalog carries (gfx950 runs mfma16_* too; its 16 x 32 atoms halve
# the steps).
FAMILIES = {
    "HALF": ["mfma16_f16", "mfma16x32_f16", "wmma11_f16", "wmma12_f16", "fma"],
    "BFLOAT16": [
        "mfma16_bf16",
        "mfma16_bf16_908",
        "mfma16x32_bf16",
        "wmma11_bf16",
        "wmma12_bf16",
        "fma",
    ],
    "FLOAT": ["mfma16_f32", "fma"],
}
# 16-bit FMA configs serve only devices without a matrix core for the type: they ship in
# conv_stem_bwd_<dtype>_fma906 packs that list these architectures.
FMA16_ARCHS = ["gfx906"]
GEOMETRIES = [
    (2, 4),
    (2, 2),
    (2, 3),
    (2, 1),
    (1, 3),
    (1, 5),
    (1, 7),
    (1, 1),
    (4, 1),
    (4, 4),
]
# 3-D (ft depth taps, unit depth stride: the dx planes in the grid, the taps in the reduction):
# (stride, extent, taps). 1x7x7 s(1,2,2) p(0,3,3) -> (2, 4, 1); 3x7x7 s(1,2,2) p(1,3,3) ->
# (2, 4, 3); 3x3x3 s1 "same" -> (1, 3, 3). After every 2-D config, which keep their order.
GEOMETRIES_3D = [(2, 4, 1), (2, 4, 3), (1, 3, 3)]
# Stride 1 on a space-to-depth view of dy (sd = 2: 2 x 2 super-pixels, the phases filling the
# GEMM columns, the sub-pixels joining the reduction): the window extents
# (ConvStemBwdProblem.cpp spaceToDepthWindowOf) of 3x3 p1 -> 2, 5x5 p2 -> 3, 7x7 p3 -> 4. After
# the 3-D configs.
SPACE_TO_DEPTH = 2
SD_WINDOWS = [2, 3, 4]
# Space-to-depth only pays where it fills idle matrix-core columns, so no FMA configs (their
# columns are exact: measured 1.3-1.5x slower on ssd300 3x3 fp32, gfx1101 / gfx1201). Per layout,
# the chunk / tile pairs that won or came within 5% on the RDNA A/B (unet, realesrgan, superres
# NCHW fp16; sd15 / sdxl NHWC bf16); the rest never did there and each adds a candidate per graph.
# Keyed by (layout, vector loads); channels-first column vectors read both sub-columns at once.
SD_KC_TILES = {
    (0, False): [(16, (4, 2)), (16, (8, 2)), (32, (8, 2))],
    (0, True): [(16, (4, 2)), (16, (8, 2)), (32, (8, 2))],
    (1, True): [(64, (4, 2)), (32, (4, 4))],
}
INTS = [
    "atom_id",
    "layout",
    "vec",
    "u",
    "v",
    "nb",
    "kc",
    "th",
    "fm",
    "sh",
    "sw",
    "ft",
    "sd",
]
STRINGS = ["direction", "dtype", "atom"]
FIELDS = STRINGS + INTS


def eb(dtype):
    return 4 if dtype == "FLOAT" else 2


def lds_bytes(e, u, v, nb, kc, th, tw):  # ConvStemBwdArgs.h sbLdsBytes
    return ((th + u - 1) * (tw + v - 1) + nb * u * v) * (kc + 16 // e) * e


def prefetch_vgprs(e, layout, vec, w, nb, kc, th, tw, wave):
    """VGPRs of a chunk's register prefetch (ConvStemBwdCore.h issue / store): past ~96 the
    kernel spills."""
    ev = 16 // e
    wr, wc = th + w - 1, tw + w - 1
    threads = th * wave
    if layout == 1:
        ua, regs = wr * wc * (kc // ev), 4
    elif vec > 1:
        ua, regs = wr * kc * ((wc + 2 * ev - 2) // ev), 4
    else:
        ua, regs = wr * wc * kc, 1
    ub = nb * w * w * (kc // ev)
    return -(-ua // threads) * regs + -(-ub // threads) * 4


def lds_limit(atom):
    return 160 * 1024 if atom.startswith("mfma16x32") else 64 * 1024


def nbs(atom, s):
    """GEMM column counts: for 3 channels (RGB) and 4, and 1 (grey) on FMA, rounded to whole
    fragments on a matrix atom; distinct, at most 64."""
    dim = SHAPE[atom][0]
    cs = [3, 4]
    got = []
    for c in cs:
        n = c * s * s
        if atom != "fma":
            n = -(-n // dim) * dim
        if n <= 64 and n not in got:
            got.append(n)
    return got


def build():
    catalog = {}
    for dtype, atoms in FAMILIES.items():
        e = eb(dtype)
        ev = 16 // e
        cfgs = []
        # 2-D configs first (their order, and so their priorities, as before), then 3-D ones.
        plan = (
            [(atom, s, w, 0, 0) for atom in atoms for s, w in GEOMETRIES]
            + [(atom, s, w, ft, 0) for atom in atoms for s, w, ft in GEOMETRIES_3D]
            + [(atom, 1, w, 0, SPACE_TO_DEPTH) for atom in atoms for w in SD_WINDOWS]
        )
        for atom, s, w, ft, sd in plan:
            if sd and atom == "fma":
                continue
            dim, ka, wave = SHAPE[atom]
            # Wide windows stage more of B per chunk: smaller chunks.
            kcs = (
                [8, 16]
                if dtype == "FLOAT"
                else ([16, 32, 64] if w * w <= 4 else [16, 32])
            )
            tiles = (
                # 16 threads a row; 8 x 1 for the wide windows' 48 / 64 columns (11x11 s4),
                # whose B chunk leaves the larger tiles no room in 64 KiB or registers.
                [(8, 4), (16, 2), (16, 4), (32, 2)]
                + ([(8, 1)] if w * w > 4 and s > 2 else [])
                if atom == "fma"
                else [(2, 4), (4, 2), (4, 4), (8, 2)]
            )
            layouts = [(1, ev), (0, ev), (0, 1)]
            for nb in nbs(atom, sd or s):
                for layout, vec in layouts:
                    for kc in kcs:
                        if kc % ka or kc % ev:
                            continue
                        for th, fm in tiles:
                            if (
                                sd
                                and (kc, (th, fm)) not in SD_KC_TILES[(layout, vec > 1)]
                            ):
                                continue
                            if th * wave > 1024:
                                continue
                            if lds_bytes(e, w, w, nb, kc, th, fm * dim) > lds_limit(
                                atom
                            ):
                                continue
                            if (
                                prefetch_vgprs(
                                    e, layout, vec, w, nb, kc, th, fm * dim, wave
                                )
                                > 96
                            ):
                                continue
                            cfgs.append(
                                {
                                    "atom": atom,
                                    "atom_id": ATOMS.index(atom),
                                    "layout": layout,
                                    "vec": vec,
                                    "u": w,
                                    "v": w,
                                    "nb": nb,
                                    "kc": kc,
                                    "th": th,
                                    "fm": fm,
                                    "sh": s,
                                    "sw": s,
                                    "ft": ft,
                                    "sd": sd,
                                }
                            )
        catalog[dtype] = cfgs
    return catalog


def emit(catalog):
    kmd, umd, udd = U("kmd"), U("umd"), U("udd")
    write(
        "conv_stem_bwd.kmd.json",
        {
            "version": "1.0",
            "id": kmd,
            "name": "conv_stem_bwd config fields",
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
            "name": "kernel dtype, atom (device), layout, strides, window and columns match the graph",
            "scope": "kernel",
            "match_symbol": "hipkernel.conv_stem_bwd.kernel_match",
        },
    )
    write(
        "conv_stem_bwd.udd.json",
        {
            "version": "1.0",
            "id": udd,
            "name": "conv_stem_bwd dispatch",
            "dispatch_symbol": "hipkernel.conv_stem_bwd.dispatch",
        },
    )
    ued = U("ued")
    write(
        "conv_stem_bwd.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": "hipkernel:CONV_STEM_BWD",
            "graph_match": {"native": "hipkernel.conv_stem_bwd.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    total = 0
    tuples = set()
    for stale in out.glob("conv_stem_bwd*.kdp.json"):
        stale.unlink()
    kept = None if UNIVERSE else read_kept()
    if kept is not None:
        universe = {config_name(d, c) for d, cfgs in catalog.items() for c in cfgs}
        stale = sorted(kept - universe)
        if stale:
            sys.exit(
                f"{KEPT_FILE.name} lists configs the generator no longer makes: {stale[:5]}"
            )
    for dtype, cfgs in catalog.items():
        dt = DT[dtype]
        packs = {}
        if kept is not None:
            cfgs = [c for c in cfgs if config_name(dtype, c) in kept]
        for i, c in enumerate(cfgs):
            fam = family_of(c["atom"])
            fam = "fma906" if fam == "fma" and dtype != "FLOAT" else fam
            pack = packs.setdefault(fam, [])
            name = config_name(dtype, c)
            key = (dtype,) + tuple(c[f] for f in INTS)
            assert key not in tuples, name
            tuples.add(key)
            pack.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvStemBwd.cpp",
                        "entry_point": "ConvStemBwd",
                    },
                    "metadata": {"direction": "BWD", "dtype": dtype, **c},
                    "priority": 1000 - i,
                }
            )
        # One pack per dtype and atom family (the repository caps a file at 500 KB).
        for fam, pack in sorted(packs.items()):
            fname = f"conv_stem_bwd_{dt}_{fam}.kdp.json"
            write(
                fname,
                {
                    "version": "1.0",
                    "id": U(f"kdp.{dt}.{fam}"),
                    "name": f"hipkernel:conv_stem_bwd_{dt}_{fam}",
                    "arch": FMA16_ARCHS if fam == "fma906" else [],
                    "matchers": [umd],
                    "engine": ued,
                    "dispatch": udd,
                    "kernelDescriptors": pack,
                },
            )
            assert (out / fname).stat().st_size < 500 * 1000, f"{fname} over 500 KB"
            total += len(pack)
    return total


DT = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}


def config_name(dtype, c):
    lay = ["nchw", "nhwc"][c["layout"]]
    return (
        f"conv_stem_bwd.{DT[dtype]}.{c['atom']}.{lay}.v{c['vec']}.s{c['sh']}x{c['sw']}"
        f".w{c['u']}x{c['v']}.nb{c['nb']}.kc{c['kc']}.t{c['th']}f{c['fm']}"
        + (f".d{c['ft']}" if c["ft"] else "")
        + (f".sd{c['sd']}" if c["sd"] else "")
    )


def read_kept():
    """The shipped configs' names: one per line of conv_stem_bwd_kept.txt; # starts a comment."""
    names = set()
    for line in KEPT_FILE.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            assert name not in names, f"{name} listed twice"
            names.add(name)
    return names


def family_of(atom):
    if atom == "fma":
        return "fma"
    if atom.startswith("wmma"):
        return "rdna"
    if atom.startswith("mfma16x32"):
        return "gfx950"
    # gfx908's bf16 atoms apart: one bf16 CDNA file would pass the 500 KB cap.
    if atom.endswith("_908"):
        return "cdna908"
    return "cdna"


if __name__ == "__main__":
    cat = build()
    print(
        emit(cat),
        "kernel configs",
        "(the universe)" if UNIVERSE else f"(of {KEPT_FILE.name})",
    )
    for dtype, cfgs in cat.items():
        print("  universe", dtype, len(cfgs))
