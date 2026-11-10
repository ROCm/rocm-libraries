# Generates the CONV_STEM_BWD descriptors (backward data for at most 4 input channels: stems).
# Each config is admitted on correctness alone (its atom on the device and taking the graph's
# dtype, its layout dy's, its strides and window the problem's, its GEMM columns at least the
# channels times the phases, its LDS within the device's); autotune (UHD later) ranks them.
#   python3 gen_conv_stem_bwd.py <descriptors/conv/stem_bwd>
#
# The window of one axis: offset floor((s - 1 + p) / s), extent offset - ceil((p - f + 1) / s) + 1
# (ConvStemBwdProblem.cpp windowOf). GEOMETRIES lists the (stride, extent) pairs the catalog
# covers, square: 7x7 s2 p3 -> (2, 4); 3x3 s2 p1 -> (2, 2); 6x6 s2 p2 / 5x5 s2 p2 -> (2, 3);
# 2x2 s2 -> (2, 1); 3x3 / 5x5 / 7x7 s1 "same" -> (1, 3 / 5 / 7); 1x1 -> (1, 1); 4x4 s4 patchify
# -> (4, 1); 11x11 s4 p2 -> (4, 4).
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
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
    "HALF": ["mfma16_f16", "mfma16x32_f16", "wmma11_f16", "wmma12_f16"],
    "BFLOAT16": [
        "mfma16_bf16",
        "mfma16_bf16_908",
        "mfma16x32_bf16",
        "wmma11_bf16",
        "wmma12_bf16",
    ],
    "FLOAT": ["mfma16_f32", "fma"],
}
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
INTS = ["atom_id", "layout", "vec", "u", "v", "nb", "kc", "th", "fm", "sh", "sw"]
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
        for atom in atoms:
            dim, ka, wave = SHAPE[atom]
            for s, w in GEOMETRIES:
                # Wide windows stage more of B per chunk: smaller chunks.
                kcs = (
                    [8, 16]
                    if dtype == "FLOAT"
                    else ([16, 32, 64] if w * w <= 4 else [16, 32])
                )
                tiles = (
                    [(8, 4), (16, 2), (16, 4), (32, 2)]
                    if atom == "fma"  # 16 threads a row
                    else [(2, 4), (4, 2), (4, 4), (8, 2)]
                )
                for nb in nbs(atom, s):
                    for layout, vec in [(1, ev), (0, ev), (0, 1)]:
                        for kc in kcs:
                            if kc % ka or kc % ev:
                                continue
                            for th, fm in tiles:
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
    for dtype, cfgs in catalog.items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        packs = {}
        for i, c in enumerate(cfgs):
            pack = packs.setdefault(family_of(c["atom"]), [])
            lay = ["nchw", "nhwc"][c["layout"]]
            name = (
                f"conv_stem_bwd.{dt}.{c['atom']}.{lay}.v{c['vec']}.s{c['sh']}x{c['sw']}"
                f".w{c['u']}x{c['v']}.nb{c['nb']}.kc{c['kc']}.t{c['th']}f{c['fm']}"
            )
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
                    "arch": [],
                    "matchers": [umd],
                    "engine": ued,
                    "dispatch": udd,
                    "kernelDescriptors": pack,
                },
            )
            assert (out / fname).stat().st_size < 500 * 1000, f"{fname} over 500 KB"
            total += len(pack)
    return total


def family_of(atom):
    if atom == "fma":
        return "fma"
    if atom.startswith("wmma"):
        return "rdna"
    if atom.startswith("mfma16x32"):
        return "gfx950"
    return "cdna"


if __name__ == "__main__":
    cat = build()
    print(emit(cat), "kernel configs")
    for dtype, cfgs in cat.items():
        print(" ", dtype, len(cfgs))
