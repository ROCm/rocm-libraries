# Generates the CONV_NARROW_FWD / CONV_NARROW_BWD / CONV_NARROW_WRW descriptors: convolutions
# with at most 8 channels on one side (descriptors/conv/narrow/kernels/): network output layers
# (K <= 8) forward and backward weights, stem input gradients (C <= 8) backward data and stem
# backward weights. The taps go on the GEMM's N axis (ConvNarrow.cpp, ConvNarrowWrw.cpp).
#
# Forward / backward data (ConvNarrow.cpp): a config fixes the direction, layout, atom (a
# matrix-core atom of ConvIgemmCore.h, or `valu`), the narrow channels it serves (kmin..ko), the
# filter (fd x fr x fs) and width stride (sw), depth streaming (ds), the tile (wr pixels a product
# row, rb in rows a step, nw waves a block, op output pixels a thread) and, on matrix cores, the
# reduction channels it holds (cm; it serves the graphs whose channels round up into cmin..cm), the
# A prefetch depth (pd) and the folded filter columns (sf; 0 in the catalog). Backward weights
# (ConvNarrowWrw.cpp): the side (k: an output layer; c: a stem), layout, atom, kmin..ko, the
# filter, in columns (wr) and rows (rb) a step, waves a block (nw) and row tiles a wave (mpw; valu:
# column groups). Every extent, channel count, stride (but the forward / backward-data width
# stride), padding and dilation is a runtime argument. Each config is admitted on correctness
# alone; autotune picks. `archs` is a config's ship list (architectures, `gfx11`, `gfx12` or
# `any`). The tiles are the fastest of a sweep over output-layer, stem and 3-D shapes on gfx950,
# gfx942, gfx908 and gfx1101.
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("5d0c7e4b-2a91-4f3e-8c6d-1b7a9e2f4c03")
U = lambda name: str(uuid.uuid5(NS, "narrow." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


# Atoms: name -> (id, element, k-step, lanes' k run, tile size, wave); valu: the device's lanes.
ATOMS = {
    "valu": (0, "any", 1, 1, 1, 0),
    "mfma16_f16": (1, "HALF", 16, 4, 16, 64),
    "mfma16_bf16": (2, "BFLOAT16", 16, 4, 16, 64),
    "mfma16_bf16_908": (3, "BFLOAT16", 8, 2, 16, 64),
    "mfma16_f32": (4, "FLOAT", 4, 1, 16, 64),
    "mfma16x32_f16": (5, "HALF", 32, 8, 16, 64),
    "mfma16x32_bf16": (6, "BFLOAT16", 32, 8, 16, 64),
    "wmma11_f16": (7, "HALF", 16, 16, 16, 32),
    "wmma11_bf16": (8, "BFLOAT16", 16, 16, 16, 32),
    "wmma12_f16": (9, "HALF", 16, 8, 16, 32),
    "wmma12_bf16": (10, "BFLOAT16", 16, 8, 16, 32),
}
ES = {"FLOAT": 4, "HALF": 2, "BFLOAT16": 2}
# Ship groups: (archs, LDS bytes, atoms). An atom ships to the architectures that run it best;
# valu everywhere (the floor: any channel count, gfx906, and fp32 on RDNA, which has no fp32
# matrix core).
SHIPS = [
    ("gfx950", 160 * 1024, ["mfma16x32_f16", "mfma16x32_bf16", "mfma16_f32"]),
    ("gfx908,gfx90a,gfx942", 64 * 1024, ["mfma16_f16", "mfma16_f32"]),
    ("gfx90a,gfx942", 64 * 1024, ["mfma16_bf16"]),
    ("gfx908", 64 * 1024, ["mfma16_bf16_908"]),
    ("gfx11", 64 * 1024, ["wmma11_f16", "wmma11_bf16"]),
    ("gfx12", 64 * 1024, ["wmma12_f16", "wmma12_bf16"]),
    ("any", 64 * 1024, ["valu"]),
]
# Catalog files: one a ship group (the repository caps a file at 500 KB).
PACK_OF = {
    "gfx950": "gfx950",
    "gfx908,gfx90a,gfx942": "cdna",
    "gfx90a,gfx942": "cdna",
    "gfx908": "gfx908",
    "gfx11": "gfx11",
    "gfx12": "gfx12",
    "any": "valu",
}
KOS = [1, 3, 4, 8]
# The narrow channels a config serves: those above the next smaller config's, up to its own.
KMIN = {ko: (KOS[i - 1] + 1 if i else 1) for i, ko in enumerate(KOS)}
LAYOUTS = ["nchw", "nhwc"]


def run_of(atom):
    _, el, ka, kpl, _, _ = ATOMS[atom]
    es = ES[el]
    return 1 if kpl * es >= 16 else 16 // (kpl * es)


# ---- forward / backward data
# Filter classes (fd, fr, fs, sw): forward, network output layers (VAE / super-resolution 3x3 and
# 3x3x3 heads, 1x1 heads, vocoder 1 x 7); backward data, stem input gradients (7x7 and 3x3 at
# stride 2, 3x7x7 video stems, stride-1 3x3 / 3x3x3 and 1x1).
FILTERS = {
    "FWD": [(3, 3, 3, 1), (1, 3, 3, 1), (1, 1, 1, 1), (1, 1, 7, 1)],
    "BWD": [
        (1, 7, 7, 2),
        (3, 7, 7, 2),
        (1, 3, 3, 2),
        (1, 3, 3, 1),
        (3, 3, 3, 1),
        (1, 1, 1, 1),
    ],
}
# Reduction-channel tiers of the matrix-core configs: a config holds cm channels and serves the
# graphs whose channels round up into (previous tier, cm]; a graph's channels pick one tier.
CM_TIERS = [32, 64, 128, 320, 512]
# Tiles (wr, rb, nw, op, pd, ds): pixels a product row, in rows a step, waves a block, output
# pixels a thread, A prefetch depth (matrix cores), depth streaming (unit depth stride; each list
# also has a ds 0 tile, which takes any depth stride). Per atom family: the tiles of single-plane
# filters (2-D graphs, 1x1x1) and of 3-D filters.
A_TILE, B_TILE, C_TILE = (64, 1, 4, 2, 3, 0), (32, 2, 4, 2, 1, 1), (64, 2, 8, 1, 2, 0)
TILES = {
    "mfma": ([A_TILE, B_TILE], [A_TILE, C_TILE, B_TILE]),
    "wmma": ([B_TILE, C_TILE], [C_TILE, A_TILE, B_TILE]),
    "valu": (
        # 32x2 (64 rows of products) also carries the 8-channel heads within 64 KiB.
        [(32, 4, 4, 2, 0, 0), (64, 2, 4, 2, 0, 0), (32, 2, 4, 2, 0, 0)],
        [
            (32, 4, 4, 2, 0, 0),
            (64, 2, 4, 2, 0, 0),
            (32, 2, 4, 2, 0, 1),
            (32, 2, 4, 2, 0, 0),
        ],
    ),
}
# A step's A fragments live in registers (CN_PD + 1 buffers): at most this many VGPRs.
A_VGPRS = 128


def fwd_plan(atom, dtype, tile, ko, filt, cm, lds_limit):
    """Whether a config compiles within the budgets on every architecture it ships to (the
    host's planFor arithmetic: products, zero run, filter in LDS or registers)."""
    wr, rb, nw, op, pd, ds = tile
    fd, fr, fs, _ = filt
    m = wr * rb
    ncol = (fd if ds else 1) * ko * fr * fs
    if atom == "valu":
        for wave in (32, 64):
            threads = nw * wave
            if m % wave or threads % m:
                return False
            groups = threads // m
            ncp = groups * -(-ncol // groups)
            if (2 * m * (ncp + 1) + ko * fr * fs) * 4 > lds_limit:
                return False
        return True
    _, el, ka, kpl, dim, wave = ATOMS[atom]
    es = ES[el]
    if m % (dim * nw):
        return False
    ncp = -(-ncol // dim) * dim
    frag = max(1, kpl * es // 4)
    ks = cm // ka
    if (pd + 1) * (m // dim // nw) * ks * frag > A_VGPRS:
        return False
    groups = 1 if ds else fd
    br = groups == 1 and ncp // dim * ks * frag <= 64
    lds = (2 * m * (ncp + 1) + ko * fr * fs) * 4 + (
        0 if br else groups * ncp * (cm + 16 // es) * es
    )
    return lds <= lds_limit


def fwd_configs(direction, dtype):
    for group, lds_limit, atoms in SHIPS:
        for atom in atoms:
            if ATOMS[atom][1] not in ("any", dtype):
                continue
            valu = atom == "valu"
            step = 1 if valu else ATOMS[atom][2] * run_of(atom)
            tiers = (
                [(0, 0)]
                if valu
                else [
                    (lo + 1, cm)
                    for lo, cm in zip([0] + CM_TIERS, CM_TIERS)
                    if cm % step == 0
                ]
            )
            for filt in FILTERS[direction]:
                for tile in TILES[atom[:4]][0 if filt[0] == 1 else 1]:
                    for ko in KOS:
                        for cmin, cm in tiers:
                            if not fwd_plan(atom, dtype, tile, ko, filt, cm, lds_limit):
                                continue
                            for layout in LAYOUTS:
                                yield group, atom, layout, ko, filt, tile, cmin, cm


# ---- backward weights
# Filter classes (fd, fr, fs) per side: output layers (k: 3x3x3, 3x3, 1x1x1 / 1x1, 1 x 7) and
# stems (c: 7x7, 3x7x7, 1x7x7, 3x3, 4x4, 3x3x3).
WRW_FILTERS = {
    "k": [(3, 3, 3), (1, 3, 3), (1, 1, 1), (1, 1, 7)],
    "c": [(1, 7, 7), (3, 7, 7), (1, 3, 3), (1, 4, 4), (3, 3, 3)],
}
# Tiles (wr, rb, nw, mpw); valu (wr, rb, nw, column groups).
WRW_TILES_MMA = [(32, 2, 8, 1), (64, 1, 4, 1), (32, 2, 4, 1), (32, 2, 4, 2)]
WRW_TILES_VALU = [(32, 2, 8, 2), (64, 2, 4, 2), (32, 2, 4, 4)]


def wrw_configs(dtype):
    for group, lds_limit, atoms in SHIPS:
        for atom in atoms:
            if ATOMS[atom][1] not in ("any", dtype):
                continue
            _, _, ka, kpl, dim, wave = ATOMS[atom]
            for side in ("k", "c"):
                for fd, fr, fs in WRW_FILTERS[side]:
                    for ko in KOS:
                        ncol = ko * fd * fr * fs
                        for layout in LAYOUTS:
                            for tile in (
                                WRW_TILES_VALU if atom == "valu" else WRW_TILES_MMA
                            ):
                                wr, rb, nw, mpw = tile
                                kr = wr * rb
                                if atom == "valu":
                                    if nw % mpw:
                                        continue
                                    ncp = mpw * -(-ncol // mpw)
                                    if ncp // mpw > 64:
                                        continue
                                    lds = ncp * (kr + 4) * 4
                                else:
                                    if kr % ka or wr % kpl:
                                        continue
                                    ncp = -(-ncol // dim) * dim
                                    lds = ncp * (kr + 16 // ES[dtype]) * ES[dtype]
                                # The narrow window a step reads (stems at stride 2).
                                xw = (
                                    (wr + fs) // 1 + 1
                                    if side == "k"
                                    else (wr - 1) * 2 + fs
                                )
                                lds += ko * fd * fr * rb * xw * ES[dtype]
                                if lds > lds_limit:
                                    continue
                                yield group, atom, side, layout, ko, (fd, fr, fs), tile


DTYPES = (("FLOAT", "f32"), ("HALF", "f16"), ("BFLOAT16", "bf16"))


def kmd(name, strings, ints):
    write(
        f"{name}.kmd.json",
        {
            "version": "1.0",
            "id": U(f"kmd.{name}"),
            "name": f"{name} config fields",
            "fields": [{"name": f, "type": "string"} for f in strings]
            + [{"name": f, "type": "int"} for f in ints],
        },
    )


def descriptor(name, source, entry, metadata, priority):
    return {
        "version": "1.0",
        "id": U(name),
        "name": name,
        "kernel_source": {
            "kind": "embedded_source",
            "source_file": source,
            "entry_point": entry,
        },
        "metadata": metadata,
        "priority": priority,
    }


INTS = [
    "atom_id",
    "kmin",
    "ko",
    "fd",
    "fr",
    "fs",
    "sw",
    "ds",
    "cmin",
    "cm",
    "pd",
    "sf",
    "wr",
    "rb",
    "nw",
    "op",
]
STRINGS = ["direction", "dtype", "layout", "atom", "archs"]
WRW_INTS = [
    "atom_id",
    "side_id",
    "kmin",
    "ko",
    "fd",
    "fr",
    "fs",
    "wr",
    "rb",
    "nw",
    "mpw",
]
WRW_STRINGS = ["side", "dtype", "layout", "atom", "archs"]

kmd("conv_narrow", STRINGS, INTS)
kmd("conv_narrow_wrw", WRW_STRINGS, WRW_INTS)
for name, symbol, what in (
    (
        "kernel_matches_graph",
        "hipkernel.conv_narrow.kernel_match",
        "kernel direction, dtype, layout, filter, width stride, narrow channels, atom (device, ship list) match the graph",
    ),
    (
        "kernel_matches_graph_wrw",
        "hipkernel.conv_narrow_wrw.kernel_match",
        "kernel side, dtype, layout, filter, narrow channels, atom (device, ship list) match the graph",
    ),
):
    write(
        f"{name}.umd.json",
        {
            "version": "1.0",
            "id": U(f"umd.{name}"),
            "name": what,
            "scope": "kernel",
            "match_symbol": symbol,
        },
    )
for name in ("conv_narrow", "conv_narrow_wrw"):
    write(
        f"{name}.udd.json",
        {
            "version": "1.0",
            "id": U(f"udd.{name}"),
            "name": f"{name} dispatch",
            "dispatch_symbol": f"hipkernel.{name}.dispatch",
        },
    )
for low, knobs, kmdName in (
    ("fwd", INTS, "conv_narrow"),
    ("bwd", INTS, "conv_narrow"),
    ("wrw", WRW_INTS, "conv_narrow_wrw"),
):
    write(
        f"conv_narrow_{low}.ued.json",
        {
            "version": "1.0",
            "id": U(f"ued.{low}"),
            "name": f"hipkernel:CONV_NARROW_{low.upper()}",
            "graph_match": {"native": f"hipkernel.conv_narrow_{low}.graph_match"},
            "metadata": U(f"kmd.{kmdName}"),
            "knobs": knobs,
            "behavior_notes": ["runtime_compilation"],
            # RFC 0025 numerical notes. No "nondeterministic": forward and backward data sum
            # every output element in one thread in a fixed order, backward weights add their
            # partials in split order; the family's device test relaunches each config.
            "numerical_notes": [],
        },
    )

total = 0
for low in ("fwd", "bwd", "wrw"):
    for dtype, dt in DTYPES:
        packs = {p: [] for p in dict.fromkeys(PACK_OF.values())}
        if low == "wrw":
            configs = list(wrw_configs(dtype))
        else:
            configs = list(fwd_configs(low.upper(), dtype))
        for i, cfg in enumerate(configs):
            if low == "wrw":
                group, atom, side, layout, ko, (fd, fr, fs), tile = cfg
                wr, rb, nw, mpw = tile
                name = (
                    f"conv_narrow.wrw.{dt}.{atom}.{PACK_OF[group]}.{side}.{layout}.k{ko}.f{fd}x{fr}x{fs}"
                    f".w{wr}r{rb}n{nw}m{mpw}"
                )
                meta = {
                    "side": side,
                    "dtype": dtype,
                    "layout": layout,
                    "atom": atom,
                    "archs": group,
                    "atom_id": ATOMS[atom][0],
                    "side_id": 1 if side == "k" else 0,
                    "kmin": KMIN[ko],
                    "ko": ko,
                    "fd": fd,
                    "fr": fr,
                    "fs": fs,
                    "wr": wr,
                    "rb": rb,
                    "nw": nw,
                    "mpw": mpw,
                }
                d = descriptor(
                    name, "kernels/ConvNarrowWrw.cpp", "ConvNarrowWrw", meta, 1000 - i
                )
            else:
                group, atom, layout, ko, (fd, fr, fs, sw), tile, cmin, cm = cfg
                wr, rb, nw, op, pd, ds = tile
                name = (
                    f"conv_narrow.{low}.{dt}.{atom}.{PACK_OF[group]}.{layout}.k{ko}.f{fd}x{fr}x{fs}s{sw}d{ds}"
                    f".w{wr}r{rb}n{nw}o{op}p{pd}c{cm}"
                )
                meta = {
                    "direction": low.upper(),
                    "dtype": dtype,
                    "layout": layout,
                    "atom": atom,
                    "archs": group,
                    "atom_id": ATOMS[atom][0],
                    "kmin": KMIN[ko],
                    "ko": ko,
                    "fd": fd,
                    "fr": fr,
                    "fs": fs,
                    "sw": sw,
                    "ds": ds,
                    "cmin": cmin,
                    "cm": cm,
                    "pd": pd,
                    "sf": 0,
                    "wr": wr,
                    "rb": rb,
                    "nw": nw,
                    "op": op,
                }
                d = descriptor(
                    name, "kernels/ConvNarrow.cpp", "ConvNarrow", meta, 1000 - i
                )
            packs[PACK_OF[group]].append(d)
        for pack, kernels in packs.items():
            if not kernels:
                continue
            fname = f"conv_narrow_{low}_{dt}_{pack}.kdp.json"
            write(
                fname,
                {
                    "version": "1.0",
                    "id": U(f"kdp.{low}.{dt}.{pack}"),
                    "name": f"hipkernel:conv_narrow_{low}_{dt}_{pack}",
                    "arch": [],
                    "matchers": [
                        U(
                            "umd.kernel_matches_graph_wrw"
                            if low == "wrw"
                            else "umd.kernel_matches_graph"
                        )
                    ],
                    "engine": U(f"ued.{low}"),
                    "dispatch": U(
                        "udd.conv_narrow_wrw" if low == "wrw" else "udd.conv_narrow"
                    ),
                    "kernelDescriptors": kernels,
                },
            )
            assert (out / fname).stat().st_size < 500 * 1000, f"{fname} over 500 KB"
            total += len(kernels)
            print(fname, len(kernels), "kernel configs")
print(total, "kernel configs")
