# Generates the CONV_WRW descriptors (backward weights, K3). Each config is admitted on
# correctness alone (its atom on the device and taking the graph's dtype, its layout the graph's,
# its filter width / stride / dilation and, for the row rings, filter height equal to the
# problem's, its LDS within the device's); autotune (UHD later) ranks them. Priority orders the
# declared fallback.
#
# Every config-distinguishing field is an integer knob (hipDNN's candidate enumeration tells
# configs apart by their integer knobs only); the strings (direction, dtype, atom) are
# descriptive and checked by the kernel match (atom_id must name the same atom as atom).
#   python3 gen_conv_wrw.py <descriptors/conv/wrw>
#
# To extend the catalog, add rows to FAMILIES (which atoms, per dtype, with which tiles, loops
# and layouts) and FILTERS (filter-width / stride / dilation / row combos and the loop they run
# on). The LDS arithmetic is ConvWrwGeom.h's k3LdsBytes (ported below; keep it in step) and the
# load passes ConvWrwProblem.cpp's loadPassesOf; a config that does not fit is dropped.
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS_UUID = uuid.UUID("3b7e2d91-6c4a-4f08-9a1e-5d2c8b0f7a63")
U = lambda name: str(uuid.uuid5(NS_UUID, "conv_wrw." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


# atom_id order: the pack's conv_wrw::ATOMS. (DIM, KA, KPL) per atom; the archs that run it.
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
    "mfma16_f16": (16, 16, 4),
    "mfma16_bf16": (16, 16, 4),
    "mfma16_bf16_908": (16, 8, 2),
    "mfma16_f32": (16, 4, 1),
    "mfma32_f16": (32, 8, 4),
    "mfma32_bf16": (32, 8, 4),
    "mfma32_bf16_908": (32, 4, 2),
    "mfma32_f32": (32, 2, 1),
    "mfma16x32_f16": (16, 32, 8),
    "mfma16x32_bf16": (16, 32, 8),
    "mfma32x16_f16": (32, 16, 8),
    "mfma32x16_bf16": (32, 16, 8),
    "wmma11_f16": (16, 16, 16),
    "wmma11_bf16": (16, 16, 16),
    "wmma12_f16": (16, 16, 8),
    "wmma12_bf16": (16, 16, 8),
    "fma": (0, 0, 0),
}
CDNA = ["gfx908", "gfx90a", "gfx942", "gfx950"]
ARCHS = {
    "mfma16_f16": CDNA,
    "mfma32_f16": CDNA,
    "mfma16_f32": CDNA,
    "mfma32_f32": CDNA,
    "mfma16_bf16": CDNA[1:],
    "mfma32_bf16": CDNA[1:],
    "mfma16_bf16_908": ["gfx908"],
    "mfma32_bf16_908": ["gfx908"],
    "mfma16x32_f16": ["gfx950"],
    "mfma16x32_bf16": ["gfx950"],
    "mfma32x16_f16": ["gfx950"],
    "mfma32x16_bf16": ["gfx950"],
    "wmma11_f16": ["gfx11"],
    "wmma11_bf16": ["gfx11"],
    "wmma12_f16": ["gfx12"],
    "wmma12_bf16": ["gfx12"],
    "fma": CDNA + ["gfx11", "gfx12"],
}
INTS = [
    "atom_id",
    "loop",
    "layout",
    "bk",
    "bc",
    "bq",
    "fr",
    "s",
    "sw",
    "dw",
    "wk",
    "wc",
    "tk",
    "tc",
    "vec",
    "xcd",
    "align",
    "ns",
    "bpu",
    "wq",
    "dws",
    "pb",
    "math",
]
STRINGS = ["direction", "dtype", "atom"]
FIELDS = STRINGS + INTS
CF, CL, TR = 0, 1, 2


def eb(dtype):
    return 4 if dtype == "FLOAT" else 2


def wave(atom):
    return 32 if atom.startswith("wmma") else 64


def du(a, b):
    return -(-a // b)


def tr_pitch(n):  # ConvWrwGeom.h k3TrPitch
    p = n + 8
    while p % 128 in (0, 8, 32, 96):
        p += 8
    return p


def rot_bit(r, i):  # ConvWrwGeom.h k3dBit / k3dRot / k3dConflictFree / k3dRotation
    return [(3, 1, 0), (1, 3, 0), (0, 1, 3)][r][i]


def rot(r, ch, x):
    g = (
        ((x >> rot_bit(r, 0)) & 1)
        | (((x >> rot_bit(r, 1)) & 1) << 1)
        | (((x >> rot_bit(r, 2)) & 1) << 2)
    )
    return g & (ch // 16 - 1)


def conflict_free(r, ch, shifts):
    c4 = ch // 4
    for cb in range(0, c4, 4):
        for sh in range(shifts):
            for ph in range(2):
                for rd in range(2):
                    seen = set()
                    for l in range(ph * 32, ph * 32 + 32):
                        x = (l // 16) * 8 + rd * 4 + (l % 16) // 4 + sh
                        off = (x * c4 + ((cb + l % 4) ^ (4 * rot(r, ch, x)))) % 32
                        if off in seen:
                            return False
                        seen.add(off)
    return True


def rotation(ch, shifts):
    if ch < 32 or ch > 512 or ch & (ch - 1):
        return -1
    for r in range(3):
        if conflict_free(r, ch, shifts):
            return r
    return -1


def dx_cols(bc, s, dw):  # k3dXCols
    return du((32 + (s - 1) * dw) * bc, 512) * 512 // bc


def d_loads(c):  # k3dLoads
    nw = 8 // c["wq"]
    return du(dx_cols(c["bc"], c["s"], c["dw"]) * c["bc"] // 512, nw) + du(
        32 * c["bk"] // 512, nw
    )


def lds_bytes(c, e):  # ConvWrwGeom.h k3LdsBytes
    loop, layout, bk, bc, bq, fr, s, sw, dw = (
        c[k] for k in ("loop", "layout", "bk", "bc", "bq", "fr", "s", "sw", "dw")
    )
    vec, align, ns = c["vec"], c["align"], c["ns"]
    if loop == 8:  # generic taps: a bk-row dy chunk and fr bc-row x tiles, two buffers
        if c["tk"]:  # the FMA form: [position][channel] rows of bk / bc + 16 bytes
            u = 16 // e
            return 2 * (bq * (bk + u) + (bq + dw if dw > 0 else fr * bq) * (bc + u)) * e
        if dw > 0:  # one shared window of bq + dw positions per x row (k3WinPitch)
            unit = 16 // e
            pitch = bq + dw + unit
            pitch += unit if (pitch // unit) % 2 == 0 else 0
            return 2 * (bk * (bq + unit) + bc * pitch) * e
        return 2 * (bk + fr * bc) * (bq + 16 // e) * e
    if loop == 5:
        wq, fc, fk = c["wq"], bc // (16 * c["wc"]), bk // (16 * c["wk"])
        slot = wq * (dx_cols(bc, s, dw) * bc + 32 * bk) + 512
        red = (wq - 1) * (8 // wq) * s * fc * fk * 64 * 16 if wq > 1 else 0
        return ns * slot * e + red
    if loop == 4:
        win = (bq - 1) * sw + (s - 1) * dw + 1
        nbv = du(win, vec) + (1 if vec > 1 else 0)
        return 2 * (bq * (bk + 4) + fr * nbv * vec * (bc + 4)) * e
    if loop == 3:
        win = bq + (s - 1) * dw
        nbv = du(win, vec) + 1
        return ((fr + 1) * bq * (bk + 4) + 2 * nbv * vec * (bc + 4)) * e
    waves = c["wk"] * c["wc"]
    if loop == 2 and layout == TR:
        na, nb = tr_pitch(bk) // 8, tr_pitch(bc) // 8
        win = bq + (s - 1) * dw
        return (
            ns
            * (du(du(bq * na, 64), waves) + du(du(win * nb, 64), waves))
            * waves
            * 64
            * 8
            * e
        )
    if loop == 2:
        v = 16 // e
        nqa = bq // v + 1
        win = bq + (s - 1) * dw
        nbv = du(win + 2, v) | 1
        return (
            ns
            * (du(du(bk * nqa, 64), waves) + du(du(bc * nbv, 64), waves))
            * waves
            * 64
            * v
            * e
        )
    rows = fr if loop in (0, 6, 7) else 1
    layout = (
        CF if loop == 6 else layout
    )  # small-C rows stage channels-last into the CF tiles
    tr = layout == TR
    kv = 4 // e if layout == CL else 1
    win = (bq - 1) * sw + (s - 1) * dw + 1
    nbv = du(win + (vec - 1 if align else 0), vec)
    ph_raw = nbv * vec // sw if layout == CF else du(win, sw)
    ph = du(ph_raw, kv) * kv
    skew = 16 // e
    lda = tr_pitch(bk) if tr else bq + skew
    ldb = tr_pitch(bc) if tr else du(ph + 2, skew) * skew + skew
    a = (bq if tr else bk) * lda + (vec if align else 0)
    b = rows * sw * (ph if tr else bc) * ldb
    return 2 * (a + b) * e


def passes(c, e):  # ConvWrwProblem.cpp loadPassesOf
    vec = c["vec"]
    if c["loop"] == 8:  # k3WrwGeneric's loads a thread holds
        threads = (
            (c["bk"] // c["tk"]) * (c["bc"] // c["tc"])
            if c["tk"]
            else c["wk"] * c["wc"] * wave(c["atom"])
        )
        if c["layout"] == CF:
            rows = threads // c["bq"]
            return du(c["bk"], rows) + du(c["fr"] * c["bc"], rows)
        kw = 4 // e
        words = du(c["bq"], kw)
        x_units = (
            du(c["bc"], vec) * du(c["bq"] + c["dw"], kw)
            if c["dw"] > 0
            else c["fr"] * du(c["bc"], vec) * words
        )
        return (du(du(c["bk"], vec) * words, threads) + du(x_units, threads)) * kw
    if c["loop"] == 5:
        return d_loads(c)
    if c["loop"] == 4:
        threads = (c["bk"] // c["tk"]) * (c["bc"] // c["tc"])
        win = (c["bq"] - 1) * c["sw"] + (c["s"] - 1) * c["dw"] + 1
        nbv = du(win, vec) + (1 if vec > 1 else 0)
        cf = c["layout"] == CF
        au = (c["bk"] // 4) * (c["bq"] // vec) if cf else c["bq"] * (c["bk"] // vec)
        bu = (
            c["fr"] * (c["bc"] // 4) * nbv
            if cf
            else c["fr"] * nbv * vec * (c["bc"] // vec)
        )
        return max(du(au, threads), du(bu, threads))
    if c["loop"] == 3:
        threads = (c["bk"] // c["tk"]) * (c["bc"] // c["tc"])
        nbv = du(c["bq"] + (c["s"] - 1) * c["dw"], vec) + 1
        cf = c["layout"] == CF
        au = (c["bk"] // 4) * (c["bq"] // vec) if cf else c["bq"] * (c["bk"] // vec)
        bu = (c["bc"] // 4) * nbv if cf else nbv * vec * (c["bc"] // vec)
        return max(du(au, threads), du(bu, threads))
    waves = c["wk"] * c["wc"]
    if c["loop"] == 2:
        win = c["bq"] + (c["s"] - 1) * c["dw"]
        if c["layout"] == TR:
            ag, bg = du(c["bq"] * (tr_pitch(c["bk"]) // 8), 64), du(
                win * (tr_pitch(c["bc"]) // 8), 64
            )
        else:
            v = 16 // e
            ag, bg = du(c["bk"] * (c["bq"] // v + 1), 64), du(
                c["bc"] * (du(win + 2, v) | 1), 64
            )
        return max(du(ag, waves), du(bg, waves))
    threads = waves * wave(c["atom"])
    if c["loop"] == 7:  # K3PackTiles: dy and x one element a unit
        win = (c["bq"] - 1) * c["sw"] + (c["s"] - 1) * c["dw"] + 1
        span = du(win, vec) * vec
        return max(
            du(c["bk"] * c["bq"], threads), du(c["fr"] * c["bc"] * span, threads)
        )
    if c["loop"] == 6 and c["layout"] == CL:  # K3SmallClTiles: x one element a unit
        kv = 4 // e
        win = (c["bq"] - 1) * c["sw"] + (c["s"] - 1) * c["dw"] + 1
        ph = du(win, vec) * vec // c["sw"]
        au = (c["bq"] // kv) * (c["bk"] // vec)
        bu = c["fr"] * c["sw"] * ph * c["bc"]
        return max(du(au, threads), du(bu, threads))
    cf = c["layout"] == CF
    kv = 4 // e if c["layout"] == CL else 1
    rows = c["fr"] if c["loop"] in (0, 6) else 1
    win = (c["bq"] - 1) * c["sw"] + (c["s"] - 1) * c["dw"] + 1
    nbv = du(win + (vec - 1 if c["align"] else 0), vec)
    ph = du(nbv * vec // c["sw"] if cf else du(win, c["sw"]), kv) * kv
    au = c["bk"] * (c["bq"] // vec) if cf else (c["bq"] // kv) * (c["bk"] // vec)
    bu = rows * c["bc"] * nbv if cf else rows * c["sw"] * (ph // kv) * (c["bc"] // vec)
    return max(du(au, threads), du(bu, threads))


def archs_of(c):
    return ["gfx950"] if c["loop"] in (2, 5) or c["layout"] == TR else ARCHS[c["atom"]]


def lds_limit(c):
    return 160 * 1024 if archs_of(c) == ["gfx950"] else 64 * 1024


def valid(c, dtype, fma16=False):
    """Whether the pack admits config c for dtype on some architecture. fma16: the FMA loops on
    16-bit graphs, which only devices without a matrix core for the type take (gfx906).
    """
    e = eb(dtype)
    fma = c["loop"] in (3, 4) or (c["loop"] == 8 and c["atom"] == "fma")
    ok = (c["vec"] * e == 16 or (c["loop"] == 4 and c["vec"] == 1)) and (
        c["atom"] == "fma"
    ) == fma
    if fma:
        threads = (c["bk"] // c["tk"]) * (c["bc"] // c["tc"])
        ok = (
            ok
            and (dtype == "FLOAT" or fma16)
            and c["wk"] == c["wc"] == 0
            and c["tk"] % 4 == 0
            and c["tc"] % 4 == 0
        )
        ok = (
            ok
            and c["bk"] % c["tk"] == 0
            and c["bc"] % c["tc"] == 0
            and c["bq"] % c["vec"] == 0
            and c["bk"] % c["vec"] == 0
            and c["bc"] % c["vec"] == 0
        )
        ok = (
            ok
            and (c["sw"] == 1 or c["loop"] in (4, 8))
            and c["layout"] in (CF, CL)
            and c["xcd"] == 0
        )
    else:
        dim, ka, kpl = SHAPE[c["atom"]]
        threads = c["wk"] * c["wc"] * wave(c["atom"])
        if c["loop"] == 6:  # small-C rows: (channel, tap) N axis, waves on k only
            ok = (
                ok
                and c["bk"] % (c["wk"] * dim) == 0
                and c["wc"] == 1
                and c["bc"] <= 16
                and c["bq"] % ka == 0
            )
            ok = ok and c["layout"] in (CF, CL)
            # The CF tiles' geometry.
            ok = ok and c["bq"] % c["vec"] == 0 and c["vec"] % c["sw"] == 0
        else:
            ok = (
                ok
                and c["bk"] % (c["wk"] * dim) == 0
                and c["bc"] % (c["wc"] * dim) == 0
                and c["bq"] % ka == 0
            )
        if c["layout"] == TR:
            ok = ok and kpl == 8 and e == 2 and "gfx950" in ARCHS[c["atom"]]
    if c["loop"] == 8:
        # Generic taps: runtime geometry, a position per thread, whole channel rows a pass.
        rows = threads // c["bq"] if threads % c["bq"] == 0 else 0
        ok = (
            ok
            and c["layout"] in (CF, CL)
            and c["s"] == c["sw"] == 0
            and (c["dw"] == 0 or (c["layout"] == CL and c["dw"] % 8 == 0))
            and c["fr"] >= 1
            and rows > 0
            and c["bk"] % rows == 0
            and c["bc"] % rows == 0
            and c["align"] == c["ns"] == 0
        )
    elif c["layout"] == CF:
        ok = ok and c["bq"] % c["vec"] == 0 and (fma or c["vec"] % c["sw"] == 0)
    if c["loop"] == 5:
        # The direct loop: 8 waves, 16 x 16 x 32 atoms, rows kept channels-last, a rotation for
        # both rows, the loads in flight within vmcnt, the accumulators within the register
        # budget (taps x fc x fk x 4 at most 160 VGPRs).
        fc, fk = c["bc"] // (16 * c["wc"]), c["bk"] // (16 * c["wk"])
        threads = 512
        ok = (
            ok
            and c["wq"] in (1, 2, 4)
            and c["wk"] * c["wc"] * c["wq"] == 8
            and c["layout"] == TR
        )
        ok = (
            ok
            and SHAPE[c["atom"]][:2] == (16, 32)
            and c["bq"] == 32
            and c["sw"] == 1
            and 2 <= c["ns"] <= 5
        )
        ok = (
            ok
            and rotation(c["bc"], (c["s"] - 1) * c["dw"] + 1) >= 0
            and rotation(c["bk"], 1) >= 0
        )
        ok = (
            ok
            and d_loads(c) <= 63
            and (c["ns"] - 2) * d_loads(c) <= 63
            and c["bpu"] == 1
        )
        ok = ok and c["fr"] * c["s"] * fc * fk * 4 <= 160
    elif c["wq"]:
        ok = False
    if c["loop"] == 2:
        ok = (
            ok
            and c["layout"] in (CF, TR)
            and c["sw"] == 1
            and 2 <= c["ns"] <= 4
            and "gfx950" in ARCHS[c["atom"]]
        )
    if c["align"] == 2:  # dy units at any element: the ring, channels-last transposed
        ok = ok and c["loop"] == 1 and c["layout"] == CL
    elif c["align"]:
        ok = ok and c["loop"] == 1 and c["layout"] == CF and c["sw"] == 1
    if c["xcd"]:  # admitted on gfx942 / gfx950 only (one XCD elsewhere)
        ok = ok and bool({"gfx942", "gfx950"} & set(archs_of(c)))
    if c["loop"] == 7:  # packed rows: channels-first, a dy position per thread
        ok = ok and c["layout"] == CF and threads % c["bq"] == 0 and c["ns"] == 0
    return (
        ok
        and 1 <= threads <= 1024
        and passes(c, e) <= (64 if c["loop"] in (7, 8) else 32)
        and lds_bytes(c, e) <= lds_limit(c)
    )


# Tiles (bk, bc, wk, wc, bq); FMA tiles (bk, bc, tk, tc, bq). CDNA 32 x 32 atoms: one fragment per
# wave; RDNA (wave32, 16 x 16 fragments) and the 16 x 16 fp32 MFMA: measured tiles.
CDNA16 = [
    (128, 64, 4, 2, 64),
    (64, 64, 2, 2, 32),
    (64, 128, 2, 4, 64),
    (32, 64, 1, 2, 32),
]
CDNA32 = [
    (128, 64, 4, 2, 16),
    (64, 64, 2, 2, 16),
    (64, 128, 2, 4, 32),
    (32, 64, 1, 2, 32),
]
MFMA16F32 = [(32, 64, 2, 4, 32), (64, 128, 4, 2, 32), (32, 32, 2, 2, 32)]
RDNA16 = [
    (64, 32, 4, 2, 32),
    (64, 64, 4, 2, 32),
    (128, 32, 4, 2, 32),
    (64, 64, 4, 2, 64),
]
# RDNA 1 x N rows (hifigan-like long 1-D maps), measured on gfx1201: a one-fragment-per-wave tile
# for narrow channel counts and long filters, a tall one for 128+ channels.
RDNA_1D = [(32, 32, 2, 2, 64), (128, 32, 4, 2, 64)]
ONE_D = {
    "1x3",
    "1x3d3",
    "1x3d5",
    "1x4",
    "1x7",
    "1x7d3",
    "1x7d5",
    "1x11",
    "1x11d3",
    "1x11d5",
}
# Tiles a family's list lacks for some rows, measured on gfx942 (atom prefix, row tags) -> tiles:
# the 128 x 64 ring at 32 positions for large width dilations (64 overflows the window's LDS),
# a 16-position tile for stride-8 1 x 16, the 32 x 32 16 x 16 fp32 tile for 3-D stride 2.
EXTRA_TILES = {
    ("mfma32", ("3x3d12", "3x3d24", "3x3d36")): [(128, 64, 4, 2, 32)],
    # fp32's 32-position windows at dilation 24 / 36 overflow 64 KiB: 16 positions.
    ("mfma32_f32", ("3x3d12", "3x3d24", "3x3d36")): [(64, 64, 2, 2, 16)],
    ("mfma32_", ("1x16s8",)): [(64, 32, 2, 1, 16)],
    ("mfma16_f32", ("3x3s2r1", "2x2s2r1")): [(32, 32, 2, 2, 32)],
    # RDNA 3x3 (deeplab dilations, unet2d), measured on gfx1100: 32-position tiles, up to 128 x 64.
    (
        "wmma",
        (
            "3x3d4",
            "3x3d6",
            "3x3d12",
            "3x3d24",
            "3x3d36",
            "3x3r1",
            "3x3d4r1",
            "3x3d6r1",
            "3x3d12r1",
            "3x3d24r1",
            "3x3d36r1",
        ),
    ): [(64, 32, 4, 2, 32), (64, 64, 4, 2, 32), (128, 64, 4, 2, 32)],
    # Measured on gfx1201 (fp16, n128 / n16): inception 5x5 on the ring (gfx12 only: on gfx11
    # the output rows' 64 x 64 tile is 3.4x faster), deeplab 1x1 on the rows (gfx11 too:
    # 1.3-2.2x over the ring's tiles on gfx1101).
    ("wmma12", ("5x5",)): [(64, 32, 4, 2, 32)],
    ("wmma", ("1x1r0",)): [(128, 64, 4, 2, 32)],
    # Packed rows, measured (fp16, n128): gfx950 resnet50 c512 7^2 / 14^2 s2, inception c288
    # 35^2 s2; gfx1201 the same and resnet50 c256 14^2 (its 128 x 32 tile at 64 positions ran
    # within 1% of the 64 x 64 one: not shipped); gfx11 (gfx1101) the WMMA tiles again, 2.4x.
    ("mfma32x16", ("3x3pk",)): [(128, 64, 4, 2, 64), (64, 128, 2, 4, 64)],
    ("mfma32x16", ("3x3s2pk",)): [(128, 64, 4, 2, 64), (128, 64, 4, 2, 32)],
    ("wmma", ("3x3pk",)): [(64, 64, 4, 2, 64), (128, 32, 4, 2, 32)],
    # 32 x 32 rings, measured on resnext50 c1024 g32 14^2 (fp16 n128): gfx950 294 against
    # 361 us, gfx1201 953 against 1052.
    ("mfma32x16", ("3x3g32",)): [(32, 32, 1, 1, 32)],
    ("wmma12", ("3x3g32",)): [(32, 32, 2, 2, 32)],
    # Channels-last 1x1 on the output rows, measured (bf16) on gfx1201 and gfx1101: the wide
    # tile for most maps, the 64 x 128 one for sdxl c960 k320 192^2.
    ("wmma", ("1x1clr0",)): [(128, 128, 4, 2, 32), (64, 128, 2, 4, 32)],
    # 5x5x5 3-D (N1 c16-c64, 128^3-132^3), measured on gfx950 and gfx1201 (fp32 / bf16).
    ("mfma32x16", ("5x5s2r1",)): [(64, 64, 2, 2, 32)],
    ("mfma32_f16", ("5x5s2r1",)): [(64, 64, 2, 2, 32)],
    ("mfma32_bf16", ("5x5s2r1",)): [(64, 64, 2, 2, 32)],
    ("mfma32_f32", ("5x5s2r1",)): [(64, 64, 2, 2, 16)],
    ("mfma16_f32", ("5x5s2r1", "5x5f")): [(32, 32, 2, 2, 32)],
    ("wmma", ("5x5s2r1",)): [(64, 32, 4, 2, 32)],
    ("fma", ("5x5s2r1",)): [(64, 64, 4, 4, 32)],
    ("fma", ("5x5rf",)): [(64, 64, 4, 4, 32), (32, 64, 4, 4, 32)],
    # gfx942 and gfx90a (the 32 x 32 atom): the fastest tile per stride on both.
    ("mfma32_", ("3x3pk",)): [(128, 64, 4, 2, 64)],
    ("mfma32_", ("3x3s2pk",)): [(64, 64, 2, 2, 64)],
    ("wmma", ("3x3s2pk",)): [(64, 32, 4, 2, 32), (64, 64, 4, 2, 32)],
}
FMA32 = [(32, 64, 4, 4, 32), (64, 64, 4, 4, 32)]
# FMA tiles per filter row where FMA32's 32-position steps overflow the LDS (a stride-8 window
# of 32 outputs spans 264 inputs): 8 positions per step.
FMA_TILES = {"1x16s8": [(32, 64, 4, 4, 8), (64, 64, 4, 4, 8)]}
BIG1x1 = [
    (128, 128, 2, 2, 64),
    (128, 256, 2, 4, 64),
]  # gfx950 1x1, channels-last kept (layout 2)

# Filter combos: (tag, s, sw, dw, fr, loop, tiles (indices into the family's list; None: all),
# CF aligns, layouts allowed (None: the family's)). loop 1 (row ring) for unit height stride with fr the filter height (the FMA family
# runs these on loop 3); loop 0 (output rows) for stride 2 with fr filter rows per block. 3x3 also
# serves 3x3x3 (the depth taps are grid rows); the deeplab rows use dilation 12 / 24 / 36 on both
# axes.
FILTERS = [
    ("3x3", 3, 1, 1, 3, 1, None, [0, 1], None),
    ("3x3d2", 3, 1, 2, 3, 1, [0, 1], [0], None),
    ("1x1", 1, 1, 1, 1, 1, [0, 1], [0, 1], None),
    ("1x3", 3, 1, 1, 1, 1, [0, -1], [0, 1], None),
    ("1x3d5", 3, 1, 5, 1, 1, [0, -1], [0, 1], [CF]),
    # HiFi-GAN's dilation-3 1x3 (the d5 row's tiles).
    ("1x3d3", 3, 1, 3, 1, 1, [0, -1], [0, 1], [CF]),
    ("1x4", 4, 1, 1, 1, 1, [0, -1], [0], None),
    ("1x7", 7, 1, 1, 1, 1, [-1], [0], None),
    ("1x7d3", 7, 1, 3, 1, 1, [-1], [0], [CF]),
    ("1x7d5", 7, 1, 5, 1, 1, [-1], [0], [CF]),
    ("1x11", 11, 1, 1, 1, 1, [-1], [0], None),
    ("1x11d3", 11, 1, 3, 1, 1, [-1], [0], [CF]),
    ("1x11d5", 11, 1, 5, 1, 1, [-1], [0], [CF]),
    ("3x3d4", 3, 1, 4, 3, 1, [0, -1], [0], [CF]),
    # SSD300's dilation-6 3x3 (the d4 row's tiles).
    ("3x3d6", 3, 1, 6, 3, 1, [0, -1], [0], [CF]),
    ("3x3d12", 3, 1, 12, 3, 1, [0, -1], [0], None),
    ("3x3d24", 3, 1, 24, 3, 1, [0, -1], [0], [CF]),
    ("3x3d36", 3, 1, 36, 3, 1, [0, -1], [0], [CF]),
    ("3x3s2", 3, 2, 1, 3, 0, [0, 1], [0], None),
    ("3x3s2r1", 3, 2, 1, 1, 0, [0, 1], [0], None),
    ("2x2s2", 2, 2, 1, 2, 0, [0, 1], [0], None),
    # Stride-1 3x3 on the output rows too, plain and dilated (RDNA: deeplab 65^2 and unet2d run
    # fastest there; gfx1100).
    ("3x3r1", 3, 1, 1, 1, 0, [0, 1], [0], None),
    ("3x3d4r1", 3, 1, 4, 1, 0, [0, 1], [0], [CF]),
    ("3x3d6r1", 3, 1, 6, 1, 0, [0, 1], [0], [CF]),
    ("3x3d12r1", 3, 1, 12, 1, 0, [0, 1], [0], None),
    ("3x3d24r1", 3, 1, 24, 1, 0, [0, 1], [0], [CF]),
    ("3x3d36r1", 3, 1, 36, 1, 0, [0, 1], [0], [CF]),
    # 7x7 stride-2 stems (channels-first; few input channels): one filter row per block.
    ("7x7s2r1", 7, 2, 1, 1, 0, [0, 1], [0], [CF]),
    # Strided 1-D downsampling (MelGAN / HiFi-GAN discriminators): 1 x 4 at stride 2, 1 x 16 at 8.
    ("1x4s2", 4, 2, 1, 1, 0, [0, 1, -1], [0], None),
    ("1x16s8", 16, 8, 1, 1, 0, [0, 1, -1], [0], None),
    ("2x2s2r1", 2, 2, 1, 1, 0, [0, 1], [0], None),
    # Stride-2 rows on the matrix ring over height phases (ceil(fr / 2) rows per phase).
    ("3x3s2ph", 3, 2, 1, 2, 1, [0, 1], [0], None),
    ("2x2s2ph", 2, 2, 1, 1, 1, [0, 1], [0], None),
    ("1x1s2", 1, 2, 1, 1, 0, [0, 1], [0], None),
    # Inception's asymmetric and 5x5 rows (channels-first), from the tiles measured best on
    # gfx950 and gfx1201 (n128 fp16): 7x1 and 3x1 on the ring with the 64 x 64 tile, 5x5 on the
    # output rows with the 64 x 64 tile (gfx950) and on the ring with RDNA's 64 x 32 tile
    # (EXTRA_TILES, gfx1201).
    ("7x1", 1, 1, 1, 7, 1, [1], [0, 1], [CF]),
    ("3x1", 1, 1, 1, 3, 1, [1], [0, 1], [CF]),
    ("5x5r1", 5, 1, 1, 1, 0, [1], [0], [CF]),
    ("5x5", 5, 1, 1, 5, 1, [], [0], [CF]),
    # Pointwise on RDNA's output rows (EXTRA_TILES): deeplab's 1x1 at 128^2 runs 1.5x faster
    # there than on the ring's tiles (gfx1201).
    ("1x1r0", 1, 1, 1, 1, 0, [], [0], [CF]),
    # Packed output rows (loop 7) for narrow outputs (ResNet / Inception 7^2, 14^2 s2, 35^2 s2
    # layers), on the tiles measured best on gfx950 and gfx1201 (EXTRA_TILES).
    ("3x3pk", 3, 1, 1, 1, 7, [], [0], [CF]),
    ("3x3s2pk", 3, 2, 1, 1, 7, [], [0], [CF]),
    # 32 x 32 rings for 32-channel groups (ResNeXt c1024 g32 14^2; EXTRA_TILES).
    ("3x3g32", 3, 1, 1, 3, 1, [], [0], [CF]),
    # Channels-last pointwise on RDNA's output rows (diffusion VAE / UNet 1x1; EXTRA_TILES).
    ("1x1clr0", 1, 1, 1, 1, 0, [], [0], [CL]),
    # 5-wide filters (3-D 5x5x5 volumes, AIPYTORCH-1356; EXTRA_TILES): stride 2 on the output
    # rows (every dtype), fp32 on the 16 x 16 fp32 ring (5x5f) and the FMA rows (5x5rf).
    ("5x5s2r1", 5, 2, 1, 1, 0, [], [0], [CF]),
    ("5x5f", 5, 1, 1, 5, 1, [], [0], [CF]),
    ("5x5rf", 5, 1, 1, 1, 0, [], [0], [CF]),
]
# fp32-only rows; and the rows with no gfx906 (FMA) configs.
FLOAT_ONLY = {"5x5f", "5x5rf"}
NO_FMA906 = FLOAT_ONLY | {"5x5s2r1"}
# Rows measured on the 16-bit matrix atoms only: no fp32 or FMA (gfx906) configs.
MATRIX16_ONLY = {
    "7x1",
    "3x1",
    "5x5r1",
    "5x5",
    "1x1r0",
    "3x3pk",
    "3x3s2pk",
    "3x3g32",
    "1x1clr0",
}
BASE = {
    "3x3",
    "1x1",
    "3x3s2",
    "3x3s2r1",
    "1x1s2",
}  # what the gfx908-only bf16 atom ships
LOOP2 = {
    "3x3",
    "3x3d2",
    "1x1",
    "3x3d12",
    "3x3d24",
    "3x3d36",
}  # the DMA ring's filters (gfx950)

# Families: (atoms per dtype, tiles, xcd, layouts for loops 0/1, loop-2 layouts, filters or None
# for all). An atom several archs run ships with xcd 0 (gfx908 and gfx90a have one XCD); gfx950
# and gfx942/gfx950-only families with xcd 1.
FAMILIES = [
    (
        {"HALF": "mfma32x16_f16", "BFLOAT16": "mfma32x16_bf16"},
        CDNA16,
        1,
        [CF, CL, TR],
        [CF, TR],
        None,
    ),
    ({"HALF": "mfma32_f16", "BFLOAT16": "mfma32_bf16"}, CDNA16, 0, [CF, CL], [], None),
    ({"FLOAT": "mfma32_f32"}, CDNA32, 0, [CF, CL], [], None),
    ({"FLOAT": "mfma32_f32"}, CDNA32[:2], 1, [], [CF], LOOP2),
    (
        {"FLOAT": "mfma16_f32"},
        MFMA16F32,
        1,
        [CF, CL],
        [],
        {"3x3", "3x3d2", "1x1", "1x7", "1x11", "3x3s2r1", "2x2s2r1", "5x5s2r1", "5x5f"},
    ),
    ({"BFLOAT16": "mfma32_bf16_908"}, CDNA16, 0, [CF, CL], [], BASE),
    ({"HALF": "wmma11_f16", "BFLOAT16": "wmma11_bf16"}, RDNA16, 0, [CF, CL], [], None),
    ({"HALF": "wmma12_f16", "BFLOAT16": "wmma12_bf16"}, RDNA16, 0, [CF, CL], [], None),
    ({"FLOAT": "fma"}, FMA32, 0, [CF, CL], [], None),
]
LOOP2_NS = 2

CATALOG = {"FLOAT": [], "HALF": [], "BFLOAT16": []}


def C(atom, loop, layout, tile, s, sw, dw, fr, vec, xcd, align, ns):
    if atom == "fma":
        bk, bc, tk, tc, bq = tile
        wk = wc = 0
    else:
        bk, bc, wk, wc, bq = tile
        tk = tc = 0
    c = dict(
        atom=atom,
        atom_id=ATOMS.index(atom),
        loop=loop,
        layout=layout,
        bk=bk,
        bc=bc,
        bq=bq,
        fr=fr,
        s=s,
        sw=sw,
        dw=dw,
        wk=wk,
        wc=wc,
        tk=tk,
        tc=tc,
        vec=vec,
        xcd=xcd,
        align=align,
        ns=ns,
        bpu=0,
        wq=0,
    )
    return dict(c, dws=dws_of(c), pb=0, math=0)


def D(atom, tile, s, dw, fr, ns):
    """A direct-loop config: tile (bk, bc, wk, wc, wq)."""
    bk, bc, wk, wc, wq = tile
    return dict(
        atom=atom,
        atom_id=ATOMS.index(atom),
        loop=5,
        layout=TR,
        bk=bk,
        bc=bc,
        bq=32,
        fr=fr,
        s=s,
        sw=1,
        dw=dw,
        wk=wk,
        wc=wc,
        tk=0,
        tc=0,
        vec=8,
        xcd=1,
        align=0,
        ns=ns,
        bpu=1,
        wq=wq,
        dws=0,
        pb=0,
        math=0,
    )


def with_bpu(c):
    """The planner's blocks per dispatch unit: for the gfx942 / gfx950 row rings and packed rows
    on matrix cores, 1, 2 and 4 with the 16 x 16 fp32 atom (3-D fp32 layers want 4) and 1 and 2 otherwise
    (autotune picks); the FMA output rows 2 and 4; the generic taps 2 (as measured); elsewhere
    one value, 1 for blocks of 512 threads or more, else 2."""
    if c["loop"] == 5:
        return [c]
    if c["loop"] == 8:
        return [dict(c, bpu=2)]
    if c["loop"] in (3, 4):
        threads = (c["bk"] // c["tk"]) * (c["bc"] // c["tc"])
    else:
        threads = c["wk"] * c["wc"] * wave(c["atom"])
    if (
        c["loop"] in (1, 2, 7)
        and c["atom"] != "fma"
        and {"gfx942", "gfx950"} & set(archs_of(c))
    ):
        return [
            dict(c, bpu=b) for b in ((1, 2, 4) if c["atom"] == "mfma16_f32" else (1, 2))
        ]
    if c["loop"] in (
        4,
        6,
    ):  # FMA output rows (3-D stride 2, gfx1201) and small-C rows want 2 or 4
        return [dict(c, bpu=b) for b in (2, 4)]
    return [dict(c, bpu=1 if threads >= 512 else 2)]


# Direct-loop tiles (bk, bc, wk, wc, wq), gfx950 16-bit: the wide ones measured best on the
# flux / sdxl 3x3 class, waves_q for narrow channel counts, one-fragment waves for the long
# 1 x N filters, and wide pointwise tiles. Ring slots: 3 (2 rows in flight); 1x1 also 4.
DIRECT_TILES = [
    (128, 64, 4, 2, 1),
    (64, 128, 2, 4, 1),
    (64, 64, 2, 2, 2),
    (32, 128, 1, 4, 2),
    (32, 32, 2, 2, 2),
    (32, 64, 2, 4, 1),
]
DIRECT_1x1 = [(128, 128, 2, 4, 1), (128, 256, 2, 4, 1)]
DIRECT_FILTERS = [
    ("3x3", 3, 1, 3),
    ("1x1", 1, 1, 1),
    ("1x7", 7, 1, 1),
    ("1x7d3", 7, 3, 1),
    ("1x7d5", 7, 5, 1),
    ("1x11", 11, 1, 1),
    ("1x11d3", 11, 3, 1),
    ("1x11d5", 11, 5, 1),
]


def build_direct():
    dropped = 0
    for dtype, atom in (("HALF", "mfma16x32_f16"), ("BFLOAT16", "mfma16x32_bf16")):
        for tag, s, dw, fr in DIRECT_FILTERS:
            for t in DIRECT_TILES + (DIRECT_1x1 if tag == "1x1" else []):
                for ns in ((3, 4) if tag == "1x1" else (3,)):
                    c = D(atom, t, s, dw, fr, ns)
                    if valid(c, dtype):
                        CATALOG[dtype].append(c)
                    else:
                        dropped += 1
    return dropped


# Output channel counts that are not whole vectors (e.g. a VAE's 3-channel output layer),
# channels-last: the ring with dy units at any element (align 2), one fragment of output
# channels per block. (atoms per dtype, tiles (bk, bc, wk, wc, bq), xcd).
UNALIGNED_K = [
    (
        {"HALF": "wmma11_f16", "BFLOAT16": "wmma11_bf16"},
        [(16, 128, 1, 4, 32), (16, 64, 1, 4, 64)],
        0,
    ),
    (
        {"HALF": "wmma12_f16", "BFLOAT16": "wmma12_bf16"},
        [(16, 128, 1, 4, 32), (16, 64, 1, 4, 64)],
        0,
    ),
    (
        {"HALF": "mfma32x16_f16", "BFLOAT16": "mfma32x16_bf16"},
        [(32, 128, 1, 4, 64), (32, 64, 1, 2, 64)],
        1,
    ),
    (
        {"HALF": "mfma32_f16", "BFLOAT16": "mfma32_bf16"},
        [(32, 128, 1, 4, 32), (32, 64, 1, 2, 32)],
        0,
    ),
    ({"FLOAT": "mfma32_f32"}, [(32, 128, 1, 4, 16), (32, 64, 1, 2, 16)], 0),
]
UNALIGNED_K_FILTERS = [("3x3", 3, 3), ("1x1", 1, 1)]


def build_unaligned_k():
    dropped = 0
    for atoms, tiles, xcd in UNALIGNED_K:
        for dtype, atom in atoms.items():
            for tag, s, fr in UNALIGNED_K_FILTERS:
                for t in tiles:
                    c = C(atom, 1, CL, t, s, 1, 1, fr, 16 // eb(dtype), xcd, 2, 0)
                    if valid(c, dtype):
                        CATALOG[dtype].extend(with_bpu(c))
                    else:
                        dropped += 1
    return dropped


# Small-C rows (loop 6, stems): 16 x 16 atoms per dtype (xcd), tiles (bk, bc, wk, bq), filters
# (tag, s, sw, fr). Measured on gfx950: 7x7 s2 fr 7 64 x 4 at 64 positions; 3x3 32 x 4 at 64.
SMALLC_ATOMS = [
    ({"HALF": "mfma16x32_f16", "BFLOAT16": "mfma16x32_bf16"}, 1),
    ({"HALF": "mfma16_f16", "BFLOAT16": "mfma16_bf16", "FLOAT": "mfma16_f32"}, 0),
    ({"HALF": "wmma11_f16", "BFLOAT16": "wmma11_bf16"}, 0),
    ({"HALF": "wmma12_f16", "BFLOAT16": "wmma12_bf16"}, 0),
]
SMALLC_TILES = [(64, 4, 4, 64), (32, 4, 2, 64), (64, 8, 4, 32), (64, 16, 4, 32)]
# Channels-last stems also take longer position steps: EfficientNet-v2 stems (c3 k24 3x3 s2,
# 480^2 and 384^2) ran 13-22% faster at 128 or 256 positions a step on gfx942 / gfx950.
SMALLC_TILES_CL = [(32, 4, 2, 128), (64, 4, 4, 128), (64, 4, 4, 256)]
SMALLC_FILTERS = [
    ("7x7s2", 7, 2, 7),
    ("7x7s2r1", 7, 2, 1),
    ("3x3s2", 3, 2, 3),
    ("3x3", 3, 1, 3),
    ("5x5", 5, 1, 5),
]
# 5x5 small-C rows (3-D 5x5x5 with 16 input channels, AIPYTORCH-1356): 16-bit, channels-first,
# the tile measured best per atom family (gfx950: 32 x 4 at 64 positions; gfx1201: 64 x 8 at 32).
SMALLC_5X5_TILES = {"mfma": [(32, 4, 2, 64)], "wmma": [(64, 8, 4, 32)]}


def build_smallc():
    dropped = 0
    for atoms, xcd in SMALLC_ATOMS:
        for dtype, atom in atoms.items():
            vec = 16 // eb(dtype)
            for tag, s, sw, fr in SMALLC_FILTERS:
                for lay in (CF, CL):
                    tiles = SMALLC_TILES + (SMALLC_TILES_CL if lay == CL else [])
                    if tag == "5x5":
                        if dtype == "FLOAT" or lay != CF:
                            continue
                        tiles = SMALLC_5X5_TILES[atom[:4]]
                    for bk, bc, wk, bq in tiles:
                        c = C(
                            atom,
                            6,
                            lay,
                            (bk, bc, wk, 1, bq),
                            s,
                            sw,
                            1,
                            fr,
                            vec,
                            xcd,
                            0,
                            0,
                        )
                        if valid(c, dtype):
                            CATALOG[dtype].extend(with_bpu(c))
                        else:
                            dropped += 1
    return dropped


def fma_cands(tiles, layouts, lays, s, sw, dw, fr, loop, vec):
    """The FMA configs of a filter row. Unit height stride: the FMA ring (aligned rows) and the
    FMA output rows with scalar units (any row); stride 2: the FMA output rows, vectors and
    scalars."""
    cands = []
    for t in tiles:
        for lay in layouts if lays is None else [x for x in layouts if x in lays]:
            if loop == 1:
                cands.append(C("fma", 3, lay, t, s, sw, dw, fr, vec, 0, 0, 0))
                cands.append(C("fma", 4, lay, t, s, sw, dw, fr, 1, 0, 0, 0))
            else:
                cands.append(C("fma", 4, lay, t, s, sw, dw, fr, vec, 0, 0, 0))
                cands.append(C("fma", 4, lay, t, s, sw, dw, fr, 1, 0, 0, 0))
    return cands


# gfx906 (no matrix core): the FMA loops for every element type, 16-bit accumulating in fp32, on
# FMA32's tiles plus 16-input-channel tiles (small-C layers: SqueezeNet's c16, stems), and on the
# output rows 4-position steps for maps of a few output positions (FCN's 7x7 point_output on a
# 7x7 map: one output per image). Shipped in conv_wrw_<dtype>_fma906 packs that list gfx906
# only, so none of it is a candidate where matrix cores run.
FMA906_SMALLC = [(64, 16, 4, 4, 32), (32, 16, 4, 4, 32)]
FMA906_ROWS = [(64, 64, 4, 4, 4), (64, 128, 4, 4, 4)]
# The filter rows that take the small-C tiles (stems, SqueezeNet-like 3x3 and 1x1, the 1-D
# rows) and the short-step output rows (1 x S rows serve any filter height on the output rows).
SMALLC906 = {
    "3x3",
    "1x1",
    "1x3",
    "1x7",
    "3x3r1",
    "3x3s2",
    "3x3s2r1",
    "7x7s2r1",
    "2x2s2",
}
ROWS906 = {"1x1", "1x3", "1x7", "1x11"}
FMA906_ARCHS = ["gfx906"]
CATALOG906 = {"FLOAT": [], "HALF": [], "BFLOAT16": []}


def build_fma906():
    dropped = 0
    for dtype in CATALOG906:
        vec = 16 // eb(dtype)
        for tag, s, sw, dw, fr, loop, idx, aligns, lays in FILTERS:
            if tag in MATRIX16_ONLY or tag in NO_FMA906:
                continue
            # fp32 already has FMA32's tiles (and FMA_TILES) in the shared FMA pack.
            tiles = [] if dtype == "FLOAT" else FMA_TILES.get(tag, FMA32)
            tiles = tiles + (FMA906_SMALLC if tag in SMALLC906 else [])
            # Channels-last 7x7 stride-2 stems too: elsewhere the small-C rows serve them, on
            # gfx906 only these FMA rows can.
            if tag == "7x7s2r1" and CL not in lays:
                lays = lays + [CL]
            cands = fma_cands(tiles, [CF, CL], lays, s, sw, dw, fr, loop, vec)
            if tag in ROWS906:
                cands += fma_cands(FMA906_ROWS, [CF, CL], lays, s, sw, dw, fr, 0, vec)
            # Dilated filters: one filter row per block on the output rows, whose dilated
            # window of three rows overflows 64 KiB in fp32 (deeplab d2 to d36).
            if dw > 1:
                cands += fma_cands(
                    FMA32 + FMA906_SMALLC, [CF, CL], lays, s, sw, dw, 1, 0, vec
                )
            for c in cands:
                if valid(c, dtype, fma16=True):
                    CATALOG906[dtype].extend(with_bpu(c))
                else:
                    dropped += 1
    # Drop what the shared FMA pack already holds (fp32) and repeats (as build()).
    for dtype, cfgs in CATALOG906.items():
        seen = {tuple(c[f] for f in INTS) for c in CATALOG[dtype]}
        kept = []
        for c in cfgs:
            key = tuple(c[f] for f in INTS)
            if key not in seen:
                seen.add(key)
                kept.append(c)
        CATALOG906[dtype] = kept
    return dropped


# RDNA fp32 (no fp32 matrix core): FMA tiles sized for few channels, on few-channel layers on
# large maps (issue #15446: IGM's 3x3 C = K = 24 on 1505 x 2255 single images). FMA32's 32 x 64
# tiles fill 24 of 64 input and 24 of 32 output channels there. These tiles are 24 x 24 at 32
# positions and 48 x 48 at 64, measured on gfx1100 (1.3-1.5x on C = 24). Their blocks are small
# (36 threads), so the planner also targets 8 blocks per unit. The pack lists the RDNA archs, so
# none of it becomes a candidate where an fp32 matrix core runs.
FMARDNA_TILES = [(24, 24, 4, 4, 32), (48, 48, 4, 4, 64)]
FMARDNA_TAGS = {"3x3", "3x3r1", "1x1", "1x3", "3x3s2", "3x3s2r1"}
FMARDNA_ARCHS = [
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1200",
    "gfx1201",
]
CATALOGRDNA = {"FLOAT": []}


def build_fmardna():
    dropped = 0
    vec = 4
    for tag, s, sw, dw, fr, loop, idx, aligns, lays in FILTERS:
        if tag not in FMARDNA_TAGS:
            continue
        for c in fma_cands(FMARDNA_TILES, [CF, CL], lays, s, sw, dw, fr, loop, vec):
            if valid(c, "FLOAT"):
                CATALOGRDNA["FLOAT"].extend(dict(c, bpu=b) for b in (2, 4, 8))
            else:
                dropped += 1
    seen = {tuple(c[f] for f in INTS) for c in CATALOG["FLOAT"] + CATALOG906["FLOAT"]}
    kept = []
    for c in CATALOGRDNA["FLOAT"]:
        key = tuple(c[f] for f in INTS)
        if key not in seen:
            seen.add(key)
            kept.append(c)
    CATALOGRDNA["FLOAT"] = kept
    return dropped


# Block-based buffers (pb 1): channels-first images past the 32-bit buffer range (3-D
# ConvTranspose3d / volumes of 64 channels at 257^3). A block's bc x channel planes must fit the
# range, so the tiles take 32 (16-bit) or 16 (fp32) input channels. Measured on 257^3 3x3x3
# stride 2 (gfx950 / gfx1201): (atom, loop, filter rows, tile).
PB_CONFIGS = [
    ({"HALF": "mfma32x16_f16", "BFLOAT16": "mfma32x16_bf16"}, 1, 3, (64, 32, 2, 1, 32)),
    ({"FLOAT": "mfma16_f32"}, 1, 1, (32, 16, 2, 1, 64)),
    ({"HALF": "wmma12_f16", "BFLOAT16": "wmma12_bf16"}, 0, 3, (64, 32, 4, 2, 32)),
    ({"HALF": "wmma11_f16", "BFLOAT16": "wmma11_bf16"}, 0, 3, (64, 32, 4, 2, 32)),
]


def build_pb():
    dropped = 0
    for atoms, xcd, fr, tile in PB_CONFIGS:
        for dtype, atom in atoms.items():
            c = C(atom, 0, CF, tile, 3, 2, 1, fr, 16 // eb(dtype), xcd, 0, 0)
            if valid(c, dtype):
                CATALOG[dtype].extend(dict(x, pb=1) for x in with_bpu(c))
            else:
                dropped += 1
    return dropped


# The generic tier (loop 8): few configs of runtime filter, strides and dilation, each declined
# wherever a specialized config's class serves (its "spec", written by emit). (atoms per dtype,
# xcd, [(layout, (bk, bc, wk, wc, bq), taps a block[, shared-window dw])]), the measured best per arch: gfx950 on
# 2-D 5x5 / 7x7 and 3-D 1x5x5 channels-last fp16 (1.5-1.9x over the BLAS path), gfx1201 on the
# same and 1-D k3 (1.7-3.2x).
GENERIC = [
    (
        {"HALF": "mfma32x16_f16", "BFLOAT16": "mfma32x16_bf16"},
        1,
        [
            (CL, (64, 64, 2, 2, 64), 3),
            (CL, (128, 64, 4, 2, 32), 3),
            (CL, (64, 64, 2, 2, 32), 5),
            (CF, (128, 64, 4, 2, 32), 2),
            # Shared width window (dw 8): 2-D 5x5 / 7x7, 1x5x5 and 1-D k9 5-12% faster.
            (CL, (64, 64, 2, 2, 64), 9, 8),
        ],
    ),
    (
        {"HALF": "wmma12_f16", "BFLOAT16": "wmma12_bf16"},
        0,
        [
            (CL, (64, 64, 4, 2, 32), 5),
            (CL, (128, 64, 4, 2, 32), 3),
            (CL, (64, 32, 4, 2, 32), 9),
            (CF, (64, 64, 4, 2, 32), 2),
            # Shared width window (dw 16): 1-D k9 776 against 838 us.
            (CL, (128, 64, 4, 2, 32), 5, 16),
        ],
    ),
    # gfx908 to gfx950 (64 KiB of LDS): gfx950's tiles at 32 positions.
    (
        {"HALF": "mfma32_f16", "BFLOAT16": "mfma32_bf16"},
        0,
        [
            (CL, (128, 64, 4, 2, 32), 3),
            (CL, (64, 64, 2, 2, 32), 5),
            (CF, (128, 64, 4, 2, 32), 2),
        ],
    ),
    (
        {"FLOAT": "mfma32_f32"},
        0,
        [(CF, (64, 64, 2, 2, 16), 3), (CL, (64, 64, 2, 2, 16), 3)],
    ),
    # The FMA form, fp32 where no matrix core takes it (RDNA; tiles (bk, bc, tk, tc, bq)),
    # measured on gfx1201 against the BLAS path: 5x5 1.6x, 1-D k9 1.4x, 3x3 s3 4.2x, 5x5x5 3.5x.
    (
        {"FLOAT": "fma"},
        0,
        [
            (CF, (64, 64, 4, 4, 16), 5),
            (CF, (128, 64, 8, 4, 16), 3),
            (CL, (64, 64, 8, 4, 16), 3),
            (CL, (64, 64, 4, 4, 32), 5, 8),
        ],
    ),
]


def build_generic():
    dropped = 0
    for atoms, xcd, tiles in GENERIC:
        for dtype, atom in atoms.items():
            for lay, tile, taps, *window in tiles:
                wx = window[0] if window else 0  # dw: shared-window positions
                c = C(atom, 8, lay, tile, 0, 0, wx, taps, 16 // eb(dtype), xcd, 0, 0)
                if valid(c, dtype):
                    CATALOG[dtype].extend(with_bpu(c))
                    # Its block-based buffer twin: images past the 32-bit buffer range.
                    CATALOG[dtype].extend(with_bpu(dict(c, pb=1)))
                else:
                    dropped += 1
    return dropped


SPEC_KIND = {0: "R", 6: "C", 1: "P", 2: "G", 5: "D"}


def spec_of(g, cfgs):
    """The specialized classes generic config g declines: every matrix-core config of the dtype
    (the K3Tiles rows and rings, small-C rows, the DMA ring and the direct loop, and the
    block-based buffer forms of rows, rings and small-C rows, in lower case; not the FMA loops
    or packed rows (narrow outputs only)) whose atom runs on one of g's archs, as
    "<atom_id><kind><layout>.<s>.<sw>.<dw>.<fr>" keys (ConvWrwProblem.hpp). The FMA form's
    configs (atom "fma") also decline the FMA loops' classes: F (output rows of vectors), S (of
    scalar units), Q (the FMA ring).
    """
    keys = set()
    for c in cfgs:
        if c["atom"] == "fma":
            if g["atom"] != "fma" or c["loop"] not in (3, 4):
                continue
            kind = "Q" if c["loop"] == 3 else "S" if c["vec"] == 1 else "F"
        elif c["loop"] not in SPEC_KIND or not set(archs_of(c)) & set(archs_of(g)):
            continue
        else:
            kind = "U" if c["loop"] == 1 and c["align"] == 2 else SPEC_KIND[c["loop"]]
            kind = kind.lower() if c["pb"] else kind
        fr = 0 if kind.upper() in "RCFS" else c["fr"]  # the rows' kinds carry no height
        keys.add(f"{c['atom_id']}{kind}{c['layout']}.{c['s']}.{c['sw']}.{c['dw']}.{fr}")
    return ";".join(sorted(keys))


def build():
    dropped = build_direct() + build_unaligned_k() + build_smallc() + build_pb()
    for atoms, tiles, xcd, layouts, loop2, only in FAMILIES:
        for dtype, atom in atoms.items():
            vec = 16 // eb(dtype)
            for tag, s, sw, dw, fr, loop, idx, aligns, lays in FILTERS:
                if only is not None and tag not in only:
                    continue
                if tag in MATRIX16_ONLY and dtype == "FLOAT":
                    continue
                if tag in FLOAT_ONLY and dtype != "FLOAT":
                    continue
                picked = (
                    tiles
                    if idx is None
                    else [tiles[i] for i in idx if -len(tiles) <= i < len(tiles)]
                )
                picked = list(dict.fromkeys(picked))  # indices may name one tile twice
                if tag in ONE_D and atom.startswith("wmma"):
                    picked = picked + [t for t in RDNA_1D if t not in picked]
                for (prefix, tags), extra in EXTRA_TILES.items():
                    if atom.startswith(prefix) and tag in tags:
                        picked = picked + [t for t in extra if t not in picked]
                if atom == "fma":
                    picked = FMA_TILES.get(tag, picked)
                    cands = fma_cands(picked, layouts, lays, s, sw, dw, fr, loop, vec)
                    for c in cands:
                        if valid(c, dtype):
                            CATALOG[dtype].extend(with_bpu(c))
                        else:
                            dropped += 1
                    continue
                cands = []
                for t in picked:
                    for lay in (
                        layouts if lays is None else [x for x in layouts if x in lays]
                    ):
                        for al in (aligns if lay == CF and loop == 1 else [0]):
                            cands.append(
                                C(atom, loop, lay, t, s, sw, dw, fr, vec, xcd, al, 0)
                            )
                    if loop == 1 and tag in LOOP2:
                        for lay in loop2:
                            cands.append(
                                C(atom, 2, lay, t, s, sw, dw, fr, vec, 1, 0, LOOP2_NS)
                            )
                if tag == "1x1" and TR in layouts:
                    for t in BIG1x1:
                        cands.append(C(atom, 1, TR, t, s, sw, dw, fr, vec, xcd, 0, 0))
                        cands.append(
                            C(atom, 2, TR, t, s, sw, dw, fr, vec, 1, 0, LOOP2_NS)
                        )
                for c in cands:
                    if valid(c, dtype):
                        CATALOG[dtype].extend(with_bpu(c))
                    else:
                        dropped += 1
    # The generic tier last: the specialized configs keep their priorities.
    dropped += build_generic()
    # Two rows can give one config (the FMA output rows ignore the filter height: 1x3 and 3x3r1
    # map alike); keep the first.
    for dtype, cfgs in CATALOG.items():
        seen = set()
        kept = []
        for c in cfgs:
            key = tuple(c[f] for f in INTS)
            if key not in seen:
                seen.add(key)
                kept.append(c)
        CATALOG[dtype] = kept
    return dropped


# Three-bf16 (math 1): fp32 graphs under TF32 permission run a bf16 config on x and dy split
# into 3N bf16 images (ConvWrwSplit), dw in fp32 through the reduce (so dws 0). The configs are
# a subset of the bf16 catalog on the atoms where three bf16 products beat exact fp32: per
# (atom, loop, layout, filter, align) its first two tiles, loops 0, 1, 5 (gfx950 direct) and 6
# (small-C). Not gfx908 (its half-rate bf16 atom makes three products slower than fp32 MFMA).
X3_ATOMS = {
    "mfma16_bf16",  # the CDNA small-C rows (stems)
    "mfma32_bf16",
    "mfma32x16_bf16",
    "mfma16x32_bf16",
    "wmma11_bf16",
    "wmma12_bf16",
}
X3_LOOPS = (0, 1, 5, 6)
X3_TILES = 2
CATALOGX3 = []


def build_x3():
    groups = {}
    for c in CATALOG["BFLOAT16"]:
        # Not the block-buffer forms (pb): the 3N problem keeps the 32-bit buffer range.
        if c["atom"] not in X3_ATOMS or c["loop"] not in X3_LOOPS or c["pb"]:
            continue
        g = (
            c["atom"],
            c["loop"],
            c["layout"],
            c["s"],
            c["sw"],
            c["dw"],
            c["fr"],
            c["align"],
        )
        t = tuple(c[f] for f in ("bk", "bc", "bq", "wk", "wc", "wq", "ns", "xcd"))
        tiles = groups.setdefault(g, [])
        if t in tiles or len(tiles) >= X3_TILES:
            continue
        tiles.append(t)
        CATALOGX3.append(dict(c, dws=0, math=1))
    seen = set()
    kept = []
    for c in CATALOGX3:
        key = tuple(c[f] for f in INTS)
        if key not in seen:
            seen.add(key)
            kept.append(c)
    CATALOGX3[:] = kept


def dws_of(c):
    """CONV_K3_DWS: 1 compiles the direct dw store (one split into K T R S C or one tap) into
    the epilogue, 0 keeps the partials-only epilogue. 0 for loop 5 (partials only) and for
    gfx12's WMMA tiles of 512 or more accumulators per wave (16 x 32 and up), where the second
    epilogue cost 7-17% on gfx1201 at any split."""
    if c["loop"] == 5:
        return 0
    if (
        c["atom"].startswith("wmma12")
        and (c["bk"] // c["wk"]) * (c["bc"] // c["wc"]) >= 512
    ):
        return 0
    return 1


def family_of(c):
    """The pack a config ships in: the direct loop, gfx950's 16 x 32 / 32 x 16 atoms, the other
    MFMA atoms, RDNA's WMMA, or the FMA loops. The 32 x 32 atom's packed rows ship in a pack
    without gfx950, which has its own measured packed-rows tiles."""
    if c["loop"] == 5:
        return "direct"
    if c["loop"] == 8:
        return "genericfma" if c["atom"] == "fma" else "generic"
    if c["loop"] == 7 and c["atom"].startswith("mfma32_"):
        return "cdnapk"
    if c["atom"] == "fma":
        return "fma"
    if c["atom"].startswith("wmma"):
        return "rdna"
    if c["atom"].startswith(("mfma16x32", "mfma32x16")):
        return "gfx950"
    return "cdna"


def emit():
    kmd, umd, udd = U("kmd"), U("umd"), U("udd")
    write(
        "conv_wrw.kmd.json",
        {
            "version": "1.0",
            "id": kmd,
            "name": "conv_wrw config fields",
            "fields": [
                {"name": f, "type": "string" if f in STRINGS else "int"} for f in FIELDS
            ]
            # The generic loop's specialized classes; empty for every other loop.
            + [{"name": "spec", "type": "string", "default_value": ""}],
        },
    )
    write(
        "kernel_matches_graph.umd.json",
        {
            "version": "1.0",
            "id": umd,
            "name": "kernel dtype, atom (device), layout, loop, filter width and tile match the graph",
            "scope": "kernel",
            "match_symbol": "hipkernel.conv_wrw.kernel_match",
        },
    )
    write(
        "conv_wrw.udd.json",
        {
            "version": "1.0",
            "id": udd,
            "name": "conv_wrw dispatch",
            "dispatch_symbol": "hipkernel.conv_wrw.dispatch",
        },
    )
    ued = U("ued")
    write(
        "conv_wrw.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": "hipkernel:CONV_WRW",
            "graph_match": {"native": "hipkernel.conv_wrw.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    # One pack per dtype and kernel family (the repository caps a file at 500 KB); all serve the
    # one engine. Priorities run over the dtype's whole list, as before the split.
    for stale in out.glob("conv_wrw*.kdp.json"):
        stale.unlink()
    kernels = []
    for dtype, cfgs in CATALOG.items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        packs = {}
        # The gfx906 and RDNA FMA configs follow the dtype's list, in their own packs that
        # list only those archs.
        # The gfx906 list splits in two packs (fma906, fma906b) under the size cap.
        half906 = (len(CATALOG906[dtype]) + 1) // 2
        # The generic tier comes last, after every specialized config.
        tagged = (
            [(c, family_of(c)) for c in cfgs if c["loop"] != 8]
            + [
                (c, "fma906" if j < half906 else "fma906b")
                for j, c in enumerate(CATALOG906[dtype])
            ]
            + [(c, "fmardna") for c in CATALOGRDNA.get(dtype, [])]
            # Three-bf16 (TF32 permission only) in packs of their own, ahead of the generic tier.
            + [
                (c, "x3" + family_of(c))
                for c in (CATALOGX3 if dtype == "FLOAT" else [])
            ]
            + [(c, family_of(c)) for c in cfgs if c["loop"] == 8]
        )
        for i, (c, fam) in enumerate(tagged):
            pack = packs.setdefault(fam, [])
            lay = ["nchw", "nhwc", "nhwctr"][c["layout"]]
            thr = (
                f"t{c['tk']}x{c['tc']}"
                if c["atom"] == "fma"
                else f"w{c['wk']}x{c['wc']}"
            )
            if c["loop"] == 5:
                thr += f"q{c['wq']}"
            name = (
                f"conv_wrw.{dt}{'x3' if c['math'] else ''}.{c['atom']}.l{c['loop']}.{lay}.{c['bk']}x{c['bc']}x{c['bq']}"
                f".{thr}.fr{c['fr']}s{c['s']}sw{c['sw']}dw{c['dw']}"
                f".v{c['vec']}.x{c['xcd']}a{c['align']}n{c['ns']}b{c['bpu']}"
                + (".pb" if c["pb"] else "")
            )
            pack.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvWrw.cpp",
                        "entry_point": "ConvWrw",
                    },
                    "metadata": {
                        "direction": "WRW",
                        "dtype": dtype,
                        **c,
                        **(
                            {"spec": spec_of(c, cfgs + CATALOGRDNA.get(dtype, []))}
                            if c["loop"] == 8
                            else {}
                        ),
                    },
                    "priority": 1000 - i,
                }
            )
        for fam, pack in sorted(packs.items()):
            write(
                f"conv_wrw_{dt}_{fam}.kdp.json",
                {
                    "version": "1.0",
                    "id": U(f"kdp.{dt}.{fam}"),
                    "name": f"hipkernel:conv_wrw_{dt}_{fam}",
                    "arch": {
                        "fma906": FMA906_ARCHS,
                        "fma906b": FMA906_ARCHS,
                        "fmardna": FMARDNA_ARCHS,
                        "cdnapk": CDNA[:3],
                        "genericfma": FMARDNA_ARCHS,
                    }.get(fam, []),
                    "matchers": [umd],
                    "engine": ued,
                    "dispatch": udd,
                    "kernelDescriptors": pack,
                },
            )
            assert (
                out / f"conv_wrw_{dt}_{fam}.kdp.json"
            ).stat().st_size < 500 * 1000, f"{dt}_{fam} over 500 KB"
            kernels += pack
    names = [k["name"] for k in kernels]
    assert len(names) == len(set(names)), "duplicate config names"
    tuples = [
        (k["metadata"]["dtype"],) + tuple(k["metadata"][f] for f in INTS)
        for k in kernels
    ]
    assert len(tuples) == len(set(tuples)), "duplicate knob tuples"
    return len(kernels)


if __name__ == "__main__":
    dropped = build()
    dropped += build_fma906()
    dropped += build_fmardna()
    build_x3()
    n = emit()
    print(n, "kernel configs;", dropped, "candidates dropped (LDS, passes or tile)")
    for dtype, cfgs in CATALOG.items():
        print(
            " ",
            dtype,
            len(cfgs),
            "+",
            len(CATALOG906[dtype]),
            "gfx906 FMA +",
            len(CATALOGRDNA.get(dtype, [])),
            "RDNA FMA",
        )
    print("  FLOAT three-bf16 (TF32 permission):", len(CATALOGX3))
