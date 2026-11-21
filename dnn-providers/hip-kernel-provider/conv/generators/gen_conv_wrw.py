# Generates the CONV_WRW descriptors (backward weights, K3). Each config is admitted on
# correctness alone (its atom on the device and taking the graph's dtype, its layout the graph's,
# its specialized filter geometry equal to the problem's (loops 8-12 instead keep geometry
# runtime), its LDS within the device's); autotune (UHD later) ranks them. Loop 10 is a complete
# FLOAT candidate with FP64 products, partials and fold, not an alternate mode of an old kernel.
# Priority orders the declared fallback; accurate entries follow all legacy entries.
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
# The accurate loop's bounded float staging uses (BK + BC) * (BQ + 1) * 4 bytes instead.
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
    "wmma125_f16",
    "wmma125_bf16",
    "wmma125_f32",
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
    "wmma125_f16": (16, 32, 16),
    "wmma125_bf16": (16, 32, 16),
    "wmma125_f32": (16, 4, 2),
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
    "wmma125_f16": ["gfx1250"],
    "wmma125_bf16": ["gfx1250"],
    "wmma125_f32": ["gfx1250"],
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
    "flat",
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


def cl_pitch(n, e, fb):  # ConvWrwGeom.h k3ClPitch
    g = 4 if fb >= 16 else 2 if fb >= 8 else 1
    d = du(du(n * e, 4), g) * g
    if (d // g) % 2 == 0:
        d += g
    return d * 4 // e


def lds_bytes(c, e):  # ConvWrwGeom.h k3LdsBytes
    loop, layout, bk, bc, bq, fr, s, sw, dw = (
        c[k] for k in ("loop", "layout", "bk", "bc", "bq", "fr", "s", "sw", "dw")
    )
    vec, align, ns = c["vec"], c["align"], c["ns"]
    if loop == 10:
        return (bk + bc) * (bq + 1) * 4
    if loop == 9:
        return 0
    if loop == 12:
        return (bk + bc) * (bq + 16 // e) * e + bq * 32
    if loop == 11:
        return (bk + bc) * (bq + 16 // e) * e
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
    units = du(ph + 2, skew) + 1  # k3CfPitch: odd at unit width stride, channels-first
    ldb = (
        tr_pitch(bc)
        if tr
        else (units | 1 if sw == 1 and layout == CF else units) * skew
    )
    if loop == 1 and layout == CL and c["atom"].startswith("mfma"):
        # The CDNA ring's channels-last rows: k3ClPitch of the fragment read's bytes.
        fb = SHAPE[c["atom"]][2] * e
        lda, ldb = cl_pitch(bq, e, fb), cl_pitch(ph + 2, e, fb)
    a = (bq if tr else bk) * lda + (vec if align else 0)
    b = rows * sw * (ph if tr else bc) * ldb
    slots = c["wq"] if loop == 1 and c["wq"] > 1 else 1  # the ring's steps a stage
    return 2 * slots * (a + b) * e


def passes(c, e):  # ConvWrwProblem.cpp loadPassesOf
    vec = c["vec"]
    if c["loop"] in (9, 10, 11, 12):
        return 0
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
    if c["loop"] in (9, 10, 11, 12):
        return ["gfx1250"]
    return ["gfx950"] if c["loop"] in (2, 5) or c["layout"] == TR else ARCHS[c["atom"]]


def lds_limit(c):
    archs = archs_of(c)
    return (
        320 * 1024
        if archs == ["gfx1250"]
        else 160 * 1024 if archs == ["gfx950"] else 64 * 1024
    )


def valid(c, dtype, fma16=False):
    """Whether the pack admits config c for dtype on some architecture. fma16: the FMA loops on
    16-bit graphs, which only devices without a matrix core for the type take (gfx906).
    """
    e = eb(dtype)
    if c["loop"] == 10:
        return (
            dtype == "FLOAT"
            and c["atom"] == "fma"
            and c["atom_id"] == 16
            and c["layout"] in (CF, CL)
            and c["bk"] in (8, 16)
            and c["bc"] in (8, 16)
            and c["bq"] in (32, 64)
            and c["tk"] == c["tc"] == c["vec"] == 1
            and c["fr"] == c["s"] == c["sw"] == c["dw"] == 0
            and c["wk"] == c["wc"] == c["xcd"] == c["align"] == c["ns"] == c["wq"] == 0
            and c["dws"] == 0
            and c["bpu"] in (1, 2, 4, 8)
            and lds_bytes(c, e) <= lds_limit(c)
        )
    if c["loop"] == 9:
        return (
            c["atom"] == "fma"
            and c["atom_id"] == 16
            and c["layout"] == CL
            and c["bk"] == c["tk"] == 4
            and c["tc"] == c["vec"] == 1
            and c["bc"] in (64, 128, 256)
            and c["bq"] in (64, 128, 256)
            and c["fr"] == c["s"] == c["sw"] == c["dw"] == 0
            and c["wk"] == c["wc"] == c["xcd"] == c["align"] == c["ns"] == c["wq"] == 0
            and c["dws"] == 1
            and c["bpu"] in (1, 2, 4, 8)
            and lds_bytes(c, e) == 0
        )
    if c["loop"] in (11, 12):
        dim, ka, _ = SHAPE[c["atom"]]
        return (
            c["atom"].startswith("wmma125_")
            and c["layout"] in (CF, CL)
            and c["vec"] == 1
            and c["fr"] == c["s"] == c["sw"] == c["dw"] == 0
            and c["tk"] == c["tc"] == c["xcd"] == c["align"] == c["ns"] == c["wq"] == 0
            and (
                c["dws"] == 1
                or (
                    c["dws"] == 2
                    and c["loop"] == 12
                    and c["layout"] == CF
                    and c["atom"] == "wmma125_f16"
                    and (c["bk"], c["bc"], c["bq"], c["wk"], c["wc"])
                    == (64, 64, 32, 2, 2)
                )
            )
            and c["bpu"] in (1, 2, 4, 8)
            and 1 <= c["wk"] <= 8
            and 1 <= c["wc"] <= 8
            and 1 <= c["bk"] <= 128
            and 1 <= c["bc"] <= 128
            and 1 <= c["bq"] <= 256
            and c["bk"] % (c["wk"] * dim) == c["bc"] % (c["wc"] * dim) == 0
            and c["bq"] % ka == 0
            and c["wk"] * c["wc"] * wave(c["atom"]) <= 1024
            and lds_bytes(c, e) <= lds_limit(c)
            and (
                c["loop"] == 11
                or (
                    (c["wk"] * c["wc"] * wave(c["atom"])) % c["bk"] == 0
                    and (c["wk"] * c["wc"] * wave(c["atom"])) % c["bc"] == 0
                    and (c["wk"] * c["wc"] * wave(c["atom"]))
                    % min(c["bq"], c["wk"] * c["wc"] * wave(c["atom"]))
                    == 0
                    and c["bq"] % min(c["bq"], c["wk"] * c["wc"] * wave(c["atom"])) == 0
                )
            )
        )
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
        # The matrix-core ring's steps a stage (K3 WQ): 2 or 4, channels-first or -last.
        ok = (
            ok
            and c["loop"] == 1
            and c["wq"] in (2, 4)
            and c["layout"] in (CF, CL)
            and c["atom"] != "fma"
            and c["align"] == 0
        )
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
    "1x2",
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
    # The width fold's 1x2 rows over C x s channels (HiFi-GAN 1x16 stride 8): wide channel tiles
    # on the 32 x 32 atoms every CDNA part runs (the gfx950 pack is at its size cap).
    ("mfma32_", ("1x2",)): [
        (128, 128, 4, 2, 64),
        (128, 128, 4, 2, 32),
        (256, 64, 4, 2, 64),
    ],
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
    ("wmma", ("1x1r0",)): [(128, 64, 4, 2, 32), (128, 128, 4, 2, 32)],
    # gfx950 channels-first pointwise on the output rows, wide tiles (fp16, best config before
    # -> after; MIOpen): deeplab c256 k1024 128^2 0.38 -> 0.23 ms (0.38), c2048 k256 65^2 0.44
    # -> 0.22 (0.38), resnext c1024 n128 16^2 0.71 -> 0.22 (0.16).
    ("mfma32x16", ("1x1r0",)): [(256, 128, 4, 2, 32), (128, 128, 2, 2, 64)],
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
# Channels-first ring tiles (loop 1, layout CF only) for dilated 3x3 rows on maps whose width
# a 32- or 64-position chunk splits badly (deeplab ASPP 65^2: 65 = 64 + 1): one chunk of 72 / 80
# positions a row. Measured (fp16 n16, best config against the next): gfx90a 65^2 d12 7.74 against
# 8.70 ms, d36 9.44 / 9.50, c512 d2 4.16 / 4.47; gfx942 MI325X 65^2 d12 1.60 / 1.68, c512 d4
# 0.84 / 0.91; gfx950 65^2 d12 0.96 / 1.24, d24 1.03 / 1.63, c512 d4 0.44 / 0.58.
RING_CF_TILES = {
    ("mfma32_f16", ("3x3d2", "3x3d4", "3x3d12", "3x3d24", "3x3d36")): [
        (64, 64, 2, 2, 72)
    ],
    ("mfma32_bf16", ("3x3d2", "3x3d4", "3x3d12", "3x3d24", "3x3d36")): [
        (64, 64, 2, 2, 72)
    ],
    ("mfma32x16", ("3x3d4", "3x3d12", "3x3d24")): [(128, 64, 4, 2, 80)],
    # Channels-first pointwise on RDNA's ring: wide tiles for a long reduction behind a big
    # output (deeplab 1x1 c1024 k2048 128^2: gfx1100 71.0 -> 23.7 ms, MIOpen 36.0; c256 k1024
    # 5.43 -> 2.97 / 5.04; gfx1201 3.33 -> 2.21 / 2.31).
    ("wmma", ("1x1",)): [
        (128, 64, 4, 2, 64),
        (128, 128, 4, 2, 32),
        (64, 128, 2, 4, 64),
    ],
}
# Dilated 3x3 ring rows on the 16-bit 32 x 32 atoms of gfx908 to gfx942 (xcd 0) also at 4 blocks per
# dispatch unit: gfx942 MI325X 128^2 d24 4.01 against 5.56 ms at 2, d36 4.58 / 4.73, 65^2 d12
# 1.60 / 1.70; gfx90a 65^2 d36 9.44 / 9.50 (bpu only sets the split count: no new program). Both
# layouts.
BPU4_RING_ROWS = {"3x3d2", "3x3d4", "3x3d12", "3x3d24", "3x3d36"}
# Ring stages of wq steps of fewer positions (K3 WQ), for output rows no wider than bq (the
# host admits them only there): the 3-D small-map layers' 16- and 8-wide rows fill the atoms
# instead of padding a 32-position step with zeros. (atom prefix, filter tags) -> [(layout, tile
# (bk, bc, wk, wc, bq), wq)].
STAGE_RING = {
    (("mfma32_f16", "mfma32_bf16"), ("3x3",)): [
        (CF, (64, 64, 2, 2, 16), 2),
        (CL, (64, 64, 2, 2, 16), 2),
        (CF, (64, 64, 2, 2, 8), 4),
        (CL, (64, 64, 2, 2, 8), 4),
    ],
    (("mfma32_f32",), ("3x3",)): [
        (CF, (64, 64, 2, 2, 16), 2),
        (CL, (64, 64, 2, 2, 16), 2),
        (CF, (64, 64, 2, 2, 8), 4),
        (CL, (64, 64, 2, 2, 8), 4),
    ],
    (("mfma32x16_f16", "mfma32x16_bf16"), ("3x3",)): [
        (CF, (64, 64, 2, 2, 16), 2),
        (CL, (64, 64, 2, 2, 16), 2),
    ],
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
    # HiFi-GAN's dilated 1-D rows, in both layouts (channels-last otherwise falls to the
    # generic tier).
    ("1x3d5", 3, 1, 5, 1, 1, [0, -1], [0, 1], None),
    # HiFi-GAN's dilation-3 1x3 (the d5 row's tiles).
    ("1x3d3", 3, 1, 3, 1, 1, [0, -1], [0, 1], None),
    ("1x4", 4, 1, 1, 1, 1, [0, -1], [0], None),
    # Two-tap rows: the width fold's HiFi-GAN class (1x16 stride 8 is 1x2 stride 1 over C x 8
    # channels, widthFoldOf) on the 1x4 row's tiles.
    ("1x2", 2, 1, 1, 1, 1, [0, -1], [0], None),
    ("1x7", 7, 1, 1, 1, 1, [-1], [0], None),
    ("1x7d3", 7, 1, 3, 1, 1, [-1], [0], None),
    ("1x7d5", 7, 1, 5, 1, 1, [-1], [0], None),
    ("1x11", 11, 1, 1, 1, 1, [-1], [0], None),
    ("1x11d3", 11, 1, 3, 1, 1, [-1], [0], None),
    ("1x11d5", 11, 1, 5, 1, 1, [-1], [0], None),
    # Dilated 3x3 rows in both layouts: without channels-last configs a channels-last deeplab
    # ASPP layer (d4 / d24 / d36) has no specialized config, the generic tier declines it (d6+,
    # except on gfx11) and it falls to the BLAS path (30x on gfx1100).
    ("3x3d4", 3, 1, 4, 3, 1, [0, -1], [0], None),
    # SSD300's dilation-6 3x3 (the d4 row's tiles).
    ("3x3d6", 3, 1, 6, 3, 1, [0, -1], [0], None),
    ("3x3d12", 3, 1, 12, 3, 1, [0, -1], [0], None),
    ("3x3d24", 3, 1, 24, 3, 1, [0, -1], [0], None),
    ("3x3d36", 3, 1, 36, 3, 1, [0, -1], [0], None),
    ("3x3s2", 3, 2, 1, 3, 0, [0, 1], [0], None),
    ("3x3s2r1", 3, 2, 1, 1, 0, [0, 1], [0], None),
    ("2x2s2", 2, 2, 1, 2, 0, [0, 1], [0], None),
    # Stride-1 3x3 on the output rows too, plain and dilated (RDNA: deeplab 65^2 and unet2d run
    # fastest there; gfx1100).
    ("3x3r1", 3, 1, 1, 1, 0, [0, 1], [0], None),
    ("3x3d4r1", 3, 1, 4, 1, 0, [0, 1], [0], None),
    ("3x3d6r1", 3, 1, 6, 1, 0, [0, 1], [0], None),
    ("3x3d12r1", 3, 1, 12, 1, 0, [0, 1], [0], None),
    ("3x3d24r1", 3, 1, 24, 1, 0, [0, 1], [0], None),
    ("3x3d36r1", 3, 1, 36, 1, 0, [0, 1], [0], None),
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
    # 5-wide filters (3-D 5x5x5 volumes; EXTRA_TILES): stride 2 on the output
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

# Catalog breadth: geometries the families' tiles and the planner's default blocks per unit
# (with_bpu) miss, each the measured best on a corpus layer (fp16 and bf16, n128 unless noted;
# a geometry sweep around the top configs, compiled from this source and planned by
# wrwLaunchFor). (atoms, filter tag) -> [(layout, tile (bk, bc, wk, wc, bq), bpus)]: the config
# at exactly these bpus, added to the family's (a tile the family has gets the bpus too). Not
# three-bf16 sources (build_x3): measured on 16-bit graphs only.
GFX950_16 = ("mfma32x16_f16", "mfma32x16_bf16")
CDNA32_16 = ("mfma32_f16", "mfma32_bf16")
WMMA11_16 = ("wmma11_f16", "wmma11_bf16")
WMMA12_16 = ("wmma12_f16", "wmma12_bf16")
RDNA_16 = WMMA11_16 + WMMA12_16
BREADTH = {
    # gfx950: resnet50 c512 7^2 144 against 162 us, c512 14^2 s2 170 against 200, c256 k1024
    # 14^2 1x1 116 against 194 (4 blocks a unit), inception 35^2 5x5 110 against 140, c288 k384
    # 35^2 s2 390 against 432.
    (GFX950_16, "3x3pk"): [(CF, (128, 128, 4, 2, 64), (2,))],
    (GFX950_16, "3x3s2pk"): [(CF, (256, 64, 4, 2, 32), (2,))],
    (GFX950_16, "1x1"): [(CF, (64, 128, 2, 2, 32), (4,))],
    (GFX950_16, "5x5r1"): [(CF, (64, 64, 2, 2, 64), (2,))],
    (GFX950_16, "3x3s2ph"): [(CF, (128, 64, 4, 2, 32), (2,))],
    # gfx942 (MI325X): resnet50 c256 k1024 14^2 1x1 158 against 245 us, inception c288 k384
    # 35^2 s2 595 against 862.
    (CDNA32_16, "1x1"): [(CF, (128, 64, 2, 2, 32), (2,))],
    (CDNA32_16, "3x3s2ph"): [(CF, (128, 64, 2, 2, 32), (1,))],
    # Stride-2 3x3 rings (fp16): more waves a block and more blocks a unit (the 128 x 64 two-wave
    # tile left MI325X at three waves a CU). CONV_WRW alone over the corpus's fp16 channels-first
    # stride-2 3x3 layers: 1.06x on MI325X, 1.03x on gfx950.
    (("mfma32x16_f16",), "3x3s2ph"): [
        (CF, (64, 64, 2, 2, 32), (4,)),
        (CF, (64, 32, 2, 1, 32), (2, 4)),
    ],
    (("mfma32_f16",), "3x3s2ph"): [
        (CF, (128, 64, 4, 2, 32), (1, 2)),
        (CF, (64, 64, 2, 2, 32), (4,)),
        (CF, (64, 32, 2, 1, 32), (2, 4)),
    ],
    # gfx1201 (gfx1101 alike where noted): the same layers 656 against 784 us (gfx1101 2111 against
    # 2204), 750 against 986 (2365 against 2927), 477 against 622 (1676 against 2176), 576 against
    # 700 (3214 against 3894), 1888 against 2461 (6472 against 7702); gfx1201 only: inception c80
    # 73^2 2992 against 3339 and c32 147^2 (3x3r1) 2087 against 3037, and channels-last sd unet,
    # flux vae and efficientnet layers 4-12% faster at 4 blocks a unit.
    # 1 x N rows (HiFi-GAN 1 x 3 / 7 / 11 at dilation 1, 3, 5, Inception's 17-wide 1 x 7; fp16),
    # channels-first, CDNA ring: one-fragment waves at 4 blocks a unit and 2 x 2 / 4 x 2-wave
    # tiles. The 32 x 64 tile alone left one wave a SIMD; over the corpus's 152 HiFi-GAN wrw
    # layers (autotuned, CONV_WRW alone) these tiles run 1.43x on gfx950 and 1.16x on MI325X, and
    # Inception's 1 x 7 layers 1.2-1.7x on gfx950.
    **{
        (atoms, tag): [
            (CF, (32, 64, 1, 2, 32), (4,)),
            (CF, (64, 64, 2, 2, 32), (2, 4)),
            (CF, (64, 64, 2, 2, 64), (2,)),
            (CF, (128, 64, 4, 2, 32), (1, 2)),
            (CF, (64, 32, 2, 1, 32), (4,)),
        ]
        for atoms in (("mfma32x16_f16",), ("mfma32_f16",))
        for tag in (
            "1x3",
            "1x3d3",
            "1x3d5",
            "1x7",
            "1x7d3",
            "1x7d5",
            "1x11",
            "1x11d3",
            "1x11d5",
        )
    },
    (RDNA_16, "3x3pk"): [(CF, (128, 32, 4, 2, 64), (2,))],
    (RDNA_16, "3x3s2pk"): [(CF, (128, 32, 4, 2, 32), (2,))],
    (RDNA_16, "1x1"): [(CF, (128, 64, 4, 2, 32), (4,))],
    (RDNA_16, "5x5r1"): [(CF, (64, 64, 4, 2, 64), (4,))],
    (RDNA_16, "3x3s2ph"): [(CF, (128, 32, 4, 2, 32), (4,))],
    (WMMA12_16, "3x3"): [
        (CF, (64, 32, 4, 2, 32), (4,)),
        (CL, (64, 64, 4, 2, 32), (4,)),
    ],
    (WMMA12_16, "3x3r1"): [(CF, (64, 32, 4, 2, 64), (4,))],
}
# INTS keys of the BREADTH configs (build_x3 leaves them out).
BREADTH_KEYS = set()

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
    return dict(c, dws=dws_of(c), pb=0, flat=0, math=0)


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
        flat=0,
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
    # gfx908 bf16 (no bf16_1k atom): the half-rate 32 x 32 atom.
    ({"BFLOAT16": "mfma32_bf16_908"}, [(32, 128, 1, 4, 32), (32, 64, 1, 2, 32)], 0),
]
# 1x7: HiFi-GAN's output layer (one output channel), which has 32 input channels: also tiles 32
# input channels wide (atom family prefix -> tiles), so a block's channel tile is not half empty.
UNALIGNED_K_NARROW = {
    "mfma": [(32, 32, 1, 1, 32), (32, 32, 1, 1, 64)],
    "wmma": [(16, 32, 1, 2, 32), (16, 32, 1, 2, 64)],
}
UNALIGNED_K_FILTERS = [("3x3", 3, 3), ("1x1", 1, 1), ("1x7", 7, 1)]


def build_unaligned_k():
    dropped = 0
    for atoms, tiles, xcd in UNALIGNED_K:
        for dtype, atom in atoms.items():
            for tag, s, fr in UNALIGNED_K_FILTERS:
                extra = UNALIGNED_K_NARROW.get(atom[:4], []) if tag == "1x7" else []
                for t in tiles + [x for x in extra if x not in tiles]:
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
    ({"BFLOAT16": "mfma16_bf16_908"}, 0),  # gfx908 bf16
    ({"HALF": "wmma11_f16", "BFLOAT16": "wmma11_bf16"}, 0),
    ({"HALF": "wmma12_f16", "BFLOAT16": "wmma12_bf16"}, 0),
    ({"HALF": "wmma125_f16", "BFLOAT16": "wmma125_bf16"}, 0),
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
# 5x5 small-C rows (3-D 5x5x5 with 16 input channels): 16-bit, channels-first,
# the tile measured best per atom family (gfx950: 32 x 4 at 64 positions; gfx1201: 64 x 8 at 32).
SMALLC_5X5_TILES = {"mfma": [(32, 4, 2, 64)], "wmma": [(64, 8, 4, 32)]}


# Configs appended after every other legacy config of a dtype (their priorities follow the
# others', which keep theirs): (dtype) -> configs.
LATE = {"FLOAT": [], "HALF": [], "BFLOAT16": []}
# 3-D small-C stems: every (depth, height) row of a 3x3x3 filter a block (fr 9), so a long
# reduction reads dy once instead of once a depth tap.
SMALLC_LATE_FILTERS = [("3x3x3", 3, 1, 9)]
# Ring stages for the 3-D small-map layers' 8- and 16-wide rows on the 32 x 32 16-bit atoms,
# channels-last: tiles of 128 output channels by 64 input ones (wide-channel layers).
STAGE_RING_LATE = {
    ("mfma32_f16", "mfma32_bf16"): [
        ((128, 64, 4, 2, 16), 2),
        ((128, 64, 4, 2, 8), 4),
    ]
}


def build_late():
    """The LATE configs: small-C rows over all (depth, height) filter rows, wide ring stages."""
    dropped = 0
    for atoms, xcd in SMALLC_ATOMS:
        for dtype, atom in atoms.items():
            if not atom.startswith("mfma"):
                continue  # measured on the CDNA atoms
            vec = 16 // eb(dtype)
            for tag, s, sw, fr in SMALLC_LATE_FILTERS:
                for lay in (CF, CL):
                    tiles = SMALLC_TILES + (SMALLC_TILES_CL if lay == CL else [])
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
                            LATE[dtype].extend(with_bpu(c))
                        else:
                            dropped += 1
    for atoms, extra in STAGE_RING_LATE.items():
        for atom in atoms:
            dtype = "HALF" if atom.endswith("f16") else "BFLOAT16"
            for t, wq in extra:
                c = dict(C(atom, 1, CL, t, 3, 1, 1, 3, 8, 0, 0, 0), wq=wq)
                if valid(c, dtype):
                    LATE[dtype].extend(with_bpu(c))
                else:
                    dropped += 1
    return dropped


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
# xcd, [(layout, (bk, bc, wk, wc, bq), taps a block[, shared-window dw])][, the tiles with a
# block-based buffer twin, default all]), the measured best per arch: gfx950 on
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
    # gfx908 bf16 (no bf16_1k atom): the half-rate 32 x 32 atom, the 32 x 32 tiles.
    (
        {"BFLOAT16": "mfma32_bf16_908"},
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


# gfx11 channels-last rows that the families' channels-first-only filter rows leave out (after
# every other specialized config, so those keep their priorities): Inception's 5x5 on the output
# rows (channels-first gfx11 runs 5x5 there, 3.4x ahead of the ring) on the channels-first tiles,
# and the 3x3 ring on gfx12's 32 x 32 tile for 32-channel groups (ResNeXt / RegNet grouped
# layers). (tag, s, sw, dw, fr, loop, tiles).
GFX11_CL_ATOMS = {"HALF": "wmma11_f16", "BFLOAT16": "wmma11_bf16"}
GFX11_CL_ROWS = [
    ("5x5r1", 5, 1, 1, 1, 0, [RDNA16[1], (64, 64, 4, 2, 64)]),
    ("3x3g32", 3, 1, 1, 3, 1, [(32, 32, 2, 2, 32)]),
]


def build_gfx11_cl():
    dropped = 0
    for dtype, atom in GFX11_CL_ATOMS.items():
        vec = 16 // eb(dtype)
        have = {tuple(x[f] for f in INTS) for x in CATALOG[dtype]}
        for tag, s, sw, dw, fr, loop, tiles in GFX11_CL_ROWS:
            for t in tiles:
                c = C(atom, loop, CL, t, s, sw, dw, fr, vec, 0, 0, 0)
                if not valid(c, dtype):
                    dropped += 1
                    continue
                for v in with_bpu(c):
                    assert tuple(v[f] for f in INTS) not in have, f"{tag} {t} ships"
                    CATALOG[dtype].append(v)
    return dropped


# Flattened positions (flat 1): steps of bq consecutive positions of the flattened (image, depth,
# row, column) order, for output rows shorter than bq (small 2-D and 3-D maps with wide channels:
# GEMM-shaped wrw). Twins of the generic configs without a shared window, plus wider tiles: one
# or two taps of 128 x 128 channels a block.
GENERIC_FLAT = [
    (
        {"HALF": "mfma32x16_f16", "BFLOAT16": "mfma32x16_bf16"},
        1,
        [(CL, (128, 128, 2, 2, 32), 2), (CL, (128, 128, 2, 2, 32), 3)],
    ),
    (
        {"HALF": "mfma32_f16", "BFLOAT16": "mfma32_bf16"},
        0,
        [
            (CL, (128, 128, 2, 2, 32), 1),
            (CL, (128, 128, 2, 2, 32), 2),
            (CL, (128, 128, 4, 2, 32), 2),
        ],
    ),
]


# gfx11's generic tier, built after GENERIC_FLAT (its own entries: the tiles, then the tiles
# with a block-based buffer twin).
GENERIC_GFX11 = [
    # gfx11 (wave32, 16 x 16 x 16 WMMA), chosen by a greedy cover of 19 candidate tiles timed on
    # gfx1101 over the corpus wrw layers BLAS serves there and the specialized classes' layers.
    # Channels-first elements: 128 x 64 at 2 taps for wide channel counts, 64 x 64 at 3, 64 x 64
    # on 4 waves at 1 tap for short maps, 32 x 32 for few output channels (HiFi-GAN's 1-channel
    # outputs, 1.34x the BLAS path where 64 x 64 gives 1.15x). Channel vectors: 64 x 32 at 9 taps
    # and the shared windows for unit width stride. Within 1.2% of all 19 (geomean) on
    # channels-first layers and 7% on channels-last. Block-based buffer twins on one tile a
    # layout only (images past 2 GiB; listed last, the "pb" set); flattened-position twins on
    # every tile without a shared window, as in GENERIC.
    (
        {"HALF": "wmma11_f16", "BFLOAT16": "wmma11_bf16"},
        0,
        [
            (CF, (128, 64, 4, 2, 32), 2),
            (CF, (64, 64, 4, 2, 32), 3),
            (CF, (64, 64, 2, 2, 32), 1),
            (CF, (32, 32, 2, 2, 32), 3),
            (CL, (64, 32, 4, 2, 32), 9),
            (CL, (128, 64, 4, 2, 32), 5, 16),
            (CL, (64, 64, 4, 2, 32), 9, 8),
        ],
        {0, 4},
    ),
]


def build_generic():
    dropped = 0

    def add(entries):
        dropped = 0
        for atoms, xcd, tiles, *pb_tiles in entries:
            for dtype, atom in atoms.items():
                for t, (lay, tile, taps, *window) in enumerate(tiles):
                    wx = window[0] if window else 0  # dw: shared-window positions
                    c = C(
                        atom, 8, lay, tile, 0, 0, wx, taps, 16 // eb(dtype), xcd, 0, 0
                    )
                    if valid(c, dtype):
                        CATALOG[dtype].extend(with_bpu(c))
                        # Its block-based buffer twin: images past the 32-bit buffer range
                        # (every tile, or the entry's "pb" set).
                        if not pb_tiles or t in pb_tiles[0]:
                            CATALOG[dtype].extend(with_bpu(dict(c, pb=1)))
                        if not wx:
                            CATALOG[dtype].extend(with_bpu(dict(c, flat=1)))
                    else:
                        dropped += 1
        return dropped

    dropped += add(GENERIC)
    for atoms, xcd, tiles in GENERIC_FLAT:
        for dtype, atom in atoms.items():
            for lay, tile, taps in tiles:
                c = C(atom, 8, lay, tile, 0, 0, 0, taps, 16 // eb(dtype), xcd, 0, 0)
                if valid(c, dtype):
                    CATALOG[dtype].extend(with_bpu(dict(c, flat=1)))
                else:
                    dropped += 1
    dropped += add(GENERIC_GFX11)
    return dropped


DILATED_ROWS_CF_1250 = {
    "1x3d3",
    "1x3d5",
    "1x7d3",
    "1x7d5",
    "1x11d3",
    "1x11d5",
    "3x3d4",
    "3x3d6",
    "3x3d24",
    "3x3d36",
    "3x3d4r1",
    "3x3d6r1",
    "3x3d24r1",
    "3x3d36r1",
}


def build_native1250():
    # Native matrix atoms: retain window-reuse loops for common filters, while loop 11 covers
    # every runtime filter/stride/dilation and channel tail without vector-alignment gates.
    dropped = 0
    for dtype, atom in (
        ("HALF", "wmma125_f16"),
        ("BFLOAT16", "wmma125_bf16"),
        ("FLOAT", "wmma125_f32"),
    ):
        tiles = [(32, 32, 2, 2, 64), (64, 64, 2, 2, 64), (64, 32, 4, 2, 64)]
        # The dilated rows ship channels-first only here: channels-last dilated problems run
        # loop 11, except SSD300's dilation-6 rows, which take both layouts below.
        native_filters = [
            row if row[0] not in DILATED_ROWS_CF_1250 else row[:8] + ([CF],)
            for row in FILTERS
        ] + [
            ("3x3d6", 3, 1, 6, 3, 1, [0, 1], [0], [CF, CL]),
            ("3x3d6r1", 3, 1, 6, 1, 0, [0, 1], [0], [CF, CL]),
        ]
        for _, s, sw, dw, fr, loop, idx, aligns, lays in native_filters:
            picked = (
                tiles
                if idx is None
                else list(
                    dict.fromkeys(
                        tiles[i] for i in idx if -len(tiles) <= i < len(tiles)
                    )
                )
            )
            for tile in picked:
                for layout in ([CF, CL] if lays is None else lays):
                    if layout not in (CF, CL):
                        continue
                    for align in (aligns if layout == CF and loop == 1 else [0]):
                        c = C(
                            atom,
                            loop,
                            layout,
                            tile,
                            s,
                            sw,
                            dw,
                            fr,
                            16 // eb(dtype),
                            0,
                            align,
                            0,
                        )
                        if valid(c, dtype):
                            CATALOG[dtype].extend(with_bpu(c))
                        else:
                            dropped += 1
        generic_tiles = [
            (16, 16, 1, 1, 32),
            (32, 32, 2, 2, 64),
            (64, 64, 2, 2, 64),
            (64, 32, 4, 2, 128),
            (32, 64, 2, 4, 128),
            (64, 128, 2, 4, 64),
        ]
        for tile in generic_tiles:
            for layout in (CF, CL):
                for bpu in (2, 4):
                    c = C(atom, 11, layout, tile, 0, 0, 0, 0, 1, 0, 0, 0)
                    c.update(dws=1, bpu=bpu)
                    if valid(c, dtype):
                        CATALOG[dtype].append(c)
                    else:
                        dropped += 1
        # A bounded candidate set, not an exhaustive product. BC packs small-C taps into
        # useful matrix columns; BQ explores amortizing metadata and LDS staging. These are
        # measurement candidates, not claimed winning tiles.
        tap_tiles = [
            (32, 32, 2, 2, 64),
            (32, 64, 2, 2, 64),
            (64, 32, 2, 2, 64),
            (64, 64, 2, 2, 64),
            (32, 128, 2, 4, 64),
            (64, 128, 2, 4, 64),
            (32, 64, 2, 2, 128),
            (64, 64, 2, 2, 128),
            (64, 128, 2, 4, 128),
            (64, 64, 2, 2, 256),
            (32, 128, 2, 4, 256),
        ]
        for tile in tap_tiles:
            for layout in (CF, CL):
                for bpu in (2, 4):
                    c = C(atom, 12, layout, tile, 0, 0, 0, 0, 1, 0, 0, 0)
                    c.update(dws=1, bpu=bpu)
                    if valid(c, dtype):
                        CATALOG[dtype].append(c)
                    else:
                        dropped += 1
        for bc in (64, 128, 256):
            for bq in (64, 128, 256):
                for bpu in (2, 4):
                    c = C("fma", 9, CL, (4, bc, 4, 1, bq), 0, 0, 0, 0, 1, 0, 0, 0)
                    c.update(dws=1, bpu=bpu)
                    if valid(c, dtype):
                        CATALOG[dtype].append(c)
                    else:
                        dropped += 1
    return dropped


def build_dilation_rows():
    # Append only the two measured NCHW row windows, preserving all old identities and order.
    for bq in (32, 128):
        c = C("wmma125_f16", 0, CF, (64, 64, 2, 2, bq), 3, 1, 24, 1, 8, 0, 0, 0)
        c.update(dws=1, bpu=2)
        assert valid(c, "HALF")
        CATALOG["HALF"].append(c)


def build_accurate():
    # Small independent measurement candidates; all use the same FP64 arithmetic contract.
    for bk, bc, bq in ((8, 8, 32), (16, 16, 32), (16, 16, 64)):
        for layout in (CF, CL):
            for bpu in (2, 4):
                c = C("fma", 10, layout, (bk, bc, 1, 1, bq), 0, 0, 0, 0, 1, 0, 0, 0)
                c.update(dws=0, bpu=bpu)
                assert valid(c, "FLOAT")
                CATALOG["FLOAT"].append(c)


def build_family_tiles():
    # Append after accuracy/dilation entries without changing their fallback priorities.
    dropped = 0
    for dtype, atom in (("HALF", "wmma125_f16"), ("BFLOAT16", "wmma125_bf16")):
        # One wave packs small input-channel groups across the three width taps.
        for sw in (1, 2):
            c = C(atom, 6, CF, (16, 8, 1, 1, 32), 3, sw, 1, 3, 8, 0, 0, 0)
            if valid(c, dtype):
                CATALOG[dtype].extend(with_bpu(c))
            else:
                dropped += 1
        # Longer pointwise ring chunks reuse each staged row across channel tiles.
        for layout in (CF, CL):
            for bq in (128, 256):
                c = C(atom, 1, layout, (64, 64, 2, 2, bq), 1, 1, 1, 1, 8, 0, 0, 0)
                if valid(c, dtype):
                    CATALOG[dtype].extend(
                        dict(c, bpu=b) for b in ((2, 4) if bq == 128 else (2,))
                    )
                else:
                    dropped += 1
    return dropped


SPEC_KIND = {0: "R", 6: "C", 1: "P", 2: "G", 5: "D"}
SPEC_MAX_ATOM_ID = ATOMS.index("fma")


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
        if c["atom_id"] > SPEC_MAX_ATOM_ID:
            continue
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
                if loop == 1 and CF in layouts and (lays is None or CF in lays):
                    for (prefix, tags), extra in RING_CF_TILES.items():
                        if atom.startswith(prefix) and tag in tags:
                            for t in extra:
                                if t not in picked:
                                    cands.append(
                                        C(atom, 1, CF, t, s, sw, dw, fr, vec, xcd, 0, 0)
                                    )
                if loop == 1:
                    for (stage_atoms, tags), extra in STAGE_RING.items():
                        if atom in stage_atoms and tag in tags:
                            for lay, t, wq in extra:
                                if lay in layouts and (lays is None or lay in lays):
                                    cands.append(
                                        dict(
                                            C(
                                                atom,
                                                1,
                                                lay,
                                                t,
                                                s,
                                                sw,
                                                dw,
                                                fr,
                                                vec,
                                                xcd,
                                                0,
                                                0,
                                            ),
                                            wq=wq,
                                        )
                                    )
                if tag == "1x1" and TR in layouts:
                    for t in BIG1x1:
                        cands.append(C(atom, 1, TR, t, s, sw, dw, fr, vec, xcd, 0, 0))
                        cands.append(
                            C(atom, 2, TR, t, s, sw, dw, fr, vec, 1, 0, LOOP2_NS)
                        )
                for c in cands:
                    if valid(c, dtype):
                        variants = with_bpu(c)
                        if (
                            tag in BPU4_RING_ROWS
                            and c["loop"] == 1
                            and c["layout"] in (CF, CL)
                            and c["atom"] in ("mfma32_f16", "mfma32_bf16")
                            and not any(v["bpu"] == 4 for v in variants)
                        ):
                            variants = variants + [dict(c, bpu=4)]
                        # RDNA channels-first pointwise: also one block per dispatch unit
                        # (gfx1100 deeplab c256 k1024 128^2: 2.97 ms at 1 against 3.87 at 2).
                        if (
                            tag in ("1x1", "1x1r0")
                            and c["layout"] == CF
                            and c["atom"].startswith("wmma")
                            and not any(v["bpu"] == 1 for v in variants)
                        ):
                            variants = variants + [dict(c, bpu=1)]
                        CATALOG[dtype].extend(variants)
                    else:
                        dropped += 1
                for (atoms_of, btag), extra in BREADTH.items():
                    if atom not in atoms_of or btag != tag:
                        continue
                    for lay, t, bpus in extra:
                        if lay not in layouts or (lays is not None and lay not in lays):
                            continue
                        c = C(atom, loop, lay, t, s, sw, dw, fr, vec, xcd, 0, 0)
                        assert valid(
                            c, dtype
                        ), f"breadth config {atom} {tag} {t} invalid"
                        have = {tuple(x[f] for f in INTS) for x in CATALOG[dtype]}
                        for b in bpus:
                            key = tuple(dict(c, bpu=b)[f] for f in INTS)
                            assert (
                                key not in have
                            ), f"breadth config {atom} {tag} {t} b{b} ships"
                            CATALOG[dtype].append(dict(c, bpu=b))
                            BREADTH_KEYS.add(key)
    dropped += build_gfx11_cl()
    # The generic tier last: the specialized configs keep their priorities.
    dropped += build_generic()
    dropped += build_native1250()
    build_accurate()
    build_dilation_rows()
    dropped += build_family_tiles()
    c = C("wmma125_f16", 12, CF, (64, 64, 2, 2, 32), 0, 0, 0, 0, 1, 0, 0, 0)
    c.update(dws=2, bpu=1)
    assert valid(c, "HALF")
    CATALOG["HALF"].append(c)
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


# Three-bf16 (math 1, the CONV_WRW_TF32 engine's configs): fp32 graphs run a bf16 config on x and dy split
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
    "wmma125_bf16",
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
        if tuple(c[f] for f in INTS) in BREADTH_KEYS:
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
        c["atom"].startswith("wmma12_")
        and (c["bk"] // c["wk"]) * (c["bc"] // c["wc"]) >= 512
    ):
        return 0
    return 1


def family_of(c):
    """The pack a config ships in: the direct loop, gfx950's 16 x 32 / 32 x 16 atoms (3-wide
    filters in one pack, every other width in a second, each under the 500 KB file cap), the
    other MFMA atoms, RDNA's WMMA per generation (gfx11, gfx12), gfx1250's native packs, or the
    FMA loops. The 32 x 32 atom's packed rows ship in a pack without gfx950, which has its own
    measured packed-rows tiles.
    """
    if c["loop"] == 10:
        return "accurate"
    if c["loop"] == 9:
        return "smallk1250"
    if c["loop"] == 12:
        return "taptiled1250"
    if c["atom"].startswith("wmma125_"):
        return "generic1250" if c["loop"] == 11 else "gfx1250"
    if c["loop"] == 5:
        return "direct"
    if c["loop"] == 8:
        return "genericfma" if c["atom"] == "fma" else "generic"
    if c["loop"] == 7 and c["atom"].startswith("mfma32_"):
        return "cdnapk"
    if c["atom"] == "fma":
        return "fma"
    if c["atom"].startswith("wmma11"):
        return "rdna11"
    if c["atom"].startswith("wmma12"):
        return "rdna12"
    if c["atom"].startswith(("mfma16x32", "mfma32x16")):
        return "gfx950" if c["s"] == 3 else "gfx950b"
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
            # RFC 0025 numerical notes. No "nondeterministic": every config is
            # bitwise reproducible (fixed-order sums, no atomics), which the
            # family's device test checks by relaunching each config.
            "numerical_notes": [],
        },
    )
    # The three-bf16 configs make an engine of their own, which reports the TENSOR_CORE numerical
    # note (cuDNN's note for TF32 engines), so a caller can exclude them by note.
    ued_tf32 = U("ued.tf32")
    write(
        "conv_wrw_tf32.ued.json",
        {
            "version": "1.0",
            "id": ued_tf32,
            "name": "hipkernel:CONV_WRW_TF32",
            "graph_match": {"native": "hipkernel.conv_wrw.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
            # RFC 0025 numerical notes: TENSOR_CORE (fp32 through bf16 operand parts).
            # Deterministic as the exact engine.
            "numerical_notes": ["tensor_core"],
        },
    )
    # One pack per dtype and kernel family (the repository caps a file at 500 KB); all serve the
    # one engine but the three-bf16 packs, which serve CONV_WRW_TF32. Priorities run over the
    # dtype's whole list, as before the split.
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
            [(c, family_of(c)) for c in cfgs if c["loop"] not in (8, 10)]
            + [
                (c, "fma906" if j < half906 else "fma906b")
                for j, c in enumerate(CATALOG906[dtype])
            ]
            + [(c, "fmardna") for c in CATALOGRDNA.get(dtype, [])]
            # Three-bf16 in packs of their own (CONV_WRW_TF32's), ahead of the generic tier.
            + [
                (c, "x3" + family_of(c))
                for c in (CATALOGX3 if dtype == "FLOAT" else [])
            ]
            + [(c, family_of(c)) for c in cfgs if c["loop"] == 8]
            + [(c, family_of(c)) for c in LATE[dtype]]
            + [(c, "accurate") for c in cfgs if c["loop"] == 10]
        )
        legacy_i = native_i = accurate_i = 0
        for c, fam in tagged:
            native = c["atom"].startswith("wmma125_") or c["loop"] == 9
            accurate = c["loop"] == 10
            priority = (
                -10000 - accurate_i
                if accurate
                else 2000 - native_i if native else 1000 - legacy_i
            )
            accurate_i += int(accurate)
            native_i += int(native)
            legacy_i += int(not native and not accurate)
            pack = packs.setdefault(fam, [])
            lay = ["nchw", "nhwc", "nhwctr"][c["layout"]]
            thr = (
                f"t{c['tk']}x{c['tc']}"
                if c["atom"] == "fma"
                else f"w{c['wk']}x{c['wc']}"
            )
            if c["loop"] == 5 or (c["loop"] == 1 and c["wq"]):
                thr += f"q{c['wq']}"
            name = (
                f"conv_wrw.{dt}{'x3' if c['math'] else ''}.{c['atom']}.l{c['loop']}.{lay}.{c['bk']}x{c['bc']}x{c['bq']}"
                f".{thr}.fr{c['fr']}s{c['s']}sw{c['sw']}dw{c['dw']}"
                f".v{c['vec']}.x{c['xcd']}a{c['align']}n{c['ns']}b{c['bpu']}"
                + (".pb" if c["pb"] else "")
                + (".flat" if c["flat"] else "")
            )
            if native:
                name += f".dws{c['dws']}"
            pack.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": (
                            "kernels/ConvWrwAccurate.cpp"
                            if accurate
                            else "kernels/ConvWrw.cpp"
                        ),
                        "entry_point": "ConvWrwAccurate" if accurate else "ConvWrw",
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
                    "priority": priority,
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
                        "gfx1250": ["gfx1250"],
                        "generic1250": ["gfx1250"],
                        "taptiled1250": ["gfx1250"],
                        "smallk1250": ["gfx1250"],
                        "accurate": ["gfx1250"],
                        "x3gfx1250": ["gfx1250"],
                    }.get(fam, []),
                    "matchers": [umd],
                    "engine": ued_tf32 if fam.startswith("x3") else ued,
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
    dropped += build_late()
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
