# Generates the CONV_DIRECT_FWD / CONV_DIRECT_BWD descriptors (stride-1 halo-tile direct
# convolution, K2). Each config is admitted on correctness alone (its atom on the device, its
# filter, its channel vector dividing the reduction channels) plus its ship list: the arch groups
# whose autotuned sweeps it won (~/old-asic-hipdnn/k2/STATUS.md, round 2). FMA configs ship
# everywhere as the floor. Autotune (UHD later) ranks the admitted configs; priority orders the
# declared fallback: the arch's measured winners first, FMA last.
#
# Every config-distinguishing field is an integer knob (hipDNN's candidate enumeration tells
# configs apart by their integer knobs only); strings (direction, dtype, kind, atom, archs) are
# descriptive and checked by the kernel match.
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("2f8e3c6a-61b4-4d0e-9a57-c3d81e0b5a42")
U = lambda name: str(uuid.uuid5(NS, "direct." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


ATOMS = [
    "fma",
    "mfma16_f16",
    "mfma32_f16",
    "mfma16_bf16",
    "mfma32_bf16",
    "mfma16_bf16_908",
    "mfma32_bf16_908",
    "mfma16_f32",
    "mfma32_f32",
    "mfma16x32_f16",
    "mfma16x32_bf16",
    "mfma32x16_f16",
    "mfma32x16_bf16",
    "wmma11_f16",
    "wmma11_bf16",
    "wmma12_f16",
    "wmma12_bf16",
]
INTS = [
    "atom_id",
    "math",
    "kd",
    "r",
    "s",
    "bm",
    "bn",
    "bc",
    "wm",
    "wn",
    "vec",
    "ring",
    "bl",
    "split",
    "tm",
    "tn",
    "nt",
    "threads",
    "pipe",
    "rows",
    "pw",
    "fn",
    "wp",
    "wq",
    "wk",
    "pp",
    "epi",
]
ROWS0 = dict(rows=0, pw=0, fn=0, wp=0, wq=0, wk=0, pp=0, epi=0)
STRINGS = ["direction", "dtype", "kind", "atom", "archs"]
F33, F55, F77, F17, F71, F333 = (
    (1, 3, 3),
    (1, 5, 5),
    (1, 7, 7),
    (1, 1, 7),
    (1, 7, 1),
    (3, 3, 3),
)
FILTERS2D = [F33, F55, F77, F17, F71]


def M(atom, f, bm, bn, bc, wm, wn, vec, ring=1, bl=0, split=0, math=0):
    return dict(
        kind="mma",
        atom=atom,
        atom_id=ATOMS.index(atom),
        math=math,
        kd=f[0],
        r=f[1],
        s=f[2],
        bm=bm,
        bn=bn,
        bc=bc,
        wm=wm,
        wn=wn,
        vec=vec,
        ring=ring,
        bl=bl,
        split=split,
        tm=0,
        tn=0,
        nt=0,
        threads=0,
        pipe=0,
        **ROWS0,
    )


def F(f, bc, tm, tn, nt, vec, ring=1, pipe=0):
    return dict(
        kind="fma",
        atom="fma",
        atom_id=0,
        math=0,
        kd=f[0],
        r=f[1],
        s=f[2],
        bm=(256 // nt) * tm,
        bn=nt * tn,
        bc=bc,
        wm=0,
        wn=0,
        vec=vec,
        ring=ring,
        bl=0,
        split=0,
        tm=tm,
        tn=tn,
        nt=nt,
        threads=256,
        pipe=pipe,
        **ROWS0,
    )


def RW(atom, f, pw, fn, wp, wq, wk, bc, vec, pp=0, epi=1, math=0):
    """Filter-row reuse kernel (ConvDirectRows.h): LDS DMA (16-byte on gfx950, dword elsewhere),
    16x16 MFMA atoms; 2-D, or 3-D (each output plane walks its depth taps' input planes).
    """
    return dict(
        kind="rows",
        atom=atom,
        atom_id=ATOMS.index(atom),
        math=math,
        kd=f[0],
        r=f[1],
        s=f[2],
        bm=wp * pw,
        bn=wk * fn * 16,
        bc=bc,
        wm=0,
        wn=0,
        vec=vec,
        ring=0,
        bl=0,
        split=0,
        tm=0,
        tn=0,
        nt=0,
        threads=0,
        pipe=0,
        rows=1,
        pw=pw,
        fn=fn,
        wp=wp,
        wq=wq,
        wk=wk,
        pp=pp,
        epi=epi,
    )


def nr(f):
    """3-D configs here stream without the ring unless asked."""
    return 0 if f[0] > 1 else 1


# catalog[dtype] = list of (archs, config); archs is a set of groups.
catalog = {"HALF": [], "BFLOAT16": [], "FLOAT": []}


def add(dtype, archs, cfgs):
    for c in cfgs:
        catalog[dtype].append((set(archs), c))


def x16(dt):
    return {"HALF": "f16", "BFLOAT16": "bf16"}[dt]


# Rows tiles for the 64 KiB MFMA parts (dword LDS DMA): the ping-pong tile won every shape of
# 256+ channels on gfx90a (1.04-1.12x over the halo tile); the two PP0 tiles take 192-384.
ROWS_GFX9 = ["gfx90a"]

for dt in ("HALF", "BFLOAT16"):
    e = x16(dt)
    add(
        dt,
        ROWS_GFX9,
        [
            RW(f"mfma16_{e}", F33, 8, 4, 2, 2, 2, 16, 8, pp=1, epi=0),
            RW(f"mfma16_{e}", F33, 4, 4, 2, 1, 2, 32, 8),
            RW(f"mfma16_{e}", F33, 8, 2, 2, 2, 2, 16, 8),
            # 3x3x3: the 96-wide ping-pong tile won Wan VAE's 96-channel layers (1.07-1.14x over
            # the halo tile), pw4 fn4 its 384-channel 3x62x106 (1.08x); 192+ wide stay on K2.
            RW(f"mfma16_{e}", F333, 8, 3, 2, 2, 2, 16, 8, pp=1, epi=0),
            RW(f"mfma16_{e}", F333, 4, 4, 2, 2, 2, 16, 8, pp=1, epi=0),
        ],
    )
    # gfx942
    add(
        dt,
        ["gfx942"],
        [
            M(f"mfma16_{e}", F33, 128, 64, 32, 2, 2, 8),
            M(f"mfma16_{e}", F33, 128, 64, 16, 2, 2, 8),
            M(f"mfma16_{e}", F33, 128, 128, 16, 2, 2, 8),
            M(f"mfma16_{e}", F33, 128, 128, 32, 2, 2, 8),
            M(f"mfma16_{e}", F333, 128, 128, 16, 2, 2, 8, ring=0),
            M(f"mfma16_{e}", F333, 128, 64, 16, 2, 2, 8, ring=0),
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=0),
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=1),
        ],
    )
    # gfx950: the rows kernel (LDS DMA), then the halo-tile configs
    add(
        dt,
        ["gfx950"],
        [
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 64, 8, pp=1, epi=0),
            RW(f"mfma16x32_{e}", F33, 8, 2, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 8, 1, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 4, 4, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 1, 2, 32, 8),
            # 48-wide tiles: images whose width is a multiple of 48 (sdxl 48^2 / 96^2)
            RW(f"mfma16x32_{e}", F33, 8, 2, 2, 3, 2, 32, 8),
            RW(f"mfma16x32_{e}", F33, 8, 4, 1, 3, 2, 64, 8),
            RW(f"mfma16x32_{e}", F55, 8, 2, 2, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F77, 8, 2, 2, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F17, 8, 4, 2, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F71, 8, 2, 2, 2, 2, 32, 8),
            # 3x3x3 (Wan VAE, nnunet bf16): 1.03-1.79x over the halo tile, fwd and bwd.
            RW(f"mfma16x32_{e}", F333, 8, 2, 2, 3, 2, 32, 8),
            RW(f"mfma16x32_{e}", F333, 8, 2, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F333, 8, 4, 2, 2, 2, 64, 8),
        ],
    )
    add(
        dt,
        ["gfx950"],
        [
            M(f"mfma16x32_{e}", F33, 128, 64, 32, 2, 2, 8),
            M(f"mfma32x16_{e}", F33, 128, 128, 32, 2, 2, 8),
            M(f"mfma16_{e}", F33, 128, 64, 16, 2, 2, 8),
            M(f"mfma16_{e}", F33, 128, 128, 32, 2, 2, 8),
            M(f"mfma16x32_{e}", F333, 128, 128, 32, 2, 2, 8, ring=0),
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=0),
            M(f"mfma32_{e}", F333, 128, 128, 32, 2, 2, 8, ring=0),
        ],
    )
    # gfx90a
    add(
        dt,
        ["gfx90a"],
        [
            M(f"mfma32_{e}", F33, 64, 128, 32, 1, 4, 8),
            M(f"mfma16_{e}", F33, 128, 64, 32, 2, 2, 8),
            M(f"mfma16_{e}", F33, 128, 128, 32, 2, 2, 8),
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=0),
            M(f"mfma16_{e}", F333, 128, 128, 16, 2, 2, 8, ring=0),
        ],
    )
    if dt == "BFLOAT16":
        # 192-wide K blocks (2 x 4 waves of 64 x 48, 16-channel steps): K = 192 and 384 layers
        # (wan VAE) stage each input halo once per 192 output channels at 4 waves per SIMD.
        add(
            dt,
            ["gfx90a"],
            [
                M(f"mfma16_{e}", F333, 128, 192, 16, 2, 4, 8, ring=0),
                M(f"mfma16_{e}", F33, 128, 192, 16, 2, 4, 8),
            ],
        )
    # gfx11 / gfx12
    for fam in ("11", "12"):
        a = f"wmma{fam}_{e}"
        cfgs = [
            M(a, F33, 128, 128, 32, 2, 4, 8),
            M(a, F33, 128, 64, 32, 4, 2, 8),
            M(a, F33, 128, 128, 16, 2, 4, 8),
            M(a, F333, 128, 128, 32, 2, 4, 8, ring=0),
            M(a, F333, 128, 128, 16, 2, 4, 8, ring=0),
        ]
        if fam == "12":
            cfgs += [
                M(a, F33, 128, 64, 16, 4, 2, 8),
                M(a, F333, 128, 64, 32, 4, 2, 8, ring=1),
                M(a, F333, 64, 64, 32, 2, 2, 8, ring=1),
            ]
        # 96-wide K blocks (4 x 2 waves of 32 x 48): K = 96 layers (wan VAE) fill them.
        cfgs += [
            M(a, F333, 128, 96, 32, 4, 2, 8, ring=0),
            M(a, F33, 128, 96, 32, 4, 2, 8),
        ]
        # 16- and 32-wide K blocks (8 x 1 waves): K <= 16 output layers (VAE / ESRGAN to RGB) and
        # K = 32 layers; the input streams once and the narrow block keeps the MMAs busy.
        cfgs += [
            M(a, F33, 128, 16, 32, 8, 1, 8),
            M(a, F333, 128, 16, 32, 8, 1, 8, ring=0),
            M(a, F333, 128, 32, 32, 8, 1, 8, ring=0),
            M(a, F33, 128, 32, 32, 8, 1, 8),
        ]
        add(dt, [f"gfx{fam}"], cfgs)
        # The other filters, one config each.
        add(
            dt,
            [f"gfx{fam}"],
            [
                M(a, f, 128, 64, 16 if f in (F55, F77) else 32, 4, 2, 8)
                for f in (F55, F77, F17, F71)
            ],
        )
    # CDNA, other filters
    cdna = ["gfx90a", "gfx942", "gfx950"]
    # 96-, 32- and 16-wide K blocks for K = 96, 32 and <= 16 layers.
    add(
        dt,
        cdna,
        [
            M(f"mfma16_{e}", F333, 128, 96, 32, 4, 2, 8, ring=0),
            M(f"mfma16_{e}", F33, 128, 96, 32, 4, 2, 8),
            M(f"mfma16_{e}", F333, 128, 32, 32, 4, 1, 8, ring=0),
            M(f"mfma16_{e}", F33, 128, 32, 32, 4, 1, 8),
            M(f"mfma16_{e}", F333, 128, 16, 32, 4, 1, 8, ring=0),
            M(f"mfma16_{e}", F33, 128, 16, 32, 4, 1, 8),
        ],
    )
    add(
        dt,
        cdna,
        [
            M(f"mfma16_{e}", f, 128, 64, 16 if f in (F55, F77) else 32, 2, 2, 8)
            for f in (F55, F77, F17, F71)
        ],
    )

# gfx908: f16 on the full-rate atoms, bf16 on the half-rate 908 forms.
add(
    "HALF",
    ["gfx908"],
    [
        M("mfma16_f16", F33, 128, 64, 32, 2, 2, 8),
        M("mfma16_f16", F33, 128, 128, 32, 2, 2, 8),
        M("mfma32_f16", F33, 64, 128, 32, 1, 4, 8),
        M("mfma16_f16", F333, 128, 64, 16, 2, 2, 8, ring=0),
        M("mfma16_f16", F333, 128, 128, 16, 2, 2, 8, ring=0),
    ]
    + [
        M("mfma16_f16", f, 128, 64, 16 if f in (F55, F77) else 32, 2, 2, 8)
        for f in (F55, F77, F17, F71)
    ],
)
add(
    "BFLOAT16",
    ["gfx908"],
    [
        M("mfma32_bf16_908", F33, 64, 128, 32, 1, 4, 8),
        M("mfma32_bf16_908", F33, 128, 128, 32, 2, 2, 8, bl=1),
        M("mfma16_bf16_908", F33, 128, 64, 32, 2, 2, 8, bl=1),
        M("mfma32_bf16_908", F333, 128, 128, 32, 2, 2, 8, ring=0, bl=1),
    ]
    + [
        M("mfma16_bf16_908", f, 128, 64, 16 if f in (F55, F77) else 32, 2, 2, 8)
        for f in (F55, F77, F17, F71)
    ],
)

# Exact fp32
add(
    "FLOAT",
    ["gfx942"],
    [
        M("mfma16_f32", F33, 128, 128, 16, 2, 2, 4, bl=1),
        M("mfma16_f32", F33, 128, 64, 16, 2, 2, 4, bl=1),
        M("mfma16_f32", F33, 128, 64, 8, 2, 2, 4),
        M("mfma16_f32", F333, 128, 64, 16, 2, 2, 4, ring=0),
        M("mfma16_f32", F333, 128, 128, 16, 2, 2, 4, ring=0),
    ],
)
add(
    "FLOAT",
    ["gfx950"],
    [
        RW("mfma16_f32", F33, 8, 2, 2, 2, 2, 32, 4, epi=0),
        RW("mfma16_f32", F33, 8, 1, 2, 2, 2, 32, 4, epi=0),
        RW("mfma16_f32", F55, 8, 2, 2, 2, 2, 16, 4, epi=0),
        RW("mfma16_f32", F77, 8, 2, 2, 2, 2, 16, 4, epi=0),
        RW("mfma16_f32", F17, 8, 4, 2, 2, 2, 16, 4, epi=0),
        RW("mfma16_f32", F71, 8, 2, 2, 2, 2, 16, 4, epi=0),
    ],
)
add(
    "FLOAT",
    ROWS_GFX9,
    [
        RW("mfma16_f32", F33, 8, 2, 2, 2, 2, 8, 4, epi=0),
        RW("mfma16_f32", F33, 8, 2, 2, 2, 2, 8, 4, pp=1, epi=0),
        # 3x3x3 fp32 (Wan VAE, nnunet): 1.29-1.79x over every fp32 halo-tile config on gfx90a;
        # fn 1 for 32-channel outputs.
        RW("mfma16_f32", F333, 8, 2, 2, 2, 2, 8, 4, pp=1, epi=0),
        RW("mfma16_f32", F333, 8, 1, 2, 2, 2, 8, 4, epi=0),
    ],
)
# gfx942 (MI325X) takes the fp32 rows tile alone: 1.05-1.17x over every fp32 halo tile on 9 of 11
# 2-D 3x3 shapes, fwd and bwd. Its 16-bit rows (0.80-1.08x, ping-pong never best) and every
# 3x3x3 rows tile lost there.
add("FLOAT", ["gfx942"], [RW("mfma16_f32", F33, 8, 2, 2, 2, 2, 8, 4, epi=0)])
# 32-wide K blocks (4 x 1 waves): K = 32 layers (nnunet 3-D) waste half a 64-wide block.
BN32_F32 = [
    M("mfma16_f32", F333, 256, 32, 16, 4, 1, 4, ring=0),
    M("mfma16_f32", F333, 128, 32, 16, 4, 1, 4, ring=0),
    M("mfma16_f32", F33, 256, 32, 16, 4, 1, 4),
    M("mfma16_f32", F33, 128, 32, 16, 4, 1, 4),
]
add("FLOAT", ["gfx950", "gfx942", "gfx90a", "gfx908"], BN32_F32)
add(
    "FLOAT",
    ["gfx950"],
    [
        M("mfma16_f32", F33, 128, 64, 16, 2, 2, 4, bl=1),
        M("mfma16_f32", F33, 128, 128, 16, 2, 2, 4, bl=1),
        M("mfma16_f32", F33, 128, 64, 16, 2, 2, 4, bl=1, split=1),
        M("mfma16_f32", F333, 128, 64, 16, 2, 2, 4, ring=0, bl=1),
    ],
)
add(
    "FLOAT",
    ["gfx90a"],
    [
        M("mfma16_f32", F33, 128, 128, 16, 2, 2, 4, bl=1),
        M("mfma32_f32", F33, 128, 64, 16, 2, 2, 4, bl=1),
        M("mfma32_f32", F333, 128, 64, 16, 2, 2, 4, ring=0),
    ],
)
add(
    "FLOAT",
    ["gfx908"],
    [
        M("mfma16_f32", F33, 128, 64, 16, 2, 2, 4, bl=1),
        M("mfma16_f32", F33, 128, 128, 16, 2, 2, 4, bl=1),
        M("mfma32_f32", F333, 128, 64, 16, 2, 2, 4, ring=0),
    ],
)
add(
    "FLOAT",
    ["gfx908", "gfx90a", "gfx942", "gfx950"],
    [
        M("mfma16_f32", f, 128, 64, 8 if f in (F55, F77) else 16, 2, 2, 4)
        for f in (F55, F77, F17, F71)
    ],
)
add(
    "FLOAT",
    ["gfx11", "gfx12"],
    [F(F33, 8, 8, 4, 16, 4, pipe=1), F(F333, 8, 8, 4, 16, 4, ring=0, pipe=1)],
)
add("FLOAT", ["gfx12"], [F(F333, 8, 8, 4, 16, 4, ring=0)])
# Three-bf16 fp32 (TF32 permission only): bf16 operands, 3x reduction, fp32 output.
X3 = dict(math=1)
add(
    "FLOAT",
    ["gfx942", "gfx90a"],
    [
        M("mfma16_bf16", F33, 128, 64, 32, 2, 2, 8, **X3),
        M("mfma16_bf16", F333, 128, 64, 32, 2, 2, 8, ring=0, **X3),
    ],
)
add("FLOAT", ["gfx942"], [M("mfma16_bf16", F33, 128, 128, 32, 2, 2, 8, **X3)])
add(
    "FLOAT",
    ROWS_GFX9,
    [
        RW("mfma16_bf16", F33, 8, 4, 2, 2, 2, 16, 8, pp=1, epi=0, **X3),
        RW("mfma16_bf16", F33, 8, 2, 2, 2, 2, 16, 8, epi=0, **X3),
    ],
)
add(
    "FLOAT",
    ["gfx950"],
    [
        RW("mfma16x32_bf16", F33, 8, 4, 2, 2, 2, 64, 8, epi=0, **X3),
        RW("mfma16x32_bf16", F33, 8, 2, 2, 2, 2, 64, 8, epi=0, **X3),
    ],
)
add(
    "FLOAT",
    ["gfx950"],
    [
        M("mfma16x32_bf16", F33, 128, 64, 32, 2, 2, 8, **X3),
        M("mfma16x32_bf16", F333, 128, 64, 32, 2, 2, 8, ring=0, **X3),
        M("mfma16_bf16", F33, 128, 64, 32, 2, 2, 8, **X3),
    ],
)
add(
    "FLOAT",
    ["gfx908"],
    [
        M("mfma32_bf16_908", F33, 64, 128, 32, 1, 4, 8, **X3),
        M("mfma16_bf16_908", F33, 128, 64, 32, 2, 2, 8, **X3),
        M("mfma16_bf16_908", F333, 128, 64, 32, 2, 2, 8, ring=0, **X3),
    ],
)
for fam in ("11", "12"):
    add(
        "FLOAT",
        [f"gfx{fam}"],
        [
            M(f"wmma{fam}_bf16", F33, 128, 64, 32, 4, 2, 8, **X3),
            M(f"wmma{fam}_bf16", F33, 128, 128, 32, 2, 4, 8, **X3),
            M(f"wmma{fam}_bf16", F333, 128, 128, 32, 2, 4, 8, ring=0, **X3),
        ],
    )

# The floor, every arch: FMA for every filter and dtype, a vector form and a scalar (any C) form.
for dt in ("FLOAT", "HALF", "BFLOAT16"):
    v = 4 if dt == "FLOAT" else 8
    for f in FILTERS2D + [F333]:
        add(
            dt,
            ["any"],
            [F(f, 8, 8, 4, 16, v, ring=nr(f)), F(f, 4, 4, 8, 8, 1, ring=nr(f))],
        )

ORDER = ["gfx908", "gfx90a", "gfx942", "gfx950", "gfx11", "gfx12", "any"]
kmd, umd, udd = U("kmd"), U("umd"), U("udd")
write(
    "conv_direct.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_direct config fields",
        "fields": [{"name": f, "type": "string"} for f in STRINGS]
        + [{"name": f, "type": "int"} for f in INTS],
    },
)
write(
    "kernel_matches_graph.umd.json",
    {
        "version": "1.0",
        "id": umd,
        "name": "kernel direction, dtype, filter, atom (device, ship list) and channel vector match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_direct.kernel_match",
    },
)
write(
    "conv_direct.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_direct dispatch",
        "dispatch_symbol": "hipkernel.conv_direct.dispatch",
    },
)


def merged(entries):
    """One entry per distinct integer tuple: the union of its arch groups."""
    seen = {}
    for archs, c in entries:
        key = tuple(c[k] for k in INTS)
        if key in seen:
            seen[key][0].update(archs)
        else:
            seen[key] = [set(archs), c]
    return list(seen.values())


total = 0
for direction, tag in (("FWD", "fwd"), ("BWD", "bwd")):
    ued = U("ued." + tag)
    write(
        f"conv_direct_{tag}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_DIRECT_{direction}",
            "graph_match": {"native": f"hipkernel.conv_direct_{tag}.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    for dtype, entries in catalog.items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        for i, (archs, c) in enumerate(merged(entries)):
            groups = (
                "any" if "any" in archs else ",".join(a for a in ORDER if a in archs)
            )
            mathTag = "bf16x3" if c["math"] else "native"
            if c["rows"]:
                # ConvDirectRows.h's two halo stages; the pack gates the same (rowsLdsBytes).
                eb = 4 if c["atom"] == "mfma16_f32" else 2
                lds = 2 * (c["wp"] * c["pw"] + c["r"] - 1) * (c["wq"] * 16 + c["s"] - 1)
                lds *= c["bc"] * eb
                limit = 160 * 1024 if archs <= {"gfx950"} else 64 * 1024
                assert lds <= limit, (c, archs, lds)
                name = (
                    f"conv_direct.{tag}.{dt}.{c['kd']}x{c['r']}x{c['s']}.{c['atom']}.{mathTag}.rows"
                    f".p{c['pw']}f{c['fn']}.w{c['wp']}x{c['wq']}x{c['wk']}c{c['bc']}.v{c['vec']}"
                    f".pp{c['pp']}.epi{c['epi']}"
                )
            else:
                name = (
                    f"conv_direct.{tag}.{dt}.{c['kd']}x{c['r']}x{c['s']}.{c['atom']}.{mathTag}"
                    f".{c['bm']}x{c['bn']}c{c['bc']}.w{c['wm']}x{c['wn']}.t{c['tm']}x{c['tn']}n{c['nt']}"
                    f".v{c['vec']}.ring{c['ring']}.bl{c['bl']}.split{c['split']}.pipe{c['pipe']}"
                )
            kernels.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvDirect.cpp",
                        "entry_point": "ConvDirect",
                    },
                    "metadata": {
                        "direction": direction,
                        "dtype": dtype,
                        "kind": c["kind"],
                        "atom": c["atom"],
                        "archs": groups,
                        **{k: c[k] for k in INTS},
                    },
                    "priority": 1000 - i,
                }
            )
    total += len(kernels)
    write(
        f"conv_direct_{tag}.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp." + tag),
            "name": f"hipkernel:conv_direct_{tag}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
print(total, "kernel configs")
