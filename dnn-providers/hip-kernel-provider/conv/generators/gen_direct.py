# Generates the CONV_DIRECT_FWD / CONV_DIRECT_BWD descriptors (stride-1 halo-tile direct
# convolution, K2). Each config is admitted on correctness alone (its atom on the device, its
# filter, its channel vector dividing the reduction channels) plus its ship list: the arch groups
# whose autotuned sweeps it won. FMA configs ship
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
    "wmma125_f16",
    "wmma125_bf16",
    "wmma125_f32",
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
    "un",
    "cf",
    "xpf",
    "dil",
    "line",
    "bpu",
]
ROWS0 = dict(
    rows=0, pw=0, fn=0, wp=0, wq=0, wk=0, pp=0, epi=0, un=0, cf=0, dil=1, line=0, bpu=1
)
STRINGS = ["direction", "dtype", "kind", "atom", "archs"]
F33, F55, F77, F17, F71, F333, F555 = (
    (1, 3, 3),
    (1, 5, 5),
    (1, 7, 7),
    (1, 1, 7),
    (1, 7, 1),
    (3, 3, 3),
    (5, 5, 5),
)
FILTERS2D = [F33, F55, F77, F17, F71]


def M(atom, f, bm, bn, bc, wm, wn, vec, ring=1, bl=0, split=0, math=0, cf=0, xpf=0):
    """Halo-tile matrix-core kernel; cf 1 (16-bit): packed channels-first activations in place.
    xpf 1 (register-path weights, channels-last): weight fragments prefetched across channel
    chunks, ring sets skipped by scalar branches, 16-bit halo rows padded by 8 bytes."""
    c = dict(
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
    c["cf"] = cf
    c["xpf"] = xpf
    return c


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
        xpf=0,
    )


def RW(
    atom,
    f,
    pw,
    fn,
    wp,
    wq,
    wk,
    bc,
    vec,
    pp=0,
    epi=1,
    math=0,
    un=1,
    split=0,
    cf=0,
    dil=1,
    line=0,
    bpu=1,
):
    """Filter-row reuse kernel (ConvDirectRows.h): LDS DMA (16-byte on gfx950, dword elsewhere),
    16x16 MFMA atoms; 2-D, or 3-D (each output plane walks its depth taps' input planes). un > 1
    (2-D, gfx950): each MFMA's 16 columns are 16 / un columns of un consecutive images. cf 1
    (16-bit): packed channels-first activations read and written in place; cf 2 is the
    dword-DMA form (even input widths) on any MFMA part. dil (2-D): the filter dilation it serves.
    line 1 (1 x S filters on one-row images): a tile is wp pw segments of 16 wq columns of the row,
    staged as one halo line. bpu: persistent blocks resident per unit (their LDS stages together).
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
        split=split,
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
        un=un,
        cf=cf,
        xpf=0,
        dil=dil,
        line=line,
        bpu=bpu,
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
            RW(f"mfma16_{e}", F33, 8, 4, 2, 2, 2, 16, 8, pp=1),
            RW(f"mfma16_{e}", F33, 4, 4, 2, 1, 2, 32, 8),
            RW(f"mfma16_{e}", F33, 8, 2, 2, 2, 2, 16, 8),
            # 3x3x3: the 96-wide ping-pong tile won Wan VAE's 96-channel layers (1.07-1.14x over
            # the halo tile), pw4 fn4 its 384-channel 3x62x106 (1.08x); 192+ wide stay on K2.
            RW(f"mfma16_{e}", F333, 8, 3, 2, 2, 2, 16, 8, pp=1, epi=0),
            RW(f"mfma16_{e}", F333, 4, 4, 2, 2, 2, 16, 8, pp=1, epi=0),
            # 4-image 8 x 8 tiles (dword DMA batch unfold), the ping-pong one and a 256-wide K
            # one: MI210 kernel bench over 37 SD/SDXL/VAE 2-D 3x3 shapes, weighted time -5.8% vs the
            # gfx90a catalog tiles (40^2 c1280 1.36-1.51x -> 1.10-1.17x of MIOpen's search time).
            RW(f"mfma16_{e}", F33, 8, 4, 2, 2, 2, 16, 8, pp=1, un=4),
            RW(f"mfma16_{e}", F33, 8, 4, 1, 2, 4, 16, 8, un=4),
        ],
    )
    # gfx908 rows, dword LDS DMA; bf16 on the 16x16x8 atom. Eight-wave blocks run two waves per
    # SIMD at 128 VGPRs + 128 AGPRs, so the tiles keep 8-12 accumulator sets a wave; the four-wave
    # tiles (one wave per SIMD) take the 20- and 24-row tiles.
    a908 = "mfma16_f16" if dt == "HALF" else "mfma16_bf16_908"
    add(
        dt,
        ["gfx908"],
        [
            RW(a908, F33, 8, 2, 2, 2, 2, 16, 8),
            RW(a908, F33, 6, 2, 2, 2, 2, 32, 8),
            RW(a908, F33, 4, 4, 2, 2, 2, 32, 8),
            RW(a908, F33, 8, 3, 2, 2, 2, 16, 8),
            RW(a908, F33, 8, 3, 2, 1, 2, 16, 8),
            RW(a908, F33, 5, 4, 4, 1, 2, 16, 8),
            RW(a908, F33, 6, 2, 4, 1, 2, 16, 8),
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
    # gfx942 3x3x3 (large 3-D bf16 volumes, MI300A / MI325X, autotuned, interleaved A/B):
    # the 128 x 192 tile (8 waves) takes the 512 / 1024-channel layers on small 3-D outputs
    # (1.08-1.1x); rows tiles take 256 / 512-channel layers on mid-size 3-D outputs on MI325X
    # (1.2-1.3x); the cross-chunk prefetch (xpf) twins of the halo tiles take the
    # 512 / 1024-channel layers (1.1-1.13x).
    add(
        dt,
        ["gfx942"],
        [
            M(f"mfma16_{e}", F333, 128, 192, 16, 2, 4, 8, ring=0),
            RW(f"mfma16_{e}", F333, 8, 4, 1, 2, 4, 16, 8),
            RW(f"mfma16_{e}", F333, 8, 4, 2, 2, 2, 16, 8),
            M(f"mfma16_{e}", F333, 128, 128, 16, 2, 2, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F333, 128, 64, 16, 2, 2, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=1, xpf=1),
            M(f"mfma16_{e}", F333, 128, 32, 32, 4, 1, 8, ring=0, xpf=1),
            # One wave row (wm 1): each wave reuses its weight fragments over all 128 rows, so
            # the register-path weights are read once per block and half as often per MMA.
            M(f"mfma16_{e}", F333, 128, 128, 16, 1, 4, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F333, 128, 128, 32, 1, 4, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F333, 128, 64, 32, 1, 4, 8, ring=1, xpf=1),
            M(f"mfma16_{e}", F333, 128, 192, 16, 1, 4, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F333, 128, 256, 16, 1, 8, 8, ring=0, xpf=1),
            # Split-C twins for under-filled grids (the planner splits the channel chunks only
            # below two blocks a CU): the 512 / 1024-channel layers on small 3-D and smaller
            # outputs run 64-192 blocks on 228-304 CUs unsplit.
            M(f"mfma16_{e}", F333, 128, 192, 16, 2, 4, 8, ring=0, split=1),
            M(f"mfma16_{e}", F333, 128, 128, 16, 2, 2, 8, ring=0, split=1, xpf=1),
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=0, split=1, xpf=1),
            M(f"mfma16_{e}", F333, 128, 128, 16, 1, 4, 8, ring=0, split=1, xpf=1),
            # 2-D 3x3: the xpf twins of the gfx942 halo tiles.
            M(f"mfma16_{e}", F33, 128, 64, 32, 2, 2, 8, xpf=1),
            M(f"mfma16_{e}", F33, 128, 64, 16, 2, 2, 8, xpf=1),
            M(f"mfma16_{e}", F33, 128, 128, 16, 2, 2, 8, xpf=1),
            M(f"mfma16_{e}", F33, 128, 128, 32, 2, 2, 8, xpf=1),
        ],
    )
    # gfx942 (MI325X) rows, dword LDS DMA, 16-channel chunks: the 16 x 16 x 256 tile and the
    # 4-image 8 x 8 tile (40^2 layers 0.80-0.91x of MIOpen's search time) take most 2-D 3x3 16-bit
    # layers from the halo tile (kernel-only sweep, 37 SD/SDXL/VAE shapes: 7% of the weighted time).
    add(
        dt,
        ["gfx942"],
        [
            RW(f"mfma16_{e}", F33, 8, 4, 2, 1, 4, 16, 8),
            RW(f"mfma16_{e}", F33, 8, 4, 2, 2, 2, 16, 8),
            RW(f"mfma16_{e}", F33, 8, 4, 1, 2, 4, 16, 8, un=4),
            RW(f"mfma16_{e}", F33, 5, 4, 4, 1, 2, 16, 8),
        ],
    )
    # Packed channels-first activations in place on the dword-DMA parts (gather transpose in
    # LDS): the 16 x 16 x 256, 16 x 32 x 128 and 20-row tiles at 16-channel chunks, and the
    # 8 x 16 x 128 tile at 32.
    add(
        dt,
        ["gfx90a", "gfx942"],
        [
            RW(f"mfma16_{e}", F33, 8, 4, 1, 2, 4, 16, 8, epi=0, cf=1),
            RW(f"mfma16_{e}", F33, 8, 4, 2, 2, 2, 16, 8, epi=0, cf=1),
            RW(f"mfma16_{e}", F33, 5, 4, 4, 1, 2, 16, 8, epi=0, cf=1),
            RW(f"mfma16_{e}", F33, 4, 4, 2, 1, 2, 32, 8, epi=0, cf=1),
            RW(f"mfma16_{e}", F333, 8, 4, 1, 2, 4, 16, 8, epi=0, cf=1),
            RW(f"mfma16_{e}", F333, 8, 4, 2, 2, 2, 16, 8, epi=0, cf=1),
        ],
    )
    # gfx950: the rows kernel (LDS DMA), then the halo-tile configs
    add(
        dt,
        ["gfx950"],
        [
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 64, 8),
            # Ping-pong with the staged epilogue (the direct-store epilogue cost it 4%): 1.05x over
            # the single-group tile on 16-bit VAE / UNet layers (MI355X; 1.04x on > 96^2 maps).
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 64, 8, pp=1),
            RW(f"mfma16x32_{e}", F33, 8, 2, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 8, 1, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 4, 4, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 1, 2, 32, 8),
            # 48-wide tiles: images whose width is a multiple of 48 (sdxl 48^2 / 96^2)
            RW(f"mfma16x32_{e}", F33, 8, 2, 2, 3, 2, 32, 8),
            RW(f"mfma16x32_{e}", F33, 8, 4, 1, 3, 2, 64, 8),
            # Tile counts and image extents the 32-wide x 16-row tile fits badly (SD/SDXL UNet
            # and VAE, 24^2 to 192^2, 320-2560 channels, fwd and bwd), each the best of a
            # 15-tile sweep on some of them (MI355X, same GPU as ConvHipConv): 20-row
            # (40^2: 1.2x), 256-wide K at 16 or 32 columns (hipconv direct_l1's K256 tile;
            # 80^2 / 24^2: 1.1-1.3x), 24- and 12-row (48^2 / 96^2), 8-row 16-column (80^2).
            RW(f"mfma16x32_{e}", F33, 5, 4, 4, 1, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 1, 4, 64, 8),
            RW(f"mfma16x32_{e}", F33, 6, 4, 2, 1, 4, 64, 8),
            RW(f"mfma16x32_{e}", F33, 8, 4, 1, 2, 4, 64, 8),
            RW(f"mfma16x32_{e}", F33, 4, 4, 4, 1, 2, 64, 8),
            RW(f"mfma16x32_{e}", F33, 6, 4, 2, 2, 2, 64, 8),
            # Two images of 8 columns per MFMA column group (hipconv's batch-packed tiles):
            # 24^2 (1.25x over the best plain tile) and 40^2 / 16^2 maps (1.03-1.16x).
            RW(f"mfma16x32_{e}", F33, 6, 4, 4, 1, 2, 64, 8, un=2),
            RW(f"mfma16x32_{e}", F33, 5, 4, 2, 1, 4, 64, 8, un=2),
            # Four images of 4 columns per MFMA column group: an 8 x 8-pixel tile of 4 images
            # (hipconv's batch-packed direct tile), so 40^2 c1280 layers fill one persistent round
            # (MI355X vs ConvHipConv: 40^2 1.35-1.42x -> 1.13-1.18x behind, 80^2 1.16x -> 0.99x).
            RW(f"mfma16x32_{e}", F33, 8, 4, 1, 2, 4, 64, 8, un=4),
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 32, 8, un=4),
            RW(f"mfma16x32_{e}", F33, 4, 4, 2, 2, 2, 64, 8, un=4),
            # Split-C of the main tile for grids with at most a quarter of the units busy (n1-n2
            # layers): fp32 partials and an in-order reduce (n1 c640 64^2: 1.4x).
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 64, 8, split=1),
            RW(f"mfma16x32_{e}", F55, 8, 2, 2, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F77, 8, 2, 2, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F17, 8, 4, 2, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F71, 8, 2, 2, 2, 2, 32, 8),
            # 3x3x3 (Wan VAE, nnunet bf16): 1.03-1.79x over the halo tile, fwd and bwd.
            RW(f"mfma16x32_{e}", F333, 8, 2, 2, 3, 2, 32, 8),
            RW(f"mfma16x32_{e}", F333, 8, 2, 2, 2, 2, 64, 8),
            RW(f"mfma16x32_{e}", F333, 8, 4, 2, 2, 2, 64, 8),
            # 32-wide 3x3x3 tiles of 1024 pixels (32 x 32 and 16 x 64): 32-channel outputs (nnU-Net
            # c32 / c64 -> k32, 128^3-160^3), where the 64-wide tile leaves half of every MMA
            # idle: 1.3-1.6x over the best 3x3x3 tile before them (MI355X kernel bench).
            RW(f"mfma16x32_{e}", F333, 8, 1, 4, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F333, 8, 2, 2, 4, 1, 32, 8),
            # Packed channels-first activations in place (no staging transposes): the 256-wide K
            # tile at 64- and 32-channel chunks, the 16 x 32 tile, and the 20-row tile for 40^2.
            RW(f"mfma16x32_{e}", F33, 8, 4, 1, 2, 4, 64, 8, epi=0, cf=1),
            RW(f"mfma16x32_{e}", F33, 8, 4, 1, 2, 4, 32, 8, epi=0, cf=1),
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 32, 8, epi=0, cf=1),
            RW(f"mfma16x32_{e}", F33, 5, 4, 4, 1, 2, 64, 8, epi=0, cf=1),
            # 3x3x3 in place (input widths of whole 16-byte runs), and the dword-DMA form for
            # even widths (video VAE inputs padded by two columns), 2-D and 3-D.
            RW(f"mfma16x32_{e}", F333, 8, 4, 1, 2, 4, 64, 8, epi=0, cf=1),
            RW(f"mfma16x32_{e}", F333, 8, 2, 2, 3, 2, 32, 8, epi=0, cf=1),
            RW(f"mfma16x32_{e}", F333, 8, 4, 1, 2, 4, 32, 8, epi=0, cf=1),
            RW(f"mfma16x32_{e}", F333, 8, 2, 2, 3, 2, 32, 8, epi=0, cf=2),
            RW(f"mfma16x32_{e}", F333, 8, 4, 1, 2, 4, 32, 8, epi=0, cf=2),
            RW(f"mfma16x32_{e}", F33, 8, 4, 1, 2, 4, 64, 8, epi=0, cf=2),
            RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 32, 8, epi=0, cf=2),
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
    # xpf twins of the gfx950 and gfx90a halo tiles (3x3x3 and 3x3).
    add(
        dt,
        ["gfx950"],
        [
            M(f"mfma16x32_{e}", F333, 128, 128, 32, 2, 2, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=0, xpf=1),
            M(f"mfma16x32_{e}", F33, 128, 64, 32, 2, 2, 8, xpf=1),
        ],
    )
    add(
        dt,
        ["gfx90a"],
        [
            M(f"mfma16_{e}", F333, 128, 64, 32, 2, 2, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F333, 128, 128, 16, 2, 2, 8, ring=0, xpf=1),
            M(f"mfma16_{e}", F33, 128, 64, 32, 2, 2, 8, xpf=1),
            M(f"mfma16_{e}", F33, 128, 128, 32, 2, 2, 8, xpf=1),
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
    # 5x5x5 (ConvTranspose3d grad-input, c64 -> k16 / k32): 16- and 32-wide K blocks; the
    # halo in LDS feeds all 125 taps where the implicit GEMM re-gathers them.
    add(
        dt,
        ["gfx950"],
        [
            M(f"mfma16x32_{e}", F555, 128, 16, 32, 4, 1, 8, ring=0),
            M(f"mfma16x32_{e}", F555, 128, 32, 32, 4, 1, 8, ring=0),
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
    # Packed channels-first activations in place on the halo tile (wide tiles, a register
    # transpose of 8-channel runs): RDNA 2-D and 3-D, and CDNA 3-D (the rows tiles take 2-D).
    for fam in ("11", "12"):
        a = f"wmma{fam}_{e}"
        add(
            dt,
            [f"gfx{fam}"],
            [
                M(a, F33, 128, 128, 16, 2, 4, 8, cf=1),
                M(a, F33, 128, 128, 32, 2, 4, 8, cf=1),
                M(a, F33, 128, 64, 32, 4, 2, 8, cf=1),
                M(a, F333, 128, 128, 16, 2, 4, 8, ring=0, cf=1),
                M(a, F333, 128, 128, 32, 2, 4, 8, ring=0, cf=1),
            ],
        )
    add(
        dt,
        ["gfx90a", "gfx942", "gfx950"],
        [
            M(f"mfma16_{e}", F333, 128, 128, 16, 2, 2, 8, ring=0, cf=1),
            M(f"mfma16_{e}", F333, 128, 64, 16, 2, 2, 8, ring=0, cf=1),
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
    # Narrow channels: 32- and 64-wide output blocks at 32-channel chunks (16 on the dword-DMA
    # parts), channels-last and channels-first in place. On K <= 64 outputs (and the C <= 64
    # reductions of backward data) a 128-wide block leaves half or more of every MMA idle and a
    # 64-channel chunk half of every step's reduction.
    add(
        dt,
        ["gfx950"],
        [
            RW(f"mfma16x32_{e}", F33, 8, 2, 2, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F33, 8, 1, 2, 2, 2, 32, 8),
            RW(f"mfma16x32_{e}", F33, 8, 2, 2, 2, 2, 32, 8, epi=0, cf=1),
            RW(f"mfma16x32_{e}", F33, 8, 1, 2, 2, 2, 32, 8, epi=0, cf=1),
            RW(f"mfma16x32_{e}", F33, 8, 2, 1, 2, 2, 64, 8, epi=0, cf=1),
        ],
    )
    add(
        dt,
        ["gfx90a", "gfx942"],
        [
            RW(f"mfma16_{e}", F33, 8, 2, 2, 2, 2, 16, 8),
            RW(f"mfma16_{e}", F33, 8, 1, 2, 2, 2, 16, 8),
            RW(f"mfma16_{e}", F33, 8, 2, 2, 2, 2, 16, 8, epi=0, cf=1),
            RW(f"mfma16_{e}", F33, 8, 1, 2, 2, 2, 16, 8, epi=0, cf=1),
        ],
    )
    # Narrow 8-row tiles of 4 waves, two or three resident per unit: the blocks' chunk DMA,
    # transposes and barriers overlap each other's MMAs, which a K <= 64 tile's short chunks (three
    # steps of 32 channels) cannot hide alone.
    add(
        dt,
        ["gfx950"],
        [
            RW(f"mfma16x32_{e}", F33, 8, 2, 1, 2, 2, 32, 8, epi=0, cf=1, bpu=b)
            for b in (2, 3)
        ]
        + [
            RW(f"mfma16x32_{e}", F33, 8, 1, 1, 2, 2, 32, 8, epi=0, cf=1, bpu=b)
            for b in (2, 3)
        ]
        + [RW(f"mfma16x32_{e}", F33, 8, 2, 1, 2, 2, 32, 8, bpu=b) for b in (2, 3)],
    )
    add(
        dt,
        ["gfx90a", "gfx942"],
        [
            RW(f"mfma16_{e}", F33, 8, 2, 1, 2, 2, 16, 8, epi=0, cf=1, bpu=2),
            RW(f"mfma16_{e}", F33, 8, 1, 1, 2, 2, 16, 8, epi=0, cf=1, bpu=2),
        ],
    )
    # Dilated 3x3 (DeepLab's dilation 2 and 4 stages, any channel count): the halo grows by
    # (3 - 1) d rows and columns and each filter row reads its input rows d apart, still from the
    # one staged halo; 32-channel chunks keep the larger halo within the LDS.
    for d in (2, 4):
        # 8-row tiles two per unit at dilation 2 (dilation 4's halo passes half the LDS).
        pair = [
            RW(f"mfma16x32_{e}", F33, 8, 4, 1, 2, 2, 32, 8, epi=0, cf=1, dil=d, bpu=2)
        ]
        add(
            dt,
            ["gfx950"],
            (pair if d == 2 else [])
            + [
                RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 32, 8, dil=d),
                RW(f"mfma16x32_{e}", F33, 8, 2, 2, 2, 2, 32, 8, dil=d),
                RW(f"mfma16x32_{e}", F33, 8, 4, 2, 2, 2, 32, 8, epi=0, cf=1, dil=d),
                RW(f"mfma16x32_{e}", F33, 8, 2, 2, 2, 2, 32, 8, epi=0, cf=1, dil=d),
            ],
        )
        # 64 KiB parts: 8-row tiles at dilation 4.
        wp = 2 if d == 2 else 1
        add(
            dt,
            ["gfx90a", "gfx942"],
            [
                RW(f"mfma16_{e}", F33, 8, 4, wp, 2, 2, 16, 8, dil=d),
                RW(f"mfma16_{e}", F33, 8, 4, wp, 2, 2, 16, 8, epi=0, cf=1, dil=d),
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
# The cross-chunk prefetch on the gfx942 fp32 3x3x3 halo tiles.
add(
    "FLOAT",
    ["gfx942"],
    [
        M("mfma16_f32", F333, 128, 64, 16, 2, 2, 4, ring=0, xpf=1),
        M("mfma16_f32", F333, 128, 128, 16, 2, 2, 4, ring=0, xpf=1),
    ],
)
# gfx942 (MI325X) takes the fp32 rows tile alone: 1.05-1.17x over every fp32 halo tile on 9 of 11
# 2-D 3x3 shapes, fwd and bwd. Its 16-bit rows (0.80-1.08x, ping-pong never best) and every
# 3x3x3 rows tile lost there.
add("FLOAT", ["gfx942"], [RW("mfma16_f32", F33, 8, 2, 2, 2, 2, 8, 4, epi=0)])
# Line tiles: 1 x S filters on one-row images (1-D audio convolutions, HiFi-GAN's 3/7/11-tap
# residual stacks at dilation 1, 3 and 5). A wave owns 8 segments of 16 columns of the row, so each
# weight fragment serves 8 segments and the staged line serves every tap; channels-first (16-bit,
# in place) and channels-last.
LINE_GEOMETRIES = [(s, d) for s in (3, 7, 11) for d in (1, 3, 5)]
for s_, d in LINE_GEOMETRIES:
    f = (1, 1, s_)
    L = dict(dil=d, line=1)
    for dt in ("HALF", "BFLOAT16"):
        e = x16(dt)
        add(
            dt,
            ["gfx950"],
            [
                RW(f"mfma16x32_{e}", f, 8, 4, 1, 2, 2, 32, 8, epi=0, cf=1, bpu=2, **L),
                RW(f"mfma16x32_{e}", f, 8, 2, 1, 2, 2, 32, 8, epi=0, cf=1, bpu=2, **L),
                RW(f"mfma16x32_{e}", f, 8, 4, 2, 2, 2, 32, 8, epi=0, cf=1, **L),
                RW(f"mfma16x32_{e}", f, 8, 2, 2, 2, 2, 32, 8, epi=0, cf=1, **L),
                RW(f"mfma16x32_{e}", f, 8, 2, 2, 2, 2, 32, 8, **L),
                RW(f"mfma16x32_{e}", f, 8, 4, 1, 2, 2, 32, 8, **L),
            ],
        )
        add(
            dt,
            ["gfx90a", "gfx942"],
            [
                RW(f"mfma16_{e}", f, 8, 4, 1, 2, 2, 16, 8, epi=0, cf=1, bpu=2, **L),
                RW(f"mfma16_{e}", f, 8, 4, 2, 2, 2, 16, 8, epi=0, cf=1, **L),
                RW(f"mfma16_{e}", f, 8, 2, 2, 2, 2, 16, 8, **L),
                RW(f"mfma16_{e}", f, 8, 4, 2, 2, 2, 16, 8, **L),
            ],
        )
    add("FLOAT", ["gfx950"], [RW("mfma16_f32", f, 8, 2, 2, 2, 2, 16, 4, epi=0, **L)])
    add(
        "FLOAT",
        ["gfx90a", "gfx942"],
        [RW("mfma16_f32", f, 8, 2, 2, 2, 2, 8, 4, epi=0, **L)],
    )
# Dilated 3x3 fp32 (dilation 2 and 4): the 16-bit dilated tiles at half the chunk.
for d in (2, 4):
    add(
        "FLOAT", ["gfx950"], [RW("mfma16_f32", F33, 8, 2, 2, 2, 2, 16, 4, epi=0, dil=d)]
    )
    add(
        "FLOAT",
        ["gfx90a", "gfx942"],
        [RW("mfma16_f32", F33, 8, 2, 2 if d == 2 else 1, 2, 2, 8, 4, epi=0, dil=d)],
    )
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
        M("mfma16_f32", F555, 128, 16, 8, 4, 1, 4, ring=0),
        M("mfma16_f32", F555, 256, 32, 16, 4, 1, 4, ring=0),
    ],
)
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
# Three-bf16 fp32 (TF32 configs, the CONV_DIRECT_*_TF32 engines'): bf16 operands, 3x reduction,
# fp32 output.
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

# Native gfx1250 atoms. Keep these separate from the older gfx12 instruction contracts.
for dt, atom, bc, vec in (
    ("HALF", "wmma125_f16", 32, 8),
    ("BFLOAT16", "wmma125_bf16", 32, 8),
    ("FLOAT", "wmma125_f32", 4, 4),
):
    for f in FILTERS2D + [F333]:
        add(
            dt,
            ["gfx1250"],
            [
                M(atom, f, 64, 64, bc, 2, 2, vec, ring=nr(f)),
                M(atom, f, 128, 64, bc, 2, 2, vec, ring=nr(f)),
                M(atom, f, 64, 64, bc, 2, 2, 1, ring=nr(f)),
            ],
        )

add(
    "FLOAT",
    ["gfx1250"],
    [
        M("wmma125_bf16", F33, 64, 64, 32, 2, 2, 8, **X3),
        M("wmma125_bf16", F33, 128, 64, 32, 2, 2, 8, **X3),
        M("wmma125_bf16", F333, 64, 64, 32, 2, 2, 8, ring=0, **X3),
    ],
)

# The floor, every arch: FMA for every filter and dtype, a vector form and a scalar (any C) form.
for dt in ("FLOAT", "HALF", "BFLOAT16"):
    v = 4 if dt == "FLOAT" else 8
    for f in FILTERS2D + [F333, F555]:
        add(
            dt,
            ["any"],
            [F(f, 8, 8, 4, 16, v, ring=nr(f)), F(f, 4, 4, 8, 8, 1, ring=nr(f))],
        )

# Extra native candidates follow the existing entries so fallback priorities stay stable.
for dt, atom in (("HALF", "wmma125_f16"), ("BFLOAT16", "wmma125_bf16")):
    # Reuse the 3x3 input halo across twice as many output channels per block.
    add(dt, ["gfx1250"], [M(atom, F33, 64, 128, 32, 2, 4, 8)])

# gfx942 fp32 3x3x3 trial candidates (nnU-Net 32-128 channel layers).
add(
    "FLOAT",
    ["gfx942"],
    [
        M("mfma16_f32", F333, 128, 64, 16, 2, 2, 4, ring=0, bl=1),
        M("mfma16_f32", F333, 128, 128, 16, 2, 2, 4, ring=0, bl=1),
        M("mfma16_f32", F333, 256, 32, 16, 4, 1, 4, ring=0, xpf=1),
        M("mfma16_f32", F333, 128, 32, 16, 4, 1, 4, ring=0, xpf=1),
        M("mfma16_f32", F333, 256, 32, 8, 4, 1, 4, ring=0),
        M("mfma16_f32", F333, 256, 64, 16, 4, 2, 4, ring=0, xpf=1),
        M("mfma16_f32", F333, 256, 64, 16, 4, 2, 4, ring=0),
        M("mfma16_f32", F333, 256, 32, 16, 4, 1, 4, ring=1),
        M("mfma16_f32", F333, 128, 64, 16, 2, 2, 4, ring=1, xpf=1),
        M("mfma16_f32", F333, 256, 32, 16, 4, 1, 4, ring=0, bl=1),
    ],
)

ORDER = ["gfx908", "gfx90a", "gfx942", "gfx950", "gfx11", "gfx12", "gfx1250", "any"]
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
            # RFC 0025 numerical notes. No "nondeterministic": every config is
            # bitwise reproducible (fixed-order sums, no atomics), which the
            # family's device test checks by relaunching each config.
            "numerical_notes": [],
        },
    )
    kernels = []
    for dtype, entries in catalog.items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        legacy_index = native_index = 0
        for archs, c in merged(entries):
            if c["atom"].startswith("wmma125"):
                priority = 1100 - native_index
                native_index += 1
            else:
                priority = 1000 - legacy_index
                legacy_index += 1
            groups = (
                "any" if "any" in archs else ",".join(a for a in ORDER if a in archs)
            )
            mathTag = "bf16x3" if c["math"] else "native"
            if c["rows"]:
                # ConvDirectRows.h's two halo stages; the pack gates the same (rowsLdsBytes).
                eb = 4 if c["atom"] == "mfma16_f32" else 2
                hh = 1 if c["line"] else c["wp"] * c["pw"] + (c["r"] - 1) * c["dil"]
                cols = c["wq"] * 16 * (c["wp"] * c["pw"] if c["line"] else 1)
                reach = (c["s"] - 1) * c["dil"]
                lds = 2 * hh * c["un"] * (cols // c["un"] + reach) * c["bc"] * eb
                if c["cf"]:
                    runs = ((7 + cols + reach + 7) // 8 + 1) // 2 * 2
                    lds = lds // 2 + hh * c["bc"] * runs * 16
                limit = 160 * 1024 if archs <= {"gfx950"} else 64 * 1024
                assert lds * c["bpu"] <= limit, (c, archs, lds)
                name = (
                    f"conv_direct.{tag}.{dt}.{c['kd']}x{c['r']}x{c['s']}.{c['atom']}.{mathTag}.rows"
                    f".p{c['pw']}f{c['fn']}.w{c['wp']}x{c['wq']}x{c['wk']}c{c['bc']}.v{c['vec']}"
                    f".pp{c['pp']}.epi{c['epi']}"
                    + (f".u{c['un']}" if c["un"] > 1 else "")
                    + (".split" if c["split"] else "")
                    + {0: "", 1: ".cf", 2: ".cfd"}[c["cf"]]
                    + (f".d{c['dil']}" if c["dil"] > 1 else "")
                    + (".line" if c["line"] else "")
                    + (f".b{c['bpu']}" if c["bpu"] > 1 else "")
                )
            else:
                name = (
                    f"conv_direct.{tag}.{dt}.{c['kd']}x{c['r']}x{c['s']}.{c['atom']}.{mathTag}"
                    f".{c['bm']}x{c['bn']}c{c['bc']}.w{c['wm']}x{c['wn']}.t{c['tm']}x{c['tn']}n{c['nt']}"
                    f".v{c['vec']}.ring{c['ring']}.bl{c['bl']}.split{c['split']}.pipe{c['pipe']}"
                    + (".cf" if c["cf"] else "")
                    + (".xpf" if c["xpf"] else "")
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
                    "priority": priority,
                }
            )
    total += len(kernels)
    # The three-bf16 configs make an engine of their own, which reports the TENSOR_CORE
    # numerical note (cuDNN's note for TF32 engines), so a caller can exclude them by note.
    ued_tf32 = U("ued." + tag + ".tf32")
    write(
        f"conv_direct_{tag}_tf32.ued.json",
        {
            "version": "1.0",
            "id": ued_tf32,
            "name": f"hipkernel:CONV_DIRECT_{direction}_TF32",
            "graph_match": {"native": f"hipkernel.conv_direct_{tag}.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
            # RFC 0025 numerical notes: TENSOR_CORE (fp32 through bf16 operand parts).
            # Deterministic as the exact engine.
            "numerical_notes": ["tensor_core"],
        },
    )
    for suffix, engine, members in (
        ("", ued, [k for k in kernels if not k["metadata"]["math"]]),
        ("_tf32", ued_tf32, [k for k in kernels if k["metadata"]["math"]]),
    ):
        write(
            f"conv_direct_{tag}{suffix}.kdp.json",
            {
                "version": "1.0",
                "id": U("kdp." + tag + suffix.replace("_", ".")),
                "name": f"hipkernel:conv_direct_{tag}{suffix}",
                "arch": [],
                "matchers": [umd],
                "engine": engine,
                "dispatch": udd,
                "kernelDescriptors": members,
            },
        )
print(total, "kernel configs")
