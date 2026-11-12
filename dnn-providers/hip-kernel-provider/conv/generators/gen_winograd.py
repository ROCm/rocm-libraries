# Generates the CONV_WINOGRAD_FWD / CONV_WINOGRAD_BWD descriptors (Winograd F(2,3) / F(4,3) for 3x3
# stride 1, F(3,2) on the phase sub-filters for 3x3 stride 2; K5) and CONV_WINOGRAD_WRW (backward
# weights, F(3,2) / F(3,4) / F(3,6) for 3x3 stride 1). Each config is admitted on correctness alone; autotune (UHD later) ranks them.
# Priority orders the declared fallback: fused matrix-core (cooperative, then register-built operands), fused
# VALU (register operands first), multipass, then the
# multipass LLC-fit plan (an extra config, never forced).
#   python3 gen_winograd.py <descriptors/conv/winograd>
import json, uuid, pathlib, sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("9b3f6a2e-1c4d-4e8f-a7b5-2d0c8e6f4a13")
U = lambda name: str(uuid.uuid5(NS, "winograd." + name))


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


FIELDS = [
    "direction",
    "dtype",
    "variant",
    "m",
    "layout",
    "arch",
    "llc",
    "gemm_bm",
    "gemm_bn",
    "gemm_bk",
    "bk",
    "bt",
    "ccv",
    "tm",
    "tn",
    "waves",
    "bins",
    "f16math",
    "pf",
]
STRINGS = {"direction", "dtype", "variant", "layout", "arch"}
# The integer fields identify a config among those one graph admits (the enrolled knob tuple):
# llc tells the multipass LLC-fit plan from the plain one.
KNOBS = [
    "m",
    "llc",
    "gemm_bm",
    "gemm_bn",
    "gemm_bk",
    "bk",
    "bt",
    "ccv",
    "tm",
    "tn",
    "waves",
    "bins",
    "f16math",
    "pf",
]
ENTRY = {
    "multipass": "ConvWinogradGemm",
    "fused": "ConvWinogradFused",
    "wmma": "ConvWinogradWmma",
    "wreg": "ConvWinogradWreg",
    "sreg": "ConvWinogradSreg",
    "coop": "ConvWinogradCoop",
    "wrw": "ConvWinogradWrwGemm",
}


def C(
    variant,
    m=2,
    layout="NCHW",
    arch="any",
    plan="none",
    gemm=(0, 0, 0),
    bk=0,
    bt=0,
    ccv=0,
    tm=0,
    tn=0,
    waves=0,
    bins=0,
    f16math=0,
    pf=0,
):
    return dict(
        variant=variant,
        m=m,
        layout=layout,
        arch=arch,
        llc=1 if plan == "llc" else 0,
        gemm_bm=gemm[0],
        gemm_bn=gemm[1],
        gemm_bk=gemm[2],
        bk=bk,
        bt=bt,
        ccv=ccv,
        tm=tm,
        tn=tn,
        waves=waves,
        bins=bins,
        f16math=f16math,
        pf=pf,
    )


GEMMS = [(128, 64, 8), (64, 64, 16)]


def multipass(ms, plan):
    return [
        C("multipass", m, lay, plan=plan, gemm=g)
        for m in ms
        for lay in ("NCHW", "NHWC")
        for g in GEMMS
    ]


def wmma():
    return [
        C("wmma", arch=a, bk=bk, waves=wv)
        for bk, wv in ((64, 8), (32, 4))
        for a in ("gfx11", "gfx12")
    ]


# Register-built WMMA operands (gfx11 / gfx12): K block, 16-tile groups, bins per wave (4: a whole
# transform row; 2: half), fp16 packed input-transform math.
def wreg(blocks):
    return [
        C(
            "wreg",
            arch=a,
            bk=bk,
            bt=16 * tg,
            waves=4 * (4 // bins) * tg,
            bins=bins,
            f16math=h,
        )
        for bk, tg, bins, h in blocks
        for a in ("gfx11", "gfx12")
    ]


WREG_F16 = [(32, 1, 4, 1), (32, 2, 4, 1), (64, 1, 2, 1), (32, 2, 4, 0)]
WREG_BF16 = [(32, 2, 4, 0), (64, 1, 2, 0)]


# Cooperative WMMA (gfx11 / gfx12): the block transforms each input chunk once into LDS. Waves =
# 4 x kw (one per transform row per K group), each 16 x mt out channels by 16 x nt tiles.
# m = 3 is F(3,2): stride 2 by phases. pf = 1 loads the next item's first input chunk behind the
# epilogue (CDNA, few input chunks per item).
def coop(blocks, archs=("gfx11", "gfx12"), m=2, pf=0):
    return [
        C(
            "coop",
            m,
            arch=a,
            bk=16 * mt * kw,
            bt=16 * nt,
            waves=4 * kw,
            f16math=h,
            pf=pf,
        )
        for mt, nt, kw, h in blocks
        for a in archs
    ]


# 16- and 32-wide K blocks for K <= 16 (to-RGB, K = 3) and K = 32 layers: the transform is shared
# by the out channels the layer has, not by 64 that are mostly padding.
COOP_F16 = [
    (1, 2, 4, 1),
    (2, 2, 4, 1),
    (1, 2, 8, 1),
    (2, 2, 2, 1),
    (1, 4, 4, 1),
    (1, 2, 1, 1),
    (1, 2, 2, 1),
]
COOP_BF16 = [(1, 2, 4, 0), (2, 2, 2, 0), (1, 4, 4, 0), (1, 2, 1, 0), (1, 2, 2, 0)]
# F(3,2) (stride 2): 4x the channels on the phase side (in forward, out backward data).
COOP3_F16 = [(1, 2, 4, 1), (1, 2, 2, 1), (2, 2, 2, 1), (1, 2, 1, 1)]
COOP3_BF16 = [(1, 2, 4, 0), (1, 2, 2, 0), (1, 2, 1, 0)]
# CDNA ("gfx9": wave64 16x16x16 MFMA, gfx908 and later; at most 16 waves).
COOP_CDNA_F16 = [
    (2, 4, 2, 1),
    (2, 2, 2, 1),
    (4, 4, 1, 1),
    (1, 4, 4, 1),
    (1, 4, 1, 1),
    (2, 2, 1, 1),
]
COOP_CDNA_BF16 = [(2, 4, 2, 0), (2, 2, 2, 0), (1, 4, 1, 0), (2, 2, 1, 0)]
# Cross-item prefetch on CDNA, small C (gfx950 inception c32 k32 149² n128: 7-8%).
COOPPF_CDNA_F16 = [(2, 2, 1, 1)]
COOPPF_CDNA_BF16 = [(2, 2, 1, 0)]
COOP3_CDNA_F16 = [(1, 4, 4, 1), (2, 2, 1, 1), (1, 2, 1, 1), (2, 2, 2, 1)]
COOP3_CDNA_BF16 = [(1, 4, 4, 0), (2, 2, 1, 0), (1, 2, 1, 0)]


# Register-operand VALU on wave64 gfx9: lane = tile, the transformed filter through the scalar cache.
def sreg(variants):
    return [C("sreg", arch="gfx9", bk=16, bt=64, waves=4, f16math=h) for h in variants]


# m = 3 is F(3,2): stride 2 by phases (2x2 sub-filters over 4x the channels); the others stride 1.
# pf = 1 loads input chunks two ahead (two register sets).
def fused(blocks, ms=(2, 3), pf=0):
    return [
        C("fused", m, bk=bk, bt=bt, ccv=ccv, tm=tm, tn=tn, pf=pf)
        for m in ms
        for bk, bt, ccv, tm, tn in blocks
    ]


# The 16-wide K blocks serve groups of up to 16 output channels.
# 24-wide K blocks (12-row micro-tiles) for K = 24 / 48 layers: no padded FMAs (IGM 3x3 c24 k24 fp32).
F32_BLOCKS = [
    (64, 32, 4, 8, 8),
    (32, 32, 8, 8, 4),
    (32, 32, 8, 4, 8),
    (16, 32, 4, 8, 4),
    (24, 64, 4, 12, 8),
    (24, 32, 4, 12, 4),
]
# Prefetch depth 2 (RDNA fp32: maskrcnn c256 200² / 14², ssd300 c128 150², stride-2 c128 200²:
# 5-13% on gfx1201, 2-10% on gfx1101).
F32_PF_BLOCKS = [(64, 32, 4, 8, 8)]
F16_BLOCKS = [(64, 32, 2, 8, 8), (32, 32, 8, 8, 4), (16, 32, 2, 8, 4)]
BF16_BLOCKS = [(64, 32, 4, 8, 8), (32, 32, 8, 8, 4), (16, 32, 4, 8, 4)]
CATALOG = {
    "FLOAT": sreg([0])
    + fused(F32_BLOCKS)
    + fused(F32_PF_BLOCKS, pf=1)
    + multipass((4, 2, 3), "none")
    + multipass((4, 2, 3), "llc"),
    "HALF": coop(COOP_F16)
    + coop(COOP3_F16, m=3)
    + coop(COOP_CDNA_F16, ("gfx9",))
    + coop(COOP3_CDNA_F16, ("gfx9",), m=3)
    + coop(COOPPF_CDNA_F16, ("gfx9",), pf=1)
    + wreg(WREG_F16)
    + wmma()
    + sreg([1, 0])
    + fused(F16_BLOCKS)
    + multipass((2, 3), "none")
    + multipass((2, 3), "llc"),
    "BFLOAT16": coop(COOP_BF16)
    + coop(COOP3_BF16, m=3)
    + coop(COOP_CDNA_BF16, ("gfx9",))
    + coop(COOP3_CDNA_BF16, ("gfx9",), m=3)
    + coop(COOPPF_CDNA_BF16, ("gfx9",), pf=1)
    + wreg(WREG_BF16)
    + wmma()
    + fused(BF16_BLOCKS)
    + multipass((2, 3), "none")
    + multipass((2, 3), "llc"),
}


# Backward weights, F(3,r) (m = r: 2, 4 or 6), x and dy read channels-first (packed channels-last
# ones transposed in, any other layout copied). The GEMM per
# Winograd position: (out channels, in channels, tiles per step) and tm x tn = the FMA thread tile
# ("any": fp32 operands; f16math 1: fp16 operands through v_dot2_f32_f16) or a wave's 16x16 atoms
# (MFMA "gfx9": fp32 or fp16; "gfx950" 16x16x32 and WMMA "gfx11" / "gfx12": fp16).
# Each dtype's configs stay inside its error band against an fp64 reference (the dynamic-tolerance
# model): fp16 operands only for fp16 graphs' F(3,2); bf16 graphs on fp32 operands only; fp32
# graphs F(3,2) and F(3,4) (the latter admitted from 512 products per dW element on).
def wrw(arch, tiles, rs=(2, 4, 6), f16math=0):
    return [
        C(
            "wrw",
            r,
            lay,
            arch=arch,
            gemm=(bm, bn, bk),
            tm=tm,
            tn=tn,
            waves=0 if arch == "any" else (bm // (16 * tm)) * (bn // (16 * tn)),
            f16math=f16math,
        )
        for r in rs
        for lay in ("NCHW",)
        for bm, bn, bk, tm, tn in tiles
    ]


WRW_FMA = [
    (128, 128, 16, 8, 8),
    (64, 64, 16, 4, 4),
    (128, 64, 16, 8, 4),
    (64, 128, 16, 4, 8),
    (128, 128, 8, 8, 8),
]
WRW_ATOM = [
    (128, 128, 32, 4, 4),
    (64, 64, 32, 2, 2),
    (128, 128, 64, 4, 4),
    (128, 64, 32, 4, 2),
]
WRW_CATALOG = {
    "FLOAT": wrw("gfx9", WRW_ATOM, (2, 4)) + wrw("any", WRW_FMA, (2, 4)),
    # fp16: F(3,2) on fp16 operands (matrix cores; FMA through v_dot2 for tile steps of 16+), and
    # F(3,4) / F(3,6) on fp32 operands (CDNA fp32 MFMA, FMA).
    "HALF": wrw("gfx9", WRW_ATOM)
    + wrw("gfx950", WRW_ATOM, (2,))
    + wrw("gfx11", WRW_ATOM, (2,))
    + wrw("gfx12", WRW_ATOM, (2,))
    + wrw("any", WRW_FMA)
    + wrw("any", [t for t in WRW_FMA if t[2] >= 16], (2,), 1),
    # bf16: fp32 operands (CDNA fp32 MFMA for F(3,4) / F(3,6), FMA for every tile).
    "BFLOAT16": wrw("gfx9", WRW_ATOM, (4, 6)) + wrw("any", WRW_FMA),
}


def label(c):
    if c["variant"] == "wrw":
        return f"wrw.f3x{c['m']}.{c['layout'].lower()}.{c['arch']}.{c['gemm_bm']}x{c['gemm_bn']}x{c['gemm_bk']}t{c['tm']}{c['tn']}{'.h' if c['f16math'] else ''}"
    if c["variant"] == "multipass":
        return f"multipass.f{c['m']}.{c['layout'].lower()}.g{c['gemm_bm']}x{c['gemm_bn']}x{c['gemm_bk']}.{'llc' if c['llc'] else 'none'}"
    if c["variant"] == "fused":
        return f"fused.f{c['m']}.nchw.{c['bk']}x{c['bt']}c{c['ccv']}t{c['tm']}{c['tn']}{'.pf1' if c['pf'] else ''}"
    if c["variant"] == "wreg":
        return f"wreg.f2.nchw.{c['arch']}.{c['bk']}g{c['bt'] // 16}b{c['bins']}{'h' if c['f16math'] else 'f'}"
    if c["variant"] == "coop":
        kw = c["waves"] // 4
        return f"coop.f{c['m']}.nchw.{c['arch']}.m{c['bk'] // (16 * kw)}n{c['bt'] // 16}k{kw}{'h' if c['f16math'] else 'f'}{'.pf1' if c['pf'] else ''}"
    if c["variant"] == "sreg":
        return f"sreg.f2.nchw.{c['arch']}.{c['bk']}{'h' if c['f16math'] else 'f'}"
    return f"wmma.f2.nchw.{c['arch']}.{c['bk']}w{c['waves']}"


kmd, umd, udd = U("kmd"), U("umd"), U("udd")
w(
    "conv_winograd.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_winograd config fields",
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
        "name": "kernel direction, dtype, variant (device), layout and plan match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_winograd.kernel_match",
    },
)
w(
    "conv_winograd.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_winograd dispatch",
        "dispatch_symbol": "hipkernel.conv_winograd.dispatch",
    },
)
total = 0
for direction, tag in (("FWD", "fwd"), ("BWD", "bwd"), ("WRW", "wrw")):
    ued = U("ued." + tag)
    w(
        f"conv_winograd_{tag}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_WINOGRAD_{direction}",
            "graph_match": {"native": f"hipkernel.conv_winograd_{tag}.graph_match"},
            "metadata": kmd,
            "knobs": KNOBS,
            "behavior_notes": ["runtime_compilation"],
            # RFC 0020 4.2 numerical notes: held as authored, not yet surfaced by hipDNN.
            "numerical_notes": ["WINOGRAD"],
        },
    )
    kernels = []
    for dtype, cfgs in (WRW_CATALOG if direction == "WRW" else CATALOG).items():
        dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}[dtype]
        for i, c in enumerate(cfgs):
            name = f"conv_winograd.{tag}.{dt}.{label(c)}"
            kernels.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvWinograd.cpp",
                        "entry_point": ENTRY[c["variant"]],
                    },
                    "metadata": {"direction": direction, "dtype": dtype, **c},
                    "priority": 1000 - i,
                }
            )
    total += len(kernels)
    w(
        f"conv_winograd_{tag}.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp." + tag),
            "name": f"hipkernel:conv_winograd_{tag}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
print(total, "kernel configs")
