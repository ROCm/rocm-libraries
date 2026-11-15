# Generates the CONV_DEPTHWISE_VALU_WRW descriptors (backward weights, partial + fold kernels).
# Every config is admitted on correctness alone; autotune (UHD later) ranks them. Floors first:
# with no ranker, declared order is the default.
import json, uuid, pathlib, sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("0a6f3c1e-9b52-4d7e-8e0f-5c2d1b7a4e90")
U = lambda name: str(uuid.uuid5(NS, "wrw." + name))


def w(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


FIELDS = [
    "dtype",
    "layout",
    "variant",
    "kd",
    "kh",
    "kw",
    "q",
    "g",
    "nwg",
    "bpc",
    "mult",
    "s",
    "cv",
    "strip",
    "form",
    "fill",
]
STRINGS = {"dtype", "layout", "variant"}
# The matrix-core fields (q output columns a tile, g input rows a step, nwg 16-channel groups a
# wave, s the stride of both axes, form the kernel body) are 0 for the VALU variants. bpc is the
# blocks a dispatch unit the partitions aim for; 0 on a VALU variant is the default 4. fill 1 lets
# the planner trade partitions for whole rounds of blocks (rounds x rows a partition); fill 0 keeps
# the target.
NO_MMA = dict(q=0, g=0, nwg=0, bpc=0, s=0, cv=0, strip=0, form=0, fill=0)
CONFIGS = [dict(variant="floor", kd=0, kh=0, kw=0, **NO_MMA, mult=1)] + [
    dict(variant="tile", kd=kd, kh=kh, kw=kw, **NO_MMA, mult=1)
    for kd, kh, kw in (
        (1, 3, 3),
        (1, 5, 5),
        (1, 7, 7),
        (1, 1, 4),
        (1, 1, 31),
        (3, 3, 3),
    )
]
# Depth multiplier 2 (two output channels per input channel; VALU variants only): the floor and
# the common 2-D tiles.
CONFIGS += [dict(c, mult=2) for c in CONFIGS if c["kd"] in (0, 1) and c["kh"] != 1]
# Matrix core (gfx908 fp16, gfx90a / gfx942 / gfx950 fp16 and bf16; channels-last 16-bit,
# stride 1 or 2): the schedules that won somewhere on gfx950's ConvNeXt / EfficientNet / MobileNet
# shapes (q, g, nwg, bpc).
HANKEL_SCHEDULES = (
    (32, 1, 1, 2),
    (32, 2, 1, 2),
    (32, 1, 1, 4),
    (32, 2, 1, 4),
    (32, 1, 2, 2),
    (16, 2, 1, 2),
    (16, 2, 2, 1),
    (16, 1, 2, 2),
    # Four input rows a step (more loads in flight): ConvNeXt 7x7 and EfficientNet 3x3 on
    # gfx950, 5-10% over two.
    (16, 4, 1, 2),
    (16, 4, 1, 1),
    (32, 4, 1, 1),
)
# Schedules of the row-staging body (form 1) only: tiles that fit 7-, 24- and 28-column images
# (ConvNeXt 7x7 at 7 and 28 columns, EfficientNet at 24) and four blocks a dispatch unit at two
# rows a step, each the best form-1 schedule on some gfx950 ConvNeXt or EfficientNet shape.
HANKEL_FORM1_SCHEDULES = (
    (8, 4, 2, 2),
    (8, 4, 1, 4),
    (8, 4, 1, 1),
    (16, 2, 1, 4),
    (24, 4, 1, 2),
    (28, 2, 1, 1),
    (28, 4, 1, 1),
)
# Stride 2 (both axes; EfficientNet / MobileNet downsampling layers) stages each x row as its
# even and odd columns: the same schedules. Every schedule ships in both bodies (form 0 and 1),
# the form-1 schedules in form 1 alone.
HANKEL = [
    dict(
        variant="hankel",
        kd=1,
        kh=k,
        kw=k,
        q=q,
        g=g,
        nwg=nwg,
        bpc=bpc,
        mult=1,
        s=s,
        cv=0,
        strip=0,
        form=form,
        fill=0,
    )
    for s in (1, 2)
    for k in (3, 5, 7)
    for (q, g, nwg, bpc), forms in [(c, (0, 1)) for c in HANKEL_SCHEDULES]
    + [(c, (1,)) for c in HANKEL_FORM1_SCHEDULES]
    for form in forms
]

# Vector tiles (VALU, channels-last, unit width stride, every type and architecture): cv
# channels a thread read as one vector, strips of `strip` output columns; (cv, strip) per filter
# within the register budget.
VTILE_SHAPES = {
    3: ((4, 8), (8, 4), (2, 8)),
    5: ((4, 4), (2, 8)),
    7: ((2, 8), (2, 4), (1, 8)),
}
VTILE = [
    dict(
        variant="vtile",
        kd=1,
        kh=k,
        kw=k,
        q=0,
        g=0,
        nwg=0,
        bpc=0,
        mult=1,
        s=0,
        cv=cv,
        strip=strip,
        form=0,
        fill=0,
    )
    for k, shapes in VTILE_SHAPES.items()
    for cv, strip in shapes
]
# Partition targets besides the default for the register and vector tiles: few rows a partition
# re-read the filter's halo rows (gfx950 ConvNeXt c512 14^2 n16 fp32: 1.6x at 1 or 2 blocks a
# unit; NCHW fp16 at one partition, 1.6x), large batches fill more (EfficientNet c1536 12^2
# n128: 1.15-1.2x near 8).
VALU_BPC = (1, 2, 8)
CONFIGS += [dict(c, bpc=b) for c in CONFIGS if c["variant"] == "tile" for b in VALU_BPC]
VTILE += [dict(c, bpc=b) for c in VTILE for b in VALU_BPC]
# The same schedules with the partitions traded for whole rounds of blocks (fill 1): small
# batches gain (gfx950 EfficientNet c960 30^2 n16: 1.12x); some n128 shapes run 4-5% slower, so
# both plans ship and autotune picks. The row-staging Hankel body (form 1) ships at fill 1 only in
# the schedules where whole rounds beat the target on gfx950 (four blocks a unit at two rows a
# step: EfficientNet c1536 12^2 and c960 30^2 n16, 1.11-1.13x); elsewhere its fill 1 measured
# within 2% of fill 0 or slower.
HANKEL_FORM1_FILL_SCHEDULES = ((16, 2, 1, 4), (32, 2, 1, 4))
CONFIGS += [dict(c, fill=1) for c in CONFIGS if c["variant"] == "tile"]
VTILE += [dict(c, fill=1) for c in VTILE]
HANKEL += [
    dict(c, fill=1)
    for c in HANKEL
    if c["form"] == 0
    or (c["q"], c["g"], c["nwg"], c["bpc"]) in HANKEL_FORM1_FILL_SCHEDULES
]

kmd, umd, udd, ued = U("kmd"), U("umd"), U("udd"), U("ued")
w(
    "conv_depthwise_valu_wrw.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_depthwise_valu_wrw variant fields",
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
        "name": "kernel dtype, layout and variant constraints match the graph",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_depthwise_valu_wrw.kernel_match",
    },
)
w(
    "conv_depthwise_valu_wrw.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_depthwise_valu_wrw dispatch",
        "dispatch_symbol": "hipkernel.conv_depthwise_valu_wrw.dispatch",
    },
)
w(
    "conv_depthwise_valu_wrw.ued.json",
    {
        "version": "1.0",
        "id": ued,
        "name": "hipkernel:CONV_DEPTHWISE_VALU_WRW",
        "graph_match": {"native": "hipkernel.conv_depthwise_valu_wrw.graph_match"},
        "metadata": kmd,
        "knobs": [
            "variant",
            "q",
            "g",
            "nwg",
            "bpc",
            "mult",
            "s",
            "cv",
            "strip",
            "form",
            "fill",
        ],
        "behavior_notes": ["runtime_compilation"],
        # RFC 0025 numerical notes. No "nondeterministic": every config is
        # bitwise reproducible (fixed-order sums, no atomics), which the
        # family's device test checks by relaunching each config.
        "numerical_notes": [],
    },
)
kernels = []
for layout in ("NHWC", "NCHW"):
    for c in CONFIGS:
        for dtype, dt in (("FLOAT", "f32"), ("HALF", "f16"), ("BFLOAT16", "bf16")):
            k = (
                f"{c['kh']}x{c['kw']}"
                if c["kd"] == 1
                else f"{c['kd']}x{c['kh']}x{c['kw']}"
            )
            tag = "floor" if c["variant"] == "floor" else f"tile_k{k}"
            tag += f".m{c['mult']}" if c["mult"] != 1 else ""
            tag += f".b{c['bpc']}" if c["bpc"] != 0 else ""
            tag += ".fill" if c["fill"] else ""
            name = f"conv_depthwise_valu_wrw.{dt}.{layout.lower()}.{tag}"
            kernels.append(
                {
                    "version": "1.0",
                    "id": U(name),
                    "name": name,
                    "kernel_source": {
                        "kind": "embedded_source",
                        "source_file": "kernels/ConvDepthwiseValuWrw.cpp",
                        "entry_point": "ConvDepthwiseValuWrwPartial",
                    },
                    "metadata": {"dtype": dtype, "layout": layout, **c},
                    "priority": 0,
                }
            )
for c in HANKEL:
    for dtype, dt in (("HALF", "f16"), ("BFLOAT16", "bf16")):
        tag = f"hankel_k{c['kh']}x{c['kw']}.q{c['q']}g{c['g']}w{c['nwg']}b{c['bpc']}"
        tag += f".s{c['s']}" if c["s"] != 1 else ""
        tag += f".f{c['form']}" if c["form"] != 0 else ""
        tag += ".fill" if c["fill"] else ""
        name = f"conv_depthwise_valu_wrw.{dt}.nhwc.{tag}"
        kernels.append(
            {
                "version": "1.0",
                "id": U(name),
                "name": name,
                "kernel_source": {
                    "kind": "embedded_source",
                    "source_file": "kernels/ConvDepthwiseValuWrw.cpp",
                    "entry_point": "ConvDepthwiseValuWrwPartial",
                },
                "metadata": {"dtype": dtype, "layout": "NHWC", **c},
                "priority": 0,
            }
        )
for c in VTILE:
    for dtype, dt in (("FLOAT", "f32"), ("HALF", "f16"), ("BFLOAT16", "bf16")):
        name = f"conv_depthwise_valu_wrw.{dt}.nhwc.vtile_k{c['kh']}x{c['kw']}.v{c['cv']}s{c['strip']}"
        name += f".b{c['bpc']}" if c["bpc"] != 0 else ""
        name += ".fill" if c["fill"] else ""
        kernels.append(
            {
                "version": "1.0",
                "id": U(name),
                "name": name,
                "kernel_source": {
                    "kind": "embedded_source",
                    "source_file": "kernels/ConvDepthwiseValuWrw.cpp",
                    "entry_point": "ConvDepthwiseValuWrwPartial",
                },
                "metadata": {"dtype": dtype, "layout": "NHWC", **c},
                "priority": 0,
            }
        )
# Floors first across both layouts.
kernels.sort(key=lambda k: k["metadata"]["variant"] != "floor")
w(
    "conv_depthwise_valu_wrw.kdp.json",
    {
        "version": "1.0",
        "id": U("kdp"),
        "name": "hipkernel:conv_depthwise_valu_wrw",
        "arch": [],
        "matchers": [umd],
        "engine": ued,
        "dispatch": udd,
        "kernelDescriptors": kernels,
    },
)
print(len(kernels), "kernel configs")
