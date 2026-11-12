# Generates the CONV_FFT_FWD / CONV_FFT_BWD / CONV_FFT_WRW descriptors: overlap-save tiled FFT
# convolution (descriptors/conv/fft/kernels/ConvFft.cpp). One tier: every config is generic
# (filter, dilation, padding, extents and channels are runtime); a config is a transform size,
# its transform block (tiles a block, threads) and the product tile. The pack admits a config only
# above the FFT admission floor (ConvFftProblem.hpp), so common small-filter graphs gain no
# candidates.
#
# Every config-distinguishing field is an integer knob; direction and dtype are descriptive and
# checked by the kernel match.
import json, pathlib, sys, uuid

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
NS = uuid.UUID("3c8f1d42-7a9e-4b65-b0d1-5e2f7c9a8b13")
U = lambda name: str(uuid.uuid5(NS, "fft." + name))


def write(name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n")


INTS = ["th", "tw", "pb", "tb", "gn", "tm"]
STRINGS = ["direction", "dtype"]

# Transform sizes: 2-D square tiles for square filters up to 63 taps of span, 1-D rows for
# 1-D (and 1 x s) filters. Larger tiles amortise a larger filter; smaller ones waste less on
# small images. Per tile, transform blocks of pb tiles (more tiles a block: longer contiguous
# runs of the spectra, more LDS) and tb threads (at least one a row pair).
TILES = {
    (16, 16): [8, 16],
    (32, 32): [2, 4, 8],
    (64, 64): [1, 2],
    (1, 32): [64, 128],
    (1, 64): [32, 64],
}
# Product tiles (gn output channels, tm tiles a thread): 64 x 64 and 128 x 64 for wide channel
# counts, 256 tiles x 16 channels for narrow ones.
GNS = [(64, 4), (64, 8), (16, 4)]


def threads_of(th, tw, pb):
    """Threads of a transform block: one a row pair at least (pb th / 2, or pb / 2 for rows), and
    also 256 (more waves for the same LDS, for the loads, stores and column passes)."""
    pairs = pb // 2 if th == 1 else pb * th // 2
    least = min(256, max(64, (pairs + 63) // 64 * 64))
    return sorted({least, 256})


kmd, umd, udd = U("kmd"), U("umd"), U("udd")
write(
    "conv_fft.kmd.json",
    {
        "version": "1.0",
        "id": kmd,
        "name": "conv_fft config fields",
        "fields": [{"name": f, "type": "string"} for f in STRINGS]
        + [{"name": f, "type": "int"} for f in INTS],
    },
)
write(
    "kernel_matches_graph.umd.json",
    {
        "version": "1.0",
        "id": umd,
        "name": "kernel direction, dtype and transform size serve the graph above the FFT admission floor",
        "scope": "kernel",
        "match_symbol": "hipkernel.conv_fft.kernel_match",
    },
)
write(
    "conv_fft.udd.json",
    {
        "version": "1.0",
        "id": udd,
        "name": "conv_fft dispatch",
        "dispatch_symbol": "hipkernel.conv_fft.dispatch",
    },
)
dt = {"FLOAT": "f32", "HALF": "f16", "BFLOAT16": "bf16"}
for direction in ("fwd", "bwd", "wrw"):
    ued = U("ued." + direction)
    write(
        f"conv_fft_{direction}.ued.json",
        {
            "version": "1.0",
            "id": ued,
            "name": f"hipkernel:CONV_FFT_{direction.upper()}",
            "graph_match": {"native": f"hipkernel.conv_fft_{direction}.graph_match"},
            "metadata": kmd,
            "knobs": INTS,
            "behavior_notes": ["runtime_compilation"],
        },
    )
    kernels = []
    i = 0
    for dtype in ("FLOAT", "HALF", "BFLOAT16"):
        for (th, tw), pbs in TILES.items():
            for pb, tb in [(pb, tb) for pb in pbs for tb in threads_of(th, tw, pb)]:
                for gn, tm in GNS:
                    # Backward weights' product has no tm (it contracts over tiles).
                    if direction == "wrw" and tm != 4:
                        continue
                    name = (
                        f"conv_fft.{direction}.{dt[dtype]}.t{th}x{tw}.pb{pb}.tb{tb}.g{gn}"
                        + (f".m{tm}" if tm != 4 else "")
                    )
                    kernels.append(
                        {
                            "version": "1.0",
                            "id": U(name),
                            "name": name,
                            "kernel_source": {
                                "kind": "embedded_source",
                                "source_file": "kernels/ConvFft.cpp",
                                "entry_point": "ConvFftData",
                            },
                            "metadata": {
                                "direction": direction.upper(),
                                "dtype": dtype,
                                "th": th,
                                "tw": tw,
                                "pb": pb,
                                "tb": tb,
                                "gn": gn,
                                "tm": tm,
                            },
                            "priority": 1000 - i,
                        }
                    )
                    i += 1
    write(
        f"conv_fft_{direction}.kdp.json",
        {
            "version": "1.0",
            "id": U("kdp." + direction),
            "name": f"hipkernel:conv_fft_{direction}",
            "arch": [],
            "matchers": [umd],
            "engine": ued,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    )
    print(direction, len(kernels), "kernel configs")
