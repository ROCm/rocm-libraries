"""Writes every descriptor of hipkernel:hip_conv_fwd_rdna in this folder.

Run from this folder after changing a variant axis or a descriptor field:
    python3 generate_descriptors.py .
Every id is a uuid5 of the descriptor kind and name, so regenerating keeps every id and a
new descriptor needs no hand-picked one. Renaming a descriptor changes its id.
"""

import json
import sys
import uuid
from pathlib import Path

NAMESPACE = uuid.UUID("6b1f0f5e-3c7a-4d0e-9a51-2f4c8e7d1a30")


def uid(kind, name):
    """The id of the `kind` descriptor (ued, udd, kmd, uhd, umd, kdp, ukd) named `name`."""
    return str(uuid.uuid5(NAMESPACE, f"{kind}:{name}"))


# The RDNA targets, listed one by one rather than as gfx11-generic / gfx12-generic, so each
# build compiles its own targets and the packer stamps each shard with its architecture.
ARCH = [
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
    "gfx1200",
    "gfx1201",
]

DTYPES = [("f32", "FLOAT", 0), ("f16", "HALF", 1), ("bf16", "BFLOAT16", 2)]
TILES = [32, 64, 128]
LAYOUTS = [("nchw", "CHANNELS_FIRST", 0), ("nhwc", "CHANNELS_LAST", 1)]

FWD_ENGINE_NAME = "hipkernel:hip_conv_fwd_rdna"
FWD_SYMBOL = "hipkernel.hip_conv_fwd_rdna"
FWD_FIELDS = ["kernel_family", "dtype", "layout", "tile_n"]

FWD_UED = uid("ued", FWD_ENGINE_NAME)
FWD_UDD_NAME = "hip_conv_fwd_rdna dispatch"
FWD_KMD_NAME = "hip_conv_fwd_rdna variant fields"
FWD_UHD_NAME = "hip_conv_fwd_rdna selector"
FWD_UMD_NAME = (
    "hip_conv_fwd_rdna: the kernel serves the graph's family, dtype, layout and tile"
)


def contract(engine, kmd, fields):
    return {
        "specialization_contract": {
            "schema_version": 1,
            "consumers": [
                {
                    "engine_id": engine,
                    "kmd_id": kmd,
                    "metadata_fields": [],
                    "matcher_only_fields": fields,
                    "bindings": {},
                    "vocabulary": {},
                }
            ],
        }
    }


def fwd_descriptors():
    """The conv_fwd_rdna.* documents, keyed by file name."""
    kmd = uid("kmd", FWD_KMD_NAME)
    umd = uid("umd", FWD_UMD_NAME)
    udd = uid("udd", FWD_UDD_NAME)
    uhd = uid("uhd", FWD_UHD_NAME)

    kernels = []
    for short, dtype, dcode in DTYPES:
        for lshort, layout, lcode in LAYOUTS:
            for tile in TILES:
                name = f"hip_conv_fwd_rdna.fma_{short}_{lshort}_tile_n{tile}"
                kernels.append(
                    {
                        "version": "1.0",
                        "id": uid("ukd", name),
                        "name": name,
                        "kernel_source": {
                            "kind": "hip",
                            "source": "kernels/ConvFwdFmaIgemm.cpp",
                            "entry": "ConvFwdFmaIgemm",
                            "build": {
                                "defines": {
                                    "CONV_DTYPE": dcode,
                                    "CONV_CHANNELS_LAST": lcode,
                                    # Channels-first weights are read in place; channels-last
                                    # weights are K x taps x C already.
                                    "CONV_WEI_KCZYX": 1 - lcode,
                                    "TILE_N": tile,
                                }
                            },
                        },
                        "metadata": {
                            "kernel_family": "fma_igemm",
                            "dtype": dtype,
                            "layout": layout,
                            "tile_n": tile,
                        },
                        "priority": 0,
                        "provenance": contract(FWD_UED, kmd, FWD_FIELDS),
                    }
                )

    return {
        "conv_fwd_rdna.ued.json": {
            "version": "1.0",
            "id": FWD_UED,
            "name": FWD_ENGINE_NAME,
            "graph_match": {"native": f"{FWD_SYMBOL}.graph_match"},
            "metadata": kmd,
            "sort_kernel_catalog": {"default": uhd},
        },
        "conv_fwd_rdna.udd.json": {
            "version": "1.0",
            "id": udd,
            "name": FWD_UDD_NAME,
            "dispatch_symbol": f"{FWD_SYMBOL}.dispatch",
        },
        "conv_fwd_rdna.kmd.json": {
            "version": "1.0",
            "id": kmd,
            "name": FWD_KMD_NAME,
            "fields": [
                {"name": "kernel_family", "type": "string"},
                {"name": "dtype", "type": "string"},
                {"name": "layout", "type": "string"},
                {"name": "tile_n", "type": "int"},
            ],
        },
        "conv_fwd_rdna.uhd.json": {
            "version": "1.0",
            "id": uhd,
            "name": FWD_UHD_NAME,
            "adapter": "native",
            "objective": "max",
            "native": {"symbol": f"{FWD_SYMBOL}.score"},
        },
        "conv_fwd_rdna.kernel_match.umd.json": {
            "version": "1.0",
            "id": umd,
            "name": FWD_UMD_NAME,
            "scope": "kernel",
            "match_symbol": f"{FWD_SYMBOL}.kernel_match",
        },
        "conv_fwd_rdna.kdp.json": {
            "version": "1.0",
            "id": uid("kdp", FWD_ENGINE_NAME),
            "name": FWD_ENGINE_NAME,
            "arch": ARCH,
            "matchers": [umd],
            "engine": FWD_UED,
            "dispatch": udd,
            "kernelDescriptors": kernels,
        },
    }


def write(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for file_name, doc in fwd_descriptors().items():
        write(out / file_name, doc)
