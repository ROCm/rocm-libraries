"""Writes conv_fwd_rdna.kdp.json, the variant list of hipkernel:hip_conv_fwd_rdna.

Run from this folder after changing a variant axis: python3 generate_kdp.py .
Every id is a uuid5 of the descriptor name, so regenerating is stable and a new variant
needs no hand-picked id.
"""

import json
import sys
import uuid
from pathlib import Path

NAMESPACE = uuid.UUID("6b1f0f5e-3c7a-4d0e-9a51-2f4c8e7d1a30")


def uid(name):
    return str(uuid.uuid5(NAMESPACE, name))


FWD_ENGINE = "3099827c-98e8-443c-aa7c-9ad3bec8d14a"
FWD_KMD = "549d3ad2-2cb7-42ab-aefa-36933279cb7f"
FWD_UMD = "4eca87aa-649d-45c3-b7d9-fcb0ae6d1ba1"
FWD_UDD = "ccbd5790-6e75-4375-9452-3c16bd8c5316"
ARCH = ["gfx11-generic", "gfx12-generic"]

DTYPES = [("f32", "FLOAT", 0), ("f16", "HALF", 1), ("bf16", "BFLOAT16", 2)]
TILES = [32, 64, 128]
LAYOUTS = [("nchw", "CHANNELS_FIRST", 0), ("nhwc", "CHANNELS_LAST", 1)]


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


def fwd_kdp():
    kernels = []
    for short, dtype, dcode in DTYPES:
        for lshort, layout, lcode in LAYOUTS:
            for tile in TILES:
                name = f"hip_conv_fwd_rdna.fma_{short}_{lshort}_tile_n{tile}"
                kernels.append(
                    {
                        "version": "1.0",
                        "id": uid(name),
                        "name": name,
                        "kernel_source": {
                            "kind": "hip",
                            "source": "kernels/MIOpenConv3dFwdFmaHip.cpp",
                            "entry": "MIOpenConv3dFwdFmaHip",
                            "build": {
                                "defines": {
                                    "CONV3D_DTYPE": dcode,
                                    "CONV3D_NDHWC": lcode,
                                    # Channels-first weights are read in place; channels-last
                                    # weights are K x taps x C already.
                                    "CONV3D_WEI_KCZYX": 1 - lcode,
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
                        "provenance": contract(
                            FWD_ENGINE,
                            FWD_KMD,
                            ["kernel_family", "dtype", "layout", "tile_n"],
                        ),
                    }
                )
    return {
        "version": "1.0",
        "id": "752302de-deb9-43ce-82ae-f2038c1a8e70",
        "name": "hipkernel:hip_conv_fwd_rdna",
        "arch": ARCH,
        "matchers": [FWD_UMD],
        "engine": FWD_ENGINE,
        "dispatch": FWD_UDD,
        "kernelDescriptors": kernels,
    }


def write(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    write(out / "conv_fwd_rdna.kdp.json", fwd_kdp())
