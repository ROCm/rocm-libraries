#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/tensor_pack_emit.py -- Python reference emitter for the tensor pack
# (instances/common/tensor/pack.py): every form (pairs, tiled, elements), both dtypes, strided
# sources, group / block variants and specs both engines reject. C twin: tensor_pack_emit.c (same
# rows). Generated rows: edit both files together.
import sys

from _emit_common import run_emit

ARCH = "gfx950"

ROWS = [
    {
        "name": "pairs_f16",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "pairs_bf16",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "bf16",
        "kw": {},
    },
    {
        "name": "pairs_g1",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"groups": 1},
    },
    {
        "name": "pairs_g4_b128",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"groups": 4, "block": 128},
    },
    {
        "name": "pairs_b512",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"block": 512},
    },
    {
        "name": "pairs_b64",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"block": 64},
    },
    {
        "name": "tiled_f16",
        "src": ["M1", "M0", "K1", "K0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "tiled_bf16",
        "src": ["M1", "M0", "K1", "K0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "bf16",
        "kw": {},
    },
    {
        "name": "tiled_ragged",
        "src": ["M1", "M0", "K1", "K0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 40, "M1": 36, "K0": 24, "K1": 26},
        "strides": None,
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "tiled_kfirst",
        "src": ["K1", "K0", "M1", "M0"],
        "dst": ["M1", "M0", "K1", "K0"],
        "ext": {"M0": 72, "M1": 64, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "elements_odd",
        "src": ["M1", "K0", "M0"],
        "dst": ["K0", "M1", "M0"],
        "ext": {"M0": 5, "M1": 8, "K0": 3},
        "strides": None,
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "elements_bf16",
        "src": ["M1", "K0", "M0"],
        "dst": ["K0", "M1", "M0"],
        "ext": {"M0": 5, "M1": 8, "K0": 3},
        "strides": None,
        "dtype": "bf16",
        "kw": {},
    },
    {
        "name": "force_elements",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"form": "elements"},
    },
    {
        "name": "force_pairs",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"form": "pairs"},
    },
    {
        "name": "strided_pairs",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": {"M0": 1, "K0": 64, "M1": 1600, "K1": 52000},
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "strided_tiled",
        "src": ["M1", "M0", "K1", "K0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": {"K0": 1, "K1": 32, "M0": 800, "M1": 40000},
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "strided_elems",
        "src": ["M1", "M0", "K1", "K0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": {"K0": 3, "K1": 80, "M0": 2000, "M1": 100000},
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "bad_numel",
        "src": ["M0", "K0"],
        "dst": ["K0", "M0"],
        "ext": {"M0": 5, "K0": 3},
        "strides": None,
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "bad_tiled_block",
        "src": ["M1", "M0", "K1", "K0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"form": "tiled", "block": 128},
    },
    {
        "name": "bad_pairs",
        "src": ["M1", "M0", "K1", "K0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"form": "pairs"},
    },
    {
        "name": "bad_groups",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24},
        "strides": None,
        "dtype": "fp16",
        "kw": {"groups": 3},
    },
    {
        "name": "bad_modes",
        "src": ["K1", "M1", "K0", "M0"],
        "dst": ["K1", "K0", "M1", "M2"],
        "ext": {"M0": 48, "M1": 32, "K0": 24, "K1": 24, "M2": 2},
        "strides": None,
        "dtype": "fp16",
        "kw": {},
    },
    {
        "name": "bad_span",
        "src": ["M1", "M0", "K1", "K0"],
        "dst": ["K1", "K0", "M1", "M0"],
        "ext": {"M0": 1024, "M1": 1024, "K0": 1024, "K1": 2},
        "strides": None,
        "dtype": "fp16",
        "kw": {},
    },
]


def _spec(idx: int):
    from rocke.instances.common.tensor.pack import TensorPackSpec, is_valid_spec

    if not 0 <= idx < len(ROWS):
        raise SystemExit(f"unknown config index {idx}")
    r = ROWS[idx]
    spec = TensorPackSpec.make(
        r["src"],
        r["dst"],
        r["ext"],
        strides_src=r["strides"],
        dtype=r["dtype"],
        **r["kw"],
    )
    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        sys.stderr.write(f"invalid spec: {why}\n")
        sys.exit(1)
    return spec, ARCH


def _build(spec, arch):
    from rocke.instances.common.tensor.pack import build_tensor_pack

    return build_tensor_pack(spec, arch=arch)


def main() -> int:
    return run_emit(_spec, _build, usage="usage: tensor_pack_emit.py <config_index>\n")


if __name__ == "__main__":
    sys.exit(main())
