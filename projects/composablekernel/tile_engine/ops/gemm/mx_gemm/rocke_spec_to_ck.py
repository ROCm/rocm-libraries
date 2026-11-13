# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocKE mx_gemm kernels as CK comp_tdm_scheduled configs, and back.

A rocKE kernel (its name, e.g. ``mx_gemm_mxfp4_M4096N4096K65536_t256x256x512_c4x4_hblate``,
or its spec dict) becomes a tile_engine config (one instance: a single value per tile, trait
and tuning key) or a dispatcher config (unified_mx_gemm_codegen.py). With the problem's M, N
and K the instance gets the shape traits that problem needs exactly (no edge code when M and N
are whole tiles, the K tail, the tile parity), so it is the instance rocKE would build.

tile_engine and the dispatcher build A and B of one type (fp4 or fp8) and an fp16 D. A rocKE
kernel outside that (bf8 or mixed inputs, its default fp8 D) needs ``out_dtype="fp16"`` to
take that D type, and is rejected otherwise.

    python rocke_spec_to_ck.py mx_gemm_mxfp4_M4096N4096K65536_t256x256x512_c4x4_fp16
    python rocke_spec_to_ck.py spec.json --format dispatcher --out-dtype fp16 -o cfg.json
"""

import argparse
import dataclasses
import json
import os
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mx_tdm_scheduled  # noqa: E402
from mx_tdm_scheduled import MxTdmScheduledSpec  # noqa: E402

PIPELINE = "comp_tdm_scheduled"
EPILOGUE = "direct"
SCHEDULER = "intrawave"
CK_OUT_DTYPES = ("fp16",)


def spec_of(kernel) -> MxTdmScheduledSpec:
    """A spec from a rocKE kernel name, a spec dict or a spec."""
    if isinstance(kernel, MxTdmScheduledSpec):
        return kernel
    if isinstance(kernel, str):
        return mx_tdm_scheduled.parse_kernel_name(kernel)[1]
    return MxTdmScheduledSpec.from_dict(kernel)


def _checked(spec: MxTdmScheduledSpec, out_dtype: Optional[str]) -> MxTdmScheduledSpec:
    if (
        spec.a_dtype != spec.b_dtype
        or spec.a_dtype not in mx_tdm_scheduled.TE_DTYPES.values()
    ):
        raise ValueError(
            f"{spec.a_dtype}x{spec.b_dtype}: CK configs take one input type, fp4 or fp8"
        )
    if spec.a_dtype == "bf8":
        raise ValueError("bf8: CK configs take one input type, fp4 or fp8")
    if out_dtype is not None:
        spec = dataclasses.replace(
            spec, out_dtype=mx_tdm_scheduled.TE_OUT_DTYPES[out_dtype]
        )
    out = {v: k for k, v in mx_tdm_scheduled.TE_OUT_DTYPES.items()}[spec.out_dtype]
    if out not in CK_OUT_DTYPES:
        raise ValueError(
            f"D type {spec.out_dtype}: CK configs build {CK_OUT_DTYPES}; pass out_dtype"
        )
    reason = spec.validate(check_shape=bool(spec.M and spec.N and spec.K))
    if reason != "ok":
        raise ValueError(f"{spec.kernel_name()}: rocKE rejects it ({reason})")
    return spec


def _tuning(spec: MxTdmScheduledSpec, shape_traits: bool) -> Dict[str, object]:
    t = mx_tdm_scheduled.te_tuning_from_spec(spec, shape_traits=shape_traits)
    t["cluster_dim"] = list(spec.cluster_dim)
    return t


def to_dispatcher_config(
    kernel,
    gpu_target: str = "gfx1250",
    out_dtype: Optional[str] = None,
    shape_traits: bool = True,
) -> Dict[str, object]:
    """A unified_mx_gemm_codegen.py config (tuning keys at the top level)."""
    spec = _checked(spec_of(kernel), out_dtype)
    return {
        "datatype": spec.a_dtype,
        "layout": "rcr",
        "gpu_target": gpu_target,
        "pipeline": PIPELINE,
        "epilogue": EPILOGUE,
        "scheduler": SCHEDULER,
        "tile_config": mx_tdm_scheduled.tile_from_spec(spec),
        **_tuning(spec, shape_traits),
    }


def to_tile_engine_config(
    kernel, out_dtype: Optional[str] = None, shape_traits: bool = True
) -> Dict[str, object]:
    """A tile_engine mx_gemm config holding exactly this instance."""
    spec = _checked(spec_of(kernel), out_dtype)
    traits = {
        "pipeline": PIPELINE,
        "scheduler": SCHEDULER,
        "epilogue": EPILOGUE,
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "persistent": False,
    }
    return {
        "tile_config": {
            k: {"values": [v]} for k, v in mx_tdm_scheduled.tile_from_spec(spec).items()
        },
        "trait_config": {k: {"values": [v]} for k, v in traits.items()},
        "tuning_config": mx_tdm_scheduled.tuning_config_section(
            [_tuning(spec, shape_traits)]
        ),
        "k_block_per_cu": 1,
    }


def from_dispatcher_config(
    cfg: Dict, M: int = 0, N: int = 0, K: int = 0, out_dtype: str = "fp16"
) -> Tuple[MxTdmScheduledSpec, Dict[str, object]]:
    """The rocKE spec (for M, N, K) and the normalized tuning of a dispatcher config."""
    if cfg.get("pipeline") != PIPELINE:
        raise ValueError(f"not a {PIPELINE} config: pipeline {cfg.get('pipeline')!r}")
    keys = (*mx_tdm_scheduled.TUNING_DEFAULTS, *mx_tdm_scheduled.SHAPE_DEFAULTS)
    tuning = mx_tdm_scheduled.normalize_tuning({k: cfg[k] for k in keys if k in cfg})
    spec = mx_tdm_scheduled.spec_from_tile(
        cfg["tile_config"],
        tuning,
        cfg["datatype"],
        cfg["datatype"],
        mx_tdm_scheduled.TE_OUT_DTYPES[out_dtype],
        M,
        N,
        K,
    )
    return spec, tuning


def from_tile_engine_config(
    config: Dict, M: int = 0, N: int = 0, K: int = 0, datatype: str = "fp4"
) -> Tuple[MxTdmScheduledSpec, Dict[str, object]]:
    """The spec and tuning of a single-instance tile_engine config."""

    def single(section: Dict, key: str):
        values = section[key]["values"]
        if len(values) != 1:
            raise ValueError(f"{key} has {len(values)} values; one instance has one")
        return values[0]

    tile = {k: single(config["tile_config"], k) for k in config["tile_config"]}
    tunings = mx_tdm_scheduled.expand_tuning_config(config.get("tuning_config"))
    if len(tunings) != 1:
        raise ValueError(
            f"tuning_config makes {len(tunings)} tunings; one instance has one"
        )
    if single(config["trait_config"], "pipeline") != PIPELINE:
        raise ValueError(f"not a {PIPELINE} config")
    cfg = {
        "pipeline": PIPELINE,
        "datatype": datatype,
        "tile_config": tile,
        **tunings[0],
    }
    return from_dispatcher_config(cfg, M, N, K)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "kernel", help="a rocKE kernel name, or a JSON file of its spec dict"
    )
    ap.add_argument(
        "--format", choices=("tile_engine", "dispatcher"), default="tile_engine"
    )
    ap.add_argument("--out-dtype", choices=CK_OUT_DTYPES, help="the D type to build")
    ap.add_argument("--gpu-target", default="gfx1250")
    ap.add_argument(
        "--no-shape-traits",
        action="store_true",
        help="rocKE's general traits (any M, N, tile count) instead of the problem's",
    )
    ap.add_argument("-o", "--output", help="write here instead of stdout")
    args = ap.parse_args()

    kernel = args.kernel
    if os.path.isfile(kernel):
        with open(kernel) as f:
            kernel = json.load(f)
    shape_traits = not args.no_shape_traits
    if args.format == "dispatcher":
        out = to_dispatcher_config(
            kernel, args.gpu_target, args.out_dtype, shape_traits
        )
    else:
        out = to_tile_engine_config(kernel, args.out_dtype, shape_traits)
    text = json.dumps(out, indent=4) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
