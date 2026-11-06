# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate the paired ML4 expert-major fused-MoE scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from types import SimpleNamespace

from rocke.core.ir import I32, I64, IRBuilder

from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function, write_device_function
from . import moe_fused as MF
from .mono_layout import from_aiter
from .stage_fn import ML4_FAMILY, parse_variant


def pair_id(variant: str, knobs: MF.FusedKnobs) -> str:
    return hashlib.sha256(
        json.dumps(["ml4-fused", variant, knobs.suffix()], sort_keys=True).encode()
    ).hexdigest()[:16]


def build_function(stage: str, rows: int, seq: bool, layout, knobs: MF.FusedKnobs):
    if stage not in MF.STAGES:
        raise ValueError(stage)
    plan = MF.FusedPlan(rows, knobs=knobs)
    plan.validate()
    b = IRBuilder(f"rocke_ml4_fused_{stage}_r{rows}" + ("s" if seq else ""))
    b.kernel.attrs["max_workgroup_size"] = layout.threads
    vals = {}
    for name, typ in ML4_FAMILY.abi_params:
        vals[name] = b.param(name, I32 if typ == "i32" else I64)
    if stage == "m3":
        e = MF.FusedCtx(b, SimpleNamespace(**vals), plan)
        e.emit()
    b.ret()
    return b.kernel, plan


def write_functions(out_dir, variants, *, layout, knobs=MF.FusedKnobs(), flavor=None, arch="gfx950", bitcode=False):
    written = []
    for variant in variants:
        rows, seq, group = parse_variant(variant)
        if group:
            raise ValueError(f"{variant}: grouped builds are not covered")
        pair = pair_id(variant, knobs)
        for stage in MF.STAGES:
            kernel, plan = build_function(stage, rows, seq, layout, knobs)
            fn = lower_device_function(
                kernel,
                arch=arch,
                symbol=f"rocke_ml4_fused_{stage}_{variant}",
                llvm_flavor=flavor,
                knobs=DeviceFunctionKnobs(lds="arg", inline="always"),
            )
            if stage == "m3" and fn.lds_bytes > layout.arena_words(rows) * 4:
                raise ValueError(f"{stage} {variant}: fused LDS exceeds aiter arena")
            paths = write_device_function(fn, out_dir, stem=f"{stage}_{variant}", bitcode=bitcode)
            with open(paths["json"], encoding="utf-8") as f:
                meta = json.load(f)
            meta.update(
                {
                    "family": "ml4_mono",
                    "fused": True,
                    "pair": pair,
                    "requires": {other: pair for other in MF.STAGES if other != stage},
                    "mailboxes": "plain",
                    "stage_knobs": knobs.suffix() or "scaffold",
                    "open_todo": "Implement expert-major router/UG/down/combine stream",
                }
            )
            with open(paths["json"], "w", encoding="utf-8", newline="\n") as f:
                json.dump(meta, f, indent=1)
                f.write("\n")
            written.append({"stage": stage, "variant": variant, **paths, "lds_bytes": fn.lds_bytes})
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit paired ML4 expert-major fused-MoE scaffold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="r1,r2,r4,r4s")
    ap.add_argument("--aiter-root", default=None)
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    ap.add_argument("--no-arena-check", action="store_true")
    args = ap.parse_args(argv)
    layout = from_aiter(args.aiter_root)
    knobs = MF.FusedKnobs()
    arena_layout = layout
    # ``--no-arena-check`` is intentionally accepted for parity with the
    # split generator; the scaffold's tiny arena fits every ML4 variant.
    _ = args.no_arena_check
    bitcode = {"auto": "auto", "yes": True, "no": False}[args.bitcode]
    variants = [v for v in args.variants.split(",") if v]
    written = write_functions(
        args.out,
        variants,
        layout=arena_layout,
        knobs=knobs,
        flavor=args.flavor,
        arch=args.arch,
        bitcode=bitcode,
    )
    print(json.dumps({"fused": True, "written": written}, indent=1))
    return 0


__all__ = ["build_function", "main", "pair_id", "write_functions"]


if __name__ == "__main__":
    sys.exit(main())
