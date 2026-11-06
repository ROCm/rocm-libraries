# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate ML4 router, UG and down-GEMV stage functions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from . import moe
from .mono_layout import from_aiter
from .stage_fn import (
    DEFAULT_VARIANTS,
    StageKnobs,
    StageSpec,
    lower_stage,
    parse_variant,
)
from ..qwen38_mono.device_function import DeviceFunctionKnobs, write_device_function

MOE_STAGES = ("router", "ug", "down_gemv")


def pair_id(variant: str, stages, knobs: StageKnobs, mutant: str | None) -> str:
    payload = json.dumps(
        ["ml4", variant, list(stages), knobs.suffix(), mutant or ""], sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def sidecar_extra(stage: str, stages, pair: str, knobs: StageKnobs, layout, variant: str, rows: int, mutant):
    order = {"router": 0, "ug": 1, "down_gemv": 2}
    requires = {}
    for prior in stages:
        if order[prior] < order[stage]:
            requires[prior] = pair
    out = {
        "family": "ml4_mono",
        "variant": variant,
        "pair": pair,
        "mailboxes": "plain" if knobs.mailbox_impl == "asm" else "ir",
        "requires": requires,
        "stage_knobs": knobs.suffix() or "default",
        "aiter_layout": {
            "source": str(layout.source_dir),
            "sha256": layout.source_digest,
            "tasks": dict(layout.stage_tasks(rows)),
            "bases": layout.stage_bases(rows),
            "arena_words": layout.arena_words(rows),
        },
    }
    if mutant:
        out["mutant"] = mutant
    if stage == "down_gemv":
        # This extern intentionally replaces only FlyDSL's GEMV region.  The
        # peer exchange, residual add, and x_out store remain in its epilogue.
        out.update(
            {
                "replacement": "down_gemv_only",
                "epilogue": "flydsl_peer_reduce",
                "partial_output": "lds:outs",
            }
        )
    return out


def write_functions(
    out_dir,
    variants,
    *,
    stages=MOE_STAGES,
    layout,
    knobs=StageKnobs(),
    arena_words=None,
    arch="gfx950",
    flavor=None,
    bitcode=False,
    mutant=None,
):
    if knobs.mailbox_impl == "ir":
        from ..qwen38_mono import mailbox_ir  # noqa: F401
    moe.MUTATE = mutant
    written = []
    for variant in variants:
        rows, seq, group = parse_variant(variant)
        if group:
            raise ValueError(f"{variant}: grouped ML4 builds are not covered")
        if rows > 4 or (seq and rows < 2):
            raise ValueError(f"{variant}: unsupported ML4 variant")
        have = layout.arena_words(rows) if arena_words is None else arena_words
        pair = pair_id(variant, stages, knobs, mutant)
        for stage in stages:
            spec = StageSpec(stage, rows, seq, knobs=knobs)
            fn = lower_stage(spec, layout, arch=arch, flavor=flavor)
            if fn.lds_bytes > have * 4:
                raise ValueError(
                    f"{stage} {variant}: LDS arena {fn.lds_bytes} bytes > "
                    f"aiter arena {have * 4} bytes"
                )
            stem = f"{stage}_{variant}"
            paths = write_device_function(fn, out_dir, stem=stem, bitcode=bitcode)
            with open(paths["json"], encoding="utf-8") as f:
                meta = json.load(f)
            meta.update(sidecar_extra(stage, stages, pair, knobs, layout, variant, rows, mutant))
            with open(paths["json"], "w", encoding="utf-8", newline="\n") as f:
                json.dump(meta, f, indent=1)
                f.write("\n")
            written.append(
                {"stage": stage, "variant": variant, "lds_bytes": fn.lds_bytes, **paths}
            )
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="emit ML4 rocKE MoE stage functions (router, ug, down_gemv)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--stages", default=",".join(MOE_STAGES))
    ap.add_argument("--aiter-root", default=None, help="aiter checkout or ML4 gfx950 directory")
    ap.add_argument("--mono-lds-words", type=int, default=None)
    ap.add_argument("--no-arena-check", action="store_true")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    ap.add_argument("--mailbox-impl", choices=("asm", "ir"), default="asm")
    ap.add_argument("--inline", choices=("default", "always", "never"), default="always")
    ap.add_argument(
        "--mutate",
        choices=("router_score", "ug_scale", "down_order"),
        default=None,
        help="emit a deliberately different body for the bitwise mutant gate",
    )
    args = ap.parse_args(argv)
    stages = tuple(x for x in args.stages.split(",") if x)
    bad = set(stages) - set(MOE_STAGES)
    if bad:
        raise SystemExit(f"--stages: unknown {sorted(bad)} (known {list(MOE_STAGES)})")
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    layout = from_aiter(args.aiter_root)
    knobs = StageKnobs(
        mailbox_impl=args.mailbox_impl,
        fn=DeviceFunctionKnobs(lds="arg", inline=args.inline),
    )
    if args.no_arena_check:
        arena = 1 << 30
    elif args.mono_lds_words is not None:
        arena = args.mono_lds_words
    else:
        arena = None
    bitcode = {"auto": "auto", "yes": True, "no": False}[args.bitcode]
    written = write_functions(
        args.out,
        variants,
        stages=stages,
        layout=layout,
        knobs=knobs,
        arena_words=arena,
        arch=args.arch,
        flavor=args.flavor,
        bitcode=bitcode,
        mutant=args.mutate,
    )
    print(
        json.dumps(
            {
                "family": "ml4_mono",
                "stages": list(stages),
                "variants": variants,
                "mutant": args.mutate,
                "written": written,
            },
            indent=1,
        )
    )
    return 0


__all__ = ["MOE_STAGES", "pair_id", "sidecar_extra", "write_functions", "main"]


if __name__ == "__main__":
    sys.exit(main())
