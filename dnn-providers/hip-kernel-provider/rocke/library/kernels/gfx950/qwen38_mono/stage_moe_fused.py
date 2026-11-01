# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The fused MoE (:mod:`.moe_fused`) as hybrid stage functions.

The fused body runs in the mono layer kernel's M3 slot (``m3``: router, gate_up,
down and combine, self-placed from the block id); ``m4`` and ``m5`` are empty
functions, so the mono's own M4 / M5 do not run. The three are one
generation: each sidecar names the others in ``"requires"`` with the
generation's ``"pair"`` id, so ATOM (``extern_stages.resolve``) refuses a build
that links only some of them::

    python -m kernels.gfx950.qwen38_mono.stage_moe_fused --out DIR [--variants r1,r4s]
        [--knobs best | '{"iod_split": 0}'] [--flavor llvm23]

(``--knobs`` defaults to ``best``, the tuned serving sets with knob ``numa=1``; ``best_plain`` the tuned sets
for a server without ATOM's numa lever; a JSON builds those knobs on top of the design defaults, e.g. ``'{}'`` =
first light, which spills at rows > 1.) Bodies built with ``numa=1`` address the
mailboxes of ATOM's ``numa=1`` schedule lever (run them with ``ATOM_Q38_SCHED=numa=1``; the sidecars say so and
ATOM keeps the lever for those builds). Link with
``ATOM_Q38_EXTERN_STAGES=m3=DIR/m3_{variant}.ll,m4=DIR/m4_{variant}.ll,m5=DIR/m5_{variant}.ll``.

The private ``ys`` mailbox and the phase stamps live in the ``ss`` region
(:mod:`.moe_fused`): a whole-scratch bitwise compare against the mono masks
``ss``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from types import SimpleNamespace

from rocke.core.ir import I32, I64, IRBuilder

from . import abi as A
from . import moe_fused as MF
from .device_function import (
    DeviceFunctionKnobs,
    lower_device_function,
    write_device_function,
)
from .stage_fn import ABI_PARAMS, parse_variant, variant_name

STAGES = ("m3", "m4", "m5")

# The tuned sets: hill climb with lookback on gfx950, in situ (the mono layer with these functions linked vs the
# plain mono layer, shuffled interleaved rounds with a mono-vs-mono control); ratios in
# hackathon-prep/reports/qwen38-rocke/ledger/fused-moe.md. Rows 1 (decode) and rows > 1 (MTP verify / batched).
# ``--knobs best``: the NUMA sets (knob numa=1: the layer runs ATOM's numa=1 schedule lever, ATOM_Q38_SCHED=numa=1,
# which ATOM keeps for these builds and requires); ``--knobs best_plain``: the sets for a server without the lever.
BEST_KNOBS = {
    1: {"dn_inflight": 0, "line_impl": "const", "guard": "loads", "slots": "arith", "xchg_pre": "none",
        "sh_ctas": 32, "sh_base": 160, "numa": 1, "lg_copy": "numa2", "xchg_home": "numa2", "ys_home": "numa2",
        "iod_split": 0, "line_order": "router_dn"},
    0: {"dn_inflight": 0, "line_impl": "const", "guard": "loads", "xchg_pre": "none", "router_knobs": "mono",
        "combine_base": -1, "numa": 1, "lg_copy": "numa2", "line_order": "router_dn"},
}  # fmt: skip
BEST_KNOBS_PLAIN = {
    1: {"dn_inflight": 0, "line_impl": "const", "guard": "loads", "map": "groups", "slots": "arith", "xchg_pre": "none",
        "sh_ctas": 32, "sh_base": 160},
    0: {"dn_inflight": 0, "line_impl": "const", "guard": "loads", "xchg_pre": "none", "router_knobs": "mono",
        "combine_base": -1},
}  # fmt: skip
BEST_SETS = {"best": BEST_KNOBS, "best_plain": BEST_KNOBS_PLAIN}


def best_knobs(rows: int, which: str = "best") -> MF.FusedKnobs:
    return knobs_from_json(BEST_SETS[which][1 if rows == 1 else 0])


def knobs_from_json(d: dict) -> MF.FusedKnobs:
    d = dict(d)
    if isinstance(d.get("mailbox"), dict):
        d["mailbox"] = MF.M.MailboxKnobs(**d["mailbox"])
    return MF.FusedKnobs(**d)


def build_function(stage: str, rows: int, seq: bool, knobs: MF.FusedKnobs, *, experts=A.EXPERTS,
                   arena_words="atom"):
    """``m3``: the fused body; ``m4`` / ``m5``: empty. ``arena_words`` as :func:`.stage_moe.build_function`."""
    if stage not in STAGES:
        raise ValueError(stage)
    if not (1 <= rows <= A.MAX_ROWS and (not seq or rows > 1)):
        raise ValueError(f"no fused function for rows {rows} seq {seq}")
    if not knobs.router:
        raise ValueError("stage functions run the router (knobs.router)")
    plan = MF.FusedPlan(rows, experts, knobs)
    plan.validate()
    if stage == "m3":
        if arena_words == "atom":
            from .stage_moe import mono_lds_words

            arena_words = mono_lds_words(rows, seq)
        words = MF.lds_layout(plan)[2]
        if arena_words is not None and words > arena_words:
            raise ValueError(f"fused r{rows}: LDS arena {words} words > the mono arena's {arena_words}")
    b = IRBuilder(f"rocke_q38_{stage}_{variant_name(rows, seq)}")
    b.kernel.attrs["max_workgroup_size"] = A.THREADS
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    if stage == "m3":
        for name in ("tag", "layer"):
            vals[name] = b.readfirstlane(vals[name])
        a = SimpleNamespace(**vals)
        addr = {n: getattr(a, n) for n in A.LAYER_ARGS}
        e = MF.FusedCtx(b, plan, tag=a.tag, layer=a.layer, addr=addr)
        MF.emit_fused(e)
    b.ret()
    return b.kernel, plan


def pair_id(variant: str, knobs: MF.FusedKnobs) -> str:
    key = ["fused", variant, repr(knobs)] + ([["m3_rows", knobs.m3_rows]] if knobs.m3_rows else [])  # not in repr
    key += [["route", knobs.route]] if knobs.route != "cta" else []  # P15, not in repr
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]


def sidecar_extra(stage: str, pair: str, knobs: MF.FusedKnobs | None = None) -> dict:
    """The generation's pairing, and (ATOM ``extern_stages``, "NUMA") the mailbox layout the body addresses:
    ``"mailboxes": "numa1"`` with knob ``numa`` (ATOM keeps its numa=1 lever for the build, and refuses the build
    without it), ``"numa_hdr"`` whenever the body uses the NUMA area (ATOM checks the area exists there)."""
    out = {"pair": pair, "requires": {s: pair for s in STAGES if s != stage}, "fused": True}
    if knobs is not None and knobs.uses_area:
        out["mailboxes"] = "numa1" if knobs.numa else "plain"
        out["numa_hdr"] = knobs.numa_hdr or MF.NUMA_HDR
    return out


def write_functions(out: str, variants, knobs: MF.FusedKnobs, *, experts=A.EXPERTS, arena_words="atom",
                    flavor=None, arch="gfx950", inline="always", bitcode=False) -> list:
    """Emit ``<stage>_<variant>.{ll,json}`` for every variant; returns the written entries."""
    fk = DeviceFunctionKnobs(lds="arg", inline=inline)
    written = []
    for v in variants:
        rows, seq, group = parse_variant(v)
        if group:
            raise ValueError(f"{v}: grouped builds are not covered")
        pair = pair_id(v, knobs)
        for stage in STAGES:
            have = arena_words
            if arena_words == "atom" and stage != "m3":
                have = None
            kern, _plan = build_function(stage, rows, seq, knobs, experts=experts, arena_words=have)
            fn = lower_device_function(kern, arch=arch, symbol=f"rocke_q38_{stage}_{v}", llvm_flavor=flavor, knobs=fk)
            paths = write_device_function(fn, out, stem=f"{stage}_{v}", bitcode=bitcode)
            with open(paths["json"]) as f:
                meta = json.load(f)
            meta.update(sidecar_extra(stage, pair, knobs), moe_knobs="fused:" + (knobs.suffix() or "default"))
            with open(paths["json"], "w", newline="\n") as f:
                json.dump(meta, f, indent=1)
                f.write("\n")
            written.append({"stage": stage, "variant": v, "lds_bytes": fn.lds_bytes, **paths})
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the fused rocKE MoE stage functions (m3 fused, m4 / m5 empty)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="r1,r2,r3,r4,r2s,r3s,r4s")
    ap.add_argument("--knobs", default="best",
                    help="'best' (BEST_KNOBS per variant, the default: numa=1 bodies, run with ATOM_Q38_SCHED=numa=1), "
                         "'best_plain' (BEST_KNOBS_PLAIN: servers without the lever), or FusedKnobs as JSON ('{}': "
                         "the design's first-light defaults, which spill at rows > 1)")
    ap.add_argument("--experts", type=int, default=A.EXPERTS)
    ap.add_argument("--mono-lds-words", type=int, default=None)
    ap.add_argument("--no-arena-check", action="store_true")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--inline", default="always", choices=("always", "default", "never"))
    a = ap.parse_args(argv)
    arena = None if a.no_arena_check else (a.mono_lds_words if a.mono_lds_words is not None else "atom")
    variants = [v for v in a.variants.split(",") if v]
    written = []
    if a.knobs in BEST_SETS:
        for v in variants:
            knobs = best_knobs(parse_variant(v)[0], a.knobs)
            written += write_functions(a.out, [v], knobs, experts=a.experts, arena_words=arena, flavor=a.flavor,
                                       arch=a.arch, inline=a.inline)
        print(json.dumps({"knobs": a.knobs, "written": written}, indent=1))
        return 0
    knobs = knobs_from_json(json.loads(a.knobs))
    written = write_functions(a.out, variants, knobs, experts=a.experts,
                              arena_words=arena, flavor=a.flavor, arch=a.arch, inline=a.inline)
    print(json.dumps({"knobs": knobs.suffix(), "written": written}, indent=1))
    return 0


__all__ = ["BEST_KNOBS", "BEST_KNOBS_PLAIN", "BEST_SETS", "STAGES", "best_knobs", "build_function", "knobs_from_json", "main", "pair_id", "sidecar_extra",
           "write_functions"]

if __name__ == "__main__":
    import importlib

    sys.exit(importlib.import_module(__spec__.name).main())
