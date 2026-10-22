# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""M3 / M4 / M5 of the mono layer as rocKE stage functions (hybrid path).

The bodies are :mod:`.moe`'s emitters; the function ABI, symbols and files are
:mod:`.stage_fn`'s (``rocke_q38_<stage>_<variant>``, ``lds="arg"``: the
emitters' LDS arena is the start of the mono kernel's arena). Registered with
:data:`.stage_fn.STAGES` at the default :class:`.moe.MoeKnobs`; the generator
here takes the MoE knobs too::

    python -m kernels.gfx950.qwen38_mono.stage_moe --out DIR [--stages m3,m4,m5]
        [--variants r1,r4s] [--knobs '{"dedupe": true}'] [--flavor llvm23]

then ``ATOM_Q38_EXTERN_STAGES=m3=DIR/m3_{variant}.ll,m4=DIR/m4_{variant}.ll,m5=DIR/m5_{variant}.ll``.

``--knobs best`` (the serving set) writes, per layer kind and variant, only the
stages that beat the mono there in situ (:data:`BEST`; a ``None`` entry, or a
stage left out, keeps the FlyDSL stage), as ``<stage>_<kind>_<variant>``::

    python -m kernels.gfx950.qwen38_mono.stage_moe --out DIR --knobs best --flavor llvm23

then ``ATOM_Q38_EXTERN_MISSING=mono`` and
``ATOM_Q38_EXTERN_STAGES=m3=DIR/m3_{kind}_{variant}.ll,m4=DIR/m4_{kind}_{variant}.ll,m5=DIR/m5_{kind}_{variant}.ll``
(ATOM ``extern_stages``: ``{kind}`` is the build's layer kind). Grouped and wide
builds have no function either.

Pairing rule: at rows > 1 the mono M5 reads routing that the mono M4's CTAs
leave in LDS, and this M4 leaves its own tables instead, so ``m4`` must be
swapped together with ``m5`` (the generator refuses ``m4`` alone). ``m5``
reuses this M4's slot tables only when both are swapped (``--stages`` holds
both); alone it routes for itself. The sidecars say so and ATOM refuses a
build that breaks it: ``m4``'s ``"requires": {"m5": null}``, a reusing
``m5``'s ``"requires": {"m4": <pair>}``, ``"pair"`` the generation's id.

The mono arena must hold the emitters' (:func:`.moe.lds_layout`): checked
against ATOM's ``layout.for_rows`` (ATOM importable, or ``Q38_ATOM_DIR``); the
generator refuses to run without it unless given ``--mono-lds-words`` or
``--no-arena-check``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from rocke.core.ir import I32, I64, IRBuilder

from . import abi as A
from . import moe as MO
from .device_function import DeviceFunctionKnobs, lower_device_function, write_device_function
from .stage_fn import ABI_PARAMS, StageSpec, parse_variant, register_stage, variant_name

MOE_STAGES = ("m3", "m4", "m5")

# The tuned knob sets per row class (used by :data:`BEST`): hill climb with
# lookback on gfx950, in situ (the mono layer with these functions linked),
# confirmed by an exhaustive sweep over the retained knobs (rows > 1 at delta
# and qsa r4s, rows 1 at delta r1); ratios in
# hackathon-prep/reports/qwen38-rocke/ledger/moe.md. Rows 1 (decode) and rows > 1
# (MTP verify) differ: the expert-union dedupe needs several rows.
BEST_KNOBS = {
    1: {
        "m5_ranks": "pre", "a_pre": True, "slots": "arith", "m5_extras": "first",
        "shared_early": True, "mailbox": {"impl": "ir"},
    },
    0: {
        "dedupe": True, "m4_waves": 4, "m4_map": "balanced", "m5_ranks": "pre", "a_pre": True,
        "m4_guard": "none", "m4_interleave": True, "m4_apipe": 1, "m5_extras": "m4",
        "m4_weights": "late", "mailbox": {"impl": "ir"}, "w_cpol": 18,
    },
}  # fmt: skip


_ROWS1 = BEST_KNOBS[1]
_ROWSN = BEST_KNOBS[0]
_ALL = ("m3", "m4", "m5")
_M3 = ("m3",)

# The serving set (``--knobs best``): per (layer kind, variant), the stages to
# swap and their knobs, or ``None`` (every FlyDSL stage stays). A cell ships
# only what beat the mono layer in EVERY repeat (ATOM_Q38_STAMPS=0, 3 cells on
# each of the two gfx950 nodes, max ratio < 0.9995; jobs moefix-dec1 / dec2,
# ledger hackathon-prep/reports/qwen38-rocke/ledger/moe.md): the whole set,
# M3 alone (rows 2: the whole set loses, M3 alone wins), or nothing (non-seq
# rows 3 / 4 on delta and qsa, qsa r4s: node-dependent or above 1).
BEST = {
    "delta": {
        "r1": (_ALL, _ROWS1), "r2": (_M3, _ROWSN), "r2s": (_M3, _ROWSN), "r3": None, "r3s": (_ALL, _ROWSN),
        "r4": None, "r4s": (_ALL, _ROWSN),
    },
    "delta_ple": {
        "r1": (_ALL, _ROWS1), "r2": None, "r2s": None, "r3": None, "r3s": (_ALL, _ROWSN),
        "r4": (_ALL, _ROWSN), "r4s": (_ALL, _ROWSN),
    },
    "qsa": {
        "r1": (_ALL, _ROWS1), "r2": None, "r2s": (_M3, _ROWSN), "r3": None, "r3s": (_ALL, _ROWSN),
        "r4": None, "r4s": None,
    },
}  # fmt: skip
KINDS = tuple(BEST)


def knobs_from_json(d: dict) -> MO.MoeKnobs:
    """``MoeKnobs`` from a JSON dict (``mailbox`` as a dict of ``MailboxKnobs`` fields)."""
    d = dict(d)
    if isinstance(d.get("mailbox"), dict):
        d["mailbox"] = MO.M.MailboxKnobs(**d["mailbox"])
    return MO.MoeKnobs(**d)


def best_knobs(rows: int) -> MO.MoeKnobs:
    """:data:`BEST_KNOBS` for a ``rows``-row variant."""
    return knobs_from_json(BEST_KNOBS[1 if rows == 1 else 0])


def _supported(spec, layout) -> bool:
    return spec.group == 0 and 1 <= spec.rows <= A.MAX_ROWS and (not spec.seq or spec.rows > 1)


def _ctx(b, a, plan):
    addr = {n: getattr(a, n) for n in A.LAYER_ARGS}
    return MO.MoeCtx(b, plan, tag=a.tag, layer=a.layer, addr=addr)


def emit_stage(b, a, stage: str, plan: MO.MoePlan, m5_reuse: bool) -> None:
    e = _ctx(b, a, plan)
    if stage == "m3":
        MO.emit_m3(e)
    elif stage == "m4":
        MO.emit_m4(e)
    elif stage == "m5":
        routed = MO.m5_routed_by_m4(e) if m5_reuse else None
        MO.emit_m5(e, routed)
    else:
        raise ValueError(stage)


def _body(stage):
    def body(b, a, layout, spec):
        plan = MO.MoePlan(spec.rows, A.EXPERTS, MO.MoeKnobs(), MO.STAGES)
        emit_stage(b, a, stage, plan, m5_reuse=False)

    return body


for _s in MOE_STAGES:
    register_stage(_s, _body(_s), _supported)


def build_function(stage: str, rows: int, seq: bool, knobs: MO.MoeKnobs, *, experts=A.EXPERTS,
                   m5_reuse=False, arena_words="atom"):
    """The stage function's ``KernelDef`` (:data:`.stage_fn.ABI_PARAMS`, then the body).

    ``arena_words``: the mono LDS arena's int32 words the emitters must fit in:
    ``"atom"`` (ATOM's ``layout.for_rows``; raises when ATOM is not importable),
    an int, or ``None`` (no check: the caller takes the risk)."""
    spec = StageSpec(stage, rows, seq)
    if not _supported(spec, None):
        raise ValueError(f"{stage}: no rocKE function for variant {spec.variant}")
    plan = MO.MoePlan(rows, experts, knobs, MO.STAGES)
    plan.validate()
    have = mono_lds_words(rows, seq) if arena_words == "atom" else arena_words
    words = MO.lds_layout(plan)[2]
    if have is not None and words > have:
        # the function's arena is the mono kernel's (lds="arg"): past its end the
        # writes are dropped and the reads return zeros (seen: NaN in h_out)
        raise ValueError(f"{stage} {spec.variant}: LDS arena {words} words > the mono arena's {have}")
    b = IRBuilder(spec.symbol)
    b.kernel.attrs["max_workgroup_size"] = A.THREADS
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    for name in ("tag", "task", "layer"):
        vals[name] = b.readfirstlane(vals[name])
    from types import SimpleNamespace

    emit_stage(b, SimpleNamespace(**vals), stage, plan, m5_reuse)
    b.ret()
    return b.kernel, plan


def mono_lds_words(rows: int, seq: bool) -> int:
    """The mono arena's int32 words for the variant (ATOM ``layout.for_rows``:
    the ``atom`` package, else the checkout at ``Q38_ATOM_DIR``). Raises when
    neither is importable: without the check a knob set past the arena runs
    with NaN in ``h_out``."""
    try:
        from atom.models.qwen4_exp_mono.kernels import layout as LY
    except Exception as e:  # noqa: BLE001
        root = os.environ.get("Q38_ATOM_DIR")
        if not root:
            raise RuntimeError(
                "the mono LDS arena check needs ATOM (importable, or Q38_ATOM_DIR); "
                "give the generator --mono-lds-words N or --no-arena-check"
            ) from e
        from .mono_layout import _import_atom

        LY = _import_atom("atom.models.qwen4_exp_mono.kernels.layout", root)
    return LY.for_rows(rows, seq).lds_words


def pair_id(kind, variant: str, stages, knobs: MO.MoeKnobs) -> str:
    """One generation's id (sidecar ``pair``): its kind, variant, stages and knobs."""
    text = json.dumps([kind, variant, list(stages), repr(knobs)])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sidecar_extra(stage: str, stages, pair: str) -> dict:
    """The pairing fields of ``stage``'s sidecar (module docstring)."""
    out = {"pair": pair}
    if stage == "m4":
        out["requires"] = {"m5": None}
    elif stage == "m5" and "m4" in stages:
        out["requires"] = {"m4": pair}
    return out


def serving_plan(kinds=KINDS, variants=None) -> list:
    """``[(kind, variant, stages, MoeKnobs)]``: the :data:`BEST` cells that have a function."""
    out = []
    for kind in kinds:
        for v, entry in BEST[kind].items():
            if entry is None or (variants is not None and v not in variants):
                continue
            stages, kn = entry
            out.append((kind, v, tuple(stages), knobs_from_json(kn)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE MoE stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="m3,m4,m5")
    ap.add_argument("--variants", default="r1,r2,r3,r4,r2s,r3s,r4s")
    ap.add_argument("--knobs", default="{}",
                    help="MoeKnobs as JSON (files <stage>_<variant>), or 'best' (the serving set BEST: "
                         "files <stage>_<kind>_<variant>, only where they beat the mono)")
    ap.add_argument("--kinds", default=",".join(KINDS), help="--knobs best: the layer kinds")
    ap.add_argument("--mono-lds-words", type=int, default=None,
                    help="the mono arena's int32 words (when ATOM is not importable)")
    ap.add_argument("--no-arena-check", action="store_true",
                    help="skip the mono LDS arena check (a set past the arena runs with NaN in h_out)")
    ap.add_argument("--experts", type=int, default=A.EXPERTS)
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    # always: LLVM's inliner left M5 a real call in the QSA r4s layer (callee saves to scratch, 1.08 of the
    # mono layer); "default" (the inliner decides) is the retained-off value
    ap.add_argument("--inline", default="always", choices=("always", "default", "never"))
    a = ap.parse_args(argv)
    stages = tuple(s for s in a.stages.split(",") if s)
    if "m4" in stages and "m5" not in stages:
        raise SystemExit("m4 must be swapped together with m5 (module docstring)")
    best = a.knobs == "best"
    variants = [v for v in a.variants.split(",") if v]
    for v in variants:
        if parse_variant(v)[2]:
            raise SystemExit(f"{v}: grouped builds are not covered")
    if best:
        kinds = [k for k in a.kinds.split(",") if k]
        bad = set(kinds) - set(KINDS)
        if bad:
            raise SystemExit(f"--kinds: unknown {sorted(bad)} (known {list(KINDS)})")
        cells = [
            (kind, v, tuple(s for s in st if s in stages), kn)
            for kind, v, st, kn in serving_plan(kinds, set(variants))
        ]
        cells = [c for c in cells if c[2]]
        if any("m4" in st and "m5" not in st for _k, _v, st, _kn in cells):
            raise SystemExit("m4 must be swapped together with m5 (module docstring)")
    else:
        knobs = knobs_from_json(json.loads(a.knobs))
        cells = [(None, v, stages, knobs) for v in variants]
    if any(c[3].stamps for c in cells):
        raise SystemExit("stage functions take no stamps knob (the mono kernel stamps them)")
    fk = DeviceFunctionKnobs(lds="arg", inline=a.inline)
    bitcode = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    written = []
    for kind, v, st, knobs in cells:
        rows, seq, _group = parse_variant(v)
        if a.no_arena_check:
            have = None
        elif a.mono_lds_words is not None:
            have = a.mono_lds_words
        else:
            try:
                have = mono_lds_words(rows, seq)
            except RuntimeError as e:
                raise SystemExit(str(e)) from e
        pair = pair_id(kind, v, st, knobs)
        for stage in st:
            kern, plan = build_function(
                stage, rows, seq, knobs, experts=a.experts, m5_reuse=("m4" in st), arena_words=have
            )
            fn = lower_device_function(
                kern, arch=a.arch, symbol=f"rocke_q38_{stage}_{variant_name(rows, seq)}",
                llvm_flavor=a.flavor, knobs=fk,
            )  # fmt: skip
            if have is not None and fn.lds_bytes > 4 * have:
                raise SystemExit(f"{stage} {v}: needs {fn.lds_bytes} LDS bytes, mono arena {4 * have}")
            stem = f"{stage}_{v}" if kind is None else f"{stage}_{kind}_{v}"
            paths = write_device_function(fn, a.out, stem=stem, bitcode=bitcode)
            with open(paths["json"]) as f:
                meta = json.load(f)
            meta.update(sidecar_extra(stage, st, pair), moe_knobs=knobs.suffix() or "default")
            if kind is not None:
                meta["kind"] = kind
            with open(paths["json"], "w", newline="\n") as f:
                json.dump(meta, f, indent=1)
                f.write("\n")
            written.append({"stage": stage, "kind": kind, "variant": v, "lds_bytes": fn.lds_bytes, **paths})
    print(json.dumps({"knobs": "best" if best else knobs.suffix(), "written": written}, indent=1))
    return 0


__all__ = [
    "BEST", "BEST_KNOBS", "KINDS", "MOE_STAGES", "best_knobs", "build_function", "emit_stage", "knobs_from_json",
    "main", "mono_lds_words", "pair_id", "serving_plan", "sidecar_extra",
]  # fmt: skip

if __name__ == "__main__":
    import importlib

    sys.exit(importlib.import_module(__spec__.name).main())
