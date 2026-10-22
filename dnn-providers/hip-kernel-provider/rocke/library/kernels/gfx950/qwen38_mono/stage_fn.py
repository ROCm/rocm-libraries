# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocKE stage functions for the FlyDSL mono layer kernel (hybrid path).

A stage function is one mono stage's body (e.g. Q2) authored in rocKE and
lowered as an LLVM device function (:mod:`.device_function`) that the mono
kernel calls in place of its own stage code (ATOM
``ATOM_Q38_EXTERN_STAGES``). Same inputs, outputs, mailboxes and task
assignment: the mono computes the launch tag and the CTA's task index exactly
as for its own stage and passes them in.

ABI (one for every stage; unused arguments are dead after inlining)::

    void @rocke_q38_<stage>_<variant>(
        i32 tag,            # the launch tag (mailbox.launch_tag)
        i32 task,           # this CTA's task of the stage (common.task_index;
                            # >= the stage's task count: no task)
        i64 step, i32 layer,          # the kernel's own first two arguments
        i64 <LAYER_ARGS...>,          # the 41 kernel arguments, in order
        i32 l0, i32 l1, i32 l2, i32 l3,  # QSA builds: each row's cache length
                                         # (the mono's p.Ls, 0 past the rows)
        i32 lds_base)       # byte address of the mono LDS arena (lds="arg")

``<variant>`` is ``r<rows>`` plus ``s`` for a seq (MTP verify) build and
``g<S>`` for a grouped one: ``r1``, ``r4s``, ... The ``.json`` sidecar holds the
ABI (``abi``), which the ATOM side checks against its own argument list.

Generator (needs the ATOM checkout for the layout, ``mono_layout``)::

    python -m kernels.gfx950.qwen38_mono.stage_fn --out DIR [--stages q2]
        [--variants r1,r2,r3,r4,r2s,r3s,r4s] [--atom-root PATH] [--flavor llvm23]

writes ``DIR/<stage>_<variant>.{ll,json}`` (and ``.bc`` with an ``llvm-as``)
per variant; ``ATOM_Q38_EXTERN_STAGES=q2=DIR/q2_{variant}.ll``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

from rocke.core.ir import I32, I64, IRBuilder

from .device_function import (
    DeviceFunctionKnobs,
    lower_device_function,
    write_device_function,
)
from .mono_layout import LAYER_ARGS, MonoLayout

__all__ = [
    "ABI_PARAMS",
    "STAGES",
    "StageKnobs",
    "StageSpec",
    "build_stage_kernel",
    "lower_stage",
    "register_stage",
    "variant_name",
]

ABI_PARAMS = (
    ("tag", "i32"),
    ("task", "i32"),
    ("step", "i64"),
    ("layer", "i32"),
    *((n, "i64") for n in LAYER_ARGS),
    ("l0", "i32"),
    ("l1", "i32"),
    ("l2", "i32"),
    ("l3", "i32"),
)
SYMBOL_PREFIX = "rocke_q38_"


def variant_name(rows: int, seq: bool = False, group: int = 0) -> str:
    return f"r{rows}" + ("s" if seq else "") + (f"g{group}" if group else "")


def parse_variant(v: str) -> tuple[int, bool, int]:
    import re

    m = re.fullmatch(r"r(\d+)(s?)(?:g(\d+))?", v)
    if not m:
        raise ValueError(f"bad variant {v!r} (r<rows>[s][g<S>])")
    return int(m.group(1)), bool(m.group(2)), int(m.group(3) or 0)


@dataclass(frozen=True)
class StageKnobs:
    """Build knobs of a stage function (defaults = the shipped behaviour).

    mailbox_impl: ``rocke.helpers.mailbox`` implementation of the polls and
      puts (``"asm"``; ``"ir"`` needs ``mailbox_ir`` imported). Shared with
      the b3 and moe stage functions, so its default stays ``"asm"``.
    xshfl: lane exchanges of the q2 norm / RoPE butterflies: ``"dpp2"``
      (default; VALU only, the mono's form: 1, 2, 8 by DPP, 4 from
      ``row_shl:4`` / ``row_shr:4`` and a select, 16 and 32 as the pair sums
      of ``permlane16_swap`` / ``permlane32_swap``) | ``"dpp"`` (the first
      port: ``ds_swizzle`` for 4 and 16) | ``"lds"`` (``warp_shuffle_xor``
      everywhere: ``ds_swizzle``, ``ds_bpermute`` for 32).
    tbl: loads of the per-step QSA address table and the page index:
      ``"smem"`` (default; scalar ``s_load`` through a constant-space pointer,
      as the mono's ``common.table_entry`` / ``qsa.page_block(scalar=True)``:
      no ``vmcnt`` wait, which on gfx9 also counts the previous stage's
      mailbox stores; only for memory no wave writes during the launch) |
      ``"vmem"`` (the first port: vector load + ``readfirstlane``, NOT the
      mono's form; its entry ``s_waitcnt vmcnt(0)`` is the mono's missing
      wait).
    fn: device-function emission knobs. Default ``lds="arg"``,
      ``inline="always"``: the mono must inline a stage (a real call costs
      9x the inlined sequence and moves the caller's registers); with
      ``"default"`` LLVM's inliner heuristics decide, and the r4s build of
      the (dpp2, smem, ir) set crossed its threshold and ran as a call.
    """

    mailbox_impl: str = "asm"
    xshfl: str = "dpp2"
    tbl: str = "smem"
    fn: DeviceFunctionKnobs = field(
        default_factory=lambda: DeviceFunctionKnobs(lds="arg", inline="always")
    )

    def __post_init__(self) -> None:
        if self.xshfl not in ("dpp", "lds", "dpp2"):
            raise ValueError(f"xshfl must be dpp, lds or dpp2, got {self.xshfl!r}")
        if self.tbl not in ("vmem", "smem"):
            raise ValueError(f"tbl must be vmem or smem, got {self.tbl!r}")

    def suffix(self) -> str:
        d = StageKnobs()
        parts = []
        if self.mailbox_impl != d.mailbox_impl:
            parts.append(f"mb{self.mailbox_impl}")
        if self.xshfl != d.xshfl:
            parts.append(f"x{self.xshfl}")
        if self.tbl != d.tbl:
            parts.append(f"t{self.tbl}")
        # fn knobs relative to this class's fn default (not DeviceFunctionKnobs')
        for k, short in (("lds", "lds"), ("inline", "inl")):
            if getattr(self.fn, k) != getattr(d.fn, k):
                parts.append(f"{short}{getattr(self.fn, k)}")
        if self.fn.flat_wg_attr != d.fn.flat_wg_attr:
            parts.append("nowg")
        if self.fn.convergent != d.fn.convergent:
            parts.append("nocv")
        return "_".join(parts)


@dataclass(frozen=True)
class StageSpec:
    stage: str
    rows: int = 1
    seq: bool = False
    group: int = 0
    knobs: StageKnobs = field(default_factory=StageKnobs)

    @property
    def variant(self) -> str:
        return variant_name(self.rows, self.seq, self.group)

    @property
    def symbol(self) -> str:
        return f"{SYMBOL_PREFIX}{self.stage}_{self.variant}"


# stage -> (body(b, a, layout, spec), supported(spec) -> bool)
STAGES: dict = {}


def register_stage(name: str, body, supported):
    STAGES[name] = (body, supported)
    return body


def build_stage_kernel(spec: StageSpec, layout: MonoLayout):
    """The stage function's ``KernelDef`` (the ABI's parameters, then the
    stage body)."""
    if spec.stage not in STAGES:
        _load_stages(spec.stage)
    if spec.stage not in STAGES:
        raise KeyError(f"no rocKE stage function {spec.stage!r}; have {sorted(STAGES)}")
    body, supported = STAGES[spec.stage]
    if not supported(spec, layout):
        raise ValueError(f"{spec.stage} has no rocKE function for variant {spec.variant}")
    b = IRBuilder(spec.symbol)
    b.kernel.attrs["max_workgroup_size"] = layout.threads
    b.fence_barriers = "compiler"
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    # Wave-uniform inputs: SGPRs once inlined into the mono (a no-op there);
    # a real call (``inline="never"``) passes arguments in VGPRs.
    for name in ("tag", "task", "layer", "l0", "l1", "l2", "l3"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(
        **vals,
        tid=tid,
        lane=b.land(tid, b.const_i32(63)),
        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))),
        ls=[vals["l0"], vals["l1"], vals["l2"], vals["l3"]],
    )
    body(b, a, layout, spec)
    b.ret()
    return b.kernel


def lower_stage(spec: StageSpec, layout: MonoLayout, *, arch="gfx950", flavor=None):
    return lower_device_function(
        build_stage_kernel(spec, layout),
        arch=arch,
        symbol=spec.symbol,
        llvm_flavor=flavor,
        knobs=spec.knobs.fn,
    )


# stage -> the module that registers it, imported on first use of that stage
# (a stage module mid-edit does not break the other stages' generation)
STAGE_MODULES = {"q2": "stage_q2", "b3": "stage_b3"}


def _load_stages(stage: str | None = None) -> None:
    import importlib

    mods = (
        [STAGE_MODULES[stage]]
        if stage in STAGE_MODULES
        else sorted(set(STAGE_MODULES.values()))
    )
    for m in mods:
        importlib.import_module(f"{__package__}.{m}")


DEFAULT_VARIANTS = ("r1", "r2", "r3", "r4", "r2s", "r3s", "r4s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit rocKE stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="q2")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--atom-root", default=None, help="ATOM checkout (default: import atom)")
    ap.add_argument("--flavor", default=None, help="LLVM flavor (default: autodetect)")
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--mailbox-impl", default="asm")
    d = StageKnobs()
    ap.add_argument("--xshfl", default=d.xshfl, choices=("dpp", "lds", "dpp2"))
    ap.add_argument("--tbl", default=d.tbl, choices=("vmem", "smem"))
    ap.add_argument("--lds", default=d.fn.lds)
    ap.add_argument("--inline", default=d.fn.inline)
    ap.add_argument(
        "--eps-bits-delta", type=int, default=0,
        help="test only: perturb the norm epsilon's bits (a mutant the A/B must catch)",
    )  # fmt: skip
    ap.add_argument(
        "--mutate", default=None,
        help="test only: a stage's named mutant (stage_q2.MUTANTS) the A/B must catch",
    )  # fmt: skip
    a = ap.parse_args(argv)
    from dataclasses import replace

    from .mono_layout import from_atom

    layout = from_atom(a.atom_root)
    if a.eps_bits_delta:
        layout = replace(layout, eps_bits=layout.eps_bits + a.eps_bits_delta)
    if a.mutate:
        for stage in a.stages.split(","):
            _load_stages(stage)
            mod = sys.modules[f"{__package__}.{STAGE_MODULES[stage]}"]
            if a.mutate not in getattr(mod, "MUTANTS", ()):
                raise SystemExit(f"{stage}: no mutant {a.mutate!r}")
            mod.MUTATE = a.mutate
    if a.mailbox_impl == "ir":
        from . import mailbox_ir  # noqa: F401  (registers "ir")
    knobs = StageKnobs(
        mailbox_impl=a.mailbox_impl,
        xshfl=a.xshfl,
        tbl=a.tbl,
        fn=DeviceFunctionKnobs(lds=a.lds, inline=a.inline),
    )
    bitcode = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    written = []
    for stage in a.stages.split(","):
        for v in a.variants.split(","):
            rows, seq, group = parse_variant(v)
            spec = StageSpec(stage, rows, seq, group, knobs)
            fn = lower_stage(spec, layout, arch=a.arch, flavor=a.flavor)
            paths = write_device_function(fn, a.out, stem=f"{stage}_{v}", bitcode=bitcode)
            written.append({"stage": stage, "variant": v, **paths, "lds_bytes": fn.lds_bytes})
    print(json.dumps(written, indent=1))
    return 0


if __name__ == "__main__":
    # ``python -m``: run the package's module instance, whose STAGES registry
    # the stage modules register into (this one is ``__main__``).
    import importlib

    sys.exit(importlib.import_module(__spec__.name).main())
