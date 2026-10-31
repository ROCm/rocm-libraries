#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_dq_emit.py -- Python reference emitter for the
# whole split dQ kernel of the layout-general attention backward
# (kernels/common/attention_bwd_dq.py: build_attn_bwd_dq with the body of
# _attention_bwd_dqbody). The C twin attention_bwd_dq_emit.c builds the same
# kernel through rocke_build_attn_bwd_dq_new (rocke/instance_attention_bwd_dq.h)
# and the two .ll must match.
#
# The C builder does not consult the recorded resource-gate refusals
# (DQ_SPILLING_COMBOS, DQ_RESOURCE_REFUSED, DQ_SCHED_SPILLING stay Python), so
# no config here is one of them.
#
# The tail round of store_dq (a chunk guarded by chunk < total) cannot be
# reached: a valid spec has block_m = 16 * waves * k, head_size in 32/64/128 and
# wave64, so total = block_m * head_size / 8 is a multiple of 64 * waves threads.
#
# Configs:
#   * "@<case id>": the dQ kernel of a case of the phase-1 lowerer gate
#     (library/tests/test_attention_bwd_ir_cpp_parity.py, _cases / _plan):
#     every case whose plan launches a "dq" kernel, planned here exactly as
#     the gate plans it;
#   * "@<case id>;<field>:<value>;...": that spec with knob overrides;
#   * "<arch>|<field>:<value>;...": an AttnBwdDqSpec from scratch;
#   * "main=<arch>|<main fields>#<dq knobs>": AttnBwdSpec(<main fields>)
#     .dq_spec(<dq knobs>) (the main-spec spellings of
#     attention_bwd_spec_emit.py), the derivation of the plan.
# The C table holds each config resolved ("@" cases planned to
# "<arch>|<field>:<value>;..."); regenerate it with
# `attention_bwd_dq_emit.py --print-c-configs` whenever the shipped tables or
# _CONFIGS change.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
import dataclasses
import importlib.util
import sys
from functools import cache
from pathlib import Path

from _emit_common import run_emit
from attention_bwd_spec_emit import _parse as _parse_main
from kernels.common.attention_bwd import AttnBwdSpec
from kernels.common.attention_bwd_dq import AttnBwdDqSpec, build_attn_bwd_dq

_GATE = Path(__file__).resolve().parents[1] / "test_attention_bwd_ir_cpp_parity.py"
_INT = {
    "head_size",
    "stage_vec",
    "waves",
    "block_m",
    "block_n",
    "kv_pad",
    "waves_per_eu",
    "narrow_vec",
}
_BOOL = {"stats_prescale", "edge_tiles", "kv_prefetch", "xcd_remap"}


@cache
def _gate():
    """The phase-1 gate module (loaded by path: its directory would shadow the
    library's ``dispatch`` package on sys.path)."""
    spec = importlib.util.spec_from_file_location("_attn_bwd_gate", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@cache
def _planned() -> tuple:
    """``((case_id, arch, dq spec), ...)``: every gate case whose plan
    launches a "dq" kernel, in the gate's order."""
    gate = _gate()
    out = []
    for cid, family, arch, recipe in gate._cases():
        if family == "unified":
            continue
        plan = gate._plan(family, arch, recipe)
        if any(step.stage == "dq" for step in plan.launches):
            out.append((cid, arch, plan.specs["dq"]))
    return tuple(out)


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, tuple):
        return f"{v[0]}x{v[1]}"
    return str(v)


def _spec_text(arch: str, spec: AttnBwdDqSpec) -> str:
    """``spec`` as an "<arch>|<field>:<value>;..." config (non-default fields)."""
    items = []
    for f in dataclasses.fields(spec):
        v = getattr(spec, f.name)
        if f.default is dataclasses.MISSING or v != f.default:
            items.append(f"{f.name}:{_fmt(v)}")
    return f"{arch}|" + ";".join(items)


def _case_text(cid: str) -> str:
    for case_id, arch, spec in _planned():
        if case_id == cid:
            return _spec_text(arch, spec)
    raise SystemExit(f"gate case {cid!r} has no dq kernel")


def _resolve(config: str) -> str:
    """A config as "<arch>|<fields>" (gate cases planned, overrides last)."""
    if not config.startswith("@"):
        return config
    cid, _, over = config[1:].partition(";")
    text = _case_text(cid)
    return f"{text};{over}" if over else text


def _value(name: str, text: str):
    if text == "None":
        return None
    if name in _BOOL:
        if text not in ("True", "False"):
            raise SystemExit(f"bad bool {text!r} for {name}")
        return text == "True"
    if name in _INT:
        return int(text)
    if name == "agpr_alloc" and "x" in text:
        r, c = text.split("x")
        return (int(r), int(c))
    return text


def _parse(config: str):
    arch, _, rest = config.partition("|")
    fields = {}
    for item in rest.split(";"):
        if item:
            name, _, text = item.partition(":")
            fields[name] = _value(name, text)
    return arch, fields


def _case(cid: str, over: str = "") -> str:
    return f"@{cid};{over}" if over else f"@{cid}"


# The phase-1 dQ cases (gfx950 d64, 4 waves, 128 x 64) and two hand bases,
# one per arch, at the default tile.
_T11 = "general/gfx950/tuned11"
_T14 = "general/gfx950/tuned14"
_T17 = "general/gfx950/tuned17"
_B942 = "gfx942|head_size:64"
_B950 = "gfx950|head_size:128"
_SCHEDS = (
    "none",
    "iglp0",
    "iglp1",
    "fences_only",
    "stage_table",
    "stage_table_setprio",
)


def _knobs() -> tuple:
    out = []
    # every key-step schedule value, on both arches, plain and with the
    # edge / prefetch bodies the stage table counts
    for sc in _SCHEDS:
        out.append((f"sched {sc} gfx942", f"{_B942};sched:{sc}"))
        out.append((f"sched {sc} gfx950", f"{_B950};sched:{sc}"))
        out.append((f"sched {sc} tuned11", _case(_T11, f"sched:{sc}")))
        out.append(
            (
                f"sched {sc} prefetch edge gfx950 d64",
                f"gfx950|head_size:64;sched:{sc};kv_prefetch:True;edge_tiles:True",
            )
        )
    # keys per step (dq_block_n), both arches
    for kn in (16, 32, 64, 128, 256):
        out.append((f"block_n {kn} gfx942 d32", f"gfx942|head_size:32;block_n:{kn}"))
        out.append((f"block_n {kn} gfx950 d64", f"gfx950|head_size:64;block_n:{kn}"))
    out += [
        # query rows / waves
        ("block_m 16 waves 1 gfx942", f"{_B942};block_m:16;waves:1"),
        ("block_m 32 waves 2 gfx950", f"{_B950};block_m:32;waves:2"),
        ("block_m 128 waves 8 gfx942 d32", "gfx942|head_size:32;block_m:128;waves:8"),
        ("block_m 128 waves 4 gfx950 d64", "gfx950|head_size:64;block_m:128"),
        ("block_m 32 waves 1 gfx950 d32", "gfx950|head_size:32;block_m:32;waves:1"),
        # dtypes, head sizes, masks, sequence modes, access widths
        ("bf16 gfx942", f"{_B942};dtype:bf16"),
        ("bf16 gfx950", f"{_B950};dtype:bf16"),
        ("d32 gfx942", "gfx942|head_size:32"),
        ("d128 gfx942", "gfx942|head_size:128"),
        ("d32 bf16 gfx950", "gfx950|head_size:32;dtype:bf16"),
        ("d64 gfx950", "gfx950|head_size:64"),
        ("no mask gfx942", f"{_B942};mask_class:none"),
        ("no mask gfx950 edge", f"{_B950};mask_class:none;edge_tiles:True"),
        ("thd gfx942", f"{_B942};seq_mode:thd"),
        ("thd gfx950 band edge", f"{_B950};seq_mode:thd;edge_tiles:True"),
        (
            "thd no mask gfx950 group",
            f"{_B950};seq_mode:thd;mask_class:none;head_loop:group",
        ),
        ("stage_vec 1 gfx942", f"{_B942};stage_vec:1"),
        ("narrow 2 gfx942", f"{_B942};stage_vec:1;narrow_vec:2"),
        ("narrow 4 gfx950", f"{_B950};stage_vec:1;narrow_vec:4"),
        (
            "narrow 4 thd bf16 gfx950 d64",
            "gfx950|head_size:64;stage_vec:1;narrow_vec:4;seq_mode:thd;dtype:bf16",
        ),
        ("token-major gfx942", f"{_B942};ws_layout:token_major"),
        ("token-major group gfx950", f"{_B950};ws_layout:token_major;head_loop:group"),
        # stats / scale placement
        ("stats_prescale off gfx942", f"{_B942};stats_prescale:False"),
        (
            "stats_prescale off edge gfx950",
            f"{_B950};stats_prescale:False;edge_tiles:True",
        ),
        (
            "stats_prescale off runtime gfx942",
            f"{_B942};stats_prescale:False;static_scale:runtime",
        ),
        (
            "stats_prescale off convert gfx950",
            f"{_B950};stats_prescale:False;static_scale:convert",
        ),
        ("convert gfx942", f"{_B942};static_scale:convert"),
        ("runtime gfx950", f"{_B950};static_scale:runtime"),
        (
            "runtime edge group gfx942",
            f"{_B942};static_scale:runtime;edge_tiles:True;head_loop:group",
        ),
        # head loop, edge tiles, prefetch, padding, CTA remap
        ("group gfx942", f"{_B942};head_loop:group"),
        ("group gfx950", f"{_B950};head_loop:group"),
        ("group edge gfx942", f"{_B942};head_loop:group;edge_tiles:True"),
        (
            "group prefetch gfx950 d64",
            "gfx950|head_size:64;head_loop:group;kv_prefetch:True",
        ),
        ("edge gfx942", f"{_B942};edge_tiles:True"),
        ("edge gfx950", f"{_B950};edge_tiles:True"),
        ("prefetch gfx942", f"{_B942};kv_prefetch:True"),
        ("prefetch gfx950", f"{_B950};kv_prefetch:True"),
        ("prefetch edge gfx942", f"{_B942};kv_prefetch:True;edge_tiles:True"),
        (
            "prefetch tail block_n 16 waves 8 gfx942",
            "gfx942|head_size:64;block_n:16;block_m:128;waves:8;kv_prefetch:True",
        ),
        (
            "prefetch tail block_n 32 waves 8 gfx950 d32",
            "gfx950|head_size:32;block_n:32;block_m:128;waves:8;kv_prefetch:True",
        ),
        ("kv_pad 8 gfx942", f"{_B942};kv_pad:8"),
        ("kv_pad 16 gfx950", f"{_B950};kv_pad:16"),
        ("kv_pad 32 prefetch gfx942", f"{_B942};kv_pad:32;kv_prefetch:True"),
        ("xcd_remap gfx942", f"{_B942};xcd_remap:True"),
        ("xcd_remap group gfx950", f"{_B950};xcd_remap:True;head_loop:group"),
        # atoms and transpose sources
        ("lds_plain gfx950", f"{_B950};transpose_source:lds_plain"),
        (
            "lds_plain atom_g4 32 gfx950",
            f"{_B950};transpose_source:lds_plain;atom_g4:16x16x32",
        ),
        ("rejected: atom_g4 32 on gfx942", f"{_B942};atom_g4:16x16x32"),
        ("atom_g02 16 gfx950", f"{_B950};atom_g02:16x16x16"),
        ("rejected: atom_g02 32 on gfx942", "gfx942|head_size:32;atom_g02:16x16x32"),
        # kernel attributes
        ("waves_per_eu 1 gfx942", f"{_B942};waves_per_eu:1"),
        ("waves_per_eu 2 gfx950 d32", "gfx950|head_size:32;waves_per_eu:2"),
        ("agpr_alloc backend gfx942", f"{_B942};agpr_alloc:backend"),
        ("agpr_alloc 128x128 gfx950", f"{_B950};agpr_alloc:128x128"),
        ("agpr_alloc 0x0 gfx942 d128", "gfx942|head_size:128;agpr_alloc:0x0"),
        ("scheduler_strategy max-ilp gfx942", f"{_B942};scheduler_strategy:max-ilp"),
        (
            "scheduler_strategy iterative-minreg gfx950",
            f"{_B950};scheduler_strategy:iterative-minreg",
        ),
        ("name stem gfx942", f"{_B942};name:my_bwd"),
        # knobs on the phase-1 cases
        ("tuned11 edge", _case(_T11, "edge_tiles:True")),
        ("tuned11 group", _case(_T11, "head_loop:group")),
        ("tuned11 prefetch", _case(_T11, "kv_prefetch:True")),
        ("tuned14 xcd", _case(_T14, "xcd_remap:True")),
        ("tuned14 stats_prescale off", _case(_T14, "stats_prescale:False")),
        ("tuned17 band", _case(_T17, "mask_class:band")),
        ("tuned17 thd", _case(_T17, "seq_mode:thd")),
        ("tuned17 lds_plain", _case(_T17, "transpose_source:lds_plain")),
        # the main spec's derivation (AttnBwdSpec.dq_spec)
        ("dq_spec gfx942 d64", "main=gfx942|head_size:64;dq_mode:split#"),
        (
            "dq_spec gfx950 d128 bf16 thd dq_before_atomic",
            "main=gfx950|head_size:128;dtype:bf16;seq_mode:thd;dq_mode:split;static_scale:dq_before_atomic#edge_tiles:True",
        ),
        (
            "dq_spec gfx942 d32 runtime token-major",
            "main=gfx942|head_size:32;static_scale:runtime;ws_layout:token_major#head_loop:group;sched:iglp1",
        ),
        (
            "dq_spec gfx950 d64 narrow none convert",
            "main=gfx950|head_size:64;stage_vec:1;narrow_vec:2;mask_class:none;static_scale:convert#kv_prefetch:True",
        ),
        # rejections (both engines)
        ("rejected: rdna arch", "gfx1201|head_size:64"),
        ("rejected: unknown arch", "gfx1100|head_size:64"),
        ("rejected: block_m not a multiple of 16 waves", f"{_B942};block_m:32;waves:4"),
        ("rejected: tr_read on gfx942", f"{_B942};transpose_source:tr_read"),
        ("rejected: narrow_vec with stage_vec 8", f"{_B942};narrow_vec:2"),
        ("rejected: bad sched", f"{_B942};sched:stage_table_atomics"),
        ("rejected: tr_read K-permuted G4'", f"{_B950};atom_g4:16x16x32"),
        ("rejected: LDS plan", "gfx942|head_size:128;block_m:128;block_n:256;waves:2"),
        (
            "rejected: AGPR estimate",
            "gfx942|head_size:128;block_m:128;block_n:128;waves:2",
        ),
        (
            "rejected: agpr_alloc below the AGPR estimate",
            "gfx942|head_size:128;block_m:128;block_n:64;waves:2;agpr_alloc:128x128",
        ),
        (
            "rejected: dq_spec static_scale",
            "main=gfx942|head_size:64;static_scale:bogus#",
        ),
        (
            "rejected: block_n vs the G4' atom K",
            "gfx950|head_size:64;block_n:16;atom_g4:16x16x32;transpose_source:lds_plain",
        ),
        (
            "rejected: arch-VGPR estimate past the slack",
            "gfx942|head_size:128;block_m:128;block_n:64;waves:2;agpr_alloc:0x0",
        ),
        (
            "rejected: agpr_alloc 0x0 over the arch VGPRs",
            "gfx942|head_size:64;block_m:64;block_n:32;waves:1;agpr_alloc:0x0",
        ),
        (
            "rejected: waves_per_eu 2 register share",
            "gfx942|head_size:128;block_m:128;block_n:64;waves:2;waves_per_eu:2",
        ),
        (
            "rejected: workgroup share of eight waves",
            "gfx950|head_size:128;block_m:128;block_n:256;waves:8",
        ),
        ("rejected: bad name", f"{_B942};name:bad-name"),
    ]
    return tuple(out)


def _configs() -> tuple:
    out = [(f"case {cid}", _case(cid)) for cid, _a, _s in _planned()]
    out += list(_knobs())
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    config = _resolve(_CONFIGS[idx][1])
    if config.startswith("main="):
        main, _, knobs = config[len("main=") :].partition("#")
        arch, mfields = _parse_main(main)
        _a, kfields = _parse(f"{arch}|{knobs}")
        return ("main", tuple(mfields.items()), tuple(kfields.items())), arch
    arch, fields = _parse(config)
    return ("dq", tuple(fields.items()), ()), arch


def build_attention_bwd_dq_kernel(spec, *, arch: str):
    """The split dQ kernel of one config (a spec error is a build error)."""
    kind, fields, knobs = spec
    if kind == "main":
        dq = AttnBwdSpec(**dict(fields)).dq_spec(**dict(knobs))
    else:
        dq = AttnBwdDqSpec(**dict(fields))
    return build_attn_bwd_dq(dq, arch=arch)


def print_c_configs() -> None:
    for comment, config in _CONFIGS:
        c = comment.replace("*/", "* /")
        print(f'    /* {c} */\n    "{_resolve(config)}",')


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--print-c-configs":
        print_c_configs()
        return 0
    return run_emit(
        _spec,
        build_attention_bwd_dq_kernel,
        usage="usage: attention_bwd_dq_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
