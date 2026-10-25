#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_dense_bwd_main_emit.py -- Python reference emitter for
# the dense attention backward main kernel (kernels/<arch>/attention_dense_bwd:
# build_attention_dense_bwd, the shell of _attention_dense_bwd_shell driving
# the general family's _kv_tile) and for the shared aux kernels (prep,
# convert) the dense plans launch. The C twin attention_dense_bwd_main_emit.c
# builds the same kernels through rocke_build_attn_dense_bwd_main_new
# (rocke/instance_attention_dense_bwd_spec.h) and rocke_build_attn_bwd_aux_new
# and the two .ll must match.
#
# Configs:
#   * "@<case id>": the dense main kernel of a dense case of the phase-1
#     lowerer gate (library/tests/test_attention_bwd_ir_cpp_parity.py, _cases
#     / _plan), planned here exactly as the gate plans it;
#   * "@<case id>;<dense field>:<value>;...#<body knob>:<value>;...": that
#     spec with dense-field and body-knob overrides;
#   * "<arch>|<dense field>:<value>;...#<body knob>:<value>;...": a dense
#     spec from scratch. The spec class is the arch's per-arch class
#     (Gfx942AttentionDenseBwdSpec / Gfx950AttentionDenseBwdSpec), or the one
#     the pseudo-field cls names ("base", "gfx942", "gfx950"). Body knobs use
#     the AttnBwdSpec spellings of attention_bwd_spec_emit.py;
#   * "aux:<stage>|<arch>|<field>:<value>;...": a prep / convert kernel of a
#     dense plan (AttnBwdAuxSpec fields), every one the dense cases launch.
# The C table holds each config resolved (cases planned); regenerate it with
# `attention_dense_bwd_main_emit.py --print-c-configs` whenever the shipped
# tables or _CONFIGS change.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
import dataclasses
import importlib.util
import sys
from functools import cache
from pathlib import Path

from _emit_common import run_emit
from attention_bwd_spec_emit import _parse as _parse_knobs
from kernels.common.attention_bwd_aux import AttnBwdAuxSpec
from kernels.common.attention_bwd_run import _BUILDERS
from kernels.common.attention_dense_bwd_spec import AttentionDenseBwdSpec
from kernels.gfx942.attention_dense_bwd import Gfx942AttentionDenseBwdSpec
from kernels.gfx950.attention_dense_bwd import Gfx950AttentionDenseBwdSpec

_GATE = Path(__file__).resolve().parents[1] / "test_attention_bwd_ir_cpp_parity.py"
_CLASSES = {
    "base": AttentionDenseBwdSpec,
    "gfx942": Gfx942AttentionDenseBwdSpec,
    "gfx950": Gfx950AttentionDenseBwdSpec,
}
_INT = {
    "seqlen_q",
    "seqlen_kv",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
    "batch",
    "sliding_window",
    "window_left",
    "window_right",
    # AttnBwdAuxSpec
    "stage_vec",
    "rows_per_cta",
    "convert_vec",
    "narrow_vec",
}
_BOOL = {
    "causal",
    "causal_bottom_right",
    "ragged",
    "edge_tiles",
    "persistent",
    "xcd_remap",
    "worklist",
}


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
    """``((case_id, arch, plan), ...)``: every dense case of the gate."""
    gate = _gate()
    out = []
    for cid, family, arch, recipe in gate._cases():
        if family == "dense":
            out.append((cid, arch, gate._plan(family, arch, recipe)))
    return tuple(out)


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, tuple):
        return f"{v[0]}x{v[1]}"
    return str(v)


def _fields_text(spec, skip=()) -> str:
    items = []
    for f in dataclasses.fields(spec):
        if f.name in skip:
            continue
        v = getattr(spec, f.name)
        if f.default is dataclasses.MISSING or v != f.default:
            items.append(f"{f.name}:{_fmt(v)}")
    return ";".join(items)


def _dense_text(arch: str, spec) -> str:
    """A dense spec as "<arch>|<fields>#<knobs>"."""
    knobs = ";".join(f"{k}:{_fmt(v)}" for k, v in spec.body_knobs)
    return f"{arch}|{_fields_text(spec, skip=('body_knobs',))}#{knobs}"


@cache
def _case_text(cid: str) -> str:
    for case_id, arch, plan in _planned():
        if case_id == cid:
            return _dense_text(arch, plan.specs["dense"])
    raise SystemExit(f"unknown dense gate case {cid!r}")


def _aux_configs() -> list:
    """Every prep / convert kernel the dense cases launch (once each)."""
    out, seen = [], set()
    for cid, arch, plan in _planned():
        for step in plan.launches:
            if step.stage not in ("prep", "convert"):
                continue
            text = f"aux:{step.stage}|{arch}|{_fields_text(plan.specs[step.stage])}"
            if text not in seen:
                seen.add(text)
                out.append((f"{step.stage} of {cid}", text))
    return out


def _resolve(config: str) -> str:
    """A config with its gate case planned (overrides appended)."""
    if not config.startswith("@"):
        return config
    cid, _, over = config[1:].partition(";")
    text = _case_text(cid)
    if not over:
        return text
    dense_over, _, knob_over = over.partition("#")
    dense, _, knobs = text.partition("#")
    if dense_over:
        dense = f"{dense};{dense_over}"
    if knob_over:
        knobs = f"{knobs};{knob_over}" if knobs else knob_over
    return f"{dense}#{knobs}"


def _value(name: str, text: str):
    if text == "None":
        return None
    if name in _BOOL:
        if text not in ("True", "False"):
            raise SystemExit(f"bad bool {text!r} for {name}")
        return text == "True"
    if name in _INT:
        return int(text)
    return text


def _parse_fields(rest: str) -> dict:
    fields = {}
    for item in rest.split(";"):
        if item:
            name, _, text = item.partition(":")
            fields[name] = _value(name, text)
    return fields


def _case(cid: str, over: str = "") -> str:
    return f"@{cid};{over}" if over else f"@{cid}"


_T0 = "dense/gfx950/tuned0"  # causal BR, d128, split dQ, DMA ring 2, asm_g13
_T1 = "dense/gfx950/tuned1"  # no mask, d64, split dQ, kt from LDS
_T2 = "dense/gfx950/tuned2"  # causal TL, ragged, atomic dK / dV, kN0 = 192
_D942 = "dense/gfx942/default0"  # causal BR, d128, the shipped body
_D942N = "dense/gfx942/default1"  # no mask, d64
_D950 = "dense/gfx950/default0"
_D950N = "dense/gfx950/default1"
_D942C = "dense/gfx942/default2"  # causal TL, 64 / 8 heads
# small problems from scratch (one per arch), aligned to the default tiles
_S942 = (
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64"
)
_S950 = (
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128"
)


def _knobs() -> tuple:
    out = []
    # the masks: causal TL / BR, sliding windows, explicit two-sided bands,
    # no mask, on both arches (compile-time classification of every kv tile)
    for base, tag in ((_S942, "gfx942"), (_S950, "gfx950")):
        out += [
            (f"causal tl {tag}", f"{base};causal:True#"),
            (f"causal br {tag}", f"{base};causal_bottom_right:True#"),
            (f"no mask {tag}", f"{base};causal:False#"),
            (f"window 64 {tag}", f"{base};sliding_window:64#"),
            (
                f"window 64 br {tag}",
                f"{base};sliding_window:64;causal_bottom_right:True#",
            ),
            (
                f"band 32x16 {tag}",
                f"{base};causal:False;window_left:32;window_right:16#",
            ),
            (f"band left only {tag}", f"{base};causal:False;window_left:100#"),
            (
                f"band right only br {tag}",
                f"{base};causal:False;window_right:0;causal_bottom_right:True#",
            ),
            (f"ragged causal {tag}", f"{base};seqlen_q:200;seqlen_kv:300;ragged:True#"),
            (
                f"ragged no mask {tag}",
                f"{base};seqlen_q:200;seqlen_kv:300;ragged:True;causal:False#",
            ),
            (
                f"ragged clamp {tag}",
                f"{base};seqlen_q:200;seqlen_kv:300;ragged:True;oob:clamp#",
            ),
            (f"edge tiles off {tag}", f"{base};edge_tiles:False#"),
            (f"edge tiles off no mask {tag}", f"{base};edge_tiles:False;causal:False#"),
            (f"bhsd {tag}", f"{base};layout:bhsd#"),
            (f"fp16 {tag}", f"{base};dtype:fp16#"),
            (f"pinned batch {tag}", f"{base};batch:3#"),
            (f"oob clamp {tag}", f"{base};oob:clamp#"),
            (f"atomic dkv {tag}", f"{base};dkv_mode:atomic#"),
            (
                f"atomic dkv token-major {tag}",
                f"{base};dkv_mode:atomic;ws_layout:token_major#",
            ),
            (f"split dq {tag}", f"{base};dq_mode:split#"),
            (f"epilogue direct {tag}", f"{base};epilogue:direct#"),
            (f"epilogue pair {tag}", f"{base};epilogue:lds_staged_pair#"),
            (f"token-major {tag}", f"{base};ws_layout:token_major#"),
            (f"persistent {tag}", f"{base};persistent:True#"),
            (
                f"persistent hkv {tag}",
                f"{base};persistent:True;persist_decode:hkv_major#",
            ),
            (f"persistent atomic {tag}", f"{base};persistent:True;dkv_mode:atomic#"),
            (f"xcd remap {tag}", f"{base};xcd_remap:True#"),
            (f"xcd remap atomic {tag}", f"{base};xcd_remap:True;dkv_mode:atomic#"),
            (f"name stem {tag}", f"{base};name:my_dense#"),
            # body knobs through the shared body
            (f"ring vgpr 2 {tag}", f"{base}#ring_depth:2"),
            (f"ring vgpr 3 {tag} (gfx942 refuses)", f"{base}#ring_depth:3"),
            (f"ring vgpr 2 clamp {tag}", f"{base};oob:clamp#ring_depth:2"),
            (
                f"ring vgpr 2 ragged {tag}",
                f"{base};seqlen_q:200;seqlen_kv:300;ragged:True#ring_depth:2",
            ),
            (f"static_scale runtime {tag}", f"{base}#static_scale:runtime"),
            (
                f"static_scale convert atomic {tag}",
                f"{base};dkv_mode:atomic#static_scale:convert",
            ),
            (f"lb_interleave {tag}", f"{base}#lb_interleave:True"),
            (
                f"lb_interleave persistent {tag}",
                f"{base};persistent:True#lb_interleave:True",
            ),
            (f"ptr_noalias workspace {tag}", f"{base}#ptr_noalias:workspace"),
            (
                f"ptr_noalias all atomic {tag}",
                f"{base};dkv_mode:atomic#ptr_noalias:all",
            ),
            (f"head_outer {tag}", f"{base}#step_order:head_outer"),
            (f"head_outer no mask {tag}", f"{base};causal:False#step_order:head_outer"),
            (f"q_rotate {tag}", f"{base}#q_rotate:True"),
            (f"counted {tag}", f"{base}#loop_sync:counted"),
            (f"stage_table_atomics {tag}", f"{base}#sched:stage_table_atomics"),
            (f"stats_vec 4 {tag}", f"{base}#stats_vec:4;stats_fma:True"),
            (f"waves_per_eu 1 {tag}", f"{base}#waves_per_eu:1"),
            (f"scheduler_strategy {tag}", f"{base}#scheduler_strategy:max-ilp"),
            (f"interior_clamp {tag}", f"{base}#interior_clamp:True"),
            # the counted / deferred-dQ / rotated q-step forms of _kv_tile
            (
                f"counted stage_table_atomics {tag}",
                f"{base}#loop_sync:counted;sched:stage_table_atomics",
            ),
            (f"counted no mask {tag}", f"{base};causal:False#loop_sync:counted"),
            (
                f"stage_table_atomics no mask {tag}",
                f"{base};causal:False#sched:stage_table_atomics",
            ),
            (
                f"head_outer q_rotate {tag}",
                f"{base}#step_order:head_outer;q_rotate:True",
            ),
            (
                f"head_outer q_rotate no mask {tag}",
                f"{base};causal:False#step_order:head_outer;q_rotate:True",
            ),
            (
                f"head_outer stage_table_atomics {tag}",
                f"{base}#step_order:head_outer;sched:stage_table_atomics",
            ),
            # not built / rejected (both engines refuse)
            (f"rejected: lb_mirror {tag}", f"{base}#lb_mirror:True"),
            (f"rejected: slab_carry {tag}", f"{base}#slab_carry:True"),
            (f"rejected: ws_bounds buffer {tag}", f"{base}#ws_bounds:buffer"),
            (f"rejected: owned knob {tag}", f"{base}#dtype:bf16"),
            (f"rejected: owned knob name {tag}", f"{base}#name:other"),
            (f"rejected: unaligned not ragged {tag}", f"{base};seqlen_q:200#"),
        ]
    out += [
        ("global_path dma ring 2 gfx950", f"{_S950}#global_path:dma;ring_depth:2"),
        ("global_path dma ring 3 gfx950", f"{_S950}#global_path:dma;ring_depth:3"),
        ("global_path dma ring 1 gfx950", f"{_S950}#global_path:dma;ring_depth:1"),
        (
            "global_path dma ring 2 split gfx950",
            f"{_S950};dq_mode:split#global_path:dma;ring_depth:2",
        ),
        ("barrier merge gfx950", f"{_S950}#barrier_merge:on"),
        ("asm_g13 gfx950", f"{_S950}#agpr_alloc:128x128;agpr_route:asm_g13"),
        (
            "vgpr_form gfx942",
            f"{_S942};head_size:128#agpr_alloc:192x192;agpr_route:vgpr_form",
        ),
        ("agpr_alloc 0x0 gfx942", f"{_S942}#agpr_alloc:0x0"),
        ("body_variant xt not built gfx942", f"{_S942};body_variant:xt#"),
        ("body_variant trload not built gfx950", f"{_S950};body_variant:trload#"),
        # body_knobs names as passed: Python refuses a dense-owned name at its
        # default and a repeated name; a tuning knob at its default is dropped
        ("rejected: owned knob at default", f"{_S942}#dtype:fp16"),
        (
            "rejected: owned opt knob at default",
            f"{_S950}#q_rotate:True;edge_tiles:None",
        ),
        ("rejected: repeated knob", f"{_S942}#q_rotate:True;q_rotate:True"),
        ("rejected: repeated owned knob", f"{_S950}#dtype:fp16;dtype:fp16"),
        ("knob at default dropped", f"{_S942}#q_rotate:False"),
        ("rejected: base class", f"{_S942};cls:base#"),
        ("rejected: gfx950 class on gfx942", f"{_S942};cls:gfx950#"),
        (
            "rejected: unknown arch",
            "gfx1201|seqlen_q:256;seqlen_kv:256;num_query_heads:8;num_kv_heads:8;head_size:64#",
        ),
        ("rejected: head size 32", f"{_S942};head_size:32#"),
        ("rejected: window without causal", f"{_S942};causal:False;sliding_window:8#"),
        ("rejected: xcd and persistent", f"{_S942};persistent:True;xcd_remap:True#"),
        (
            "rejected: heads not a multiple",
            f"{_S942};num_query_heads:6;num_kv_heads:4#",
        ),
        ("rejected: seqlen_q 0", f"{_S942};seqlen_q:0#"),
        ("rejected: batch 0", f"{_S942};batch:0#"),
        ("rejected: negative window", f"{_S942};sliding_window:-1#"),
        ("rejected: causal and explicit band", f"{_S942};window_left:4#"),
        ("rejected: window_left -2", f"{_S942};causal:False;window_left:-2#"),
        (
            "rejected: persist_decode without persistent",
            f"{_S942};persist_decode:hkv_major#",
        ),
        ("rejected: bad name", f"{_S942};name:bad-name#"),
        ("rejected: bad layout", f"{_S942};layout:sbhd#"),
        ("rejected: bad oob", f"{_S942};oob:wrap#"),
        (
            "rejected: slab past the 32-bit offset",
            f"{_S950};seqlen_q:140000;ragged:True;num_query_heads:64;num_kv_heads:8#",
        ),
        ("rejected: pinned batch past 2^31", f"{_S942};batch:200000#"),
        # knobs on the shipped specs
        ("tuned0 persistent", _case(_T0, "persistent:True#")),
        ("tuned0 xcd remap", _case(_T0, "xcd_remap:True#")),
        ("tuned0 atomic dQ", _case(_T0, "dq_mode:atomic#")),
        ("tuned0 epilogue pair", _case(_T0, "epilogue:lds_staged_pair#")),
        ("tuned0 ring 3", _case(_T0, "#ring_depth:3")),
        ("tuned1 causal", _case(_T1, "causal:True#")),
        ("tuned1 ragged", _case(_T1, "ragged:True;seqlen_q:2000#")),
        ("tuned1 clamp", _case(_T1, "oob:clamp#")),
        ("tuned2 direct dkv", _case(_T2, "dkv_mode:direct#")),
        ("tuned2 split", _case(_T2, "dq_mode:split#")),
        ("tuned2 aligned", _case(_T2, "ragged:False;seqlen_q:4096;seqlen_kv:3840#")),
        ("default942 split", _case(_D942, "dq_mode:split#")),
        ("default942 atomic", _case(_D942, "dkv_mode:atomic#")),
        ("default942 window", _case(_D942, "sliding_window:512#")),
        ("default942N bhsd", _case(_D942N, "layout:bhsd#")),
        (
            "default942C persistent hkv",
            _case(_D942C, "persistent:True;persist_decode:hkv_major#"),
        ),
        ("default950 edge off", _case(_D950, "edge_tiles:False#")),
        ("default950N ring 2", _case(_D950N, "#ring_depth:2")),
    ]
    return tuple(out)


def _configs() -> tuple:
    out = [(f"case {cid}", _case(cid)) for cid, _a, _p in _planned()]
    out += _aux_configs()
    out += list(_knobs())
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    config = _resolve(_CONFIGS[idx][1])
    if config.startswith("aux:"):
        stage, arch, rest = config[len("aux:") :].split("|", 2)
        return ("aux", stage, tuple(_parse_fields(rest).items()), ()), arch
    main, _, knobs = config.partition("#")
    arch, _, rest = main.partition("|")
    fields = _parse_fields(rest)
    # one item per knob as written (a repeated name reaches the spec)
    kitems = tuple(
        item
        for part in knobs.split(";")
        if part
        for item in _parse_knobs(f"{arch}|{part}")[1].items()
    )
    return ("dense", None, tuple(fields.items()), kitems), arch


def build_attention_dense_bwd_kernel(spec, *, arch: str):
    """The kernel of one config (a spec error is a build error)."""
    kind, stage, fields, knobs = spec
    if kind == "aux":
        return _BUILDERS[stage](AttnBwdAuxSpec(**dict(fields)), arch=arch)
    fields = dict(fields)
    cls = _CLASSES.get(fields.pop("cls", arch), AttentionDenseBwdSpec)
    dense = cls(**fields, body_knobs=knobs)
    return _BUILDERS["dense"](dense, arch=arch)


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
        build_attention_dense_bwd_kernel,
        usage="usage: attention_dense_bwd_main_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
