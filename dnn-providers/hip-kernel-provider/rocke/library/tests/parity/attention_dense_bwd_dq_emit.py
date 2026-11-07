#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_dense_bwd_dq_emit.py -- Python reference emitter for
# the split dQ kernel of the dense attention backward
# (kernels/common/attention_dense_bwd_dq.py: build_attention_dense_bwd_dq, the
# general split dQ body behind the dense addressing). The C twin
# attention_dense_bwd_dq_emit.c builds the same kernel through
# rocke_build_attn_dense_bwd_dq_new (rocke/instance_attention_dense_bwd_dq.h)
# and the two .ll must match.
#
# Neither engine consults the recorded dQ resource-gate refusals here (the C
# validator has no copy of them), so no config is one of them.
#
# Configs:
#   * "@<case id>": the dense dQ kernel of a dense case of the phase-1 lowerer
#     gate (library/tests/test_attention_bwd_ir_cpp_parity.py, _cases /
#     _plan) whose plan launches one, planned exactly as the gate plans it;
#   * "@<case id>;<dense field>:<value>;...#<dq field>:<value>;...": that
#     spec with overrides;
#   * "<arch>|<dense field>:<value>;...#<dq field>:<value>;...": a spec from
#     scratch: AttentionDenseBwdDqSpec(<dense fields>) (dq=None: the
#     problem's default AttnBwdDqSpec), then, when "#" is present, its dq
#     replaced by dataclasses.replace(spec.dq, <dq fields>) (the dQ field
#     spellings of attention_bwd_dq_emit.py).
# The C table holds each config resolved (cases planned); regenerate it with
# `attention_dense_bwd_dq_emit.py --print-c-configs` whenever the shipped
# tables or _CONFIGS change.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
import dataclasses
import importlib.util
import sys
from functools import cache
from pathlib import Path

from _emit_common import run_emit
from attention_bwd_dq_emit import _parse as _parse_dq
from kernels.common.attention_dense_bwd_dq import (
    AttentionDenseBwdDqSpec,
    build_attention_dense_bwd_dq,
)

_GATE = Path(__file__).resolve().parents[1] / "test_attention_bwd_ir_cpp_parity.py"
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
}
_BOOL = {"causal", "causal_bottom_right"}


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
    """``((case_id, arch, dense dq spec), ...)``: every gate case whose plan
    launches a "dense_dq" kernel."""
    gate = _gate()
    out = []
    for cid, family, arch, recipe in gate._cases():
        if family != "dense":
            continue
        plan = gate._plan(family, arch, recipe)
        if any(step.stage == "dense_dq" for step in plan.launches):
            out.append((cid, arch, plan.specs["dense_dq"]))
    return tuple(out)


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, tuple):
        return f"{v[0]}x{v[1]}"
    return str(v)


def _spec_text(arch: str, spec: AttentionDenseBwdDqSpec) -> str:
    """``spec`` as "<arch>|<dense fields>#<dq fields>": the dq fields that
    differ from the problem's default dQ spec."""
    items = []
    for f in dataclasses.fields(spec):
        if f.name == "dq":
            continue
        v = getattr(spec, f.name)
        if f.default is dataclasses.MISSING or v != f.default:
            items.append(f"{f.name}:{_fmt(v)}")
    base = dataclasses.replace(spec, dq=None).dq
    dq = [
        f"{f.name}:{_fmt(getattr(spec.dq, f.name))}"
        for f in dataclasses.fields(spec.dq)
        if getattr(spec.dq, f.name) != getattr(base, f.name)
    ]
    return f"{arch}|" + ";".join(items) + "#" + ";".join(dq)


@cache
def _case_text(cid: str) -> str:
    for case_id, arch, spec in _planned():
        if case_id == cid:
            return _spec_text(arch, spec)
    raise SystemExit(f"gate case {cid!r} has no dense dq kernel")


def _resolve(config: str) -> str:
    if not config.startswith("@"):
        return config
    cid, _, over = config[1:].partition(";")
    text = _case_text(cid)
    if not over:
        return text
    dense_over, _, dq_over = over.partition("#")
    dense, _, dq = text.partition("#")
    if dense_over:
        dense = f"{dense};{dense_over}"
    if dq_over:
        dq = f"{dq};{dq_over}" if dq else dq_over
    return f"{dense}#{dq}"


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


def _case(cid: str, over: str = "") -> str:
    return f"@{cid};{over}" if over else f"@{cid}"


_T0 = "dense/gfx950/tuned0"  # causal BR, d128, GQA 4, head_loop group
_T1 = "dense/gfx950/tuned1"  # no mask, d64, MHA, single head
_S942 = (
    "gfx942|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:64"
)
_S950 = (
    "gfx950|seqlen_q:256;seqlen_kv:512;num_query_heads:8;num_kv_heads:2;head_size:128"
)
_R942 = (
    "gfx942|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:64"
)
_R950 = (
    "gfx950|seqlen_q:200;seqlen_kv:300;num_query_heads:8;num_kv_heads:2;head_size:128"
)


def _knobs() -> tuple:
    out = []
    for base, ragged, tag in ((_S942, _R942, "gfx942"), (_S950, _R950, "gfx950")):
        out += [
            (f"default {tag}", base),
            (f"no mask {tag}", f"{base};causal:False"),
            (f"causal br {tag}", f"{base};causal_bottom_right:True"),
            (f"window 64 {tag}", f"{base};sliding_window:64"),
            (
                f"band 32x16 {tag}",
                f"{base};causal:False;window_left:32;window_right:16",
            ),
            (f"band right only {tag}", f"{base};causal:False;window_right:0"),
            (f"ragged {tag}", ragged),
            (f"ragged no mask {tag}", f"{ragged};causal:False"),
            (f"ragged clamp {tag}", f"{ragged};oob:clamp"),
            (f"bhsd {tag}", f"{base};layout:bhsd"),
            (f"fp16 {tag}", f"{base};dtype:fp16"),
            (f"batch 2 {tag}", f"{base};batch:2"),
            (f"clamp {tag}", f"{base};oob:clamp"),
            (f"token-major {tag}", f"{base};ws_layout:token_major"),
            (f"name stem {tag}", f"{base};name:my_dense"),
            # the dQ tuning values
            (f"edge tiles {tag}", f"{base}#edge_tiles:True"),
            (f"edge tiles no mask {tag}", f"{base};causal:False#edge_tiles:True"),
            (f"edge tiles ragged {tag}", f"{ragged}#edge_tiles:True"),
            (
                f"edge tiles ragged no mask {tag}",
                f"{ragged};causal:False#edge_tiles:True",
            ),
            (f"edge tiles window {tag}", f"{base};sliding_window:64#edge_tiles:True"),
            (f"group {tag}", f"{base}#head_loop:group"),
            (f"group ragged {tag}", f"{ragged}#head_loop:group"),
            (f"group clamp {tag}", f"{base};oob:clamp#head_loop:group"),
            (f"group edge {tag}", f"{base}#head_loop:group;edge_tiles:True"),
            (f"group xcd {tag}", f"{base}#head_loop:group;xcd_remap:True"),
            (f"xcd {tag}", f"{base}#xcd_remap:True"),
            (f"prefetch {tag}", f"{base}#kv_prefetch:True"),
            (f"prefetch clamp {tag}", f"{base};oob:clamp#kv_prefetch:True"),
            (
                f"prefetch edge ragged {tag}",
                f"{ragged}#kv_prefetch:True;edge_tiles:True",
            ),
            (f"stats_prescale off {tag}", f"{base}#stats_prescale:False"),
            (f"runtime scale {tag}", f"{base}#static_scale:runtime"),
            (f"convert scale {tag}", f"{base}#static_scale:convert"),
            (f"kv_pad 16 {tag}", f"{base}#kv_pad:16"),
            (f"block_n 32 {tag}", f"{base}#block_n:32"),
            (f"sched iglp1 {tag}", f"{base}#sched:iglp1"),
            (f"sched stage_table {tag}", f"{base}#sched:stage_table"),
            (f"waves_per_eu 2 {tag}", f"{base}#waves_per_eu:2"),
            (f"agpr_alloc 0x0 {tag}", f"{base}#agpr_alloc:0x0"),
            (f"scheduler_strategy {tag}", f"{base}#scheduler_strategy:max-ilp"),
            # rejected (both engines)
            (
                f"rejected: group not dividing block_m {tag}",
                f"{base};num_query_heads:6;num_kv_heads:2#head_loop:group;block_m:64",
            ),
            (f"rejected: dq mask mismatch {tag}", f"{base}#mask_class:none"),
            (f"rejected: dq head size mismatch {tag}", f"{base}#head_size:32"),
            (f"rejected: thd dq {tag}", f"{base}#seq_mode:thd"),
        ]
    out += [
        ("lds_plain gfx950", f"{_S950}#transpose_source:lds_plain"),
        (
            "rejected: unknown arch",
            "gfx1201|seqlen_q:256;seqlen_kv:256;num_query_heads:8;num_kv_heads:8;head_size:64",
        ),
        ("rejected: head size 32", f"{_S942};head_size:32"),
        ("rejected: window without causal", f"{_S942};causal:False;sliding_window:8"),
        (
            "rejected: slab past the 32-bit offset",
            f"{_S950};seqlen_q:140000;num_query_heads:64;num_kv_heads:8",
        ),
        ("rejected: pinned batch past 2^31", f"{_S942};batch:200000"),
        ("rejected: bad oob", f"{_S942};oob:wrap"),
        # knobs on the shipped specs
        ("tuned0 single", _case(_T0, "#head_loop:single")),
        ("tuned0 prefetch", _case(_T0, "#kv_prefetch:True")),
        ("tuned0 xcd", _case(_T0, "#xcd_remap:True")),
        ("tuned0 edge off", _case(_T0, "#edge_tiles:False")),
        ("tuned0 clamp", _case(_T0, "oob:clamp#")),
        ("tuned0 ragged q", _case(_T0, "seqlen_q:1000#")),
        ("tuned0 window", _case(_T0, "sliding_window:300#")),
        ("tuned1 group", _case(_T1, "#head_loop:group")),
        ("tuned1 causal", _case(_T1, "causal:True#mask_class:band")),
        ("tuned1 ragged", _case(_T1, "seqlen_q:2000;seqlen_kv:2001#")),
        ("tuned1 batch", _case(_T1, "batch:4#")),
        ("tuned1 bhsd", _case(_T1, "layout:bhsd#")),
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
    dense, hash_, dq = config.partition("#")
    arch, _, rest = dense.partition("|")
    fields = {}
    for item in rest.split(";"):
        if item:
            name, _, text = item.partition(":")
            fields[name] = _value(name, text)
    dq_fields = None
    if hash_:
        _a, dq_fields = _parse_dq(f"{arch}|{dq}")
        dq_fields = tuple(dq_fields.items())
    return (tuple(fields.items()), dq_fields), arch


def build_attention_dense_bwd_dq_kernel(spec, *, arch: str):
    """The dense dQ kernel of one config (a spec error is a build error)."""
    fields, dq_fields = spec
    dense = AttentionDenseBwdDqSpec(**dict(fields))
    if dq_fields is not None:
        dq = dataclasses.replace(dense.dq, **dict(dq_fields))
        dense = dataclasses.replace(dense, dq=dq)
    return build_attention_dense_bwd_dq(dense, arch=arch)


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
        build_attention_dense_bwd_dq_kernel,
        usage="usage: attention_dense_bwd_dq_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
