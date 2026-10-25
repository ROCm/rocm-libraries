#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_unified_main_emit.py -- Python reference emitter
# for the whole unified (paged) attention backward main kernel
# (kernels/common/attention_bwd_unified.py: build_attn_bwd_unified_main, i.e.
# _emit_main with the general _kv_tile, the paged K / V slabs and the page
# atomic / direct dK / dV epilogues). The C twin
# attention_bwd_unified_main_emit.c builds the same kernel through
# rocke_build_attn_bwd_unified_main_new (rocke/instance_attention_bwd_unified.h)
# and the two .ll must match.
#
# Configs:
#   * "@<case id>": the main kernel of a unified case of the phase-1 lowerer
#     gate (library/tests/test_attention_bwd_ir_cpp_parity.py, _cases /
#     _plan): every unified tuned page tile on gfx942 and gfx950, in direct
#     and atomic page mode, planned here exactly as the gate plans them;
#   * "@<case id>;<field>:<value>;...": that spec with overrides;
#   * "<arch>|<field>:<value>;...": a spec from scratch.
# Field spellings are those of attention_bwd_spec_emit.py (a warp grid or an
# agpr_alloc pair is "RxC"), plus the unified page_mode and sub_page. The C
# table holds each config resolved to "<arch>|<field>:<value>;..." (the
# planned spec's non-default fields, then the overrides); regenerate it with
# `attention_bwd_unified_main_emit.py --print-c-configs` whenever the shipped
# tables or _CONFIGS change.
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
import dataclasses
import importlib.util
import sys
from functools import cache
from pathlib import Path

from _emit_common import run_emit
from attention_bwd_spec_emit import _value
from kernels.common.attention_bwd_unified import (
    AttnBwdUnifiedSpec,
    build_attn_bwd_unified_main,
)

_GATE = Path(__file__).resolve().parents[1] / "test_attention_bwd_ir_cpp_parity.py"
_ARCHES = ("gfx942", "gfx950")


@cache
def _gate():
    """The phase-1 gate module (loaded by path: its directory would shadow the
    library's ``dispatch`` package on sys.path)."""
    spec = importlib.util.spec_from_file_location("_attn_bwd_gate", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case_ids() -> list:
    """The unified cases of the gate on gfx942 / gfx950, in its order."""
    return [
        cid
        for cid, family, arch, _recipe in _gate()._cases()
        if family == "unified" and arch in _ARCHES
    ]


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, tuple):
        return f"{v[0]}x{v[1]}"
    return str(v)


def _spec_text(arch: str, spec) -> str:
    """``spec`` as an "<arch>|<field>:<value>;..." config (non-default fields)."""
    items = []
    for f in dataclasses.fields(spec):
        v = getattr(spec, f.name)
        if f.default is dataclasses.MISSING or v != f.default:
            items.append(f"{f.name}:{_fmt(v)}")
    return f"{arch}|" + ";".join(items)


@cache
def _case_text(cid: str) -> str:
    """The main spec the gate plans for case ``cid``, as a config."""
    gate = _gate()
    for case_id, family, arch, recipe in gate._cases():
        if case_id == cid:
            plan = gate._plan(family, arch, recipe)
            return _spec_text(arch, plan.specs["main"])
    raise SystemExit(f"unknown gate case {cid!r}")


def _resolve(config: str) -> str:
    """A config as "<arch>|<fields>" (gate cases planned, overrides last)."""
    if not config.startswith("@"):
        return config
    cid, _, over = config[1:].partition(";")
    text = _case_text(cid)
    return f"{text};{over}" if over else text


def _parse(config: str):
    arch, _, rest = config.partition("|")
    fields = {}
    for item in rest.split(";"):
        if item:
            name, _, text = item.partition(":")
            if name == "sub_page":
                fields[name] = None if text == "None" else int(text)
            else:
                fields[name] = _value(name, text)
    return arch, fields


def _case(cid: str, over: str = "") -> str:
    return f"@{cid};{over}" if over else f"@{cid}"


# Shipped cases the knob variants start from.
_SUB942 = "unified/gfx942/d64_p64_sub64_atomic"  # sub class, 8 waves
_SUB942D = "unified/gfx942/d64_p64_sub64_direct"
_MUL942 = "unified/gfx942/d128_p64_k64_atomic"  # multi class, 4 waves
_MUL942D = "unified/gfx942/d128_p64_k64_direct"
_SUB950 = "unified/gfx950/d32_p64_sub64_atomic"  # sub class, tr_read
_SUB950D = "unified/gfx950/d32_p64_sub64_direct"
_MUL950 = "unified/gfx950/d128_p128_k64_atomic"  # multi class, kt in LDS
_MUL950D = "unified/gfx950/d128_p128_k64_direct"
_SUB950W = "unified/gfx950/d64_p128_sub128_atomic"  # kN0 = 256
_BASE942 = "gfx942|head_size:64"
_BASE950 = "gfx950|head_size:128"


def _knobs() -> tuple:
    out = []
    # the mask, page class and dtype axes on every page mode
    for mode in ("atomic", "direct"):
        for label, base in (("gfx942 d64", _BASE942), ("gfx950 d128", _BASE950)):
            out += [
                (f"{mode} multi {label}", f"{base};page_mode:{mode}"),
                (f"{mode} no mask {label}", f"{base};page_mode:{mode};mask_class:none"),
                (f"{mode} fp16 {label}", f"{base};page_mode:{mode};dtype:fp16"),
                (f"{mode} stage_vec 1 {label}", f"{base};page_mode:{mode};stage_vec:1"),
                (f"{mode} sub 16 {label}", f"{base};page_mode:{mode};sub_page:16"),
                (f"{mode} sub 32 {label}", f"{base};page_mode:{mode};sub_page:32"),
                (
                    f"{mode} runtime scale {label}",
                    f"{base};page_mode:{mode};static_scale:runtime",
                ),
                (
                    f"{mode} dq_before_atomic {label}",
                    f"{base};page_mode:{mode};static_scale:dq_before_atomic",
                ),
                (
                    f"{mode} convert {label}",
                    f"{base};page_mode:{mode};static_scale:convert",
                ),
                (
                    f"{mode} head pack {label}",
                    f"{base};page_mode:{mode};head_pack:True",
                ),
                (
                    f"{mode} edge tiles {label}",
                    f"{base};page_mode:{mode};edge_tiles:True",
                ),
                (
                    f"{mode} head_outer grouped {label}",
                    f"{base};page_mode:{mode};step_order:head_outer;edge_tiles:True",
                ),
                (
                    f"{mode} head_outer {label}",
                    f"{base};page_mode:{mode};step_order:head_outer",
                ),
                (f"{mode} q_rotate {label}", f"{base};page_mode:{mode};q_rotate:True"),
                (
                    f"{mode} interior_clamp {label}",
                    f"{base};page_mode:{mode};edge_tiles:True;interior_clamp:True",
                ),
                (
                    f"{mode} token major {label}",
                    f"{base};page_mode:{mode};ws_layout:token_major",
                ),
                (
                    f"{mode} stats_vec 4 {label}",
                    f"{base};page_mode:{mode};stats_vec:4;stats_fma:True",
                ),
                (
                    f"{mode} counted {label}",
                    f"{base};page_mode:{mode};loop_sync:counted",
                ),
                (
                    f"{mode} lds_only {label}",
                    f"{base};page_mode:{mode};loop_sync:lds_only",
                ),
                (
                    f"{mode} barrier merge {label}",
                    f"{base};page_mode:{mode};ring_depth:2;barrier_merge:on",
                ),
                (
                    f"{mode} acc_in_lds {label} d32",
                    f"{base.split('|')[0]}|head_size:32;page_mode:{mode};acc_in_lds:True",
                ),
                (
                    f"{mode} kv in LDS {label}",
                    f"{base};page_mode:{mode};kv_residency:lds;kt_source:lds",
                ),
            ]
            out += [
                (f"{mode} sched {sc} {label}", f"{base};page_mode:{mode};sched:{sc}")
                for sc in ("iglp0", "fences_only", "stage_table", "stage_table_atomics")
            ]
    # dK / dV epilogues: atomic guards; direct-mode store paths and widths
    out += [
        ("atomic select gfx942", f"{_BASE942};dkv_atomic_guard:select"),
        ("atomic select gfx950 sub", f"{_BASE950};dkv_atomic_guard:select;sub_page:32"),
        (
            "atomic select runtime gfx942",
            f"{_BASE942};dkv_atomic_guard:select;static_scale:runtime",
        ),
    ]
    for ep in ("direct", "lds_staged", "lds_staged_pair"):
        for label, base in (("gfx942 d64", _BASE942), ("gfx950 d128", _BASE950)):
            out += [
                (f"direct {ep} {label}", f"{base};page_mode:direct;dkv_epilogue:{ep}"),
                (
                    f"direct {ep} sub 16 {label}",
                    f"{base};page_mode:direct;dkv_epilogue:{ep};sub_page:16",
                ),
                (
                    f"direct {ep} runtime {label}",
                    f"{base};page_mode:direct;dkv_epilogue:{ep};static_scale:runtime",
                ),
            ]
    for vec in (1, 2, 4):
        out += [
            (
                f"direct lds_staged vec {vec} gfx942",
                f"{_BASE942};page_mode:direct;dkv_store_vec:{vec}",
            ),
            (
                f"direct pair vec {vec} gfx950",
                f"{_BASE950};page_mode:direct;dkv_epilogue:lds_staged_pair;dkv_store_vec:{vec}",
            ),
        ]
    # 4 waves, kM0 = kN0 = 32 on the small-tile grids: the LDS-staged
    # epilogue leaves a tail round
    small = (
        "gfx942|head_size:32;waves:4;block_m:32;block_n:32;block_k4:32;"
        "warp_grid_g4:2x2;warp_grid_g02:2x2;warp_grid_g13:2x2;pt_route:lds"
    )
    out += [
        ("direct tail round gfx942 d32", f"{small};page_mode:direct"),
        (
            "direct pair tail round gfx942 d32",
            f"{small};page_mode:direct;dkv_epilogue:lds_staged_pair",
        ),
        ("atomic tail round gfx942 d32", small),
        (
            "direct tail round sub 16 gfx942 d32",
            f"{small};page_mode:direct;sub_page:16",
        ),
    ]
    # RDNA (wave32, WMMA)
    out += [
        ("rdna gfx1201 d64 atomic", "gfx1201|head_size:64"),
        ("rdna gfx1201 d64 direct", "gfx1201|head_size:64;page_mode:direct"),
        ("rdna gfx1151 d64 atomic no mask", "gfx1151|head_size:64;mask_class:none"),
        (
            "rdna gfx1151 d64 direct fp16",
            "gfx1151|head_size:64;page_mode:direct;dtype:fp16",
        ),
    ]
    # knobs on shipped specs
    for cid in (
        _SUB942,
        _SUB942D,
        _MUL942,
        _MUL942D,
        _SUB950,
        _SUB950D,
        _MUL950,
        _MUL950D,
        _SUB950W,
    ):
        out += [
            (f"{cid} no mask", _case(cid, "mask_class:none")),
            (f"{cid} head pack", _case(cid, "head_pack:True")),
            (f"{cid} edge tiles", _case(cid, "edge_tiles:True")),
            (f"{cid} counted", _case(cid, "loop_sync:counted")),
            (f"{cid} stage_table_atomics", _case(cid, "sched:stage_table_atomics")),
            (f"{cid} runtime scale", _case(cid, "static_scale:runtime")),
            (f"{cid} fp16", _case(cid, "dtype:fp16")),
        ]
    out += [
        ("multi gfx942 atomic select", _case(_MUL942, "dkv_atomic_guard:select")),
        ("multi gfx942 direct pair", _case(_MUL942D, "dkv_epilogue:lds_staged_pair")),
        ("multi gfx942 direct cells", _case(_MUL942D, "dkv_epilogue:direct")),
        ("sub gfx950 direct pair", _case(_SUB950D, "dkv_epilogue:lds_staged_pair")),
        ("sub gfx950 direct cells", _case(_SUB950D, "dkv_epilogue:direct")),
        ("sub gfx942 sub 32", _case(_SUB942, "sub_page:32")),
        ("sub gfx942 direct sub 16", _case(_SUB942D, "sub_page:16")),
    ]
    # kernel names and the _kernel_attrs knobs (agpr pair on MFMA,
    # mfma_vgpr_form, waves_per_eu, scheduler_strategy)
    out += [
        ("name latin-1 letter", f"{_BASE942};name:attn_\u00e9"),
        ("name latin-1 digit", f"{_BASE950};page_mode:direct;name:k\u00b2_\u00bd"),
        (
            "agpr pair agpr_fence wpe 2 max-ilp gfx942",
            (
                f"{_BASE942};agpr_alloc:128x128;agpr_route:agpr_fence;waves_per_eu:2;"
                "scheduler_strategy:max-ilp"
            ),
        ),
        (
            "agpr pair vgpr_form max-memory-clause gfx950 direct",
            (
                f"{_BASE950};page_mode:direct;agpr_alloc:128x128;agpr_route:vgpr_form;"
                "scheduler_strategy:max-memory-clause"
            ),
        ),
        (
            "agpr asm_g13 iterative-minreg gfx942 direct",
            (
                f"{_BASE942};page_mode:direct;agpr_alloc:192x192;agpr_route:asm_g13;"
                "scheduler_strategy:iterative-minreg"
            ),
        ),
        ("agpr backend gfx942", f"{_BASE942};agpr_alloc:backend"),
        (
            "agpr pair on shipped multi gfx942",
            _case(_MUL942, "agpr_alloc:128x128;waves_per_eu:2"),
        ),
    ]
    # refused by both engines (spec or validation errors)
    out += [
        ("rejected: last_cta sink", f"{_BASE942};dkv_sink:last_cta"),
        ("rejected: empty name", f"{_BASE942};name:"),
        ("rejected: name with dash", f"{_BASE942};name:bad-name"),
        ("rejected: name of underscores", f"{_BASE942};name:__"),
        ("rejected: name with multiply sign", f"{_BASE942};name:a\u00d7b"),
        ("rejected: wide name with dash", f"{_BASE942};name:attn_\u03a9-x"),
        ("rejected: scheduler_strategy", f"{_BASE942};scheduler_strategy:bogus"),
        ("rejected: xcd_remap", f"{_BASE942};xcd_remap:True"),
        ("rejected: lb_mirror", f"{_BASE942};lb_mirror:True"),
        ("rejected: ring without merge", f"{_BASE942};ring_depth:2"),
        ("rejected: dma path", f"{_BASE950};global_path:dma;ring_depth:2"),
        ("rejected: sub_page 0", f"{_BASE942};sub_page:0"),
        ("rejected: sub_page 24", f"{_BASE942};sub_page:24"),
        ("rejected: sub_page = block_n", _case(_SUB942, "sub_page:128")),
        ("rejected: slices sink", f"{_BASE942};dkv_sink:slices"),
        ("rejected: ws_bounds buffer", f"{_BASE942};ws_bounds:buffer"),
        ("rejected: narrow_vec 2", f"{_BASE942};stage_vec:1;narrow_vec:2"),
        ("rejected: page_mode", f"{_BASE942};page_mode:scatter"),
        ("rejected: head_size", "gfx942|head_size:48"),
        ("rejected: unknown arch", "gfx1100|head_size:64"),
    ]
    return tuple(out)


def _configs() -> tuple:
    out = [(f"case {cid}", _case(cid)) for cid in _case_ids()]
    out += list(_knobs())
    return tuple(out)


_CONFIGS = _configs()


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    arch, fields = _parse(_resolve(_CONFIGS[idx][1]))
    return fields, arch


def build_attention_bwd_unified_main_kernel(spec, *, arch: str):
    """The unified main kernel of one config."""
    return build_attn_bwd_unified_main(AttnBwdUnifiedSpec(**dict(spec)), arch=arch)


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
        build_attention_bwd_unified_main_kernel,
        usage="usage: attention_bwd_unified_main_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
