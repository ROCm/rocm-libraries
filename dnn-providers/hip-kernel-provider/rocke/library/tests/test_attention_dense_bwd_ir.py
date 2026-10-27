# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense SDPA backward IR (CPU: pure text lowering, no device, no comgr).

* Golden LLVM-IR hashes of representative dense backward instances on gfx942
  and gfx950 (``tests/golden/attention_dense_bwd_ir_sha256.json``, one entry
  per LLVM flavor): every mask form, both layouts and dtypes, both head
  sizes, ragged, atomic dK / dV, both persistent decodes, the chiplet remap,
  the clamp addressing, every epilogue, the token-major workspace, edge tiles
  off, a pinned batch, the interleave decode, every multiplier applied at run
  time, ``noalias`` workspace pointers, and the Q / dO prefetch ring (register
  path on both arches; depth 3 and the DMA path at depths 1-3 on gfx950, with
  the best general tile, atomic dK / dV, a persistent grid, both OOB modes
  and ragged kernels), the split-dQ main kernel (``dq_mode = "split"``: plain,
  unmasked, atomic dK / dV, ragged, the prefetch ring, the DMA ring with the
  best tiles) and the dense split dQ kernel (``<arch>/dq/<id>``: both OOB
  modes, ragged, edge / interior key loops, the key prefetch, head packing,
  the chiplet remap, a sliding window, a pinned batch with the token-major
  workspace, the convert scale placement). The resource gate runs on every
  instance. Re-bless with ``python tests/test_attention_dense_bwd_ir.py
  --write``.
* The emitted kernel signature is the ``rocke.attn_dense_bwd.v1`` list.
* The dense backward modules never import the dense forward modules, so the
  forward golden IR (``attention_dense_gfx942_ir_sha256.json`` and the
  gfx950 dense tests) cannot move with them.
* IR facts: buffer resources only under ``oob = "buffer_oob"``; the edge
  q loops are not emitted when no tile is an edge tile; a runtime batch reads
  the ``batch`` kernarg; knob values that are not built raise before IR; the
  body's ``ptr_noalias`` marks the dense pointers by the general family's
  rule; the multipliers that are 1.0 under ``static_scale`` are never read.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
from ._attention_bwd_test_utils import kernel_signature

_TESTS = Path(__file__).resolve().parent
_LIB_ROOT = str(_TESTS.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)

_GOLDEN = _TESTS / "golden" / "attention_dense_bwd_ir_sha256.json"
_FLAVORS = ("llvm20", "llvm22", "llvm23")
_ARCHS = ("gfx942", "gfx950")


def _spec_cls(arch):
    if arch == "gfx942":
        from kernels.gfx942.attention_dense_bwd import Gfx942AttentionDenseBwdSpec

        return Gfx942AttentionDenseBwdSpec
    from kernels.gfx950.attention_dense_bwd import Gfx950AttentionDenseBwdSpec

    return Gfx950AttentionDenseBwdSpec


def _builder(arch):
    import importlib

    return importlib.import_module(
        f"kernels.{arch}.attention_dense_bwd"
    ).build_attention_dense_bwd


_BASE = dict(
    seqlen_q=256, seqlen_kv=256, num_query_heads=8, num_kv_heads=2, head_size=128
)
# (case id, spec fields over _BASE)
_CASES = (
    ("causal_tl_bf16_d128_bshd", {}),
    (
        "none_fp16_d64_bhsd",
        dict(causal=False, dtype="fp16", head_size=64, layout="bhsd"),
    ),
    (
        "causal_br_fp16_d128_sq128",
        dict(causal_bottom_right=True, dtype="fp16", seqlen_q=128),
    ),
    ("swa64_bf16_d128", dict(sliding_window=64)),
    (
        "win7x3_br_bf16_d64_bhsd",
        dict(
            causal=False,
            window_left=7,
            window_right=3,
            causal_bottom_right=True,
            head_size=64,
            layout="bhsd",
        ),
    ),  # fmt: skip
    ("ragged_causal_bf16_d128_s200", dict(seqlen_q=200, seqlen_kv=200, ragged=True)),
    (
        "ragged_none_fp16_d64_s100x72",
        dict(
            causal=False,
            seqlen_q=100,
            seqlen_kv=72,
            ragged=True,
            dtype="fp16",
            head_size=64,
        ),
    ),  # fmt: skip
    ("atomic_causal_bf16_d128", dict(dkv_mode="atomic")),
    ("persist_kvtile_bf16_d128", dict(persistent=True)),
    ("persist_hkv_bf16_d128", dict(persistent=True, persist_decode="hkv_major")),
    ("xcd_remap_bf16_d128", dict(xcd_remap=True)),
    ("clamp_ragged_bf16_d128", dict(oob="clamp", seqlen_kv=200, ragged=True)),
    ("epilogue_direct_bf16_d128", dict(epilogue="direct")),
    ("epilogue_pair_bf16_d128", dict(epilogue="lds_staged_pair")),
    ("token_major_bf16_d128", dict(ws_layout="token_major")),
    ("no_edge_tiles_bf16_d128", dict(edge_tiles=False)),
    ("batch2_bf16_d128", dict(batch=2)),
    ("mha_none_bf16_d128", dict(num_kv_heads=8, causal=False)),
    ("interleave_bf16_d128", dict(body_knobs={"lb_interleave": True})),
    ("static_runtime_bf16_d128", dict(body_knobs={"static_scale": "runtime"})),
    ("noalias_ws_bf16_d128", dict(body_knobs={"ptr_noalias": "workspace"})),
    # the Q / dO prefetch ring of the dense shell (register path, depth 2)
    ("ring_vgpr2_bf16_d128", dict(body_knobs={"ring_depth": 2})),
    (
        "ring_vgpr2_clamp_ragged_bf16_d128",
        dict(oob="clamp", seqlen_kv=200, ragged=True, body_knobs={"ring_depth": 2}),
    ),
    (
        "ring_vgpr2_ragged_none_fp16_d64",
        dict(
            causal=False,
            seqlen_q=100,
            seqlen_kv=72,
            ragged=True,
            dtype="fp16",
            head_size=64,
            body_knobs={"ring_depth": 2},
        ),
    ),  # fmt: skip
    (
        "ring_vgpr2_atomic_bf16_d128",
        dict(dkv_mode="atomic", body_knobs={"ring_depth": 2}),
    ),
    # split dQ: the main kernel computes dK / dV only
    ("split_causal_bf16_d128", dict(dq_mode="split")),
    (
        "split_none_fp16_d64_bhsd",
        dict(dq_mode="split", causal=False, dtype="fp16", head_size=64, layout="bhsd"),
    ),
    (
        "split_ring_vgpr2_atomic_bf16_d128",
        dict(dq_mode="split", dkv_mode="atomic", body_knobs={"ring_depth": 2}),
    ),
    (
        "split_ragged_br_fp16_d64",
        dict(
            dq_mode="split", causal_bottom_right=True, seqlen_q=136, seqlen_kv=200,
            ragged=True, dtype="fp16", head_size=64,
        ),
    ),  # fmt: skip
)
# The gfx950 tile of the best general configuration (stats levers on).
_WIN_A = {
    "atom_g02": "16x16x32", "atom_g13": "16x16x16", "atom_g4": "16x16x32",
    "block_k4": 32, "block_m": 32, "block_n": 128, "pt_route": "relabel",
    "transpose_source": "tr_read", "warp_grid_g02": (1, 8),
    "warp_grid_g13": (8, 1), "warp_grid_g4": (2, 4), "waves": 8,
    "waves_per_eu": 1, "stats_vec": 4, "stats_fma": True,
}  # fmt: skip
# The gfx950 d64 four-wave tile of the split-dQ search (DMA ring, stats
# levers, fast exp2).
_W4_D64 = {
    "atom_g02": "16x16x32", "atom_g13": "16x16x16", "atom_g4": "16x16x32",
    "block_k4": 32, "block_m": 32, "block_n": 128, "pt_route": "relabel",
    "transpose_source": "tr_read", "warp_grid_g02": (1, 4),
    "warp_grid_g13": (4, 1), "warp_grid_g4": (2, 2), "waves": 4,
    "waves_per_eu": 2, "stats_vec": 4, "stats_fma": True, "bwd_exp2": "fast",
    "kv_residency": "reg", "kt_source": "reg", "global_path": "dma",
    "ring_depth": 2,
}  # fmt: skip
# Instances of one arch only (values the other arch's catalog refuses: the
# DMA path and depth 3 are gfx950 values).
_ARCH_CASES = {
    "gfx950": (
        ("ring_vgpr3_bf16_d128", dict(body_knobs={"ring_depth": 3})),
        ("ring_dma1_bf16_d128", dict(body_knobs={"global_path": "dma"})),
        (
            "ring_dma2_bf16_d128",
            dict(body_knobs={"global_path": "dma", "ring_depth": 2}),
        ),
        (
            "ring_dma3_bf16_d128",
            dict(body_knobs={"global_path": "dma", "ring_depth": 3}),
        ),
        (
            "ring_dma2_winA_stats_bf16_d128",
            dict(body_knobs=dict(_WIN_A, global_path="dma", ring_depth=2)),
        ),
        (
            "ring_dma2_winA_atomic_bf16_d128",
            dict(
                dkv_mode="atomic", edge_tiles=False,
                body_knobs=dict(_WIN_A, global_path="dma", ring_depth=2),
            ),
        ),  # fmt: skip
        (
            "ring_dma2_persist_bf16_d128",
            dict(persistent=True, body_knobs={"global_path": "dma", "ring_depth": 2}),
        ),
        (
            "ring_dma2_clamp_ragged_bf16_d128",
            dict(
                oob="clamp", seqlen_q=200, ragged=True,
                body_knobs={"global_path": "dma", "ring_depth": 2},
            ),
        ),  # fmt: skip
        (
            "ring_dma2_ragged_br_fp16_d64_bhsd",
            dict(
                causal_bottom_right=True, seqlen_q=136, seqlen_kv=200, ragged=True,
                dtype="fp16", head_size=64, layout="bhsd",
                body_knobs={"global_path": "dma", "ring_depth": 2},
            ),
        ),  # fmt: skip
        # split dQ with the DMA ring: the best general tile, and the d64
        # four-wave tile of the split search (stats levers, fast exp2)
        (
            "split_ring_dma2_winA_bf16_d128",
            dict(dq_mode="split", body_knobs=dict(_WIN_A, global_path="dma",
                                                  ring_depth=2)),
        ),  # fmt: skip
        (
            "split_ring_dma2_g2_winA_bf16_d128",
            dict(dq_mode="split", dkv_mode="atomic", edge_tiles=False,
                 body_knobs=dict(_WIN_A, global_path="dma", ring_depth=2)),
        ),  # fmt: skip
        (
            "split_ring_dma2_w4_none_bf16_d64",
            dict(dq_mode="split", causal=False, head_size=64,
                 seqlen_q=512, seqlen_kv=512, body_knobs=_W4_D64),
        ),  # fmt: skip
    ),
}


def _cases(arch):
    """Every golden instance of ``arch``."""
    return _CASES + _ARCH_CASES.get(arch, ())


_ALL = [(a, cid, over) for a in ("gfx942", "gfx950") for cid, over in _cases(a)]

# The dense split dQ kernel (``dq_mode = "split"``): (case id, problem fields
# over _BASE, with its dQ tuning values under "dq"). Keyed "<arch>/dq/<id>".
_DQ_CASES = (
    ("causal_bf16_d128", {}),
    (
        "none_fp16_d64_bhsd_edge",
        dict(
            causal=False,
            dtype="fp16",
            head_size=64,
            layout="bhsd",
            dq=dict(edge_tiles=True),
        ),
    ),  # fmt: skip
    (
        "causal_edge_prefetch_bf16_d128",
        dict(dq=dict(edge_tiles=True, kv_prefetch=True)),
    ),
    ("causal_group_edge_bf16_d128", dict(dq=dict(head_loop="group", edge_tiles=True))),
    (
        "clamp_ragged_br_fp16_d64",
        dict(
            oob="clamp",
            causal_bottom_right=True,
            seqlen_q=136,
            seqlen_kv=200,
            dtype="fp16",
            head_size=64,
            dq=dict(edge_tiles=True),
        ),
    ),  # fmt: skip
    (
        "buffer_ragged_br_fp16_d64",
        dict(
            causal_bottom_right=True,
            seqlen_q=136,
            seqlen_kv=200,
            dtype="fp16",
            head_size=64,
            dq=dict(edge_tiles=True),
        ),
    ),  # fmt: skip
    (
        "group_ragged_br_fp16_d64",
        dict(
            causal_bottom_right=True,
            seqlen_q=136,
            seqlen_kv=200,
            dtype="fp16",
            head_size=64,
            dq=dict(head_loop="group", edge_tiles=True),
        ),
    ),  # fmt: skip
    ("xcd_bf16_d128", dict(dq=dict(xcd_remap=True))),
    ("swa64_bf16_d128", dict(sliding_window=64, dq=dict(edge_tiles=True))),
    (
        "bm128_none_bf16_d64",
        dict(
            causal=False,
            head_size=64,
            seqlen_q=512,
            seqlen_kv=512,
            dq=dict(block_m=128, edge_tiles=True),
        ),
    ),  # fmt: skip
    ("batch2_token_major_bf16_d128", dict(batch=2, ws_layout="token_major")),
    ("static_convert_bf16_d128", dict(dq=dict(static_scale="convert"))),
)
_DQ_ALL = [(a, cid, over) for a in _ARCHS for cid, over in _DQ_CASES]


def _dq_spec(over):
    from kernels.common.attention_bwd_dq import AttnBwdDqSpec
    from kernels.common.attention_dense_bwd_dq import AttentionDenseBwdDqSpec

    d = dict(_BASE)
    d.update(over)
    knobs = d.pop("dq", {})
    causal = d.get("causal", True) or d.get("sliding_window", 0)
    dq = AttnBwdDqSpec(
        head_size=d["head_size"], dtype=d.get("dtype", "bf16"),
        mask_class="band" if causal else "none",
        ws_layout=d.get("ws_layout", "head_major"), **knobs,
    )  # fmt: skip
    return AttentionDenseBwdDqSpec(**d, dq=dq)


def _dq_kernel(arch, over):
    from kernels.common.attention_dense_bwd_dq import build_attention_dense_bwd_dq

    return build_attention_dense_bwd_dq(_dq_spec(over), arch=arch)


def _spec(arch, over):
    d = dict(_BASE)
    d.update(over)
    return _spec_cls(arch)(**d)


def _kernel(arch, over):
    return _builder(arch)(_spec(arch, over), arch=arch)


def _lower(kernel, arch, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


def _sha(arch, over, flavor, *, dq=False):
    kernel = _dq_kernel(arch, over) if dq else _kernel(arch, over)
    data = _lower(kernel, arch, flavor).encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _golden_rows(arch):
    """``(golden key, over, is a dQ instance)`` of every instance of ``arch``."""
    return [(f"{arch}/{cid}", over, False) for cid, over in _cases(arch)] + [
        (f"{arch}/dq/{cid}", over, True) for cid, over in _DQ_CASES
    ]


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _build_doc():
    doc = {"schema": "attention_dense_bwd.ir_golden_sha256/v1", "flavors": {}}
    for flavor in _FLAVORS:
        cases = {}
        for arch in _ARCHS:
            for key, over, dq in _golden_rows(arch):
                sha, nbytes = _sha(arch, over, flavor, dq=dq)
                cases[key] = {"sha256": sha, "bytes": nbytes}
        doc["flavors"][flavor] = {"cases": cases}
    return doc


@pytest.mark.parametrize("flavor", _FLAVORS)
@pytest.mark.parametrize("arch", _ARCHS)
def test_dense_bwd_ir_matches_golden(arch, flavor):
    """Every flavor is checked in every run (the lowering takes the flavor as
    an argument); a missing fixture, flavor or case fails, never skips."""
    if not _GOLDEN.exists():
        pytest.fail("dense backward golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    gflav = golden.get("flavors", {}).get(flavor)
    if not gflav:
        pytest.fail(f"no dense backward golden recorded for llvm flavor {flavor!r}")
    drift = []
    for key, over, dq in _golden_rows(arch):
        want = gflav["cases"].get(key, {}).get("sha256")
        if want is None:
            drift.append(f"{key}: missing from the golden")
            continue
        got, _ = _sha(arch, over, flavor, dq=dq)
        if got != want:
            drift.append(f"{key}: {want} -> {got}")
    assert not drift, "dense backward IR drift vs golden:\n  " + "\n  ".join(drift)


def test_golden_covers_every_case_and_flavor():
    golden = json.loads(_GOLDEN.read_text())
    want = {key for a in _ARCHS for key, _, _ in _golden_rows(a)}
    for flavor in _FLAVORS:
        got = set(golden["flavors"][flavor]["cases"])
        assert got == want, flavor


_INPUTS = {"Q", "K", "V", "O", "dO", "LSE", "WS_LSE2", "WS_DSUM"}


@pytest.mark.parametrize("arch", _ARCHS)
@pytest.mark.parametrize(
    "over", [{}, dict(dkv_mode="atomic"), dict(persistent=True), dict(xcd_remap=True)]
)
def test_signature_is_the_abi(arch, over):
    from kernels.common.attention_dense_bwd_spec import attn_dense_bwd_params

    sig = kernel_signature(_lower(_kernel(arch, over), arch, "llvm22"))
    assert [(n, k) for n, k, _ in sig] == list(attn_dense_bwd_params())
    for name, kind, text in sig:
        assert "noalias" not in text
        if kind == "Q":
            assert ("readonly" in text) == (name in _INPUTS), text


@pytest.mark.parametrize("arch", _ARCHS)
@pytest.mark.parametrize(
    "value,over", [("workspace", {}), ("all", dict(dkv_mode="atomic"))]
)
def test_ptr_noalias_marks_the_dense_pointers(arch, value, over):
    over = dict(over, body_knobs={"ptr_noalias": value})
    sig = kernel_signature(_lower(_kernel(arch, over), arch, "llvm22"))
    want = {
        n for n, k, _ in sig if k == "Q" and (value == "all" or n.startswith("WS_"))
    }
    assert {n for n, _, text in sig if "noalias" in text} == want


@pytest.mark.parametrize("arch", _ARCHS)
def test_ptr_noalias_all_needs_atomic_dkv(arch):
    with pytest.raises(ValueError):
        _kernel(arch, dict(body_knobs={"ptr_noalias": "all"}))


def _read_multipliers(text):
    """Multiplier kernargs the kernel body reads (beyond the signature)."""
    return {
        m
        for m in ("ds_mult", "dk_mult", "dq_mult")
        if len(re.findall(rf"%{m}(?![\w.])", text)) > 1
    }


@pytest.mark.parametrize("arch", _ARCHS)
@pytest.mark.parametrize(
    "over,read",
    [
        ({}, {"ds_mult"}),
        (dict(body_knobs={"static_scale": "fold_ds"}), {"ds_mult"}),
        (dict(body_knobs={"static_scale": "convert"}), {"dk_mult"}),
        (dict(body_knobs={"static_scale": "dq_before_atomic"}), {"dk_mult", "dq_mult"}),
        (dict(body_knobs={"static_scale": "runtime"}), {"ds_mult", "dk_mult", "dq_mult"}),
        (dict(dkv_mode="atomic", body_knobs={"static_scale": "convert"}), set()),
        (dict(dkv_mode="atomic", body_knobs={"static_scale": "runtime"}), {"ds_mult", "dk_mult", "dq_mult"}),
    ],
)  # fmt: skip
def test_static_scale_reads_only_the_non_unit_multipliers(arch, over, read):
    text = _lower(_kernel(arch, over), arch, "llvm22")
    assert _read_multipliers(text) == read


@pytest.mark.parametrize("arch", _ARCHS)
def test_interleave_decode_only_with_the_switch(arch):
    plain = _lower(_kernel(arch, {}), arch, "llvm22")
    woven = _lower(
        _kernel(arch, dict(body_knobs={"lb_interleave": True})), arch, "llvm22"
    )
    assert plain != woven


def _walk(ops):
    for op in ops:
        yield op
        for r in op.regions:
            yield from _walk(r.ops)


def _ops(kernel):
    return list(_walk(kernel.body.ops))


def _count(kernel, name):
    return sum(1 for op in _ops(kernel) if op.name == name)


def _q_loops(kernel):
    """Innermost ``scf.for`` loops that hold a matrix op (the q loops)."""
    loops = [
        op
        for op in _ops(kernel)
        if op.name == "scf.for"
        and any(o.name == "tile.mma" for o in _walk(op.regions[0].ops))
    ]
    ids = {id(lp) for lp in loops}
    return [
        lp
        for lp in loops
        if not any(id(o) in ids for o in _walk(lp.regions[0].ops) if o is not lp)
    ]


@pytest.mark.parametrize("arch", _ARCHS)
def test_buffer_resources_only_under_buffer_oob(arch):
    assert _count(_kernel(arch, {}), "tile.buffer_rsrc") > 0
    assert _count(_kernel(arch, dict(oob="clamp")), "tile.buffer_rsrc") == 0


@pytest.mark.parametrize("arch", _ARCHS)
def test_edge_loops_dropped_when_no_tile_is_an_edge(arch):
    """Aligned unmasked: one q loop; causal: edge and interior loops."""

    def q_loops(over):
        return len(_q_loops(_kernel(arch, over)))

    assert q_loops(dict(causal=False)) == 1
    assert q_loops({}) == 3
    assert q_loops(dict(causal=False, edge_tiles=False)) == 1
    # a ragged unmasked kernel has tail tiles: edge loops come back
    assert q_loops(dict(causal=False, seqlen_kv=200, ragged=True)) == 3
    # a persistent kernel holds the q loops inside its work loop
    assert q_loops(dict(persistent=True)) == 3


@pytest.mark.parametrize("arch", _ARCHS)
def test_runtime_batch_vs_pinned(arch):
    text = _lower(_kernel(arch, {}), arch, "llvm22")
    pinned = _lower(_kernel(arch, dict(batch=2)), arch, "llvm22")
    assert text != pinned


@pytest.mark.parametrize("arch,cid,over", _ALL, ids=[f"{a}-{c}" for a, c, _ in _ALL])
def test_zero_scratch_no_spills_and_lds_within_the_plan(arch, cid, over):
    """The general family's resource gate on every golden dense instance:
    no scratch, no spills, arch VGPR / AGPR budgets, occupancy >= 1, q-loop
    AGPR copies within the ceiling of the body tile, LDS within the plan."""
    pytest.importorskip("rocke.runtime.comgr")
    from kernels.common.attention_bwd import attn_bwd_arch_facts, lds_plan
    from kernels.common.attention_dense_bwd_spec import validate_attn_dense_bwd_spec

    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        compile_or_skip,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    # torch's bundled (production) comgr, pinned by path; torch not imported
    pin_production_comgr()
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(
            f"resource gate is valid on {GATE_FLAVOR} with the production comgr "
            f"only (flavor {flavor}, comgr {comgr})"
        )
    spec = _spec(arch, over)
    body = validate_attn_dense_bwd_spec(spec, arch)
    art = compile_or_skip(
        _builder(arch)(spec, arch=arch),
        arch=arch,
        capture_ir_text=False,
        backend="python",
    )
    facts = attn_bwd_arch_facts(arch)
    res = evaluate_resource_gate(
        art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, spec=body
    )
    assert res.valid, (res.flavor, res.comgr)
    assert not res.failures, res.failures
    assert not res.warnings, (res.warnings, res.resources.sgpr)
    plan_bytes = lds_plan(body, arch).total_bytes
    assert res.resources.lds_bytes <= plan_bytes, (res.resources.lds_bytes, plan_bytes)


@pytest.mark.parametrize(
    "arch,cid,over", _DQ_ALL, ids=[f"{a}-dq-{c}" for a, c, _ in _DQ_ALL]
)
def test_dense_dq_resource_gate(arch, cid, over):
    """The split dQ kernel's resource gate on every golden dense dQ instance
    (the sweep's ``static_prune_dq`` rules): no scratch, no spills, the
    budgets, occupancy, LDS within the dQ plan, no SGPR-margin warning, and
    with VGPR-form MFMA no AGPR read in a key loop."""
    pytest.importorskip("rocke.runtime.comgr")
    from kernels.common.attention_bwd import attn_bwd_arch_facts
    from kernels.common.attention_bwd_dq import dq_lds_plan
    from kernels.common.attention_dense_bwd_dq import validate_attn_dense_bwd_dq_spec

    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        compile_or_skip,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
        pin_production_comgr,
    )

    pin_production_comgr()
    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the key-loop AGPR rule)")
    flavor, comgr = current_flavor(), comgr_version()
    if flavor != GATE_FLAVOR or not comgr_is_production(comgr):
        pytest.skip(
            f"resource gate is valid on {GATE_FLAVOR} with the production comgr "
            f"only (flavor {flavor}, comgr {comgr})"
        )
    spec = _dq_spec(over)
    dq = validate_attn_dense_bwd_dq_spec(spec, arch)
    art = compile_or_skip(
        _dq_kernel(arch, over), arch=arch, capture_ir_text=False, backend="python"
    )
    facts = attn_bwd_arch_facts(arch)
    tile = {}
    if dq.agpr_alloc == (0, 0):
        tile = dict(block_m=dq.block_m, block_n=dq.block_n, waves=dq.waves)
    res = evaluate_resource_gate(
        art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, **tile
    )
    assert res.valid, (res.flavor, res.comgr)
    assert not res.failures, res.failures
    assert not res.warnings, (res.warnings, res.resources.sgpr)
    if tile:
        assert not any(x["reads"] for x in res.loops or ()), res.loops
    plan_bytes = dq_lds_plan(dq, arch).total_bytes
    assert res.resources.lds_bytes <= plan_bytes, (res.resources.lds_bytes, plan_bytes)


def test_dense_bwd_never_imports_the_forward_dense_modules():
    lib = _TESTS.parent
    files = [
        lib / "kernels" / "common" / "attention_dense_bwd_spec.py",
        lib / "kernels" / "common" / "attention_dense_bwd_plan.py",
        lib / "kernels" / "common" / "_attention_dense_bwd_shell.py",
        lib / "kernels" / "common" / "attention_dense_bwd_dq.py",
        lib / "kernels" / "gfx942" / "attention_dense_bwd.py",
        lib / "kernels" / "gfx950" / "attention_dense_bwd.py",
    ]
    forward = {
        "kernels.gfx942.attention_dense",
        "kernels.gfx950.attention_dense",
        "kernels.common.attention_dense_spec",
    }
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
            for m in mods:
                assert m not in forward, f"{f.name} imports {m}"


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        for a in _ARCHS:
            test_dense_bwd_ir_matches_golden(a, _current_flavor())
