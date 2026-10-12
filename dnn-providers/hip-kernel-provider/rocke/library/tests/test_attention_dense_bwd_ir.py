# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense SDPA backward IR (CPU: pure text lowering, no device, no comgr).

* Golden LLVM-IR hashes of representative dense backward instances on gfx942
  and gfx950 (``tests/golden/attention_dense_bwd_ir_sha256.json``, one entry
  per LLVM flavor): every mask form, both layouts and dtypes, both head
  sizes, ragged, atomic dK / dV, both persistent decodes, the chiplet remap,
  the clamp addressing, every epilogue, the token-major workspace, edge tiles
  off, a pinned batch, the interleave decode, every multiplier applied at run
  time and ``noalias`` workspace pointers. Re-bless with
  ``python tests/test_attention_dense_bwd_ir.py --write``.
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
)


def _spec(arch, over):
    d = dict(_BASE)
    d.update(over)
    return _spec_cls(arch)(**d)


def _kernel(arch, over):
    return _builder(arch)(_spec(arch, over), arch=arch)


def _lower(kernel, arch, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


def _sha(arch, over, flavor):
    data = _lower(_kernel(arch, over), arch, flavor).encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _build_doc():
    doc = {"schema": "attention_dense_bwd.ir_golden_sha256/v1", "flavors": {}}
    for flavor in _FLAVORS:
        cases = {}
        for arch in _ARCHS:
            for cid, over in _CASES:
                sha, nbytes = _sha(arch, over, flavor)
                cases[f"{arch}/{cid}"] = {"sha256": sha, "bytes": nbytes}
        doc["flavors"][flavor] = {"cases": cases}
    return doc


@pytest.mark.parametrize("arch", _ARCHS)
def test_dense_bwd_ir_matches_golden(arch):
    if not _GOLDEN.exists():
        pytest.fail("dense backward golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    gflav = golden.get("flavors", {}).get(flavor)
    if not gflav:
        pytest.skip(f"no dense backward golden recorded for llvm flavor {flavor!r}")
    drift = []
    for cid, over in _CASES:
        key = f"{arch}/{cid}"
        want = gflav["cases"].get(key, {}).get("sha256")
        if want is None:
            drift.append(f"{key}: missing from the golden")
            continue
        got, _ = _sha(arch, over, flavor)
        if got != want:
            drift.append(f"{key}: {want} -> {got}")
    assert not drift, "dense backward IR drift vs golden:\n  " + "\n  ".join(drift)


def test_golden_covers_every_case_and_flavor():
    golden = json.loads(_GOLDEN.read_text())
    want = {f"{a}/{cid}" for a in _ARCHS for cid, _ in _CASES}
    for flavor in _FLAVORS:
        got = set(golden["flavors"][flavor]["cases"])
        assert got == want, flavor


_INPUTS = {"Q", "K", "V", "O", "dO", "LSE", "WS_LSE2", "WS_DSUM"}


def _signature(ir):
    head = re.search(r"define amdgpu_kernel void @\S+\((.*?)\) #0", ir).group(1)
    out = []
    for p in head.split(", "):
        name = p.split("%")[-1]
        if p.startswith("ptr"):
            kind = "Q"
        elif p.startswith("float"):
            kind = "f"
        else:
            kind = "q" if p.startswith("i64") else "i"
        out.append((name, kind, p))
    return out


@pytest.mark.parametrize("arch", _ARCHS)
@pytest.mark.parametrize(
    "over", [{}, dict(dkv_mode="atomic"), dict(persistent=True), dict(xcd_remap=True)]
)
def test_signature_is_the_abi(arch, over):
    from kernels.common.attention_dense_bwd_spec import attn_dense_bwd_params

    sig = _signature(_lower(_kernel(arch, over), arch, "llvm22"))
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
    sig = _signature(_lower(_kernel(arch, over), arch, "llvm22"))
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


@pytest.mark.parametrize("cid,over", _CASES, ids=[c for c, _ in _CASES])
@pytest.mark.parametrize("arch", _ARCHS)
def test_zero_scratch_no_spills_and_lds_within_the_plan(arch, cid, over):
    """The general family's resource gate on every golden dense instance:
    no scratch, no spills, arch VGPR / AGPR budgets, occupancy >= 1, q-loop
    AGPR copies within the ceiling of the body tile, LDS within the plan."""
    pytest.importorskip("rocke.runtime.comgr")
    # torch first: rocKE then compiles with torch's bundled (production) comgr
    pytest.importorskip("torch")
    from rocke.helpers.compile import compile_kernel

    from kernels.common.attention_bwd import attn_bwd_arch_facts, lds_plan
    from kernels.common.attention_dense_bwd_spec import validate_attn_dense_bwd_spec

    from ._attention_bwd_resources import (
        GATE_FLAVOR,
        comgr_is_production,
        comgr_version,
        current_flavor,
        evaluate_resource_gate,
        find_objdump,
    )

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
    try:
        art = compile_kernel(
            _builder(arch)(spec, arch=arch),
            arch=arch,
            capture_ir_text=False,
            backend="python",
        )
    except Exception as exc:  # noqa: BLE001 - comgr cannot target this arch here
        pytest.skip(f"comgr cannot compile for {arch}: {exc}")
    facts = attn_bwd_arch_facts(arch)
    res = evaluate_resource_gate(
        art.hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, spec=body
    )
    assert res.valid, (res.flavor, res.comgr)
    assert not res.failures, res.failures
    assert not res.warnings, (res.warnings, res.resources.sgpr)
    plan_bytes = lds_plan(body, arch).total_bytes
    assert res.resources.lds_bytes <= plan_bytes, (res.resources.lds_bytes, plan_bytes)


def test_dense_bwd_never_imports_the_forward_dense_modules():
    lib = _TESTS.parent
    files = [
        lib / "kernels" / "common" / "attention_dense_bwd_spec.py",
        lib / "kernels" / "common" / "attention_dense_bwd_plan.py",
        lib / "kernels" / "common" / "_attention_dense_bwd_shell.py",
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
            test_dense_bwd_ir_matches_golden(a)
