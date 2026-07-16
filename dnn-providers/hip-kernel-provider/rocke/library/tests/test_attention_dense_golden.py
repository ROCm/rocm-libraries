# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR byte-stability test for the gfx950 dense flash-attn kernel.

Hashes the Python-lowered LLVM IR (SHA256) of representative ``attention_dense``
specs and compares against a checked-in per-flavor golden fixture, catching any
unintended codegen drift. Pure text lowering — no GPU / no comgr required.

NOT WIRED INTO CI (by request): this file lives under ``library/tests/`` (which
``platform/tests/run_all.py``'s CI gate does NOT collect — it only pytests
``platform/tests/``), registers NO byte-identity ``*_emit`` parity pair, adds NO
case to ``rocke_ir_parity_harness.cases()`` / the platform golden, and adds NO
``add_test(...)`` CMake entry. Run it manually:

    cd rocke/library
    PYTHONPATH=../platform/python:. python -m pytest tests/test_attention_dense_golden.py

Re-bless after an intended codegen change:

    cd rocke/library
    PYTHONPATH=../platform/python:. python tests/test_attention_dense_golden.py --write
"""
import hashlib
import json
import sys
from pathlib import Path

_GOLDEN = Path(__file__).resolve().parent / "golden" / "attention_dense_ir_sha256.json"
_FLAVORS = ("llvm20", "llvm22")


def _cases():
    """cid -> zero-arg builder returning a KernelDef. Small Sq keeps the IR compact
    while still exercising the full pipeline (both grid variants)."""
    from kernels.gfx950.attention_dense import (
        AttentionDenseSpec,
        build_attention_dense,
    )

    base = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=128,
        num_kv_heads=8,
        head_size=128,
        causal=True,
        dtype="bf16",
    )
    return {
        "attention_dense/default_causal_sq512": lambda: build_attention_dense(
            AttentionDenseSpec(**base)
        ),
        "attention_dense/persistent_causal_sq512": lambda: build_attention_dense(
            AttentionDenseSpec(**base, persistent=True, num_persistent=256)
        ),
    }


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch="gfx950", llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc():
    doc = {"schema": "attention_dense.ir_golden_sha256/v1", "flavors": {}}
    for flavor in _FLAVORS:
        cases = {}
        for cid, build in _cases().items():
            try:
                sha, nbytes = _sha_for(build, flavor)
                cases[cid] = {"sha256": sha, "bytes": nbytes}
            except Exception as e:  # pragma: no cover - diagnostic
                cases[cid] = {"error": str(e)[:160]}
        doc["flavors"][flavor] = {"cases": cases}
    return doc


def test_attention_dense_ir_matches_golden():
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    gflav = golden.get("flavors", {}).get(flavor)
    if not gflav:
        pytest.skip(f"no golden recorded for llvm flavor {flavor!r}")
    drift = []
    for cid, build in _cases().items():
        want = gflav["cases"].get(cid, {}).get("sha256")
        if want is None:
            continue
        got, _ = _sha_for(build, flavor)
        if got != want:
            drift.append(f"{cid}: {want} -> {got}")
    assert not drift, "attention_dense IR drift vs golden:\n  " + "\n  ".join(drift)


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_attention_dense_ir_matches_golden()
        print("PASS")
