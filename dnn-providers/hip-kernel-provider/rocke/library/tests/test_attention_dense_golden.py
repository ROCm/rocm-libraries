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

    def mk(**over):
        d = dict(base)
        d.update(over)
        return lambda: build_attention_dense(AttentionDenseSpec(**d))

    return {
        # --- default (one CTA per q-block/head) grid ---
        "attention_dense/default_causal_sq512": mk(),
        "attention_dense/swa_w128_sq512": mk(sliding_window=128),
        "attention_dense/varlen_sq512": mk(varlen=True),
        "attention_dense/lazy_off_sq512": mk(lazy_rescale=False),
        "attention_dense/fp16_h64_sq512": mk(dtype="fp16", head_size=64),
        "attention_dense/bn128_sq512": mk(block_n=128),
        "attention_dense/noncausal_sq512": mk(causal=False),
        # --- persistent (grid-stride) grid + decode variants ---
        "attention_dense/persistent_causal_sq512": mk(
            persistent=True, num_persistent=256
        ),
        "attention_dense/persist_qbmaj_sq512": mk(
            persistent=True, num_persistent=256, persist_decode="qb_major"
        ),
        "attention_dense/persist_hkvmaj_sq512": mk(
            persistent=True, num_persistent=256, persist_decode="hkv_major"
        ),
        "attention_dense/persist_intl_sq512": mk(
            persistent=True, num_persistent=256, interleave=True
        ),
        "attention_dense/persist_lazy_off_sq512": mk(
            persistent=True, num_persistent=256, lazy_rescale=False
        ),
        "attention_dense/persist_swa_w128_sq512": mk(
            persistent=True, num_persistent=256, sliding_window=128
        ),
        # D=64 packed-row DMA loader (2 rows/instr, unpadded LDS) on the
        # persistent builder -- locks the head_size=64 fix (fp16_h64 above only
        # exercises the default builder).
        "attention_dense/persist_h64_sq512": mk(
            persistent=True, num_persistent=256, head_size=64
        ),
        # ragged (non-256 seqlen) in-kernel path: on-chip boundary padding.
        # causal (no key mask), non-causal (ktok<seqlen_kv key mask), D=64, and
        # the persistent variant -- lock all four ragged codegen shapes.
        "attention_dense/ragged_causal_sq500": mk(
            seqlen_q=500, seqlen_kv=500, ragged=True
        ),
        "attention_dense/ragged_full_sq500": mk(
            seqlen_q=500, seqlen_kv=500, ragged=True, causal=False
        ),
        "attention_dense/ragged_h64_sq500": mk(
            seqlen_q=500, seqlen_kv=500, ragged=True, head_size=64
        ),
        "attention_dense/persist_ragged_sq500": mk(
            seqlen_q=500,
            seqlen_kv=500,
            ragged=True,
            persistent=True,
            num_persistent=256,
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


def test_attention_dense_cpp_python_byte_identity():
    """Stream-4 parity gate: the C++ engine (rocke_engine) lowers every
    dense-prefill variant to byte-identical LLVM IR vs the Python lowerer.

    Both sides go through ``_lower_llvm_via_backend`` so they resolve the same
    llvm flavor; ``ROCKE_CPP_STRICT=1`` disables the silent python fallback so a
    missing/stale C++ engine surfaces as a skip (not a false pass). Requires the
    C++ ``exp2_fast`` op (added for the dense softmax hot path).
    """
    import os

    import pytest

    try:
        from rocke.helpers.compile import _lower_llvm_via_backend
    except Exception as e:  # pragma: no cover
        pytest.skip(f"backend lowering unavailable: {e}")

    prev = os.environ.get("ROCKE_CPP_STRICT")
    os.environ["ROCKE_CPP_STRICT"] = "1"
    mism = []
    try:
        for cid, build in _cases().items():
            k = build()
            py = _lower_llvm_via_backend(k, arch="gfx950", backend="python", spec=None)
            try:
                cpp = _lower_llvm_via_backend(
                    k, arch="gfx950", backend="cpp", spec=None
                )
            except Exception as e:  # C++ engine not built / opcode gap
                pytest.skip(f"C++ engine unavailable ({cid}): {str(e)[:140]}")
            if py != cpp:
                mism.append(cid)
    finally:
        if prev is None:
            os.environ.pop("ROCKE_CPP_STRICT", None)
        else:
            os.environ["ROCKE_CPP_STRICT"] = prev
    assert not mism, "attention_dense cpp/python IR byte-mismatch:\n  " + "\n  ".join(
        mism
    )


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_attention_dense_ir_matches_golden()
        print("PASS")
