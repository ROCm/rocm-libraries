# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR byte-stability test for the MLA decode kernels.

Hashes the Python-lowered LLVM IR (SHA256) of the decode latent kernel (both
arches, single pass and split-KV) and of the split-KV combine, and compares
against a checked-in per-flavor fixture, catching unintended codegen drift.
Pure text lowering -- no GPU / no comgr required.

The ``dispatch_*`` cases build through the spec the MLA dispatcher selects, so
they pin the *shipped* configuration and move with it.

MLA has no C++ *instance* mirror (``platform/cpp/instances``), so the kernel
IR is built by the Python builders only. Its *lowering* is still dual-engine:
``test_mla_decode_cpp_python_byte_identity`` lowers every case through both
engines and requires identical bytes (skipped where the C++ engine is not
built).

Re-bless, from ``rocke/library``:

    PYTHONPATH=../platform/python:. python tests/test_mla_decode_golden.py --write

Re-blessing is a deliberate act, never a way to turn red green: a hash that moved
without a matching, explained emission change is a defect, not a stale fixture.
"""

import hashlib
import json
import sys
from pathlib import Path

_GOLDEN = Path(__file__).resolve().parent / "golden" / "mla_decode_ir_sha256.json"
_FLAVORS = ("llvm20", "llvm22", "llvm23")
_ARCHES = ("gfx942", "gfx950")
_SCHEMA = "mla_decode.ir_golden_sha256/v1"

# Pin the ``library/`` root ahead of everything on sys.path: run directly,
# ``tests/dispatch/`` would shadow the real ``library/dispatch`` package.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)


def _cases():
    """``cid -> (arch, zero-arg builder)``.

    Only the head count and single-pass vs split vary: those are the axes
    that change the IR. The split count itself is a kernel argument.
    """
    from kernels.mla.mla_decode import MlaDecodeSpec, build_mla_decode
    from kernels.mla.mla_decode_combine import (
        MlaDecodeCombineSpec,
        build_mla_decode_combine,
    )
    from kernels.mla.mla_decode_gfx950 import (
        MlaDecodeGfx950Spec,
        build_mla_decode_gfx950,
    )

    # Each arch through its own builder, the one a descriptor would name.
    arch_build = {
        "gfx942": (MlaDecodeSpec, build_mla_decode),
        "gfx950": (MlaDecodeGfx950Spec, build_mla_decode_gfx950),
    }

    def latent(arch, num_heads, num_seg):
        spec_cls, build = arch_build[arch]
        return arch, lambda: build(
            spec_cls(num_heads=num_heads, num_seg=num_seg), arch=arch
        )

    def combine(arch):
        return arch, lambda: build_mla_decode_combine(
            MlaDecodeCombineSpec(num_heads=128), arch=arch
        )

    def dispatched(arch, num_heads, max_k):
        def build():
            from dispatch.mla import MLARequest, dispatch_mla

            result = dispatch_mla(
                MLARequest(
                    num_heads=num_heads,
                    total_q=4,
                    num_seqs=4,
                    max_seqlen_q=1,
                    max_seqlen_k=max_k,
                    arch=arch,
                    # Pinned so the split choice does not come from the host.
                    num_cus=256,
                )
            )
            return result.candidate.built(result.spec, arch)

        return arch, build

    cases = {}
    for arch in _ARCHES:
        for heads in (64, 128):
            cases[f"mla_decode_{arch}/latent_h{heads}"] = latent(arch, heads, 1)
            cases[f"mla_decode_{arch}/latent_split_h{heads}"] = latent(arch, heads, 4)
        cases[f"mla_decode_{arch}/combine_h128"] = combine(arch)
        cases[f"mla_decode_{arch}/dispatch_h128_short"] = dispatched(arch, 128, 256)
        cases[f"mla_decode_{arch}/dispatch_h128_long"] = dispatched(arch, 128, 8192)
    return cases


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(arch, build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=arch, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc():
    doc = {"schema": _SCHEMA, "flavors": {}}
    for flavor in _FLAVORS:
        cases = {}
        for cid, (arch, build) in _cases().items():
            try:
                sha, nbytes = _sha_for(arch, build, flavor)
                cases[cid] = {"sha256": sha, "bytes": nbytes}
            except Exception as e:  # pragma: no cover - diagnostic
                cases[cid] = {"error": str(e)[:160]}
        doc["flavors"][flavor] = {"cases": cases}
    return doc


def _golden():
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("MLA decode golden fixture missing; generate with --write")
    return json.loads(_GOLDEN.read_text())


def test_mla_decode_golden_covers_every_case():
    """The fixture and the case table name exactly the same cases."""
    golden = _golden()
    assert golden.get("schema") == _SCHEMA
    want = set(_cases())
    for flavor in _FLAVORS:
        recorded = set(golden["flavors"][flavor]["cases"])
        assert recorded == want, (
            f"{flavor}: golden case set differs from the case table; "
            f"missing={sorted(want - recorded)} stale={sorted(recorded - want)}"
        )


def test_mla_decode_golden_records_no_build_errors():
    """A blessed ``{"error": ...}`` entry is a failed bless, not a golden."""
    golden = _golden()
    broken = [
        f"{flavor}/{cid}: {entry['error']}"
        for flavor in _FLAVORS
        for cid, entry in golden["flavors"][flavor]["cases"].items()
        if "error" in entry
    ]
    assert not broken, "golden holds build errors:\n  " + "\n  ".join(broken)


def test_mla_decode_ir_matches_golden():
    import pytest

    golden = _golden()
    flavor = _current_flavor()
    gflav = golden.get("flavors", {}).get(flavor)
    if not gflav:
        pytest.skip(f"no MLA decode golden recorded for llvm flavor {flavor!r}")
    drift = []
    for cid, (arch, build) in _cases().items():
        want = gflav["cases"].get(cid, {}).get("sha256")
        if want is None:
            continue  # reported once, by the coverage test
        got, _ = _sha_for(arch, build, flavor)
        if got != want:
            drift.append(f"{cid}: {want} -> {got}")
    assert not drift, "MLA decode IR drift vs golden:\n  " + "\n  ".join(drift)


def test_mla_decode_cpp_python_byte_identity(monkeypatch):
    """Every case lowers to identical bytes through the C++ engine."""
    import pytest
    from rocke.core.backend import BackendCoverageGap, BackendError
    from rocke.helpers.compile import _lower_llvm_via_backend

    monkeypatch.setenv("ROCKE_CPP_STRICT", "1")
    mismatched, gaps = [], []
    for cid, (arch, build) in _cases().items():
        kernel = build()
        py = _lower_llvm_via_backend(kernel, arch=arch, backend="python", spec=None)
        try:
            cpp = _lower_llvm_via_backend(kernel, arch=arch, backend="cpp", spec=None)
        except BackendCoverageGap:
            gaps.append(cid)
            continue
        except BackendError as e:
            pytest.skip(f"C++ engine not importable: {str(e)[:200]}")
        if py != cpp:
            mismatched.append(cid)
    assert not mismatched, "MLA decode cpp/python IR mismatch:\n  " + "\n  ".join(
        mismatched
    )
    assert not gaps, "C++ engine cannot lower:\n  " + "\n  ".join(gaps)


def test_dispatch_cases_track_the_shipped_spec():
    """The ``dispatch_*`` cases build what dispatch selects: one single-pass
    and one split-KV launch per arch."""
    from dispatch.mla import MLARequest, dispatch_mla

    for arch in _ARCHES:
        for max_k, split in ((256, False), (8192, True)):
            spec = dispatch_mla(
                MLARequest(
                    num_heads=128,
                    total_q=4,
                    num_seqs=4,
                    max_seqlen_q=1,
                    max_seqlen_k=max_k,
                    arch=arch,
                    num_cus=256,
                )
            ).spec
            assert spec.split_kv is split
            assert spec.ARCH == arch


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_mla_decode_golden_covers_every_case()
        test_mla_decode_ir_matches_golden()
