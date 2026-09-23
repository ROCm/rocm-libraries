# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR byte-stability test for the gfx942 MLA prefill kernels.

Hashes the Python-lowered LLVM IR (SHA256) of representative ``mla_prefill``
specs and compares against a checked-in per-flavor golden fixture, catching
unintended codegen drift. Pure text lowering -- no GPU / no comgr required.

Covers both entry points in the family and both shipped head counts:

  * ``fwd_*``      -- the chunked forward kernel (online softmax, LSE, mask).
  * ``probe_*``    -- the score-tile probe, which the on-device verify driver
    compares against the numpy oracle. It is a shipped entry point, so its IR
    is pinned too; drift there would silently weaken the parity harness.
  * ``dispatch_*`` -- the forward kernel built through the spec the MLA
    dispatcher actually selects, rather than one hard-coded here. This is what
    guards the *shipped* configuration: if a future change moves a lever in
    ``dispatch.mla.gfx942``, this case moves with it and re-blessing stays a
    one-command operation.

There is NO cpp/python byte-identity companion gate for this family: MLA has no
C++ engine mirror (settled when the family was scoped), so the dual-engine parity
gate does not apply. The Python lowering IS the ground truth here.

Re-bless, from ``rocke/library``:

    PYTHONPATH=../platform/python:. python tests/test_mla_prefill_gfx942_golden.py --write

Re-blessing is a deliberate act, never a way to turn red green: a hash that moved
without a matching, explained emission change is a defect, not a stale fixture.
"""

import hashlib
import json
import sys
from pathlib import Path

_GOLDEN = (
    Path(__file__).resolve().parent / "golden" / "mla_prefill_gfx942_ir_sha256.json"
)
_FLAVORS = ("llvm20", "llvm22")
_ARCH = "gfx942"
_SCHEMA = "mla_prefill_gfx942.ir_golden_sha256/v1"

# Pin the ``library/`` root ahead of everything on sys.path. When this file is
# run directly Python puts ``tests/`` on sys.path[0], where ``tests/dispatch/``
# would shadow the real ``library/dispatch`` package the dispatch-built case
# imports. Harmless under pytest.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)


def _cases():
    """``cid -> zero-arg builder``.

    Only ``num_heads`` varies. The codegen levers (``block_q``, ``block_k``,
    ``r_kv_tile``, ``num_warps``) are deliberately left at their defaults: this
    fixture pins the *shipped* IR, and sweeping levers here would bless
    configurations nothing has measured or run on device.
    """
    from kernels.mla.mla_prefill_gfx942 import (
        MlaPrefillSpec,
        build_mla_prefill_fwd,
        build_mla_prefill_score_probe,
    )

    def fwd(num_heads):
        return lambda: build_mla_prefill_fwd(
            MlaPrefillSpec(num_heads=num_heads), arch=_ARCH
        )

    def probe(num_heads):
        return lambda: build_mla_prefill_score_probe(
            MlaPrefillSpec(num_heads=num_heads), arch=_ARCH
        )

    def dispatched(num_heads):
        def build():
            from dispatch.mla import MLARequest, dispatch_mla

            result = dispatch_mla(
                MLARequest(
                    num_heads=num_heads,
                    total_q=64,
                    num_seqs=2,
                    max_seqlen_k=96,
                    arch=_ARCH,
                )
            )
            return result.candidate.built(result.spec, _ARCH)

        return build

    return {
        # 128 heads is the DeepSeek-V3 shape; 64 is the smaller supported count.
        "mla_prefill_gfx942/fwd_h128": fwd(128),
        "mla_prefill_gfx942/fwd_h64": fwd(64),
        "mla_prefill_gfx942/probe_h128": probe(128),
        "mla_prefill_gfx942/probe_h64": probe(64),
        "mla_prefill_gfx942/dispatch_fwd_h128": dispatched(128),
        "mla_prefill_gfx942/dispatch_fwd_h64": dispatched(64),
    }


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc():
    doc = {"schema": _SCHEMA, "flavors": {}}
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


def test_mla_prefill_gfx942_golden_covers_every_case():
    """The fixture and the case table name exactly the same cases.

    Without this, a case added to ``_cases()`` but never blessed would be
    silently skipped rather than failing, and a typo'd cid would pin nothing.
    """
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("MLA golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    assert golden.get("schema") == _SCHEMA
    want = set(_cases())
    for flavor in _FLAVORS:
        recorded = set(golden["flavors"][flavor]["cases"])
        assert recorded == want, (
            f"{flavor}: golden case set differs from the case table; "
            f"missing={sorted(want - recorded)} stale={sorted(recorded - want)}"
        )


def test_mla_prefill_gfx942_golden_records_no_build_errors():
    """A blessed ``{"error": ...}`` entry is a failed bless, not a golden."""
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("MLA golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    broken = [
        f"{flavor}/{cid}: {entry['error']}"
        for flavor in _FLAVORS
        for cid, entry in golden["flavors"][flavor]["cases"].items()
        if "error" in entry
    ]
    assert not broken, "golden holds build errors:\n  " + "\n  ".join(broken)


def test_mla_prefill_gfx942_ir_matches_golden():
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("MLA golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    gflav = golden.get("flavors", {}).get(flavor)
    if not gflav:
        pytest.skip(f"no MLA golden recorded for llvm flavor {flavor!r}")
    drift = []
    for cid, build in _cases().items():
        want = gflav["cases"].get(cid, {}).get("sha256")
        # A missing case is reported by the coverage test above; do not also
        # fail here, so one omission produces one diagnosis.
        if want is None:
            continue
        got, _ = _sha_for(build, flavor)
        if got != want:
            drift.append(f"{cid}: {want} -> {got}")
    assert not drift, "gfx942 MLA prefill IR drift vs golden:\n  " + "\n  ".join(drift)


def test_dispatch_case_tracks_the_shipped_spec():
    """The ``dispatch_*`` case must build the same spec the dispatcher selects.

    Guards the case's whole purpose: if it silently diverged from dispatch, it
    would pin a configuration nothing ships.
    """
    from dispatch.mla import MLARequest, dispatch_mla
    from kernels.mla.mla_prefill_gfx942 import MlaPrefillSpec

    for num_heads in (64, 128):
        result = dispatch_mla(
            MLARequest(
                num_heads=num_heads,
                total_q=64,
                num_seqs=2,
                max_seqlen_k=96,
                arch=_ARCH,
            )
        )
        assert result.spec == MlaPrefillSpec(num_heads=num_heads)


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_mla_prefill_gfx942_golden_covers_every_case()
        test_mla_prefill_gfx942_ir_matches_golden()
