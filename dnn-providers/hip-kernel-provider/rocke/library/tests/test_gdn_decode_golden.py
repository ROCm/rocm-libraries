# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the GDN decode kernel.

Hashes the lowered IR for a fixed set of specs and compares against a recorded
fixture. This catches a class of change nothing else does: a refactor of the
emitter that alters the generated code *without* making it wrong. The numeric
test would still pass, because the kernel is still correct -- just different.

The two checks are complementary, not redundant:

* the numeric test catches a kernel that is **wrong**, but not one that merely
  **changed**;
* this test catches a kernel that **changed**, but says nothing about whether
  either version is correct.

A failure here is not automatically a bug. It is a claim that the emitted code
moved, and it demands an answer: intended, or not? When intended, re-record in
the same change so the diff states it out loud::

    python3 library/tests/test_gdn_decode_golden.py --write

Lowering needs no GPU and no comgr, so this runs anywhere.
"""

from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import sys
from pathlib import Path

_GOLDEN = (
    Path(__file__).resolve().parent / "golden" / "gdn_decode_gfx950_ir_sha256.json"
)
_FLAVORS = ("llvm20", "llvm22", "llvm23")
_ARCH = "gfx950"

# Pin the library root ahead of everything on sys.path so that running this file
# directly does not let tests/dispatch/ shadow the real library/dispatch package.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)


def _cases():
    """case id -> zero-arg builder returning a KernelDef.

    Covers the default spec, the reference path, and every tile the dispatcher
    can select, so a change to any shipped configuration is visible.
    """
    from dispatch.gdn.gfx950 import _TUNED_TILES
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, build_gdn_decode

    def build(**overrides):
        spec = dc.replace(GdnDecodeSpec(), **overrides)
        return lambda: build_gdn_decode(spec, arch=_ARCH)

    cases = {
        "default": build(),
        "simple": build(simple=True),
        "no_l2norm": build(use_qk_l2norm=False),
    }
    for _, tile, spec_id in _TUNED_TILES:
        cases[f"tuned_{spec_id}"] = build(
            num_warps=tile[0],
            warp_threads_k=tile[1],
            blocks_per_v_dim=tile[2],
        )
    return cases


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc():
    doc = {"schema": "gdn_decode_gfx950.ir_golden_sha256/v1", "flavors": {}}
    for flavor in _FLAVORS:
        cases = {}
        for cid, build in _cases().items():
            try:
                sha, nbytes = _sha_for(build, flavor)
                cases[cid] = {"sha256": sha, "bytes": nbytes}
            except Exception as exc:  # pragma: no cover - diagnostic only
                cases[cid] = {"error": str(exc)[:160]}
        doc["flavors"][flavor] = {"cases": cases}
    return doc


def test_gdn_decode_ir_matches_golden():
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("gdn_decode golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    recorded = golden.get("flavors", {}).get(flavor)
    if not recorded:
        pytest.skip(f"no gdn_decode golden recorded for llvm flavor {flavor!r}")
    drift = []
    for cid, build in _cases().items():
        want = recorded["cases"].get(cid, {}).get("sha256")
        if want is None:
            continue
        got, _ = _sha_for(build, flavor)
        if got != want:
            drift.append(f"{cid}: {want} -> {got}")
    assert not drift, (
        "gdn_decode IR drift vs golden (re-record with --write if intended):\n  "
        + "\n  ".join(drift)
    )


def test_every_shipped_configuration_is_recorded():
    """A new tuned tile must arrive with a golden entry, not silently uncovered."""
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("gdn_decode golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    recorded = golden.get("flavors", {}).get(flavor)
    if not recorded:
        pytest.skip(f"no gdn_decode golden recorded for llvm flavor {flavor!r}")
    missing = sorted(set(_cases()) - set(recorded["cases"]))
    assert not missing, f"configurations with no golden entry: {missing}"


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_gdn_decode_ir_matches_golden()
        test_every_shipped_configuration_is_recorded()
