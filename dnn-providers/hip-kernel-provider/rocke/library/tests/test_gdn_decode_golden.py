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
_ARCH = "gfx950"

# Pin the library root ahead of everything on sys.path so that running this file
# directly does not let tests/dispatch/ shadow the real library/dispatch package.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)


def _cases():
    """case id -> zero-arg builder returning a KernelDef.

    Covers the default spec, the reference path, and every legal registered
    tile, so a change to any selectable configuration is visible.
    """
    from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode_all
    from dispatch.gdn.gfx950 import _TUNED_TILES_KDA
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, build_gdn_decode

    def build(**overrides):
        spec = dc.replace(GdnDecodeSpec(), **overrides)
        return lambda: build_gdn_decode(spec, arch=_ARCH)

    cases = {
        "default": build(),
        "simple": build(simple=True),
        "no_l2norm": build(use_qk_l2norm=False),
        # KDA gate kind. Pinned for the same reason the GDN cases are: the
        # per-channel gate is emitted code, and a refactor that changed it
        # without breaking it would pass every other test in the tree.
        "kda_default": build(gate_kind="kda"),
        "kda_simple": build(gate_kind="kda", simple=True),
        "kda_raw_gate": build(gate_kind="kda", fuse_gate=False),
    }
    request = GdnDecodeRequest(batch=16, arch=_ARCH)
    for result in dispatch_gdn_decode_all(request):
        cases[f"registered_{result.candidate.spec_id}"] = (
            lambda spec=result.spec: build_gdn_decode(spec, arch=_ARCH)
        )
    for _, tile, spec_id in _TUNED_TILES_KDA:
        cases[f"tuned_{spec_id}"] = build(
            gate_kind="kda",
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


def _run(flavor):
    """One flavor's golden sub-document, for ``check_golden``. A lowering error
    propagates with its traceback."""
    cases = {}
    for cid, build in _cases().items():
        sha, nbytes = _sha_for(build, flavor)
        cases[cid] = {"sha256": sha, "bytes": nbytes}
    return {"cases": cases}


def _build_doc():
    from rocke.core.ir_golden import GOLDEN_FLAVORS

    doc = {"schema": "gdn_decode_gfx950.ir_golden_sha256/v1", "flavors": {}}
    failures = []
    for flavor in GOLDEN_FLAVORS:
        cases = {}
        for cid, build in _cases().items():
            try:
                sha, nbytes = _sha_for(build, flavor)
            except Exception as exc:  # pragma: no cover - diagnostic only
                failures.append(f"{flavor}/{cid}: {exc}")
                continue
            cases[cid] = {"sha256": sha, "bytes": nbytes}
        doc["flavors"][flavor] = {"cases": cases}
    if failures:
        raise RuntimeError(
            "refusing to write golden fixture with lowering failures:\n  "
            + "\n  ".join(failures)
        )
    return doc


def test_gdn_decode_ir_matches_golden():
    import pytest
    from rocke.core.ir_golden import check_golden

    if not _GOLDEN.exists():
        pytest.skip("gdn_decode golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    # An entry without a hash would otherwise surface as a sha mismatch; name it.
    unrecorded = [
        f"{flavor}/{cid}: no sha256 recorded ({entry})"
        for flavor, sub in golden.get("flavors", {}).items()
        for cid, entry in sub.get("cases", {}).items()
        if not entry.get("sha256")
    ]
    assert not unrecorded, "\n  ".join(unrecorded)
    # Every flavor in LLVM_FLAVORS, from any host, so a datalayout or intrinsic
    # change for a flavor this host does not run still fails here.
    drift = check_golden(_GOLDEN, _run)
    assert not drift, (
        "gdn_decode IR drift vs golden (re-record with --write if intended):\n  "
        + "\n  ".join(drift)
    )


def test_every_shipped_configuration_is_recorded():
    """A new tuned tile must arrive with a golden entry, not silently uncovered."""
    import pytest
    from rocke.core.ir_golden import GOLDEN_FLAVORS

    if not _GOLDEN.exists():
        pytest.skip("gdn_decode golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavors = golden.get("flavors", {})
    # A flavor with no sub-document leaves every configuration unrecorded.
    missing = sorted(
        f"{flavor}/{cid}"
        for flavor in GOLDEN_FLAVORS
        for cid in _cases()
        if not flavors.get(flavor, {}).get("cases", {}).get(cid, {}).get("sha256")
    )
    assert not missing, f"configurations without a SHA-256: {missing}"


def test_golden_ir_check_rejects_entry_without_sha256(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(sys.modules[__name__], "_cases", lambda: {"default": object()})
    fixture = tmp_path / "gdn_decode_gfx950_ir_sha256.json"
    fixture.write_text(
        json.dumps(
            {
                "flavors": {
                    _current_flavor(): {
                        "cases": {
                            cid: {"error": "synthetic lowering failure"}
                            for cid in _cases()
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setattr(sys.modules[__name__], "_GOLDEN", fixture)

    with pytest.raises(AssertionError, match="no sha256 recorded"):
        test_gdn_decode_ir_matches_golden()


def test_config_coverage_rejects_entry_without_sha256(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(sys.modules[__name__], "_cases", lambda: {"default": object()})
    fixture = tmp_path / "gdn_decode_gfx950_ir_sha256.json"
    fixture.write_text(
        json.dumps(
            {
                "flavors": {
                    _current_flavor(): {
                        "cases": {
                            cid: {"error": "synthetic lowering failure"}
                            for cid in _cases()
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setattr(sys.modules[__name__], "_GOLDEN", fixture)

    with pytest.raises(AssertionError, match="without a SHA-256"):
        test_every_shipped_configuration_is_recorded()


def test_build_doc_refuses_lowering_failure(monkeypatch):
    import pytest

    monkeypatch.setattr(sys.modules[__name__], "_cases", lambda: {"default": object()})

    def fail_lowering(*_):
        raise RuntimeError("synthetic lowering failure")

    monkeypatch.setattr(sys.modules[__name__], "_sha_for", fail_lowering)

    with pytest.raises(RuntimeError, match="refusing to write"):
        _build_doc()


def test_gate_kind_actually_moves_the_ir():
    """A mutation check: the golden gate must be able to detect this change.

    "Golden untouched" only means something if the golden *could* have moved.
    Flipping gate_kind changes emitted code, so it must change both the IR hash
    and the kernel name -- otherwise the KDA cases above are pinning nothing and
    two different kernels would share one compile-cache entry.
    """
    import dataclasses as _dc

    from kernels.gfx950.gdn_decode import GdnDecodeSpec, build_gdn_decode

    flavor = _current_flavor()
    gdn = GdnDecodeSpec()
    kda = _dc.replace(gdn, gate_kind="kda")

    gdn_sha, _ = _sha_for(lambda: build_gdn_decode(gdn, arch=_ARCH), flavor)
    kda_sha, _ = _sha_for(lambda: build_gdn_decode(kda, arch=_ARCH), flavor)

    assert gdn_sha != kda_sha, "gate_kind did not change the emitted IR"
    assert gdn.kernel_name() != kda.kernel_name(), "gate_kind did not change the name"


def test_gdn_cases_carry_no_kda_marker():
    """Every pre-existing GDN entry must stay a GDN entry.

    Guards the additive claim from the fixture side: if a GDN case id ever
    starts resolving to a KDA spec, the "GDN goldens unchanged" evidence is
    quietly measuring the wrong kernel.

    KDA appears in an id two ways -- as a prefix for the hand-written cases
    (``kda_default``) and as an infix for the tuned ones (``tuned_kda_w128``,
    which inherits its gate kind from the spec id in the KDA table) -- so the
    split is on containment, not prefix.
    """
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    assert GdnDecodeSpec().gate_kind == "gdn"
    ids = list(_cases())
    gdn_ids = [cid for cid in ids if "kda" not in cid]
    kda_ids = [cid for cid in ids if "kda" in cid]

    # The original GDN set: default, simple, no_l2norm + one per GDN tuned tile.
    assert len(gdn_ids) >= 7, f"expected the original GDN case set, got {gdn_ids}"
    assert kda_ids, "the KDA gate kind is unpinned"
    assert not set(gdn_ids) & set(kda_ids)


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_gdn_decode_ir_matches_golden()
        test_every_shipped_configuration_is_recorded()
