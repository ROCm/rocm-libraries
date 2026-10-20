# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR byte-stability test for the gfx942 MLA prefill kernels.

Hashes the Python-lowered LLVM IR (SHA256) of representative ``mla_prefill``
specs and compares against a checked-in per-flavor golden fixture, catching
unintended codegen drift. Pure text lowering -- no GPU / no comgr required.
Every flavor in ``LLVM_FLAVORS`` is checked from any host, through the shared
``rocke.core.ir_golden`` comparator, so a newly added flavor fails here until
the fixture is re-blessed.

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
  * ``fwd_h128_agpr`` -- the agpr_acc layout built directly. Admission refuses
    it on llvm20, which the golden records as an expected failure for that
    flavor rather than a case.
  * ``*_aq`` -- the unfused ``absorbed_query`` kernel, which a request reaches
    through ``MLARequest.absorbed_query``: the dispatched default layout and
    the agpr_acc layout (the latter refused on llvm20 like ``fwd_h128_agpr``).

There is no C++ *builder* mirror for this family, but the C++ lowerer is
family-agnostic: ``test_attention_ir_cpp_parity.py`` lowers these same cases
through both engines and requires byte-identical IR.

Re-bless, from ``rocke/library``:

    PYTHONPATH=../platform/python:. python tests/test_mla_prefill_gfx942_golden.py --write

Re-blessing is a deliberate act, never a way to turn red green: a hash that moved
without a matching, explained emission change is a defect, not a stale fixture.
"""

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

_GOLDEN = (
    Path(__file__).resolve().parent / "golden" / "mla_prefill_gfx942_ir_sha256.json"
)
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

    def agpr(num_heads, absorbed_query=False):
        # The shipped long-key layout, built directly so llvm20's refusal is
        # recorded rather than hidden behind dispatch's fallback.
        return lambda: build_mla_prefill_fwd(
            MlaPrefillSpec(
                num_heads=num_heads,
                heads_per_wg=2,
                q_tiles_per_wg=2,
                num_warps=4,
                block_q=64,
                agpr_acc=True,
                absorbed_query=absorbed_query,
            ),
            arch=_ARCH,
        )

    def probe(num_heads):
        # The probe is single-tile, so it pins block_q to one MFMA tile.
        return lambda: build_mla_prefill_score_probe(
            MlaPrefillSpec(num_heads=num_heads, block_q=16), arch=_ARCH
        )

    def dispatched(num_heads, total_q=64, max_k=96, absorbed_query=False):
        def build():
            from dispatch.mla import MLARequest, dispatch_mla

            result = dispatch_mla(
                MLARequest(
                    num_heads=num_heads,
                    total_q=total_q,
                    num_seqs=2,
                    max_seqlen_k=max_k,
                    arch=_ARCH,
                    # 228-CU gfx942, pinned: the XCD count would otherwise come from
                    # the host's device and make the fixture host-dependent.
                    num_cus=228,
                    # Two sequences of ceil(max_k / 16) pages each.
                    num_kv_blocks=2 * -(-max_k // 16),
                    absorbed_query=absorbed_query,
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
        # 4096 query rows per sequence: the agpr_acc layout at llvm22 (llvm20
        # refuses it and dispatch falls back to 16 rows per wave).
        "mla_prefill_gfx942/dispatch_fwd_h128_long": dispatched(128, 8192, 8192),
        "mla_prefill_gfx942/fwd_h128_agpr": agpr(128),
        # The absorbed_query kernel: rule-picked default layout, and agpr_acc.
        "mla_prefill_gfx942/dispatch_fwd_h128_aq": dispatched(128, absorbed_query=True),
        "mla_prefill_gfx942/fwd_h128_agpr_aq": agpr(128, absorbed_query=True),
    }


@contextmanager
def _flavor_env(flavor):
    """Build under the flavor being lowered: admission is flavor-dependent
    (agpr_acc refuses llvm20), so a case must select and admit its spec for the
    same flavor its IR is recorded for."""
    old = os.environ.get("ROCKE_LLVM_FLAVOR")
    os.environ["ROCKE_LLVM_FLAVOR"] = flavor
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("ROCKE_LLVM_FLAVOR", None)
        else:
            os.environ["ROCKE_LLVM_FLAVOR"] = old


def _run(flavor):
    """One flavor's golden sub-document.

    An admission refusal (``NotImplementedError`` from
    :func:`~kernels.mla.mla_prefill_gfx942.require_mla_prefill`) is recorded
    under ``expected_failures`` with its message, so the refusal itself is
    pinned. Any other build error propagates with its traceback, in the test
    and under ``--write`` alike, so a fixture is never blessed with a case that
    fails to lower.
    """
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    cases, failures = {}, {}
    for cid, build in _cases().items():
        with _flavor_env(flavor):
            try:
                kernel = build()
            except NotImplementedError as e:
                failures[cid] = {"type": type(e).__name__, "message": str(e)}
                continue
        llvm = _lower_kernel_to_llvm_python(kernel, arch=_ARCH, llvm_flavor=flavor)
        data = llvm.encode("utf-8")
        cases[cid] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    doc = {"cases": cases}
    if failures:
        doc["expected_failures"] = failures
    return doc


def _build_doc():
    from rocke.core.ir_golden import GOLDEN_FLAVORS

    return {
        "schema": _SCHEMA,
        "flavors": {fl: _run(fl) for fl in GOLDEN_FLAVORS},
    }


def _golden():
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("MLA golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    assert golden.get("schema") == _SCHEMA
    return golden


def test_mla_prefill_gfx942_golden_covers_every_case():
    """Every flavor accounts for exactly the case table: each case is either
    hashed or a recorded expected failure, never both, never neither.

    Without this, a case added to ``_cases()`` but never blessed would be
    silently skipped rather than failing, and a typo'd cid would pin nothing.
    """
    from rocke.core.ir_golden import GOLDEN_FLAVORS

    golden = _golden()
    want = set(_cases())
    for flavor in GOLDEN_FLAVORS:
        sub = golden["flavors"][flavor]
        hashed = set(sub["cases"])
        refused = set(sub.get("expected_failures", {}))
        assert not hashed & refused, f"{flavor}: {sorted(hashed & refused)}"
        recorded = hashed | refused
        assert recorded == want, (
            f"{flavor}: golden case set differs from the case table; "
            f"missing={sorted(want - recorded)} stale={sorted(recorded - want)}"
        )


def test_mla_prefill_gfx942_golden_expected_failures_are_the_llvm20_agpr_refusal():
    """The only refusals the golden may hold are the agpr_acc cases at llvm20;
    anything else recorded as an expected failure is a failed bless, not a
    golden."""
    from rocke.core.ir_golden import GOLDEN_FLAVORS

    agpr_cases = {
        "mla_prefill_gfx942/fwd_h128_agpr",
        "mla_prefill_gfx942/fwd_h128_agpr_aq",
    }
    golden = _golden()
    for flavor in GOLDEN_FLAVORS:
        refused = golden["flavors"][flavor].get("expected_failures", {})
        if flavor == "llvm20":
            assert set(refused) == agpr_cases
            for cid in agpr_cases:
                entry = refused[cid]
                assert entry["type"] == "NotImplementedError"
                assert "agpr_acc" in entry["message"]
        else:
            assert not refused, f"{flavor}: unexpected refusals {sorted(refused)}"


def test_mla_prefill_gfx942_ir_matches_golden():
    from rocke.core.ir_golden import check_golden

    _golden()
    drift = check_golden(_GOLDEN, _run)
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
                num_cus=228,
                num_kv_blocks=12,
            )
        )
        assert result.spec == MlaPrefillSpec(
            num_heads=num_heads,
            heads_per_wg=2,
            q_tiles_per_wg=2,
            num_warps=4,
            block_q=32,
            num_xcds=6,
        )
    # The ``dispatch_fwd_h128_aq`` case: the same spec plus the request's flag.
    absorbed = dispatch_mla(
        MLARequest(
            num_heads=128,
            total_q=64,
            num_seqs=2,
            max_seqlen_k=96,
            arch=_ARCH,
            num_cus=228,
            num_kv_blocks=12,
            absorbed_query=True,
        )
    )
    assert absorbed.spec == MlaPrefillSpec(
        num_heads=128,
        heads_per_wg=2,
        q_tiles_per_wg=2,
        num_warps=4,
        block_q=32,
        num_xcds=6,
        absorbed_query=True,
    )


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_mla_prefill_gfx942_golden_covers_every_case()
        test_mla_prefill_gfx942_golden_expected_failures_are_the_llvm20_agpr_refusal()
        test_mla_prefill_gfx942_ir_matches_golden()
