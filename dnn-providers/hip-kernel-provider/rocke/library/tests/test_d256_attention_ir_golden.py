#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Library-side IR golden for the gfx950 D256 bf16 prefill *fast* spec.

Pins the emitted LLVM IR (sha256, both ``llvm20``/``llvm22`` flavors) of the
D256 fast spec: 32x32-transposed + ``use_kq_lds_pad`` (slab-granularity K_lds
pad) + ``use_softmax_mfma_interleave``. The slab pad's async-DMA write and its
padded read remap must agree byte-for-byte or numerics silently corrupt, but
only the spec dataclass is otherwise asserted -- an addressing drift would pass
every CPU test and fail only on GPU (#9233 review).

Lives in the LIBRARY test layer (not ``platform/tests``) because it builds a
library kernel (``kernels`` / ``builders``); platform tests must not import
library (one-way ``platform <- library`` dependency). It imports only the
pure-Python lowerer from ``rocke.core.lower_llvm`` (``library -> platform`` is
the allowed direction) with an EXPLICIT llvm flavor, so the recorded sha is a
function of the committed IR alone -- reproducible in CI on any host, GPU or
not (no ``rocke_engine`` / comgr / real device needed).

Bless (only from a verified-good tree, alongside a reviewed IR change)::

    python3 library/tests/test_d256_attention_ir_golden.py --write
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "golden" / "d256_attention_ir_sha256.json"
GOLDEN_FLAVORS = ("llvm20", "llvm22")
GOLDEN_SCHEMA = "rocke.d256_attention_ir_sha256/v1"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _current_flavor() -> str:
    """The llvm flavor this host autodetects (the golden stores both; the gate
    compares only this one, so one committed golden is valid on every host)."""
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _build_d256_fast(arch: str):
    """Build the *production* D256 fast-spec kernel under a pinned arch.

    ``_d256_gfx950_fast`` gates on the *memoized runtime device arch* via
    ``_resolve_attention_arch`` -- NOT any per-case argument -- and the spec
    builder holds a *bound* import of it. Both bindings are patched wholesale
    (per its docstring: "Tests that monkeypatch this function replace it
    wholesale") to ``arch`` so the fast spec is selected deterministically on
    any host; otherwise a non-gfx950 host silently pins the FALLBACK spec (no
    pad/interleave) and the golden becomes host-dependent. Raises if the fast
    spec / pad+interleave is not selected, so the golden can never pin the
    fallback by accident.
    """
    import builders.common.attention_spec_builder as asb
    import kernels.common.attention_unified as au
    from kernels import build_unified_attention_2d_tiled
    from kernels.common.attention_unified import (
        UnifiedAttentionProblem,
        _d256_gfx950_fast,
        _tiled_spec_from_problem,
    )

    # Validated D256 cohort point (GQA 16/2, hd256, bs16, sq4096 bf16).
    problem = UnifiedAttentionProblem(
        total_q=4096,
        num_seqs=1,
        num_query_heads=16,
        num_kv_heads=2,
        head_size=256,
        block_size=16,
        max_seqlen_q=4096,
        max_seqlen_k=4096,
        dtype="bf16",
        num_sms=120,
    )
    pin = lambda: arch  # noqa: E731
    o_au, o_asb = au._resolve_attention_arch, asb._resolve_attention_arch
    au._resolve_attention_arch = pin
    asb._resolve_attention_arch = pin
    try:
        if not _d256_gfx950_fast(problem):
            raise RuntimeError(
                f"D256 fast spec not selected under pinned arch {arch!r}; "
                "golden would pin the fallback (no pad/interleave)"
            )
        spec = _tiled_spec_from_problem(problem)
        if not (spec.use_kq_lds_pad and spec.use_softmax_mfma_interleave):
            raise RuntimeError(
                "expected pad+interleave in the D256 fast spec; got "
                f"pad={spec.use_kq_lds_pad} "
                f"interleave={spec.use_softmax_mfma_interleave}"
            )
        return build_unified_attention_2d_tiled(spec, arch=arch)
    finally:
        au._resolve_attention_arch = o_au
        asb._resolve_attention_arch = o_asb


def _cases():
    """case_id -> (arch, build thunk). Add future D256 IR pins here."""
    return {
        "d256_pad_interleave/gfx950": ("gfx950", lambda: _build_d256_fast("gfx950")),
    }


def _lower(arch, build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    kernel = build()
    llvm = _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)
    return {
        "arch": arch,
        "kernel_name": getattr(kernel, "name", "<unknown>"),
        "sha256": _sha(llvm),
        "bytes": len(llvm.encode("utf-8")),
    }


def _run(flavor):
    return {cid: _lower(arch, build, flavor) for cid, (arch, build) in _cases().items()}


def build_golden() -> dict:
    return {
        "schema": GOLDEN_SCHEMA,
        "flavors": {fl: {"cases": _run(fl)} for fl in GOLDEN_FLAVORS},
    }


class TestD256AttentionIrGolden(unittest.TestCase):
    def test_ir_matches_golden(self):
        """CPU-only: recompute the D256 fast-spec IR and compare its sha256 to
        the blessed golden for this host's llvm flavor. Any drift in the
        emitted addressing (slab-pad write/read, interleave) fails here."""
        self.assertTrue(
            GOLDEN.is_file(),
            f"missing golden {GOLDEN}; bless with "
            "`python3 library/tests/test_d256_attention_ir_golden.py --write`",
        )
        doc = json.loads(GOLDEN.read_text())
        flavor = _current_flavor()
        base = doc.get("flavors", {}).get(flavor)
        self.assertIsNotNone(
            base,
            f"golden has no entry for flavor {flavor!r} (have {sorted(doc.get('flavors', {}))})",
        )
        cur = _run(flavor)
        self.assertEqual(
            set(base["cases"]),
            set(cur),
            "D256 IR golden case set changed; re-bless with --write",
        )
        for cid, brec in base["cases"].items():
            crec = cur[cid]
            self.assertEqual(
                brec["sha256"],
                crec["sha256"],
                f"IR DRIFT [{flavor}] {cid}: {brec['sha256']} -> {crec['sha256']} "
                f"(kernel {crec['kernel_name']}); re-bless only with a reviewed, "
                "expected IR change",
            )


if __name__ == "__main__":
    import sys

    if "--write" in sys.argv:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(build_golden(), indent=2, sort_keys=True) + "\n")
        summary = ", ".join(f"{fl}={len(_run(fl))} case(s)" for fl in GOLDEN_FLAVORS)
        print(f"wrote {GOLDEN}: {summary}")
    else:
        unittest.main(verbosity=2)
