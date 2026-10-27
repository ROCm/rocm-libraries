# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the gfx942 KDA kernel family.

The fixture pins the Python-lowered LLVM IR for the C16 raw-input
prep, scan, and fused kernels. This is a CPU-only test: it does not require a
GPU or comgr.

Run or re-bless from ``rocke/library``::

    python tests/run_all.py --only kda_gfx942
    python tests/test_kda_gfx942_golden.py --write

Review the IR change before re-blessing the hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

_TESTS = Path(__file__).resolve().parent
_LIBRARY = _TESTS.parent
_PLATFORM_PYTHON = _LIBRARY.parent / "platform" / "python"
_GOLDEN = _TESTS / "golden" / "kda_gfx942_ir_sha256.json"
_ARCH = "gfx942"

# Keep the file directly executable without requiring callers to construct a
# PYTHONPATH.  The library path must precede tests/ so tests/dispatch cannot
# shadow the real dispatch package.
for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cases() -> dict[str, Callable]:
    """Representative builders for every gfx942 KDA kernel in this PR."""
    from kernels.gfx942.kda_chunkwise import (
        KdaChunkFusedSpec,
        KdaChunkPrepSpec,
        KdaChunkScanSpec,
        build_kda_chunk_fused,
        build_kda_chunk_prep,
        build_kda_chunk_scan,
    )

    return {
        "kda_gfx942/c16_prep_default": lambda: build_kda_chunk_prep(KdaChunkPrepSpec()),
        "kda_gfx942/c16_scan_default": lambda: build_kda_chunk_scan(KdaChunkScanSpec()),
        "kda_gfx942/c16_scan_initial_state": lambda: build_kda_chunk_scan(
            KdaChunkScanSpec(has_initial_state=True)
        ),
        "kda_gfx942/c16_fused_prefetch": lambda: build_kda_chunk_fused(
            KdaChunkFusedSpec()
        ),
        "kda_gfx942/c16_fused_initial_state": lambda: build_kda_chunk_fused(
            KdaChunkFusedSpec(has_initial_state=True)
        ),
    }


def _sha_for(build: Callable, flavor: str) -> tuple[str, int]:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _run(flavor: str) -> dict:
    """One flavor's golden sub-document. A build error propagates, so a fixture
    can never be blessed with a case that fails to lower."""
    return {
        "cases": {
            cid: {"sha256": sha, "bytes": nbytes}
            for cid, build in _cases().items()
            for sha, nbytes in [_sha_for(build, flavor)]
        }
    }


def _build_doc() -> dict:
    from rocke.core.ir_golden import GOLDEN_FLAVORS

    return {
        "schema": "kda_gfx942.ir_golden_sha256/v1",
        "flavors": {flavor: _run(flavor) for flavor in GOLDEN_FLAVORS},
    }


def test_kda_gfx942_ir_matches_golden():
    from rocke.core.ir_golden import check_golden

    assert _GOLDEN.exists(), (
        f"missing gfx942 KDA golden fixture; generate it with "
        f"`python {Path(__file__).name} --write`"
    )
    golden = json.loads(_GOLDEN.read_text())
    assert golden.get("schema") == "kda_gfx942.ir_golden_sha256/v1"

    # Every flavor in LLVM_FLAVORS, from any host, so a datalayout or intrinsic
    # change for a flavor this host does not run still fails here.
    drift = check_golden(_GOLDEN, _run)
    assert not drift, "gfx942 KDA LLVM IR drift vs golden:\n  " + "\n  ".join(drift)


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_kda_gfx942_ir_matches_golden()
