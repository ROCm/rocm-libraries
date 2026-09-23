# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the gfx942 lightning-indexer kernel.

Pins the Python-lowered LLVM IR for a few representative indexer specs. CPU-only:
no GPU, no comgr. This is the byte-identity anchor for the DSA indexer family
(the KDA-style golden-sha tier; there is no hand-written C++ mirror).

Run or re-bless from ``rocke/library``::

    python tests/run_all.py --only dsa_indexer_gfx942
    python tests/test_dsa_indexer_gfx942_golden.py --write

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
_GOLDEN = _TESTS / "golden" / "dsa_indexer_gfx942_ir_sha256.json"
_FLAVORS = ("llvm20", "llvm22")
_ARCH = "gfx942"

for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cases() -> dict[str, Callable]:
    """Representative builders for the gfx942 lightning indexer.

    Two model-shaped cases (DeepSeek H_I=64, GLM H_I=32) plus a small case, all
    at D_I=128 (the real index_head_dim). seqlen only sizes the grid, so it is
    kept small to keep the golden cheap.
    """
    from kernels.gfx942.lightning_indexer import (
        IndexerSpec,
        IndexerTileSpec,
        build_lightning_indexer,
    )

    return {
        "dsa_indexer_gfx942/deepseek_hi64": lambda: build_lightning_indexer(
            IndexerSpec(n_index_heads=64, index_head_dim=128, seqlen_q=8, seqlen_k=64)
        ),
        "dsa_indexer_gfx942/glm_hi32": lambda: build_lightning_indexer(
            IndexerSpec(n_index_heads=32, index_head_dim=128, seqlen_q=8, seqlen_k=64)
        ),
        "dsa_indexer_gfx942/small": lambda: build_lightning_indexer(
            IndexerSpec(
                n_index_heads=4,
                index_head_dim=16,
                seqlen_q=8,
                seqlen_k=32,
                tile=IndexerTileSpec(block_size=64),
            )
        ),
        # MFMA body (matrix-core), 16-aligned shapes, one wave64.
        "dsa_indexer_gfx942/mfma_deepseek_hi64": lambda: build_lightning_indexer(
            IndexerSpec(
                n_index_heads=64,
                index_head_dim=128,
                seqlen_q=16,
                seqlen_k=64,
                body="mfma",
                tile=IndexerTileSpec(block_size=64),
            )
        ),
        "dsa_indexer_gfx942/mfma_glm_hi32": lambda: build_lightning_indexer(
            IndexerSpec(
                n_index_heads=32,
                index_head_dim=128,
                seqlen_q=16,
                seqlen_k=64,
                body="mfma",
                tile=IndexerTileSpec(block_size=64),
            )
        ),
    }


def _current_flavor() -> str:
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build: Callable, flavor: str) -> tuple[str, int]:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc() -> dict:
    cases = _cases()
    return {
        "schema": "dsa_indexer_gfx942.ir_golden_sha256/v1",
        "flavors": {
            flavor: {
                "cases": {
                    cid: {"sha256": sha, "bytes": nbytes}
                    for cid, build in cases.items()
                    for sha, nbytes in [_sha_for(build, flavor)]
                }
            }
            for flavor in _FLAVORS
        },
    }


def test_dsa_indexer_gfx942_ir_matches_golden():
    assert _GOLDEN.exists(), (
        f"missing gfx942 indexer golden fixture; generate it with "
        f"`python {Path(__file__).name} --write`"
    )
    golden = json.loads(_GOLDEN.read_text())
    assert golden.get("schema") == "dsa_indexer_gfx942.ir_golden_sha256/v1"

    flavor = _current_flavor()
    assert flavor in golden.get("flavors", {}), (
        f"no gfx942 indexer golden recorded for LLVM flavor {flavor!r}; "
        "review and re-bless the fixture"
    )

    cases = _cases()
    recorded = golden["flavors"][flavor]["cases"]
    assert set(recorded) == set(cases), (
        "gfx942 indexer golden case set drifted: "
        f"recorded={sorted(recorded)}, current={sorted(cases)}"
    )

    drift = []
    for cid, build in cases.items():
        want = recorded[cid]["sha256"]
        got, nbytes = _sha_for(build, flavor)
        if got != want:
            drift.append(
                f"{cid}: {want} -> {got} ({recorded[cid]['bytes']} -> {nbytes} bytes)"
            )
    assert not drift, "gfx942 indexer LLVM IR drift vs golden:\n  " + "\n  ".join(drift)


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_dsa_indexer_gfx942_ir_matches_golden()
