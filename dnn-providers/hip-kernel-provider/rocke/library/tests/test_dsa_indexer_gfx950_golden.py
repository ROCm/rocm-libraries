# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the gfx950 lightning-indexer kernel.

Same arch-neutral kernel as gfx942 (kernels/common/lightning_indexer.py), lowered
for gfx950 -- the emitted IR differs per arch, so gfx950 carries its own golden.
CPU-only: no GPU, no comgr.

Run or re-bless from ``rocke/library``::

    python tests/test_dsa_indexer_gfx950_golden.py --write
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
_GOLDEN = _TESTS / "golden" / "dsa_indexer_gfx950_ir_sha256.json"
_FLAVORS = ("llvm20", "llvm22")
_ARCH = "gfx950"

for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cases() -> dict[str, Callable]:
    """Representative builders for the gfx950 lightning indexer (both bodies)."""
    from kernels.common.lightning_indexer import (
        IndexerSpec,
        IndexerTileSpec,
        build_lightning_indexer,
    )

    return {
        "dsa_indexer_gfx950/deepseek_hi64": lambda: build_lightning_indexer(
            IndexerSpec(n_index_heads=64, index_head_dim=128, seqlen_q=8, seqlen_k=64),
            arch=_ARCH,
        ),
        "dsa_indexer_gfx950/glm_hi32": lambda: build_lightning_indexer(
            IndexerSpec(n_index_heads=32, index_head_dim=128, seqlen_q=8, seqlen_k=64),
            arch=_ARCH,
        ),
        "dsa_indexer_gfx950/mfma_deepseek_hi64": lambda: build_lightning_indexer(
            IndexerSpec(
                n_index_heads=64,
                index_head_dim=128,
                seqlen_q=16,
                seqlen_k=64,
                body="mfma",
                tile=IndexerTileSpec(block_size=64),
            ),
            arch=_ARCH,
        ),
        "dsa_indexer_gfx950/mfma_glm_hi32": lambda: build_lightning_indexer(
            IndexerSpec(
                n_index_heads=32,
                index_head_dim=128,
                seqlen_q=16,
                seqlen_k=64,
                body="mfma",
                tile=IndexerTileSpec(block_size=64),
            ),
            arch=_ARCH,
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
        "schema": "dsa_indexer_gfx950.ir_golden_sha256/v1",
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


def test_dsa_indexer_gfx950_ir_matches_golden():
    assert _GOLDEN.exists(), (
        f"missing gfx950 indexer golden fixture; generate it with "
        f"`python {Path(__file__).name} --write`"
    )
    golden = json.loads(_GOLDEN.read_text())
    assert golden.get("schema") == "dsa_indexer_gfx950.ir_golden_sha256/v1"

    flavor = _current_flavor()
    assert flavor in golden.get("flavors", {}), (
        f"no gfx950 indexer golden recorded for LLVM flavor {flavor!r}; "
        "review and re-bless the fixture"
    )

    cases = _cases()
    recorded = golden["flavors"][flavor]["cases"]
    assert set(recorded) == set(cases), (
        "gfx950 indexer golden case set drifted: "
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
    assert not drift, "gfx950 indexer LLVM IR drift vs golden:\n  " + "\n  ".join(drift)


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_dsa_indexer_gfx950_ir_matches_golden()
