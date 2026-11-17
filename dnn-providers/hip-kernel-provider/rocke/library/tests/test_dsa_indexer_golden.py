# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the lightning-indexer kernel (both arches).

Pins the Python-lowered LLVM IR for a few representative indexer specs on gfx942
and gfx950. The kernel is arch-neutral, but the lowered IR carries the arch
target, so the SHAs differ per arch; one arch-keyed golden holds both. CPU-only:
no GPU, no comgr. This is the byte-identity anchor for the DSA indexer family (the
KDA-style golden-sha tier; there is no hand-written C++ mirror).

Run or re-bless from ``rocke/library``::

    python tests/test_dsa_indexer_golden.py --write

Review the IR change before re-blessing the hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import pytest

_TESTS = Path(__file__).resolve().parent
_LIBRARY = _TESTS.parent
_PLATFORM_PYTHON = _LIBRARY.parent / "platform" / "python"
_GOLDEN = _TESTS / "golden" / "dsa_indexer_ir_sha256.json"
_SCHEMA = "dsa_indexer.ir_golden_sha256/v1"
_FLAVORS = ("llvm20", "llvm22", "llvm23")
_ARCHES = ("gfx942", "gfx950")

for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cases() -> dict[str, Callable]:
    """Representative builders, arch-neutral (the arch is applied at lower time).

    Two model-shaped cases (DeepSeek H_I=64, GLM H_I=32) plus a small case, all at
    D_I=128 (the real index_head_dim), for each body. seqlen only sizes the grid,
    so it is kept small to keep the golden cheap.
    """
    from kernels.common.lightning_indexer import (
        IndexerSpec,
        IndexerTileSpec,
        build_lightning_indexer,
    )

    return {
        "deepseek_hi64": lambda: build_lightning_indexer(
            IndexerSpec(n_index_heads=64, index_head_dim=128, seqlen_q=8, seqlen_k=64)
        ),
        "glm_hi32": lambda: build_lightning_indexer(
            IndexerSpec(n_index_heads=32, index_head_dim=128, seqlen_q=8, seqlen_k=64)
        ),
        "small": lambda: build_lightning_indexer(
            IndexerSpec(
                n_index_heads=4,
                index_head_dim=16,
                seqlen_q=8,
                seqlen_k=32,
                tile=IndexerTileSpec(block_size=64),
            )
        ),
        "mfma_deepseek_hi64": lambda: build_lightning_indexer(
            IndexerSpec(
                n_index_heads=64,
                index_head_dim=128,
                seqlen_q=16,
                seqlen_k=64,
                body="mfma",
                tile=IndexerTileSpec(block_size=64),
            )
        ),
        "mfma_glm_hi32": lambda: build_lightning_indexer(
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


def _sha_for(build: Callable, arch: str, flavor: str) -> tuple[str, int]:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=arch, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc() -> dict:
    cases = _cases()
    return {
        "schema": _SCHEMA,
        "arches": {
            arch: {
                "flavors": {
                    flavor: {
                        "cases": {
                            cid: {"sha256": sha, "bytes": nbytes}
                            for cid, build in cases.items()
                            for sha, nbytes in [_sha_for(build, arch, flavor)]
                        }
                    }
                    for flavor in _FLAVORS
                }
            }
            for arch in _ARCHES
        },
    }


@pytest.mark.parametrize("arch", _ARCHES)
def test_dsa_indexer_ir_matches_golden(arch: str):
    assert _GOLDEN.exists(), (
        f"missing indexer golden fixture; generate it with "
        f"`python {Path(__file__).name} --write`"
    )
    golden = json.loads(_GOLDEN.read_text())
    assert golden.get("schema") == _SCHEMA
    assert arch in golden.get("arches", {}), f"no golden recorded for arch {arch!r}"

    flavor = _current_flavor()
    arch_doc = golden["arches"][arch]
    assert flavor in arch_doc.get("flavors", {}), (
        f"no {arch} indexer golden recorded for LLVM flavor {flavor!r}; "
        "review and re-bless the fixture"
    )

    cases = _cases()
    recorded = arch_doc["flavors"][flavor]["cases"]
    assert set(recorded) == set(cases), (
        f"{arch} indexer golden case set drifted: "
        f"recorded={sorted(recorded)}, current={sorted(cases)}"
    )

    drift = []
    for cid, build in cases.items():
        want = recorded[cid]["sha256"]
        got, nbytes = _sha_for(build, arch, flavor)
        if got != want:
            drift.append(
                f"{cid}: {want} -> {got} ({recorded[cid]['bytes']} -> {nbytes} bytes)"
            )
    assert not drift, f"{arch} indexer LLVM IR drift vs golden:\n  " + "\n  ".join(
        drift
    )


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        for _arch in _ARCHES:
            test_dsa_indexer_ir_matches_golden(_arch)
        print("PASS")
