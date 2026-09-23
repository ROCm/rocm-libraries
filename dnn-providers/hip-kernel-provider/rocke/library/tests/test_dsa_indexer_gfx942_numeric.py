# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx942 lightning indexer (torch-free).

Compiles the kernel, launches it through the numpy ``run_manifest`` path, and
checks the emitted scores against the CPU oracle in
``builders.gfx942.dsa.hostpack.ref_indexer_scores``. Gated on an actual gfx942
device via a torch-free probe, so it skips cleanly on any other host (this lane
cannot run on gfx90a/gfx950).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_LIBRARY = _TESTS.parent
_PLATFORM_PYTHON = _LIBRARY.parent / "platform" / "python"
for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _gpu_arch() -> str:
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) or ""
    except Exception:  # noqa: BLE001
        return ""


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "gfx942" not in _gpu_arch(), reason="needs a gfx942 GPU"
    ),
]

_ARCH = "gfx942"
# One multiply per fp8/bf16 pair plus per-head weights stay in f32; the score is
# a ranking input, so a relaxed relative tolerance is the correct gate.
_TOL = 3e-2


@pytest.mark.parametrize(
    "Q,Sk,HI,D,block,q_pos_base,body",
    [
        (8, 64, 4, 16, 64, 0, "scalar"),      # small, decode-like base
        (8, 128, 32, 128, 256, 0, "scalar"),  # GLM-shaped H_I=32, D_I=128
        (4, 96, 8, 128, 128, 50, "scalar"),   # nonzero causal offset
        (16, 64, 32, 128, 64, 0, "mfma"),     # MFMA, GLM-shaped, 16-aligned
        (32, 128, 64, 128, 64, 0, "mfma"),    # MFMA, DeepSeek-shaped, multi-tile
        (16, 96, 8, 128, 64, 48, "mfma"),     # MFMA, nonzero causal offset
    ],
)
def test_indexer_scores_match_oracle(Q, Sk, HI, D, block, q_pos_base, body):
    from rocke.helpers.compile import compile_kernel
    from rocke.run_manifest import run_manifest

    from builders.gfx942.dsa.manifest import make_lightning_indexer_manifest
    from kernels.gfx942.lightning_indexer import (
        IndexerSpec,
        IndexerTileSpec,
        build_lightning_indexer,
        lightning_indexer_signature,
    )

    spec = IndexerSpec(
        n_index_heads=HI,
        index_head_dim=D,
        seqlen_q=Q,
        seqlen_k=Sk,
        body=body,
        tile=IndexerTileSpec(block_size=block),
    )
    artifact = compile_kernel(build_lightning_indexer(spec, arch=_ARCH), arch=_ARCH)
    assert artifact.hsaco_bytes > 0

    manifest = make_lightning_indexer_manifest(
        artifact=artifact,
        spec=spec,
        args_signature=lightning_indexer_signature(spec),
        q_pos_base=q_pos_base,
        default_shape=(Q, Sk, HI),
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        man_path = tmp / "indexer.manifest.json"
        hsaco_path = tmp / f"{spec.kernel_name()}.hsaco"
        man_path.write_text(json.dumps(manifest))
        hsaco_path.write_bytes(artifact.hsaco)
        summary = run_manifest(man_path, hsaco_path, shape=None, verify=True)

    assert summary.bad_count == 0, (
        f"{summary.bad_count}/{summary.total} scores exceed tol={_TOL} "
        f"(max_abs_diff={summary.max_abs_diff:.3e})"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
