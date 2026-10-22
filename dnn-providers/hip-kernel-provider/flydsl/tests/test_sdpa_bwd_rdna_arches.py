# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The arch allow-lists of the SDPA backward ops, refused before anything compiles.

The RDNA smoke op builds only for ``gfx11-generic`` and the six CDNA ops only for
``gfx950``; each side's generator must refuse the other's target on the command line
(exit 1, nothing written) rather than hand a foreign ABI to the compiler. Host-only:
no FlyDSL install is needed, because the refusal precedes every compile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import (  # noqa: E402
    gen_sdpa_bwd_dbias_reduce,
    gen_sdpa_bwd_dkdv,
    gen_sdpa_bwd_dq,
    gen_sdpa_bwd_dsink_reduce,
    gen_sdpa_bwd_merged,
    gen_sdpa_bwd_prep,
    gen_sdpa_bwd_rdna_dbias_reduce,
)
from generators._instances import instances_for  # noqa: E402

CDNA_GENERATORS = (
    gen_sdpa_bwd_prep,
    gen_sdpa_bwd_dkdv,
    gen_sdpa_bwd_dq,
    gen_sdpa_bwd_merged,
    gen_sdpa_bwd_dbias_reduce,
    gen_sdpa_bwd_dsink_reduce,
)


def _run(module, arch: str, out_dir: Path) -> int:
    return module.main(["--arch", arch, "--out-dir", str(out_dir)])


@pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx90a", "gfx12-generic"])
def test_the_rdna_smoke_op_refuses_every_target_but_gfx11_generic(
    arch: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(gen_sdpa_bwd_rdna_dbias_reduce, arch, tmp_path) == 1
    assert "gfx11-generic" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("module", CDNA_GENERATORS, ids=lambda m: m.OP)
def test_the_cdna_ops_refuse_gfx11_generic(
    module, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(module, "gfx11-generic", tmp_path) == 1
    assert "gfx950" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_the_smoke_op_has_a_row_on_gfx11_generic_only() -> None:
    assert len(instances_for("sdpa_bwd_rdna_dbias_reduce", "gfx11-generic")) == 1
    for arch in ("gfx950", "gfx942", "gfx90a", "gfx12-generic"):
        assert instances_for("sdpa_bwd_rdna_dbias_reduce", arch) == []
