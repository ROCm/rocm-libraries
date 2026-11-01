# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The arch allow-lists of the SDPA backward ops, refused before anything compiles.

The split ops (prep, dkdv, dq) build ``gfx950`` and ``gfx11-generic``; the merged and reduce ops build only
``gfx950``. Each generator must refuse any other target on the command line (exit 1, nothing written) rather
than hand a foreign ABI to the compiler. Host-only: no FlyDSL install is needed, because the refusal precedes
every compile.
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
)
from generators._instances import instances_for  # noqa: E402

SPLIT_GENERATORS = (gen_sdpa_bwd_prep, gen_sdpa_bwd_dkdv, gen_sdpa_bwd_dq)
CDNA_ONLY_GENERATORS = (
    gen_sdpa_bwd_merged,
    gen_sdpa_bwd_dbias_reduce,
    gen_sdpa_bwd_dsink_reduce,
)


def _run(module, arch: str, out_dir: Path) -> int:
    return module.main(["--arch", arch, "--out-dir", str(out_dir)])


@pytest.mark.parametrize("module", CDNA_ONLY_GENERATORS, ids=lambda m: m.OP)
def test_the_cdna_only_ops_refuse_gfx11_generic(
    module, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(module, "gfx11-generic", tmp_path) == 1
    assert "gfx950" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a", "gfx12-generic"])
@pytest.mark.parametrize("module", SPLIT_GENERATORS, ids=lambda m: m.OP)
def test_the_split_ops_refuse_every_other_target(
    module, arch: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(module, arch, tmp_path) == 1
    assert "gfx950" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("module", SPLIT_GENERATORS, ids=lambda m: m.OP)
def test_a_gfx11_build_needs_the_s1_reference_dir(module, tmp_path: Path) -> None:
    """The pins are the oracle of every gfx11 object: no reference directory, no build."""
    from generators._codeobject import GeneratorError  # noqa: PLC0415

    with pytest.raises(GeneratorError, match="reference-dir"):
        module.generate("gfx11-generic", tmp_path)


def test_only_the_split_ops_have_rows_on_gfx11_generic() -> None:
    for module in SPLIT_GENERATORS:
        assert instances_for(module.OP, "gfx11-generic")
    for module in CDNA_ONLY_GENERATORS:
        assert instances_for(module.OP, "gfx11-generic") == []
