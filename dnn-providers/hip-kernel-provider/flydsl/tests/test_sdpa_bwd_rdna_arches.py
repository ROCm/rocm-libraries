# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The arch allow-lists of the SDPA backward ops, refused before anything compiles.

The split ops (prep, dkdv, dq) build ``gfx950``, ``gfx11-generic``, ``gfx12-generic``, ``gfx942`` and ``gfx90a``; the merged and reduce ops build only
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


@pytest.mark.parametrize("arch", ["gfx11-generic", "gfx12-generic"])
@pytest.mark.parametrize("module", CDNA_ONLY_GENERATORS, ids=lambda m: m.OP)
def test_the_cdna_only_ops_refuse_the_generic_arches(
    module, arch: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(module, arch, tmp_path) == 1
    assert "gfx950" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1200", "gfx1201"])
@pytest.mark.parametrize("module", SPLIT_GENERATORS, ids=lambda m: m.OP)
def test_the_split_ops_refuse_every_other_target(
    module, arch: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(module, arch, tmp_path) == 1
    assert "gfx950" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("module", SPLIT_GENERATORS, ids=lambda m: m.OP)
def test_gfx12_generic_builds_the_three_split_ops_through_the_rdna_builder(
    module,
) -> None:
    """gfx12-generic is a generic arch of the RDNA builder: each split op routes it there, and its tables are the arch's own."""
    from generators import _instances as inst  # noqa: PLC0415
    from generators import _sdpa_bwd_rdna_build as rdna  # noqa: PLC0415

    assert "gfx12-generic" in rdna.ARCHES
    assert "gfx12-generic" in rdna.GENERIC_ARCHES
    assert "gfx12-generic" not in rdna.GATED_ARCHES
    assert "gfx12-generic" in module._OTHER_ARCHES
    assert rdna._REFERENCE_CSVS["gfx12-generic"].name == "sdpa_bwd_gfx12_reference.csv"
    # Every row of a generic arch is an RDNA-family row: the builder takes the arch's whole table.
    rows = inst.instances_for(module.OP, "gfx12-generic")
    assert rows
    assert rdna.rows_of(module.OP, "gfx12-generic")[0] == rows


@pytest.mark.parametrize("arch", ["gfx11-generic", "gfx12-generic", "gfx942", "gfx90a"])
@pytest.mark.parametrize("module", SPLIT_GENERATORS, ids=lambda m: m.OP)
def test_a_non_gfx950_build_needs_the_s1_reference_dir(
    module, arch: str, tmp_path: Path
) -> None:
    """The pins are the oracle of every object of these arches: no reference directory, no build."""
    from generators._codeobject import GeneratorError  # noqa: PLC0415

    with pytest.raises(GeneratorError, match="reference-dir"):
        module.generate(arch, tmp_path)


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
def test_the_mixed_arches_are_built_through_the_merge_module_only(
    arch: str, tmp_path: Path
) -> None:
    """A manifest holds both builder families' rows: neither builder writes one alone, and only the split ops have rows."""
    from generators import _sdpa_bwd_cdna_build as cdna  # noqa: PLC0415
    from generators import _sdpa_bwd_rdna_build as rdna  # noqa: PLC0415
    from generators._codeobject import GeneratorError  # noqa: PLC0415

    with pytest.raises(GeneratorError, match="_sdpa_bwd_split_build"):
        cdna.generate("sdpa_bwd_dq", arch, tmp_path, None, tmp_path)
    with pytest.raises(GeneratorError, match="_sdpa_bwd_split_build"):
        rdna.generate("sdpa_bwd_dq", arch, tmp_path, None, tmp_path)
    for module in CDNA_ONLY_GENERATORS:
        assert _run(module, arch, tmp_path) == 1


def test_only_the_split_ops_have_rows_on_gfx11_generic() -> None:
    for module in SPLIT_GENERATORS:
        assert instances_for(module.OP, "gfx11-generic")
    for module in CDNA_ONLY_GENERATORS:
        assert instances_for(module.OP, "gfx11-generic") == []


def test_a_dq_policy_scheduler_reaches_the_build_hints_and_the_launcher_hints(
    monkeypatch,
) -> None:
    """G16: ``build`` writes ``llvm_options["amdgpu-sched-strategy"]`` into the per-build hints copy when the dQ policy row spells it, and
    leaves the shared ``_COMPILE_HINTS`` and every other Spec's hints alone; the launcher's own hints (the JIT path) carry it too.
    """
    pytest.importorskip("flydsl")
    sys.path.insert(0, str(FLYDSL / "kernels_src"))
    from generators import _sdpa_bwd_rdna_build as rdna  # noqa: PLC0415
    from generators._sdpa_bwd_spec import Spec  # noqa: PLC0415
    from kernels.attention_bwd_rdna import fmha_bwd_dq_gfx11 as dq  # noqa: PLC0415
    from kernels.attention_bwd_rdna import (
        fmha_tuning_bwd_dq_gfx11 as tuning,
    )  # noqa: PLC0415

    key = "amdgpu-sched-strategy"
    spec = Spec("dq", "f16", 128, "plain", "bk", 0, 1, "g1", arch="gfx12-generic")
    base = rdna.build(spec)
    assert key not in base.hints["llvm_options"] and base.knobs.sched_strategy == ""
    shared = {
        k: dict(v) if isinstance(v, dict) else v for k, v in dq._COMPILE_HINTS.items()
    }
    monkeypatch.setitem(tuning.POLICY_GFX12, 128, (2, 8, 16, "max-memory-clause"))
    built = rdna.build(spec)
    assert built.knobs.sched_strategy == "max-memory-clause"
    assert built.hints["llvm_options"][key] == "max-memory-clause"
    assert {
        k: v for k, v in built.hints["llvm_options"].items() if k != key
    } == dq._COMPILE_HINTS["llvm_options"]
    assert dq._COMPILE_HINTS == shared
    other = rdna.build(
        Spec("dq", "bf16", 96, "plain", "bk", 0, 1, "g1", arch="gfx12-generic")
    )
    assert key not in other.hints["llvm_options"]
    # dK/dV keeps its inert function attribute: its record's default ``sched_strategy`` never becomes an LLVM option
    for arch in ("gfx11-generic", "gfx12-generic"):
        dkdv = rdna.build(
            Spec("dkdv", "f16", 128, "plain", "bk", 0, 1, "g1", arch=arch)
        )
        assert dkdv.knobs.sched_strategy == "max-memory-clause"
        assert key not in dkdv.hints["llvm_options"]
