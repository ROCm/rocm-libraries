# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The rne shipping rule: a bf16 dK/dV or dQ row of an arch with the ``rne`` knob ships only when it spells ``rne=1``.

Both gates are exercised through their entry points: the table installer's ``read_pins`` and the descriptor
generator's ``require_rne_pins`` (what ``gen_descriptors.py --check`` runs after ``collect``). The rule is not in
``Spec``: a generic bf16 key without ``rne`` stays a valid Spec for sweeps to build and measure.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "tools"))
import gen_descriptors as gd  # noqa: E402
import sdpa_bwd_tables_from_sweep as tool  # noqa: E402
from generators._sdpa_bwd_spec import Spec  # noqa: E402

G11 = ";arch=gfx11-generic"
ROWS = {
    "dkdv bf16 rne=1": f"dkdv,bf16,64,causal,bk,0,1,g1,0,1{G11};rne=1",
    "dq bf16 rne=1": f"dq,bf16,128,bias,bk,0,1,g1,0,1{G11};rne=1",
    "dkdv bf16 generic": f"dkdv,bf16,64,causal,bk,0,1,g1,0,1{G11}",
    "dq bf16 generic": f"dq,bf16,128,bias,bk,0,1,g1,0,1{G11}",
}
# Rows the rule never touches: another dtype, another kernel, and gfx950 (hardware RNE conversion, no rne knob).
EXEMPT = {
    "dkdv f16": f"dkdv,f16,64,causal,bk,0,1,g1,0,1{G11}",
    "dq f16": f"dq,f16,128,bias,bk,0,1,g1,0,1{G11}",
    "dkdv fp32": f"dkdv,fp32,224,plain,bk,0,1,g1,0,1{G11}",
    "dq fp32": f"dq,fp32,128,bias,bk,0,1,g1,0,1{G11}",
    "prep bf16": f"prep,bf16,128,plain,bk,0,1,tail,0,1{G11}",
    "gfx950 dkdv bf16": "dkdv,bf16,128,bias,bk,0,0,tail,0,1",
    "gfx950 dq bf16": "dq,bf16,128,bias,bk,0,0,tail,0,1",
}
SPELLED = [k for k in ROWS if k.endswith("rne=1")]
BARE = [k for k in ROWS if not k.endswith("rne=1")]


def _s1_dir(tmp_path: Path, *keys: str) -> Path:
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh)
        out.writerow(["hash", "spec", "sha256", "rmir_scan"])
        for i, key in enumerate(keys):
            spec = Spec.parse(key)
            out.writerow([spec.hash(), spec.key(), f"{i:064x}", "clean"])
    (tmp_path / "known-flaky.csv").write_text("hash\n", encoding="utf-8")
    return tmp_path


def _entry(key: str) -> gd.KernelEntry:
    spec = Spec.parse(key)
    return gd.KernelEntry(
        f"sdpa_bwd_{spec.kernel}",
        Path(f"{spec.hash()}.hsaco"),
        b"",
        {"spec": spec.key()},
    )


@pytest.mark.parametrize("name", SPELLED)
def test_an_rne_row_is_installed_and_shipped(tmp_path, name) -> None:
    arch, pins, _, _ = tool.read_pins(_s1_dir(tmp_path, ROWS[name]))
    assert arch == "gfx11-generic" and len(pins) == 1
    gd.require_rne_pins([_entry(ROWS[name])])


@pytest.mark.parametrize("name", BARE)
def test_a_generic_bf16_row_without_rne_is_refused_by_both_gates(
    tmp_path, name
) -> None:
    with pytest.raises(SystemExit, match="must spell rne=1"):
        tool.read_pins(_s1_dir(tmp_path, ROWS[name]))
    with pytest.raises(ValueError, match="1 shipped row.*rne"):
        gd.require_rne_pins([_entry(ROWS["dkdv bf16 rne=1"]), _entry(ROWS[name])])


def test_the_refusal_counts_every_offending_row() -> None:
    with pytest.raises(ValueError, match=r"2 shipped row\(s\)"):
        gd.require_rne_pins([_entry(ROWS[name]) for name in BARE])


@pytest.mark.parametrize("name", EXEMPT)
def test_other_dtypes_kernels_and_gfx950_are_not_subject_to_the_rule(
    tmp_path, name
) -> None:
    tool.read_pins(_s1_dir(tmp_path, EXEMPT[name]))
    gd.require_rne_pins([_entry(EXEMPT[name])])
