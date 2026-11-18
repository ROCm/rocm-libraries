# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The rocDojo gfx1250 bf16 custom kernels and their Prediction logic.

Each ``*_rocdojo.yaml`` Origami file sits beside the stock gfx1250 Origami file
whose name it extends, so the library build merges both into one Prediction
table. That holds only while the two carry the same ProblemType and
DeviceNames. TT has no stock Origami file: its rocDojo file must carry the NT
ProblemType with TN's A-operand keys. Every kernel a file names must be bundled,
every bundled ``rocdojo_*`` kernel must be named by exactly one file, and its
``custom.config`` must agree with that file's data types and transposes.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from logic_corpus import needs_logic_corpus, require_logic_file

from tensilelite.Common.DataType import DataType
from tensilelite.CustomKernels import readCustomKernelConfig
from tensilelite.resources import custom_kernel_names, custom_kernel_text

pytestmark = pytest.mark.unit

_ORIGAMI = "gfx1250/gfx1250/Origami"
_NN = "gfx1250_Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs"
_NT = "gfx1250_Cijk_Ailk_Bjlk_BBS_BH_BiasSB_HAS_SAV_UserArgs"
_TN = "gfx1250_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs"
_TT = "gfx1250_Cijk_Alik_Bjlk_BBS_BH_BiasSB_HAS_SAV_UserArgs"
_STOCK_STEMS = (_NN, _NT, _TN)
_A_KEYS = ("TransposeA", "TLUA", "IndexAssignmentsA", "IndexUnrollA")
_PREFIX = "rocdojo_"
_MANIFEST = Path(__file__).resolve().parents[2] / "CustomKernels" / "rocdojo" / "MANIFEST.json"


def _read(rel):
    with open(require_logic_file(f"{_ORIGAMI}/{rel}.yaml")) as f:
        return yaml.load(f, Loader=yaml.CSafeLoader)


def _rocdojo(stem):
    return _read(f"{stem}_rocdojo")


@needs_logic_corpus
@pytest.mark.parametrize("stem", _STOCK_STEMS)
def test_logic_joins_its_stock_sibling(stem):
    ours, stock = _rocdojo(stem), _read(stem)
    assert ours["ProblemType"] == stock["ProblemType"]
    assert ours["DeviceNames"] == stock["DeviceNames"]
    assert ours["LibraryType"] == stock["LibraryType"] == "Prediction"
    assert ours["CUCount"] == stock["CUCount"]


@needs_logic_corpus
def test_tt_logic_is_nt_with_tn_a_operand():
    ours, nt, tn = _rocdojo(_TT), _read(_NT), _read(_TN)
    want = dict(nt["ProblemType"])
    want.update({key: tn["ProblemType"][key] for key in _A_KEYS})
    assert ours["ProblemType"] == want
    assert ours["DeviceNames"] == nt["DeviceNames"]
    assert ours["LibraryType"] == "Prediction"


@needs_logic_corpus
def test_every_named_kernel_is_bundled_and_every_bundled_kernel_is_named():
    bundled = {name for name in custom_kernel_names() if name.startswith(_PREFIX)}
    named = {}
    for stem in (*_STOCK_STEMS, _TT):
        for solution in _rocdojo(stem)["Solutions"]:
            named.setdefault(solution["CustomKernelName"], []).append(stem)
    assert sorted(set(named) - bundled) == []
    assert sorted(bundled - set(named)) == []
    assert {name: stems for name, stems in named.items() if len(stems) > 1} == {}


@needs_logic_corpus
@pytest.mark.parametrize("stem", (*_STOCK_STEMS, _TT))
def test_kernel_config_matches_its_logic(stem):
    logic = _rocdojo(stem)
    problem = logic["ProblemType"]
    indices = [solution["SolutionIndex"] for solution in logic["Solutions"]]
    assert indices == list(range(len(indices)))
    for solution in logic["Solutions"]:
        name = solution["CustomKernelName"]
        assert solution["SolutionNameMin"] == name
        config = readCustomKernelConfig(name)
        assert config["Source"]["Origin"] == "rocdojo"
        # custom.config spells data types as characters, logic files as enum values.
        for key in ("DataType", "DestDataType", "ComputeDataType"):
            assert DataType(config["ProblemType"][key]) == DataType(problem[key]), (name, key)
        for key in ("TransposeA", "TransposeB"):
            assert config["ProblemType"][key] == problem[key], (name, key)
        assert config["MatrixInstruction"] == solution["MatrixInstruction"], name


def test_manifest_hashes_the_bundled_kernels():
    manifest = json.loads(_MANIFEST.read_text())
    records = {record["name"]: record for record in manifest["kernels"]}
    assert set(records) == {name for name in custom_kernel_names() if name.startswith(_PREFIX)}
    for name, record in records.items():
        text = custom_kernel_text(name)
        header, _, body = text.partition("\n\t.amdgcn_target")
        assert all(line.startswith("//") for line in header.splitlines()), name
        body = "\t.amdgcn_target" + body
        assert hashlib.sha256(body.encode()).hexdigest() == record["body_sha256"], name
