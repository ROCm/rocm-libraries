# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scoped dual-address partitions, operand identity, and public JSON boundaries."""

from __future__ import annotations

from dataclasses import replace
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import rocke
from rocke.analysis.lds import LdsAccess, LdsPredictionError, predict_lds_conflicts
from rocke.analysis.lds.model import ModelValidationError
from rocke.analysis.lds.serialization import dumps, loads
from rocke.assets import dsl_docs_dir


_CASES = [
    (f"ds_{direction}_2addr{suffix}_b{bits}", bits // 8, spacing)
    for direction in ("load", "store")
    for bits, suffix, spacing in (
        (32, "", 64),
        (64, "", 64),
        (32, "_stride64", 256),
        (64, "_stride64", 512),
    )
]


def _accesses(width, spacing, lanes=(0, 1), bases=(0, 128)):
    return [
        LdsAccess(
            2 * i + operand,
            lane,
            base + operand * spacing,
            width,
            operand_index=operand,
        )
        for i, (lane, base) in enumerate(zip(lanes, bases))
        for operand in (0, 1)
    ]


def _predict(opcode, accesses, target="gfx1201", wave_size=32):
    return predict_lds_conflicts(
        target=target,
        opcode=opcode,
        wave_size=wave_size,
        accesses=accesses,
    )


@pytest.mark.parametrize(("opcode", "width", "spacing"), _CASES)
def test_all_ordered_lane_pairs_keep_operands_separate(opcode, width, spacing):
    # Golden membership is explicit, independent of the profile's bit formula.
    phases = (
        (set(range(32)),)
        if width == 4
        else (
            set(range(16)),
            set(range(16, 32)),
        )
    )
    for lane_a in range(32):
        for lane_b in range(32):
            if lane_a == lane_b:
                continue
            result = _predict(opcode, _accesses(width, spacing, (lane_a, lane_b)))
            expected = any({lane_a, lane_b} <= phase for phase in phases)
            assert [g.access_ids for g in result.conflict_groups] == (
                [(0, 2), (1, 3)] if expected else []
            )
            assert all(g.multiplicity == 2 for g in result.conflict_groups)
            assert [a.classification.value for a in result.accesses] == (
                ["conflict"] * 4 if expected else ["normal"] * 4
            )
            assert result.request.access_width_bytes == width
            assert result.request.active_lanes == tuple(sorted((lane_a, lane_b)))
            assert result.schema_version == 2


@pytest.mark.parametrize(("opcode", "width", "spacing"), _CASES)
def test_aligned_sweeps_and_cross_operand_addresses(opcode, width, spacing):
    for distance in range(width, 769, width):
        result = _predict(opcode, _accesses(width, spacing, bases=(0, distance)))
        assert len(result.conflict_groups) == (2 if distance % 128 == 0 else 0)
    # Ordinary operands {0,64} and {64,128} must not invent a cross-operand
    # conflict. Stride64 forms also never group the two operands of one lane.
    result = _predict(opcode, _accesses(width, spacing, bases=(0, 64)))
    assert not result.conflict_groups
    single = _predict(opcode, _accesses(width, spacing, lanes=(0,), bases=(0,)))
    assert single.summary.active_access_count == 2
    assert not single.conflict_groups


@pytest.mark.parametrize(("opcode", "width", "spacing"), _CASES)
def test_legacy_spellings_and_json_round_trip(opcode, width, spacing):
    legacy = opcode.replace("load_2addr", "read2").replace("store_2addr", "write2")
    legacy = legacy.replace("_stride64", "st64")
    result = _predict(opcode, _accesses(width, spacing))
    alias = _predict(legacy, _accesses(width, spacing))
    assert alias.accesses == result.accesses
    assert alias.conflict_groups == result.conflict_groups
    assert loads(dumps(result)) == result
    assert dumps(loads(dumps(result))) == dumps(result)
    assert [a.operand_index for a in result.accesses] == [0, 1, 0, 1]
    schema = json.loads(
        (
            files("rocke.analysis.lds")
            / "schema"
            / "lds-conflict-result-v2.schema.json"
        ).read_text()
    )
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(json.loads(dumps(result)), schema)


@pytest.mark.parametrize(("opcode", "width", "spacing"), _CASES)
def test_rejects_incomplete_duplicate_and_unsupported_spacing(opcode, width, spacing):
    accesses = _accesses(width, spacing)
    with pytest.raises(LdsPredictionError, match="both active address operands"):
        _predict(opcode, accesses[:-1])
    with pytest.raises(LdsPredictionError, match="one active access per lane"):
        _predict(opcode, accesses + [replace(accesses[0], access_id=4)])
    with pytest.raises(LdsPredictionError, match="operand spacing"):
        _predict(
            opcode,
            [
                (
                    replace(a, lds_byte_address=a.lds_byte_address + width)
                    if a.operand_index
                    else a
                )
                for a in accesses
            ],
        )
    with pytest.raises(LdsPredictionError, match="at most two active lanes"):
        _predict(opcode, _accesses(width, spacing, (0, 1, 2), (0, 128, 256)))
    with pytest.raises(LdsPredictionError, match="distinct, non-overlapping"):
        _predict(opcode, _accesses(width, spacing, bases=(0, 0)))
    with pytest.raises(LdsPredictionError, match="both active address operands"):
        _predict(opcode, [replace(accesses[0], active=False)] + accesses[1:])


@pytest.mark.parametrize(("opcode", "width", "spacing"), _CASES)
def test_both_operands_are_validated_and_inactive_pairs_preserved(
    opcode, width, spacing
):
    accesses = _accesses(width, spacing)
    with pytest.raises(LdsPredictionError, match="byte aligned"):
        _predict(
            opcode,
            [
                (
                    replace(a, lds_byte_address=a.lds_byte_address + 1)
                    if a.operand_index
                    else a
                )
                for a in accesses
            ],
        )
    with pytest.raises(LdsPredictionError, match="LDS capacity"):
        _predict(opcode, _accesses(width, spacing, (0,), (65536 - width,)))
    last = _predict(opcode, _accesses(width, spacing, (0,), (65536 - spacing - width,)))
    assert last.accesses[1].lds_byte_address + width == 65536
    inactive = [replace(a, active=False) for a in _accesses(width, spacing, (2,), (0,))]
    inactive = [replace(a, access_id=a.access_id + 4) for a in inactive]
    result = _predict(opcode, accesses + inactive)
    assert result.summary.inactive_access_count == 2
    assert result.request.active_lanes == (0, 1)
    assert loads(dumps(result)) == result


@pytest.mark.parametrize("target", ["gfx90a", "gfx942", "gfx950"])
@pytest.mark.parametrize(("opcode", "width", "spacing"), _CASES)
def test_dual_opcodes_do_not_expand_gfx9(target, opcode, width, spacing):
    with pytest.raises(LdsPredictionError, match=f"not supported by {target}"):
        _predict(opcode, _accesses(width, spacing), target, 64)


def test_single_address_contract_and_v1_documents_stay_strict():
    access = LdsAccess(0, 0, 0, 4)
    assert "operand_index" not in access.as_dict()
    assert LdsAccess.from_dict(access.as_dict()) == access
    with pytest.raises(LdsPredictionError, match="no address operand 1"):
        _predict("ds_load_b32", [replace(access, operand_index=1)])
    with pytest.raises(ModelValidationError, match="operand_index"):
        replace(access, operand_index=True)
    with pytest.raises(ModelValidationError, match="operand_index"):
        replace(access, operand_index=2)
    result = _predict("ds_load_b32", [access])
    assert result.schema_version == 1
    assert loads(dumps(result)) == result
    document = result.as_dict()
    document["accesses"][0]["operand_index"] = 0
    with pytest.raises(ModelValidationError, match="does not encode operand_index"):
        loads(json.dumps(document))
    dual = _predict("ds_load_2addr_b32", _accesses(4, 64))
    with pytest.raises(ModelValidationError, match="requires schema_version 2"):
        replace(dual, schema_version=1)
    cross_group = replace(dual.conflict_groups[0], access_ids=(0, 1))
    changed = (
        replace(dual.accesses[0], conflict_group_ids=(0,)),
        replace(dual.accesses[1], conflict_group_ids=(0,)),
        replace(dual.accesses[2], conflict_group_ids=(1,)),
        replace(dual.accesses[3], conflict_group_ids=(1,)),
    )
    groups = (cross_group, replace(dual.conflict_groups[1], access_ids=(2, 3)))
    with pytest.raises(ModelValidationError, match="one address operand"):
        replace(dual, accesses=changed, conflict_groups=groups)


def test_cli_serializes_operands_and_rejects_unsupported_spacing():
    script = Path(__file__).with_name("lds-bank-conflict-expert") / "predict.py"
    if not script.is_file():
        script = (
            dsl_docs_dir()
            / "optimization/utilities/skills/lds-bank-conflict-expert/scripts/predict.py"
        )
    env = dict(os.environ, PYTHONPATH=str(Path(rocke.__file__).resolve().parent.parent))
    request = dict(
        target="gfx1201",
        opcode="ds_load_2addr_b32",
        wave_size=32,
        accesses=[a.as_dict() for a in _accesses(4, 64)],
    )
    run = subprocess.run(
        [sys.executable, str(script), "-"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    result = loads(run.stdout)
    assert result.schema_version == 2
    assert [g.access_ids for g in result.conflict_groups] == [(0, 2), (1, 3)]
    request["accesses"][1]["lds_byte_address"] = 68
    run = subprocess.run(
        [sys.executable, str(script), "-"],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=env,
    )
    assert run.returncode != 0
    assert "operand spacing" in run.stderr
