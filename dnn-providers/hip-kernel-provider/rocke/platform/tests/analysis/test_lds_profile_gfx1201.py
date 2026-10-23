# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden phase partitions and request boundaries for gfx1201 wave32."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import rocke
from rocke.analysis.lds import LdsAccess, LdsPredictionError, predict_lds_conflicts
from rocke.analysis.lds import opcodes
from rocke.analysis.lds.model import GroupKind
from rocke.analysis.lds.profiles import Gfx1201Profile
from rocke.analysis.lds.registry import UnsupportedLdsTargetError, resolve_profile
from rocke.analysis.lds.serialization import dumps, loads
from rocke.assets import dsl_docs_dir

# Independent, explicit expected memberships rather than the profile's bit rule.
_ALL = (tuple(range(32)),)
_HALVES = (tuple(range(16)), tuple(range(16, 32)))
_QUARTERS = tuple(tuple(range(start, start + 8)) for start in (0, 8, 16, 24))
_CASES = [
    ("ds_load_b32", 4, 4, _ALL),
    ("ds_load_b64", 8, 8, _HALVES),
    ("ds_load_b96", 12, 16, _QUARTERS),
    ("ds_load_b128", 16, 16, _QUARTERS),
    ("ds_store_b32", 4, 4, _ALL),
    ("ds_store_b64", 8, 8, _HALVES),
    ("ds_store_b96", 12, 16, _QUARTERS),
    ("ds_store_b128", 16, 16, _QUARTERS),
    ("ds_load_u8", 1, 4, _ALL),
    ("ds_load_i8", 1, 4, _ALL),
    ("ds_load_u16", 2, 4, _ALL),
    ("ds_load_i16", 2, 4, _ALL),
    ("ds_store_b8", 1, 4, _ALL),
    ("ds_store_b16", 2, 4, _ALL),
]


def _predict(opcode: str, width: int, lanes=(0, 1), addresses=(0, 128)):
    return predict_lds_conflicts(
        target="gfx1201",
        opcode=opcode,
        wave_size=32,
        accesses=[
            LdsAccess(i, lane, address, width)
            for i, (lane, address) in enumerate(zip(lanes, addresses))
        ],
    )


@pytest.mark.parametrize(("opcode", "width", "alignment", "groups"), _CASES)
def test_every_ordered_lane_pair_matches_golden_partition(
    opcode, width, alignment, groups
):
    for lane_a in range(32):
        for lane_b in range(32):
            if lane_a == lane_b:
                continue
            expected = any(lane_a in group and lane_b in group for group in groups)
            result = _predict(opcode, width, (lane_a, lane_b))
            assert bool(result.conflict_groups) is expected
            assert [a.classification.value for a in result.accesses] == (
                ["conflict", "conflict"] if expected else ["normal", "normal"]
            )
            if expected:
                assert result.conflict_groups[0].access_ids == (0, 1)
                assert result.conflict_groups[0].multiplicity == 2
                assert (
                    result.conflict_groups[0].kind
                    is GroupKind.DISTINCT_ADDRESS_CONFLICT
                )


@pytest.mark.parametrize(("opcode", "width", "alignment", "groups"), _CASES)
def test_aligned_address_sweep(opcode, width, alignment, groups):
    for distance in range(alignment, 769, alignment):
        result = _predict(opcode, width, addresses=(0, distance))
        assert bool(result.conflict_groups) is (distance % 128 == 0)


@pytest.mark.parametrize("width", [4, 8, 16])
@pytest.mark.parametrize(
    ("canonical", "native"), [("read", "load"), ("write", "store")]
)
def test_legacy_and_native_spellings_have_the_same_groups(width, canonical, native):
    legacy = _predict(f"ds_{canonical}_b{width * 8}", width)
    result = _predict(f"  DS_{native.upper()}_B{width * 8}  ", width)
    assert result.profile.target == "gfx1201"
    assert result.profile.profile_version == 1
    assert result.request.opcode == f"ds_{native}_b{width * 8}"
    assert legacy.accesses == result.accesses
    assert legacy.conflict_groups == result.conflict_groups
    assert loads(dumps(result)) == result


@pytest.mark.parametrize(("opcode", "width", "alignment", "groups"), _CASES)
def test_scope_rejects_same_address_and_multiple_active_lanes(
    opcode, width, alignment, groups
):
    with pytest.raises(LdsPredictionError, match="distinct, non-overlapping"):
        _predict(opcode, width, addresses=(0, 0))
    with pytest.raises(LdsPredictionError, match="at most two active lanes"):
        _predict(opcode, width, lanes=(0, 1, 2), addresses=(0, 128, 256))


@pytest.mark.parametrize(("opcode", "width", "alignment", "groups"), _CASES)
def test_scope_alignment_and_capacity(opcode, width, alignment, groups):
    with pytest.raises(LdsPredictionError, match=f"{alignment}-byte aligned"):
        _predict(opcode, width, lanes=(0,), addresses=(alignment // 2,))
    with pytest.raises(LdsPredictionError, match="exceeds.*LDS capacity"):
        _predict(opcode, width, lanes=(0,), addresses=(65536,))
    last = ((65536 - width) // alignment) * alignment
    result = _predict(opcode, width, lanes=(0,), addresses=(last,))
    assert result.accesses[0].lds_byte_address == last


def test_inactive_accesses_do_not_expand_pair_scope():
    result = predict_lds_conflicts(
        target="gfx1201",
        opcode="ds_load_b32",
        wave_size=32,
        accesses=[
            LdsAccess(0, 0, 0, 4),
            LdsAccess(1, 1, 128, 4),
            LdsAccess(2, 2, 0, 4, active=False),
        ],
    )
    assert [a.classification.value for a in result.accesses] == [
        "conflict",
        "conflict",
        "inactive",
    ]
    assert result.request.active_lanes == (0, 1)


@pytest.mark.parametrize("wave_size", [16, 64])
def test_only_wave32_is_admitted(wave_size):
    with pytest.raises(LdsPredictionError, match="unsupported wave_size"):
        predict_lds_conflicts(
            target="gfx1201", opcode="ds_load_b32", wave_size=wave_size, accesses=[]
        )


@pytest.mark.parametrize("target", ["gfx1200", "gfx12-generic", "GFX1201"])
def test_no_architecture_aliases(target):
    with pytest.raises(UnsupportedLdsTargetError):
        resolve_profile(target)


def test_opcode_growth_does_not_expand_profile_support(monkeypatch):
    opcode = "ds_load_test_b64"
    monkeypatch.setitem(
        opcodes._OPCODE_SPECS, opcode, opcodes.OpcodeSpec(opcode, "read", 8)
    )
    with pytest.raises(LdsPredictionError, match="not supported by gfx1201"):
        _predict(opcode, 8)
    for method in ("phase_key", "collision_key"):
        with pytest.raises(ValueError, match="not supported by gfx1201"):
            getattr(Gfx1201Profile(), method)(opcode, 0)


@pytest.mark.parametrize("target", ["gfx90a", "gfx942", "gfx950"])
def test_native_opcodes_do_not_expand_gfx9_support(target):
    with pytest.raises(LdsPredictionError, match=f"not supported by {target}"):
        predict_lds_conflicts(
            target=target, opcode="ds_load_b96", wave_size=64, accesses=[]
        )


@pytest.mark.parametrize("unsupported", ["ds_load_2addr_b32", "ds_load_tr16_b128"])
def test_unrepresented_opcodes_are_rejected(unsupported):
    with pytest.raises(opcodes.UnsupportedLdsOpcodeError):
        _predict(unsupported, 4)


def test_cli_preserves_wave32_result_and_rejects_broadcast():
    script = (
        Path(__file__).resolve().with_name("lds-bank-conflict-expert") / "predict.py"
    )
    if not script.is_file():
        script = (
            dsl_docs_dir()
            / "optimization"
            / "utilities"
            / "skills"
            / "lds-bank-conflict-expert"
            / "scripts"
            / "predict.py"
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(rocke.__file__).resolve().parent.parent)
    request = {
        "target": "gfx1201",
        "opcode": "ds_load_b96",
        "wave_size": 32,
        "accesses": [
            LdsAccess(0, 0, 0, 12).as_dict(),
            LdsAccess(1, 1, 128, 12).as_dict(),
        ],
    }
    run = subprocess.run(
        [sys.executable, str(script), "-"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    result = loads(run.stdout)
    assert result.request.wave_size == 32
    assert result.summary.maximum_multiplicity == 2
    request["accesses"][1]["lds_byte_address"] = 0
    run = subprocess.run(
        [sys.executable, str(script), "-"],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        env=env,
    )
    assert run.returncode != 0
    assert "distinct, non-overlapping" in run.stderr
