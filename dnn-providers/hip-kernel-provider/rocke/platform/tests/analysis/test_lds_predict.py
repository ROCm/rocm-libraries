# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-instruction input validation across registered LDS profiles."""

import pytest

from rocke.analysis.lds import LdsAccess, LdsPredictionError, predict_lds_conflicts
from rocke.analysis.lds.opcodes import get_opcode_spec
from rocke.analysis.lds.registry import registered_targets, resolve_profile


@pytest.mark.parametrize(
    ("target", "opcode"),
    [
        (target, opcode)
        for target in registered_targets()
        for opcode in sorted(resolve_profile(target).supported_opcodes)
    ],
)
@pytest.mark.parametrize("address", [0, 128])
@pytest.mark.parametrize("active", [True, False])
def test_repeated_lane_requires_inactive_access(target, opcode, address, active):
    width = get_opcode_spec(opcode).access_width_bytes
    accesses = [
        LdsAccess(0, 0, 0, width),
        LdsAccess(1, 0, address, width, active=active),
    ]
    request = dict(target=target, opcode=opcode, wave_size=64, accesses=accesses)

    if active:
        with pytest.raises(LdsPredictionError, match="one active access per lane"):
            predict_lds_conflicts(**request)
    else:
        result = predict_lds_conflicts(**request)
        assert [access.classification.value for access in result.accesses] == [
            "normal",
            "inactive",
        ]
        assert not result.conflict_groups


@pytest.mark.parametrize("target", registered_targets())
@pytest.mark.parametrize("width", [4, 8, 16])
@pytest.mark.parametrize("direction", ["read", "write"])
def test_profile_capacity_and_alignment_boundaries(target, width, direction):
    capacity = {"gfx90a": 65536, "gfx942": 65536, "gfx950": 163840}[target]
    assert resolve_profile(target).lds_capacity_bytes == capacity

    def predict(address):
        return predict_lds_conflicts(
            target=target,
            opcode=f"ds_{direction}_b{width * 8}",
            wave_size=64,
            accesses=[LdsAccess(0, 0, address, width)],
        )

    result = predict(capacity - width)
    assert result.accesses[0].lds_byte_address == capacity - width
    assert not result.conflict_groups
    with pytest.raises(LdsPredictionError, match="exceeds.*LDS capacity"):
        predict(capacity)
    with pytest.raises(LdsPredictionError, match=f"{width}-byte aligned"):
        predict(1 if width == 4 else 4)
    if width > 4:
        with pytest.raises(LdsPredictionError, match="exceeds.*LDS capacity"):
            predict(capacity - 4)
