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
