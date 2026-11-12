# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only golden cases for the explicit gfx942 LDS profile.

Cases that gfx942 and gfx950 share are in `test_lds_profile_cdna_common.py`.
"""

import pytest
from rocke.analysis.lds.model import LdsAccess
from rocke.analysis.lds.predict import predict_lds_conflicts
from rocke.analysis.lds.registry import resolve_profile
from rocke.tests.arch_marks import arch_specific


def _access(access_id: int, lane: int, address: int, *, width: int = 4) -> LdsAccess:
    return LdsAccess(
        access_id=access_id,
        lane=lane,
        lds_byte_address=address,
        access_width_bytes=width,
    )


def _predict(target: str, opcode: str, accesses: list[LdsAccess]):
    return predict_lds_conflicts(
        target=target, opcode=opcode, wave_size=64, accesses=accesses
    )


_gfx942_profile = arch_specific(
    ("gfx942",), reason="golden cases for the explicit gfx942 LDS profile"
)


@_gfx942_profile
def test_gfx942_has_distinct_profile_identity():
    gfx90a = resolve_profile("gfx90a")
    gfx942 = resolve_profile("gfx942")

    assert gfx942 is not gfx90a
    assert gfx942.identity.target == "gfx942"
    assert gfx942.identity != gfx90a.identity


@_gfx942_profile
@pytest.mark.parametrize(
    ("opcode", "width"),
    [
        ("ds_read_b32", 4),
        ("ds_write_b32", 4),
        ("ds_read_b64", 8),
        ("ds_write_b64", 8),
        ("ds_read_b128", 16),
        ("ds_write_b128", 16),
    ],
)
def test_reviewed_scope_agrees_with_gfx90a(opcode, width):
    accesses = [_access(0, 0, 0, width=width), _access(1, 1, 128, width=width)]

    gfx942 = _predict("gfx942", opcode, accesses)
    gfx90a = _predict("gfx90a", opcode, accesses)

    assert gfx942.profile.target == "gfx942"
    assert gfx90a.profile.target == "gfx90a"
    assert gfx942.request == gfx90a.request
    assert gfx942.accesses == gfx90a.accesses
    assert gfx942.conflict_groups == gfx90a.conflict_groups
    assert gfx942.summary == gfx90a.summary
