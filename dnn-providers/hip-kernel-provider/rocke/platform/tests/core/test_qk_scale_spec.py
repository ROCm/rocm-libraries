# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""QkScaleSpec rejects a bad layout or block size, naming the bad value.

The C++ engine's rocke_qk_scale_spec_validate reports the same text; the native
test rocke_qk_scale_spec_validate pins it.
"""

import pytest

from rocke.helpers.qk_scale import QkScaleSpec


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"layout": "per_tile"},
            r"layout must be 'per_head' or 'per_block', got 'per_tile'",
        ),
        (
            {"layout": "per_block", "scale_block": 0},
            r"scale_block must be > 0 \(got 0\)",
        ),
        (
            {"layout": "per_block", "scale_block": -4},
            r"scale_block must be > 0 \(got -4\)",
        ),
    ],
)
def test_qk_scale_spec_rejects_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        QkScaleSpec(**kwargs)


def test_qk_scale_spec_accepts_valid_layouts():
    QkScaleSpec(layout="per_head")
    QkScaleSpec(layout="per_block", scale_block=64)
