# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
from rocke.core.ir import F16
from rocke.helpers.io import iter_global_fragment_scalars, load_clipped_global_fragment


@pytest.mark.parametrize(
    "n, clipping, match",
    [
        (0, {}, "positive width"),
        (1, {"mask": object(), "clip": (object(), object())}, "one clipping mode"),
        (1, {"mask": object()}, "needs a pad value"),
        (1, {"clip": (object(), object())}, "needs a pad value"),
    ],
)
def test_fragment_rejects_invalid_clipping_before_emitting(n, clipping, match):
    with pytest.raises(ValueError, match=match):
        list(
            iter_global_fragment_scalars(
                None, None, None, dtype=F16, n=n, align=1, **clipping
            )
        )


def test_clipped_load_rejects_empty_fragment_before_reading():
    def unexpected(*_):
        pytest.fail("empty fragment issued a read")

    with pytest.raises(ValueError, match="fragment needs positive width"):
        load_clipped_global_fragment(
            None, None, None, None, dtype=F16, n=0, read=unexpected
        )
