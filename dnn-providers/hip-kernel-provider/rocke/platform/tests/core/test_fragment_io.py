# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
from rocke.core.ir import F16
from rocke.helpers.io import iter_global_fragment_scalars, load_clipped_global_fragment


@pytest.mark.parametrize(
    "n, mask, match",
    [(0, None, "positive width"), (1, object(), "needs a pad value")],
)
def test_fragment_rejects_invalid_width_or_mask_before_emitting(n, mask, match):
    with pytest.raises(ValueError, match=match):
        list(
            iter_global_fragment_scalars(
                None, None, None, dtype=F16, n=n, align=1, mask=mask
            )
        )


def test_clipped_load_rejects_empty_fragment_before_reading():
    def unexpected(*_):
        pytest.fail("empty fragment issued a read")

    with pytest.raises(ValueError, match="fragment needs positive width"):
        load_clipped_global_fragment(
            None, None, None, None, dtype=F16, n=0, read=unexpected
        )
