# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.KernelWriterAssembly import _enableLdsTr6Rearrange

pytestmark = pytest.mark.unit


def _kernel(*, tdm_a=False, tdm_b=False, lds_tr_a=True, lds_tr_b=True):
    return {
        "enableTDMA": tdm_a,
        "enableTDMB": tdm_b,
        "enableLDSTrA": lds_tr_a,
        "enableLDSTrB": lds_tr_b,
    }


@pytest.mark.parametrize(
    "tail,tdm_a,tdm_b,expected",
    [
        (False, False, False, True),  # Main loop always rearranges tr6 output.
        (True, False, False, False),  # Non-TDM tail does it in shiftK.
        (True, True, False, True),
        (True, False, True, True),
        (True, True, True, True),
    ],
)
@pytest.mark.parametrize("tensor_char", ["A", "B"])
def test_tdm_tail_reuses_main_loop_tr6_rearrangement(
    tensor_char, tail, tdm_a, tdm_b, expected
):
    kernel = _kernel(tdm_a=tdm_a, tdm_b=tdm_b)
    tensor = {"tensorChar": tensor_char, "bpe": 0.75}

    assert _enableLdsTr6Rearrange(kernel, tensor, tail) is expected


@pytest.mark.parametrize("tensor_char", ["A", "B"])
@pytest.mark.parametrize("bpe,enable_lds_tr", [(2, True), (0.75, False)])
def test_rearrangement_requires_6bit_lds_transpose(tensor_char, bpe, enable_lds_tr):
    kernel = _kernel(
        lds_tr_a=enable_lds_tr if tensor_char == "A" else True,
        lds_tr_b=enable_lds_tr if tensor_char == "B" else True,
    )
    tensor = {"tensorChar": tensor_char, "bpe": bpe}

    assert _enableLdsTr6Rearrange(kernel, tensor, tail=False) is False
