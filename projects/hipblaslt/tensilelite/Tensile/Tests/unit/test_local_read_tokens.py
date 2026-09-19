# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
from rocisa.container import MemTokenData

from Tensile.Component import Component

pytestmark = pytest.mark.unit

LocalRead = Component.LocalRead


def _writer():
    return SimpleNamespace(
        states=SimpleNamespace(
            ldsReadTokenIdx=0,
            memTokenLdsSplit=[[11, 12], [21, 22]],
        ),
        tdmSplitLdsBoundary=lambda kernel, tensor_parameters: 64,
    )


def _kernel():
    return {"TDMSplit": True, "ProblemType": {"Sparse": False}}


def _tensor_parameters():
    return {"isM": False, "localReadSwapByteOffset": 128}


def test_split_read_can_depend_on_both_half_tokens():
    token, index = LocalRead._getLdsReadMemToken(
        SimpleNamespace(),
        _writer(),
        _kernel(),
        _tensor_parameters(),
        ldsByteOffset=128,
        bothHalves=True,
    )

    assert token.tokens == [11, 12]
    assert index == 11


@pytest.mark.parametrize(("lds_byte_offset", "expected_token"), [(191, 11), (192, 12)])
def test_split_read_selects_token_from_lds_offset(lds_byte_offset, expected_token):
    token, index = LocalRead._getLdsReadMemToken(
        SimpleNamespace(),
        _writer(),
        _kernel(),
        _tensor_parameters(),
        ldsByteOffset=lds_byte_offset,
    )

    assert token.tokens == [expected_token]
    assert index == expected_token


def test_emitted_read_comment_lists_every_token():
    class Instruction:
        def __init__(self, **kwargs):
            self.comment = kwargs["comment"]
            self.token = None

        def setMemToken(self, token):
            self.token = token

    token = MemTokenData([4, 7])
    instructions = []
    local_read = SimpleNamespace(
        _getLdsReadMemToken=lambda *args: (token, token.tokens[0]),
    )

    LocalRead._emitLdsRead(
        local_read,
        None,
        None,
        None,
        Instruction,
        "dst",
        "src",
        "ds",
        SimpleNamespace(add=instructions.append),
        comment="read",
    )

    assert len(instructions) == 1
    assert instructions[0].comment == "read sync LDS4, sync LDS7"
    assert instructions[0].token is token
