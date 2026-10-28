# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Analytic CPU checks for shared decode normalization and feature ordering."""

import math

import pytest

from tests._decode_reference import _reference, _feature_reference, reference


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("window", [0, 1, 2])
def test_equal_scores_gqa_and_window(torch, window):
    q = torch.zeros(1, 4, 2)
    k = torch.zeros(1, 2, 3, 2)
    v = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            ]
        ]
    )
    begin = 0 if window == 0 else 3 - window
    expected = v[:, :, begin:].mean(2).repeat_interleave(2, dim=1)
    torch.testing.assert_close(_reference(torch, q, k, v, [3], window), expected)
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    kp, vp = k.permute(0, 2, 1, 3).reshape(3, 1, 2, 2), v.permute(0, 2, 1, 3).reshape(
        3, 1, 2, 2
    )
    kp, vp = kp[torch.tensor([1, 2, 0])], vp[torch.tensor([1, 2, 0])]
    output, lse = reference(torch, q, kp, vp, table, [3], 1, window)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(lse, torch.full((1, 4, 1), math.log(3 - begin)))


def test_empty_row_skips_poisoned_page_lookup(torch):
    q = torch.ones(1, 2, 2)
    output, lse = reference(
        torch,
        q,
        torch.empty(0, 1, 1, 2),
        torch.empty(0, 1, 1, 2),
        torch.tensor([[-1]]),
        [0],
        1,
        0,
    )
    assert torch.count_nonzero(output) == 0
    assert torch.isneginf(lse).all()


@pytest.mark.parametrize("sink", [-1.0, 0.0, 2.0])
def test_zero_valued_sink_normalization(torch, sink):
    q, k, v = torch.zeros(1, 2, 2), torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)
    features = dict(
        softcap=0, alibi_slopes=None, qq_bias=None, sinks=torch.full((2,), sink)
    )
    output = _feature_reference(torch, q, k, v, [2], 0, features)
    torch.testing.assert_close(output, torch.full((1, 2, 2), 2 / (2 + math.exp(sink))))


def test_softcap_precedes_position_and_last_key_bias(torch):
    q, k = torch.tensor([[[2.0]]]), torch.tensor([[[[0.0], [2.0], [4.0]]]])
    v = torch.tensor([[[[1.0], [0.0], [0.0]]]])
    features = dict(
        softcap=1.0,
        alibi_slopes=torch.tensor([0.5]),
        qq_bias=torch.tensor([[0.75]]),
        sinks=None,
    )
    output = _feature_reference(torch, q, k, v, [3], 0, features)
    scores = torch.tensor([0.0, math.tanh(4.0), math.tanh(8.0)]) + torch.tensor(
        [-1.0, -0.5, 0.75]
    )
    torch.testing.assert_close(output[0, 0, 0], scores.softmax(0)[0])
