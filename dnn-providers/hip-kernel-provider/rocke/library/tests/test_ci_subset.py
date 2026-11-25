# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host contracts of ``@ci_subset``, which trims GPU numeric matrices in CI."""

from __future__ import annotations

import itertools

import pytest

from ._ci_subset import FULL_MATRIX_ENV, ci_subset


@ci_subset(cover=("shape",), rotate=("d", "dtype"))
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("shape", ["decode", "prefill", "ragged"])
def _probe(shape, d, dtype):
    return shape, d, dtype


@ci_subset(cover=("q_lens,kv_lens",), rotate=("dtype",))
@pytest.mark.parametrize("dtype", ["fp16", pytest.param("bf16", id="b")])
@pytest.mark.parametrize(
    "q_lens,kv_lens", [([1], [64]), pytest.param([5], [3], id="br")]
)
def _joint_probe(q_lens, kv_lens, dtype):
    return q_lens, kv_lens, dtype


_ALL = list(
    itertools.product(["decode", "prefill", "ragged"], [64, 128, 256], ["fp16", "bf16"])
)


def _kept(probe, cases, names):
    return [c for c in cases if probe.ci_kept(dict(zip(names, c)))]


def test_each_cover_value_runs_once_with_rotated_values():
    kept = _kept(_probe, _ALL, ("shape", "d", "dtype"))
    assert kept == [
        ("decode", 64, "bf16"),
        ("prefill", 128, "fp16"),
        ("ragged", 256, "bf16"),
    ]


def test_joint_argnames_and_pytest_param_values():
    cases = [
        ([1], [64], "fp16"),
        ([1], [64], "bf16"),
        ([5], [3], "fp16"),
        ([5], [3], "bf16"),
    ]
    kept = _kept(_joint_probe, cases, ("q_lens", "kv_lens", "dtype"))
    assert kept == [([1], [64], "fp16"), ([5], [3], "bf16")]


def test_case_outside_the_subset_skips(monkeypatch):
    monkeypatch.delenv(FULL_MATRIX_ENV, raising=False)
    assert _probe(shape="decode", d=64, dtype="bf16") == ("decode", 64, "bf16")
    with pytest.raises(pytest.skip.Exception, match=FULL_MATRIX_ENV):
        _probe(shape="decode", d=128, dtype="bf16")


@pytest.mark.parametrize("value", ["1", "yes"])
def test_full_matrix_runs_every_case(monkeypatch, value):
    monkeypatch.setenv(FULL_MATRIX_ENV, value)
    assert _probe(shape="decode", d=128, dtype="bf16") == ("decode", 128, "bf16")


def test_zero_does_not_turn_on_the_full_matrix(monkeypatch):
    monkeypatch.setenv(FULL_MATRIX_ENV, "0")
    with pytest.raises(pytest.skip.Exception):
        _probe(shape="decode", d=128, dtype="bf16")


def test_keep_callable_replaces_the_rotation(monkeypatch):
    monkeypatch.delenv(FULL_MATRIX_ENV, raising=False)

    @ci_subset(keep=lambda path, **_: path != "auto")
    @pytest.mark.parametrize("path", ["2d", "auto"])
    def probe(path):
        return path

    assert probe(path="2d") == "2d"
    with pytest.raises(pytest.skip.Exception):
        probe(path="auto")


def test_too_few_cover_combinations_to_rotate_every_value_is_an_error():
    with pytest.raises(AssertionError, match="cannot rotate all 3 values"):

        @ci_subset(cover=("dtype",), rotate=("d",))
        @pytest.mark.parametrize("d", [64, 128, 256])
        @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
        def probe(dtype, d):
            pass


def test_unknown_argnames_is_an_error():
    with pytest.raises(LookupError, match="'heads'"):

        @ci_subset(cover=("heads",))
        @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
        def probe(dtype):
            pass
