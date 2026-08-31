# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for custom-kernel name enumeration order."""

import pytest

from Tensile.CustomKernels import getAllCustomKernelNames

pytestmark = pytest.mark.unit


def _write_kernels(directory, names):
    for name in names:
        (directory / name).write_text("s_endpgm\n", encoding="utf-8")


def test_names_are_sorted_independently_of_listdir_order(tmp_path, monkeypatch):
    _write_kernels(tmp_path, ["b.s", "a.s", "c.s"])

    listdir_orders = [["b.s", "a.s", "c.s"], ["c.s", "b.s", "a.s"]]
    monkeypatch.setattr(
        "Tensile.CustomKernels.os.listdir", lambda directory: listdir_orders.pop(0)
    )

    assert getAllCustomKernelNames(directory=str(tmp_path)) == ["a", "b", "c"]
    assert getAllCustomKernelNames(directory=str(tmp_path)) == ["a", "b", "c"]


def test_names_drop_the_dot_s_suffix_and_ignore_other_files(tmp_path):
    _write_kernels(tmp_path, ["kernel.s", "notes.txt", "kernel.s.bak"])

    assert getAllCustomKernelNames(directory=str(tmp_path)) == ["kernel"]


def test_bundled_custom_kernel_names_are_sorted():
    names = getAllCustomKernelNames()

    assert names
    assert names == sorted(names)
