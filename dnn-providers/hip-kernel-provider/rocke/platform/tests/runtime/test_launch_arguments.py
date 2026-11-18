# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What a plain launch hands ``hipModuleLaunchKernel``, with HIP mocked out.

HIP reads the packed-args buffer of the ``extra`` path after the call returns,
so every launch needs a buffer of its own that the runtime keeps alive, and the
launch geometry reaches HIP through ctypes scalars that launches share. These
pin both. They need no GPU and no HIP library; the cluster path is covered in
``test_cluster_launch.py``.
"""

from __future__ import annotations

import ctypes
from unittest import mock

import pytest

from rocke.runtime import hip_module
from rocke.runtime.hip_module import Runtime

# A stream nothing else uses, so the class-level bucket can be inspected.
_STREAM = 0xA265


class _Hip:
    """Stands in for ``hipModuleLaunchKernel``: records the geometry of each
    call and reads the ``extra`` array while the call is in progress."""

    def __init__(self):
        self.calls = []

    def module_launch(self, fn, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        size = ctypes.c_size_t.from_address(extra[3]).value
        self.calls.append(
            {
                "geometry": (
                    (gx.value, gy.value, gz.value),
                    (bx.value, by.value, bz.value),
                    shared.value,
                    stream.value,
                ),
                "params": params,
                "markers": (extra[0], extra[2], extra[4]),
                "args_address": extra[1],
                "size_address": extra[3],
                "args": ctypes.string_at(extra[1], size),
            }
        )
        return 0


@pytest.fixture
def hip():
    fake = _Hip()
    with mock.patch.object(
        hip_module, "_hipModuleLaunchKernel", fake.module_launch
    ), mock.patch.object(
        hip_module, "_hipStreamSynchronize", return_value=0
    ), mock.patch.dict(
        Runtime._pending_args
    ), mock.patch.dict(
        Runtime._launch_scalars, clear=True
    ):
        yield fake


def test_each_launch_parks_the_buffer_hip_was_pointed_at(hip):
    packed = [b"\x01" * 8, b"\x02" * 16]
    for args in packed:
        Runtime().launch(object(), (4, 1, 1), (64, 1, 1), args, stream=_STREAM)

    assert [call["args"] for call in hip.calls] == packed
    assert [call["markers"] for call in hip.calls] == [(1, 2, 3)] * 2
    assert all(call["params"] is None for call in hip.calls)
    # The same geometry twice still gives each launch a buffer of its own, and
    # the parked objects are the ones whose addresses HIP holds.
    parked = [refs for refs, _event in Runtime._pending_args[_STREAM]]
    assert [bytes(refs[0]) for refs in parked] == packed
    assert [ctypes.addressof(refs[0]) for refs in parked] == [
        call["args_address"] for call in hip.calls
    ]
    assert [ctypes.addressof(refs[1]) for refs in parked] == [
        call["size_address"] for call in hip.calls
    ]


def test_blocking_launch_points_hip_at_the_packed_args(hip):
    Runtime().launch_blocking(
        object(), (2, 3, 1), (128, 1, 1), b"\x07" * 24, shared_bytes=64, stream=_STREAM
    )
    (call,) = hip.calls
    assert call["args"] == b"\x07" * 24
    assert call["markers"] == (1, 2, 3)
    assert call["geometry"] == ((2, 3, 1), (128, 1, 1), 64, _STREAM)
    assert _STREAM not in Runtime._pending_args


def test_a_launch_differing_in_one_argument_reaches_hip_with_it(hip):
    base = ((4, 2, 1), (64, 1, 1), 0, _STREAM)
    variants = [
        ((5, 2, 1), (64, 1, 1), 0, _STREAM),
        ((4, 3, 1), (64, 1, 1), 0, _STREAM),
        ((4, 2, 2), (64, 1, 1), 0, _STREAM),
        ((4, 2, 1), (32, 1, 1), 0, _STREAM),
        ((4, 2, 1), (64, 2, 1), 0, _STREAM),
        ((4, 2, 1), (64, 1, 2), 0, _STREAM),
        ((4, 2, 1), (64, 1, 1), 512, _STREAM),
        ((4, 2, 1), (64, 1, 1), 0, _STREAM + 1),
    ]
    # The base geometry goes first and again between variants, so a variant
    # that wrongly shared the base launch's scalars would show up as the base.
    order = [base]
    for variant in variants:
        order += [variant, base]
    for grid, block, shared, stream in order:
        Runtime().launch(
            object(), grid, block, b"\0" * 8, shared_bytes=shared, stream=stream
        )
    assert [call["geometry"] for call in hip.calls] == order


def test_launch_scalars_stay_bounded(hip):
    capacity = 4
    grids = [(n, 1, 1) for n in range(1, 3 * capacity)]
    with mock.patch.object(Runtime, "_LAUNCH_SCALARS_CAPACITY", capacity):
        for grid in grids:
            Runtime().launch(object(), grid, (64, 1, 1), b"\0" * 8, stream=_STREAM)
            assert len(Runtime._launch_scalars) <= capacity
    assert [call["geometry"][0] for call in hip.calls] == grids
