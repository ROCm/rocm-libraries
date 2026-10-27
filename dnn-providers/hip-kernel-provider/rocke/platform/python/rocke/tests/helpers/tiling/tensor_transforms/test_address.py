# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the address functions, the golden every emitted address is compared with: a plain
window's graph gives the plain strided address, the conv graph gives numpy's NHWC element
offsets, and the rank, stride and positional errors fire with their text (no GPU).
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest

from rocke.core.ir import F16, F32
from rocke.helpers.tiling import tensor_transforms as tt
from rocke.helpers.tiling.descriptors import TensorDesc
from rocke.helpers.tiling.tensor_transforms import (
    address_of,
    embed,
    inputs,
    make_transform_graph,
    offset,
    outputs,
    transform_address_of,
    unmerge,
)
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph


def _raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


def _numpy_strides(lengths: tuple[int, ...], order: str = "C") -> tuple[int, ...]:
    """Element strides of a numpy array of these lengths (not bytes)."""
    arr = np.empty(lengths, dtype=np.float32, order=order)
    return tuple(s // arr.itemsize for s in arr.strides)


def test_the_package_door_exports_both_address_functions():
    assert tt.address_of is address_of
    assert tt.transform_address_of is transform_address_of


# --- a plain window: the identity graph is the plain strided address ---------------------------


@pytest.mark.parametrize(
    "lengths, strides",
    [
        ((4, 8), _numpy_strides((4, 8))),  # row-major
        ((4, 8), _numpy_strides((4, 8), "F")),  # column-major
        ((3, 5), (16, 1)),  # padded leading dimension
        ((2, 3, 4), _numpy_strides((2, 3, 4))),
        ((2, 3, 4), (1, 2, 6)),  # permuted memory order
        ((5,), (3,)),  # rank 1, non-unit stride
        ((2, 3), (0, 1)),  # a broadcast axis (stride 0) reads one row for every index
    ],
)
def test_the_identity_graph_gives_the_plain_strided_address(lengths, strides):
    tensor = TensorDesc(lengths, strides, F16)
    graph = identity_graph(lengths)
    for position in itertools.product(*(range(n) for n in lengths)):
        plain = sum(p * s for p, s in zip(position, strides))
        assert address_of(tensor, position) == plain
        assert transform_address_of(graph, tensor, position) == plain


def test_address_of_matches_numpy_element_offsets_of_a_strided_view():
    """A view with a step on each axis, starting one element in: each view element holds its
    flat index in the parent, so ``1 + address_of`` over the view's element strides must equal
    the value at that position."""
    parent = np.arange(60, dtype=np.int64).reshape(6, 10)
    view = parent[::2, 1::3]  # element strides (20, 3)
    tensor = TensorDesc(
        view.shape, tuple(s // view.itemsize for s in view.strides), F32
    )
    assert tensor.strides == (20, 3)
    for position in itertools.product(*(range(n) for n in view.shape)):
        assert 1 + address_of(tensor, position) == view[position]


def test_address_of_does_not_bound_the_position():
    """The golden answers which element a lane would address, valid or not: positions past a
    length, or negative ones (a halo), are computed, not refused."""
    tensor = TensorDesc((4, 8), (8, 1), F16)
    assert address_of(tensor, (4, 0)) == 32
    assert address_of(tensor, (-1, 3)) == -5


# --- the conv graph against numpy NHWC offsets -------------------------------------------------


@pytest.mark.parametrize(
    "N, Hi, Wi, C, Y, X, s, d, p",
    [
        (2, 5, 4, 3, 3, 3, 1, 1, 1),  # same-size conv, halo both sides
        (1, 7, 6, 2, 3, 2, 2, 1, 1),  # stride 2, rectangular filter
        (2, 6, 6, 4, 2, 2, 1, 2, 0),  # dilation 2, no pad
    ],
)
def test_the_conv_graph_gives_numpy_nhwc_offsets(N, Hi, Wi, C, Y, X, s, d, p):
    Ho = (Hi + 2 * p - d * (Y - 1) - 1) // s + 1
    Wo = (Wi + 2 * p - d * (X - 1) - 1) // s + 1
    graph = make_transform_graph(
        inputs(m=N * Ho * Wo, k=Y * X * C),
        unmerge("m", into=("n", "ho", "wo"), lengths=(N, Ho, Wo)),
        unmerge("k", into=("y", "x", "c"), lengths=(Y, X, C)),
        embed(("ho", "y"), into="h_dil", coeffs=(s, d)),
        offset("h_dil", into="hi", shift=-p),
        embed(("wo", "x"), into="w_dil", coeffs=(s, d)),
        offset("w_dil", into="wi", shift=-p),
        outputs("n", "hi", "wi", "c"),
    )
    image = np.arange(N * Hi * Wi * C).reshape(
        N, Hi, Wi, C
    )  # value == NHWC element offset
    tensor = TensorDesc((N, Hi, Wi, C), _numpy_strides((N, Hi, Wi, C)), F16)
    halo_taps = 0
    for m in range(N * Ho * Wo):
        n, ho, wo = np.unravel_index(m, (N, Ho, Wo))
        for k in range(Y * X * C):
            y, x, c = np.unravel_index(k, (Y, X, C))
            h, w = ho * s + y * d - p, wo * s + x * d - p
            got = transform_address_of(graph, tensor, (m, k))
            if 0 <= h < Hi and 0 <= w < Wi:
                assert got == image[n, h, w, c]
            else:
                # A halo tap still has an address (what the lane computes before its check);
                # it is the same strided sum, outside the image.
                halo_taps += 1
                assert got == ((n * Hi + h) * Wi + w) * C + c
    assert (halo_taps > 0) == (p > 0)


# --- errors --------------------------------------------------------------------------------------


def test_a_position_of_the_wrong_length_raises_the_rank_message():
    tensor = TensorDesc((4, 8), (8, 1), F16)
    with _raises(
        ValueError,
        "address_of got 3 coordinates but the tensor has rank 2 -- give one int per tensor "
        "axis, in axis order",
    ):
        address_of(tensor, (0, 0, 0))


def test_a_graph_whose_outputs_do_not_match_the_tensor_rank_raises():
    tensor = TensorDesc((4, 8, 2), (16, 2, 1), F16)
    with _raises(
        ValueError, "graph has 2 outputs ('axis0', 'axis1') but the tensor has rank 3"
    ):
        transform_address_of(identity_graph((4, 8)), tensor, (0, 0))


def test_a_negative_stride_raises_the_stride_message():
    tensor = TensorDesc((4, 8), (8, -1), F16)
    with _raises(
        ValueError,
        "structured address needs non-negative strides; tensor axis 1 has stride -1",
    ):
        address_of(tensor, (0, 0))
    with _raises(
        ValueError,
        "structured address needs non-negative strides; tensor axis 1 has stride -1",
    ):
        transform_address_of(identity_graph((4, 8)), tensor, (0, 0))


@pytest.mark.parametrize(
    "position", [{"axis0": 1, "axis1": 2}, [1, 2], np.array([1, 2])]
)
def test_address_of_refuses_anything_but_a_tuple(position):
    tensor = TensorDesc((4, 8), (8, 1), F16)
    with _raises(
        TypeError,
        "address_of takes the position as a tuple in tensor-axis order, got "
        f"{type(position).__name__} -- coordinates are positional",
    ):
        address_of(tensor, position)


@pytest.mark.parametrize("coords", [{"m": 1, "k": 2}, [1, 2]])
def test_transform_address_of_refuses_anything_but_a_tuple(coords):
    tensor = TensorDesc((4, 8), (8, 1), F16)
    graph = make_transform_graph(
        inputs(m=4, k=8),
        embed(("m",), into="r", coeffs=(1,)),
        embed(("k",), into="c", coeffs=(1,)),
        outputs("r", "c"),
    )
    with _raises(
        TypeError,
        "transform_address_of takes the coordinates as a tuple in inputs(...) order, got "
        f"{type(coords).__name__} -- coordinates are positional",
    ):
        transform_address_of(graph, tensor, coords)


def test_transform_address_of_with_the_wrong_input_count_raises_the_count_message():
    tensor = TensorDesc((4, 8), (8, 1), F16)
    with _raises(
        ValueError,
        "transform_address_of got 1 coordinates, the graph has 2 inputs ('axis0', 'axis1') -- "
        "give one int per input, in inputs(...) order",
    ):
        transform_address_of(identity_graph((4, 8)), tensor, (0,))
