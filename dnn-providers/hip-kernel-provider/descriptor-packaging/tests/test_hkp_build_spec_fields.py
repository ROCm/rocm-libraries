# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""build_spec over tuple and multi-arm union spec fields (stub dataclasses, no rocKE)."""

from __future__ import annotations

import dataclasses
from typing import Literal

import pytest
from hkp_pack.errors import HkpPackError
from hkp_pack.rocke_compile import build_spec


@dataclasses.dataclass
class _Leaf:
    x: int


@dataclasses.dataclass
class _Spec:
    grid: tuple[int, int] | None = None
    nested: tuple[tuple[int, int], ...] | None = None
    alloc: tuple[int, int] | str | None = None
    noalias: bool | str = False
    mode: Literal["a", "b"] | int = "a"
    leaf_or_name: _Leaf | str = "n"
    scale: float | str = 1.0


@dataclasses.dataclass
class _Ambiguous:
    v: int | float = 0


@dataclasses.dataclass
class _Lists:
    items: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Untyped:
    items: tuple = ()


@pytest.mark.quick
def test_tuple_from_list():
    spec = build_spec(_Spec, {"grid": [2, 4]})
    assert spec.grid == (2, 4) and isinstance(spec.grid, tuple)


@pytest.mark.quick
def test_nested_variadic_tuple():
    spec = build_spec(_Spec, {"nested": [[1, 2], [3, 4], [5, 6]]})
    assert spec.nested == ((1, 2), (3, 4), (5, 6))
    assert all(isinstance(t, tuple) for t in spec.nested)


@pytest.mark.quick
def test_optional_tuple_none():
    assert build_spec(_Spec, {"grid": None, "alloc": None}).grid is None


@pytest.mark.quick
def test_tuple_arity_and_shape_refused():
    with pytest.raises(HkpPackError, match="needs 2 items"):
        build_spec(_Spec, {"grid": [1, 2, 3]})
    with pytest.raises(HkpPackError, match="needs a JSON list"):
        build_spec(_Spec, {"nested": [1, 2]})


@pytest.mark.quick
def test_union_arm_by_type():
    assert build_spec(_Spec, {"alloc": [64, 192]}).alloc == (64, 192)
    assert build_spec(_Spec, {"alloc": "auto"}).alloc == "auto"
    assert build_spec(_Spec, {"noalias": True}).noalias is True
    assert build_spec(_Spec, {"noalias": "all"}).noalias == "all"
    assert build_spec(_Spec, {"mode": "b"}).mode == "b"
    assert build_spec(_Spec, {"mode": 3}).mode == 3
    assert build_spec(_Spec, {"leaf_or_name": {"x": 7}}).leaf_or_name == _Leaf(7)
    assert build_spec(_Spec, {"scale": 2}).scale == 2


@pytest.mark.quick
def test_union_without_a_single_arm_refused():
    with pytest.raises(HkpPackError, match="matches 0 arms"):
        build_spec(_Spec, {"alloc": 5})
    with pytest.raises(HkpPackError, match="matches 0 arms"):
        build_spec(_Spec, {"mode": "c"})
    with pytest.raises(HkpPackError, match="matches 0 arms"):
        build_spec(_Spec, {"noalias": 1})
    with pytest.raises(HkpPackError, match="matches 2 arms"):
        build_spec(_Ambiguous, {"v": 1})


@pytest.mark.quick
def test_list_and_untyped_tuple_still_refused():
    with pytest.raises(HkpPackError, match="unsupported spec field type"):
        build_spec(_Lists, {"items": [1, 2]})
    with pytest.raises(HkpPackError, match="unsupported spec field type"):
        build_spec(_Untyped, {"items": [1, 2]})
