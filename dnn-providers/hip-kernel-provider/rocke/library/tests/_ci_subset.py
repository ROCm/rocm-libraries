# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run a CI subset of a parametrized GPU numeric matrix; the rest on request.

Each kernel case compiles its own binary, so a full dtype x head size x heads
cross-product costs CI minutes for checks that mostly repeat. ``@ci_subset``
keeps every combination of the ``cover`` parameters and runs each of them with
one value of every ``rotate`` parameter, picked in turn, so each value still runs
somewhere. The cases it drops skip with a reason, and they run again when
``ROCKE_TEST_FULL_MATRIX=1`` is set::

    ROCKE_TEST_FULL_MATRIX=1 python -m pytest tests/test_attention_unified_lse_numeric.py

The values come from the test's own ``pytest.mark.parametrize`` marks, so put
``@ci_subset`` above them.
"""

from __future__ import annotations

import functools
import itertools
import os

import pytest

FULL_MATRIX_ENV = "ROCKE_TEST_FULL_MATRIX"

_SKIP_REASON = f"outside the CI subset; set {FULL_MATRIX_ENV}=1 to run it"


def full_matrix() -> bool:
    """True when ``ROCKE_TEST_FULL_MATRIX`` asks for every case."""
    return os.environ.get(FULL_MATRIX_ENV, "").strip() not in ("", "0")


def _choices(fn, argnames: str) -> list:
    """The values of the ``parametrize`` mark on ``fn`` whose argnames are ``argnames``."""
    for mark in getattr(fn, "pytestmark", []):
        if mark.name == "parametrize" and mark.args[0] == argnames:
            multi = "," in argnames
            out = []
            for value in mark.args[1]:
                if isinstance(value, type(pytest.param(None))):
                    value = value.values if multi else value.values[0]
                out.append(tuple(value) if multi else value)
            return out
    raise LookupError(f"{fn.__name__} has no parametrize mark for {argnames!r}")


def _value(kwargs, argnames: str):
    names = [n.strip() for n in argnames.split(",")]
    return tuple(kwargs[n] for n in names) if len(names) > 1 else kwargs[names[0]]


def ci_subset(*, cover=(), rotate=(), keep=None):
    """Skip the cases of the decorated test that are outside its CI subset.

    ``cover`` and ``rotate`` name ``parametrize`` argnames as written in the
    mark (``"q_lens,kv_lens"`` for a joint one). Combination ``k`` of the
    ``cover`` values runs with value ``(k + j) % n`` of the ``j``-th ``rotate``
    parameter. ``keep(**params)``, when given, is checked instead.
    """

    def decorate(fn):
        if keep is None:
            cover_choices = [_choices(fn, a) for a in cover]
            rotate_choices = [_choices(fn, a) for a in rotate]
            combos = list(itertools.product(*cover_choices))
            for argnames, choices in zip(rotate, rotate_choices):
                assert len(combos) >= len(choices), (
                    f"{fn.__name__}: {len(combos)} cover combinations cannot "
                    f"rotate all {len(choices)} values of {argnames!r}"
                )

        def kept(kwargs) -> bool:
            if keep is not None:
                return bool(keep(**kwargs))
            k = combos.index(tuple(_value(kwargs, a) for a in cover))
            return all(
                _value(kwargs, a) == choices[(k + j) % len(choices)]
                for j, (a, choices) in enumerate(zip(rotate, rotate_choices))
            )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not full_matrix() and not kept(kwargs):
                pytest.skip(_SKIP_REASON)
            return fn(*args, **kwargs)

        wrapper.ci_kept = kept
        return wrapper

    return decorate
