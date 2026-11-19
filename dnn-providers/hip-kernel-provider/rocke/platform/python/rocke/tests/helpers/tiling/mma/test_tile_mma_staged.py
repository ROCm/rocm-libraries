# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from rocke.helpers.tiling import TileMmaDriver


def test_staged_reads_once_and_keeps_per_atom_accumulators():
    events = []
    initial = [0, 1, 2, 3]

    def read_a(k, m):
        events.append(("a", k, m))
        return k, m

    def read_b(k, n):
        events.append(("b", k, n))
        return k, n

    def mma(a, b, acc):
        k, m = a
        assert k == b[0]
        n = b[1]
        events.append(("mma", k, m, n, acc))
        return acc * 10 + (k + 1) * (m + n + 1)

    out = TileMmaDriver.accumulate_staged(
        subtiles=(2, 2, 2),
        accs=initial,
        read_a=read_a,
        read_b=read_b,
        mma=mma,
        before_row=lambda: events.append("begin"),
        after_row=lambda: events.append("end"),
    )

    assert initial == [0, 1, 2, 3]
    assert out == [12, 124, 224, 336]
    assert events == [
        ("a", 0, 0),
        ("a", 0, 1),
        ("b", 0, 0),
        ("b", 0, 1),
        "begin",
        ("mma", 0, 0, 0, 0),
        ("mma", 0, 0, 1, 1),
        "end",
        "begin",
        ("mma", 0, 1, 0, 2),
        ("mma", 0, 1, 1, 3),
        "end",
        ("a", 1, 0),
        ("a", 1, 1),
        ("b", 1, 0),
        ("b", 1, 1),
        "begin",
        ("mma", 1, 0, 0, 1),
        ("mma", 1, 0, 1, 12),
        "end",
        "begin",
        ("mma", 1, 1, 0, 22),
        ("mma", 1, 1, 1, 33),
        "end",
    ]


@pytest.mark.parametrize("subtiles, accs", [((2, 2, 0), [0] * 4), ((2, 2, 1), [0])])
def test_staged_rejects_invalid_grid_before_reading(subtiles, accs):
    def unexpected(*_):
        pytest.fail("invalid grid emitted an atom or a read")

    with pytest.raises(ValueError, match="one accumulator per output atom"):
        TileMmaDriver.accumulate_staged(
            subtiles=subtiles,
            accs=accs,
            read_a=unexpected,
            read_b=unexpected,
            mma=unexpected,
        )
