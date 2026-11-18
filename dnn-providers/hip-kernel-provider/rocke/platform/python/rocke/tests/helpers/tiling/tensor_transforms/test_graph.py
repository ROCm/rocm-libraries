# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the coordinate graph: each kind against numpy, the conv graph against an im2col
index, the positional call rule, canonical equality, and every build / call error (no GPU).
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pytest

from rocke.helpers.tiling import tensor_transforms as tt
from rocke.helpers.tiling.tensor_transforms import (
    broadcast,
    embed,
    freeze,
    inputs,
    make_transform_graph,
    merge,
    offset,
    outputs,
    slice_window,
    unmerge,
    xor,
)

# Square and rectangular, power-of-two and not.
_SHAPES_2D = [(4, 4), (3, 3), (4, 8), (3, 5), (8, 2), (7, 1)]
_SHAPES_3D = [(2, 2, 2), (3, 3, 4), (2, 5, 3), (4, 1, 6)]


def _raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


# --- each kind against numpy over its whole domain -----------------------------------------


@pytest.mark.parametrize("lengths", _SHAPES_2D + _SHAPES_3D)
def test_unmerge_matches_numpy_unravel_over_the_whole_domain(lengths):
    names = tuple(f"p{i}" for i in range(len(lengths)))
    g = make_transform_graph(
        inputs(x=int(np.prod(lengths))),
        unmerge("x", into=names, lengths=lengths),
        outputs(*names),
    )
    flat = np.arange(int(np.prod(lengths)))
    expected = np.stack(np.unravel_index(flat, lengths), axis=1)
    got = np.array([g.evaluate((int(x),)) for x in flat])
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("lengths", [(4, 8), (3, 5), (2, 5, 3)])
def test_unmerge_outer_piece_is_not_wrapped_past_the_domain(lengths):
    """A read past the domain (a tile-rounded index) drives the outer piece past its length;
    the inner pieces are the remainders. This is what every lane computes, valid or not.
    """
    names = tuple(f"p{i}" for i in range(len(lengths)))
    total = int(np.prod(lengths))
    g = make_transform_graph(
        inputs(x=total), unmerge("x", into=names, lengths=lengths), outputs(*names)
    )
    inner = int(np.prod(lengths[1:]))
    for x in range(total, 3 * total):
        got = g.evaluate((x,))
        assert got[0] == x // inner and got[0] >= lengths[0]
        assert got[1:] == tuple(np.unravel_index(x % inner, lengths[1:]))


@pytest.mark.parametrize("lengths", _SHAPES_2D + _SHAPES_3D)
def test_merge_matches_numpy_ravel_over_the_whole_domain(lengths):
    names = tuple(f"r{i}" for i in range(len(lengths)))
    g = make_transform_graph(
        inputs(**dict(zip(names, lengths))),
        merge(names, into="flat", lengths=lengths),
        outputs("flat"),
    )
    assert g.edge("flat").domain == int(np.prod(lengths))
    for idx in itertools.product(*(range(n) for n in lengths)):
        assert g.evaluate(idx) == (int(np.ravel_multi_index(idx, lengths)),)


def test_merge_then_unmerge_round_trips():
    g = make_transform_graph(
        inputs(a=3, b=5),
        merge(("a", "b"), into="flat", lengths=(3, 5)),
        unmerge("flat", into=("a2", "b2"), lengths=(3, 5)),
        outputs("a2", "b2"),
    )
    for idx in itertools.product(range(3), range(5)):
        assert g.evaluate(idx) == idx


@pytest.mark.parametrize(
    "lengths, coeffs", [((4, 4), (2, 1)), ((3, 5), (7, 3)), ((8, 3), (0, 5))]
)
def test_embed_matches_numpy_dot_over_the_whole_domain(lengths, coeffs):
    g = make_transform_graph(
        inputs(a=lengths[0], b=lengths[1]),
        embed(("a", "b"), into="e", coeffs=coeffs),
        outputs("e"),
    )
    grid = np.stack(
        np.meshgrid(*(np.arange(n) for n in lengths), indexing="ij"), axis=-1
    )
    expected = grid @ np.array(coeffs)
    for idx in itertools.product(*(range(n) for n in lengths)):
        assert g.evaluate(idx) == (int(expected[idx]),)


@pytest.mark.parametrize("shift", [-3, 0, 5])
def test_offset_adds_the_shift_over_the_whole_domain(shift):
    g = make_transform_graph(
        inputs(h=7), offset("h", into="h2", shift=shift), outputs("h2")
    )
    got = np.array([g.evaluate((h,))[0] for h in range(7)])
    np.testing.assert_array_equal(got, np.arange(7) + shift)
    assert g.edge("h2").domain is None


def test_slice_window_adds_begin_over_the_whole_domain():
    g = make_transform_graph(
        inputs(t=6), slice_window("t", into="s", begin=10, end=16), outputs("s")
    )
    got = np.array([g.evaluate((t,))[0] for t in range(6)])
    np.testing.assert_array_equal(got, np.arange(10, 16))


def test_freeze_writes_its_value_whatever_the_inputs():
    g = make_transform_graph(inputs(m=5), freeze(into="z", value=9), outputs("m", "z"))
    assert [g.evaluate((m,)) for m in range(5)] == [(m, 9) for m in range(5)]


@pytest.mark.parametrize("rows, cols", [(4, 4), (8, 4), (3, 8), (16, 2), (5, 1)])
def test_xor_matches_numpy_and_is_a_bijection_per_row(rows, cols):
    g = make_transform_graph(
        inputs(r=rows, c=cols),
        xor(("r", "c"), into=("r2", "c2")),
        outputs("r2", "c2"),
    )
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    expected_c = np.bitwise_xor(c, np.bitwise_and(r, cols - 1))
    for i, j in itertools.product(range(rows), range(cols)):
        assert g.evaluate((i, j)) == (i, int(expected_c[i, j]))
    for i in range(rows):
        assert sorted(expected_c[i]) == list(range(cols))
    assert g.edge("c2").domain == cols
    assert g.edge("r2").domain is None


def test_lds_swizzle_unmerge_xor_merge_is_a_permutation():
    g = make_transform_graph(
        inputs(i=64),
        unmerge("i", into=("row", "col"), lengths=(8, 8)),
        xor(("row", "col"), into=("row2", "col2")),
        merge(("row2", "col2"), into="o", lengths=(8, 8)),
        outputs("o"),
    )
    got = [g.evaluate((i,))[0] for i in range(64)]
    assert sorted(got) == list(range(64))
    assert got != list(range(64))


def test_broadcast_consumes_an_input_that_addresses_nothing():
    g = make_transform_graph(inputs(m=4, n=6), broadcast(("m",)), outputs("n"))
    assert [g.evaluate((m, n)) for m in range(4) for n in range(6)] == [
        (n,) for m in range(4) for n in range(6)
    ]


# --- the conv graph against a numpy im2col index -------------------------------------------


@pytest.mark.parametrize(
    "N, Hi, Wi, C, Y, X, s, d, p",
    [
        (2, 5, 4, 3, 3, 3, 1, 1, 1),  # same-size conv, halo both sides
        (1, 7, 6, 2, 3, 2, 2, 1, 1),  # stride 2, rectangular filter
        (2, 6, 6, 4, 2, 2, 1, 2, 0),  # dilation 2, no pad
    ],
)
def test_conv_input_graph_matches_numpy_im2col(N, Hi, Wi, C, Y, X, s, d, p):
    Ho = (Hi + 2 * p - d * (Y - 1) - 1) // s + 1
    Wo = (Wi + 2 * p - d * (X - 1) - 1) // s + 1
    g = make_transform_graph(
        inputs(m=N * Ho * Wo, k=Y * X * C),
        unmerge("m", into=("n", "ho", "wo"), lengths=(N, Ho, Wo)),
        unmerge("k", into=("y", "x", "c"), lengths=(Y, X, C)),
        embed(("ho", "y"), into="h_dil", coeffs=(s, d)),
        offset("h_dil", into="hi", shift=-p),
        embed(("wo", "x"), into="w_dil", coeffs=(s, d)),
        offset("w_dil", into="wi", shift=-p),
        outputs("n", "hi", "wi", "c"),
    )
    # im2col built independently: index grids over (n, ho, wo) x (y, x, c), flattened row-major.
    n, ho, wo, y, x, c = np.meshgrid(
        np.arange(N),
        np.arange(Ho),
        np.arange(Wo),
        np.arange(Y),
        np.arange(X),
        np.arange(C),
        indexing="ij",
    )
    im2col = np.stack([n, ho * s + y * d - p, wo * s + x * d - p, c], axis=-1).reshape(
        N * Ho * Wo, Y * X * C, 4
    )
    for m in range(N * Ho * Wo):
        for k in range(Y * X * C):
            assert g.evaluate((m, k)) == tuple(int(v) for v in im2col[m, k])


# --- positional calls, resolve, equality ---------------------------------------------------


def _mn_graph():
    return make_transform_graph(
        inputs(m=8, n=4),
        embed(("m", "n"), into="flat", coeffs=(4, 1)),
        outputs("flat"),
    )


def test_evaluate_and_resolve_take_a_tuple_in_inputs_order():
    g = _mn_graph()
    assert g.evaluate((3, 1)) == (13,)
    assert g.evaluate((1, 3)) == (7,)
    assert g.resolve((3, 1)) == {"m": 3, "n": 1, "flat": 13}


@pytest.mark.parametrize("call", ["evaluate", "resolve"])
@pytest.mark.parametrize("coords", [{"m": 3, "n": 1}, [3, 1]])
def test_a_non_tuple_raises_the_positional_type_error(call, coords):
    g = _mn_graph()
    with _raises(
        TypeError,
        f"{call} takes the coordinates as a tuple in inputs(...) order, got "
        f"{type(coords).__name__} -- coordinates are positional",
    ):
        getattr(g, call)(coords)


@pytest.mark.parametrize("call", ["evaluate", "resolve"])
@pytest.mark.parametrize("coords", [(3,), (3, 1, 0)])
def test_a_wrong_count_raises_the_count_message(call, coords):
    g = _mn_graph()
    with _raises(
        ValueError,
        f"{call} got {len(coords)} coordinates, the graph has 2 inputs ('m', 'n') -- give one "
        "int per input, in inputs(...) order",
    ):
        getattr(g, call)(coords)


def test_input_order_is_part_of_the_graph():
    a = make_transform_graph(inputs(m=8, n=4), outputs("m", "n"))
    b = make_transform_graph(inputs(n=4, m=8), outputs("m", "n"))
    assert a != b
    assert a.evaluate((1, 2)) == (1, 2)
    assert b.evaluate((1, 2)) == (2, 1)


def test_equality_and_hash_are_stable_under_step_order():
    steps = [
        unmerge("m", into=("n", "ho", "wo"), lengths=(2, 3, 4)),
        unmerge("k", into=("y", "x", "c"), lengths=(3, 3, 2)),
        embed(("ho", "y"), into="h_dil", coeffs=(1, 1)),
        offset("h_dil", into="hi", shift=-1),
        embed(("wo", "x"), into="w_dil", coeffs=(1, 1)),
        offset("w_dil", into="wi", shift=-1),
        broadcast(("n",)),
    ]
    graphs = [
        make_transform_graph(inputs(m=24, k=18), *perm, outputs("hi", "wi", "c"))
        for perm in (steps, steps[::-1], steps[3:] + steps[:3])
    ]
    assert graphs[0] == graphs[1] == graphs[2]
    assert len({hash(g) for g in graphs}) == 1
    assert graphs[0].nodes == graphs[1].nodes == graphs[2].nodes
    # Topological: every read is an input or written by an earlier node.
    written = set(graphs[0].inputs)
    for node in graphs[0].nodes:
        assert set(node.read) <= written
        written |= set(node.write)


def test_resolve_contains_every_intermediate():
    g = make_transform_graph(
        inputs(m=24, k=18),
        unmerge("m", into=("n", "ho", "wo"), lengths=(2, 3, 4)),
        unmerge("k", into=("y", "x", "c"), lengths=(3, 3, 2)),
        embed(("ho", "y"), into="h_dil", coeffs=(2, 1)),
        offset("h_dil", into="hi", shift=-1),
        embed(("wo", "x"), into="w_dil", coeffs=(2, 1)),
        offset("w_dil", into="wi", shift=-1),
        outputs("n", "hi", "wi", "c"),
    )
    got = g.resolve((17, 11))
    names = {"m", "k", "n", "ho", "wo", "y", "x", "c", "h_dil", "hi", "w_dil", "wi"}
    assert set(got) == names
    # m = 17 -> (n, ho, wo) = (1, 1, 1); k = 11 -> (y, x, c) = (1, 2, 1)
    assert (got["n"], got["ho"], got["wo"], got["y"], got["x"], got["c"]) == (
        1,
        1,
        1,
        1,
        2,
        1,
    )
    assert got["h_dil"] == 3 and got["hi"] == 2 and got["w_dil"] == 4 and got["wi"] == 3
    assert tuple(got[o] for o in g.outputs) == g.evaluate((17, 11))


def test_edges_record_domains_and_their_writers():
    g = make_transform_graph(
        inputs(k=12, t=None),
        unmerge("k", into=("y", "c"), lengths=(3, 4)),
        offset("t", into="t2", shift=1),
        outputs("y", "c", "t2"),
    )
    assert g.edge("k") == tt.Edge("k", 12, "inputs(...)")
    assert g.edge("t").domain is None
    assert g.edge("y") == tt.Edge("y", 3, "unmerge('k' -> 'y', 'c')")
    assert g.edge("c").domain == 4
    assert g.edge("t2") == tt.Edge("t2", None, "offset('t' -> 't2')")


def test_a_graph_is_frozen():
    g = _mn_graph()
    with pytest.raises(AttributeError):
        g.outputs = ("m",)
    with pytest.raises(TypeError):
        g.inputs["m"] = 3


# --- builder errors ------------------------------------------------------------------------


def test_builder_errors_fire_with_their_text():
    with _raises(
        ValueError,
        "embed reads 2 coordinates but has 1 coeffs -- give one coefficient per read",
    ):
        embed(("a", "b"), into="e", coeffs=(1,))
    with _raises(
        ValueError,
        "embed coefficient -2 on read 'b' is negative -- reversed axes are not supported; "
        "reorder the tensor's strides instead",
    ):
        embed(("a", "b"), into="e", coeffs=(1, -2))
    with _raises(
        ValueError,
        "offset into='h' rewrites its own read -- give the step a new output name (each "
        "coordinate is written once)",
    ):
        offset("h", into="h", shift=-1)
    with _raises(
        ValueError,
        "unmerge into='k' rewrites its own read -- give the step a new output name (each "
        "coordinate is written once)",
    ):
        unmerge("k", into=("y", "k"), lengths=(3, 4))
    with _raises(ValueError, "slice_window needs begin < end, got begin=4, end=4"):
        slice_window("t", into="s", begin=4, end=4)
    with _raises(ValueError, "unmerge lengths must be >= 1, got (3, 0)"):
        unmerge("k", into=("y", "c"), lengths=(3, 0))
    with _raises(ValueError, "merge lengths must be >= 1, got (0, 4)"):
        merge(("a", "b"), into="f", lengths=(0, 4))
    with _raises(ValueError, "inputs lengths must be >= 1, got (4, 0)"):
        inputs(m=4, t=None, n=0)
    with _raises(
        ValueError,
        "xor needs exactly two reads and two outputs, got reads=('a',), into=('b', 'c')",
    ):
        xor(("a",), into=("b", "c"))


def test_count_errors_of_merge_and_unmerge():
    with _raises(
        ValueError,
        "merge reads 2 coordinates but has 3 lengths -- give one length per read",
    ):
        merge(("a", "b"), into="f", lengths=(2, 3, 4))
    with _raises(
        ValueError,
        "unmerge writes 3 coordinates but has 2 lengths -- give one length per written "
        "coordinate",
    ):
        unmerge("k", into=("y", "x", "c"), lengths=(3, 4))


# --- assembly errors -----------------------------------------------------------------------


def test_assembly_errors_fire_with_their_text():
    with _raises(
        ValueError,
        "coordinate 'h' is written by two steps (offset('a' -> 'h'), offset('b' -> 'h')) -- "
        "give one of them a new output name",
    ):
        make_transform_graph(
            inputs(a=4, b=4),
            offset("a", into="h", shift=1),
            offset("b", into="h", shift=2),
            outputs("h"),
        )
    with _raises(
        ValueError,
        "coordinate 'm' is written by two steps (inputs(...), offset('a' -> 'm')) -- give one "
        "of them a new output name",
    ):
        make_transform_graph(
            inputs(a=4, m=4), offset("a", into="m", shift=1), outputs("m")
        )
    with _raises(
        ValueError,
        "coordinate 'q' is read by offset('q' -> 'h') but never declared or written -- "
        "declare it in inputs(...) or write it in an earlier step",
    ):
        make_transform_graph(
            inputs(a=4), offset("q", into="h", shift=1), outputs("a", "h")
        )
    with _raises(
        ValueError,
        "the steps form a cycle through 'b', 'a' -- a coordinate cannot depend on itself",
    ):
        make_transform_graph(
            inputs(m=4),
            offset("a", into="b", shift=1),
            offset("b", into="a", shift=1),
            outputs("m", "a"),
        )
    with _raises(
        ValueError, "duplicate step broadcast('m' -> ()) -- remove the repeat"
    ):
        make_transform_graph(
            inputs(m=4, n=4), broadcast(("m",)), broadcast(("m",)), outputs("n")
        )
    with _raises(
        ValueError,
        "unmerge of 'k' declares lengths (3, 5) (product 15) but 'k' has domain 12 -- the "
        "pieces must multiply to the read's domain",
    ):
        make_transform_graph(
            inputs(k=12),
            unmerge("k", into=("y", "c"), lengths=(3, 5)),
            outputs("y", "c"),
        )
    with _raises(
        ValueError,
        "merge into 'f' declares lengths (4, 4) but its reads have domains (4, 8) -- restated "
        "lengths must match",
    ):
        make_transform_graph(
            inputs(a=4, b=8), merge(("a", "b"), into="f", lengths=(4, 4)), outputs("f")
        )
    with _raises(
        ValueError,
        "xor reads 'c' whose domain is 6; xor needs a power-of-two domain to be a bijection",
    ):
        make_transform_graph(
            inputs(r=4, c=6), xor(("r", "c"), into=("r2", "c2")), outputs("r2", "c2")
        )
    with _raises(
        ValueError,
        "xor reads 'c2', which has no declared domain -- xor needs a power-of-two domain; "
        "split it with unmerge first",
    ):
        make_transform_graph(
            inputs(r=4, c=8),
            offset("c", into="c2", shift=0),
            xor(("r", "c2"), into=("r2", "c3")),
            outputs("r2", "c3"),
        )
    with _raises(
        ValueError,
        "make_transform_graph needs inputs(...) first and outputs(...) last, got "
        "outputs(...) ... inputs(...)",
    ):
        make_transform_graph(outputs("m"), inputs(m=4))
    with _raises(
        ValueError,
        "make_transform_graph needs inputs(...) first and outputs(...) last, got "
        "inputs(...) ... offset('m' -> 'h')",
    ):
        make_transform_graph(inputs(m=4), offset("m", into="h", shift=1))
    with _raises(ValueError, "output 'z' is never written or declared"):
        make_transform_graph(inputs(m=4), outputs("m", "z"))
    with _raises(
        ValueError,
        "input 'n' is never read -- remove it or use broadcast to consume it",
    ):
        make_transform_graph(inputs(m=4, n=4), outputs("m"))


def test_a_runtime_length_input_or_merge_read_without_a_domain_builds():
    """Domains that are not declared are not mismatches: an unmerge or merge read with no domain
    is checked against the kind's required domain at run time, not refused here."""
    g = make_transform_graph(
        inputs(t=None, c=4),
        unmerge("t", into=("hi", "lo"), lengths=(3, 4)),
        offset("c", into="c2", shift=0),
        merge(("hi", "c2"), into="f", lengths=(3, 4)),
        outputs("f", "lo"),
    )
    assert g.edge("f").domain == 12
    assert g.evaluate((7, 2)) == (1 * 4 + 2, 3)


def test_a_non_step_between_inputs_and_outputs_is_refused():
    with _raises(
        TypeError,
        "make_transform_graph got int between inputs(...) and outputs(...) -- every step "
        "there is what a kind builder returns (embed, merge, unmerge, ...)",
    ):
        make_transform_graph(inputs(m=4), 3, outputs("m"))


def test_edge_of_an_unknown_coordinate_names_the_known_ones():
    with _raises(
        ValueError,
        "'z' is not a coordinate of this graph -- its coordinates are ('m', 'n', 'flat')",
    ):
        _mn_graph().edge("z")
