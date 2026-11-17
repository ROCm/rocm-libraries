# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the coordinate and address emitters of tensor_transforms.

Every emitted address is checked against ``transform_address_of`` by running the emitted IR on
the host: a small interpreter computes each op with its integer width, for every lane and a set
of per-wave values, and fails on any no-signed-wrap op whose result leaves its type (the width
proof). The op-count tests pin where work lands: the uniform side once per call, a per-lane
division at most once per call, one compare and select per access. A golden string per case
pins the emitted IR text; every case lowers to the same IR in both engines and compiles for
gfx90a, gfx942, gfx950, gfx1151, gfx1201 and gfx1250. The GPU tests at the end run the same
coordinates on whatever GPU is present, compiled for it, and compare each lane's parts with the
golden.

Every block origin that feeds an unmerge goes through a range marker stating ``0 <= origin <
length``: the marker the tiling layer's tile guard emits, which a kernel author reaches through
that guard.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from rocke.core.ir import F16, F32, I32, I64, IRBuilder, PtrType, Value
from rocke.core.ir_print import print_ir
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.helpers.tiling import tensor_transforms as tt
from rocke.helpers.tiling.descriptors import TensorDesc
from rocke.helpers.tiling.tensor_transforms import (
    embed,
    emit_addresses,
    emit_coords,
    inputs,
    make_transform_graph,
    offset,
    outputs,
    transform_address_of,
    unmerge,
    xor,
)
from rocke.helpers.tiling.tensor_transforms.emit import (
    _CallTable,
    _emit_coord_table,
    _split_access,
)
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph
from rocke.helpers.tiling.tensor_transforms.ranges import (
    AffineCoord,
    Congruence,
    Interval,
    UniformPart,
)
from rocke.helpers.tiling.tensor_transforms.walk import split_coordinate
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.gpu import current_device_arch

GOLDEN_DIR = Path(__file__).parent / "golden"
WAVE = 64


def _raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


def _builder(workgroup_size: int = WAVE) -> IRBuilder:
    b = IRBuilder("emit_test")
    b.kernel.attrs["max_workgroup_size"] = workgroup_size
    return b


def _c(b: IRBuilder, value: int) -> Value:
    return b.const_i32(value)


def _guarded(b: IRBuilder, origin: Value, length: int) -> Value:
    """The marker the tile guard emits: ``0 <= origin < length``, checked by the guard."""
    return b._assume_range(origin, lo=0, hi=length)


def _ops(b: IRBuilder):
    def walk(ops):
        for op in ops:
            yield op
            for region in op.regions:
                yield from walk(region.ops)

    return list(walk(b.kernel.body.ops))


def _op_count(b: IRBuilder) -> int:
    return len(_ops(b))


def _varies_per_lane(value: Value | int, memo: dict[int, bool] | None = None) -> bool:
    """The value depends on a thread id (it lives in vector registers)."""
    if not isinstance(value, Value) or value.op is None:
        return False
    memo = {} if memo is None else memo
    if id(value) not in memo:
        op = value.op
        memo[id(value)] = op.name == "gpu.thread_id" or (
            op.name != "scf.for" and any(_varies_per_lane(v, memo) for v in op.operands)
        )
    return memo[id(value)]


def _count(b: IRBuilder, name: str, *, per_lane: bool | None = None, **attrs) -> int:
    """Ops named ``name`` with these attributes, optionally only the per-lane (or uniform) ones."""
    total = 0
    for op in _ops(b):
        if op.name != name or any(op.attrs.get(k) != v for k, v in attrs.items()):
            continue
        if per_lane is not None and _varies_per_lane(op.results[0]) != per_lane:
            continue
        total += 1
    return total


# --- running emitted IR on the host ---------------------------------------------------------------

_BITS = {"i1": 1, "i32": 32, "i64": 64}


def _run(value: Value | int, leaves: dict[int, int], memo: dict[int, int]) -> int:
    """What ``value`` computes for the leaf values ``leaves`` (keyed by ``id``), with each op's
    integer width; a no-signed-wrap op whose result leaves its type fails the test."""
    if isinstance(value, int):
        return value
    key = id(value)
    if key in leaves:
        return leaves[key]
    if key in memo:
        return memo[key]
    op = value.op
    assert op is not None, f"no value given for leaf {value}"
    bits = _BITS[value.type.name]
    span = 1 << bits
    if op.name == "arith.constant":
        result = op.attrs["value"]
    elif op.name in ("arith.assume_range", "tile.readfirstlane"):
        result = _run(op.operands[0], leaves, memo)
    else:
        a = [_run(v, leaves, memo) for v in op.operands]
        result = _apply(op, a, bits)
    if op.name == "arith.add" and op.attrs.get("wrap") == "nuw":
        assert 0 <= result < span, f"nuw add {value} wraps: {result}"
        result = result - span if result >= span // 2 else result
    elif bits > 1:
        assert (
            -(span // 2) <= result < span // 2
        ), f"{op.name} {value} overflows: {result}"
    memo[key] = result
    return result


def _apply(op, a: list[int], bits: int) -> int:
    name, span = op.name, 1 << bits
    if name == "arith.add":
        if op.attrs.get("wrap") == "nuw":
            return a[0] % span + a[1] % span
        return a[0] + a[1]
    if name == "arith.sub":
        return a[0] - a[1]
    if name == "arith.mul":
        return a[0] * a[1]
    if name in ("arith.div", "arith.mod"):
        q = abs(a[0]) // abs(a[1])
        q = q if (a[0] >= 0) == (a[1] > 0) else -q
        return q if name == "arith.div" else a[0] - q * a[1]
    if name == "arith.lshr":
        return (a[0] % span) >> a[1]
    if name == "arith.and":
        return a[0] & a[1]
    if name == "arith.xor":
        return a[0] ^ a[1]
    if name == "arith.smin":
        return min(a)
    if name == "arith.smax":
        return max(a)
    if name == "arith.cmp":
        x, y = a
        if op.attrs["pred"] == "ult":
            unsigned = 1 << _BITS[op.operands[0].type.name]
            return int(x % unsigned < y % unsigned)
        return int({"ge": x >= y, "lt": x < y, "gt": x > y}[op.attrs["pred"]])
    if name == "arith.select":
        return a[1] if a[0] else a[2]
    if name in ("arith.zext", "arith.sext"):
        return a[0]
    if name == "arith.trunc":
        low = a[0] % span
        return low - span if low >= span // 2 else low
    if name == "arith.umul_hi_i32":
        high = ((a[0] % 2**32) * (a[1] % 2**32)) >> 32
        return high - 2**32 if high >= 2**31 else high
    raise AssertionError(f"no host rule for {name}")


def _address(address: tt.ElementAddress, leaves: dict[int, int]) -> int:
    """The element address one lane computes: every part run on the host and summed."""
    memo: dict[int, int] = {}
    total = address.const
    for term in address.uniform:
        total += _run(term.value, leaves, memo)
    if address.lane is not None:
        lane = _run(address.lane, leaves, memo)
        lo, hi = address.lane_range
        assert 0 <= lo <= lane < hi <= 2**31
        total += lane
    return total


# --- shared shapes ------------------------------------------------------------------------------


def _conv(
    n: int, h: int, w: int, c: int, *, y: int = 3, x: int = 3, pad: int = 1
) -> tuple[tt.TransformGraph, TensorDesc]:
    """The implicit-GEMM convolution input graph (stride 1) and its NHWC tensor."""
    ho, wo = h + 2 * pad - y + 1, w + 2 * pad - x + 1
    graph = make_transform_graph(
        inputs(m=n * ho * wo, k=y * x * c),
        unmerge("m", into=("n", "ho", "wo"), lengths=(n, ho, wo)),
        unmerge("k", into=("y", "x", "c"), lengths=(y, x, c)),
        embed(("ho", "y"), into="h_dil", coeffs=(1, 1)),
        offset("h_dil", into="hi", shift=-pad),
        embed(("wo", "x"), into="w_dil", coeffs=(1, 1)),
        offset("w_dil", into="wi", shift=-pad),
        outputs("n", "hi", "wi", "c"),
    )
    return graph, TensorDesc((n, h, w, c), (h * w * c, w * c, c, 1), F16)


class _ConvAccess:
    """The conv coordinates a 64-lane tile builds: ``m = origin + tid / 4`` (rows) and
    ``k = k0 + (tid % 4) * 8`` (columns), with the origin guarded and ``k0`` given."""

    def __init__(
        self, b: IRBuilder, graph: tt.TransformGraph, tile_m: int = 16
    ) -> None:
        self.tid = b.thread_id_x()
        self.block = b.block_id_x()
        m_length = graph.inputs["m"]
        self.origin = _guarded(b, b.mul(self.block, _c(b, tile_m)), m_length)
        self.m = b.add(self.origin, b.div(self.tid, _c(b, 4)))
        self.k_lane = b.mul(b.mod(self.tid, _c(b, 4)), _c(b, 8))
        self.tile_m = tile_m

    def host(self, block: int, tid: int, k0: int = 0) -> tuple[int, int]:
        return block * self.tile_m + tid // 4, k0 + (tid % 4) * 8


def _check_against_golden(
    graph: tt.TransformGraph,
    tensor: TensorDesc,
    addresses: tuple[tt.ElementAddress, ...],
    host_coords,
    leaf_values,
) -> None:
    """Every access's emitted address equals ``transform_address_of`` for every leaf setting.

    ``leaf_values`` yields ``(leaves, context)``; ``host_coords(context, i)`` gives access ``i``'s
    coordinates as ints."""
    for leaves, context in leaf_values:
        for i, address in enumerate(addresses):
            expected = transform_address_of(graph, tensor, host_coords(context, i))
            assert _address(address, leaves) == expected, (context, i)


# --- uniform terms stay apart by input ----------------------------------------------------------


def _two_input_case(b: IRBuilder):
    tid = b.thread_id_x()
    row0 = _guarded(b, b.mul(b.block_id_x(), _c(b, 16)), 256)
    col0 = _guarded(b, b.mul(b.block_id_y(), _c(b, 16)), 256)
    row = b.add(b.add(row0, b.div(tid, _c(b, 16))), _c(b, 3))
    col = b.add(col0, b.mod(tid, _c(b, 16)))
    return row, col


def test_uniform_parts_of_two_inputs_stay_two_terms_with_one_lane_sum_and_an_int_constant():
    b = _builder()
    row, col = _two_input_case(b)
    tensor = TensorDesc((256, 256), (256, 1), F32)
    graph = identity_graph((256, 256))
    (address,) = emit_addresses(b, graph, tensor, [(row, col)])
    assert [term.inputs for term in address.uniform] == [frozenset({0}), frozenset({1})]
    assert address.lane.type == I32
    assert isinstance(address.const, int) and address.const == 3 * 256
    assert address.elem_bits == 32
    assert address.uniform_delta is None and address.uniform_delta_range is None

    tid, bx, by = (_leaf(b, name) for name in ("thread_id", "bx", "by"))
    _check_against_golden(
        graph,
        tensor,
        (address,),
        lambda ctx, _: (ctx[1] * 16 + ctx[0] // 16 + 3, ctx[2] * 16 + ctx[0] % 16),
        (
            ({id(tid): t, id(bx): x, id(by): y}, (t, x, y))
            for t, x, y in itertools.product(range(WAVE), range(3), range(3))
        ),
    )


def _leaf(b: IRBuilder, which: str) -> Value:
    """The kernel's thread id, block id x or block id y value."""
    name, axis = {
        "thread_id": ("gpu.thread_id", "x"),
        "bx": ("gpu.block_id", "x"),
        "by": ("gpu.block_id", "y"),
    }[which]
    for op in _ops(b):
        if op.name == name and op.attrs.get("axis") == axis:
            return op.results[0]
    raise AssertionError(f"no {which} in the kernel")


def test_an_embed_of_two_inputs_keeps_their_two_terms():
    b = _builder()
    row, col = _two_input_case(b)
    graph = make_transform_graph(
        inputs(r=256, c=256), embed(("r", "c"), into="e", coeffs=(256, 1)), outputs("e")
    )
    (address,) = emit_addresses(b, graph, TensorDesc((65536,), (1,), F32), [(row, col)])
    assert [term.inputs for term in address.uniform] == [frozenset({0}), frozenset({1})]


def test_an_unmerge_or_xor_whose_read_mixes_two_inputs_reports_one_term_keyed_by_both():
    b = _builder()
    row, col = _two_input_case(b)
    graph = make_transform_graph(
        inputs(r=256, c=256),
        embed(("r", "c"), into="e", coeffs=(256, 1)),
        unmerge("e", into=("p", "q"), lengths=(512, 128)),
        outputs("p", "q"),
    )
    tensor = TensorDesc((512, 128), (128, 1), F32)
    (address,) = emit_addresses(b, graph, tensor, [(row, col)])
    assert [term.inputs for term in address.uniform] == [frozenset({0, 1})]

    b = _builder()
    i = _guarded(b, b.block_id_x(), 1024)
    j = _guarded(b, b.block_id_y(), 8)
    graph = make_transform_graph(
        inputs(i=1024, j=8), xor(("i", "j"), into=("a", "s")), outputs("a", "s")
    )
    (address,) = emit_addresses(b, graph, TensorDesc((1024, 8), (8, 1), F32), [(i, j)])
    assert [term.inputs for term in address.uniform] == [
        frozenset({0}),
        frozenset({0, 1}),
    ]
    assert address.lane is None


# --- unmerge --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths, ops, absent",
    [
        ((16, 16), {"arith.lshr", "arith.and"}, {"arith.umul_hi_i32"}),
        ((16, 12), {"arith.umul_hi_i32", "arith.lshr"}, {"arith.and"}),
    ],
)
def test_unmerge_divides_by_a_shift_for_a_power_of_two_and_by_magic_division_otherwise(
    lengths, ops, absent
):
    b = _builder()
    total = lengths[0] * lengths[1]
    tid = b.thread_id_x()
    m = b.add(_guarded(b, b.mul(b.block_id_x(), _c(b, 8)), total), tid)
    graph = make_transform_graph(
        inputs(m=total),
        unmerge("m", into=("a", "b"), lengths=lengths),
        outputs("a", "b"),
    )
    tensor = TensorDesc(lengths, (lengths[1], 1), F32)
    emit_addresses(b, graph, tensor, [(m,)])
    names = {op.name for op in _ops(b)}
    assert ops <= names and not absent & names
    if "arith.umul_hi_i32" in ops:
        assert _count(b, "arith.add", wrap="nuw") >= 1


@pytest.mark.parametrize(
    "lane_extent, method",
    [(6, "none"), (12, "compare-select"), (40, "division")],
)
def test_the_shared_lane_split_is_picked_by_the_lane_range_once_per_call(
    lane_extent, method
):
    def build(constants):
        b = _builder()
        lane = b.mod(b.thread_id_x(), _c(b, lane_extent))
        graph = make_transform_graph(
            inputs(m=600),
            unmerge("m", into=("a", "b"), lengths=(100, 6)),
            outputs("a", "b"),
        )
        tensor = TensorDesc((100, 6), (6, 1), F32)
        accesses = [(b.add(lane, _c(b, k)) if k else lane,) for k in constants]
        return b, emit_addresses(b, graph, tensor, accesses), lane, graph, tensor

    b, _, lane, _, _ = build([0])
    splits_on_lane = {
        op.name for op in _ops(b) if op.operands and op.operands[0] is lane
    } - {"arith.add"}
    expected = {
        "none": set(),
        "compare-select": {"arith.cmp", "arith.sub"},
        "division": {"arith.umul_hi_i32", "arith.sub"},
    }[method]
    assert splits_on_lane == expected
    one = _count(b, "arith.umul_hi_i32") + _count(b, "arith.cmp", pred="ge")

    constants = [
        0,
        1,
        5,
        8,
        -2,
    ]  # distinct residues modulo 6: no two accesses share a compare
    b, addresses, lane, graph, tensor = build(constants)
    # Each extra access adds one compare and select (step 5), never another lane split.
    assert (
        _count(b, "arith.umul_hi_i32") + _count(b, "arith.cmp", pred="ge")
        == one + len(constants) - 1
    )
    tid = _leaf(b, "thread_id")
    _check_against_golden(
        graph,
        tensor,
        addresses,
        lambda t, i: (t % lane_extent + constants[i],),
        (({id(tid): t}, t) for t in range(WAVE)),
    )


def _stated(b: IRBuilder, name: str, **facts) -> Value:
    """A kernel parameter with stated facts (a marker over it)."""
    return b._assume_range(b.param(name, I32), **facts)


@pytest.mark.parametrize(
    "parts, aligned",
    [
        (({"lo": 0, "hi": 1 << 20, "multiple_of": 128},), True),
        (({"lo": 0, "hi": 1 << 20, "multiple_of": 256, "residue": 128},) * 2, True),
        (({"lo": 0, "hi": 1 << 20, "multiple_of": 64},), False),
    ],
)
def test_a_uniform_side_aligned_to_the_inner_length_emits_no_scalar_remainder(
    parts, aligned
):
    b = _builder()
    uniform = [_stated(b, f"p{i}", **facts) for i, facts in enumerate(parts)]
    # Added around the per-lane value, so the walk keeps each stated value its own part.
    m = b.add(uniform[0], b.thread_id_x())
    if len(uniform) == 2:
        m = b.add(m, uniform[1])
    graph = make_transform_graph(
        inputs(m=1 << 21),
        unmerge("m", into=("a", "b"), lengths=(1 << 14, 128)),
        outputs("a", "b"),
    )
    emit_addresses(
        b,
        graph,
        TensorDesc((1 << 14, 128), (128, 1), F32),
        [(m,), (b.add(m, _c(b, 3)),)],
    )
    scalar_mask = _count(b, "arith.and", per_lane=False)
    scalar_compare = _count(b, "arith.cmp", per_lane=False)
    if aligned:
        assert scalar_mask == 0 and scalar_compare == 0
    else:
        assert scalar_mask == 1 and scalar_compare == 1


# --- the split coordinate -----------------------------------------------------------------------


def test_adding_split_coordinates_emits_nothing_and_combines_their_facts():
    b = _builder()
    u0, u1, lane = b.block_id_x(), b.block_id_y(), b.thread_id_x()
    before = _op_count(b)
    left = AffineCoord(3, (UniformPart(u0, Interval(0, 100), Congruence(16, 0)),))
    right = AffineCoord(
        5,
        (UniformPart(u1, Interval(0, 50), Congruence(8, 4)),),
        lane,
        Interval(0, 64),
        Congruence(8, 0),
    )
    total = left + right
    assert _op_count(b) == before
    assert total.const == 8
    assert [p.value for p in total.uniform] == [u0, u1]
    assert total.lane is lane and total.lane_range == Interval(0, 64)
    # 16i + (8j + 4) + 8l + 8 is 4 more than a multiple of 8.
    assert total.congruence == Congruence(8, 4)
    assert total.range == Interval(8, 8 + 99 + 49 + 63 + 1)
    with pytest.raises(ValueError, match="both operands have a lane part"):
        total + AffineCoord(0, (), b.lane_id(), Interval(0, 64))


def test_plain_values_and_their_split_coordinates_emit_identical_ir():
    def build(split: bool) -> str:
        b = _builder()
        graph, tensor = _conv(2, 6, 6, 3)
        conv = _ConvAccess(b, graph)
        k = b.add(conv.k_lane, _c(b, 1))
        access = (conv.m, k)
        if split:
            access = tuple(split_coordinate(b, v) for v in access)
        emit_addresses(b, graph, tensor, [access])
        return print_ir(b.kernel)

    assert build(split=False) == build(split=True)


def test_the_package_door_exports_no_split_coordinate_or_factory_for_it():
    assert not hasattr(tt, "AffineCoord") and "AffineCoord" not in tt.__all__
    assert not hasattr(AffineCoord, "of_uniform") and not hasattr(
        AffineCoord, "of_lane"
    )
    for name in ("ElementAddress", "UniformTerm", "emit_coords", "emit_addresses"):
        assert name in tt.__all__


# --- the convolution K split ----------------------------------------------------------------------


@pytest.mark.parametrize("channels", [64, 3, 4, 8])
def test_the_conv_k_split_divides_once_per_level_and_compares_once_per_access(channels):
    """The K-loop counter is a multiple of the tile (not of X*C); its lane part is in
    [0, tile_k). Each extra access adds exactly one per-lane compare and select per level and
    no division; the uniform side is divided once per level per call."""
    tile_k, y_len, x_len = 32, 3, 3
    total = y_len * x_len * channels
    # The residue near the length, a constant past it and a negative one; their residues
    # modulo C differ when C >= 4, so no two accesses share their per-lane compare.
    constants = [0, channels - 1, channels + 1, -2]

    def build(consts):
        b = _builder()
        graph = make_transform_graph(
            inputs(k=total),
            unmerge("k", into=("y", "x", "c"), lengths=(y_len, x_len, channels)),
            outputs("y", "x", "c"),
        )
        tensor = TensorDesc(
            (y_len, x_len, channels), (x_len * channels, channels, 1), F16
        )
        lane = b.mod(b.thread_id_x(), _c(b, tile_k))
        # At least two trips, so the counter is a per-wave value rather than the constant 0.
        stop = max(-(-total // tile_k), 2) * tile_k
        with b.scf_for(_c(b, 0), _c(b, stop), _c(b, tile_k)) as k_loop:
            k = b.add(k_loop, lane)
            accesses = [(b.add(k, _c(b, c)) if c else k,) for c in consts]
            addresses = emit_addresses(b, graph, tensor, accesses)
        return b, graph, tensor, addresses, k_loop, stop

    b1, *_ = build(constants[:1])
    b, graph, tensor, addresses, k_loop, stop = build(constants)
    extra = len(constants) - 1
    inner_levels = 2

    def per_call(bb, per_lane):
        # Every division ends in one shift: a power of two is one, magic division ends in one.
        return _count(bb, "arith.lshr", per_lane=per_lane)

    # The uniform side: one division per level, however many accesses.
    assert per_call(b, False) == per_call(b1, False) == inner_levels
    # The lane side: at most one shared split per level, none when tile_k <= C.
    assert per_call(b, True) == per_call(b1, True) <= inner_levels
    if channels >= tile_k:
        assert per_call(b, True) == 0
    # One per-lane compare and select per level for each extra access.
    lane_compares = _count(b, "arith.cmp", per_lane=True) - _count(
        b1, "arith.cmp", per_lane=True
    )
    lane_selects = _count(b, "arith.select", per_lane=True) - _count(
        b1, "arith.select", per_lane=True
    )
    assert lane_compares == lane_selects <= extra * inner_levels
    if channels >= 4:
        assert lane_compares == extra * inner_levels
    # The uniform side: at most one compare per level per distinct constant.
    assert _count(b, "arith.cmp", per_lane=False) <= len(constants) * inner_levels

    tid = _leaf(b, "thread_id")
    _check_against_golden(
        graph,
        tensor,
        addresses,
        lambda ctx, i: (ctx[0] + ctx[1] % tile_k + constants[i],),
        (
            ({id(k_loop): k0, id(tid): t}, (k0, t))
            for k0 in range(0, stop, tile_k)
            for t in range(WAVE)
        ),
    )


def test_only_the_outer_conv_quotients_reach_the_uniform_terms():
    b = _builder()
    graph, tensor = _conv(2, 6, 6, 3)
    conv = _ConvAccess(b, graph)
    with b.scf_for(_c(b, 0), _c(b, 64), _c(b, 32)) as k_loop:
        access = (conv.m, b.add(k_loop, conv.k_lane))
        table = _CallTable(b)
        env = _emit_coord_table(table, graph, _split_access(b, access))
        address = emit_addresses(b, graph, tensor, [access])[0]
    assert [t.inputs for t in env["n"].terms] == [frozenset({0})]
    assert [t.inputs for t in env["y"].terms] == [frozenset({1})]
    for inner in ("ho", "wo", "x", "c"):
        assert env[inner].terms == () and env[inner].lane is not None
    assert [t.inputs for t in address.uniform] == [frozenset({0}), frozenset({1})]
    assert 0 <= address.lane_range[0] < address.lane_range[1] <= 2**31


def test_out_of_domain_tile_rounded_coordinates_never_overflow_an_emitted_op():
    """Rows past N*Ho*Wo and columns past Y*X*C still compute every part inside its width."""
    graph, tensor = _conv(256, 64, 64, 64)
    m_length, k_length = graph.inputs["m"], graph.inputs["k"]
    b = _builder()
    conv = _ConvAccess(b, graph, tile_m=16)
    stop = -(-k_length // 32) * 32
    with b.scf_for(_c(b, 0), _c(b, stop), _c(b, 32)) as k_loop:
        k = b.add(k_loop, conv.k_lane)
        addresses = emit_addresses(
            b, graph, tensor, [(conv.m, k), (conv.m, b.add(k, _c(b, 7)))]
        )
    last_block = (m_length - 1) // 16
    for address in addresses:
        assert address.lane_range[1] <= 2**31
    _check_against_golden(
        graph,
        tensor,
        addresses,
        lambda ctx, i: (ctx[1] * 16 + ctx[2] // 4, ctx[0] + (ctx[2] % 4) * 8 + 7 * i),
        (
            ({id(k_loop): k0, id(conv.block): blk, id(conv.tid): t}, (k0, blk, t))
            for k0 in (0, stop - 32)
            for blk in (0, last_block)
            for t in (0, 3, 60, 63)
        ),
    )


# --- attention: the grouped-query head fold -------------------------------------------------------


def test_a_grouped_query_head_fold_divides_the_head_once_per_wave():
    """Grouped-query attention reads K by KV head: query head ``h`` of 40 reads KV head
    ``h // 5`` of 8. The fold unmerges the query head into (KV head, group) and leaves the group
    unread. The head is the block id, so its division by 5 (a magic division, 5 not being a
    power of two) is emitted once, on the per-wave side, and no lane divides.
    """
    q_heads, kv_heads, tokens, head_size = 40, 8, 256, 128
    graph = make_transform_graph(
        inputs(q_head=q_heads, token=tokens, d=head_size),
        unmerge(
            "q_head", into=("kv_head", "group"), lengths=(kv_heads, q_heads // kv_heads)
        ),
        outputs("token", "kv_head", "d"),
    )
    k_cache = TensorDesc(
        (tokens, kv_heads, head_size), (kv_heads * head_size, head_size, 1), F16
    )
    b = _builder()
    tid = b.thread_id_x()
    head = _guarded(b, b.block_id_y(), q_heads)
    token0 = _guarded(b, b.mul(b.block_id_x(), _c(b, 4)), tokens)
    token = b.add(token0, b.div(tid, _c(b, 16)))
    d = b.mul(b.mod(tid, _c(b, 16)), _c(b, 8))
    (address,) = emit_addresses(b, graph, k_cache, [(head, token, d)])
    assert _count(b, "arith.umul_hi_i32", per_lane=False) == 1
    assert _count(b, "arith.umul_hi_i32", per_lane=True) == 0
    assert _count(b, "arith.add", wrap="nuw") == 1

    tid, bx, by = (_leaf(b, name) for name in ("thread_id", "bx", "by"))
    _check_against_golden(
        graph,
        k_cache,
        (address,),
        lambda ctx, _: (ctx[2], ctx[1] * 4 + ctx[0] // 16, (ctx[0] % 16) * 8),
        (
            ({id(tid): t, id(bx): x, id(by): y}, (t, x, y))
            for t, x, y in itertools.product(
                range(WAVE), (0, 1, tokens // 4 - 1), range(q_heads)
            )
        ),
    )


# --- xor and shared sub-expressions ---------------------------------------------------------------


def _xor_case(b: IRBuilder, copies: int = 1):
    tid = b.thread_id_x()
    row = b.add(_guarded(b, b.mul(b.block_id_x(), _c(b, 8)), 64), b.div(tid, _c(b, 8)))
    col = b.mod(tid, _c(b, 8))
    graph = make_transform_graph(
        inputs(r=64, c=8), xor(("r", "c"), into=("row", "col")), outputs("row", "col")
    )
    tensor = TensorDesc((64, 8), (8, 1), F16)
    return graph, tensor, emit_addresses(b, graph, tensor, [(row, col)] * copies)


def test_xor_masks_with_and_and_identical_accesses_are_emitted_once():
    one, two = _builder(), _builder()
    _xor_case(one)
    graph, tensor, addresses = _xor_case(two, copies=2)
    assert (
        _count(two, "arith.and") == 1 and _count(two, "arith.mod") == 1
    )  # the author's tid % 8
    assert _op_count(one) == _op_count(two)
    assert addresses[0].lane is addresses[1].lane
    tid, bx = _leaf(two, "thread_id"), _leaf(two, "bx")
    _check_against_golden(
        graph,
        tensor,
        addresses,
        lambda ctx, _: (ctx[1] * 8 + ctx[0] // 8, ctx[0] % 8),
        (({id(tid): t, id(bx): x}, (t, x)) for t in range(WAVE) for x in range(8)),
    )


def _halo_xor_case(b: IRBuilder):
    """A 16-column swizzle read with a left halo of 4: ``c = tid % 16 - 4`` reaches -4."""
    tid = b.thread_id_x()
    row = b.div(tid, _c(b, 16))
    col = b.sub(b.mod(tid, _c(b, 16)), _c(b, 4))
    graph = make_transform_graph(
        inputs(r=8, c=16), xor(("r", "c"), into=("row", "col")), outputs("row", "col")
    )
    return graph, TensorDesc((8, 16), (16, 1), F16), (row, col)


def test_a_halo_column_fed_into_an_xor_is_rebased_and_addressed_on_every_lane():
    """The swizzle of a column that starts below 0 stays per-lane: its range ``[-16, 16)`` is
    rebased like any per-lane part (the low end moves into the constant), and every lane,
    including the halo lanes a bound check later masks, gets the address the graph gives.
    """
    b = _builder()
    graph, tensor, access = _halo_xor_case(b)
    (address,) = emit_addresses(b, graph, tensor, [access])
    assert address.lane_range[0] == 0 and address.const == -16
    tid = _leaf(b, "thread_id")
    _check_against_golden(
        graph,
        tensor,
        (address,),
        lambda t, _: (t // 16, t % 16 - 4),
        (({id(tid): t}, t) for t in range(WAVE)),
    )


def test_emit_coords_gives_every_named_coordinate_as_one_value():
    b = _builder()
    graph, _ = _conv(2, 6, 6, 3)
    conv = _ConvAccess(b, graph)
    values = emit_coords(b, graph, (conv.m, b.add(conv.k_lane, _c(b, 5))))
    assert set(values) == {
        "m",
        "k",
        "n",
        "ho",
        "wo",
        "y",
        "x",
        "c",
        "h_dil",
        "hi",
        "w_dil",
        "wi",
    }
    for blk, t in itertools.product(range(3), range(WAVE)):
        m, k = conv.host(blk, t)
        expected = graph.resolve((m, k + 5))
        leaves = {id(conv.block): blk, id(conv.tid): t}
        memo: dict[int, int] = {}
        assert {name: _run(v, leaves, memo) for name, v in values.items()} == expected


# --- the lane rebase ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["c - lane", "smin(lane - 3, 10)"])
def test_a_lane_part_starting_below_zero_is_rebased_once_per_call(shape):
    b = _builder()
    tid = b.thread_id_x()
    if shape == "c - lane":
        value, lo, host = b.sub(_c(b, 40), tid), -23, lambda t: 40 - t
    else:
        value, lo, host = (
            b.smin(b.sub(tid, _c(b, 3)), _c(b, 10)),
            -3,
            lambda t: min(t - 3, 10),
        )
    tensor = TensorDesc((64, 16), (16, 1), F32)
    graph = identity_graph((64, 16))
    accesses = [(value, j) for j in (0, 1, 5)]
    addresses = emit_addresses(b, graph, tensor, accesses)
    rebases = [
        op
        for op in _ops(b)
        if op.name == "arith.add"
        and op.operands[0] is value
        and op.operands[1].op.attrs.get("value") == -lo
    ]
    assert len(rebases) == 1
    for address, (_, j) in zip(addresses, accesses):
        assert address.lane_range[0] == 0
        assert address.const == lo * 16 + j
    _check_against_golden(
        graph,
        tensor,
        addresses,
        lambda t, i: (host(t), accesses[i][1]),
        (({id(tid): t}, t) for t in range(WAVE)),
    )


def test_a_lane_part_starting_at_zero_emits_no_rebase():
    b = _builder()
    graph, tensor = identity_graph((64, 16)), TensorDesc((64, 16), (16, 1), F32)
    tid = b.thread_id_x()
    before = _op_count(b)
    (address,) = emit_addresses(b, graph, tensor, [(tid, 0)])
    assert address.lane is not None and address.const == 0
    # The stride constant and one multiply; nothing is added to the lane.
    assert _op_count(b) - before == 2
    assert _count(b, "arith.add") == 0


# --- emitted IR text ------------------------------------------------------------------------------


def _golden_conv() -> tuple[IRBuilder, tuple[tt.ElementAddress, ...]]:
    b = _builder()
    graph, tensor = _conv(2, 6, 6, 3)
    conv = _ConvAccess(b, graph)
    k = conv.k_lane
    return b, emit_addresses(
        b, graph, tensor, [(conv.m, k), (conv.m, b.add(k, _c(b, 1)))]
    )


def _golden_two_inputs() -> tuple[IRBuilder, tuple[tt.ElementAddress, ...]]:
    b = _builder()
    row, col = _two_input_case(b)
    tensor = TensorDesc((256, 256), (256, 1), F32)
    return b, emit_addresses(b, identity_graph((256, 256)), tensor, [(row, col)])


def _golden_xor() -> tuple[IRBuilder, tuple[tt.ElementAddress, ...]]:
    b = _builder()
    return b, _xor_case(b)[2]


def _golden_rebase() -> tuple[IRBuilder, tuple[tt.ElementAddress, ...]]:
    b = _builder()
    value = b.sub(_c(b, 40), b.thread_id_x())
    tensor = TensorDesc((64, 16), (16, 1), F32)
    return b, emit_addresses(
        b, identity_graph((64, 16)), tensor, [(value, 0), (value, 1)]
    )


GOLDEN_CASES = {
    "conv": _golden_conv,
    "two_inputs": _golden_two_inputs,
    "xor": _golden_xor,
    "rebase": _golden_rebase,
}


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_emitted_ir_text_matches_its_reviewed_golden(case):
    b, _ = GOLDEN_CASES[case]()
    assert print_ir(b.kernel) + "\n" == (GOLDEN_DIR / f"emit_{case}.txt").read_text()


def _store_parts(b: IRBuilder, addresses: tuple[tt.ElementAddress, ...]) -> int:
    """Store every access's uniform terms and lane part (as i64) to a kernel parameter ``out``,
    one slot per part in access order, lane by lane; returns the number of slots per lane.
    """
    out = b.param("out", PtrType(I64, "global"))
    parts = [
        part
        for address in addresses
        for part in (
            *(t.value for t in address.uniform),
            *(() if address.lane is None else (b.zext(address.lane, I64),)),
        )
    ]
    lane_slot = b.mul(
        b.add(b.mul(b.block_id_x(), _c(b, WAVE)), b.thread_id_x()), _c(b, len(parts))
    )
    for slot, part in enumerate(parts):
        wide = part if part.type == I64 else b.sext(part, I64)
        b.global_store(out, b.add(lane_slot, _c(b, slot)), wide)
    return len(parts)


# One target of each family the graphs' ops (unsigned compare, no-unsigned-wrap add, range
# marker) lower for: CDNA, RDNA and gfx1250.
GOLDEN_ARCHES = ("gfx90a", "gfx942", "gfx950", "gfx1151", "gfx1201", "gfx1250")


def _golden_kernel(case: str):
    b, addresses = GOLDEN_CASES[case]()
    _store_parts(b, addresses)
    return b.kernel


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("arch", GOLDEN_ARCHES)
@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_every_golden_case_lowers_to_the_same_ir_in_both_engines(case, arch, flavor):
    assert lower_with_both_engines(_golden_kernel(case), arch, flavor)


def _require_comgr_target(arch: str) -> None:
    """Skip when the loaded comgr cannot target ``arch`` (an older ROCm lacks the newest
    targets); after this, any compile error is the kernel's and fails the test."""
    comgr = pytest.importorskip("rocke.runtime.comgr")
    try:
        isas = comgr.supported_isas()
    except comgr.ComgrError as exc:
        pytest.skip(f"comgr unavailable: {exc}")
    if arch not in {name.split("--")[-1].split(":")[0] for name in isas}:
        pytest.skip(f"the loaded comgr cannot target {arch}")


@pytest.mark.parametrize("arch", GOLDEN_ARCHES)
@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_every_golden_case_compiles(case, arch):
    compile_kernel = pytest.importorskip("rocke.helpers.compile").compile_kernel
    _require_comgr_target(arch)
    assert compile_kernel(_golden_kernel(case), arch=arch).hsaco


# --- errors ----------------------------------------------------------------------------------------


def test_an_unguarded_block_origin_into_an_unmerge_raises_the_dividend_message():
    graph, tensor = _conv(2, 6, 6, 3)
    b = _builder()
    tid = b.thread_id_x()
    m = b.add(b.mul(b.block_id_x(), _c(b, 16)), b.div(tid, _c(b, 4)))
    with _raises(
        ValueError,
        "unmerge of 'm' needs its dividend in [0, 2^31) for magic division, but its range is "
        "[0, +inf) -- bound the origin with exit_if_tile_outside(b, origin, length), or state a "
        "trusted range with runtime_value(b, v, at_most=...)",
    ):
        emit_addresses(b, graph, tensor, [(m, 0)])
    b = _builder()
    conv = _ConvAccess(b, graph)
    emit_addresses(b, graph, tensor, [(conv.m, 0)])


def test_a_lane_part_outside_32_bits_raises():
    b = _builder()
    loaded = b.global_load(b.param("idx", PtrType(I32, "global")), b.thread_id_x(), I32)
    graph, tensor = identity_graph((64,)), TensorDesc((64,), (1,), F32)
    with _raises(
        ValueError,
        "lane part of 'axis0' has range [-inf, +inf), outside 32 bits -- per-lane offsets must "
        "fit i32; move the large part into the uniform coordinate",
    ):
        emit_addresses(b, graph, tensor, [(loaded,)])
    b = _builder()
    big = TensorDesc((64, 4), (1 << 26, 1), F32)
    with _raises(
        ValueError,
        "lane part of 'axis0' has range [0, 4227858433), outside 32 bits -- per-lane offsets "
        "must fit i32; move the large part into the uniform coordinate",
    ):
        emit_addresses(b, identity_graph((64, 4)), big, [(b.thread_id_x(), 0)])


def test_every_malformed_access_is_refused_before_anything_is_emitted():
    b = _builder()
    graph = identity_graph((64, 16))
    tensor = TensorDesc((64, 16), (16, 1), F32)
    tid = b.thread_id_x()
    before = _op_count(b)
    with _raises(
        TypeError,
        "access 0 is a list, not a tuple -- give each access as a tuple of coordinates in "
        "inputs(...) order",
    ):
        emit_addresses(b, graph, tensor, [[tid, 0]])
    with _raises(
        TypeError,
        "access 1 is a dict, not a tuple -- give each access as a tuple of coordinates in "
        "inputs(...) order",
    ):
        emit_addresses(b, graph, tensor, [(tid, 0), {"axis0": tid, "axis1": 0}])
    with _raises(
        TypeError,
        "coordinate 1 ('axis1') of access 0 is a float -- a coordinate is a plain i32 IR value "
        "or an int",
    ):
        emit_coords(b, graph, (tid, 1.0))
    wide = b.sext(tid, I64)
    before = _op_count(b)
    with _raises(
        TypeError,
        "coordinate 0 ('axis0') of access 0 is a Value of type i64 -- a coordinate is a plain "
        "i32 IR value or an int",
    ):
        emit_addresses(b, graph, tensor, [(wide, 0)])
    with _raises(
        ValueError,
        "access 0 gives 1 coordinates, the graph has 2 inputs ('axis0', 'axis1') -- give one "
        "coordinate per input, in inputs(...) order",
    ):
        emit_addresses(b, graph, tensor, [(tid,)])
    with _raises(
        ValueError, "graph has 2 outputs ('axis0', 'axis1') but the tensor has rank 1"
    ):
        emit_addresses(b, graph, TensorDesc((64,), (1,), F32), [(tid, 0)])
    with _raises(
        ValueError,
        "structured address needs non-negative strides; tensor axis 0 has stride -16",
    ):
        emit_addresses(b, graph, TensorDesc((64, 16), (-16, 1), F32), [(tid, 0)])
    assert _op_count(b) == before


# --- GPU: each lane's parts against the golden ----------------------------------------------------


def _lane_addresses(build, n_blocks: int, params: dict[str, int]):
    """Run a kernel whose lanes store each access's parts; return ``(addresses, sums)`` where
    ``sums[block][tid][i]`` is access ``i``'s address that lane computed (its parts summed on
    the host in Python integers, plus the constant)."""
    import numpy as np

    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    b = _builder()
    for name in params:
        b.param(name, I32)
    addresses = build(b)
    slots = _store_parts(b, addresses)
    artifact = compile_kernel(b.kernel, arch=current_device_arch())
    signature = SignatureBuilder()
    for name in params:
        signature = signature.scalar(name, "i32")
    signature = signature.ptr("out", "i64")
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=signature.build(),
    )
    host = np.zeros(n_blocks * WAVE * slots, dtype=np.int64)
    rt = Runtime()
    dev = DeviceMem(host.nbytes)
    rt.memcpy_h2d(dev.ptr(), as_u8_buffer(host), host.nbytes)
    launcher(
        {"out": dev, **params},
        config=LaunchConfig(grid=(n_blocks, 1, 1), block=(WAVE, 1, 1)),
    )
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(host), dev.ptr(), host.nbytes)
    stored = host.reshape(n_blocks, WAVE, slots).tolist()
    sums = []
    for blk in range(n_blocks):
        rows = []
        for t in range(WAVE):
            values, at = [], 0
            for address in addresses:
                n_parts = len(address.uniform) + (address.lane is not None)
                values.append(sum(stored[blk][t][at : at + n_parts]) + address.const)
                at += n_parts
            rows.append(values)
        sums.append(rows)
    return addresses, sums


def test_gpu_conv_addresses_at_the_halo_tile_and_an_interior_tile():
    """Block 0 is the tile at the image corner (negative halo constants, origin 0); blocks 1
    and 2 are interior tiles."""
    current_device_arch()
    graph, tensor = _conv(2, 6, 6, 3)
    constants = (0, 1, 26)

    def build(b):
        conv = _ConvAccess(b, graph)
        accesses = [
            (conv.m, b.add(conv.k_lane, _c(b, c)) if c else conv.k_lane)
            for c in constants
        ]
        return emit_addresses(b, graph, tensor, accesses)

    _, sums = _lane_addresses(build, 3, {})
    for blk, t in itertools.product(range(3), range(WAVE)):
        m, k = blk * 16 + t // 4, (t % 4) * 8
        for i, c in enumerate(constants):
            assert sums[blk][t][i] == transform_address_of(graph, tensor, (m, k + c))


def test_gpu_an_address_past_two_to_the_31_is_carried_by_the_wide_uniform_term():
    current_device_arch()
    rows, cols = 1 << 20, 4096
    tensor = TensorDesc((rows, cols), (cols, 1), F16)
    graph = identity_graph((rows, cols))
    start = (1 << 19) + 64

    def build(b):
        origin = _guarded(b, b.get_param("row0"), rows)
        tid = b.thread_id_x()
        row = b.add(origin, b.div(tid, _c(b, 4)))
        col = b.add(
            b.mul(b.block_id_x(), _c(b, 32)), b.mul(b.mod(tid, _c(b, 4)), _c(b, 8))
        )
        return emit_addresses(b, graph, tensor, [(row, col)])

    (address,), sums = _lane_addresses(build, 2, {"row0": start})
    assert address.uniform[0].value.type == I64
    for blk, t in itertools.product(range(2), range(WAVE)):
        coords = (start + t // 4, blk * 32 + (t % 4) * 8)
        expected = transform_address_of(graph, tensor, coords)
        assert expected >= 2**31 and sums[blk][t][0] == expected


def test_gpu_conv_over_more_than_two_to_the_31_elements_keeps_the_lane_part_in_one_image():
    """The input holds more than 2^31 elements but one image far fewer: only n's quotient
    reaches a uniform term, and the per-lane part stays below one image plus the carry.
    """
    current_device_arch()
    n, h, w, c = 131080, 16, 16, 64
    y_len = x_len = 3
    graph, tensor = _conv(n, h, w, c, y=y_len, x=x_len)
    assert n * h * w * c >= 2**31
    m_length = graph.inputs["m"]
    constants = (0, 32, 200)

    def build(b):
        tid = b.thread_id_x()
        tile = b.add(b.get_param("m0"), b.mul(b.block_id_x(), _c(b, 16)))
        m = b.add(_guarded(b, tile, m_length), b.div(tid, _c(b, 4)))
        k_lane = b.mul(b.mod(tid, _c(b, 4)), _c(b, 8))
        accesses = [(m, b.add(k_lane, _c(b, k)) if k else k_lane) for k in constants]
        return emit_addresses(b, graph, tensor, accesses)

    m0 = m_length - 32
    addresses, sums = _lane_addresses(build, 2, {"m0": m0})
    # The bound re-derived from the pieces: n's per-lane part is at most the carry 1, ho and wo
    # at most H - 1 and W - 1 (inner pieces), y at most its carry 1, x at most X - 1 and c at
    # most C - 1; each times its stride.
    largest = (
        1 * (h * w * c) + (h - 1 + 1) * (w * c) + (w - 1 + x_len - 1) * c + (c - 1)
    )
    assert largest < 2**31
    for address in addresses:
        assert [t.inputs for t in address.uniform] == [frozenset({0})]
        assert address.uniform[0].value.type == I64
        assert 0 <= address.lane_range[0] and address.lane_range[1] <= largest + 1
    for blk, t in itertools.product(range(2), range(WAVE)):
        for i, k in enumerate(constants):
            coords = (m0 + blk * 16 + t // 4, (t % 4) * 8 + k)
            expected = transform_address_of(graph, tensor, coords)
            assert sums[blk][t][i] == expected
    assert max(transform_address_of(graph, tensor, (m0 + 31, 0)), 0) >= 2**31
