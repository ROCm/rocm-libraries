# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The block-origin guard: ``exit_if_tile_outside``.

A block origin such as ``block_id_y * TM`` says nothing in the IR about where the tile sits
relative to the tensor. ``exit_if_tile_outside(b, origin, length)`` makes ``0 <= origin < length``
true -- a workgroup whose origin is outside returns before doing anything -- and returns the origin
behind a range marker, so the IR range walk reads the guard's fact from every use of the returned
value. These tests pin:

- what the guard emits: one unsigned compare of the raw origin, one uniform branch to a return in
  the kernel's top level, then the marker; the same IR from both engines; on gfx90a a scalar
  compare whose branch leads to ``s_endpgm``;
- the facts the walk reads through the returned value -- the range snapped to the origin's
  multiple-of fact, the whole tile proven when the length is a tile multiple -- and the facts of
  the origins the guard is fed (raw products, shifts, a grouped grid remap, loop counters, the
  tail-tile origin, wave offsets added to the guarded value);
- that the guard refuses, each with its message, an origin or a length that can differ between
  the lanes or the waves of a workgroup, a call inside a loop or an if, an int origin and a length
  Value with no stated facts;
- on the GPU, that an over-launched grid writes nothing past the tensor and that a workgroup whose
  remapped origin is negative does nothing.

Whether a side is proven is asked of the shared bound-check rule directly
(``required_checks_within`` over the identity graph of the tensor's lengths); the verbs use the
same rule once they read the walk.
"""

from __future__ import annotations

import ctypes
import re
import struct

import numpy as np
import pytest

from rocke.core.ir import F16, F32, I32, I64, IRBuilder, PtrType, Value
from rocke.helpers.tiling import (
    check_gemm_shapes,
    exit_if_tile_outside,
    load_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.memory import cooperative_load_desc
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph
from rocke.helpers.tiling.tensor_transforms.ranges import (
    UNBOUNDED,
    AxisBound,
    BoundCheck,
    Congruence,
    Interval,
    _RuntimeEnd,
    required_checks_within,
)
from rocke.helpers.tiling.tensor_transforms.walk import split_coordinate, value_facts
from rocke.tests.engines import lower_with_both_engines

FLAVORS = ("llvm20", "llvm22", "llvm23")
WAVE = 64


def _builder(name: str = "guard_test", workgroup_size: int = 256) -> IRBuilder:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = workgroup_size
    return b


def _raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


def _all_ops(b: IRBuilder) -> list:
    def walk(ops):
        for op in ops:
            yield op
            for region in op.regions:
                yield from walk(region.ops)

    return list(walk(b.kernel.body.ops))


def _c(b: IRBuilder, value: int) -> Value:
    return b.const_i32(value)


def _block_origin(b: IRBuilder, tile: int, axis: str = "y") -> Value:
    block = {"x": b.block_id_x, "y": b.block_id_y, "z": b.block_id_z}[axis]()
    return b.mul(block, _c(b, tile))


def _length(b: IRBuilder, name: str, **facts) -> Value:
    """A runtime length: the kernel parameter ``name`` declared ``ge=0`` and ``facts``."""
    return b.param(name, I32, **{"ge": 0, **facts})


def _loaded(b: IRBuilder, index: Value, **param_attrs) -> Value:
    """A value read from memory through the parameter ``lens`` at ``index``, made the same for
    the whole wave: the GPU can see it, the host cannot."""
    lens = b.param("lens", PtrType(I32, "global"), **param_attrs)
    return b.readfirstlane(b.global_load_i32(lens, index))


def _checks(
    b: IRBuilder,
    lengths: tuple[int | Value, ...],
    origins: tuple[Value | int, ...],
    extents: tuple[int, ...],
) -> tuple[BoundCheck, ...]:
    """The checks the shared rule needs for a plain window at ``origins`` covering ``extents``
    elements per axis, over a tensor of ``lengths``: each axis's range is its origin's walked
    range plus the tile's ``[0, extent)``."""
    ranges = tuple(
        value_facts(b, origin)[0].plus(Interval(0, extent))
        for origin, extent in zip(origins, extents)
    )
    bounds = tuple(AxisBound.of(0, length) for length in lengths)
    return required_checks_within(identity_graph(lengths), ranges, bounds)


def _sides(checks: tuple[BoundCheck, ...]) -> dict[str, tuple[bool, bool]]:
    """Which sides of which axis stay checked: ``{axis name: (lower checked, upper checked)}``."""
    return {c.coordinate: (c.lo is not None, c.hi is not None) for c in checks}


# --- what the guard emits ----------------------------------------------------------------------


def _guarded_store_kernel(name: str = "guard_emit") -> tuple[IRBuilder, Value, Value]:
    """The plan's example: ``m0 = exit_if_tile_outside(b, block_id_y * 128, M)``, then a store
    of ``m0`` so the kernel has a use after the guard. Returns the builder, the raw origin and the
    guarded value."""
    b = _builder(name, workgroup_size=WAVE)
    y = b.param("Y", PtrType(I32, "global"))
    ms = _length(b, "M", le=4096)
    raw = _block_origin(b, 128)
    m0 = exit_if_tile_outside(b, raw, ms)
    b.global_store(y, b.thread_id_x(), m0)
    b.ret()
    return b, raw, m0


def test_the_guard_is_one_unsigned_compare_one_branch_and_one_return():
    b, raw, m0 = _guarded_store_kernel()
    top = list(b.kernel.body.ops)
    names = [op.name for op in top]
    marker = m0.op
    # The guard's ops: everything between the raw origin and its marker.
    start, end = top.index(raw.op), top.index(marker)
    guard = top[start + 1 : end]
    assert [op.name for op in guard] == ["arith.cmp", "arith.not", "scf.if"]
    compare, _, branch = guard
    assert compare.attrs["pred"] == "ult"
    assert compare.operands[0] is raw  # the compare reads the RAW origin
    assert [op.name for op in branch.regions[0].ops] == ["cf.return"]
    # The marker follows the branch, states [0, M) on the raw origin, names its writer, and is
    # what was returned.
    assert marker.operands[0] is raw and marker.attrs["ge"] == 0
    assert marker.operands[1] is b.get_param("M")
    assert marker.attrs["by"] == "exit_if_tile_outside"
    assert names.count("arith.assume_range") == 1
    assert all(
        op in top for op in guard
    )  # top-level region: never inside a loop or an if


def test_both_engines_emit_the_same_ir_for_the_guard():
    b, _, _ = _guarded_store_kernel("guard_both_engines")
    for flavor in FLAVORS:
        text = lower_with_both_engines(b.kernel, "gfx90a", llvm_flavor=flavor)
        assert len(re.findall(r"icmp ult i32 %mul\d+, %M\b", text)) == 1, text
        assert text.count("ret void") == 2  # the early exit and the kernel end


def test_the_guard_is_a_scalar_compare_and_a_branch_to_the_end_in_the_isa():
    harness = pytest.importorskip(
        "rocke.tests.helpers.tiling.tensor_transforms._harness"
    )
    b, _, _ = _guarded_store_kernel("guard_isa")
    isa = harness.gfx90a_isa(b)
    body = isa[isa.index("<guard_isa>:") :]
    lines = [line.strip() for line in body.splitlines()[1:] if line.strip()]
    offsets = {
        int(m.group(1), 16): line.split()[0]
        for line in lines
        if (m := re.search(r"// ([0-9A-F]+):", line))
    }
    # The guard's compare runs on the scalar unit; no per-lane compare exists at all.
    assert harness.isa_count(isa, r"s_cmp_lt_u32") == 1
    assert harness.isa_count(isa, r"v_cmp\w*") == 0
    # Every conditional branch is forward (prologue only, no loop), and one of them is the
    # guard's: it leads to the s_endpgm that ends the program.
    base = min(offsets)
    targets = []
    for line in lines:
        m = re.match(r"s_cbranch_\w+ \S+ .*<guard_isa\+0x([0-9a-f]+)>", line)
        if m:
            here = int(re.search(r"// ([0-9A-F]+):", line).group(1), 16)
            target = base + int(m.group(1), 16)
            assert target > here
            targets.append(offsets[target])
    assert "s_endpgm" in targets


# --- the facts the walk reads through the returned value -----------------------------------------


def test_the_returned_value_walks_to_the_guard_range_snapped():
    b = _builder()
    ms = _length(b, "M", le=4096)
    m0 = exit_if_tile_outside(b, _block_origin(b, 128), ms)
    assert value_facts(b, m0) == (Interval(0, ms), Congruence(128, 0))


@pytest.mark.parametrize(
    "multiple_of, end_shift",
    [
        (
            128,
            -127,
        ),  # a tile multiple: the end snaps to the last tile start, [0, M - 127)
        (64, 0),  # not a multiple of 128: the end stays M
    ],
)
def test_a_tile_multiple_length_snaps_the_end(multiple_of, end_shift):
    b = _builder()
    ms = _length(b, "M", le=4096, multiple_of=multiple_of)
    m0 = exit_if_tile_outside(b, _block_origin(b, 128), ms)
    assert value_facts(b, m0) == (
        Interval(0, _RuntimeEnd(ms, end_shift)),
        Congruence(128, 0),
    )


def test_an_int_length_gives_a_numeric_snapped_end():
    b = _builder()
    m0 = exit_if_tile_outside(b, _block_origin(b, 128), 1000)
    assert value_facts(b, m0) == (Interval(0, 897), Congruence(128, 0))


def test_the_raw_product_and_the_raw_shift_carry_less_than_the_guarded_value():
    # The raw origins: a product by a constant keeps its lower end (block ids are >= 0), a shift
    # keeps no end; both keep the multiple-of fact. Guarded, both give the guard's facts.
    b = _builder()
    ms = _length(b, "M", le=4096)
    product = _block_origin(b, 128)
    shifted = b.shl(b.block_id_y(), _c(b, 7))
    assert value_facts(b, product) == (Interval(0, None), Congruence(128, 0))
    assert value_facts(b, shifted) == (UNBOUNDED, Congruence(128, 0))
    from_product = exit_if_tile_outside(b, product, ms)
    from_shift = exit_if_tile_outside(b, shifted, ms)
    assert value_facts(b, from_product) == value_facts(b, from_shift)
    assert value_facts(b, from_product) == (Interval(0, ms), Congruence(128, 0))
    # The shared rule answers the same for both: lower side proven, upper side checked against M.
    product_checks = _checks(b, (ms,), (from_product,), (128,))
    shift_checks = _checks(b, (ms,), (from_shift,), (128,))
    assert product_checks == shift_checks
    assert _sides(product_checks) == {"axis0": (False, True)}


# --- partial facts on the origins the guard is fed -----------------------------------------------


def test_a_parameter_times_a_tile_keeps_its_multiple_of_fact():
    b = _builder()
    p = b.param("p", I32)
    assert value_facts(b, b.mul(p, _c(b, 128))) == (UNBOUNDED, Congruence(128, 0))


def _grouped_remap_origin(b: IRBuilder, tile: int, group: int) -> Value:
    """The M origin of a 1-D grid remapped in-kernel into groups of ``group`` M-blocks (for L2
    reuse of the B tiles), from the raw parameters M and N."""
    m, n = b.param("M_raw", I32), b.param("N_raw", I32)
    bid = b.block_id_x()
    m_blocks = b.div(b.add(m, _c(b, tile - 1)), _c(b, tile))
    n_blocks = b.div(b.add(n, _c(b, tile - 1)), _c(b, tile))
    per_group = b.mul(_c(b, group), n_blocks)
    group_id = b.div(bid, per_group)
    first_m = b.mul(group_id, _c(b, group))
    group_rows = b.smin(b.sub(m_blocks, first_m), _c(b, group))
    in_group = b.mod(bid, per_group)
    m_block = b.add(first_m, b.mod(in_group, group_rows))
    return b.mul(m_block, _c(b, tile))


def test_a_grouped_grid_remap_keeps_the_tile_multiple_and_the_guard_bounds_it():
    # The remap divides by values from raw parameters of unknown sign: no end survives, but the
    # multiple-of fact of the final product does.
    tile = 128
    b = _builder()
    raw = _grouped_remap_origin(b, tile, group=8)
    raw_range, raw_congruence = value_facts(b, raw)
    assert raw_congruence == Congruence(tile, 0)
    assert raw_range.lo is None
    ms = _length(b, "M", le=1 << 20, multiple_of=tile)
    m0 = exit_if_tile_outside(b, raw, ms)
    assert value_facts(b, m0) == (
        Interval(0, _RuntimeEnd(ms, 1 - tile)),
        Congruence(tile, 0),
    )
    assert _checks(b, (ms,), (m0,), (tile,)) == ()


def test_an_int_tile_multiple_length_bounds_the_remapped_origin_to_the_last_tile():
    tile = 128
    b = _builder()
    m0 = exit_if_tile_outside(b, _grouped_remap_origin(b, tile, group=8), 4096)
    assert value_facts(b, m0) == (Interval(0, 4096 - tile + 1), Congruence(tile, 0))


# --- an origin with no derivable multiple-of fact ------------------------------------------------


def test_a_kernel_parameter_origin_gets_the_guard_range_and_no_multiple():
    b = _builder()
    origin = exit_if_tile_outside(b, b.param("row0", I32), 1000)
    assert value_facts(b, origin) == (Interval(0, 1000), Congruence(1, 0))


def test_a_readfirstlane_of_a_read_only_load_gets_the_guard_range():
    b = _builder()
    starts = b.param("starts", PtrType(I32, "global"), readonly=True, noalias=True)
    loaded = b.readfirstlane(b.global_load_i32(starts, b.block_id_x()))
    origin = exit_if_tile_outside(b, loaded, 1000)
    assert value_facts(b, origin) == (Interval(0, 1000), Congruence(1, 0))


def test_a_loaded_origin_without_readfirstlane_is_per_lane():
    b = _builder()
    starts = b.param("starts", PtrType(I32, "global"), readonly=True, noalias=True)
    loaded = b.global_load_i32(starts, b.block_id_x())
    with _raises(
        ValueError,
        f"exit_if_tile_outside: origin {loaded.name} has a per-lane part (it depends on "
        "gpu.thread_id or a per-lane load) -- the guard ends the whole workgroup, so the origin "
        "must be the same for every lane; add per-wave offsets to the value it returns",
    ):
        exit_if_tile_outside(b, loaded, 1000)


# --- loop counters (the walk's loop-counter row, relied on by the tail-tile design) -------------


def _counter(b: IRBuilder, start: int | Value, stop: int | Value, step: int):
    start = _c(b, start) if isinstance(start, int) else start
    stop = _c(b, stop) if isinstance(stop, int) else stop
    return b.scf_for(start, stop, _c(b, step))


def test_an_int_bounded_loop_counter_has_the_tight_range():
    b = _builder()
    with _counter(b, 0, 256, 32) as k:
        assert value_facts(b, k) == (Interval(0, 225), Congruence(32, 0))


@pytest.mark.parametrize(
    "start, stop, step",
    [
        (5, 5, 4),  # an empty range
        (
            0,
            20 - 31,
            32,
        ),  # an int K = 20 below TK - 1 = 31: the stop K - (TK - 1) is below 0
    ],
)
def test_a_loop_that_never_runs_gives_its_start_and_builds(start, stop, step):
    b = _builder()
    with _counter(b, start, stop, step) as k:
        assert value_facts(b, k) == (
            Interval(start, start + 1),
            Congruence(step, start),
        )


@pytest.mark.parametrize("multiple_of", [1, 32])
def test_a_full_tile_loop_over_a_runtime_length_proves_its_k_window(multiple_of):
    # for k in [0, K - (TK - 1)) step TK: k + TK <= K by identity, aligned (K a multiple of TK)
    # and ragged alike.
    tk = 32
    b = _builder()
    ks = _length(b, "K", ge=tk, le=4096, multiple_of=multiple_of)
    with _counter(b, 0, b.add(ks, _c(b, 1 - tk)), tk) as k:
        assert value_facts(b, k) == (
            Interval(0, _RuntimeEnd(ks, 1 - tk)),
            Congruence(tk, 0),
        )
        assert _checks(b, (ks,), (k,), (tk,)) == ()


def test_a_checked_length_over_a_tighter_derived_value_is_its_stated_self_everywhere():
    # The derived range of smin(loaded, 100) is tighter than the checked [0, 1001); a runtime end
    # is that checked Value itself, so the guard's end, the loop's end and the tensor's length
    # are one Value and every identity proof reaches it.
    tk = 32
    b = _builder()
    raw = b.smin(_loaded(b, b.block_id_z(), readonly=True, noalias=True), _c(b, 100))
    ls = b.gpu_checked_value(raw, ge=0, le=1000)
    m0 = exit_if_tile_outside(b, _block_origin(b, tk), ls)
    assert value_facts(b, m0)[0] == Interval(0, ls)
    with _counter(b, 0, b.add(ls, _c(b, 1 - tk)), tk) as k:
        assert value_facts(b, k)[0] == Interval(0, _RuntimeEnd(ls, 1 - tk))
        assert value_facts(b, k)[0].hi.envelope == Interval(0, 1001)
        assert _checks(b, (ls, ls), (m0, k), (1, tk)) == ()


def test_the_tail_tile_origin_has_numeric_ends_and_needs_only_its_upper_check():
    tk = 32
    b = _builder()
    ks = _length(b, "K", le=4096)
    k_tail = b.mul(b.div(ks, _c(b, tk)), _c(b, tk))
    assert value_facts(b, k_tail) == (Interval(0, 4097), Congruence(tk, 0))
    assert _sides(_checks(b, (ks,), (k_tail,), (tk,))) == {"axis0": (False, True)}


@pytest.mark.parametrize(
    "length, upper_checked",
    [
        (1024, False),  # a multiple of the tile: the whole tile is inside
        (
            1000,
            True,
        ),  # not a multiple: the last tile overhangs, its upper side stays checked
    ],
)
def test_an_int_length_proves_the_whole_tile_only_when_a_tile_multiple(
    length, upper_checked
):
    b = _builder()
    m0 = exit_if_tile_outside(b, _block_origin(b, 128), length)
    expected = {"axis0": (False, True)} if upper_checked else {}
    assert _sides(_checks(b, (length,), (m0,), (128,))) == expected


# --- wave offsets: plain adds on the guarded value -----------------------------------------------


@pytest.mark.parametrize(
    "waves_m, waves_n, m_offset, origin_m, origin_n",
    [
        # 2x2 waves of 64x64 on a 128x128 tile; M = N = 256.
        (2, 2, Interval(0, 65), Interval(0, 193), Interval(0, 193)),
        # 4x1 waves of 32x128.
        (4, 1, Interval(0, 97), Interval(0, 225), Interval(0, 129)),
    ],
)
def test_wave_offsets_split_into_the_guarded_origin_and_a_per_lane_offset(
    waves_m, waves_n, m_offset, origin_m, origin_n
):
    tile, length = 128, 256
    warp_m, warp_n = tile // waves_m, tile // waves_n
    b = _builder(workgroup_size=WAVE * waves_m * waves_n)
    m0 = exit_if_tile_outside(b, _block_origin(b, tile, "y"), length)
    n0 = exit_if_tile_outside(b, _block_origin(b, tile, "x"), length)
    wave = b.div(b.thread_id_x(), _c(b, WAVE))
    wm, wn = b.div(wave, _c(b, waves_n)), b.mod(wave, _c(b, waves_n))
    row = b.add(m0, b.mul(wm, _c(b, warp_m)))
    col = b.add(n0, b.mul(wn, _c(b, warp_n)))
    split = split_coordinate(b, row)
    assert [part.value for part in split.uniform] == [m0]
    assert split.lane_range == m_offset
    assert value_facts(b, row) == (origin_m, Congruence(warp_m, 0))
    assert value_facts(b, col)[0] == origin_n
    # The C store window of each wave is proven on both axes.
    assert _checks(b, (length, length), (row, col), (warp_m, warp_n)) == ()


def test_a_wave_offset_from_a_loaded_value_leaves_its_side_checked():
    b = _builder()
    offsets = b.param("offsets", PtrType(I32, "global"), readonly=True, noalias=True)
    m0 = exit_if_tile_outside(b, _block_origin(b, 128), 1024)
    row = b.add(m0, b.global_load_i32(offsets, b.thread_id_x()))
    make_window(make_tensor_desc((1024, 64), (64, 1), F32), (row, 0))  # builds
    assert _sides(_checks(b, (1024,), (row,), (64,))) == {"axis0": (True, True)}


def test_a_guarded_batch_index_needs_no_check():
    b = _builder()
    gs = _length(b, "G", ge=1, le=64)
    g = exit_if_tile_outside(b, b.block_id_z(), gs)
    assert _checks(b, (gs,), (g,), (1,)) == ()


# --- origins and lengths that differ between waves are refused ----------------------------------


def test_a_per_wave_origin_is_refused():
    b = _builder()
    wave = b.readfirstlane(b.div(b.thread_id_x(), _c(b, WAVE)))
    with _raises(
        ValueError,
        f"exit_if_tile_outside: origin {wave.name} can differ between the waves of a workgroup "
        f"(it depends on {_tid_leaf(b)}) -- the guard ends whole workgroups; compute it from "
        "block ids, kernel parameters and constants, and add per-wave offsets to the value it "
        "returns",
    ):
        exit_if_tile_outside(b, wave, 1000)


def _tid_leaf(b: IRBuilder) -> str:
    tid = next(op for op in _all_ops(b) if op.name == "gpu.thread_id")
    return f"{tid.result.name} (gpu.thread_id)"


def test_a_value_loaded_at_a_per_wave_address_is_refused():
    b = _builder()
    starts = b.param("starts", PtrType(I32, "global"), readonly=True, noalias=True)
    wave = b.readfirstlane(b.div(b.thread_id_x(), _c(b, WAVE)))
    origin = b.readfirstlane(b.global_load_i32(starts, wave))
    with pytest.raises(ValueError, match="can differ between the waves of a workgroup"):
        exit_if_tile_outside(b, origin, 1000)


def test_workgroup_uniform_origins_are_accepted():
    b = _builder()
    exit_if_tile_outside(b, b.readfirstlane(b.param("row0", I32)), 1000)
    checked = b.gpu_checked_value(_block_origin(b, 64, "x"), ge=0, le=1 << 20)
    exit_if_tile_outside(b, checked, 1 << 20)


def _loaded_length(b: IRBuilder, index: Value, **param_attrs) -> Value:
    return b.gpu_checked_value(_loaded(b, index, **param_attrs), ge=0, le=4096)


def test_a_length_loaded_at_a_block_index_through_a_read_only_parameter_is_accepted():
    b = _builder()
    seqlen = _loaded_length(b, b.block_id_z(), readonly=True, noalias=True)
    m0 = exit_if_tile_outside(b, _block_origin(b, 64), seqlen)
    assert value_facts(b, m0)[0] == Interval(0, seqlen)


def test_a_length_loaded_at_a_per_wave_index_is_refused():
    # The read-only load is the leaf named: its index differs between waves, so the load does.
    b = _builder()
    seqlen = _loaded_length(
        b, b.div(b.thread_id_x(), _c(b, WAVE)), readonly=True, noalias=True
    )
    load = next(op for op in _all_ops(b) if op.name == "memref.global_load_typed")
    with _raises(
        ValueError,
        f"exit_if_tile_outside: length {seqlen.name} can differ between the waves of a "
        f"workgroup (it depends on {load.result.name} (memref.global_load_typed)) -- the guard ends whole workgroups, so every "
        "wave must test the same length; compute it from block ids, kernel parameters and "
        "constants, or load it through a kernel parameter declared readonly=True and "
        "noalias=True, at a workgroup-uniform index",
    ):
        exit_if_tile_outside(b, _block_origin(b, 64), seqlen)


@pytest.mark.parametrize("param_attrs", [{}, {"readonly": True}, {"noalias": True}])
def test_a_length_loaded_through_a_writable_parameter_is_refused(param_attrs):
    # Another pointer of the kernel may write that memory, so waves may read different values.
    b = _builder()
    seqlen = _loaded_length(b, b.block_id_z(), **param_attrs)
    load = next(op for op in _all_ops(b) if op.name == "memref.global_load_typed")
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"(it depends on {load.result.name} (memref.global_load_typed))"
        ),
    ):
        exit_if_tile_outside(b, _block_origin(b, 64), seqlen)


def test_a_length_checked_twice_is_another_value_and_proves_nothing_about_the_first():
    # Each gpu_checked_value call marks the raw value anew. A guard against one checked Value says
    # nothing about a tensor whose length is the other: that side stays checked, never wrong.
    b = _builder()
    raw = _loaded(b, b.block_id_z(), readonly=True, noalias=True)
    guard_length = b.gpu_checked_value(raw, ge=0, le=4096, multiple_of=128)
    tensor_length = b.gpu_checked_value(raw, ge=0, le=2048, multiple_of=128)
    m0 = exit_if_tile_outside(b, _block_origin(b, 128), guard_length)
    assert guard_length is not tensor_length
    assert _checks(b, (guard_length,), (m0,), (128,)) == ()
    assert _sides(_checks(b, (tensor_length,), (m0,), (128,))) == {
        "axis0": (False, True)
    }


# --- every other refusal -----------------------------------------------------------------------


def test_an_int_origin_is_refused():
    b = _builder()
    with _raises(
        TypeError,
        "exit_if_tile_outside takes the origin as an i32 IR value -- got the int 128; a "
        "build-time origin needs no guard, pass it to make_window directly",
    ):
        exit_if_tile_outside(b, 128, 1000)


def test_an_origin_of_another_kind_is_refused():
    b = _builder()
    with _raises(
        TypeError,
        "exit_if_tile_outside takes the origin as an i32 IR value -- got a float",
    ):
        exit_if_tile_outside(b, 1.5, 1000)


def test_an_i64_origin_gets_the_walk_type_error():
    b = _builder()
    origin = b.param("row0", I64)
    with _raises(
        TypeError,
        f"IR range walk: {origin.name} has type i64 -- a coordinate is an i32 value",
    ):
        exit_if_tile_outside(b, origin, 1000)


def test_a_checked_origin_is_guarded_as_it_is():
    b = _builder()
    checked = b.gpu_checked_value(_block_origin(b, 64, "x"), ge=0, le=1 << 20)
    guarded = exit_if_tile_outside(b, checked, 1 << 20)
    assert guarded.op.operands[0] is checked


_NO_FACTS = (
    "which has no stated facts. State them on its b.param declaration, or check it with "
    "b.gpu_checked_value; if you checked it with gpu_checked_value, pass the Value "
    "gpu_checked_value returned"
)


def test_a_length_with_no_stated_facts_is_refused():
    b = _builder()
    with _raises(ValueError, f"exit_if_tile_outside: length is M, {_NO_FACTS}"):
        exit_if_tile_outside(b, _block_origin(b, 64), b.param("M", I32))


def test_the_raw_value_of_a_checked_length_is_refused_with_the_value_to_pass():
    # The facts live only on the Value gpu_checked_value returned; a guard against the raw value
    # would prove nothing about a tensor whose length is the checked one.
    b = _builder()
    raw = _loaded(b, b.block_id_z(), readonly=True, noalias=True)
    b.gpu_checked_value(raw, ge=0, le=99)
    with _raises(
        ValueError, f"exit_if_tile_outside: length is {raw.name}, {_NO_FACTS}"
    ):
        exit_if_tile_outside(b, _block_origin(b, 16), raw)


def test_a_length_that_could_be_negative_or_has_no_upper_bound_is_refused():
    b = _builder()
    with _raises(
        ValueError,
        "exit_if_tile_outside: length is N, which could be negative (declared with no lower "
        'bound).\nFix: b.param("N", I32, ge=0, ...)',
    ):
        exit_if_tile_outside(b, _block_origin(b, 16), b.param("N", I32, le=99))
    with _raises(
        ValueError,
        "exit_if_tile_outside: length is K, which has no upper bound: state le= or lt=, the "
        "tightest bound it can reach",
    ):
        exit_if_tile_outside(b, _block_origin(b, 16), b.param("K", I32, ge=0))


def test_a_length_of_another_kind_is_refused():
    b = _builder()
    with _raises(
        TypeError,
        "exit_if_tile_outside takes the length as an int or an i32 Value with stated facts "
        "-- got a float",
    ):
        exit_if_tile_outside(b, _block_origin(b, 64), 1000.0)


@pytest.mark.parametrize("length", [0, -64])
def test_a_length_below_one_is_refused(length):
    b = _builder()
    with _raises(
        ValueError,
        f"exit_if_tile_outside: length {length} is below 1 -- a tensor length is positive (the "
        "tensor's own length, as passed to make_tensor_desc)",
    ):
        exit_if_tile_outside(b, _block_origin(b, 64), length)


_NOT_IN_PROLOGUE = (
    "exit_if_tile_outside emits an early exit and must be called in the kernel prologue -- it "
    "was called inside a loop or an if"
)


def test_a_guard_inside_a_loop_is_refused():
    b = _builder()
    origin = _block_origin(b, 64)
    with b.scf_for(_c(b, 0), _c(b, 4), _c(b, 1)), _raises(ValueError, _NOT_IN_PROLOGUE):
        exit_if_tile_outside(b, origin, 1000)


def test_a_guard_inside_an_if_is_refused():
    b = _builder()
    origin = _block_origin(b, 64)
    inside = b.cmp_ult(b.block_id_x(), _c(b, 4))
    with b.scf_if(inside), _raises(ValueError, _NOT_IN_PROLOGUE):
        exit_if_tile_outside(b, origin, 1000)


def test_a_per_lane_origin_is_refused():
    b = _builder()
    origin = b.add(_block_origin(b, 64), b.thread_id_x())
    with _raises(
        ValueError,
        f"exit_if_tile_outside: origin {origin.name} has a per-lane part (it depends on "
        "gpu.thread_id or a per-lane load) -- the guard ends the whole workgroup, so the origin "
        "must be the same for every lane; add per-wave offsets to the value it returns",
    ):
        exit_if_tile_outside(b, origin, 1000)


def test_a_thread_id_origin_without_a_workgroup_size_gets_the_walk_error():
    b = IRBuilder("guard_no_size")
    origin = b.thread_id_x()
    with pytest.raises(
        ValueError, match=r"^IR range walk: .* states no workgroup size"
    ):
        exit_if_tile_outside(b, origin, 1000)


def test_a_proven_origin_emits_no_compare_but_still_returns_the_marked_value():
    b = _builder()
    origin = b.mod(b.block_id_x(), _c(b, 4))  # [0, 4): inside [0, 8) on both sides
    guarded = exit_if_tile_outside(b, origin, 8)
    assert not any(op.name in ("arith.cmp", "scf.if") for op in _all_ops(b))
    assert guarded.op.name == "arith.assume_range"
    assert value_facts(b, guarded)[0] == Interval(0, 4)


# --- the marker is the guard's own -----------------------------------------------------------------


def test_two_guards_of_one_value_get_no_gpu_checked_value_advice():
    # check_gemm_shapes says "check it once" only for two gpu_checked_value calls on one value; two
    # guards of one origin, used as lengths, are a plain mismatch and get the layout advice.
    b = _builder("two_guards")
    origin = _block_origin(b, 16)
    first = exit_if_tile_outside(b, origin, 64)
    second = exit_if_tile_outside(b, origin, 64)
    a = make_tensor_desc((64, first), (64, 1), F16)
    bt = make_tensor_desc((64, second), (64, 1), F16)
    with _raises(
        ValueError,
        f"check_gemm_shapes: A's K (axis 1) is {first.name} but B's K (axis 1) is {second.name} "
        "-- GEMM tensors are read as A (M, K), B (N, K), C (M, N), memory order in the strides; "
        "a B stored N-contiguous (stride 1 on N) is make_tensor_desc((N, K), (1, ldb), dtype)",
    ):
        check_gemm_shapes(a, bt)


# --- GPU: an over-launched grid and a negative remapped origin ------------------------------------

TILE_M, TILE_N = 64, 16  # one row of the tile per lane, 16 columns
ROWS, COLS = 2 * TILE_M, 2 * TILE_N  # two tiles per axis
SIZE = ROWS * COLS
PAD = SIZE  # guard words before and after the tensor
FLAG_ROW = 8  # flags per block row


def _copy_kernel(name: str, *, row_shift: int) -> IRBuilder:
    """Copy a (128, 32) f32 tensor one 64x16 tile per workgroup, both block origins guarded; the
    row origin is ``(block_id_y - row_shift) * 64``. Each workgroup that passes the guard sets its
    flag."""
    b = _builder(name, workgroup_size=WAVE)
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    flags = b.param("flags", PtrType(I32, "global"))
    td = make_tensor_desc((ROWS, COLS), (COLS, 1), F32)
    raw_m = b.mul(b.sub(b.block_id_y(), _c(b, row_shift)), _c(b, TILE_M))
    m0 = exit_if_tile_outside(b, raw_m, ROWS)
    n0 = exit_if_tile_outside(b, _block_origin(b, TILE_N, "x"), COLS)
    block = b.add(b.mul(b.block_id_y(), _c(b, FLAG_ROW)), b.block_id_x())
    b.global_store(flags, block, _c(b, 1))
    tile = cooperative_load_desc(TILE_M, TILE_N, 1, vw=1)
    lane = b.thread_id_x()
    frag = load_fragment(b, x, make_window(td, (m0, n0)), tile, lane)
    store_fragment(b, y, make_window(td, (m0, n0)), frag, lane)
    b.ret()
    return b


def _gpu_arch() -> str | None:
    try:
        from rocke.runtime.hip_module import current_device, get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch(current_device())
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _run_copy(
    row_shift: int, grid: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Launch the copy on a ``grid`` of (x, y) workgroups; return Y with its guard words before
    and after the tensor, the flags (one row per block row), and X."""
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    kernel = _copy_kernel(f"guard_copy_{row_shift}", row_shift=row_shift).kernel
    sentinel = np.float32(-7.0)
    x = np.full(PAD + SIZE + PAD, np.nan, dtype=np.float32)
    x[PAD : PAD + SIZE] = np.arange(SIZE, dtype=np.float32)
    y = np.full_like(x, sentinel)
    flags = np.zeros(FLAG_ROW * FLAG_ROW, dtype=np.int32)
    runtime = Runtime()
    function = runtime.load_module(
        compile_kernel(kernel, arch=arch).hsaco
    ).get_function(kernel.name)
    buffers = []
    for host in (x, y, flags):
        dev = runtime.alloc(host.nbytes)
        runtime.memcpy_h2d(
            dev, (ctypes.c_ubyte * host.nbytes).from_buffer(host), host.nbytes
        )
        buffers.append(dev)
    x_dev, y_dev, flags_dev = buffers
    pad_bytes = PAD * 4
    runtime.launch_blocking(
        function,
        (grid[0], grid[1], 1),
        (WAVE, 1, 1),
        struct.pack("QQQ", x_dev + pad_bytes, y_dev + pad_bytes, flags_dev),
    )
    for host, dev in ((y, y_dev), (flags, flags_dev)):
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * host.nbytes).from_buffer(host), dev, host.nbytes
        )
    return y, flags.reshape(FLAG_ROW, FLAG_ROW), x


def test_gpu_an_over_launched_grid_touches_nothing_outside_the_tensor():
    # 4x4 workgroups for a 2x2-tile tensor: the twelve outside return at the guard.
    y, flags, x = _run_copy(row_shift=0, grid=(4, 4))
    assert np.array_equal(y[PAD : PAD + SIZE], x[PAD : PAD + SIZE])
    assert np.all(y[:PAD] == -7.0) and np.all(y[PAD + SIZE :] == -7.0)
    expected = np.zeros((FLAG_ROW, FLAG_ROW), dtype=np.int32)
    expected[:2, :2] = 1
    assert np.array_equal(flags, expected)


def test_gpu_a_negative_remapped_origin_exits():
    # Row origins (block_id_y - 1) * 64 = -64, 0, 64: the first block row returns at the guard.
    y, flags, x = _run_copy(row_shift=1, grid=(2, 3))
    assert np.array_equal(y[PAD : PAD + SIZE], x[PAD : PAD + SIZE])
    assert np.all(y[:PAD] == -7.0) and np.all(y[PAD + SIZE :] == -7.0)
    expected = np.zeros((FLAG_ROW, FLAG_ROW), dtype=np.int32)
    expected[1:3, :2] = 1
    assert np.array_equal(flags, expected)
