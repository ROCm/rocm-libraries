# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS is addressed through the tensor descriptor's strides.

An LDS buffer is a flat allocation sized from its tensor descriptor,
``b.smem_alloc(td.dtype, [td.element_span])``, and ``load_fragment`` / ``store_fragment`` give every
LDS access one element index, ``sum(position * stride)``, exactly as for global memory. These tests
check:

- the emitted index, evaluated per lane from the real IR, for a padded K-contiguous buffer, an
  N-contiguous ``(N, K)`` buffer and a stage of a rank-3 buffer positioned by its origin;
- each refusal: a buffer that is not flat, holds another element type or is smaller than the
  descriptor's span; the built-in swizzle on a buffer whose stride-1 axis is not last;
  ``TensorDesc.element_span`` on a negative stride;
- ``lds_tile_desc``: the convention-order ``(free, K)`` descriptor of a cooperative LDS tile puts every
  element at the same address the buffer-order ``(K, free)`` layout did;
- the in-tree LDS-staged demos keep their vector LDS accesses;
- on a gfx90a GPU: LDS round trips whose bytes land where the strides say, and a GEMM whose B tile is
  staged ``[K][N]`` and described ``(N, K)`` with strides ``(1, N)``, bit-exact.
"""

from __future__ import annotations

import collections
import inspect
import re

import numpy as np
import pytest

pytest.importorskip("rocke.core.ir", reason="rocke IR substrate required")

from rocke.core.ir import F16, F32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers.tiling import (
    TileMma,
    Tiling,
    cooperative_load_desc,
    cooperative_load_width,
    fill_fragment,
    lds_element_span,
    lds_read,
    lds_store,
    lds_tile_desc,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.emit import emit_tensor_coordinates
from rocke.helpers.tiling.lds_conflict import NumBuilder, desc_extents

_WAVE = 64

# A 16 x 16 tile over 64 lanes, each lane holding 4 elements along axis 1 (the vector axis when axis 1
# is stride 1), and the same tile with the 4-element runs along axis 0.
_TILE_RUN_AXIS1 = make_tile_desc(
    shape=[16, 16], thread_tile=[1, 4], thread_dist=[16, 4], wave_size=_WAVE
)
_TILE_RUN_AXIS0 = make_tile_desc(
    shape=[16, 16], thread_tile=[4, 1], thread_dist=[4, 16], wave_size=_WAVE
)


# ---- evaluating the emitted LDS indices --------------------------------------------------------------

_INT_OPS = {
    "arith.add": lambda x, y: x + y,
    "arith.sub": lambda x, y: x - y,
    "arith.mul": lambda x, y: x * y,
    "arith.div": lambda x, y: x // y,
    "arith.mod": lambda x, y: x % y,
    "arith.xor": lambda x, y: x ^ y,
    "arith.shl": lambda x, y: x << y,
}


def _lds_indices(kernel, lane: int) -> list[tuple[str, tuple[int, ...]]]:
    """Run the kernel's integer arithmetic for one lane and return, in program order, every LDS
    access as ``(op name, indices)``. Only straight-line i32 arithmetic is evaluated; that is all an
    LDS index is built from in these kernels."""
    values: dict[str, int] = {}
    accesses = []
    for op in kernel.body.ops:
        if op.name == "gpu.thread_id":
            values[op.results[0].name] = lane
        elif op.name == "arith.constant":
            values[op.results[0].name] = op.attrs["value"]
        elif op.name in _INT_OPS:
            x, y = (values.get(o.name) for o in op.operands)
            if x is not None and y is not None:
                values[op.results[0].name] = _INT_OPS[op.name](x, y)
        elif op.name in ("tile.smem_load_vN", "tile.smem_load"):
            accesses.append((op.name, tuple(values[o.name] for o in op.operands[1:])))
        elif op.name in ("tile.smem_store_vN", "tile.smem_store"):
            accesses.append((op.name, tuple(values[o.name] for o in op.operands[1:-1])))
    return accesses


def _expected_indices(tile, origin, strides, n_runs, lane):
    """The flat index of each register run start: sum((origin + coordinate) * stride) over the
    tile's axes -- the window's LAST axes -- plus origin * stride of each leading axis.
    """
    run = tile.register_count // n_runs
    lead = len(origin) - len(tile.shape)
    out = []
    for register in range(0, tile.register_count, run):
        coords = emit_tensor_coordinates(NumBuilder(lane), tile.layout, lane, register)
        tile_terms = sum(
            (o + c) * s
            for o, c, s in zip(origin[lead:], coords, strides[lead:], strict=True)
        )
        out.append(
            tile_terms + sum(o * s for o, s in zip(origin[:lead], strides[:lead]))
        )
    return out


def _load_then_store(window, tile, *, buffer_elements=None, swizzle=False):
    """A kernel that loads `tile` from an LDS buffer through `window` and stores it back."""
    b = IRBuilder("lds_index_probe")
    tid = b.thread_id_x()
    tensor = window.tensor
    size = buffer_elements if buffer_elements is not None else tensor.element_span
    lds = b.smem_alloc(tensor.dtype, [size], name_hint="lds")
    fragment = load_fragment(b, lds, window, tile, tid, lds_swizzle=swizzle)
    store_fragment(b, lds, window, fragment, tid, lds_swizzle=swizzle)
    b.ret()
    return b.kernel


def _check_indices(window, tile):
    kernel = _load_then_store(window, tile)
    smem_ops = [
        op for op in kernel.body.ops if op.name.startswith("tile.smem_load")
    ] + [op for op in kernel.body.ops if op.name.startswith("tile.smem_store")]
    assert smem_ops and all(op.attrs["rank"] == 1 for op in smem_ops)
    for lane in range(_WAVE):
        accesses = _lds_indices(kernel, lane)
        loads = [idx[0] for name, idx in accesses if "load" in name]
        stores = [idx[0] for name, idx in accesses if "store" in name]
        expected = _expected_indices(
            tile, window.origin, window.tensor.strides, len(loads), lane
        )
        assert loads == expected, f"lane {lane}"
        assert stores == expected, f"lane {lane}"


def test_padded_k_contiguous_index_is_row_times_row_stride_plus_column() -> None:
    # A (rows, K) buffer whose rows are padded to 24 elements: element (row, col) is at row*24 + col.
    td = make_tensor_desc((32, 32), (24, 1), F16)
    _check_indices(make_window(td, (16, 8)), _TILE_RUN_AXIS1)


def test_n_contiguous_index_is_n_plus_k_times_n_length() -> None:
    # A B tile staged [K][N] and described (N, K) with strides (1, N): element (n, k) is at n + k*N.
    td = make_tensor_desc((32, 16), (1, 32), F16)
    _check_indices(make_window(td, (16, 0)), _TILE_RUN_AXIS0)


def test_stage_origin_adds_index_times_stage_stride() -> None:
    # Two (rows, K) stages of 16 x 24 elements. A rank-2 tile covers the last two axes; the stage is
    # the leading axis, positioned by its origin: stage 1 adds 1 * 384 to every index.
    td = make_tensor_desc((2, 16, 16), (384, 24, 1), F16)
    window = make_window(td, (1, 0, 0))
    _check_indices(window, _TILE_RUN_AXIS1)
    stage_0 = _load_then_store(make_window(td, (0, 0, 0)), _TILE_RUN_AXIS1)
    stage_1 = _load_then_store(window, _TILE_RUN_AXIS1)
    for lane in range(_WAVE):
        expected = [(name, (i + 384,)) for name, (i,) in _lds_indices(stage_0, lane)]
        assert _lds_indices(stage_1, lane) == expected, f"lane {lane}"


def test_lds_index_is_a_lane_base_plus_a_constant_per_register_run() -> None:
    # The unswizzled form computes the lane's base index once; every run adds a compile-time
    # constant to it, which the backend folds into the instruction offset and pairs.
    td = make_tensor_desc((16, 16), (24, 1), F16)
    kernel = _load_then_store(make_window(td, (0, 0)), _TILE_RUN_AXIS0)
    by_name = {r.name: op for op in kernel.body.ops for r in op.results}
    loads = [op for op in kernel.body.ops if op.name.startswith("tile.smem_load")]
    bases = set()
    for op in loads[1:]:
        index_op = by_name[op.operands[1].name]
        assert index_op.name == "arith.add"
        base, offset = index_op.operands
        assert by_name[offset.name].name == "arith.constant"
        bases.add(base.name)
    assert bases == {loads[0].operands[1].name}


# ---- refusals ----------------------------------------------------------------------------------------


def _refusal(buffer_shape, buffer_dtype=F16, *, verb="load"):
    b = IRBuilder("lds_refusal")
    tid = b.thread_id_x()
    lds = b.smem_alloc(buffer_dtype, buffer_shape)
    window = make_window(make_tensor_desc((16, 16), (16, 1), F16), (0, 0))
    with pytest.raises(ValueError) as info:
        if verb == "load":
            load_fragment(b, lds, window, _TILE_RUN_AXIS1, tid)
        else:
            fragment = make_fragment(_TILE_RUN_AXIS1, F16)
            fill_fragment(b, fragment, 0)
            store_fragment(b, lds, window, fragment, tid)
    return str(info.value)


def test_a_rank_2_buffer_is_refused() -> None:
    assert _refusal([16, 16]) == (
        "load_fragment: the LDS buffer smem<f16, [16x16]> has rank 2 -- an LDS buffer is flat; its "
        "layout lives in the tensor descriptor. Allocate smem_alloc(f16, [256])"
    )


def test_a_buffer_of_another_element_type_is_refused() -> None:
    assert (
        _refusal([256], F32)
        == "load_fragment: LDS buffer element f32 != tensor dtype f16"
    )


def test_a_buffer_smaller_than_the_descriptor_span_is_refused() -> None:
    assert _refusal([200]) == (
        "load_fragment: the LDS tensor spans 256 elements but the buffer holds 200 -- size the "
        "allocation with lds_element_span"
    )


def test_the_store_verb_runs_the_same_checks() -> None:
    assert _refusal([200], verb="store") == (
        "store_fragment: the LDS tensor spans 256 elements but the buffer holds 200 -- size the "
        "allocation with lds_element_span"
    )


def test_a_larger_buffer_is_accepted() -> None:
    td = make_tensor_desc((16, 16), (16, 1), F16)
    _load_then_store(make_window(td, (0, 0)), _TILE_RUN_AXIS1, buffer_elements=300)


def test_builtin_swizzle_needs_the_stride_1_axis_last() -> None:
    td = make_tensor_desc((32, 16), (1, 32), F16)
    with pytest.raises(ValueError) as info:
        _load_then_store(make_window(td, (0, 0)), _TILE_RUN_AXIS0, swizzle=True)
    assert str(info.value) == (
        "lds_swizzle=True needs the stride-1 axis last -- strides (1, 32) put it at axis 0; "
        "describe the buffer with its contiguous axis last or pass a swizzle callable"
    )


# ---- the built-in swizzle's reach ------------------------------------------------------------------

# A row of 48 f16 padded to a stride of 64: the swizzle moves the run at column 0 of row 7 to column
# 56, inside the row's padding, so the last row reaches element 7 * 64 + 63 = 511, past the
# descriptor's span of 7 * 64 + 48 = 496.
_PADDED_ROWS = make_tensor_desc((8, 48), (64, 1), F16)
_TILE_8X48 = make_tile_desc(
    shape=[8, 48], thread_tile=[1, 6], thread_dist=[8, 8], wave_size=_WAVE
)


def _swizzled_access_ends(kernel) -> list[int]:
    """One past the last element of every LDS access the kernel emits, over every lane."""
    widths = [
        op.attrs.get("vec", 1)
        for op in kernel.body.ops
        if op.name.startswith("tile.smem_") and op.name != "tile.smem_alloc"
    ]
    ends = []
    for lane in range(_WAVE):
        for (_, (index,)), width in zip(
            _lds_indices(kernel, lane), widths, strict=True
        ):
            ends.append(index + width)
    return ends


def test_lds_element_span_counts_the_builtin_swizzle_reach() -> None:
    assert _PADDED_ROWS.element_span == 496
    assert lds_element_span(_PADDED_ROWS) == 496
    assert lds_element_span(_PADDED_ROWS, True) == 512
    # A row that already fills its 64-element groups reaches no further.
    full_rows = make_tensor_desc((8, 128), (128, 1), F16)
    assert lds_element_span(full_rows, True) == full_rows.element_span == 1024


@pytest.mark.parametrize("verb", ["load", "store"])
def test_a_swizzled_buffer_sized_by_element_span_is_refused(verb) -> None:
    b = IRBuilder("lds_swizzle_reach")
    tid = b.thread_id_x()
    lds = b.smem_alloc(F16, [_PADDED_ROWS.element_span])
    window = make_window(_PADDED_ROWS, (0, 0))
    with pytest.raises(ValueError) as info:
        if verb == "load":
            load_fragment(b, lds, window, _TILE_8X48, tid, lds_swizzle=True)
        else:
            fragment = make_fragment(_TILE_8X48, F16)
            fill_fragment(b, fragment, 0)
            store_fragment(b, lds, window, fragment, tid, lds_swizzle=True)
    assert str(info.value) == (
        f"{verb}_fragment: the LDS tensor spans 512 elements but the buffer holds 496 -- size the "
        "allocation with lds_element_span"
    )


def test_every_swizzled_access_fits_lds_element_span() -> None:
    # Loads and stores both reach the last row's padding, and stay inside the span.
    size = lds_element_span(_PADDED_ROWS, True)
    kernel = _load_then_store(
        make_window(_PADDED_ROWS, (0, 0)),
        _TILE_8X48,
        buffer_elements=size,
        swizzle=True,
    )
    ends = _swizzled_access_ends(kernel)
    assert max(ends) == size == 512


def test_builtin_swizzle_refuses_rows_that_would_overlap() -> None:
    # Unpadded rows of 48: the swizzled columns 48..63 of a row are the next row's elements.
    td = make_tensor_desc((8, 48), (48, 1), F16)
    with pytest.raises(ValueError) as info:
        _load_then_store(make_window(td, (0, 0)), _TILE_8X48, swizzle=True)
    assert str(info.value) == (
        "lds_swizzle=True moves elements of the last axis within groups of 64, so a row of 48 "
        "reaches 64 elements, but axis 0 steps by 48 -- rows would overlap; pad that stride to at "
        "least 64 or pass a swizzle callable"
    )


def test_swizzled_demo_buffers_hold_every_access() -> None:
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (
        build_interleaved_gemm,
    )

    small = dict(arch="gfx90a", tile_m=16, tile_n=16, tile_k=16, lds_swizzle=True)
    # Rows of 32 padded to 64: the descriptor spans 992 elements, the swizzle reaches 1024.
    kernel = build_interleaved_gemm(32, 32, 32, lds_pad=32, **small)[0]
    allocs = [
        op.results[0].type.name
        for op in kernel.body.ops
        if op.name == "tile.smem_alloc"
    ]
    assert allocs == ["smem<f16, [1024]>", "smem<f16, [1024]>"]
    # Unpadded rows of 32 would overlap under the swizzle.
    with pytest.raises(ValueError, match="rows would overlap"):
        build_interleaved_gemm(32, 32, 32, **small)


# ---- element_span ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths,strides,span",
    [
        ((16, 16), (16, 1), 256),  # contiguous
        (
            (16, 16),
            (24, 1),
            15 * 24 + 16,
        ),  # padded rows: the pad after the last row is not counted
        ((32, 16), (1, 32), 512),  # (N, K) over a [K][N] buffer
        ((2, 32, 16), (512, 1, 32), 1024),  # (stages, free, K) over [stage][K][free]
        ((64,), (1,), 64),
    ],
)
def test_element_span(lengths, strides, span) -> None:
    assert make_tensor_desc(lengths, strides, F16).element_span == span


def test_element_span_refuses_a_negative_stride() -> None:
    with pytest.raises(ValueError) as info:
        _ = make_tensor_desc((4, 4), (4, -1), F16).element_span
    assert str(info.value) == (
        "TensorDesc.element_span needs non-negative strides -- strides[1]=-1"
    )


# ---- lds_tile_desc -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tile_free,tile_k,n_waves",
    [(64, 16, 1), (64, 16, 4), (256, 64, 1), (256, 64, 4)],
)
def test_lds_tile_desc_keeps_the_buffer_order_addresses(
    tile_free, tile_k, n_waves
) -> None:
    buffers = 2
    lanes = _WAVE * n_waves
    vw = cooperative_load_width(tile_free, tile_k, n_waves)
    coop = cooperative_load_desc(tile_free, tile_k, n_waves, vw=vw)
    td = lds_tile_desc(coop, F16, buffers=buffers, n_waves=n_waves)

    free_ext, k_ext = desc_extents(coop, lanes)
    row = buffers * free_ext
    assert td.lengths == (row, k_ext)
    assert td.strides == (1, row)
    # The buffer-order layout this replaces: (K, free) extents from the swapped tile, row stride
    # buffers * free_ext, element (k, f) at k * row + f, k_ext * row elements.
    assert desc_extents(coop.swap_dims(0, 1), lanes) == (k_ext, free_ext)
    assert td.element_span == k_ext * row

    swapped = coop.swap_dims(0, 1)
    for lane in range(lanes):
        for register in range(coop.register_count):
            f, k = emit_tensor_coordinates(
                NumBuilder(lane), coop.layout, lane, register
            )
            k_old, f_old = emit_tensor_coordinates(
                NumBuilder(lane), swapped.layout, lane, register
            )
            for buffer in range(buffers):
                f_pos = buffer * free_ext + f
                new = f_pos * td.strides[0] + k * td.strides[1]
                old = k_old * row + buffer * free_ext + f_old
                assert new == old


def test_lds_verbs_check_the_tile_against_the_descriptor_lengths() -> None:
    assert "alloc" not in inspect.signature(lds_store).parameters
    assert "alloc" not in inspect.signature(lds_read).parameters
    b = IRBuilder("lds_fit")
    tid = b.thread_id_x()
    td = make_tensor_desc((16, 8), (1, 16), F16)
    lds = b.smem_alloc(td.dtype, [td.element_span])
    with pytest.raises(ValueError) as info:
        lds_read(b, lds, td, (0, 0), _TILE_RUN_AXIS1, tid, n_lanes=_WAVE)
    assert str(info.value) == (
        "cooperative LDS read addresses (16, 16) but the LDS tensor descriptor has lengths (16, 8) "
        "-- the cooperative LDS access would run outside the buffer (unclipped by design). "
        "Describe the buffer with lengths that cover the tile, per axis."
    )


def test_lds_verbs_check_a_staged_tile_against_the_last_axes() -> None:
    # A (stages, rows, K) buffer read with a rank-2 tile: the tile covers the last two axes, so its
    # extents are checked against their lengths and the stage is positioned by the origin.
    b = IRBuilder("lds_fit_stages")
    tid = b.thread_id_x()
    td = make_tensor_desc((2, 16, 16), (256, 16, 1), F16)
    lds = b.smem_alloc(td.dtype, [td.element_span])
    fragment = lds_read(b, lds, td, (1, 0, 0), _TILE_RUN_AXIS1, tid, n_lanes=_WAVE)
    lds_store(b, lds, td, (0, 0, 0), fragment, tid, n_lanes=_WAVE)
    short = make_tensor_desc((2, 16, 8), (128, 8, 1), F16)
    with pytest.raises(ValueError) as info:
        lds_read(b, lds, short, (1, 0, 0), _TILE_RUN_AXIS1, tid, n_lanes=_WAVE)
    assert str(info.value).startswith(
        "cooperative LDS read addresses (16, 16) but the LDS tensor descriptor has lengths "
        "(2, 16, 8) -- "
    )


# ---- the migrated demos keep their vector LDS accesses -----------------------------------------------


def _lds_access_forms(kernel) -> dict[str, int]:
    forms: collections.Counter[str] = collections.Counter()
    for line in lower_kernel_to_llvm(kernel, arch="gfx90a").splitlines():
        if "addrspace(3)" in line:
            m = re.search(r"\b(load|store) (<\d+ x half>|half)", line)
            if m:
                forms[f"{m.group(1)} {m.group(2)}"] += 1
    return dict(forms)


def _demo_kernels():
    from rocke.helpers.tiling.kernels.tiling_gemm_crc_demo.crc_interleaved_gemm import (
        build_crc_gemm,
    )
    from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import (
        build_interleaved_gemm,
        build_lds_staged_gemm,
    )

    return {
        "lds_staged": (
            lambda: build_lds_staged_gemm(256, 256, 256, arch="gfx90a")[0],
            {"load <4 x half>": 32, "store <4 x half>": 32},
        ),
        "interleaved": (
            lambda: build_interleaved_gemm(256, 256, 256, arch="gfx90a")[0],
            {"load half": 64, "store <4 x half>": 16},
        ),
        "interleaved 2x2 waves": (
            lambda: build_interleaved_gemm(
                256,
                256,
                256,
                arch="gfx90a",
                tile_m=128,
                tile_n=128,
                waves_m=2,
                waves_n=2,
            )[0],
            {"load half": 64, "store <2 x half>": 16},
        ),
        "canonical": (
            lambda: build_interleaved_gemm(
                256,
                256,
                256,
                arch="gfx90a",
                prefetch="canonical",
                mma_layout="canonical",
            )[0],
            {"load <4 x half>": 16, "store <4 x half>": 16},
        ),
        "interleaved pad": (
            lambda: build_interleaved_gemm(256, 256, 256, arch="gfx90a", lds_pad=8)[0],
            {"load half": 64, "store <4 x half>": 16},
        ),
        "crc": (
            lambda: build_crc_gemm(256, 256, 64, arch="gfx90a")[0],
            {"load half": 128, "store <8 x half>": 4},
        ),
        "crc ab_swap": (
            lambda: build_crc_gemm(256, 256, 64, arch="gfx90a", ab_swap=True)[0],
            {"load half": 128, "store <8 x half>": 4},
        ),
    }


@pytest.mark.parametrize("label", list(_demo_kernels()))
def test_migrated_demos_keep_their_lds_access_widths(label: str) -> None:
    build, expected = _demo_kernels()[label]
    assert _lds_access_forms(build()) == expected


# ---- GPU (gfx90a) ------------------------------------------------------------------------------------


def _gfx90a_present() -> bool:
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) == "gfx90a"
    except Exception:  # noqa: BLE001
        # Any HIP or driver failure means there is no usable GPU here.
        return False


_needs_gfx90a = pytest.mark.skipif(not _gfx90a_present(), reason="needs a gfx90a GPU")

_NP = {"f16": np.float16, "f32": np.float32}
_IR = {"f16": F16, "f32": F32}


def _run(
    kernel, signature, inputs: dict, outputs: dict, *, grid=(1, 1, 1), block=(64, 1, 1)
):
    """Compile `kernel` for gfx90a, launch it on host arrays and copy `outputs` back in place."""
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    art = compile_kernel(kernel, arch="gfx90a")
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=signature
    )
    rt = Runtime()
    arrays = {**inputs, **outputs}
    dev = {name: DeviceMem(a.nbytes) for name, a in arrays.items()}
    for name, a in arrays.items():
        rt.memcpy_h2d(dev[name].ptr(), as_u8_buffer(a), a.nbytes)
    launcher(dev, config=LaunchConfig(grid=grid, block=block))
    synchronize_and_release()
    for name, a in outputs.items():
        rt.memcpy_d2h(as_u8_buffer(a), dev[name].ptr(), a.nbytes)


def _round_trip_kernel(dtype, make_windows, store_tile, read_tile, dump_elements):
    """For each (global window, LDS window) pair ``make_windows(b)`` returns: load the global tile
    from IN with `store_tile` and store it into LDS. After a barrier, read every LDS window back with
    `read_tile` and store it to OUT through its global window, then dump the whole flat buffer into
    RAW."""
    from rocke.helpers.spec import SignatureBuilder

    ir = _IR[dtype]
    b = IRBuilder(f"lds_round_trip_{dtype}")
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = b.param("IN", PtrType(ir, "global"), noalias=True, readonly=True, align=16)
    out = b.param("OUT", PtrType(ir, "global"), noalias=True, writeonly=True, align=16)
    raw = b.param("RAW", PtrType(ir, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    lds = b.smem_alloc(ir, [dump_elements], name_hint="lds")
    lds_windows = make_windows(b)
    for global_window, lds_window in lds_windows:
        fragment = load_fragment(b, src, global_window, store_tile, tid)
        store_fragment(b, lds, lds_window, fragment, tid)
    b.sync()
    for global_window, lds_window in lds_windows:
        fragment = load_fragment(b, lds, lds_window, read_tile, tid)
        store_fragment(b, out, global_window, fragment, tid)
    flat = make_tensor_desc((dump_elements,), (1,), ir)
    dump_tile = make_tile_desc(
        shape=[dump_elements],
        thread_tile=[dump_elements // _WAVE],
        thread_dist=[_WAVE],
        wave_size=_WAVE,
    )
    everything = load_fragment(b, lds, make_window(flat, (0,)), dump_tile, tid)
    store_fragment(b, raw, make_window(flat, (0,)), everything, tid)
    b.ret()
    signature = (
        SignatureBuilder().ptr("IN", dtype).ptr("OUT", dtype).ptr("RAW", dtype).build()
    )
    return b.kernel, signature


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


@_needs_gfx90a
@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize(
    "case",
    ["k_contiguous", "padded", "n_contiguous"],
)
def test_lds_round_trip_places_elements_by_strides(dtype: str, case: str) -> None:
    ir = _IR[dtype]
    # The tensor is (16, 32), last axis contiguous, in global memory; LDS describes the same logical tensor.
    rows, cols = 16, 32
    lds_strides = {"k_contiguous": (32, 1), "padded": (40, 1), "n_contiguous": (1, 16)}[
        case
    ]
    global_td = make_tensor_desc((rows, cols), (cols, 1), ir)
    lds_td = make_tensor_desc((rows, cols), lds_strides, ir)
    # Store with 8-element runs along the columns, read back with 8-element runs along the rows, so
    # the two sides agree only if both address LDS by the strides.
    store_tile = make_tile_desc(
        shape=[rows, cols], thread_tile=[1, 8], thread_dist=[16, 4], wave_size=_WAVE
    )
    read_tile = make_tile_desc(
        shape=[rows, cols], thread_tile=[8, 1], thread_dist=[2, 32], wave_size=_WAVE
    )
    dump = _round_up(lds_td.element_span, _WAVE * 8)
    kernel, signature = _round_trip_kernel(
        dtype,
        lambda b: [(make_window(global_td, (0, 0)), make_window(lds_td, (0, 0)))],
        store_tile,
        read_tile,
        dump,
    )
    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, size=(rows, cols)).astype(_NP[dtype])
    out = np.zeros_like(data)
    raw = np.zeros(dump, _NP[dtype])
    _run(kernel, signature, {"IN": data}, {"OUT": out, "RAW": raw})
    np.testing.assert_array_equal(out, data)
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    np.testing.assert_array_equal(raw[r * lds_strides[0] + c * lds_strides[1]], data)


@_needs_gfx90a
@pytest.mark.parametrize("dtype", ["f16", "f32"])
def test_lds_round_trip_rank_3_stages(dtype: str) -> None:
    # Two stages of a (free, K) = (32, 16) tile, staged [stage][K][free]: the LDS descriptor is
    # (stages, free, K) with strides (K * F, 1, F). The (free, K) tiles cover the last two axes; each
    # stage is positioned by the origin of the leading axis (an int for stage 0, a runtime value for
    # stage 1) and read with an un-swapped (free, K) tile.
    ir = _IR[dtype]
    stages, free, k = 2, 32, 16
    global_td = make_tensor_desc((stages, free, k), (free * k, k, 1), ir)
    lds_td = make_tensor_desc((stages, free, k), (k * free, 1, free), ir)
    store_tile = make_tile_desc(
        shape=[free, k], thread_tile=[1, 8], thread_dist=[32, 2], wave_size=_WAVE
    )
    read_tile = make_tile_desc(
        shape=[free, k], thread_tile=[8, 1], thread_dist=[4, 16], wave_size=_WAVE
    )

    dump = _round_up(lds_td.element_span, _WAVE * 8)

    def stage_windows(b):
        runtime_stage = b.add(b.const_i32(0), b.const_i32(1))
        return [
            (
                make_window(global_td, (stage, 0, 0)),
                make_window(lds_td, (stage, 0, 0)),
            )
            for stage in (0, runtime_stage)
        ]

    kernel, signature = _round_trip_kernel(
        dtype, stage_windows, store_tile, read_tile, dump
    )
    rng = np.random.default_rng(1)
    data = rng.integers(-1000, 1000, size=(stages, free, k)).astype(_NP[dtype])
    out_h = np.zeros_like(data)
    raw_h = np.zeros(dump, _NP[dtype])
    _run(kernel, signature, {"IN": data}, {"OUT": out_h, "RAW": raw_h})
    np.testing.assert_array_equal(out_h, data)
    s, f, kk = np.meshgrid(
        np.arange(stages), np.arange(free), np.arange(k), indexing="ij"
    )
    np.testing.assert_array_equal(raw_h[s * k * free + f + kk * free], data)


def _convention_gemm(wave):
    """C = A * B^T for one wave tile per block. B's tile is staged in LDS laid out [K][N] and
    described in convention order (N, K) with strides (1, N), read back with the un-swapped
    (N, K) B tile."""
    from rocke.helpers.spec import SignatureBuilder

    wave_m, wave_n, wave_k = wave
    m_len = n_len = k_len = 64
    mma = TileMma(
        wave,
        a="f16",
        b="f16",
        c="f32",
        target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
    )
    b = IRBuilder(f"lds_convention_gemm_{wave_m}x{wave_n}x{wave_k}")
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size
    a_ptr = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    b_ptr = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    c_ptr = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    lane = b.thread_id_x()
    m0 = b.mul(b.block_id_y(), b.const_i32(wave_m))
    n0 = b.mul(b.block_id_x(), b.const_i32(wave_n))
    a_td = make_tensor_desc((m_len, k_len), (k_len, 1), F16)
    b_td = make_tensor_desc((n_len, k_len), (k_len, 1), F16)
    c_td = make_tensor_desc((m_len, n_len), (n_len, 1), F32)
    lds_td = make_tensor_desc((wave_n, wave_k), (1, wave_n), F16)
    lds = b.smem_alloc(lds_td.dtype, [lds_td.element_span], name_hint="lds_b")
    acc = make_fragment(mma.c_desc, F32)
    fill_fragment(b, acc, 0)
    for k0 in range(0, k_len, wave_k):
        kk = b.const_i32(k0)
        fa = load_fragment(b, a_ptr, make_window(a_td, (m0, kk)), mma.a_desc, lane)
        fb = load_fragment(b, b_ptr, make_window(b_td, (n0, kk)), mma.b_desc, lane)
        staged = make_window(lds_td, (0, 0))
        store_fragment(b, lds, staged, fb, lane)
        b.sync()
        fb = load_fragment(b, lds, staged, mma.b_desc, lane)
        b.sync()
        acc = mma(b, fa, fb, acc)
    store_fragment(b, c_ptr, make_window(c_td, (m0, n0)), acc, lane)
    b.ret()
    signature = (
        SignatureBuilder().ptr("A", "f16").ptr("B", "f16").ptr("C", "f32").build()
    )
    grid = (n_len // wave_n, m_len // wave_m, 1)
    return b.kernel, signature, grid, (mma.wave_size, 1, 1), (m_len, n_len, k_len)


@_needs_gfx90a
@pytest.mark.parametrize("wave", [(16, 16, 16), (32, 32, 16), (16, 32, 32)])
def test_gemm_with_b_staged_n_contiguous_is_bit_exact(wave) -> None:
    kernel, signature, grid, block, (m_len, n_len, k_len) = _convention_gemm(wave)
    rng = np.random.default_rng(0)
    a = rng.integers(-3, 4, size=(m_len, k_len)).astype(np.float16)
    b_host = rng.integers(-3, 4, size=(n_len, k_len)).astype(np.float16)
    c = np.zeros((m_len, n_len), np.float32)
    _run(kernel, signature, {"A": a, "B": b_host}, {"C": c}, grid=grid, block=block)
    np.testing.assert_array_equal(c, a.astype(np.float32) @ b_host.astype(np.float32).T)
