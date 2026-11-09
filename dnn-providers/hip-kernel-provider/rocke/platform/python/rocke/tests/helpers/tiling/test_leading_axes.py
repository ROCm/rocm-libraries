# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A rank-t tile covers the window's last t axes; the leading axes are positioned by the origin.

There are no axis labels: a batched ``(G, M, N)`` tensor is read with a rank-2 ``(M, N)`` tile through a
rank-3 window at origin ``(g, m0, n0)``. These tests check:

- a tile of higher rank than its window is refused by ``load_fragment`` and ``store_fragment``;
- the element index sums the tile axes first and then adds ``origin * stride`` of the leading axis;
- the vector run is decided on the tensor's trailing axes (a stride-1 last axis of a rank-3 tensor
  gives wide loads), a ragged trailing axis scalarizes exactly as the rank-2 tensor does, and the
  run's width divides every other stride (so a batch offset never misaligns a wide load);
- a swizzle sees the position on every axis, so an LDS element has one address whatever the rank of
  the tile that reaches it;
- any rank-2 tensor feeds an MMA operand tile (nothing gates an operand by axis meaning);
- on a gfx90a GPU: a rank-3 load then store, with the batch index a runtime value or an int, is
  bit-exact and writes nothing outside the output tensor.
"""

from __future__ import annotations

import collections
import re

import numpy as np
import pytest

pytest.importorskip("rocke.core.ir", reason="rocke IR substrate required")

from rocke.core.ir import F16, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.helpers.tiling import (
    TileMma,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.lds_conflict import addr_map

_WAVE = 64
# A (16, 32) tile over one wave: 16 lanes down axis 0, 4 lanes across axis 1, 8 contiguous elements
# per lane along axis 1 (the vector axis when the last tensor axis is stride 1).
_TILE = make_tile_desc(
    shape=[16, 32], thread_tile=[1, 8], thread_dist=[16, 4], wave_size=_WAVE
)


def _copy_kernel(src_td, dst_td, origin_of, *, name="leading_axes_copy"):
    """Load `_TILE` from SRC through ``make_window(src_td, origin_of(b))`` and store it to DST at the
    same origin. ``origin_of`` returns one ``(batch, m, n)`` origin per tile to copy."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = _WAVE
    src = b.param("SRC", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    dst = b.param("DST", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    for origin in origin_of(b):
        fragment = load_fragment(b, src, make_window(src_td, origin), _TILE, tid)
        store_fragment(b, dst, make_window(dst_td, origin), fragment, tid)
    b.ret()
    return b.kernel


def _block_origin(b):
    """This block's tile: batch = block_id_z (runtime), (m0, n0) from block_id_y / block_id_x."""
    m0 = b.mul(b.block_id_y(), b.const_i32(_TILE.shape[0]))
    n0 = b.mul(b.block_id_x(), b.const_i32(_TILE.shape[1]))
    return [(b.block_id_z(), m0, n0)]


# ---- the rank rule -----------------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["load_fragment", "store_fragment"])
def test_a_tile_of_higher_rank_than_its_window_is_refused(verb: str) -> None:
    b = IRBuilder("tile_rank")
    ptr = b.param("P", PtrType(F16, "global"))
    tid = b.thread_id_x()
    window = make_window(make_tensor_desc((32,), (1,), F16), (0,))
    with pytest.raises(ValueError) as info:
        if verb == "load_fragment":
            load_fragment(b, ptr, window, _TILE, tid)
        else:
            fragment = make_fragment(_TILE, F16)
            fill_fragment(b, fragment, 0)
            store_fragment(b, ptr, window, fragment, tid)
    assert str(info.value) == (
        f"{verb}: the tile has rank 2 but the window has rank 1 -- a rank-t tile covers the "
        f"window's last t axes, so t <= r"
    )


@pytest.mark.parametrize("verb", ["load_fragment", "store_fragment"])
def test_a_bound_on_a_leading_axis_is_refused(verb: str) -> None:
    # The batch axis has tile extent 1: a tile is wholly inside or outside a bound on it, so the clip
    # has nothing to mask there. A bound is refused rather than silently ignored.
    b = IRBuilder("leading_bound")
    ptr = b.param("P", PtrType(F16, "global"))
    tid = b.thread_id_x()
    td = make_tensor_desc((3, 32, 64), (32 * 64, 64, 1), F16)
    window = make_window(td, (2, 0, 0), bounds=(2, None, None))
    with pytest.raises(ValueError, match="window bounds axis 0, a leading axis"):
        if verb == "load_fragment":
            load_fragment(b, ptr, window, _TILE, tid)
        else:
            store_fragment(b, ptr, window, make_fragment(_TILE, F16), tid)


def test_bounds_on_the_tile_axes_are_still_honoured() -> None:
    # Only leading-axis entries are refused: a ragged bound on a tile axis still masks.
    td = make_tensor_desc((3, 32, 64), (32 * 64, 64, 1), F16)
    kernel = _copy_kernel(td, td, lambda b: [(1, 0, 0)], name="tile_axis_bound")
    assert "icmp" not in lower_kernel_to_llvm(kernel, arch="gfx90a")
    b = IRBuilder("tile_axis_bound_ragged")
    ptr = b.param("P", PtrType(F16, "global"))
    window = make_window(td, (1, 0, 0), bounds=(None, 30, None))
    load_fragment(b, ptr, window, _TILE, b.thread_id_x())
    b.ret()
    assert "icmp slt" in lower_kernel_to_llvm(b.kernel, arch="gfx90a")


# ---- the address fold order --------------------------------------------------------------------------


def test_leading_axis_term_is_added_after_the_tile_terms() -> None:
    # (G, M, N) with strides (2048, 64, 1): every load index is add(<tile terms>, mul(g, 2048)).
    td = make_tensor_desc((3, 32, 64), (2048, 64, 1), F16)
    kernel = _copy_kernel(td, td, _block_origin)
    by_result = {r.name: op for op in kernel.body.ops for r in op.results}
    batch = next(
        op.results[0].name
        for op in kernel.body.ops
        if op.name == "gpu.block_id" and op.attrs["axis"] == "z"
    )
    loads = [op for op in kernel.body.ops if op.name.startswith("memref.global_load")]
    assert loads
    for load in loads:
        index = by_result[load.operands[1].name]
        assert index.name == "arith.add"
        leading = by_result[index.operands[1].name]
        assert leading.name == "arith.mul"
        assert leading.operands[0].name == batch
        assert by_result[leading.operands[1].name].attrs["value"] == 2048


# ---- the vector run on a trailing axis ---------------------------------------------------------------


def _load_forms(kernel) -> dict[str, int]:
    forms: collections.Counter[str] = collections.Counter()
    for line in lower_kernel_to_llvm(kernel, arch="gfx90a").splitlines():
        if "addrspace(1)" in line:
            m = re.search(r"= load (<\d+ x half>|half)", line)
            if m:
                forms[m.group(1)] += 1
    return dict(forms)


def test_vector_run_on_the_last_axis_of_a_rank_3_tensor_is_wide() -> None:
    # (G, M, K) with strides (M * K, K, 1): the tile's run axis (tile axis 1) is tensor axis 2,
    # stride 1, not the batch axis -- so every load is a <8 x half>.
    td = make_tensor_desc((3, 32, 64), (32 * 64, 64, 1), F16)
    assert _load_forms(_copy_kernel(td, td, _block_origin)) == {"<8 x half>": 1}


def test_a_ragged_last_axis_scalarizes_as_the_rank_2_tensor_does() -> None:
    # K = 60 is not a multiple of the tile's 32: the rank-3 tensor loads exactly as the rank-2
    # tensor with that K does (scalar, masked), whatever the leading axis.
    rank_3 = make_tensor_desc((3, 32, 60), (32 * 60, 60, 1), F16)
    rank_2 = make_tensor_desc((32, 60), (60, 1), F16)
    forms_3 = _load_forms(_copy_kernel(rank_3, rank_3, lambda b: [(2, 0, 32)]))
    forms_2 = _load_forms(_copy_kernel(rank_2, rank_2, lambda b: [(0, 32)]))
    assert forms_3 == forms_2 == {"half": _TILE.register_count}


@pytest.mark.parametrize(
    "batch_stride,forms",
    [
        (32 * 64, {"<8 x half>": 1}),
        (32 * 64 + 4, {"<4 x half>": 2}),
        (32 * 64 + 1, {"half": _TILE.register_count}),
    ],
)
def test_the_vector_run_divides_every_other_stride(batch_stride: int, forms) -> None:
    # Batch 1 starts at element `batch_stride`. A wide load claims the alignment of its width, so the
    # width must divide the batch stride too: an odd stride (2049) leaves only one element per load,
    # a stride that is a multiple of 4 but not of 8 leaves four.
    td = make_tensor_desc((3, 32, 64), (batch_stride, 64, 1), F16)
    assert _load_forms(_copy_kernel(td, td, lambda b: [(1, 0, 0)])) == forms


# ---- a swizzle sees every axis -----------------------------------------------------------------------


def test_a_swizzled_element_has_one_address_whatever_the_tile_rank() -> None:
    # An (8, 64) LDS buffer written by a rank-2 tile and read back one row at a time by a rank-1 tile,
    # both with the built-in swizzle (it XORs the column by the row). The rank-1 read must see its row
    # (the window origin) as the store did, or row 1 would be read from unswizzled addresses.
    buffer = make_tensor_desc((8, 64), (64, 1), F16)
    rows = make_tile_desc(
        shape=[8, 64], thread_tile=[1, 8], thread_dist=[8, 8], wave_size=_WAVE
    )
    row = make_tile_desc(shape=[64], thread_tile=[8], thread_dist=[8], wave_size=8)

    def addresses(tile, origin, lanes):
        """Element -> LDS index: the element from the unswizzled map, the index from the swizzled one."""
        leading = tuple(origin[: buffer.rank - len(tile.shape)])
        plain, _ = addr_map(
            tile, buffer, origin=origin, n_lanes=lanes, lds_swizzle=False
        )
        swizzled, _ = addr_map(
            tile, buffer, origin=origin, n_lanes=lanes, lds_swizzle=True
        )
        return {
            leading + position: access["base"] + i
            for logical, access in zip(plain, swizzled, strict=True)
            for i, position in enumerate(logical["element_positions"])
        }

    written = addresses(rows, (0, 0), _WAVE)
    for r in range(8):
        read = addresses(row, (r, 0), 8)
        assert read == {element: written[element] for element in read}
    assert written[(1, 0)] == 72


# ---- an MMA operand from any rank-2 tensor -----------------------------------------------------------


@pytest.mark.parametrize(
    "atom,lengths,strides",
    [
        ("mfma_f32_16x16x16f16", (16, 16), (16, 1)),  # B (N, K), K-contiguous
        ("mfma_f32_16x16x16f16", (16, 16), (1, 16)),  # B (N, K), N-contiguous
        ("mfma_f32_32x32x8f16", (32, 8), (8, 1)),  # B (N, K) = (32, 8), K-contiguous
    ],
)
def test_an_mma_operand_tile_loads_from_any_rank_2_tensor(
    atom, lengths, strides
) -> None:
    mma = TileMma(target="gfx90a", atom_override=atom)
    assert tuple(mma.b_desc.shape) == lengths
    b = IRBuilder("mma_operand")
    ptr = b.param("B", PtrType(F16, "global"))
    tid = b.thread_id_x()
    td = make_tensor_desc(lengths, strides, F16)
    fragment = load_fragment(b, ptr, make_window(td, (0, 0)), mma.b_desc, tid)
    assert fragment.tile_desc == mma.b_desc
    assert fragment.dtype is F16


# ---- GPU (gfx90a) ------------------------------------------------------------------------------------


def _gfx90a_present() -> bool:
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) == "gfx90a"
    except Exception:  # noqa: BLE001
        # Any HIP or driver failure means there is no usable GPU here.
        return False


_needs_gfx90a = pytest.mark.skipif(not _gfx90a_present(), reason="needs a gfx90a GPU")

_GUARD = 1 << 16  # guard elements on each side of every device buffer
_SRC_GUARD = -7777.0  # what an unmasked out-of-bounds read would return
_DST_INIT = -12345.0  # every output element and guard before the launch


def _run_guarded(kernel, src: np.ndarray, dst_elements: int, grid) -> np.ndarray:
    """Launch `kernel` with SRC and DST each placed between guard zones; return the whole DST
    allocation (front guard, the `dst_elements` the kernel addresses, back guard)."""
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

    art = compile_kernel(kernel, arch="gfx90a")
    signature = SignatureBuilder().ptr("SRC", "f16").ptr("DST", "f16").build()
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=signature
    )
    src_h = np.full(src.size + 2 * _GUARD, _SRC_GUARD, np.float16)
    src_h[_GUARD : _GUARD + src.size] = src.ravel()
    dst_h = np.full(dst_elements + 2 * _GUARD, _DST_INIT, np.float16)
    rt = Runtime()
    src_d, dst_d = DeviceMem(src_h.nbytes), DeviceMem(dst_h.nbytes)
    rt.memcpy_h2d(src_d.ptr(), as_u8_buffer(src_h), src_h.nbytes)
    rt.memcpy_h2d(dst_d.ptr(), as_u8_buffer(dst_h), dst_h.nbytes)
    item = np.dtype(np.float16).itemsize
    launcher(
        {"SRC": src_d.ptr() + item * _GUARD, "DST": dst_d.ptr() + item * _GUARD},
        config=LaunchConfig(grid=grid, block=(_WAVE, 1, 1)),
    )
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(dst_h), dst_d.ptr(), dst_h.nbytes)
    return dst_h


@_needs_gfx90a
@pytest.mark.parametrize("batch_origin", ["runtime", "int"])
def test_rank_3_load_then_store_is_bit_exact(batch_origin: str) -> None:
    # SRC (G, M, N) packed; DST (G, M, N) with a padded batch stride, so the gap after each batch
    # is a guard too. A rank-2 (M, N) tile is placed at (g, m0, n0) on both rank-3 windows.
    g_len, m_len, n_len = 3, 32, 64
    dst_batch_stride = m_len * n_len + 96
    src_td = make_tensor_desc((g_len, m_len, n_len), (m_len * n_len, n_len, 1), F16)
    dst_td = make_tensor_desc((g_len, m_len, n_len), (dst_batch_stride, n_len, 1), F16)
    tiles_m, tiles_n = m_len // _TILE.shape[0], n_len // _TILE.shape[1]
    if batch_origin == "runtime":
        kernel = _copy_kernel(src_td, dst_td, _block_origin)
        grid = (tiles_n, tiles_m, g_len)
    else:

        def every_batch(b):
            m0 = b.mul(b.block_id_y(), b.const_i32(_TILE.shape[0]))
            n0 = b.mul(b.block_id_x(), b.const_i32(_TILE.shape[1]))
            return [(g, m0, n0) for g in range(g_len)]

        kernel = _copy_kernel(src_td, dst_td, every_batch)
        grid = (tiles_n, tiles_m, 1)

    rng = np.random.default_rng(0)
    src = rng.integers(-1000, 1000, size=(g_len, m_len, n_len)).astype(np.float16)
    dst_elements = dst_td.element_span
    whole = _run_guarded(kernel, src, dst_elements, grid)
    dst = whole[_GUARD : _GUARD + dst_elements]
    for g in range(g_len):
        start = g * dst_batch_stride
        got = dst[start : start + m_len * n_len].reshape(m_len, n_len)
        np.testing.assert_array_equal(got, src[g], err_msg=f"batch {g}")
    written = np.zeros(whole.size, bool)
    for g in range(g_len):
        start = _GUARD + g * dst_batch_stride
        written[start : start + m_len * n_len] = True
    assert np.all(
        whole[~written] == _DST_INIT
    ), "a write landed outside the output tensor"
    assert not np.any(dst == _SRC_GUARD), "a read came from outside the input tensor"
