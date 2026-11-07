# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Global accesses through windows touch no element outside the tensor.

The verbs decide every bound check with the shared rule (see ``test_explain_bounds_checks``) and
take one of three global forms:

- every side proven, int strides, offsets inside i32: one i32 element index from the pointer;
- every side proven otherwise (a runtime stride, a slice's begin): the per-wave part of the address
  on the 64-bit pointer and the per-lane part as a zero-extended 32-bit byte offset;
- a side checked: the same placed address, every access under its validity -- a vector wholly
  valid is one wide access, a vector that straddles a bound has each valid element accessed alone,
  and an invalid element reads 0 and is not written.

These tests pin the IR and ISA shape of the forms (no in-bounds promise on an address an invalid
lane computes, no clamped load, the pointer-plus-offset instruction form, the row compare of an
M-ragged GEMM outside its K loop) and, on the GPU, that ragged and misplaced windows read and write
exactly the valid elements: guard zones around every buffer stay untouched, and no guard word is
read into a valid result. One test per defect the shared rule removed: a tile overhanging the
tensor although its extent divides the length, a tile wholly past the tensor, runtime origins
before and past the tensor, a runtime batch index outside the batch, a convolution halo.
"""

from __future__ import annotations

import ctypes
import re
import struct

import numpy as np
import pytest

from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType, Value
from rocke.helpers.tiling import (
    SideCheck,
    exit_if_tile_outside,
    explain_bounds_checks,
    lds_element_span,
    load_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.clipping import GlobalForm

WAVE = 64
TM = TK = 16

# (M, K) tile, K-contiguous: 4 consecutive K elements per lane (one vector), 16 lanes on M, 4 on K.
K_TILE = make_tile_desc(
    shape=[TM, TK],
    thread_tile=[1, 4],
    thread_dist=[16, 4],
    thread_order=[1, 0],
    wave_size=64,
)


def _builder(name: str, workgroup_size: int = WAVE) -> IRBuilder:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = workgroup_size
    return b


def _all_ops(b: IRBuilder) -> list:
    def walk(ops):
        for op in ops:
            yield op
            for region in op.regions:
                yield from walk(region.ops)

    return list(walk(b.kernel.body.ops))


def _block_origin(b: IRBuilder, tile: int, axis: str = "y") -> Value:
    block = {"x": b.block_id_x, "y": b.block_id_y, "z": b.block_id_z}[axis]()
    return b.mul(block, b.const_i32(tile))


# --- the GPU harness -------------------------------------------------------------------------------

GUARD = 1024  # guard elements before and after every buffer
SENTINEL = -7.0


def _gpu_arch() -> str | None:
    try:
        from rocke.runtime.hip_module import current_device, get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch(current_device())
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _launch(kernel, args, grid, block=WAVE):
    """Run ``kernel`` on the GPU present. ``args`` follows the kernel's parameters: a numpy array
    for a pointer (placed between two guard zones -- NaN around an array that holds data, the
    sentinel around one that is all sentinel) or an int. Returns every array with its guard zones.
    """
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    runtime = Runtime()
    function = runtime.load_module(
        compile_kernel(kernel, arch=arch).hsaco
    ).get_function(kernel.name)
    padded, packed, fmt, devices = [], [], "", []
    for value in args:
        if isinstance(value, np.ndarray):
            fill = SENTINEL if np.all(value == SENTINEL) else np.nan
            host = np.full(GUARD + value.size + GUARD, fill, dtype=value.dtype)
            host[GUARD : GUARD + value.size] = value.ravel()
            dev = runtime.alloc(host.nbytes)
            runtime.memcpy_h2d(
                dev, (ctypes.c_ubyte * host.nbytes).from_buffer(host), host.nbytes
            )
            padded.append(host)
            devices.append(dev)
            packed.append(dev + GUARD * host.itemsize)
            fmt += "Q"
        else:
            packed.append(int(value))
            fmt += "i"
    runtime.launch_blocking(function, grid, (block, 1, 1), struct.pack(fmt, *packed))
    for host, dev in zip(padded, devices):
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * host.nbytes).from_buffer(host), dev, host.nbytes
        )
    return padded


def _body(padded: np.ndarray, shape) -> np.ndarray:
    return padded[GUARD : GUARD + int(np.prod(shape))].reshape(shape)


def _guards_untouched(padded: np.ndarray) -> bool:
    return bool(
        np.all(padded[:GUARD] == SENTINEL) and np.all(padded[-GUARD:] == SENTINEL)
    )


# --- IR shape of the forms ------------------------------------------------------------------------


def _ragged_copy(name: str, *, dtype=F16, lda: int = 64):
    """A runtime-M (M, 64) K-contiguous copy, one 16x16 tile per workgroup, unguarded origins."""
    b = _builder(name)
    x = b.param("X", PtrType(dtype, "global"), align=16)
    y = b.param("Y", PtrType(dtype, "global"), align=16)
    ms = b.param("M", I32, ge=0, le=1 << 20)
    td = make_tensor_desc((ms, 64), (lda, 1), dtype)
    window = make_window(td, (_block_origin(b, TM), _block_origin(b, TK, "x")))
    lane = b.thread_id_x()
    store_fragment(b, y, window, load_fragment(b, x, window, K_TILE, lane), lane)
    b.ret()
    return b


def test_a_checked_access_makes_no_in_bounds_promise_and_clamps_nothing():
    from rocke.helpers.compile import compile_kernel

    b = _ragged_copy("masked_ir")
    ops = _all_ops(b)
    # No clamped load: no select feeds a load index, and every load and store sits under its
    # validity branch.
    selects = {op.result.name for op in ops if op.name == "arith.select"}
    for op in ops:
        if op.name in ("memref.global_load_vN", "memref.global_store_vN"):
            assert op.operands[1].name not in selects
    llvm = compile_kernel(b.kernel, arch="gfx90a").llvm_text
    for line in llvm.splitlines():
        if "getelementptr inbounds" in line and "addrspace(1)" in line:
            # The only in-bounds element step left is 0: every other part of the address is an
            # offset with no in-bounds promise.
            assert line.rstrip().endswith("i32 0"), line


def test_a_proven_window_with_a_runtime_stride_places_its_address_without_a_compare():
    harness = pytest.importorskip(
        "rocke.tests.helpers.tiling.tensor_transforms._harness"
    )
    b = _builder("placed_isa")
    x = b.param("X", PtrType(F16, "global"), align=16)
    y = b.param("Y", PtrType(F16, "global"), align=16)
    ms = b.param("M", I32, ge=TM, le=1 << 20, multiple_of=TM)
    lda = b.param("lda", I32, ge=64, le=1 << 16, multiple_of=8)
    td = make_tensor_desc((ms, 64), (lda, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), ms)
    k0 = exit_if_tile_outside(b, _block_origin(b, TK, "x"), 64)
    window = make_window(td, (m0, k0))
    plan = explain_bounds_checks(b, x, window, K_TILE)
    assert plan.form is GlobalForm.PLACED
    start = len(_all_ops(b))
    lane = b.thread_id_x()
    store_fragment(b, y, window, load_fragment(b, x, window, K_TILE, lane), lane)
    b.ret()
    emitted = _all_ops(b)[start:]
    assert not any(op.name in ("arith.cmp", "scf.if", "scf.if_else") for op in emitted)
    isa = harness.gfx90a_isa(b)
    body = isa[isa.index("<placed_isa>:") :]
    # Every access is the pointer-plus-32-bit-offset form: no 64-bit per-lane address arithmetic,
    # no loop to make a per-lane address uniform.
    assert harness.isa_count(body, r"global_load_dwordx2") > 0
    assert harness.isa_count(body, r"v_addc_co_u32\w*") == 0
    assert harness.isa_count(body, r"v_lshlrev_b64") == 0
    assert harness.isa_count(body, r"v_readfirstlane_b32") == 0
    assert harness.isa_count(body, r"s_cbranch_execnz") == 0


def test_a_per_lane_offset_past_the_byte_limit_is_refused():
    b = _builder("lane_limit")
    x = b.param("X", PtrType(F32, "global"), align=16)
    length = b.param("L", I32, ge=0, le=(1 << 31) - 1)
    td = make_tensor_desc((length,), (1,), F32)
    # A per-lane origin of up to 63 * 2^24 elements: 2^32 bytes of f32 past the base.
    spread = b.mul(b.thread_id_x(), b.const_i32(1 << 24))
    row = make_tile_desc(shape=[64], thread_tile=[1], thread_dist=[64], wave_size=64)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "lane offset can reach 4227858688 bytes (lane 1056964671 elements x 32 bits + 4 B "
            "vector + soffset 0 B), past the 2^31-byte buffer limit -- move a larger part of the "
            "address into the uniform (per-wave) coordinate"
        ),
    ):
        load_fragment(b, x, make_window(td, (spread,)), row, b.thread_id_x())


# --- options the verbs do not implement are refused, on every path -------------------------------

# Where an access goes: a global window proven inside its tensor (the one-index form), a global
# window with a checked side (the masked form), and an LDS buffer.
_ACCESS_PATHS = ("proven", "checked", "lds")


def _source_and_window(b: IRBuilder, path: str):
    """The source ``K_TILE`` is accessed through, and its window, for one of ``_ACCESS_PATHS``."""
    if path == "lds":
        td = make_tensor_desc((TM, TK), (TK, 1), F16)
        return b.smem_alloc(F16, [lds_element_span(td)]), make_window(td, (0, 0))
    x = b.param("X", PtrType(F16, "global"), align=16)
    if path == "proven":
        return x, make_window(make_tensor_desc((64, 64), (64, 1), F16), (0, 0))
    rows = b.param("M", I32, ge=0, le=1 << 20)
    td = make_tensor_desc((rows, 64), (64, 1), F16)
    return x, make_window(td, (_block_origin(b, TM), 0))


@pytest.mark.parametrize("path", _ACCESS_PATHS)
@pytest.mark.parametrize("pad", [1, -1.0, float("-inf")])
def test_load_fragment_refuses_a_non_zero_pad_before_emitting_anything(path, pad):
    b = _builder(f"pad_{path}")
    source, window = _source_and_window(b, path)
    lane = b.thread_id_x()
    emitted = len(_all_ops(b))
    with pytest.raises(NotImplementedError) as refused:
        load_fragment(b, source, window, K_TILE, lane, pad=pad)
    assert str(refused.value) == (
        f"load_fragment: pad={pad!r} -- only pad=0 is implemented (an element outside the "
        "tensor loads as 0); leave pad at its default"
    )
    assert len(_all_ops(b)) == emitted


def test_the_verbs_refuse_an_ir_value_as_pad_or_coherency_even_a_zero_one():
    b = _builder("option_values")
    source, window = _source_and_window(b, "checked")
    zero = b.const_i32(0)
    lane = b.thread_id_x()
    fragment = load_fragment(b, source, window, K_TILE, lane)
    emitted = len(_all_ops(b))
    with pytest.raises(TypeError) as pad_refused:
        load_fragment(b, source, window, K_TILE, lane, pad=zero)
    assert str(pad_refused.value) == (
        f"load_fragment: pad is the IR value {zero.name} -- pad is a Python number, and only "
        "pad=0 is implemented; leave pad at its default"
    )
    with pytest.raises(TypeError) as coherency_refused:
        store_fragment(b, source, window, fragment, lane, coherency=zero)
    assert str(coherency_refused.value) == (
        f"store_fragment: coherency is the IR value {zero.name} -- coherency is a Python int, "
        "and only its default 0 is implemented; leave coherency at its default"
    )
    assert len(_all_ops(b)) == emitted


@pytest.mark.parametrize("path", _ACCESS_PATHS)
def test_the_verbs_refuse_a_coherency_before_emitting_anything(path):
    b = _builder(f"coherency_{path}")
    source, window = _source_and_window(b, path)
    lane = b.thread_id_x()
    fragment = load_fragment(b, source, window, K_TILE, lane)
    emitted = len(_all_ops(b))
    with pytest.raises(NotImplementedError) as load_refused:
        load_fragment(b, source, window, K_TILE, lane, coherency=1)
    with pytest.raises(NotImplementedError) as store_refused:
        store_fragment(b, source, window, fragment, lane, coherency=1)
    for verb, refused in (("load", load_refused), ("store", store_refused)):
        assert str(refused.value) == (
            f"{verb}_fragment: coherency=1 -- no cache-coherency control is implemented; every "
            "access uses the default cache behaviour, so leave coherency at its default (0)"
        )
    assert len(_all_ops(b)) == emitted


# --- the row compare of an M-ragged GEMM stays out of its K loop --------------------------------------


def _ragged_m_gemm(name: str):
    """C = A * B^T with a runtime M and a runtime K stated a multiple of the K tile (full K
    tiles, the loop stop ``K - (TK - 1)`` proving every K step whole), an int N: A (M, K) and
    B (N, K) K-contiguous, one 16x16x16 MFMA per K step, C (M, N) N-contiguous in f32.
    """
    from rocke.helpers.tiling import TileMma, fill_fragment, make_fragment

    n_len = 64
    mma = TileMma(target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    op = mma.emit_op()
    b = _builder(name)
    a = b.param("A", PtrType(F16, "global"), align=16, readonly=True, noalias=True)
    bp = b.param("B", PtrType(F16, "global"), align=16, readonly=True, noalias=True)
    c = b.param("C", PtrType(F32, "global"), align=16, noalias=True)
    ms = b.param("M", I32, ge=0, le=1 << 20)
    ks = b.param("K", I32, ge=TK, le=1 << 16, multiple_of=TK)
    a_td = make_tensor_desc((ms, ks), (ks, 1), F16)
    b_td = make_tensor_desc((n_len, ks), (ks, 1), F16)
    c_td = make_tensor_desc((ms, n_len), (n_len, 1), F32)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), ms)
    n0 = exit_if_tile_outside(b, _block_origin(b, TM, "x"), n_len)
    c_tile = make_tile_desc(
        shape=[TM, TM],
        thread_tile=[4, 1],
        thread_dist=[4, 16],
        thread_order=[0, 1],
        wave_size=64,
    )
    lane = b.thread_id_x()
    acc = make_fragment(c_tile, F32)
    fill_fragment(b, acc, 0)
    stop = b.add(ks, b.const_i32(1 - TK))
    loop = b.scf_for_iter(
        b.const_i32(0), stop, b.const_i32(TK), [("acc", acc.value)], iv_name="k"
    )
    with loop as (k, (acc_value,)):
        a_window = make_window(a_td, (m0, k))
        plan = explain_bounds_checks(b, a, a_window, K_TILE)
        assert [(x.lower, x.upper) for x in plan.axes] == [
            (SideCheck.NOT_NEEDED, SideCheck.PER_LANE_CHECK),
            (SideCheck.NOT_NEEDED, SideCheck.NOT_NEEDED),
        ]
        fa = load_fragment(b, a, a_window, K_TILE, lane)
        fb = load_fragment(b, bp, make_window(b_td, (n0, k)), K_TILE, lane)
        b.scf_yield(b.mma(op, fa.value, fb.value, acc_value))
    result = make_fragment(c_tile, F32, loop.results[0])
    store_fragment(b, c, make_window(c_td, (m0, n0)), result, lane)
    b.ret()
    return b


def _loop_bodies(isa: str) -> list[list[str]]:
    """The instructions of every backward branch's range (a loop body), from a disassembly."""
    base = int(re.search(r"^([0-9a-f]+) <\S+>:", isa, re.MULTILINE).group(1), 16)
    instructions = []
    for line in isa.splitlines():
        m = re.match(r"\s+(\S.*?)\s*//\s*([0-9A-F]+):", line)
        if m:
            instructions.append((int(m.group(2), 16), m.group(1).strip(), line))
    bodies = []
    for address, text, line in instructions:
        target = re.search(r"<\S+\+0x([0-9a-f]+)>", line)
        if (
            target
            and text.startswith("s_c")
            and base + int(target.group(1), 16) <= address
        ):
            start = base + int(target.group(1), 16)
            bodies.append([t for a, t, _ in instructions if start <= a <= address])
    return bodies


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942"])
def test_the_row_compare_of_an_m_ragged_gemm_stays_out_of_the_k_loop(arch):
    import os
    import shutil
    import subprocess
    import tempfile

    from rocke.helpers.compile import compile_kernel

    objdump = shutil.which("llvm-objdump") or shutil.which(
        "llvm-objdump",
        path=os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm", "bin"),
    )
    if objdump is None:
        pytest.skip("llvm-objdump is not available")
    b = _ragged_m_gemm(f"ragged_m_{arch}")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kernel.hsaco")
        with open(path, "wb") as f:
            f.write(compile_kernel(b.kernel, arch=arch).hsaco)
        isa = subprocess.run(
            [objdump, "-d", path], capture_output=True, text=True, check=True
        ).stdout
    # The K loop: the smallest backward-branch range holding the MFMA.
    body = min(
        (b for b in _loop_bodies(isa) if any(t.startswith("v_mfma") for t in b)),
        key=len,
    )
    # The A rows' M compare reads loop-invariant values only: it is computed before the loop.
    assert not [t for t in body if t.startswith("v_cmp")]
    # The loop's only M-side work per vector is the access under the exec mask.
    assert any(t.startswith("s_and_saveexec") for t in body)


# --- GPU: ragged and misplaced windows read and write exactly the valid elements ---------------------


def _gather_kernel(name, *, dtype, lengths, strides, tile, origin, out_shape):
    """``Y[tile at (by*TR, bx*TC)] = X[window at origin(b)]``: X read through a window that may
    leave the tensor (``origin`` builds its origin from the builder), Y written through a proven
    window. ``lengths`` entries that are ``"M"`` / ``"K"`` become runtime parameters."""
    b = _builder(name)
    x = b.param("X", PtrType(dtype, "global"), align=16)
    y = b.param("Y", PtrType(dtype, "global"), align=16)
    stated = {}
    for symbol in ("M", "K", "O", "G"):
        if symbol in lengths or symbol in strides:
            stated[symbol] = b.param(
                symbol, I32, ge=1 if symbol in strides else 0, le=1 << 20
            )
    xt = make_tensor_desc(
        tuple(stated.get(n, n) for n in lengths),
        tuple(stated.get(s, s) for s in strides),
        dtype,
    )
    rank = len(tile.shape)
    yt = make_tensor_desc(out_shape, (out_shape[1], 1), dtype)
    y0 = exit_if_tile_outside(b, _block_origin(b, tile.shape[0]), out_shape[0])
    y1 = exit_if_tile_outside(b, _block_origin(b, tile.shape[1], "x"), out_shape[1])
    lane = b.thread_id_x()
    window = make_window(xt, origin(b, y0, y1))
    frag = load_fragment(b, x, window, tile, lane)
    assert rank == 2
    store_fragment(b, y, make_window(yt, (y0, y1)), frag, lane)
    b.ret()
    return b


def _scatter_kernel(name, *, dtype, lengths, strides, tile, origin, in_shape):
    """``Y[window at origin(b)] = X[tile at (by*TR, bx*TC)]``: the store goes through the window
    that may leave the tensor."""
    b = _builder(name)
    x = b.param("X", PtrType(dtype, "global"), align=16)
    y = b.param("Y", PtrType(dtype, "global"), align=16)
    stated = {}
    for symbol in ("M", "K", "O", "G"):
        if symbol in lengths or symbol in strides:
            stated[symbol] = b.param(
                symbol, I32, ge=1 if symbol in strides else 0, le=1 << 20
            )
    yt = make_tensor_desc(
        tuple(stated.get(n, n) for n in lengths),
        tuple(stated.get(s, s) for s in strides),
        dtype,
    )
    xt = make_tensor_desc(in_shape, (in_shape[1], 1), dtype)
    x0 = exit_if_tile_outside(b, _block_origin(b, tile.shape[0]), in_shape[0])
    x1 = exit_if_tile_outside(b, _block_origin(b, tile.shape[1], "x"), in_shape[1])
    lane = b.thread_id_x()
    frag = load_fragment(b, x, make_window(xt, (x0, x1)), tile, lane)
    store_fragment(b, y, make_window(yt, origin(b, x0, x1)), frag, lane)
    b.ret()
    return b


def _np(dtype):
    return np.float16 if dtype is F16 else np.float32


def _expected_gather(src, positions):
    """``positions[i, j]`` -> the source element there, 0 outside ``src``."""
    out = np.zeros(positions.shape[:-1], dtype=src.dtype)
    for index in np.ndindex(out.shape):
        p = tuple(positions[index])
        if all(0 <= c < n for c, n in zip(p, src.shape)):
            out[index] = src[p]
    return out


# Every case reads a (rows, cols) source through a window of a 16x16 K-contiguous tile.
_RAGGED_CASES = {
    # name: (rows, cols, row shift, extra lda): the window of output tile (r, c) starts at
    # (r*16 + row shift, c*16); the source's row stride is cols + extra lda.
    "m_ragged": (50, 64, 0, 0),
    "k_ragged": (64, 30, 0, 2),
    "runtime_length_0": (0, 64, 0, 0),
    "negative_origin": (64, 64, -5, 0),
    "odd_lda": (50, 63, 0, 1),
}


@pytest.mark.parametrize("dtype", [F16, F32], ids=["f16", "f32"])
@pytest.mark.parametrize("case", list(_RAGGED_CASES))
def test_gpu_a_ragged_window_reads_exactly_the_valid_elements(case, dtype):
    rows, cols, shift, extra = _RAGGED_CASES[case]
    lda = cols + extra
    out_shape = (64, 64)

    def origin(b, y0, y1):
        return (b.add(y0, b.const_i32(shift)), y1)

    b = _gather_kernel(
        f"gather_{case}_{dtype.name}",
        dtype=dtype,
        lengths=("M", "K"),
        strides=("O", 1),
        tile=K_TILE,
        origin=origin,
        out_shape=out_shape,
    )
    rng = np.random.default_rng(1)
    src = rng.integers(-50, 50, size=(max(rows, 1), lda)).astype(_np(dtype))
    y = np.full(out_shape, SENTINEL, dtype=_np(dtype))
    _, y_out = _launch(b.kernel, [src, y, rows, cols, lda], grid=(4, 4, 1))
    rr, cc = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    positions = np.stack([rr + shift, cc], axis=-1)
    expected = _expected_gather(src[:rows, :cols], positions)
    assert np.array_equal(_body(y_out, out_shape), expected)


@pytest.mark.parametrize("dtype", [F16, F32], ids=["f16", "f32"])
@pytest.mark.parametrize("case", list(_RAGGED_CASES))
def test_gpu_a_ragged_window_writes_exactly_the_valid_elements(case, dtype):
    rows, cols, shift, extra = _RAGGED_CASES[case]
    lda = cols + extra
    in_shape = (64, 64)

    def origin(b, x0, x1):
        return (b.add(x0, b.const_i32(shift)), x1)

    b = _scatter_kernel(
        f"scatter_{case}_{dtype.name}",
        dtype=dtype,
        lengths=("M", "K"),
        strides=("O", 1),
        tile=K_TILE,
        origin=origin,
        in_shape=in_shape,
    )
    rng = np.random.default_rng(2)
    src = rng.integers(-50, 50, size=in_shape).astype(_np(dtype))
    y = np.full((max(rows, 1), lda), SENTINEL, dtype=_np(dtype))
    _, y_out = _launch(b.kernel, [src, y, rows, cols, lda], grid=(4, 4, 1))
    assert _guards_untouched(y_out)
    out = _body(y_out, y.shape)
    expected = np.full(y.shape, SENTINEL, dtype=y.dtype)
    for r in range(rows):
        for c in range(cols):
            if 0 <= r - shift < 64:
                expected[r, c] = src[r - shift, c]
    assert np.array_equal(out, expected)


def test_gpu_a_tile_wholly_past_the_tensor_reads_zeros_and_writes_nothing():
    # A 16x16 tensor and a 16x16 tile at the int origin (16, 0): every element is outside.
    def gather_origin(b, y0, y1):
        return (16, 0)

    b = _gather_kernel(
        "past_gather",
        dtype=F32,
        lengths=(16, 16),
        strides=(16, 1),
        tile=K_TILE,
        origin=gather_origin,
        out_shape=(16, 16),
    )
    src = np.arange(256, dtype=np.float32).reshape(16, 16)
    _, y = _launch(
        b.kernel, [src, np.full((16, 16), SENTINEL, np.float32)], grid=(1, 1, 1)
    )
    assert np.all(_body(y, (16, 16)) == 0)

    b = _scatter_kernel(
        "past_scatter",
        dtype=F32,
        lengths=(16, 16),
        strides=(16, 1),
        tile=K_TILE,
        origin=gather_origin,
        in_shape=(16, 16),
    )
    _, y = _launch(
        b.kernel, [src, np.full((16, 16), SENTINEL, np.float32)], grid=(1, 1, 1)
    )
    assert _guards_untouched(y) and np.all(_body(y, (16, 16)) == SENTINEL)


@pytest.mark.parametrize("row", [-16, -8, 16, 56, 64])
def test_gpu_a_runtime_origin_before_or_past_the_tensor(row):
    # A (64, 16) tensor read and written at a runtime row origin: rows outside read 0 and are not
    # written.
    def origin(b, _y0, _y1):
        return (b.param("R", I32), 0)

    b = _gather_kernel(
        f"runtime_origin_gather_{row + 100}",
        dtype=F32,
        lengths=(64, 16),
        strides=(16, 1),
        tile=K_TILE,
        origin=origin,
        out_shape=(16, 16),
    )
    src = np.arange(1, 64 * 16 + 1, dtype=np.float32).reshape(64, 16)
    (
        _,
        y,
    ) = _launch(
        b.kernel, [src, np.full((16, 16), SENTINEL, np.float32), row], grid=(1, 1, 1)
    )
    rr, cc = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    assert np.array_equal(
        _body(y, (16, 16)), _expected_gather(src, np.stack([rr + row, cc], axis=-1))
    )

    b = _scatter_kernel(
        f"runtime_origin_scatter_{row + 100}",
        dtype=F32,
        lengths=(64, 16),
        strides=(16, 1),
        tile=K_TILE,
        origin=origin,
        in_shape=(16, 16),
    )
    tile = np.arange(1, 257, dtype=np.float32).reshape(16, 16)
    _, y = _launch(
        b.kernel, [tile, np.full((64, 16), SENTINEL, np.float32), row], grid=(1, 1, 1)
    )
    expected = np.full((64, 16), SENTINEL, np.float32)
    for r in range(16):
        if 0 <= r + row < 64:
            expected[r + row] = tile[r]
    assert _guards_untouched(y) and np.array_equal(_body(y, (64, 16)), expected)


def _batch_kernel(name, *, store: bool):
    """A (2, 16, 16) f32 tensor at a runtime batch index ``G``: read into, or written from, a
    16x16 tile."""
    b = _builder(name)
    x = b.param("X", PtrType(F32, "global"), align=16)
    y = b.param("Y", PtrType(F32, "global"), align=16)
    g = b.param("G", I32)
    batched = make_tensor_desc((2, 16, 16), (256, 16, 1), F32)
    flat = make_tensor_desc((16, 16), (16, 1), F32)
    lane = b.thread_id_x()
    if store:
        frag = load_fragment(b, x, make_window(flat, (0, 0)), K_TILE, lane)
        store_fragment(b, y, make_window(batched, (g, 0, 0)), frag, lane)
    else:
        frag = load_fragment(b, x, make_window(batched, (g, 0, 0)), K_TILE, lane)
        store_fragment(b, y, make_window(flat, (0, 0)), frag, lane)
    b.ret()
    return b


@pytest.mark.parametrize("g", [-1, 0, 1, 2])
def test_gpu_a_runtime_batch_index_outside_the_batch_reads_zeros_and_writes_nothing(g):
    src = np.arange(1, 513, dtype=np.float32).reshape(2, 16, 16)
    _, y = _launch(
        _batch_kernel(f"batch_read_{g + 1}", store=False).kernel,
        [src, np.full((16, 16), SENTINEL, np.float32), g],
        grid=(1, 1, 1),
    )
    expected = src[g] if 0 <= g < 2 else np.zeros((16, 16), np.float32)
    assert np.array_equal(_body(y, (16, 16)), expected)

    tile = np.arange(1, 257, dtype=np.float32).reshape(16, 16)
    _, y = _launch(
        _batch_kernel(f"batch_write_{g + 1}", store=True).kernel,
        [tile, np.full((2, 16, 16), SENTINEL, np.float32), g],
        grid=(1, 1, 1),
    )
    expected = np.full((2, 16, 16), SENTINEL, np.float32)
    if 0 <= g < 2:
        expected[g] = tile
    assert _guards_untouched(y) and np.array_equal(_body(y, (2, 16, 16)), expected)


def test_gpu_a_batched_window_loads_wide_and_bit_exact():
    # (G, M, K) f16 read with a rank-2 K-contiguous tile at a runtime batch index: the vector axis
    # is window axis 2 and every load is 4 halves wide.
    b = _builder("batched_wide")
    x = b.param("X", PtrType(F16, "global"), align=16)
    y = b.param("Y", PtrType(F16, "global"), align=16)
    g = b.param("G", I32)
    batched = make_tensor_desc((3, 32, 64), (32 * 64, 64, 1), F16)
    out = make_tensor_desc((32, 64), (64, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 32)
    k0 = exit_if_tile_outside(b, _block_origin(b, TK, "x"), 64)
    lane = b.thread_id_x()
    window = make_window(batched, (g, m0, k0))
    assert explain_bounds_checks(b, x, window, K_TILE).vector_axis == 2
    frag = load_fragment(b, x, window, K_TILE, lane)
    store_fragment(b, y, make_window(out, (m0, k0)), frag, lane)
    b.ret()
    loads = [op for op in _all_ops(b) if op.name == "memref.global_load_vN"]
    assert loads and all(op.attrs["vec"] == 4 for op in loads)
    src = (
        np.random.default_rng(3).integers(-50, 50, size=(3, 32, 64)).astype(np.float16)
    )
    _, y_out = _launch(
        b.kernel, [src, np.full((32, 64), SENTINEL, np.float16), 2], grid=(4, 2, 1)
    )
    assert np.array_equal(_body(y_out, (32, 64)), src[2])


def test_gpu_a_convolution_halo_matches_im2col():
    # A 3x3, padding-1 im2col of a (20, 20) f32 image: workgroup (t, row) reads image row
    # row + t // 3 - 1 from column t % 3 - 1, 256 columns wide (4 per lane), into Y[t, row, :].
    # Rows and columns outside the image read 0.
    h = w = 20
    b = _builder("conv_halo")
    x = b.param("X", PtrType(F32, "global"), align=16)
    y = b.param("Y", PtrType(F32, "global"), align=16)
    image = make_tensor_desc((h, w), (w, 1), F32)
    cols = make_tensor_desc((9, h, 256), (h * 256, 256, 1), F32)
    row_tile = make_tile_desc(
        shape=[1, 256], thread_tile=[1, 4], thread_dist=[1, 64], wave_size=64
    )
    tap = exit_if_tile_outside(b, b.block_id_x(), 9)
    row = exit_if_tile_outside(b, b.block_id_y(), h)
    dy = b.add(b.div(tap, b.const_i32(3)), b.const_i32(-1))
    dx = b.add(b.mod(tap, b.const_i32(3)), b.const_i32(-1))
    lane = b.thread_id_x()
    frag = load_fragment(b, x, make_window(image, (b.add(row, dy), dx)), row_tile, lane)
    store_fragment(b, y, make_window(cols, (tap, row, 0)), frag, lane)
    b.ret()
    img = np.random.default_rng(4).integers(-50, 50, size=(h, w)).astype(np.float32)
    _, y_out = _launch(
        b.kernel, [img, np.full((9, h, 256), SENTINEL, np.float32)], grid=(9, h, 1)
    )
    padded = np.zeros((h + 2, 256 + 2), np.float32)
    padded[1 : h + 1, 1 : w + 1] = img
    expected = np.stack(
        [padded[t // 3 : t // 3 + h, t % 3 : t % 3 + 256] for t in range(9)]
    )
    assert np.array_equal(_body(y_out, (9, h, 256)), expected)


def test_gpu_a_tile_over_the_edge_of_a_wave_split_store_writes_nothing_past_n():
    # The CRC demo at 256x128x64 with its 256x256 tile over 16 waves: each wave's 64-wide C tile
    # is a multiple of N, but the waves past N = 128 write outside C; their stores are checked.
    from rocke.helpers.tiling.kernels.tiling_gemm_crc_demo.crc_interleaved_gemm import (
        build_crc_gemm,
    )

    m_len, n_len, k_len = 256, 128, 64
    kernel, mma = build_crc_gemm(m_len, n_len, k_len, arch=_gpu_arch() or "gfx90a")
    rng = np.random.default_rng(0)
    a = rng.integers(-3, 4, size=(m_len, k_len)).astype(np.float16)
    bm = rng.integers(-3, 4, size=(n_len, k_len)).astype(np.float16)
    c = np.full((n_len, m_len), SENTINEL, np.float32)  # C (M, N) M-contiguous
    _, _, c_out = _launch(
        kernel,
        [np.ascontiguousarray(a.T), np.ascontiguousarray(bm.T), c, m_len, n_len, k_len],
        grid=(1, 1, 1),
        block=mma.wave_size * 16,
    )
    assert _guards_untouched(c_out)
    reference = a.astype(np.float32) @ bm.astype(np.float32).T
    assert np.array_equal(_body(c_out, (n_len, m_len)).T, reference)
