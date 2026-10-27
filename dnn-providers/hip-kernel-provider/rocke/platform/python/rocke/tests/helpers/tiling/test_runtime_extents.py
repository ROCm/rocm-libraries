# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime tensor lengths and strides: a ``TensorDesc`` takes an i32 ``Value`` whose facts are
stated.

The facts live where they are stated -- a kernel parameter's declaration (checked before each
launch through ``KernelLauncher.from_artifact``) or the Value ``b.gpu_checked_value`` returned (the
GPU checks it) -- and the description reads them with ``stated_facts_of``. These tests pin:

- what a runtime length and a runtime stride need, and every refusal with its exact message: no
  stated facts (including the raw value passed instead of the checked one), a length that could be
  negative or has no upper bound, a stride that could be below 1, a Value that differs between
  lanes, a non-i32 Value, a runtime stride where the contiguous axis must keep the int stride 1;
- that a window over a runtime-length, runtime-stride tensor builds and both engines lower it;
- that ``check_gemm_shapes`` agrees runtime extents only when they are one and the same Value;
- that the readers which need the whole layout at build time refuse a runtime extent;
- on the host, that a kernel's contract accepts an aligned-shape kernel exactly for tile-multiple
  shapes and a ragged one for every in-range shape; on the GPU, with launchers built by
  ``KernelLauncher.from_artifact``, that ``first_supported`` picks between them, that the ragged
  launcher refuses an out-of-range launch before it reaches the GPU, and that the copy writes
  exactly its rows.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rocke.core.ir import F16, F32, I32, I64, IRBuilder, PtrType
from rocke.helpers.tiling import (
    TensorDesc,
    check_gemm_shapes,
    load_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.memory import cooperative_load_desc
from rocke.runtime.contract import LaunchRefused, parameter_contract
from rocke.tests.engines import lower_with_both_engines

FLAVORS = ("llvm20", "llvm22", "llvm23")
TILE_ROWS = 64
ROW = 16
PLATFORM_PYTHON = Path(__file__).resolve().parents[4]


def _builder(name: str = "runtime_extent_test") -> IRBuilder:
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    return b


def _raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


def _loaded(b: IRBuilder, name: str = "cu"):
    """A wave-uniform value read from memory: the GPU can see it, the host cannot."""
    cu = b.param(name, PtrType(I32, "global"), readonly=True, noalias=True)
    return b.readfirstlane(b.global_load(cu, b.block_id_x(), I32))


# --- what a runtime extent needs -------------------------------------------------------------


def test_tensor_desc_takes_parameter_values_as_lengths_and_strides():
    b = _builder()
    m = b.param("M", I32, ge=0, le=1 << 20)
    k = b.param("K", I32, ge=1, le=1 << 20)
    td = make_tensor_desc((m, k), (k, 1), F16)
    assert td.lengths == (m, k)
    assert td.strides[0] is k
    assert td.begin == (0, 0)


def test_tensor_desc_takes_a_value_gpu_checked_value_returned():
    b = _builder()
    t = b.param("T", I32, ge=0, le=4096)
    seqlen = b.gpu_checked_value(_loaded(b), ge=0, le=t)
    td = make_tensor_desc((seqlen, 64), (64, 1), F16)
    assert td.lengths[0] is seqlen


def test_a_stride_stated_by_a_relation_takes_its_lower_bound_from_it():
    # ldc ge N with N ge 1: ldc is at least 1 without a word of its own.
    b = _builder()
    n = b.param("N", I32, ge=1, le=4096)
    ldc = b.param("ldc", I32, ge=n, le=4096)
    td = make_tensor_desc((64, n), (ldc, 1), F16)
    assert td.strides[0] is ldc


def test_a_zero_or_non_int_length_is_refused():
    with _raises(
        ValueError,
        "TensorDesc lengths[1] = 0 -- a length is a positive int or an i32 Value with stated "
        "facts",
    ):
        make_tensor_desc((16, 0), (16, 1), F16)


def test_a_value_with_no_stated_facts_is_refused():
    b = _builder()
    no_facts = (
        "which has no stated facts. State them on its b.param declaration, or check it with "
        "b.gpu_checked_value; if you checked it with gpu_checked_value, pass the Value "
        "gpu_checked_value returned"
    )
    with _raises(ValueError, f"TensorDesc lengths[0] is M, {no_facts}"):
        make_tensor_desc((b.param("M", I32), 16), (16, 1), F16)
    with _raises(ValueError, f"TensorDesc strides[0] is lda, {no_facts}"):
        make_tensor_desc((16, 16), (b.param("lda", I32), 1), F16)


def test_the_raw_value_of_a_checked_value_is_refused_with_the_value_to_pass():
    # The facts live only on the Value gpu_checked_value returned; the raw load it checked has
    # none, and the message names the fix.
    b = _builder()
    t = b.param("T", I32, ge=0, le=4096)
    raw = _loaded(b)
    b.gpu_checked_value(raw, ge=0, le=t)
    with _raises(
        ValueError,
        f"TensorDesc lengths[0] is {raw.name}, which has no stated facts. State them on its "
        "b.param declaration, or check it with b.gpu_checked_value; if you checked it with "
        "gpu_checked_value, pass the Value gpu_checked_value returned",
    ):
        make_tensor_desc((raw, 16), (16, 1), F16)


def test_a_length_that_could_be_negative_is_refused():
    b = _builder()
    with _raises(
        ValueError,
        "TensorDesc lengths[0] is M, which could be negative (declared with no lower bound).\n"
        'Fix: b.param("M", I32, ge=0, ...)',
    ):
        make_tensor_desc((b.param("M", I32, le=4096), 16), (16, 1), F16)
    with _raises(
        ValueError,
        "TensorDesc lengths[0] is N, which could be negative (declared with the lower bound "
        '-4).\nFix: b.param("N", I32, ge=0, ...)',
    ):
        make_tensor_desc((b.param("N", I32, ge=-4, le=4096), 16), (16, 1), F16)
    t = b.param("T", I32, ge=0, le=4096)
    checked = b.gpu_checked_value(_loaded(b), le=t)
    with _raises(
        ValueError,
        f"TensorDesc lengths[0] is {checked.name}, which could be negative (checked with no "
        "lower bound).\nFix: b.gpu_checked_value(..., ge=0, ...)",
    ):
        make_tensor_desc((checked, 16), (16, 1), F16)


def test_a_length_with_no_upper_bound_is_refused():
    # Every range proof over a runtime length reads its upper end, so it is always stated.
    b = _builder()
    with _raises(
        ValueError,
        "TensorDesc lengths[0] is M, which has no upper bound: state le= or lt=, the tightest "
        "bound it can reach",
    ):
        make_tensor_desc((b.param("M", I32, ge=1), 16), (16, 1), F16)


def test_a_stride_that_could_be_below_one_is_refused():
    b = _builder()
    with _raises(
        ValueError,
        "TensorDesc strides[0] is lda, which could be below 1 (declared with the lower bound 0)."
        '\nFix: b.param("lda", I32, ge=1, ...), or ge= a length it is at least',
    ):
        make_tensor_desc((16, 16), (b.param("lda", I32, ge=0, le=4096), 1), F16)
    with _raises(
        ValueError,
        "TensorDesc strides[0] is ldb, which could be below 1 (declared with no lower bound)."
        '\nFix: b.param("ldb", I32, ge=1, ...), or ge= a length it is at least',
    ):
        make_tensor_desc((16, 16), (b.param("ldb", I32, le=4096), 1), F16)


def test_an_i64_value_is_refused():
    b = _builder()
    with _raises(
        TypeError,
        "TensorDesc lengths[0] is M, a i64 Value -- a runtime length or stride is an i32 Value",
    ):
        make_tensor_desc((b.param("M", I64, ge=0, le=64), 16), (16, 1), F16)


def test_a_value_that_differs_between_lanes_is_refused():
    # A length is the same for the whole wave. A value each lane loads for itself is checked per
    # lane by gpu_checked_value, and refused as a length.
    b = _builder()
    lens = b.param("lens", PtrType(I32, "global"), readonly=True, noalias=True)
    per_lane = b.gpu_checked_value(
        b.global_load(lens, b.thread_id_x(), I32), ge=0, le=4096
    )
    with _raises(
        ValueError,
        f"TensorDesc lengths[0] is {per_lane.name}, which can differ between the lanes of a "
        "wave (it depends on a thread id or a per-lane load) -- a length or a stride is the same "
        "for the whole wave; pass a value loaded from memory through b.readfirstlane before "
        "checking it",
    ):
        make_tensor_desc((per_lane, 16), (16, 1), F16)


def test_a_runtime_stride_on_the_contiguous_axis_is_refused():
    # No axis keeps the compile-time stride 1: the runtime stride sits where the contiguous axis
    # must be, so every access would be a gather.
    b = _builder()
    k = b.param("K", I32, ge=1, le=4096)
    with _raises(
        ValueError,
        "TensorDesc strides[1] is a runtime value on the stride-1 axis -- the contiguous axis "
        "must have the compile-time stride 1",
    ):
        make_tensor_desc((16, 16), (16, k), F16)


def test_element_span_is_static():
    # An LDS buffer is sized from element_span at build, so a runtime extent has no span.
    b = _builder()
    k = b.param("K", I32, ge=1, le=4096)
    td = make_tensor_desc((16, k), (k, 1), F16)
    with _raises(
        ValueError,
        "TensorDesc.element_span is static -- lengths[1] is a runtime value (K); LDS tensors are "
        "sized at build",
    ):
        _ = td.element_span


def test_permute_carries_runtime_extents():
    b = _builder()
    m = b.param("M", I32, ge=0, le=4096)
    ld = b.param("ld", I32, ge=1, le=4096)
    view = make_tensor_desc((m, 16), (1, ld), F16).permute([1, 0])
    assert view.lengths[1] is m and view.strides[0] is ld


def test_tensor_desc_is_still_a_value():
    td = TensorDesc((16, 16), (16, 1), F16)
    assert td == make_tensor_desc((16, 16), (16, 1), F16)
    assert td.begin == (0, 0)


# --- windows over runtime extents ------------------------------------------------------------


def copy_kernel(name: str, *, runtime_stride: bool = False, **m_facts):
    """Copy the first M rows of an (M, 16) f32 tensor, one 64-row tile per workgroup. ``M`` is a
    kernel parameter declared with ``m_facts``; with ``runtime_stride`` the row stride is the
    parameter ``ld`` (16 to 64)."""
    b = _builder(name)
    x = b.param("X", PtrType(F32, "global"), align=4)
    y = b.param("Y", PtrType(F32, "global"), align=4)
    m = b.param("M", I32, **m_facts)
    ld = b.param("ld", I32, ge=ROW, le=64) if runtime_stride else ROW
    td = make_tensor_desc((m, ROW), (ld, 1), F32)
    tile = cooperative_load_desc(TILE_ROWS, ROW, 1, vw=1)
    origin = (b.mul(b.block_id_x(), b.const_i32(TILE_ROWS)), 0)
    lane = b.thread_id_x()
    frag = load_fragment(b, x, make_window(td, origin), tile, lane)
    store_fragment(b, y, make_window(td, origin), frag, lane)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("runtime_stride", [False, True])
def test_a_window_over_runtime_extents_builds_and_both_engines_lower_it(runtime_stride):
    kernel = copy_kernel(
        "runtime_extent_copy", runtime_stride=runtime_stride, ge=0, le=4096
    )
    for flavor in FLAVORS:
        text = lower_with_both_engines(kernel, "gfx90a", llvm_flavor=flavor)
        # Every row is clipped against M itself: the compares read its Value.
        assert "icmp slt" in text


# --- check_gemm_shapes -----------------------------------------------------------------------


def test_check_gemm_shapes_accepts_one_runtime_k_and_refuses_two():
    b = _builder()
    m = b.param("M", I32, ge=0, le=4096)
    n = b.param("N", I32, ge=0, le=4096)
    k = b.param("K", I32, ge=1, le=4096)
    a = make_tensor_desc((m, k), (k, 1), F16)
    bt = make_tensor_desc((n, k), (k, 1), F16)
    c = make_tensor_desc((m, n), (4096, 1), F32)
    check_gemm_shapes(a, bt, c)
    other_k = b.param("K2", I32, ge=1, le=4096)
    b_other = make_tensor_desc((n, other_k), (other_k, 1), F16)
    with _raises(
        ValueError,
        "check_gemm_shapes: A's K (axis 1) is K but B's K (axis 1) is K2 -- GEMM tensors are "
        "read as A (M, K), B (N, K), C (M, N), memory order in the strides; a B stored "
        "N-contiguous (stride 1 on N) is make_tensor_desc((N, K), (1, ldb), dtype)",
    ):
        check_gemm_shapes(a, b_other)


def test_check_gemm_shapes_names_one_value_checked_twice():
    # Two gpu_checked_value calls on one loaded value give two Values, which never agree. The
    # message says so, instead of the layout advice a real mismatch gets.
    b = _builder()
    t = b.param("T", I32, ge=1, le=4096)
    raw = _loaded(b)
    k_a = b.gpu_checked_value(raw, ge=1, le=t)
    k_b = b.gpu_checked_value(raw, ge=1, le=t)
    assert k_a is not k_b
    a = make_tensor_desc((64, k_a), (k_a, 1), F16)
    bt = make_tensor_desc((64, k_b), (k_b, 1), F16)
    with _raises(
        ValueError,
        f"check_gemm_shapes: A's K (axis 1) is {k_a.name} but B's K (axis 1) is {k_b.name} -- "
        f"both check {raw.name} with their own gpu_checked_value call; check it once and pass "
        "the one Value it returned to every tensor",
    ):
        check_gemm_shapes(a, bt)


# --- readers that need the whole layout at build time -----------------------------------------


def test_a_recorded_runtime_stride_is_refused_by_the_address_analyses():
    # The layout renders and the coalescing report compute concrete addresses from a recorded
    # access; a runtime stride has no value at build time.
    from rocke.helpers.tiling.tiling_recorder import PipelineTransaction

    b = _builder()
    ld = b.param("ld", I32, ge=16, le=64)
    tile = cooperative_load_desc(64, 16, 1, vw=1)

    def transaction(strides):
        return PipelineTransaction(
            kind="load",
            seq=0,
            space="global",
            space_id=0,
            space_name="X",
            tile_desc=tile,
            dtype_name="f32",
            strides=strides,
            lengths=(64, 16),
            origin=(0, 0),
            register_count=tile.register_count,
            vw=1,
            swizzle=None,
        )

    assert transaction((16, 1)).static_strides() == (16, 1)
    with _raises(
        ValueError,
        f"transaction load/X: strides[0] is a runtime value ({ld!r}) -- this analysis computes "
        "concrete addresses; record a kernel built with int strides",
    ):
        transaction((ld, 1)).static_strides()


def test_a_runtime_extent_on_an_lds_tensor_has_no_span():
    b = _builder()
    k = b.param("K", I32, ge=1, le=4096)
    td = make_tensor_desc((64, k), (k, 1), F32)
    tile = cooperative_load_desc(64, 16, 1, vw=1)
    lds = b.smem_alloc(F32, [64 * 16])
    with pytest.raises(
        ValueError, match=r"^TensorDesc.element_span is static -- lengths\[1\]"
    ):
        load_fragment(b, lds, make_window(td, (0, 0)), tile, b.thread_id_x())


# --- picking a kernel by its contract ----------------------------------------------------------

_M_FACTS = {"ge": 0, "le": 4096}


def test_the_contract_accepts_the_aligned_kernel_exactly_for_tile_multiple_shapes():
    aligned = parameter_contract(
        copy_kernel("choice_aligned", **_M_FACTS, multiple_of=TILE_ROWS)
    )
    ragged = parameter_contract(copy_kernel("choice_ragged", **_M_FACTS))
    for m in range(0, 4097, 13):
        assert aligned.check({"M": m}).ok == (m % TILE_ROWS == 0), m
        assert ragged.check({"M": m}).ok, m
    assert aligned.check({"M": 4096}).ok
    assert not ragged.check({"M": 4097}).ok


def test_the_names_of_the_removed_runtime_wrapper_are_gone():
    # Facts live on b.param and gpu_checked_value only: no wrapper type, constructor or tiling
    # launch record remains in the package or its docs.
    removed = ["Runtime" + "Value", "runtime" + "_value", "launch" + "_supported"]
    found = []
    for path in sorted(PLATFORM_PYTHON.joinpath("rocke").rglob("*")):
        if path.suffix not in (".py", ".md") or "tmp" in path.parts:
            continue
        text = path.read_text(errors="replace")
        found.extend(f"{path.name}: {name}" for name in removed if name in text)
    assert found == []


# --- on the GPU ------------------------------------------------------------------------------


def _gpu_launchers():
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher
    from rocke.tests.gpu import current_device_arch

    arch = current_device_arch()
    aligned = copy_kernel("gpu_copy_aligned", **_M_FACTS, multiple_of=TILE_ROWS)
    ragged = copy_kernel("gpu_copy_ragged", **_M_FACTS)
    return tuple(
        KernelLauncher.from_artifact(compile_kernel(k, arch=arch))
        for k in (aligned, ragged)
    )


def _copy_on_gpu(launcher, m: int) -> tuple[bool, bool]:
    """Launch ``launcher`` on M rows; return whether the rows were copied and whether every guard
    row past them still holds its sentinel."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem, LaunchConfig, synchronize_and_release

    tiles = max(1, -(-m // TILE_ROWS))
    rows = tiles * TILE_ROWS + TILE_ROWS
    sentinel = np.float32(-7.0)
    x = np.arange(rows * ROW, dtype=np.float32)
    y = np.full(rows * ROW, sentinel, dtype=np.float32)
    rt = Runtime()
    x_dev, y_dev = DeviceMem(x.nbytes), DeviceMem(y.nbytes)
    rt.memcpy_h2d(x_dev.ptr(), as_u8_buffer(x), x.nbytes)
    rt.memcpy_h2d(y_dev.ptr(), as_u8_buffer(y), y.nbytes)
    launcher(
        {"X": x_dev, "Y": y_dev, "M": m},
        config=LaunchConfig(grid=(tiles, 1, 1), block=(64, 1, 1)),
    )
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(y), y_dev.ptr(), y.nbytes)
    copied = bool(np.array_equal(y[: m * ROW], x[: m * ROW]))
    guard = bool(np.all(y[m * ROW :] == sentinel))
    return copied, guard


@pytest.mark.parametrize("m", [16, 104, 128, 4096])
def test_gpu_first_supported_picks_a_kernel_that_copies_exactly_its_rows(m):
    from rocke.runtime.launcher import first_supported

    aligned, ragged = _gpu_launchers()
    launcher = first_supported([aligned, ragged], {"M": m})
    assert launcher is (aligned if m % TILE_ROWS == 0 else ragged)
    assert _copy_on_gpu(launcher, m) == (True, True)


def test_gpu_a_launch_outside_the_contract_is_refused_before_it_reaches_the_gpu():
    _, ragged = _gpu_launchers()
    with pytest.raises(LaunchRefused) as refused:
        _copy_on_gpu(ragged, 4097)
    assert refused.value.reasons[0].startswith("M = 4097: needs M le 4096")


# --- imports ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first",
    ["rocke.helpers.tiling.descriptors", "rocke.helpers.tiling.tensor_transforms"],
)
def test_descriptors_and_tensor_transforms_import_in_either_order(first):
    # descriptors reads the walk's wave-uniform rule; tensor_transforms reads TensorDesc only for
    # type hints, so neither order can hit a half-initialised module.
    second = (
        "rocke.helpers.tiling.tensor_transforms"
        if first.endswith("descriptors")
        else "rocke.helpers.tiling.descriptors"
    )
    code = f"import {first}; import {second}; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr
