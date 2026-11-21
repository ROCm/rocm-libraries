# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare native-authored descriptor transport with the real Python verbs."""

from itertools import product
import os
from pathlib import Path
import subprocess

import pytest

from rocke.core.dtypes import dtype_info
from rocke.core.ir import F16, I32, IRBuilder, PtrType, dtype_to_ir_type
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.storage import BitPacking, FragmentPacking, MatrixFragmentLayout
from rocke.core.verify import verify
from rocke.core.ir_storage import storage_ir_type
from rocke.helpers.tiling import (
    load_fragment,
    make_matrix_fragment_desc,
    make_tensor_desc,
    make_window,
    store_fragment,
)

_FORMATS = ("fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1")
_CASES = []
for dtype, words, dynamic, clip, alignment in product(
    _FORMATS, (8, 16), range(3), (0, 1), (1, 16)
):
    bits = dtype_info(dtype).encoded_bits
    _CASES.append(
        (
            dtype,
            words,
            32 if words == 8 else 16,
            2,
            16 if bits == 8 else 32,
            words * 4,
            alignment,
            clip,
            dynamic,
            0,
        )
    )
for dtype, rows, dynamic, clip, alignment in product(
    ("tf32", "fp32"), (16, 32), range(3), (0, 1), (1, 4)
):
    _CASES.append((dtype, 2, rows, 64 // rows, 2, 2, alignment, clip, dynamic, 0))
for rows, clip in product((16, 32), (0, 1)):
    _CASES.append(("fp32", 2, rows, 64 // rows, 2, 2, 4, clip, 0, 1))


def build_case(case, *, batch=False, reverse=False):
    dtype, words, rows, groups, chunk, count, alignment, clip, dynamic, narrow = case
    logical = dtype_to_ir_type(dtype)
    output_dtype = F16 if narrow else logical
    layout = MatrixFragmentLayout(
        FragmentPacking(BitPacking(dtype_info(dtype).encoded_bits), count, 32, words),
        chunk,
        groups,
        rows,
    )
    tile = make_matrix_fragment_desc(layout)
    b = IRBuilder("fragment_io")
    ptr = b.param("A", PtrType(storage_ir_type(dtype), "global"))
    out = b.param("OUT", PtrType(storage_ir_type(output_dtype.name), "global"))
    k = count * groups
    lengths = (None, rows * 3, k * 3) if batch else (rows * 3, k * 3)
    stride = -1 if reverse else 1
    strides = (rows * 3 * k * 3, k * 3, stride) if batch else (k * 3, stride)
    tensor = make_tensor_desc(lengths, strides, logical)
    destination = make_tensor_desc(tensor.lengths, tensor.strides, output_dtype)
    batch_origin = (b.param("batch", I32),) if batch else ()
    origin = (*batch_origin, rows, b.param("origin_k", I32) if dynamic == 1 else k)
    bounds = ((None,) if batch else ()) + (
        rows * 2 - 1 if clip else None,
        b.param("bound_k", I32) if dynamic == 2 else k * 2 if clip else None,
    )
    lane = b.thread_id_x()
    aligned = bool(dynamic) and dtype_info(dtype).encoded_bits < 8
    fragment = load_fragment(
        b,
        ptr,
        make_window(tensor, origin, bounds),
        tile,
        lane,
        alignment_bytes=alignment,
        packed_runs_aligned=aligned,
    )
    store_fragment(
        b,
        out,
        make_window(destination, origin, bounds),
        fragment,
        lane,
        alignment_bytes=alignment,
        packed_runs_aligned=aligned,
    )
    return b.kernel


@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize("case", _CASES)
def test_python_descriptor_transport_verifies_and_lowers(case, batch):
    kernel = build_case(case, batch=batch)
    assert not verify(kernel)
    target = (
        "gfx942"
        if case[0] in ("tf32", "fp32")
        else "gfx950" if case[1] == 8 else "gfx1250"
    )
    assert "define" in lower_kernel_to_llvm(kernel, arch=target, llvm_flavor="llvm23")


def test_reversed_global_fragment_has_signed_storage_displacements():
    case = ("fp32", 2, 32, 2, 2, 2, 4, 0, 0, 0)
    kernel = build_case(case, reverse=True)
    assert not verify(kernel)
    assert not any(op.name == "tile.global_ptr_add" for op in kernel.body.ops)
    stores = [op for op in kernel.body.ops if op.name == "memref.global_store_typed"]
    assert len(stores) == 2
    offset = stores[1].operands[1].op.operands[1]
    assert offset.op.attrs["value"] == -1


@pytest.mark.parametrize(
    "case,batch,reverse",
    [(case, batch, False) for case in _CASES for batch in (False, True)]
    + [(("fp32", 2, 32, 2, 2, 2, 4, 0, 0, 0), False, True)],
)
def test_native_descriptor_transport_emits_identical_ir(case, batch, reverse):
    executable = os.environ.get("ROCKE_FRAGMENT_IO_TEST")
    if not executable:
        pytest.skip("set ROCKE_FRAGMENT_IO_TEST to the built native fragment emitter")
    assert Path(executable).is_file()
    actual = subprocess.run(
        [
            executable,
            "--emit-batch" if batch else "--emit-reversed" if reverse else "--emit",
            case[0],
            *map(str, case[1:]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert actual.stdout == serialize(build_case(case, batch=batch, reverse=reverse))
