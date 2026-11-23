# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The buffer address space of TensorView refuses each unsupported call by name.

The C++ engine's tensor view refuses the same calls with the same error kind and
text; the native test rocke_tensor_view_buffer_errors pins it with the same rows.
The tensor_view_buffer parity family only sees that both engines reject these
calls; this pair of tests pins why.
"""

import re

import pytest

from rocke.core.ir import BF16, F16, F32, F64, I32, I64, IRBuilder, PtrType
from rocke.helpers.tensor_view import (
    make_buffer_resource,
    make_buffer_view,
    make_global_view,
)

_TYPES = {"f16": F16, "bf16": BF16, "f32": F32, "i32": I32, "i64": I64, "f64": F64}
_SHAPE = (4, 64)

# (op, dtype, n, exception, message). The rows match k_cases in
# tests/core/test_tensor_view_buffer_errors.cpp.
_CASES = [
    (
        "load_scalar",
        "bf16",
        1,
        NotImplementedError,
        "buffer scalar load not yet wired for dtype bf16 "
        "(only f16 is exposed by the IR builder today)",
    ),
    (
        "load_scalar",
        "f64",
        1,
        NotImplementedError,
        "buffer-space load/store not wired for dtype f64",
    ),
    (
        "store_scalar",
        "f32",
        1,
        NotImplementedError,
        "buffer scalar store not yet wired for dtype f32",
    ),
    (
        "load_vec",
        "f16",
        3,
        ValueError,
        "buffer load f16 supports n in {2, 4, 8} (got 3)",
    ),
    (
        "load_vec",
        "f32",
        4,
        NotImplementedError,
        "buffer vec load not yet wired for dtype f32",
    ),
    (
        "store_vec",
        "f16",
        16,
        ValueError,
        "buffer store f16 supports n in {2, 4, 8} (got 16)",
    ),
    (
        "store_vec",
        "bf16",
        4,
        NotImplementedError,
        "buffer vec store not yet wired for dtype bf16",
    ),
    (
        "load_vec_at",
        "f16",
        6,
        ValueError,
        "buffer load_vec_at f16 supports n in {2, 4, 8} (got 6)",
    ),
    (
        "load_vec_at",
        "i32",
        4,
        NotImplementedError,
        "buffer load_vec_at not wired for dtype i32",
    ),
    (
        "store_vec_at_mask",
        "f16",
        1,
        ValueError,
        "buffer store_vec_at f16 supports n in {2, 4, 8} (got 1)",
    ),
    (
        "store_vec_at",
        "f32",
        4,
        NotImplementedError,
        "buffer store_vec_at not wired for dtype f32",
    ),
    (
        "load_vec_at_mask_global",
        "f16",
        4,
        NotImplementedError,
        "mask= requires addr_space='buffer' (got 'global'); "
        "software masking for global/lds is not yet wired",
    ),
    (
        "store_vec_at_mask_global",
        "f16",
        4,
        NotImplementedError,
        "mask= requires addr_space='buffer' (got 'global')",
    ),
    (
        "load_vec_no_rsrc",
        "f16",
        4,
        TypeError,
        "buffer TensorView's base must be a BufferResource",
    ),
    (
        "buffer_of_global",
        "f16",
        1,
        TypeError,
        "TensorView.buffer requires addr_space='buffer', got 'global'",
    ),
]


def _issue(op: str, dtype_name: str, n: int) -> None:
    """Issue the case's call on a (4, 64) view of dtype: a buffer view over a
    BufferResource, a buffer view without one, or a global view."""
    dtype = _TYPES[dtype_name]
    b = IRBuilder("tensor_view_buffer_errors")
    x = b.param("X", PtrType(dtype, "global"))
    n_bytes = b.param("N_bytes", I32)
    rsrc = make_buffer_resource(b, x, num_bytes=n_bytes)
    if op in (
        "load_vec_at_mask_global",
        "store_vec_at_mask_global",
        "buffer_of_global",
    ):
        view = make_global_view(x, _SHAPE, dtype)
    elif op == "load_vec_no_rsrc":
        view = make_buffer_view(x, _SHAPE, dtype)
    else:
        view = make_buffer_view(rsrc, _SHAPE, dtype)
    tid = b.thread_id_x()
    row = b.const_i32(1)
    idx = [row, tid]
    value = b.const_i32(0)
    c_lim = b.const_i32(48)
    mask = b.cmp_lt(tid, c_lim)

    if op == "load_scalar":
        view.load_scalar(b, idx)
    elif op == "store_scalar":
        view.store_scalar(b, idx, value)
    elif op in ("load_vec", "load_vec_no_rsrc"):
        view.load_vec(b, idx, n)
    elif op == "store_vec":
        view.store_vec(b, idx, value, n)
    elif op == "load_vec_at":
        view.load_vec_at(b, tid, n)
    elif op == "store_vec_at":
        view.store_vec_at(b, tid, value, n)
    elif op == "load_vec_at_mask_global":
        view.load_vec_at(b, tid, n, mask=mask)
    elif op in ("store_vec_at_mask", "store_vec_at_mask_global"):
        view.store_vec_at(b, tid, value, n, mask=mask)
    elif op == "buffer_of_global":
        view.buffer
    else:
        raise ValueError(f"no error case op named {op!r}")


@pytest.mark.parametrize(
    "op, dtype_name, n, exception, message",
    _CASES,
    ids=[f"{op}-{dtype}-n{n}" for op, dtype, n, _, _ in _CASES],
)
def test_buffer_view_refuses_with_the_python_message(
    op, dtype_name, n, exception, message
):
    with pytest.raises(exception, match=f"^{re.escape(message)}$"):
        _issue(op, dtype_name, n)
