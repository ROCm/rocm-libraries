# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate a real gfx1250 TDM transfer built by ``helpers.tdm``.

``tdm_verify.py`` proves the intrinsics lower and assemble, but launches
nothing: its descriptor groups are all zero. This probe closes that gap by
building an actual descriptor with :func:`rocke.helpers.tdm_row_major_2d`,
moving a tile global->LDS on device, and comparing the LDS image against NumPy.

Two properties are checked, both of which a bit-level review cannot settle:

``tile``
    A tile strictly smaller than the tensor lands in LDS row-major and
    contiguously, which pins down the LDS destination layout and proves the
    stride fields are interpreted as expected.

``oob``
    Reading past ``tensor_dim`` on the positive side returns zero rather than
    garbage or a fault -- the property that lets GEMM tails skip predication.

A single wave issues the transfer, so no cross-wave barrier is involved and a
failure points at the descriptor rather than at synchronization.
"""

from __future__ import annotations

import numpy as np

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, PtrType
from rocke.helpers.tdm import tdm_row_major_2d

try:
    from .common import (
        DeviceArena,
        Reporter,
        Runtime,
        ValidatedArtifact,
        launch,
        make_parser,
        record_compile_check,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        DeviceArena,
        Reporter,
        Runtime,
        ValidatedArtifact,
        launch,
        make_parser,
        record_compile_check,
    )

_THREADS = 32

# Global tensor is wider than the tile, so the row stride is exercised rather
# than degenerating into a flat copy.
_TENSOR_ROWS = 4
_TENSOR_COLS = 16
_TILE_ROWS = 4
_TILE_COLS = 8
_TILE_ELEMS = _TILE_ROWS * _TILE_COLS  # == _THREADS: one element per lane

# For the OOB case the descriptor declares a narrower tensor than the tile
# covers; columns at and beyond this must read back as zero.
_OOB_COLS = 6

_ELEM_BYTES = 4  # i32 -> data_size code 2


def _fence_lds(builder: IRBuilder) -> None:
    """Drain LDS traffic and block compiler reordering across the transfer."""
    builder.inline_asm("s_wait_dscnt 0", "~{memory}", sideeffect=True, convergent=True)


def build_tdm_kernel(*, tensor_cols: int, name: str) -> KernelDef:
    """Build a one-wave global->LDS TDM transfer that echoes LDS to memory.

    ``tensor_cols`` is what the descriptor *claims* the tensor width is. Passing
    a value below ``_TILE_COLS`` drives the out-of-bounds path; the underlying
    allocation is always the full ``_TENSOR_COLS`` wide.
    """
    builder = IRBuilder(name)
    builder.kernel.attrs["max_workgroup_size"] = _THREADS

    # The global base arrives as an integer: rocke exposes no ptrtoint, and the
    # descriptor needs the raw 64-bit address.
    source_address = builder.param("source_address", I64)
    output = builder.param(
        "output", PtrType(I32, "global"), writeonly=True, noalias=True, align=16
    )
    shared = builder.smem_alloc(I32, [_TILE_ELEMS], name_hint="tdm_tile")

    groups = tdm_row_major_2d(
        builder,
        global_addr=source_address,
        lds_addr=builder.smem_addr_of(shared),
        rows=_TENSOR_ROWS,
        cols=tensor_cols,
        row_pitch=_TENSOR_COLS,  # true allocation pitch
        tile_rows=_TILE_ROWS,
        tile_cols=_TILE_COLS,
        elem_bytes=_ELEM_BYTES,
    )
    builder.tensor_load_to_lds(*groups)
    builder.s_wait_tensorcnt(0)
    _fence_lds(builder)

    lane = builder.thread_id_x()
    value = builder.vec_extract(
        builder.smem_load_vN(shared, lane, dtype=I32, n=1), 0
    )
    builder.global_store(output, lane, value, align=4)
    builder.ret()
    return builder.kernel


def _expected_tile(tensor: np.ndarray, tensor_cols: int) -> np.ndarray:
    """Reference LDS image: the tile, with past-the-end columns zeroed."""
    tile = np.zeros((_TILE_ROWS, _TILE_COLS), dtype=np.int32)
    visible = min(tensor_cols, _TILE_COLS)
    tile[:, :visible] = tensor[:_TILE_ROWS, :visible]
    return tile.reshape(-1)


def _run_functional(
    validated: ValidatedArtifact, *, tensor_cols: int, label: str
) -> tuple[bool, str]:
    tensor = np.arange(_TENSOR_ROWS * _TENSOR_COLS, dtype=np.int32).reshape(
        _TENSOR_ROWS, _TENSOR_COLS
    )
    expected = _expected_tile(tensor, tensor_cols)

    runtime = Runtime()
    with DeviceArena(runtime) as device:
        source_dev = device.input(tensor)
        output_dev = device.output(expected.nbytes, fill=0xFF)
        launch(
            runtime,
            validated,
            grid=(1, 1, 1),
            block=(_THREADS, 1, 1),
            pack_format="<QQ",
            pack_values=(source_dev, output_dev),
        )
        actual = device.read(
            output_dev, dtype=np.dtype(np.int32), shape=expected.shape
        )

    mismatch = int(np.count_nonzero(actual != expected))
    detail = f"{label}, mismatches={mismatch}/{expected.size}"
    if mismatch:
        bad = np.flatnonzero(actual != expected)[:4]
        detail += "; first=" + ", ".join(
            f"[{i}] got {actual[i]} want {expected[i]}" for i in bad
        )
    return mismatch == 0, detail


def _check(
    reporter: Reporter, args: object, name: str, *, tensor_cols: int, label: str
) -> None:
    validated = record_compile_check(
        reporter,
        f"{name}.compile",
        build_tdm_kernel(tensor_cols=tensor_cols, name=f"gfx1250_tdm_{name}"),
        arch=args.arch,  # type: ignore[attr-defined]
        llvm_required=("call void @llvm.amdgcn.tensor.load.to.lds(",),
        isa_required=(r"\btensor_load_to_lds\b", r"\bs_wait_tensorcnt\b"),
    )
    if validated is None:
        reporter.skipped(f"{name}.functional", "compile validation failed")
        return
    if args.compile_only:  # type: ignore[attr-defined]
        reporter.skipped(f"{name}.functional", "--compile-only requested")
        return
    try:
        ok, detail = _run_functional(validated, tensor_cols=tensor_cols, label=label)
    except Exception as exc:  # noqa: BLE001 - verifier must report runtime failures
        reporter.failed(f"{name}.functional", f"{type(exc).__name__}: {exc}")
        return
    (reporter.passed if ok else reporter.failed)(f"{name}.functional", detail)


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    _check(
        reporter,
        args,
        "tile",
        tensor_cols=_TENSOR_COLS,
        label=f"{_TILE_ROWS}x{_TILE_COLS} tile from {_TENSOR_ROWS}x{_TENSOR_COLS}",
    )
    _check(
        reporter,
        args,
        "oob",
        tensor_cols=_OOB_COLS,
        label=f"columns >= {_OOB_COLS} must read zero",
    )
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
