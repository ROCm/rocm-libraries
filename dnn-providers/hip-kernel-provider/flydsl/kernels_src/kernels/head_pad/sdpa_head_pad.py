# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Copy an SDPA operand's rows between its own layout and a head-dim-padded workspace.

The FlyDSL SDPA objects serve head dims that are multiples of 8. A graph whose Q/K or V/O
head dim is not runs on those objects over padded copies (the natives' head-dim padding,
``FlydslSdpaHeadPad.hpp``): Q, K and V are copied into a workspace whose rows are a multiple
of 8 elements wide and 16-byte aligned, with the columns past the head dim zeroed, the
attention launch reads and writes the copies, and O is copied back without its padding. No
attention object changes.

The kernel moves rows of 16-bit units, so one object serves f16, bf16 and f32 (an f32
element is two units, and the host passes widths and strides in units). For each row
``(b, h, s)`` of ``[B, H, S]`` and each unit ``c`` below ``dst_units``::

    dst[dst_row + c] = c < copy_units ? src[src_row + c] : 0

where a row starts at ``b * sb + s * ss + h * sh`` (``sb``, ``ss``, ``sh`` the operand's strides
in units). With ``ROW_OFF`` (an int32 ``[B + 1]`` token table, null for a dense operand) the
operand is packed (THD): row ``(b, h, s)`` is token ``ROW_OFF[b] + s`` at
``token * ss + h * sh``, the batch strides are not read, and a row with ``s`` at or past the
sequence's slot ``ROW_OFF[b + 1] - ROW_OFF[b]`` does not exist and is neither read nor
written. Padding a row is ``copy_units`` = the head dim and ``dst_units`` = the padded width
(the tail zeroed); unpadding is ``copy_units`` = ``dst_units`` = the head dim.

One thread per 8 destination units (a chunk): a grid of ``cdiv(S * cdiv(dst_units, 8),
BLOCK_THREADS)`` by ``B * H`` workgroups, each row of the grid one (batch, head) plane, so
those two must each fit an int32 (the native checks). A chunk is read as one 16-byte vector
when the source's rows are 16-byte aligned and the chunk lies inside ``copy_units`` (the padded
copy, when unpadding), and otherwise as eight 16-bit loads, each at its own address -- a unit
past ``copy_units`` reads the row's last copied unit instead and is replaced by 0 -- so a source
at any 2-byte alignment is read in place and nothing past a row's ``copy_units`` is read. It is
written as one 16-byte vector when the destination's rows are 16-byte aligned (the padded copy,
when padding) and unit by unit otherwise, each unit below ``dst_units`` once.

Arguments, in order: ``SRC``, ``DST``, ``ROW_OFF`` (pointers), ``batch``, ``heads``, ``seq_len``,
``copy_units``, ``dst_units`` (Int32), the source's ``(batch, head, seq)`` strides and the
destination's (Int64, in units), then the stream.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "sdpa_head_pad_kernel"


def _i32_at(ptr, index):
    """``ptr[index]`` as an i64, read as an int32 (a token table in global memory)."""
    i32_ptr = fx.recast_iter(
        fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=4,
        ),
        ptr,
    )
    return fx.Int64(fx.Int32(fx.ptr_load(i32_ptr + fx.Int64(index), fx.Int32)))


def _sequence(row_off, batch, seq_len):
    """``(first, rows)``: the batch's first token and its row count. A packed operand's come
    from its token table (a branch, so a dense launch loads nothing); a dense operand's are
    ``(0, seq_len)``, its rows addressed through the batch stride."""
    from flydsl._mlir.dialects import scf as _scf  # noqa: PLC0415

    packed = fx.Int64(fx.ptrtoint(row_off)) != fx.Int64(0)
    if_op = _scf.IfOp(fx.as_ir_value(packed), results_=[T.i64, T.i64], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        first = _i32_at(row_off, batch)
        last = _i32_at(row_off, batch + fx.Int64(1))
        _scf.YieldOp([fx.as_ir_value(first), fx.as_ir_value(last - first)])
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([fx.as_ir_value(fx.Int64(0)), fx.as_ir_value(fx.Int64(seq_len))])
    return fx.Int64(if_op.results[0]), fx.Int64(if_op.results[1]), packed


CHUNK = 8


def _units(ptr):
    return fx.recast_iter(fx.PointerType.get(fx.Int16.ir_type, ptr.address_space), ptr)


def _load_chunk(ptr, index, col, copy, vector):
    """The chunk's 8 units at ``ptr[index]``: one 16-byte load when ``vector``, else eight
    16-bit loads, a unit at or past ``copy`` read from the row's last copied unit and zeroed.
    """
    from flydsl._mlir.dialects import scf as _scf  # noqa: PLC0415

    units = _units(ptr)
    vec_ty = Vec.make_type(CHUNK, fx.Int16)
    if_op = _scf.IfOp(fx.as_ir_value(vector), results_=[vec_ty], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        view = fx.make_view(fx.add_offset(units, index), fx.make_layout(CHUNK, 1))
        _scf.YieldOp([fx.as_ir_value(Vec(view.load()))])
    with ir.InsertionPoint(if_op.else_block):
        values = []
        last = copy - fx.Int64(1) - col
        for j in range_constexpr(CHUNK):
            inside = fx.Int64(j) <= last
            offset = inside.select(fx.Int64(j), last)
            view = fx.make_view(
                fx.add_offset(units, index + offset), fx.make_layout(1, 1)
            )
            value = fx.Int16(Vec(view.load())[0])
            values.append(inside.select(value, fx.Int16(0)))
        _scf.YieldOp([fx.as_ir_value(Vec.from_elements(values, fx.Int16))])
    return Vec(if_op.results[0])


def _store_chunk(ptr, index, col, width, values, vector):
    """The chunk at ``ptr[index]``: one 16-byte store when ``vector``, else each unit below
    ``width`` on its own."""
    from flydsl._mlir.dialects import scf as _scf  # noqa: PLC0415

    units = _units(ptr)
    if_op = _scf.IfOp(fx.as_ir_value(vector), has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        view = fx.make_view(fx.add_offset(units, index), fx.make_layout(CHUNK, 1))
        view.store(values)
        _scf.YieldOp([])
    with ir.InsertionPoint(if_op.else_block):
        for j in range_constexpr(CHUNK):
            inside = col + fx.Int64(j) < width
            unit_if = _scf.IfOp(fx.as_ir_value(inside))
            with ir.InsertionPoint(unit_if.then_block):
                view = fx.make_view(
                    fx.add_offset(units, index + fx.Int64(j)), fx.make_layout(1, 1)
                )
                view.store(Vec.from_elements([values[j]], fx.Int16))
                _scf.YieldOp([])
        _scf.YieldOp([])


def build_sdpa_head_pad_module(block_threads=256):
    """Build the copy's launcher (``@flyc.jit``); its arguments are the module's."""
    BLOCK_THREADS = int(block_threads)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def sdpa_head_pad_kernel(
        SRC: fx.Pointer,
        DST: fx.Pointer,
        ROW_OFF: fx.Pointer,
        batch: fx.Int32,
        heads: fx.Int32,
        seq_len: fx.Int32,
        copy_units: fx.Int32,
        dst_units: fx.Int32,
        src_sb: fx.Int64,
        src_sh: fx.Int64,
        src_ss: fx.Int64,
        dst_sb: fx.Int64,
        dst_sh: fx.Int64,
        dst_ss: fx.Int64,
    ):
        # One (batch, head) plane per workgroup row of the grid, so its indices are the
        # workgroup's own (scalar); within it, one thread per (sequence row, 8-unit chunk),
        # divided by the chunks per row in 32 bits.
        plane = fx.Int32(gpu.block_idx.y)
        b32 = plane // fx.Int32(heads)
        b = fx.Int64(b32)
        h = fx.Int64(plane - b32 * fx.Int32(heads))
        chunks = (fx.Int32(dst_units) + fx.Int32(CHUNK - 1)) // fx.Int32(CHUNK)
        t = fx.Int32(gpu.block_idx.x) * fx.Int32(BLOCK_THREADS) + fx.Int32(
            gpu.thread_idx.x
        )
        if t < fx.Int32(seq_len) * chunks:
            s32 = t // chunks
            s = fx.Int64(s32)
            col = fx.Int64(t - s32 * chunks) * fx.Int64(CHUNK)
            first, rows, packed = _sequence(ROW_OFF, b, seq_len)
            if s < rows:
                token = first + s
                src_row = packed.select(token * src_ss, b * src_sb + s * src_ss)
                dst_row = packed.select(token * dst_ss, b * dst_sb + s * dst_ss)
                src_idx = fx.Int64(src_row) + h * src_sh + col
                dst_idx = fx.Int64(dst_row) + h * dst_sh + col
                copy = fx.Int64(copy_units)
                width = fx.Int64(dst_units)
                whole = col + fx.Int64(CHUNK)
                src_vec = whole <= copy
                dst_vec = whole <= width
                values = _load_chunk(SRC, src_idx, col, copy, src_vec)
                _store_chunk(DST, dst_idx, col, width, values, dst_vec)

    @flyc.jit
    def launch_sdpa_head_pad(
        SRC: fx.Pointer,
        DST: fx.Pointer,
        ROW_OFF: fx.Pointer,
        batch: fx.Int32,
        heads: fx.Int32,
        seq_len: fx.Int32,
        copy_units: fx.Int32,
        dst_units: fx.Int32,
        src_sb: fx.Int64,
        src_sh: fx.Int64,
        src_ss: fx.Int64,
        dst_sb: fx.Int64,
        dst_sh: fx.Int64,
        dst_ss: fx.Int64,
        stream: fx.Stream = fx.Stream(  # noqa: B008  framework idiom: evaluated once on purpose
            None
        ),
    ):
        chunks = (fx.Uint64(dst_units) + CHUNK - 1) // CHUNK
        grid_x = (fx.Uint64(seq_len) * chunks + BLOCK_THREADS - 1) // BLOCK_THREADS
        grid_y = fx.Uint64(batch) * fx.Uint64(heads)
        launcher = sdpa_head_pad_kernel(
            SRC,
            DST,
            ROW_OFF,
            batch,
            heads,
            seq_len,
            copy_units,
            dst_units,
            src_sb,
            src_sh,
            src_ss,
            dst_sb,
            dst_sh,
            dst_ss,
        )
        launcher.launch(
            grid=(grid_x, grid_y, 1), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    launch_sdpa_head_pad.block_size = BLOCK_THREADS
    return launch_sdpa_head_pad
