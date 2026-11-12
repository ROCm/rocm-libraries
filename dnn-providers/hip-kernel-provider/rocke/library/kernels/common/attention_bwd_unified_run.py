# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host launch glue of the unified attention backward.

Runs a unified plan (:func:`.attention_bwd_unified_plan.attn_bwd_unified_plan`)
on one stream in the plan's launch order (direct page mode ``prep -> main ->
convert(dQ)``; atomic page mode ``prep -> page_zero -> main -> convert(dQ) ->
page_convert(dK) -> page_convert(dV)``), with every kernel argument packed in
its stage's order (``rocke.attn_bwd_unified.v1`` for the main kernel and the
page roles, ``rocke.attn_bwd.v3`` for prep and convert).

The caller owns every buffer: the workspace (``plan.workspace_bytes``,
256-byte aligned base) and the tensors. Nothing is allocated and no memset is
issued: prep and page_zero initialise every workspace byte a later kernel
reads.

Host validation (:func:`check_attn_bwd_unified_inputs`). The run function
takes device pointers only and trusts them; the request's device-data
contract (``cu_seqlens_q`` non-decreasing in ``[0, total_q]`` with spans of at
most ``max_seqlen_q``, ``seq_lens_kv`` in ``[0, max_seqlen_kv]``, every live
block-table entry in ``[0, num_blocks)``, distinct live blocks under
``block_table_injective``) and the tensors' shapes, strides, dtypes and sizes
are checked on host-visible data before launch; a violation raises
``ValueError("<CODE>: detail")`` (codes of
:data:`kernels.common.attention_bwd_run.HOST_CHECK_CODES`), never a clamp. On
the device the defined behaviour for out-of-contract data is a clamp of the kv
length and a block id outside the cache never dereferenced (see
:mod:`kernels.common._attention_bwd_paged`).
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, MutableMapping
from typing import Any

from kernels.common.attention_bwd_aux import (
    build_attn_bwd_convert,
    build_attn_bwd_page_convert,
    build_attn_bwd_page_zero,
    build_attn_bwd_prep,
)
from kernels.common.attention_bwd_run import (
    ELEM_BYTES,
    WORKSPACE_ALIGN,
    TensorMeta,
    _comgr_version,
    _fail,
    _flavor,
    _meta,
    check_tensor,
    int_list,
)
from kernels.common.attention_bwd_unified import build_attn_bwd_unified_main
from kernels.common.attention_bwd_unified_plan import unified_stage_params

__all__ = [
    "WORKSPACE_ALIGN",
    "TensorMeta",
    "attn_bwd_unified_kernels",
    "check_attn_bwd_unified_inputs",
    "pack_attn_bwd_unified_args",
    "run_attn_bwd_unified",
]

_BUILDERS = {
    "main": build_attn_bwd_unified_main,
    "prep": build_attn_bwd_prep,
    "convert": build_attn_bwd_convert,
    "page_zero": build_attn_bwd_page_zero,
    "page_convert": build_attn_bwd_page_convert,
}

_PROCESS_CACHE: dict = {}


def attn_bwd_unified_kernels(plan, *, cache: MutableMapping | None = None) -> dict:
    """``{kernel name: (artifact, module, function)}`` for every plan launch.

    Each kernel is built from the plan's spec for its stage, compiled through
    comgr on the Python lowering backend and loaded once; ``cache`` (default:
    a process-wide dict) is keyed by ``(name, arch, flavor, comgr)``.
    """
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    cache = _PROCESS_CACHE if cache is None else cache
    flavor, comgr = _flavor(), _comgr_version()
    out = {}
    for step in plan.launches:
        key = (step.kernel, plan.arch, flavor, comgr)
        if key not in cache:
            spec = plan.specs[step.stage]
            kernel = _BUILDERS[step.stage](spec, arch=plan.arch)
            art = compile_kernel(
                kernel, arch=plan.arch, capture_ir_text=False, backend="python"
            )
            if art.kernel_name != step.kernel:
                raise AssertionError(
                    f"built kernel {art.kernel_name} differs from the plan's {step.kernel}"
                )
            mod = Runtime().load_module(art.hsaco)
            cache[key] = (art, mod, mod.get_function(art.kernel_name))
        out[step.kernel] = cache[key]
    return out


def pack_attn_bwd_unified_args(
    step,
    *,
    tensors: Mapping[str, int],
    workspace: int,
    workspace_layout: Mapping[str, tuple[int, int]],
) -> bytes:
    """Kernel-argument bytes of one launch step of a unified plan.

    Pointer roles resolve to ``workspace + offset`` for workspace sub-buffers,
    to ``tensors[role]`` for tensor roles and to 0 for unused (``None``)
    pointers; a tensor role missing from ``tensors`` raises ``ValueError``.
    """
    out = []
    for name, fmt in unified_stage_params(step.stage):
        if fmt == "Q":
            role = step.pointers[name]
            if role is None:
                ptr = 0
            elif role in workspace_layout:
                ptr = int(workspace) + int(workspace_layout[role][0])
            elif role in tensors and tensors[role] is not None:
                ptr = int(tensors[role])
            else:
                raise ValueError(f"{step.stage}: no device pointer for role {role!r}")
            out.append(struct.pack("<Q", ptr))
        elif fmt == "f":
            out.append(struct.pack("<f", float(step.scalars[name])))
        else:
            v = int(step.scalars[name])
            bits = 64 if fmt == "q" else 32
            if not -(1 << (bits - 1)) <= v < (1 << (bits - 1)):
                raise ValueError(
                    f"{step.stage}: kernarg {name}={v} does not fit i{bits}"
                )
            out.append(struct.pack("<" + fmt, v))
    return b"".join(out)


def run_attn_bwd_unified(
    plan,
    *,
    tensors: Mapping[str, int],
    workspace: int,
    workspace_size: int,
    stream: int = 0,
    cache: MutableMapping | None = None,
) -> list[Any]:
    """Launch every step of a unified ``plan`` on ``stream``.

    ``tensors`` maps the plan's pointer roles (``q o do lse dq k_cache v_cache
    dk_cache dv_cache cu_seqlens_q seq_lens_kv block_table``) to device
    addresses. ``workspace`` is the caller's workspace base
    (``WORKSPACE_ALIGN``-byte aligned) of ``workspace_size >=
    plan.workspace_bytes`` bytes. Returns the launched kernel names in order.

    The pointers are trusted: a caller holding the inputs validates them
    against the planned request first (:func:`check_attn_bwd_unified_inputs`).
    """
    from rocke.runtime.hip_module import Runtime

    if workspace_size < plan.workspace_bytes:
        raise ValueError(
            f"workspace of {workspace_size} bytes is smaller than the plan's "
            f"{plan.workspace_bytes}"
        )
    if plan.workspace_bytes and int(workspace) % WORKSPACE_ALIGN:
        raise ValueError(f"workspace base is not {WORKSPACE_ALIGN}-byte aligned")
    kernels = attn_bwd_unified_kernels(plan, cache=cache)
    packed = [
        pack_attn_bwd_unified_args(
            step,
            tensors=tensors,
            workspace=workspace,
            workspace_layout=plan.workspace_layout,
        )
        for step in plan.launches
    ]
    rt = Runtime()
    names = []
    for step, args in zip(plan.launches, packed):
        _art, _mod, fn = kernels[step.kernel]
        rt.launch(fn, tuple(step.grid), (int(step.block), 1, 1), args, stream=stream)
        names.append(step.kernel)
    return names


def check_attn_bwd_unified_inputs(
    req,
    *,
    cu_seqlens_q=None,
    seq_lens_kv=None,
    block_table=None,
    tensors: Mapping[str, TensorMeta] | None = None,
) -> None:
    """Check the host-visible inputs of a unified request (``ValueError``).

    * ``cu_seqlens_q`` (``num_seqs + 1`` ints): ``cu[0] >= 0``,
      non-decreasing, ``cu[-1] <= total_q``, every span at most
      ``max_seqlen_q`` (``RAGGED_RANGE``);
    * ``seq_lens_kv`` (``num_seqs`` ints): each in ``[0, max_seqlen_kv]``
      (``SEQ_LEN_RANGE``: a longer sequence is rejected, never clamped);
    * ``block_table`` (``[num_seqs, >= bt_stride]`` rows, or the flat device
      layout of ``num_seqs`` rows of ``bt_stride`` ints; needs
      ``seq_lens_kv``): every entry of a live block ``j < ceil(len_kv /
      page)`` in ``[0, num_blocks)`` (``BLOCK_TABLE_RANGE``) and, under
      ``block_table_injective``, no physical block used twice
      (``BLOCK_TABLE_NOT_INJECTIVE``); entries past the live blocks are never
      read and may hold anything;
    * ``tensors`` (:class:`TensorMeta` by plan role: ``q o do dq lse
      k_cache v_cache dk_cache dv_cache cu_seqlens_q seq_lens_kv
      block_table``): shapes, strides, dtypes and sizes against the request.

    Arguments left ``None`` are not checked.
    """
    import numpy as np

    n = req.num_seqs
    if cu_seqlens_q is not None:
        cu = int_list("cu_seqlens_q", cu_seqlens_q, "RAGGED_RANGE")
        if len(cu) != n + 1:
            _fail("RAGGED_RANGE", f"cu_seqlens_q: {len(cu)} entries, want {n + 1}")
        if cu[0] < 0:
            _fail("RAGGED_RANGE", f"cu_seqlens_q[0] = {cu[0]} is negative")
        for i in range(n):
            if cu[i + 1] < cu[i]:
                _fail("RAGGED_RANGE", f"cu_seqlens_q decreases at {i + 1}: "
                      f"{cu[i]} -> {cu[i + 1]}")  # fmt: skip
            if cu[i + 1] - cu[i] > req.max_seqlen_q:
                _fail("RAGGED_RANGE", f"cu_seqlens_q: sequence {i} spans "
                      f"{cu[i + 1] - cu[i]} > max_seqlen_q {req.max_seqlen_q}")  # fmt: skip
        if cu[-1] > req.total_q:
            _fail(
                "RAGGED_RANGE", f"cu_seqlens_q ends at {cu[-1]} > total_q {req.total_q}"
            )
    lens = None
    if seq_lens_kv is not None:
        lens = int_list("seq_lens_kv", seq_lens_kv, "SEQ_LEN_RANGE")
        if len(lens) != n:
            _fail("SEQ_LEN_RANGE", f"seq_lens_kv: {len(lens)} values, want {n}")
        for i, v in enumerate(lens):
            if not 0 <= v <= req.max_seqlen_kv:
                _fail("SEQ_LEN_RANGE", f"seq_lens_kv[{i}] = {v} outside [0, "
                      f"max_seqlen_kv {req.max_seqlen_kv}]")  # fmt: skip
    if block_table is not None:
        if lens is None:
            raise TypeError("checking block_table needs seq_lens_kv (its live blocks)")
        bt = np.asarray(block_table)
        if bt.ndim == 1:
            if bt.size < (n - 1) * req.table_stride + req.max_blocks_per_seq:
                _fail("BLOCK_TABLE_RANGE", f"flat block_table of {bt.size} entries "
                      f"is shorter than {n} rows at stride {req.table_stride}")  # fmt: skip
            rows = [bt[s * req.table_stride : s * req.table_stride + req.max_blocks_per_seq]
                    for s in range(n)]  # fmt: skip
        elif (
            bt.ndim == 2 and bt.shape[0] >= n and bt.shape[1] >= req.max_blocks_per_seq
        ):
            rows = [bt[s] for s in range(n)]
        else:
            _fail("BLOCK_TABLE_RANGE", f"block_table shape {bt.shape}, want "
                  f"[{n}, >= {req.max_blocks_per_seq}] or flat")  # fmt: skip
        used: dict[int, tuple[int, int]] = {}
        for s in range(n):
            live = -(-lens[s] // req.page)
            if live > req.max_blocks_per_seq:
                _fail("BLOCK_TABLE_RANGE", f"sequence {s} needs {live} blocks > "
                      f"max_blocks_per_seq {req.max_blocks_per_seq}")  # fmt: skip
            for j, v in enumerate(
                int_list("block_table", rows[s][:live], "BLOCK_TABLE_RANGE")
            ):
                if not 0 <= v < req.num_blocks:
                    _fail("BLOCK_TABLE_RANGE", f"block_table[{s}][{j}] = {v} outside "
                          f"[0, num_blocks {req.num_blocks})")  # fmt: skip
                if req.block_table_injective and v in used:
                    _fail("BLOCK_TABLE_NOT_INJECTIVE", f"block {v} is used by "
                          f"{used[v]} and ({s}, {j})")  # fmt: skip
                used[v] = (s, j)
    if tensors is not None:
        _check_unified_tensors(req, tensors)


def _check_unified_tensors(req, tensors: Mapping[str, TensorMeta]) -> None:
    from kernels.common.attention_bwd_unified_plan import _lse_layout

    d = req.d_qk
    dts = {"q": req.dtype, "o": req.o_dtype, "do": req.do_dtype, "dq": req.dq_dtype}
    for name, dt in dts.items():
        check_tensor(name, _meta(tensors, name), dtype=req.dtype if dt is None else dt,
                     shape=(req.total_q, req.h_q, d), strides=req.tensor_strides(name))  # fmt: skip
    caches = (("k_cache", "k", req.k_dtype), ("v_cache", "v", req.v_dtype),
              ("dk_cache", "dk", req.dk_dtype), ("dv_cache", "dv", req.dv_dtype))  # fmt: skip
    for role, name, dt in caches:
        check_tensor(role, _meta(tensors, role), dtype=req.dtype if dt is None else dt,
                     shape=tuple(req.tensor_dims(name)), strides=req.tensor_strides(name))  # fmt: skip
    _why, (lt, lh) = _lse_layout(req)
    m = _meta(tensors, "lse")
    rank = len(m.shape)
    shape = (req.total_q, req.h_q) + ((1,) if rank == 3 else ())
    check_tensor("lse", m, dtype="fp32", shape=shape,
                 strides=(lt, lh) + ((0,) if rank == 3 else ()))  # fmt: skip
    ints = (("cu_seqlens_q", req.num_seqs + 1, req.cu_seqlens_dtype),
            ("seq_lens_kv", req.num_seqs, req.seq_lens_dtype))  # fmt: skip
    for role, count, dt in ints:
        m = _meta(tensors, role)
        if m.dtype != dt:
            _fail("TENSOR_DTYPE", f"{role}: dtype {m.dtype}, want {dt}")
        if count * ELEM_BYTES[dt] > m.nbytes:
            _fail("TENSOR_EXTENT", f"{role}: needs {count} entries, holds "
                  f"{m.nbytes // ELEM_BYTES[dt]}")  # fmt: skip
    m = _meta(tensors, "block_table")
    if m.dtype != req.block_table_dtype:
        _fail(
            "TENSOR_DTYPE",
            f"block_table: dtype {m.dtype}, want {req.block_table_dtype}",
        )
    need = ((req.num_seqs - 1) * req.table_stride + req.max_blocks_per_seq) * 4
    if need > m.nbytes:
        _fail("TENSOR_EXTENT", f"block_table: needs {need} bytes, holds {m.nbytes}")
