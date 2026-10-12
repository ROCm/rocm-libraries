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
from kernels.common.attention_bwd_run import WORKSPACE_ALIGN, _comgr_version, _flavor
from kernels.common.attention_bwd_unified import build_attn_bwd_unified_main
from kernels.common.attention_bwd_unified_plan import unified_stage_params

__all__ = [
    "WORKSPACE_ALIGN",
    "attn_bwd_unified_kernels",
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
