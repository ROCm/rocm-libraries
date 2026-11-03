# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device harness of the unified (paged) attention backward (test helper).

``run_unified_case`` materialises a unified case, plans it in one page mode
with a test policy, uploads every buffer, runs the plan through
:func:`kernels.common.attention_bwd_unified_run.run_attn_bwd_unified` on a
caller-owned workspace and returns the gradients plus the integrity facts:

* gradient buffers start as a sentinel; every element outside the dQ view and
  every element of a page no live block references (spare and trap pages)
  must keep it, and so must a tail canary after each buffer and the stride
  gaps;
* the workspace is pre-poisoned with NaN (prep and page_zero must initialise
  every byte that is read) and followed by a NaN canary that must stay;
* inputs hold NaN in every slot that must never be read (see
  :mod:`.sdpa.bwd_unified_cases`).
"""

from __future__ import annotations

import ctypes
import dataclasses
from dataclasses import dataclass

import numpy as np

from kernels.common.attention_bwd import BWD_ARCHS
from kernels.common.attention_bwd_plan import attn_bwd_stage_vec
from kernels.common.attention_bwd_unified_plan import (
    FEATURE_GROUPS,
    AttnBwdUnifiedPolicy,
    AttnBwdUnifiedRequest,
    attn_bwd_unified_plan,
    attn_bwd_unified_tile_configs,
)
from kernels.common.attention_bwd_unified_run import (
    WORKSPACE_ALIGN,
    run_attn_bwd_unified,
)

from ._attention_bwd_harness import ARCH, CANARY, SENTINEL, dec, enc
from .sdpa.bwd_unified_cases import (
    UnifiedBwdCase,
    compare_unified,
    materialize_unified,
)

__all__ = [
    "ARCH",
    "UnifiedRun",
    "plan_unified_case",
    "request_from_unified",
    "run_unified_case",
    "unified_test_policy",
]

_KERNEL_CACHE: dict = {}
_INPUT_CACHE: dict = {}
_GRADS = ("dq", "dk_cache", "dv_cache")


def inputs_of(case: UnifiedBwdCase):
    if case.id not in _INPUT_CACHE:
        _INPUT_CACHE[case.id] = materialize_unified(case)
    return _INPUT_CACHE[case.id]


def request_from_unified(
    inp, *, injective: bool | None = None
) -> AttnBwdUnifiedRequest:
    """The unified request of a materialised case."""
    case = inp.case
    left, right, tl = case.band
    st = inp.strides
    strides = {n: st[n] for n in ("q", "o", "do", "dq")}
    for n, src in (
        ("k", "k_cache"),
        ("v", "v_cache"),
        ("dk", "dk_cache"),
        ("dv", "dv_cache"),
    ):
        strides[n] = st[src]
    return AttnBwdUnifiedRequest(
        num_seqs=len(case.q_lens),
        h_q=case.h_q,
        h_kv=case.h_kv,
        d_qk=case.d,
        d_v=case.d,
        total_q=sum(case.q_lens),
        max_seqlen_q=case.max_seqlen_q,
        max_seqlen_kv=case.max_seqlen_kv,
        page=case.page,
        num_blocks=inp.num_blocks,
        max_blocks_per_seq=inp.max_blocks_per_seq,
        dtype=case.dtype,
        strides=strides,
        bt_stride=inp.bt_stride,
        left_bound=left,
        right_bound=right,
        bottom_right=not tl,
        block_table_injective=case.injective if injective is None else injective,
        scale=inp.scale,
    )


def unified_test_policy(arch: str, knobs: dict | None = None) -> AttnBwdUnifiedPolicy:
    """Every group declared, every backward target validated, fixed knobs.

    ``knobs`` may hold ``"tile"``: the name of a shipped tile configuration
    (:func:`attn_bwd_unified_tile_configs`), expanded to its knobs.
    """
    knobs = dict(knobs or {})

    def tuning(req, arch_, page_class):
        out = dict(knobs)
        tile = out.pop("tile", None)
        if tile is not None:
            sv = attn_bwd_stage_vec(req)
            cfgs = attn_bwd_unified_tile_configs(arch_, req.d_qk, sv)
            if tile not in cfgs:
                raise ValueError(f"tile configuration {tile!r} not shipped on {arch_}")
            out.update(cfgs[tile][1])
        return out

    return AttnBwdUnifiedPolicy(
        declared_groups=frozenset(FEATURE_GROUPS),
        validated_archs=frozenset(BWD_ARCHS),
        tuning=tuning,
    )


def plan_unified_case(case, *, arch: str, mode: str, overrides=None):
    """The test plan of ``case`` in page mode ``mode`` (``"direct"`` /
    ``"atomic"``) on ``arch``."""
    inp = inputs_of(case)
    knobs = {**case.knob_dict(), **(overrides or {})}
    if mode == "atomic":
        knobs["page_mode"] = "atomic"
    elif not case.injective:
        raise ValueError("direct page mode needs an injective table")
    req = request_from_unified(inp)
    plan = attn_bwd_unified_plan(req, arch, policy=unified_test_policy(arch, knobs))
    if plan.runtime["page_mode"] != mode:
        raise AssertionError(f"planned {plan.runtime['page_mode']}, wanted {mode}")
    return plan


@dataclass
class UnifiedRun:
    case: object
    plan: object
    inputs: object
    dq: np.ndarray
    dk_cache: np.ndarray
    dv_cache: np.ndarray
    integrity: list
    problems: list


def _u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def _view(buf, dims, strides):
    item = buf.itemsize
    return np.lib.stride_tricks.as_strided(
        buf, shape=dims, strides=tuple(s * item for s in strides), writeable=True
    )


def drop_launch(plan, stage, index=0):
    """A copy of ``plan`` without the ``index``-th launch of ``stage``."""
    out, seen = [], 0
    for step in plan.launches:
        if step.stage == stage:
            seen += 1
            if seen - 1 == index:
                continue
        out.append(step)
    return dataclasses.replace(plan, launches=tuple(out))


def run_unified_case(
    case,
    *,
    arch: str,
    mode: str,
    overrides=None,
    mutate_plan=None,
    cache=None,
) -> UnifiedRun:
    """Plan, upload, launch, read back, check integrity and compare."""
    from rocke.runtime.hip_module import Runtime

    inp = inputs_of(case)
    plan = plan_unified_case(case, arch=arch, mode=mode, overrides=overrides)
    if mutate_plan is not None:
        plan = mutate_plan(plan)
    dt = case.dtype
    sentinel = enc(np.float32(SENTINEL), dt)[()]
    host = {}
    for name in ("q", "o", "do", "k_cache", "v_cache"):
        host[name] = enc(inp.buffers[name], dt)
    for name in _GRADS:
        host[name] = np.full(inp.sizes[name] + CANARY, sentinel)
    lse = inp.lse.copy()
    ws = np.full(plan.workspace_bytes // 4 + CANARY, np.nan, np.float32)
    tables = {
        "cu_seqlens_q": np.ascontiguousarray(inp.cu_seqlens_q, np.int32),
        "seq_lens_kv": np.ascontiguousarray(inp.seq_lens_kv, np.int32),
        "block_table": np.ascontiguousarray(inp.block_table, np.int32),
    }
    rt = Runtime()
    dev = {}
    try:
        uploads = (
            list(host.items()) + [("lse", lse), ("__ws", ws)] + list(tables.items())
        )
        for name, a in uploads:
            p = rt.alloc(max(int(a.nbytes), 16))
            rt.memcpy_h2d(p, _u8(a), a.nbytes)
            dev[name] = p
        if dev["__ws"] % WORKSPACE_ALIGN:
            raise AssertionError("device allocation is not 256-byte aligned")
        tensors = {k: v for k, v in dev.items() if k != "__ws"}
        run_attn_bwd_unified(
            plan,
            tensors=tensors,
            workspace=dev["__ws"],
            workspace_size=plan.workspace_bytes,
            cache=_KERNEL_CACHE if cache is None else cache,
        )
        rt.sync()
        for name in _GRADS:
            rt.memcpy_d2h(_u8(host[name]), dev[name], host[name].nbytes)
        rt.memcpy_d2h(_u8(ws), dev["__ws"], ws.nbytes)
    finally:
        for p in dev.values():
            rt.free(p)
    integrity = []
    grads = {}
    for name in _GRADS:
        raw = host[name]
        dims, strides = inp.dims[name], inp.strides[name]
        vals = _view(dec(raw[: inp.sizes[name]], dt), dims, strides).astype(np.float64)
        covered = np.zeros(inp.sizes[name], bool)
        mask_view = _view(covered, dims, strides)
        if name == "dq":
            mask_view[...] = True
        else:
            # every slot of a referenced page may be written; others never
            mask_view[...] = inp.referenced[:, None, None, None]
        body = raw[: inp.sizes[name]]
        if not np.all(body[~covered] == sentinel):
            bad = int(np.sum(body[~covered] != sentinel))
            where = (
                "outside the view" if name == "dq" else "in unreferenced pages or gaps"
            )
            integrity.append(f"{name}: {bad} elements {where} changed")
        if not np.all(raw[inp.sizes[name] :] == sentinel):
            integrity.append(f"{name}: tail canary overwritten")
        if name != "dq":
            ref_mask = np.broadcast_to(inp.referenced[:, None, None, None], vals.shape)
            stale = (
                np.asarray(_view(raw[: inp.sizes[name]], dims, strides))[ref_mask]
                == sentinel
            )
            if np.any(stale):
                integrity.append(
                    f"{name}: {int(np.sum(stale))} slots of referenced pages not written"
                )
        grads[name] = vals
    if not np.all(np.isnan(ws[plan.workspace_bytes // 4 :])):
        integrity.append("workspace canary overwritten")
    problems = compare_unified(inp, grads["dq"], grads["dk_cache"], grads["dv_cache"])
    return UnifiedRun(
        case=case,
        plan=plan,
        inputs=inp,
        dq=grads["dq"],
        dk_cache=grads["dk_cache"],
        dv_cache=grads["dv_cache"],
        integrity=integrity,
        problems=problems,
    )
