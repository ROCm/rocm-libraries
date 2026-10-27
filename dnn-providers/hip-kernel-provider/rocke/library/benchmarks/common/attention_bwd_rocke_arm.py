# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocKE arm of the backward timing harness (``attention_bwd_bench.py``).

A rocKE arm is one configuration of the layout-general backward: spec knob
overrides plus runtime plan values, labelled. For a census cell it

1. builds the capability request from the very tensors the competitor arms
   use (Q, K, V, dO), with O (cast to the I/O dtype) and the natural-log LSE
   of the fixed fp32 reference forward of the cell, and the device hints
   ``num_cus`` / ``num_xcds`` (:func:`request_hints`);
2. plans it with a policy that declares every functional group, waives the
   (not yet measured) performance status and returns the configuration's
   knobs through the tuning hook, so the plan resolves exactly the spec that
   ships with those knobs (``runtime["route"] = "dense"`` plans the dense
   family instead, with dense and body knobs in ``knobs``);
3. takes every kernel of the plan from the on-disk compile cache (compiled on
   a miss; with ``load_only``, as in a timing attempt, a miss is an error and
   nothing is compiled), loads the cached code objects, and packs every
   launch's kernel arguments once (host planning is outside the timed region);
4. runs ``prep -> main -> convert(s)`` on torch's current stream into
   caller-owned dQ / dK / dV and a caller-owned workspace, each followed by a
   canary region that must stay intact.

``RockeConfig`` is the JSON-able description shared with the sweep driver.
Nothing here prints.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

CANARY_ELEMS = 256  # gradient elements after every output buffer
CANARY_BYTES = 1024  # bytes after the workspace
_CANARY_BYTE = 0xA5
_SENTINEL = -7.0
SHIPPED_LABEL = "shipped"
SPEC_FIXED_FIELDS = ("head_size", "dtype", "seq_mode", "dkv_mode", "stage_vec")
# Runtime entries of a configuration that are device hints of the request
# (``AttnBwdRequest`` fields), not plan tuning values: the XCD remap values
# (``xcd_chunk``) need the request's XCD count.
REQUEST_HINTS = ("num_xcds", "num_cus")
# A ``num_cus`` hint of this value is the CU count of the device the arm runs
# on (mirror pairing needs a grid that overfills the device).
DEVICE_NUM_CUS = "device"
# XCDs per device as the runtime exposes a CDNA target (whole-device
# partition; rocke.helpers.grid NUM_XCDS_*): the default ``num_xcds`` hint.
DEVICE_NUM_XCDS = MappingProxyType({"gfx942": 8, "gfx950": 8})


def _read_num_cus(device_index: int) -> int | None:
    from rocke.runtime.hip_module import get_device_num_cus

    return get_device_num_cus(int(device_index))


def request_hints(
    runtime: Mapping[str, Any], arch: str, device_index: int = 0, read_num_cus=None
) -> dict:
    """Device hints of an arm's request (``AttnBwdRequest`` fields).

    A configuration's own ``num_cus`` / ``num_xcds`` win; ``num_cus =
    DEVICE_NUM_CUS`` is the device's CU count and fails (``Unsupported``) when
    it cannot be read. Without a value the arm passes the device's CU count
    (when readable) and the XCD count of ``arch`` (:data:`DEVICE_NUM_XCDS`),
    so mirror pairing and the XCD remap plan as on the device. ``None`` keeps
    a hint absent. Neither hint changes a plan that does not ask for pairing,
    the remap or a persistent grid.
    """
    read = _read_num_cus if read_num_cus is None else read_num_cus
    hints = {k: runtime[k] for k in REQUEST_HINTS if k in runtime}
    if hints.get("num_cus") == DEVICE_NUM_CUS:
        n = read(device_index)
        if not n:
            raise Unsupported("the device CU count (num_cus hint) is not readable")
        hints["num_cus"] = int(n)
    elif "num_cus" not in hints:
        n = read(device_index)
        hints["num_cus"] = int(n) if n else None
    if "num_xcds" not in hints:
        hints["num_xcds"] = DEVICE_NUM_XCDS.get(arch)
    return {k: v for k, v in hints.items() if v is not None}


class Unsupported(Exception):
    """The configuration cannot serve the cell (declined or not plannable)."""


@dataclass(frozen=True)
class RockeConfig:
    """A labelled rocKE configuration: spec knobs and runtime plan values."""

    label: str
    knobs: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "knobs": _jsonable(dict(self.knobs)),
            "runtime": _jsonable(dict(self.runtime)),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RockeConfig":
        knobs = dict(d.get("knobs") or {})
        for k, v in list(knobs.items()):
            if isinstance(v, list):
                knobs[k] = tuple(v)
        return cls(str(d["label"]), knobs, dict(d.get("runtime") or {}))


SHIPPED = RockeConfig(SHIPPED_LABEL)


def _jsonable(x):
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def load_configs(path: str | Path) -> list[RockeConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["configs"] if isinstance(data, dict) else data
    out = [RockeConfig.from_dict(d) for d in items]
    labels = [c.label for c in out]
    if len(set(labels)) != len(labels):
        raise ValueError("rocKE configuration labels must be unique")
    return out


def write_configs(path: str | Path, configs: Sequence[RockeConfig]) -> None:
    Path(path).write_text(
        json.dumps({"configs": [c.to_dict() for c in configs]}, sort_keys=True),
        encoding="utf-8",
    )


def config_policy(cfg: RockeConfig):
    """Every functional group declared, unverified waived, ``cfg`` as tuning."""
    from kernels.common.attention_bwd_plan import FEATURE_GROUPS, AttnBwdPolicy

    runtime = {k: v for k, v in cfg.runtime.items() if k not in REQUEST_HINTS}
    values = {**dict(cfg.knobs), **runtime}

    def tuning(req, arch, route, perf_class):
        return dict(values)

    return AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS),
        waive_unverified=True,
        tuning=tuning,
    )


# --------------------------------------------------------------------------- request
def _alignment(ptrs: Sequence[int]) -> int:
    out = 256
    for p in ptrs:
        p = int(p)
        if p % 256:
            out = min(out, p & -p)
    return out


def _strides4(t, thd: bool) -> tuple[int, int, int, int]:
    if thd:  # [T, H, D] -> (ignored, H, token, D)
        st, sh, sd = t.stride()
        return (0, sh, st, sd)
    return tuple(int(x) for x in t.stride())


def band_fields(cell) -> dict:
    """Request band fields of a census mask (``swa``: bottom-right window)."""
    if cell.mask == "none":
        return {}
    if cell.mask == "causal_tl":
        return {"causal": True}
    if cell.mask == "causal_br":
        return {"causal_bottom_right": True}
    if cell.mask == "swa":
        # key j attends query i iff 0 <= (i + s_kv - s_q) - j < window
        return {
            "left_bound": int(cell.window) - 1,
            "right_bound": 0,
            "bottom_right": True,
        }
    raise ValueError(f"unknown mask {cell.mask!r}")


def request_for_cell(cell, tensors: Mapping[str, Any], *, scale: float):
    """``AttnBwdRequest`` of ``cell`` over the given device tensors."""
    from kernels.common.attention_bwd_plan import AttnBwdRequest

    thd = cell.is_thd
    strides = {
        n: _strides4(tensors[n], thd)
        for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv")
    }
    kw: dict[str, Any] = {}
    if thd:
        b = len(cell.seqlens)
        s_q, s_kv = max(cell.seqlens), max(cell.kv_lengths)
        kw.update(
            layout="thd",
            ragged_offset_dtype=str(tensors["offsets_q"].dtype).replace("torch.", ""),
            ragged_offset_multiplier=1,
        )
    else:
        b, s_q, s_kv = cell.b, cell.sq, cell.skv
    if cell.padded_q is not None or cell.padded_kv is not None:
        kw.update(padding=True, has_seq_len_q=True, has_seq_len_kv=True)
    ptrs = [tensors[n].data_ptr() for n in ("q", "k", "v", "o", "do", "dq", "dk", "dv")]
    return AttnBwdRequest(
        b=b,
        h_q=cell.hq,
        h_k=cell.hkv,
        h_v=cell.h_v,
        s_q=s_q,
        s_kv=s_kv,
        d_qk=cell.d,
        d_v=cell.d,
        dtype=cell.dtype,
        strides=strides,
        tensor_alignment=_alignment(ptrs),
        scale=float(scale),
        **band_fields(cell),
        **kw,
    )


# --------------------------------------------------------------------------- arm
def _dense_like(torch, t, extra: int):
    """A zero-gap tensor with ``t``'s shape and dim order, ``extra`` elements
    after it in the same allocation: ``(flat buffer, view)``."""
    order = sorted(range(t.dim()), key=lambda i: (-t.stride(i), -i))
    strides = [0] * t.dim()
    acc = 1
    for i in reversed(order):
        strides[i] = acc
        acc *= t.shape[i]
    flat = torch.empty(acc + extra, dtype=t.dtype, device=t.device)
    return flat, flat.as_strided(tuple(t.shape), tuple(strides))


class RockeArm:
    """One rocKE configuration bound to one cell's tensors (see module doc)."""

    def __init__(
        self,
        cell,
        problem,
        ref,
        *,
        arch: str,
        config: RockeConfig,
        cache=None,
        load_only: bool = False,
    ) -> None:
        import torch

        from kernels.common.attention_bwd_plan import attn_bwd_plan
        from kernels.common.attention_bwd_run import (
            WORKSPACE_ALIGN,
            attn_bwd_kernels,
            pack_attn_bwd_args,
        )

        self.cell, self.config, self.arch = cell, config, arch
        dt = problem.q.dtype
        q, k, v, do = (
            t.detach() for t in (problem.q, problem.k, problem.v, problem.do)
        )
        # O in Q's dimension order (as the forward leaves it), so a dense
        # BSHD / BHSD cell keeps one layout across Q, O and dO
        _o_flat, o = _dense_like(torch, q, 0)
        o.copy_(ref.o.to(dt))
        lse = ref.lse.to(torch.float32).contiguous()
        self._flat = {}
        outs = {}
        for name, src in (("dq", q), ("dk", k), ("dv", v)):
            flat, view = _dense_like(torch, src, CANARY_ELEMS)
            flat.fill_(_SENTINEL)
            self._flat[name] = flat
            outs[name] = view
        tensors = {"q": q, "k": k, "v": v, "o": o, "do": do, "lse": lse, **outs}
        if cell.is_thd:
            off_dt = torch.int32
            if (
                int(problem.cu_q[-1]) * q.stride(0) >= 2**31
                or int(problem.cu_kv[-1]) * k.stride(0) >= 2**31
            ):
                off_dt = torch.int64
            tensors["offsets_q"] = (problem.cu_q.to(torch.int64) * q.stride(0)).to(
                off_dt
            )
            tensors["offsets_kv"] = (problem.cu_kv.to(torch.int64) * k.stride(0)).to(
                off_dt
            )
        if cell.padded_q is not None or cell.padded_kv is not None:
            lq = cell.padded_q or (cell.sq,) * cell.b
            lk = cell.padded_kv or (cell.skv,) * cell.b
            tensors["seq_len_q"] = torch.tensor(lq, dtype=torch.int32, device=q.device)
            tensors["seq_len_kv"] = torch.tensor(lk, dtype=torch.int32, device=q.device)
        req = request_for_cell(cell, tensors, scale=problem.scale)
        hints = request_hints(config.runtime, arch, int(q.device.index or 0))
        if hints:
            req = replace(req, **hints)
        # runtime "route": "dense" plans the dense family (its knobs in
        # ``knobs``); every other configuration plans the general family.
        runtime = dict(config.runtime)
        route = runtime.pop("route", "general")
        policy = config_policy(RockeConfig(config.label, config.knobs, runtime))
        try:
            if route == "dense":
                from kernels.common.attention_dense_bwd_plan import (
                    attn_dense_bwd_plan,
                )

                plan = attn_dense_bwd_plan(req, arch, policy=policy)
            elif route == "general":
                plan = attn_bwd_plan(req, arch, policy=policy)
            else:
                raise ValueError(f"route={route!r} not in ('general', 'dense')")
        except ValueError as ex:
            raise Unsupported(str(ex)[:300]) from ex
        self.plan = plan
        self.request = req
        ws_bytes = plan.workspace_bytes
        self._ws = torch.empty(
            ws_bytes + CANARY_BYTES + WORKSPACE_ALIGN,
            dtype=torch.uint8,
            device=q.device,
        )
        base = self._ws.data_ptr()
        pad = (-base) % WORKSPACE_ALIGN
        self._ws_off = pad
        self._ws[pad + ws_bytes :].fill_(_CANARY_BYTE)
        self._ws_len = ws_bytes
        from kernels.common import attention_bwd_run

        self.entries = {}
        if cache is not None:
            from benchmarks.common.attention_bwd_compile import (
                plan_entries,
                run_cache_for,
            )

            # load_only: a timing attempt reads the code objects its job's
            # reference process compiled and never compiles (cache_miss)
            self.entries = plan_entries(plan, cache, compile_missing=not load_only)
            self.run_cache = run_cache_for(plan, self.entries, cache.toolchain)
        elif load_only:
            raise RuntimeError("cache_miss: load_only needs a compile cache")
        else:  # compiled in this process (no disk cache)
            self.run_cache = attention_bwd_run._PROCESS_CACHE
        kernels = attn_bwd_kernels(plan, cache=self.run_cache)
        ptrs = {n: int(t.data_ptr()) for n, t in tensors.items()}
        ws_ptr = base + pad
        self._launches = []
        for step in plan.launches:
            args = pack_attn_bwd_args(
                step,
                tensors=ptrs,
                workspace=ws_ptr,
                workspace_layout=plan.workspace_layout,
            )
            fn = kernels[step.kernel][2]
            self._launches.append((fn, tuple(step.grid), (int(step.block), 1, 1), args))
        self._keep = tensors  # every launch argument stays alive with the arm
        self.grads = (outs["dq"], outs["dk"], outs["dv"])
        from rocke.runtime.hip_module import Runtime

        self._rt = Runtime()
        self._torch = torch

    @property
    def main_spec(self):
        specs = self.plan.specs
        return specs["main"] if "main" in specs else specs["dense"]

    def kernel_record(self) -> dict:
        """Kernel names, cache keys and code-object digests of the plan."""
        return {
            "kernels": [s.kernel for s in self.plan.launches],
            "hsaco_sha": {n: e.hsaco_sha for n, e in self.entries.items()},
            "cache_keys": {n: e.key for n, e in self.entries.items()},
            "main_spec": repr(self.main_spec),
            "runtime": dict(self.plan.runtime),
            "workspace_bytes": self.plan.workspace_bytes,
            # performance class + kv-tile and grid-fill buckets of the request
            "tuning_key": list(self.plan.tuning_key),
            # device hints the request carried (num_cus / num_xcds)
            "request_hints": {k: getattr(self.request, k) for k in REQUEST_HINTS},
        }

    def run(self):
        stream = int(self._torch.cuda.current_stream().cuda_stream)
        for fn, grid, block, args in self._launches:
            self._rt.launch(fn, grid, block, args, stream=stream)
        return self.grads

    def drain(self) -> None:
        """Release the runtime's per-launch argument buffers (after a sync)."""
        self._rt.wait_stream(int(self._torch.cuda.current_stream().cuda_stream))

    def canary_problems(self) -> list[str]:
        """Canary regions that changed (empty when intact)."""
        torch = self._torch
        torch.cuda.synchronize()
        out = []
        for name, flat in self._flat.items():
            tail = flat[-CANARY_ELEMS:].float()
            if not bool(torch.all(tail == _SENTINEL)):
                out.append(f"{name}: tail canary overwritten")
        start = self._ws_off + self._ws_len
        if not bool(torch.all(self._ws[start : start + CANARY_BYTES] == _CANARY_BYTE)):
            out.append("workspace canary overwritten")
        return out


def scale_of(cell) -> float:
    return 1.0 / math.sqrt(cell.d)
