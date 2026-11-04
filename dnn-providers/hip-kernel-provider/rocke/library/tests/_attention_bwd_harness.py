# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device harness of the general attention backward (test helper).

``run_case`` materialises a backward table case, plans it with a test policy,
uploads every buffer (with the per-batch SEQ_LEN tensors of a padded case),
runs ``prep -> main -> convert`` through
:func:`kernels.common.attention_bwd_run.run_attn_bwd` on a caller-owned
workspace, and returns the gradients plus the integrity facts the numeric
tests assert:

* output buffers start as a sentinel; every element outside the gradient views
  (stride gaps) and a tail canary after each buffer must keep it;
* the workspace is pre-poisoned with NaN (prep must initialise every byte that
  is read) and followed by a canary that must stay intact;
* batched padding rows of dQ / dK / dV must be exact zeros;
* ragged (THD) cases run on a :class:`ThdStorage` (offset table width and
  units, tokens before, between and after the sequences, token padding,
  per-sequence counts, workspace bounds, a base offset): input tokens outside
  the sequences hold NaN, and gradient tokens outside them must keep the
  sentinel;
* :func:`narrow_inputs` re-lays a batched case out on element strides that
  are not multiples of 8 (BHSD or BSHD order) behind a base offset of 2, 4 or
  8 bytes (the narrow-access instances, ``stage_vec = 1``).

Mutation hooks (negative controls) can rewrite the uploaded inputs or the plan.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from kernels.common.attention_bwd_plan import attn_bwd_plan
from kernels.common.attention_bwd_run import WORKSPACE_ALIGN, run_attn_bwd

from .sdpa.bwd_cases import _view, evaluate_bwd, materialize_bwd
from .sdpa.bwd_kernel_cases import CUSTOM_CONFIG, config_policy, request_from_case

ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    ARCH = None

DEVICE_ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
CANARY = 512  # elements after every output buffer and the workspace
SENTINEL = -7.0
_KERNEL_CACHE: dict = {}
_REF_CACHE: dict = {}


def enc(x, dtype):
    """float values -> raw 16-bit storage (round to nearest even)."""
    x = np.ascontiguousarray(x, np.float32)
    if dtype == "fp16":
        return x.astype(np.float16).view(np.uint16)
    u = x.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def dec(u, dtype):
    u = np.asarray(u, np.uint16)
    if dtype == "fp16":
        return u.view(np.float16).astype(np.float32)
    return (u.astype(np.uint32) << 16).view(np.float32)


def reference(case):
    """Materialised inputs and the float64 oracle (cached per case id)."""
    if case.id not in _REF_CACHE:
        inp = materialize_bwd(case)
        _REF_CACHE[case.id] = (inp, evaluate_bwd(inp))
    return _REF_CACHE[case.id]


@dataclass
class CaseRun:
    case: object
    plan: object
    dq: np.ndarray
    dk: np.ndarray
    dv: np.ndarray
    ref: object
    integrity: list  # problems with sentinels / canaries / padding rows


def _u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def _replace_launch(plan, stage, *, index=0, scalars=None, drop=False):
    """A copy of ``plan`` with one launch's scalars changed or the launch dropped."""
    launches, seen = [], 0
    for step in plan.launches:
        if step.stage == stage:
            if seen == index:
                seen += 1
                if drop:
                    continue
                if scalars:
                    new = dict(step.scalars)
                    new.update(scalars)
                    step = dataclasses.replace(step, scalars=MappingProxyType(new))
            else:
                seen += 1
        launches.append(step)
    return dataclasses.replace(plan, launches=tuple(launches))


# ---------------------------------------------------------------------------
# THD storage (ragged cases)
# ---------------------------------------------------------------------------

_THD_ROLES = ("q", "k", "v", "o", "do", "dq", "dk", "dv")
_Q_SIDE_ROLES = ("q", "o", "do", "dq")


@dataclass(frozen=True)
class ThdStorage:
    """Device storage of a ragged (THD) case's packed ``[T, H, D]`` tensors.

    The case's compact tensors are placed into storage tokens: ``lead`` tokens
    before the first sequence, ``gap`` tokens after every sequence (only with
    ``seq_len_counts``: the per-sequence counts then end each sequence before
    the next offset) and ``tail`` tokens after the last one. Every tensor has
    head stride ``D`` and token stride ``H * D + token_pad`` (``uniform``:
    ``max(h_q, h_k, h_v) * D + token_pad`` for every tensor, so the q and kv
    tables can share token units). Input tokens outside the sequences hold
    NaN (``poison``) and must never be read; gradient tokens outside them must
    keep the sentinel. ``token_pad`` may make the token strides any element
    count; ``base_offset`` elements precede every tensor in its buffer, so the
    tensor pointers are only ``2 * base_offset`` bytes aligned (with either,
    the plan selects the narrow-access instances).

    The offset tables hold element offsets of Q (q side) and K (kv side) in
    units of ``multiplier`` elements (``multiplier = 0``: token units, the
    token stride, which needs one token stride for Q and K), as int32 or
    int64. ``max_total_q`` / ``max_total_kv``: the caller's workspace row
    bounds. ``bound`` decides the ``None`` ones: ``"auto"`` passes the
    storage token count when the storage is not the compact case layout (else
    no bound), ``"storage"`` always passes the storage token count, and
    ``"none"`` never passes a bound (what a hipDNN caller sends: hipDNN has no
    max-total API), so the plan indexes the workspace by sequence slot.
    """

    lead: int = 0
    gap: int = 0
    tail: int = 0
    token_pad: int = 0
    uniform: bool = False
    offset_dtype: str = "int32"
    multiplier: int = 1
    seq_len_counts: bool = False
    max_total_q: int | None = None
    max_total_kv: int | None = None
    poison: bool = True
    base_offset: int = 0
    bound: str = "auto"  # "auto" | "storage" | "none"

    def __post_init__(self) -> None:
        if self.bound not in ("auto", "storage", "none"):
            raise ValueError(f"bound={self.bound!r} not in auto / storage / none")
        if self.bound == "none" and (
            self.max_total_q is not None or self.max_total_kv is not None
        ):
            raise ValueError("bound='none' passes no max_total")

    @property
    def compact(self) -> bool:
        return (
            self.lead == self.gap == self.tail == self.token_pad == 0
            and not self.uniform
        )


@dataclass
class _ThdLayout:
    """Storage token maps, token strides, tables and request fields."""

    maps: dict  # side ("q" | "kv") -> int array: compact token -> storage token
    tokens: dict  # side -> storage token count
    stride: dict  # role -> token stride (elements)
    heads: dict  # role -> heads
    tables: dict  # device role -> int array (offsets, counts)
    request: dict  # request fields
    d: int
    base: int = 0  # elements before every tensor in its buffer


def _thd_layout(case, storage: ThdStorage) -> _ThdLayout:
    if storage.gap and not storage.seq_len_counts:
        raise ValueError("gap tokens inside the offsets need per-sequence counts")
    lq, lk = case.lens()
    d = case.d
    heads = {"q": case.h_q, "k": case.h_k, "v": case.h_v, "o": case.h_q}
    heads.update({"do": case.h_q, "dq": case.h_q, "dk": case.h_k, "dv": case.h_v})
    wide = max(case.h_q, case.h_k, case.h_v)
    stride = {
        r: (wide if storage.uniform else heads[r]) * d + storage.token_pad
        for r in _THD_ROLES
    }
    maps, tokens, starts = {}, {}, {}
    for side, lens in (("q", lq), ("kv", lk)):
        st = [storage.lead]
        for n in lens:
            st.append(st[-1] + n + storage.gap)
        starts[side] = np.array(st, np.int64)
        maps[side] = np.concatenate(
            [np.arange(s, s + n) for s, n in zip(st[:-1], lens)] or [np.zeros(0)]
        ).astype(np.int64)
        tokens[side] = st[-1] + storage.tail
    mult = storage.multiplier
    if mult == 0:
        if stride["q"] != stride["k"]:
            raise ValueError("token-unit offsets need one token stride for Q and K")
        mult = stride["q"]
    tables = {}
    for side, role in (("q", "q"), ("kv", "k")):
        elems = starts[side] * stride[role]
        if np.any(elems % mult):
            raise ValueError("offsets are not whole multiples of the multiplier")
        units = elems // mult
        if storage.offset_dtype == "int32":
            if units.max() >= 2**31:
                raise ValueError("int32 offset table overflows")
            tables[f"offsets_{side}"] = units.astype(np.int32)
        else:
            tables[f"offsets_{side}"] = units.astype(np.int64)
    request = {
        "strides": {r: (0, d, stride[r], 1) for r in _THD_ROLES},
        "ragged_offset_dtype": storage.offset_dtype,
        "ragged_offset_multiplier": mult,
    }
    if storage.base_offset:
        request["tensor_alignment"] = _alignment(storage.base_offset)
    if storage.seq_len_counts:
        tables["seq_len_q"] = np.array(lq, np.int32)
        tables["seq_len_kv"] = np.array(lk, np.int32)
        request.update(padding=True, has_seq_len_q=True, has_seq_len_kv=True)
    for side, field in (("q", "max_total_q"), ("kv", "max_total_kv")):
        bound = getattr(storage, field)
        if bound is None and (
            storage.bound == "storage"
            or (storage.bound == "auto" and not storage.compact)
        ):
            bound = tokens[side]
        if bound is not None:
            request[field] = int(bound)
    return _ThdLayout(
        maps, tokens, stride, heads, tables, request, d, storage.base_offset
    )


def _side(role):
    return "q" if role in _Q_SIDE_ROLES else "kv"


def _thd_rows(lay: _ThdLayout, role: str, buf: np.ndarray) -> np.ndarray:
    """``[storage tokens, H * D]`` view of a THD storage buffer (no copy)."""
    n, s, o = lay.tokens[_side(role)], lay.stride[role], lay.base
    return buf[o : o + n * s].reshape(n, s)[:, : lay.heads[role] * lay.d]


def _alignment(base_elems: int) -> int:
    """Byte alignment of a 16-bit tensor ``base_elems`` elements into a
    256-byte aligned allocation."""
    nbytes = 2 * base_elems
    return 256 if nbytes == 0 else min(256, nbytes & -nbytes)


# ---------------------------------------------------------------------------
# narrow-access views of batched cases
# ---------------------------------------------------------------------------

NARROW_VIEWS = ("bhsd", "bshd")


def _narrow_strides(view: str, b: int, h: int, s: int, d: int, unit: int = 1) -> tuple:
    """``(B, H, S, 1)`` element strides of ``view``: odd S and B strides and
    an H stride that is not a multiple of 8; with ``unit`` (2 or 4) every
    stride a multiple of ``unit`` and none of them of 8 (the wider accesses
    of ``narrow_vec``)."""
    if unit != 1:
        if unit not in (2, 4) or d % 8:
            raise ValueError(f"unit={unit} needs 2 or 4 and D a multiple of 8")
        if view == "bhsd":
            ss = d + unit
            sh = s * ss + unit
            return (h * sh + unit, sh, ss, 1)
        if view == "bshd":
            sh = d + unit
            ss = h * sh + unit
            return (s * ss + unit, sh, ss, 1)
        raise ValueError(f"unknown narrow view {view!r}")
    if view == "bhsd":
        ss = d + 3
        sh = s * ss + (1 if (s * ss + 1) % 8 else 3)
        sb = h * sh + 7
    elif view == "bshd":
        sh = d + 5
        ss = h * sh + (3 if (h * sh + 3) % 2 else 4)
        sb = s * ss + 9
    else:
        raise ValueError(f"unknown narrow view {view!r}")
    return (sb, sh, ss, 1)


def narrow_inputs(case, view: str, base_offset: int, *, unit: int = 1):
    """``((inputs, reference), request fields)`` of a batched case re-laid out
    for narrow access.

    Every tensor gets its own buffer with ``base_offset`` elements before it
    and the strides of :func:`_narrow_strides` (per-tensor heads and sequence
    length); the stride gaps of the inputs hold NaN, so a read outside the
    views poisons the result. The oracle is unchanged (same logical values).
    ``unit`` (2 or 4) makes every stride a multiple of it (none of 8), for the
    ``narrow_vec`` access widths; pair it with a ``base_offset`` of ``unit``
    elements (a ``2 * unit``-byte aligned base).
    """
    if case.length_mode == "ragged" or case.layout == "packed_qkv":
        raise ValueError("narrow_inputs re-lays batched, unpacked cases only")
    inp, ref = reference(case)
    heads = {"q": case.h_q, "k": case.h_k, "v": case.h_v, "o": case.h_q}
    seqs = {"q": case.s_q, "k": case.s_kv, "v": case.s_kv, "o": case.s_q}
    of = {"do": "o", "dq": "q", "dk": "k", "dv": "v"}
    buffers = {n: v for n, v in inp.buffers.items() if n not in inp.sizes}
    descs, sizes, strides = {}, {}, {}
    for role in _THD_ROLES:
        src = inp.descs[role]
        base = of.get(role, role)
        dims = (case.b, heads[base], seqs[base], case.d)
        st = _narrow_strides(view, *dims, unit=unit)
        span = 1 + sum((n - 1) * x for n, x in zip(dims, st))
        dsc = dataclasses.replace(
            src, buffer=role, offset=base_offset, dims=dims, strides=st
        )
        descs[role] = dsc
        sizes[role] = base_offset + span
        strides[role] = st
        if role in ("dq", "dk", "dv"):
            continue
        buf = np.full(sizes[role], np.nan, np.float32)
        _view(buf, dsc)[...] = _view(inp.buffers[src.buffer], src)
        buffers[role] = buf
    new = dataclasses.replace(inp, buffers=buffers, descs=descs, sizes=sizes)
    request = {"strides": strides, "tensor_alignment": _alignment(base_offset)}
    return (new, ref), request


def plan_case(case, *, arch: str, config: str, overrides=None, request=None):
    """The test plan of ``case`` on ``arch`` (configuration knobs plus overrides).

    ``overrides``: spec knobs and runtime knobs (``g_split``, ``lb_order``,
    ``scale_placement``) on top of the configuration; ``request``: request
    fields replaced on the case's request (for example ``num_cus``).

    The ``custom`` configuration is a caller's knob stack, not a shipped one:
    where the validator refuses its own cell (``case`` and ``request`` without
    ``overrides``; a coupling or a recorded resource refusal) the calling test
    is skipped with the validator's reason. A shipped configuration's refusal,
    and a refused variant of a custom cell that plans, raise.
    """
    req = request_from_case(case)
    if request:
        req = dataclasses.replace(req, **request)
    # Every device target is declared (validated): the plan is the target's
    # own, so its validator and resource refusals are the ones the CPU pins
    # and the gate rows record.
    try:
        return attn_bwd_plan(req, arch, policy=config_policy(config, arch, overrides))
    except ValueError:
        if config != CUSTOM_CONFIG:
            raise
        try:
            attn_bwd_plan(req, arch, policy=config_policy(config, arch))
        except ValueError as exc:
            import pytest

            pytest.skip(f"the custom configuration refuses {case.id} on {arch}: {exc}")
        raise


def case_plan(
    case,
    *,
    arch: str,
    config: str,
    overrides=None,
    request=None,
    thd: ThdStorage | None = None,
    mutate_plan=None,
):
    """``(plan, THD layout or None)`` that :func:`run_case` launches.

    A ragged case takes its request fields from its device storage (``thd``,
    default: the compact layout); ``request`` entries win over them;
    ``mutate_plan`` rewrites the plan last (a refused ``custom`` cell skips:
    :func:`plan_case`).
    """
    lay = None
    if case.length_mode == "ragged":
        lay = _thd_layout(case, thd or ThdStorage())
        request = {**lay.request, **(request or {})}
    elif thd is not None:
        raise ValueError("THD storage needs a ragged case")
    plan = plan_case(
        case, arch=arch, config=config, overrides=overrides, request=request
    )
    if mutate_plan is not None:
        plan = mutate_plan(plan)
    return plan, lay


def skip_unless_planned(
    case, *, arch: str, config: str, overrides=None, request=None, thd=None
):
    """The plan of ``case`` with ``overrides`` (as :func:`run_case` plans it),
    or skip the calling test with the validator's reason when that plan is
    refused while the configuration alone plans: a variant the configuration
    cannot take (a coupling, a recorded resource refusal). A refusal of a
    shipped configuration itself is raised, not skipped (the ``custom``
    configuration's own refused cells skip in :func:`plan_case`)."""
    import pytest

    kw = dict(arch=arch, config=config, request=request, thd=thd)
    try:
        return case_plan(case, overrides=overrides, **kw)[0]
    except ValueError as exc:
        case_plan(case, **kw)  # the configuration alone must plan
        pytest.skip(f"{overrides} is not legal for {case.id} ({config}): {exc}")


def _only(plan, stages):
    return dataclasses.replace(
        plan, launches=tuple(s for s in plan.launches if s.stage in stages)
    )


@contextlib.contextmanager
def main_built_with(**body_kw):
    """Build the main kernel through ``_emit_main(..., **body_kw)`` (test-only
    body switches such as ``sentinel=False``); a private kernel cache is used
    so the switched kernel never shares a cache entry with the shipped one
    (on disk it is keyed by its ``cache_variant``)."""
    from unittest import mock  # here, not at import: it pulls in asyncio

    from kernels.common import attention_bwd, attention_bwd_run

    def build(spec, *, arch):
        return attention_bwd._emit_main(spec, arch, **body_kw)

    # the on-disk cache (ROCKE_BWD_HSACO_CACHE) keys the switched kernel apart
    build.cache_variant = "_emit_main:" + ",".join(
        f"{k}={v!r}" for k, v in sorted(body_kw.items())
    )
    with mock.patch.dict(attention_bwd_run._BUILDERS, {"main": build}):
        yield {}


def run_case(
    case,
    *,
    arch: str,
    config: str,
    mutate_inputs=None,
    mutate_plan=None,
    workspace_poison: float = float("nan"),
    overrides=None,
    inputs=None,
    after_prep=None,
    cache=None,
    request=None,
    mutate_lengths=None,
    thd: ThdStorage | None = None,
) -> CaseRun:
    """Plan, upload, launch, read back. Allocates only in this harness.

    ``overrides``: spec and runtime knobs on top of the configuration;
    ``request``: request fields replaced on the case's request. ``inputs``: a
    ``(SdpaBwdInputs, reference)`` pair replacing the case's materialised
    inputs and oracle. ``after_prep(ws, plan)``: run prep alone, hand the
    fp32 workspace (a numpy view, edited in place) to the hook, upload it and
    run the remaining launches. ``cache``: kernel cache (default: shared).
    ``mutate_lengths({role: array}) -> {role: array}``: rewrite the uploaded
    SEQ_LEN tensors of a padded case (``seq_len_q`` / ``seq_len_kv``; for
    example counts at an element stride) or the tables of a ragged case
    (``offsets_q`` / ``offsets_kv`` and the optional counts). ``thd``: the
    device storage of a ragged case (default: the compact case layout with
    int32 element offsets); the gradients come back as compact ``[T, H, D]``
    arrays, and every storage element outside the sequences must keep the
    sentinel.
    """
    from rocke.runtime.hip_module import Runtime

    inp, ref = reference(case) if inputs is None else inputs
    ragged = case.length_mode == "ragged"
    storage = thd or ThdStorage()
    plan, lay = case_plan(
        case,
        arch=arch,
        config=config,
        overrides=overrides,
        request=request,
        thd=thd,
        mutate_plan=mutate_plan,
    )
    dt = case.dtype
    sentinel = enc(np.float32(SENTINEL), dt)[()]
    host = {}
    if ragged:
        poison = np.nan if storage.poison else 0.0
        for role in _THD_ROLES:
            side = _side(role)
            n = lay.base + lay.tokens[side] * lay.stride[role]
            if role in ("dq", "dk", "dv"):
                host[role] = np.full(n + CANARY, sentinel)
                continue
            dsc = inp.descs[role]
            vals = _view(inp.buffers[dsc.buffer], dsc)
            buf = np.full(n, poison, np.float32)
            _thd_rows(lay, role, buf)[lay.maps[side]] = vals.reshape(len(vals), -1)
            host[role] = enc(buf, dt)
        lse_rows = np.full((lay.tokens["q"], case.h_q), poison, np.float32)
        lse_rows[lay.maps["q"]] = np.reshape(inp.buffers["lse"], (-1, case.h_q))
        lse = lse_rows.reshape(-1)
    else:
        for name, size in inp.sizes.items():
            if name in inp.buffers:
                host[name] = enc(inp.buffers[name], dt)
            else:  # a gradient buffer: sentinel plus canary tail
                host[name] = np.full(size + CANARY, sentinel)
        lse = np.ascontiguousarray(inp.buffers["lse"], np.float32).reshape(-1).copy()
    if mutate_inputs is not None:
        lse = mutate_inputs(lse)
    ws = np.full(plan.workspace_bytes // 4 + CANARY, workspace_poison, np.float32)
    # Padded mode: per-batch SEQ_LEN tensors (int32, one count per batch at the
    # request's element stride). THD: the offset tables (int32 / int64) and,
    # with per-sequence counts, the SEQ_LEN tensors.
    lengths = {}
    if ragged:
        lengths = {k: np.ascontiguousarray(v) for k, v in lay.tables.items()}
    else:
        for role in ("seq_len_q", "seq_len_kv"):
            counts = getattr(inp.fwd, role, None)
            if counts is not None:
                lengths[role] = np.ascontiguousarray(np.reshape(counts, -1), np.int32)
    if mutate_lengths is not None:
        lengths = mutate_lengths(lengths)
    rt = Runtime()
    dev = {}
    try:
        uploads = list(host.items()) + [("lse", lse), ("__ws", ws)]
        uploads += [(f"__{role}", a) for role, a in lengths.items()]
        for name, a in uploads:
            p = rt.alloc(max(int(a.nbytes), 16))
            rt.memcpy_h2d(p, _u8(a), a.nbytes)
            dev[name] = p
        if dev["__ws"] % WORKSPACE_ALIGN:
            raise AssertionError("device allocation is not 256-byte aligned")
        tensors = {"lse": dev["lse"]}
        for role in lengths:
            tensors[role] = dev[f"__{role}"]
        for role in _THD_ROLES:
            if ragged:
                tensors[role] = dev[role] + 2 * lay.base
            else:
                dsc = inp.descs[role]
                tensors[role] = dev[dsc.buffer] + 2 * dsc.offset
        kcache = _KERNEL_CACHE if cache is None else cache
        kw = dict(
            tensors=tensors,
            workspace=dev["__ws"],
            workspace_size=plan.workspace_bytes,
            cache=kcache,
        )
        if after_prep is None:
            run_attn_bwd(plan, **kw)
        else:
            run_attn_bwd(_only(plan, ("prep",)), **kw)
            rt.sync()
            rt.memcpy_d2h(_u8(ws), dev["__ws"], ws.nbytes)
            after_prep(ws, plan)
            rt.memcpy_h2d(dev["__ws"], _u8(ws), ws.nbytes)
            stages = ("main", "dq", "convert", "reduce", "dense", "dense_dq")
            run_attn_bwd(_only(plan, stages), **kw)
        rt.sync()
        for name, a in host.items():
            rt.memcpy_d2h(_u8(a), dev[name], a.nbytes)
        rt.memcpy_d2h(_u8(ws), dev["__ws"], ws.nbytes)
    finally:
        for p in dev.values():
            rt.free(p)
    integrity = []
    grads = {}
    covered = {}
    for role in ("dq", "dk", "dv"):
        if ragged:
            side = _side(role)
            vals = _thd_rows(lay, role, dec(host[role], dt))[lay.maps[side]]
            grads[role] = vals.reshape(len(vals), lay.heads[role], case.d)
            grads[role] = grads[role].astype(np.float64)
            mask = np.zeros(host[role].shape, bool)
            _thd_rows(lay, role, mask)[lay.maps[side]] = True
            covered[role] = mask
            continue
        dsc = inp.descs[role]
        grads[role] = _view(dec(host[dsc.buffer], dt), dsc).astype(np.float64)
        mask = covered.setdefault(dsc.buffer, np.zeros(host[dsc.buffer].shape, bool))
        _view(mask, dsc)[...] = True
    where = "outside the sequences" if ragged else "outside the gradient views"
    for buf, mask in covered.items():
        raw = host[buf]
        if not np.all(raw[~mask] == sentinel):
            bad = int(np.sum(raw[~mask] != sentinel))
            integrity.append(f"{buf}: {bad} elements {where} changed")
        if not np.all(raw[-CANARY:] == sentinel):
            integrity.append(f"{buf}: tail canary overwritten")
    ws_canary = ws[plan.workspace_bytes // 4 :]
    if not np.all(np.isnan(ws_canary)) and np.isnan(workspace_poison):
        integrity.append("workspace canary overwritten")
    if not ragged:
        lq, lk = case.lens()
        for role, lens in (("dq", lq), ("dk", lk), ("dv", lk)):
            g = grads[role]
            for bi, n in enumerate(lens):
                pad = g[bi, :, n:, :]
                if pad.size and not np.all(pad == 0.0):
                    integrity.append(f"{role}: batch {bi} padding rows are not zero")
    return CaseRun(
        case=case,
        plan=plan,
        dq=grads["dq"],
        dk=grads["dk"],
        dv=grads["dv"],
        ref=ref,
        integrity=integrity,
    )


# ---------------------------------------------------------------------------
# dead-row inputs
# ---------------------------------------------------------------------------


def dead_rows(case):
    """Bool ``[B, H_q, S_q]``: rows 0, len_q // 2 and len_q - 1 of the first
    and last query head of every batch."""
    lq, _ = case.lens()
    mask = np.zeros((case.b, case.h_q, case.s_q), bool)
    for bi, n in enumerate(lq):
        for h in sorted({0, case.h_q - 1}):
            mask[bi, h, [0, n // 2, n - 1]] = True
    return mask


def dead_row_inputs(case, rows, lse_values, o_value=None, *, base=None):
    """Device inputs with ``lse_values`` (cycled) on ``rows`` and, optionally,
    ``o_value`` in O on those rows; plus the oracle with those rows removed
    (their LSE ``-inf`` and O zero). ``base``: the inputs to start from
    (default: the case's materialised inputs)."""
    inp = reference(case)[0] if base is None else base
    lse = np.array(inp.buffers["lse"], np.float32, copy=True)
    lse_rows = lse.reshape(case.b, case.h_q, case.s_q)
    idx = np.argwhere(rows)
    for i, (bi, h, q) in enumerate(idx):
        lse_rows[bi, h, q] = lse_values[i % len(lse_values)]
    o_buf = np.array(inp.buffers["o"], np.float32, copy=True)
    o_ref = np.array(inp.o, np.float64, copy=True)
    o_ref[rows] = 0.0
    if o_value is not None:
        _view(o_buf, inp.descs["o"])[rows] = o_value
    dev = dataclasses.replace(
        inp,
        buffers={**inp.buffers, "lse": lse, "o": o_buf},
        lse=lse.astype(np.float64),
    )
    ref_lse = np.array(inp.lse, np.float64, copy=True)
    ref_lse.reshape(case.b, case.h_q, case.s_q)[rows] = -np.inf
    ref = evaluate_bwd(dataclasses.replace(inp, o=o_ref, lse=ref_lse))
    return dev, ref


def ungated_dsum_hook(rows):
    """``after_prep`` hook: Dsum of ``rows`` (bool ``[B, H_q, S_q]``) becomes
    NaN, as a prep that gates Dsum on ``q < len_q`` only would leave it for a
    NaN O on a dead row."""

    def hook(ws, plan):
        main = plan.launches[1].scalars
        off, _ = plan.workspace_layout["WS_DSUM"]
        n_rows, s_q = main["ws_rows_q"], main["S_q_max"]
        dsum = ws[off // 4 : off // 4 + n_rows * main["h_q"]]
        for bi, h, q in np.argwhere(rows):
            dsum[h * n_rows + bi * s_q + q] = np.nan

    return hook
