# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Support for the torch-free strided numeric test of the dense attention kernels.

The dense kernels take twelve launch-time strides (Q, K, V and O, each batch, head
and token). This module holds everything ``test_attention_dense_strided_numeric.py``
needs and nothing that needs torch:

* the layout model: every tensor is a view (offset plus three element strides) into
  ONE backing buffer per case, the buffers are filled with a quiet NaN carrying a
  payload no computation produces, and every buffer has a NaN guard band in front
  and behind, so a read of a gap, a guard or a neighbour turns O into NaN and a
  write outside the O view is a bit change;
* the cohort matrix (one compiled kernel per cohort, launched with different
  batch, lengths, head counts and strides), the checks after every launch, the
  negative controls (a strided view launched with packed strides must fail) and the
  refusal rows;
* two backends behind one interface: :class:`HipBackend` (HIP buffers, the
  production runner) and :class:`ModelBackend` (host memory, an fp64 model of a
  kernel that honours its twelve strides, with injectable defects) so the checks
  themselves are tested on any host.

A body whose ``body_serves_strides(spec)`` is True gets the full matrix. A body for
which it is False gets "a strided view is refused, with its reason" and packed
launches that pass. Every launch, LSE and KV-split cohorts included, goes through
``run_attention_dense_torch`` of the arch's kernel module on device-buffer views
(``View``, ``LseView``) and, for the KV-split workspace, an allocator of device
buffers (``HipBackend.workspace``).
"""

from __future__ import annotations

import ctypes
import dataclasses
import importlib
import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kernels.common._attention_dense_strides import (
    DENSE_STRIDE_AXES,
    DENSE_STRIDE_TENSORS,
    admit_dense_launch_strides,
)
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.numeric.sdpa_reference import sdpa_reference

from .sdpa.helpers import strided_view

ROLES = DENSE_STRIDE_TENSORS  # ("q", "k", "v", "o")
GUARD = 4096  # poison elements (a token row of every shape fits) in front of and behind every buffer's footprint
# Quiet NaNs whose payload is not the canonical one a computation produces (fp16
# 0x7E00, bf16 0x7FC0): a NaN the kernel writes into a gap is a bit change too.
POISON_BITS = {"fp16": 0x7E5A, "bf16": 0x7FDA}
LSE_POISON_BITS = 0x7FC05A5A
KV_SEGMENTS = 3  # uneven; S=256 (four tiles of 64) leaves one segment empty
TOLERANCE = {"fp16": 2e-2, "bf16": 4e-2}  # max_abs, as the dense numeric files

# Set by a mutation run (a pytest plugin): called with ``{role: View}`` before every
# launch and returning the views to launch. None in every normal run.
VIEW_HOOK: Callable[[dict], dict] | None = None


# --------------------------------------------------------------------------- #
# dtype codec (raw 16-bit patterns; stock numpy has no bfloat16)
# --------------------------------------------------------------------------- #


def encode(values, dtype: str) -> np.ndarray:
    values = np.asarray(values)
    if dtype == "fp16":
        return np.ascontiguousarray(values.astype(np.float16)).view(np.uint16)
    return f32_to_bf16(values.astype(np.float32))


def decode(bits, dtype: str) -> np.ndarray:
    bits = np.ascontiguousarray(bits, dtype=np.uint16)
    if dtype == "fp16":
        return bits.view(np.float16).astype(np.float64)
    return bf16_to_f32(bits).astype(np.float64)


# --------------------------------------------------------------------------- #
# Problem shapes and layouts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Shape:
    name: str
    b: int
    sq: int
    skv: int
    hq: int
    hkv: int

    def dims(self, role: str, d: int) -> tuple:
        """Logical ``(B, H, S, D)`` of one tensor."""
        if role in ("q", "o"):
            return (self.b, self.hq, self.sq, d)
        return (self.b, self.hkv, self.skv, d)

    def label(self, d: int) -> str:
        return f"B{self.b} Sq{self.sq} Skv{self.skv} Hq{self.hq} Hkv{self.hkv} D{d}"


# A, B and C differ pairwise in batch, lengths and head counts (all tile multiples,
# so every factory spec stays on the runtime-shape path and shares one binary). X is
# cross length (no mask only) and N too, with K/V of exactly one tile. E has batch 1 and
# one KV head: extent-1 axes.
SHAPES = {
    s.name: s
    for s in (
        Shape("A", 2, 256, 256, 8, 2),
        Shape("B", 2, 512, 512, 8, 8),
        Shape("C", 1, 512, 512, 16, 2),
        Shape("X", 2, 256, 512, 8, 2),
        Shape("N", 2, 256, 64, 8, 2),
        Shape("E", 1, 256, 256, 8, 1),
    )
}


@dataclass(frozen=True)
class Place:
    """One view inside a backing buffer: element offset (before the guard band),
    element strides ``(batch, head, token)`` of the logical ``(B, H, S, D)`` view and
    the head-dim stride. ``reported`` is what the launch is told when it differs from
    the truth (garbage on an extent-1 axis, or a lie in a negative control)."""

    buf: str
    offset: int
    strides: tuple
    dstride: int = 1
    reported: tuple | None = None

    def need(self, dims) -> int:
        n = [dims[0], dims[1], dims[2], dims[3]]
        s = [*self.strides, self.dstride]
        return self.offset + 1 + sum((a - 1) * b for a, b in zip(n, s))

    def told(self) -> tuple:
        return self.strides if self.reported is None else self.reported


def nested(order: str, pads=(0, 0, 0), offset: int = 0):
    """Layout whose buffer dims are ``order`` (outer to inner over b/h/s), then D.
    ``pads`` adds elements to the outer, middle and inner level strides."""

    def make(role, dims):
        b, h, s, d = dims
        size = {"b": b, "h": h, "s": s}
        st = {order[2]: d + pads[2]}
        st[order[1]] = size[order[2]] * st[order[2]] + pads[1]
        st[order[0]] = size[order[1]] * st[order[1]] + pads[0]
        return Place(role, offset, (st["b"], st["h"], st["s"]))

    return make


def batch_slice(role, dims):
    """The odd batches of a ``[2B, S, H, D]`` buffer (``x[1::2]``)."""
    b, h, s, d = dims
    one = s * h * d
    return Place(role, one, (2 * one, d, h * d))


def spread(head_pad: int, token_pad: int, batch_pad: int):
    """BSHD with slack at every level, so that raising any ONE of the three strides
    by 8 elements still gives a view that overlaps nothing: the stride controls
    tell the kernel one value that is wrong and the host still admits the launch."""

    def make(role, dims):
        b, h, s, d = dims
        sh = d + head_pad
        ss = h * (sh + 8) + token_pad
        sb = s * (ss + 8) + batch_pad
        return Place(role, 0, (sb, sh, ss))

    return make


BSHD = nested("bsh")
BHSD = nested("bhs")
SBHD = nested("sbh")
PAD_TOKEN = nested("bsh", (0, 8, 0))

# Garbage for extent-1 axes: never a stride the kernel may step by, so the host
# must canonicalize them (not a multiple of 8, zero, negative, past 2^31).
GARBAGE = {
    ("q", 0): 7,
    ("k", 0): 2**33 + 3,
    ("k", 1): -5,
    ("v", 0): 0,
    ("v", 1): 2**40,
    ("o", 0): 123456789,
}

LAYOUTS = {
    "bshd": dict(q=BSHD, k=BSHD, v=BSHD, o=BSHD),
    "bhsd": dict(q=BHSD, k=BHSD, v=BHSD, o=BHSD),
    "sbhd": dict(q=SBHD, k=SBHD, v=SBHD, o=SBHD),
    "batch_slice": dict(q=batch_slice, k=batch_slice, v=batch_slice, o=batch_slice),
    "pad_token": dict(q=PAD_TOKEN, k=PAD_TOKEN, v=PAD_TOKEN, o=PAD_TOKEN),
    "o_differs": dict(q=BSHD, k=BSHD, v=BSHD, o=BHSD),
    "spread": dict(
        q=spread(16, 32, 64),
        k=spread(24, 40, 72),
        v=spread(32, 48, 80),
        o=spread(40, 56, 88),
    ),
    "all_distinct": dict(
        q=nested("bhs", (24, 16, 8)),
        k=nested("sbh", (32, 40, 16)),
        v=nested("bsh", (56, 8, 24)),
        o=nested("shb", (16, 24, 32)),
    ),
}
# Layouts whose canonical strides equal the packed ones (a body that does not serve
# strides takes them): packed BSHD, and packed BSHD with garbage on extent-1 axes.
PACKED_LAYOUTS = ("bshd", "extent1")


def packed_qkv(shape: Shape, d: int) -> dict:
    """Q, K and V as views into one ``[B, S, Hq + 2*Hkv, D]`` buffer (MHA: the strides
    of ``[B, S, 3, H, D]``)."""
    ht = shape.hq + 2 * shape.hkv
    st = (shape.sq * ht * d, d, ht * d)
    return {
        "q": Place("qkv", 0, st),
        "k": Place("qkv", shape.hq * d, st),
        "v": Place("qkv", (shape.hq + shape.hkv) * d, st),
    }


@dataclass
class Case:
    """One launch: the places the kernel is told about, the places the data is
    written through (differs only in negative controls) and what must happen."""

    layout: str
    places: dict
    data: dict | None = None
    expect: str = "pass"  # pass | fail (negative control) | refuse
    reason: str = ""  # a refuse case: text the ValueError must contain

    def data_place(self, role: str) -> Place:
        return (self.data or self.places)[role]


def _o_bshd(shape: Shape, d: int) -> Place:
    return BSHD("o", shape.dims("o", d))


def make_case(name: str, shape: Shape, d: int, *, o_packed: bool = False):
    """The :class:`Case` of layout ``name`` on ``shape``, or None where it does not
    apply. ``o_packed`` forces a packed BSHD O (the KV-segment combine writes it)."""
    if name == "packed_qkv":
        if shape.sq != shape.skv:
            return None
        places = packed_qkv(shape, d)
        places["o"] = _o_bshd(shape, d)
        label = "packed_qkv_" + ("mha" if shape.hq == shape.hkv else "gqa")
    elif name == "extent1":
        places = {r: BSHD(r, shape.dims(r, d)) for r in ROLES}
        garbage = {}
        for (role, axis), value in GARBAGE.items():
            if shape.dims(role, d)[axis] == 1:
                told = list(places[role].strides)
                told[axis] = value
                garbage[role] = told
        if not garbage:
            return None
        for role, told in garbage.items():
            places[role] = dataclasses.replace(places[role], reported=tuple(told))
        label = "extent1"
    else:
        recipe = LAYOUTS[name]
        places = {r: recipe[r](r, shape.dims(r, d)) for r in ROLES}
        label = name
    if o_packed:
        if name in ("o_differs", "bshd", "extent1"):
            if name == "o_differs":
                return None  # identical to bshd once O is packed
        else:
            places["o"] = _o_bshd(shape, d)
            label += "+oBSHD"
    if name in ("all_distinct", "spread"):
        roles = ROLES[:3] if o_packed else ROLES
        vals = [places[r].strides[i] for r in roles for i in range(3)]
        if len(set(vals)) != len(vals):
            raise AssertionError(f"{name} strides collide on {shape}: {vals}")
    return Case(label, places)


def negative_case(role: str, shape: Shape, d: int, *, o_packed: bool = False) -> Case:
    """Every tensor head-major (O packed BSHD with ``o_packed``), handed over
    truthfully except ``role``, whose strides are told as packed BSHD: the head and
    token strides are swapped against the data, so the result must be wrong."""
    data = {r: BHSD(r, shape.dims(r, d)) for r in ROLES}
    if o_packed:
        data["o"] = _o_bshd(shape, d)
    told = dict(data)
    told[role] = dataclasses.replace(
        data[role], reported=BSHD(role, shape.dims(role, d)).strides
    )
    return Case(f"neg_packed_strides_{role}", told, data=data, expect="fail")


def slot_case(
    role: str, axis: int, shape: Shape, d: int, *, o_packed: bool = False
) -> Case:
    """The ``spread`` layout with ONE of the twelve strides told 8 elements larger (a
    kernel that does not read that one launch value, or reads it wrong): the launch is
    admitted, the data is where the unchanged strides say, the result must be wrong."""
    base = make_case("spread", shape, d, o_packed=o_packed)
    places = dict(base.places)
    told = list(places[role].strides)
    told[axis] += 8
    places[role] = dataclasses.replace(places[role], reported=tuple(told))
    name = f"neg_one_stride_{role}_{DENSE_STRIDE_AXES[axis]}"
    return Case(name, places, data=base.places, expect="fail")


def refusal_cases(shape: Shape, d: int, *, served: bool, kv_split: bool) -> list:
    """Launches the runner must refuse before anything runs, with the reason."""
    out = []

    def packed_places():
        return {r: BSHD(r, shape.dims(r, d)) for r in ROLES}

    if served:
        # head-dim stride 2 (an array twice as wide, so a launch that should have
        # been refused would still stay inside the buffer)
        places = packed_places()
        b, h, s, _ = shape.dims("q", d)
        row = h * d * 2
        places["q"] = Place("q", 0, (s * row, d * 2, row), dstride=2)
        out.append(
            Case(
                "refuse_head_dim_stride2_q",
                places,
                expect="refuse",
                reason="head_dim_stride",
            )
        )
        # token stride that is not a multiple of 8 elements
        places = packed_places()
        places["q"] = nested("bsh", (0, 4, 0))("q", shape.dims("q", d))
        out.append(
            Case(
                "refuse_misaligned_q_token",
                places,
                expect="refuse",
                reason="stride_alignment",
            )
        )
        # base address not a multiple of 16 bytes
        places = packed_places()
        places["k"] = dataclasses.replace(places["k"], offset=1)
        out.append(
            Case(
                "refuse_misaligned_k_base",
                places,
                expect="refuse",
                reason="pointer_alignment",
            )
        )
        if kv_split:
            places = packed_places()
            places["o"] = BHSD("o", shape.dims("o", d))
            out.append(
                Case(
                    "refuse_kv_split_strided_o",
                    places,
                    expect="refuse",
                    reason="kv_segments",
                )
            )
    else:
        for name in ("bhsd", "batch_slice"):
            case = make_case(name, shape, d)
            out.append(
                Case(f"refuse_{name}", case.places, expect="refuse", reason="packed")
            )
    return out


# --------------------------------------------------------------------------- #
# Buffers and views
# --------------------------------------------------------------------------- #


def place_window(array: np.ndarray, place: Place, dims: tuple, writeable=False):
    """``[B, S, H, D]`` elements of the flat ``array`` (guard band included) that
    ``place`` addresses, by its true strides."""
    b, h, s, d = dims
    sb, sh, ss = place.strides
    return strided_view(
        "operand",
        array,
        GUARD + place.offset,
        (b, s, h, d),
        (sb, ss, sh, place.dstride),
        writeable,
    )


class Buffer:
    """A flat host array (the truth) with a device mirror made by a backend."""

    def __init__(self, backend, host: np.ndarray):
        self.backend = backend
        self.host = host
        self.addr, self.handle = backend.alloc(host)

    @property
    def itemsize(self) -> int:
        return self.host.itemsize

    def ptr(self, offset: int = 0) -> int:
        return self.addr + self.itemsize * (GUARD + offset)

    def fetch(self) -> np.ndarray:
        return self.backend.read(self)

    def data_ptr(self) -> int:
        return self.ptr()


class LseView:
    """The caller-owned LSE buffer as the runner reads it: a contiguous FP32
    ``[B, Hq, Sq, 1]`` tensor (the buffer's guard bands stay outside it)."""

    is_cuda = True
    device = None
    dtype = np.dtype(np.float32)

    def __init__(self, buf: Buffer, shape: tuple):
        self.buf, self.shape = buf, shape

    def data_ptr(self) -> int:
        return self.buf.ptr(0)

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return 4

    def stride(self, dim=None):
        st = (self.shape[1] * self.shape[2], self.shape[2], 1, 1)
        return st if dim is None else st[dim]

    def is_contiguous(self) -> bool:
        return True


class View:
    """A duck-typed ``[B, S, H, D]`` tensor over one :class:`Buffer`: what the runner
    reads (``shape``, ``stride()``, ``data_ptr()``), nothing else."""

    is_cuda = True
    device = None

    def __init__(self, buf: Buffer, place: Place, dims: tuple, shift: int = 0):
        self.buf, self.place, self.dims, self.shift = buf, place, dims, shift
        b, h, s, d = dims
        self.shape = (b, s, h, d)

    def stride(self, dim=None):
        sb, sh, ss = self.place.told()
        st = (sb, ss, sh, self.place.dstride)
        return st if dim is None else st[dim]

    def data_ptr(self) -> int:
        return self.buf.ptr(self.place.offset + self.shift)

    def numel(self) -> int:
        return math.prod(self.shape)

    def element_size(self) -> int:
        return 2

    def is_contiguous(self) -> bool:
        b, h, s, d = self.dims
        return tuple(self.stride()) == (s * h * d, h * d, d, 1)

    def shifted(self, elements: int) -> "View":
        return View(self.buf, self.place, self.dims, self.shift + elements)

    def window(self, array: np.ndarray, writeable: bool = False) -> np.ndarray:
        """The view's elements of ``array`` (the buffer's flat contents) as ``[B, S, H, D]``."""
        return place_window(array, self.place, self.dims, writeable)

    def footprint(self) -> np.ndarray:
        inside = np.zeros(self.buf.host.size, dtype=bool)
        self.window(inside, writeable=True)[...] = True
        return inside


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _u8(array: np.ndarray):
    return (ctypes.c_uint8 * array.nbytes).from_buffer(array)


class _DevicePtr:
    """An int device address with the ``data_ptr()`` the launcher reads, owning its memory."""

    def __init__(self, mem):
        self._mem = mem

    def data_ptr(self) -> int:
        return self._mem.ptr()


class HipBackend:
    """HIP buffers and the production runner; torch is never imported."""

    name = "hip"

    def __init__(self):
        from rocke.runtime.hip_module import Runtime

        self.rt = Runtime()

    @staticmethod
    def compiler_label() -> str:
        """The compiler COMGR loaded, once something has compiled."""
        from rocke.runtime import comgr

        info = comgr.loaded_compiler_info()
        return "llvm " + ".".join(str(n) for n in info.llvm_version)

    def alloc(self, host: np.ndarray):
        from rocke.runtime.launcher import DeviceMem

        mem = DeviceMem(host.nbytes)
        self.rt.memcpy_h2d(mem.ptr(), _u8(host), host.nbytes)
        return mem.ptr(), mem

    def read(self, buf: Buffer) -> np.ndarray:
        from rocke.runtime import synchronize_and_release

        synchronize_and_release()
        out = np.empty_like(buf.host)
        self.rt.memcpy_d2h(_u8(out), buf.addr, out.nbytes)
        return out

    @staticmethod
    def workspace(shape) -> _DevicePtr:
        """An FP32 device buffer of ``shape``: the runner's KV-split workspace."""
        from rocke.runtime.launcher import DeviceMem

        return _DevicePtr(DeviceMem(4 * math.prod(shape)))

    def launch(self, kmod, spec, arch, q, k, v, o, scale, lse) -> None:
        from rocke.runtime import synchronize_and_release

        kmod.run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=o,
            scale=scale,
            arch=arch,
            lse=lse,
            workspace_allocator=self.workspace,
        )
        synchronize_and_release()


class ModelBackend:
    """Host memory and an fp64 model of a kernel that honours its twelve strides.

    The admission is the real one (``admit_dense_launch_strides`` with the body's
    ``body_serves_strides``) and the model reads and writes through the admitted
    values, so a strided view, a lie in a negative control or a refusal behaves as on
    a device. ``defect`` injects a fault the checks must catch: ``blind`` (reads
    and writes packed BSHD whatever the strides), ``element`` (one O element is
    wrong), ``outside`` (one O-buffer element outside the view is written),
    ``input`` (one input-buffer element is written), ``shift_o`` (O is written one
    row off)."""

    name = "model"

    def __init__(self, defect: str = ""):
        self.defect = defect
        self._next = 1 << 20

    def alloc(self, host: np.ndarray):
        addr = self._next
        self._next += (host.nbytes + (1 << 16)) & ~0xFFFF
        return addr, host.copy()

    def read(self, buf: Buffer) -> np.ndarray:
        return buf.handle.copy()

    @staticmethod
    def _window(view: View, strides: dict, role: str, writeable: bool = False):
        sb, sh, ss = (strides[f"stride_{role}_{a}"] for a in DENSE_STRIDE_AXES)
        b, h, s, d = view.dims
        first = (view.data_ptr() - view.buf.addr) // view.buf.itemsize
        try:
            return strided_view(
                role,
                view.buf.handle,
                first,
                (b, s, h, d),
                (sb, ss, sh, view.place.dstride),
                writeable,
            )
        except ValueError as exc:  # a real kernel would fault: not a refusal
            raise IndexError(f"model kernel out of bounds: {exc}") from exc

    def launch(self, kmod, spec, arch, q, k, v, o, scale, lse) -> None:
        ok, why = kmod.supports_attention_dense(spec, arch=arch)
        if not ok:
            raise NotImplementedError(f"attention_dense unsupported for spec: {why}")
        views = dict(zip(ROLES, (q, k, v, o)))
        strides = admit_dense_launch_strides(
            spec, q=q, k=k, v=v, o=o, serves_strides=kmod.body_serves_strides(spec)
        )
        if self.defect == "blind":
            for role, view in views.items():
                b, h, s, d = view.dims
                strides[f"stride_{role}_batch"] = s * h * d
                strides[f"stride_{role}_head"] = d
                strides[f"stride_{role}_token"] = h * d
        qf, kf, vf = (
            decode(self._window(views[r], strides, r), spec.dtype).transpose(0, 2, 1, 3)
            for r in "qkv"
        )
        o_ref, lse_ref = sdpa_reference(qf, kf, vf, scale=scale, causal=spec.causal)
        out = encode(o_ref.transpose(0, 2, 1, 3), spec.dtype)
        target = views["o"]
        if self.defect == "shift_o":
            target = target.shifted(strides["stride_o_token"])
        if self.defect == "element":
            out[0, 0, 0, 0] = encode(np.array([77.0]), spec.dtype)[0]
        self._window(target, strides, "o", writeable=True)[...] = out
        if self.defect == "outside":
            views["o"].buf.handle[0] = 0
        if self.defect == "input":
            views["k"].buf.handle[0] = 0
        if lse is not None:
            n = lse_ref.size
            lse.buf.handle[GUARD : GUARD + n] = lse_ref.astype(np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
# Specs (from the dispatch factory)
# --------------------------------------------------------------------------- #

# body -> (candidate spec id, knobs applied on top of the candidate's default).
BODIES = {
    "gfx942": {
        "grid": ("gfx942_dense", {"persistent": False}),
        "kvseg": ("gfx942_dense", {"persistent": False, "kv_segments": KV_SEGMENTS}),
        "persist": ("gfx942_dense", {"persistent": True}),
        # The opt-in BF16 KDB pipeline: a gfx942 body that keeps packed addressing.
        "kdb": ("gfx942_dense", {"persistent": True, "lds_num_buffers": 2}),
    },
    "gfx950": {
        "grid": ("gfx950_dense_grid", {}),
        "kvseg": ("gfx950_dense_grid", {"kv_segments": KV_SEGMENTS}),
        "persist": ("gfx950_dense_persist", {}),
        "widedma": ("gfx950_dense_persist_widedma", {}),
    },
}


def kernel_module(arch: str):
    return importlib.import_module(f"kernels.{arch}.attention_dense")


def make_spec(
    arch: str, body: str, shape: Shape, d: int, dtype: str, causal: bool, lse: bool
):
    """The dispatch factory's spec for the cohort on ``shape``: the candidate's default
    with the body's knobs (and ``emit_lse``, which is not a knob). The factory pins
    nothing else, so a tuning change is picked up here with no edit."""
    import kernels.common.attention_unified as au
    from dispatch.attention import (
        AttentionRequest,
        attention_tuning_spec,
        tuning_spec_with_knobs,
    )

    spec_id, knobs = BODIES[arch][body]
    request = AttentionRequest(
        batch=shape.b,
        nhead_q=shape.hq,
        nhead_k=shape.hkv,
        seqlen_q=shape.sq,
        seqlen_k=shape.skv,
        hdim_q=d,
        hdim_v=d,
        arch=arch,
        dtype=dtype,
        mask_type=1 if causal else 0,
    )
    # Pin the resolved arch like the dispatch wiring tests, so the factory does not
    # depend on the host GPU.
    old, au._RESOLVED_ATTENTION_ARCH = au._RESOLVED_ATTENTION_ARCH, arch
    try:
        if knobs:
            spec = tuning_spec_with_knobs(request, spec_id, knobs).kernel_spec
        else:
            spec = attention_tuning_spec(request, spec_id).kernel_spec
    finally:
        au._RESOLVED_ATTENTION_ARCH = old
    for name, want in knobs.items():
        if getattr(spec, name) != want:
            raise AssertionError(f"{arch} {body}: knob {name}={want} did not stick")
    if lse:
        spec = dataclasses.replace(spec, emit_lse=True)
    return spec


# --------------------------------------------------------------------------- #
# Cohorts: one compiled kernel, launched with different shapes and strides
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cohort:
    arch: str
    body: str
    d: int
    dtype: str
    causal: bool
    lse: bool = False
    slots: bool = False  # also launch the twelve single-stride negative controls

    @property
    def id(self) -> str:
        mask = "causal" if self.causal else "full"
        return f"{self.arch}-{self.body}-d{self.d}-{self.dtype}-{mask}" + (
            "-lse" if self.lse else ""
        )


def _cohorts(arch: str, body: str, dims: tuple) -> list:
    """Per head size: both dtypes and both masks, and (except for the KV split, which
    has no LSE output) both LSE choices."""
    out = []
    for d in dims:
        out += [
            Cohort(arch, body, d, "fp16", False, slots=True),
            Cohort(arch, body, d, "bf16", True),
        ]
        if body != "kvseg":
            out += [
                Cohort(arch, body, d, "fp16", True, lse=True),
                Cohort(arch, body, d, "bf16", False, lse=True),
            ]
    return out


COHORTS = (
    *_cohorts("gfx942", "grid", (64, 128)),
    *_cohorts("gfx942", "kvseg", (64,)),
    *_cohorts("gfx942", "persist", (64, 128)),
    Cohort("gfx942", "kdb", 128, "bf16", True),
    *_cohorts("gfx950", "grid", (64, 128)),
    *_cohorts("gfx950", "kvseg", (64, 128)),
    *_cohorts("gfx950", "persist", (64, 128)),
    *_cohorts("gfx950", "widedma", (128,)),
)

# (layout, shape) launches of a cohort whose body serves strides. The cross-length
# shape X has no causal mask (causal with Sq != Skv is bottom-right on the factory).
POSITIVE_ROWS = (
    *(
        (layout, name)
        for layout in ("bshd", "bhsd", "sbhd", "batch_slice", "pad_token")
        for name in ("A", "B", "C")
    ),
    ("packed_qkv", "A"),  # GQA 8/2
    ("packed_qkv", "B"),  # MHA 8/8
    ("packed_qkv", "C"),  # GQA 16/2, batch 1
    *(
        (layout, name)
        for layout in ("o_differs", "all_distinct")
        for name in ("A", "B", "C")
    ),
    ("extent1", "E"),  # batch 1, one KV head: garbage on both
    ("extent1", "C"),  # batch 1: garbage on the batch strides
)
# Cross length, no mask only (see above); N has K/V of exactly one tile (S = block_n).
# gfx950 only: the gfx942 dense factory serves no cross-length request without a mask
# ("cross-length NO_MASK requests are not served by the dense kernel"), so a gfx942
# cohort plans none rather than skipping them (cross_rows_served).
CROSS_ROWS = (("bshd", "X"), ("bhsd", "X"), ("bshd", "N"), ("bhsd", "N"))


def cross_rows_served(cohort: Cohort) -> bool:
    """Whether the factory serves the cohort's CROSS_ROWS: no mask, and not gfx942."""
    return not cohort.causal and cohort.arch != "gfx942"


# The rows of a body that keeps packed addressing: packed launches only.
PACKED_ROWS = (
    ("bshd", "A"),
    ("bshd", "B"),
    ("bshd", "C"),
    ("extent1", "E"),
    ("extent1", "C"),
)


# --------------------------------------------------------------------------- #
# Inputs, reference, one launch with its checks
# --------------------------------------------------------------------------- #


def tolerance(dtype: str) -> float:
    return TOLERANCE[dtype]


@dataclass
class Logical:
    bits: dict  # role -> uint16 [B, S, H, D]
    o: np.ndarray  # fp64 [B, Sq, Hq, D]
    lse: np.ndarray  # fp64 [B, Hq, Sq]


def make_logical(shape: Shape, d: int, dtype: str, causal: bool) -> Logical:
    """Deterministic Q/K/V (as raw bits) and the fp64 reference of the decoded values."""
    seed = zlib.crc32(f"{shape.label(d)}:{dtype}".encode())
    rng = np.random.default_rng(seed)
    bits, f64 = {}, {}
    for role in "qkv":
        b, h, s, dd = shape.dims(role, d)
        bits[role] = encode(rng.standard_normal((b, s, h, dd)), dtype)
        f64[role] = decode(bits[role], dtype).transpose(0, 2, 1, 3)
    o, lse = sdpa_reference(
        f64["q"], f64["k"], f64["v"], scale=1.0 / math.sqrt(d), causal=causal
    )
    return Logical(bits, o.transpose(0, 2, 1, 3), lse)


class Problem:
    """One case materialized on a backend: buffers, views and snapshots."""

    def __init__(self, backend, case: Case, shape: Shape, spec, logical: Logical):
        d, dtype = spec.head_size, spec.dtype
        need = {}
        for role in ROLES:
            dims = shape.dims(role, d)
            for place in (case.places[role], case.data_place(role)):
                need[place.buf] = max(need.get(place.buf, 0), place.need(dims))
        hosts = {
            name: np.full(GUARD + n + GUARD, POISON_BITS[dtype], dtype=np.uint16)
            for name, n in need.items()
        }
        for role in "qkv":
            place = case.data_place(role)
            place_window(hosts[place.buf], place, shape.dims(role, d), True)[...] = (
                logical.bits[role]
            )
        self.buffers = {name: Buffer(backend, host) for name, host in hosts.items()}
        self.views = {
            role: View(
                self.buffers[case.places[role].buf],
                case.places[role],
                shape.dims(role, d),
            )
            for role in ROLES
        }
        self.o_buffer = self.buffers[case.places["o"].buf]
        self.input_names = sorted(
            {case.places[r].buf for r in "qkv"}
            | {case.data_place(r).buf for r in "qkv"}
        )
        self.snapshots = {n: self.buffers[n].host.copy() for n in self.input_names}
        self.lse = None
        if spec.emit_lse:
            self.lse_numel = shape.b * shape.hq * shape.sq
            host = np.full(
                GUARD + self.lse_numel + GUARD, LSE_POISON_BITS, dtype=np.uint32
            ).view(np.float32)
            self.lse_buf = Buffer(backend, host)
            self.lse = LseView(self.lse_buf, (shape.b, shape.hq, shape.sq, 1))


@dataclass
class Row:
    cohort: str
    layout: str
    shape: str
    expect: str
    status: str = ""  # pass | FAIL | refused | caught | ran
    ok: bool = False
    problems: list = dataclasses.field(default_factory=list)
    strides: dict | None = None  # role -> (batch, head, token) as the launch was told
    max_abs: float | None = None
    error: str = ""

    @property
    def id(self) -> str:
        return f"{self.layout}/{self.shape}"


def _bits_of(buf: np.ndarray, view: View) -> np.ndarray:
    return np.ascontiguousarray(view.window(buf))


def check_launch(prob: Problem, shape: Shape, spec, logical: Logical, row: Row) -> None:
    """The checks after one launch: the O view within tolerance of the fp64 reference,
    every element of the O buffer outside the view still the poison (bit compare), every
    input buffer bit-identical, and with LSE the same for the LSE buffer."""
    dtype = spec.dtype
    o_view = prob.views["o"]
    after = {name: buf.fetch() for name, buf in prob.buffers.items()}
    o_after = after[o_view.place.buf]
    got = decode(_bits_of(o_after, o_view), dtype)
    nonfinite = int((~np.isfinite(got)).sum())
    row.max_abs = float(np.abs(got - logical.o).max()) if not nonfinite else math.inf
    if nonfinite:
        row.problems.append(f"{nonfinite} non-finite elements in the O view")
    elif not row.max_abs < tolerance(dtype):
        row.problems.append(f"O max_abs {row.max_abs:.3e} >= {tolerance(dtype)}")
    outside = ~o_view.footprint()
    changed = int((o_after[outside] != POISON_BITS[dtype]).sum())
    if changed:
        row.problems.append(
            f"{changed} O-buffer elements outside the view were written"
        )
    for name in prob.input_names:
        if not np.array_equal(after[name], prob.snapshots[name]):
            row.problems.append(f"input buffer {name} changed")
    if prob.lse is not None:
        lse_after = prob.lse_buf.fetch()
        n = prob.lse_numel
        raw = lse_after.view(np.uint32)
        guards = np.concatenate([raw[:GUARD], raw[GUARD + n :]])
        if (guards != LSE_POISON_BITS).any():
            row.problems.append("LSE guard band was written")
        got_lse = lse_after[GUARD : GUARD + n].reshape(shape.b, shape.hq, shape.sq)
        if not np.isfinite(got_lse).all():
            row.problems.append("non-finite LSE")
        else:
            err = float(np.abs(got_lse.astype(np.float64) - logical.lse).max())
            if not err < tolerance(dtype):
                row.problems.append(f"LSE max_abs {err:.3e} >= {tolerance(dtype)}")


def run_case(
    backend, cohort: Cohort, case: Case, shape: Shape, spec, logical: Logical, kmod
) -> Row:
    """Materialize ``case``, launch it and classify the outcome against ``case.expect``."""
    row = Row(cohort.id, case.layout, shape.name, case.expect)
    prob = Problem(backend, case, shape, spec, logical)
    views = dict(prob.views)
    if VIEW_HOOK is not None:
        views = VIEW_HOOK(views)
    row.strides = {r: tuple(views[r].stride()[i] for i in (0, 2, 1)) for r in ROLES}
    scale = 1.0 / math.sqrt(spec.head_size)
    try:
        backend.launch(
            kmod,
            spec,
            cohort.arch,
            views["q"],
            views["k"],
            views["v"],
            views["o"],
            scale,
            prob.lse,
        )
    except ValueError as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        if case.expect != "refuse":
            row.status, row.ok = "FAIL", False
            row.problems.append(f"unexpected refusal: {exc}")
            return row
        if case.reason and case.reason not in str(exc):
            row.problems.append(f"refused for another reason: {exc}")
        if not np.array_equal(prob.o_buffer.fetch(), prob.o_buffer.host):
            row.problems.append("a refused launch changed the O buffer")
        row.status = "refused" if not row.problems else "FAIL"
        row.ok = not row.problems
        return row
    if case.expect == "refuse":
        row.status, row.ok = "ran", False
        row.problems.append("expected a refusal, the launch ran")
        return row
    check_launch(prob, shape, spec, logical, row)
    if case.expect == "fail":
        row.ok = bool(row.problems)
        row.status = "FAIL" if row.problems else "pass"
        if not row.problems:
            row.problems.append(
                "negative control PASSED: wrong strides went undetected"
            )
        else:
            row.problems = []  # caught, as it must be
            row.status = "caught"
    else:
        row.status = "FAIL" if row.problems else "pass"
        row.ok = not row.problems
    return row


@dataclass
class CohortReport:
    cohort: Cohort
    rows: list
    problems: list
    served: bool
    seconds: float = 0.0
    compiles: int | None = None  # dense kernels compiled while the cohort ran
    compiler: str = ""  # the loaded compiler, e.g. "llvm 22.0.0" ("" on a host run)

    @property
    def ok(self) -> bool:
        return not self.problems

    def counts(self) -> dict:
        out = {}
        for row in self.rows:
            out[row.status] = out.get(row.status, 0) + 1
        return out

    def text(self) -> str:
        head = (
            f"{self.cohort.id}: served={self.served} rows={len(self.rows)} "
            f"{self.counts()} compiles={self.compiles} {self.compiler} "
            f"{self.seconds:.1f}s"
        )
        return "\n".join([head, *(f"  - {p}" for p in self.problems)])


def _plan(cohort: Cohort, served: bool, kvseg: bool) -> list:
    """The ``(case, shape)`` launches of a cohort, in order."""
    d = cohort.d
    plan = []
    if served:
        rows = POSITIVE_ROWS + (CROSS_ROWS if cross_rows_served(cohort) else ())
        for layout, name in rows:
            case = make_case(layout, SHAPES[name], d, o_packed=kvseg)
            if case is not None:
                plan.append((case, SHAPES[name]))
        shape = SHAPES["A"]
        for role in ROLES:
            if kvseg and role == "o":
                continue  # the combine kernel writes packed O: nothing to lie about
            plan.append((negative_case(role, shape, d, o_packed=kvseg), shape))
        if cohort.slots:
            for role in ROLES:
                if kvseg and role == "o":
                    continue
                for axis in range(3):
                    plan.append(
                        (slot_case(role, axis, shape, d, o_packed=kvseg), shape)
                    )
    else:
        for layout, name in PACKED_ROWS:
            plan.append((make_case(layout, SHAPES[name], d), SHAPES[name]))
    shape = SHAPES["A"]
    for case in refusal_cases(shape, d, served=served, kv_split=kvseg):
        plan.append((case, shape))
    return plan


def run_cohort(backend, cohort: Cohort, *, compile_log: list | None = None):
    """Run every launch of ``cohort`` and the cohort-level checks. ``compile_log`` is a
    list the caller appends the name of every kernel compiled while this runs (None:
    the compile count is not checked)."""
    import time

    from kernels.common.attention_dense_spec import attention_dense_cache_key

    t0 = time.perf_counter()
    kmod = kernel_module(cohort.arch)
    d, dtype = cohort.d, cohort.dtype
    first = make_spec(
        cohort.arch, cohort.body, SHAPES["A"], d, dtype, cohort.causal, cohort.lse
    )
    ok, why = kmod.supports_attention_dense(first, arch=cohort.arch)
    if not ok:
        raise AssertionError(f"{cohort.id}: the factory spec is unsupported: {why}")
    served = bool(kmod.body_serves_strides(first))
    kvseg = first.kv_segments > 1
    specs, logicals, rows, problems = {}, {}, [], []
    n_compiles_before = len(compile_log) if compile_log is not None else 0
    passing = []  # (cache key, shape, strides) of the passing positive launches
    for case, shape in _plan(cohort, served, kvseg):
        if shape.name not in specs:
            try:
                specs[shape.name] = make_spec(
                    cohort.arch, cohort.body, shape, d, dtype, cohort.causal, cohort.lse
                )
            except ValueError as exc:  # the factory has no kernel for this shape
                specs[shape.name] = exc
        spec = specs[shape.name]
        if isinstance(spec, ValueError):
            # Every planned shape must have a kernel: a refusal here would drop a row
            # silently, so it is a problem of the cohort, not a skip.
            row = Row(cohort.id, case.layout, shape.name, case.expect, "FAIL", False)
            row.error = f"factory refuses the shape: {spec}"
            rows.append(row)
            problems.append(f"{row.id} [{case.layout}]: {row.error}")
            continue
        if shape.name not in logicals:
            logicals[shape.name] = make_logical(shape, d, dtype, cohort.causal)
        row = run_case(backend, cohort, case, shape, spec, logicals[shape.name], kmod)
        rows.append(row)
        if not row.ok:
            problems.append(
                f"{row.id} [{case.layout}]: " + "; ".join(row.problems or [row.status])
            )
        elif case.expect == "pass":
            passing.append(
                (
                    attention_dense_cache_key(spec, arch=cohort.arch),
                    (shape.b, shape.sq, shape.skv, shape.hq, shape.hkv),
                    tuple(sorted((r, v) for r, v in row.strides.items())),
                )
            )
    # One compiled kernel served every launch, and the launches differ.
    keys = {k for k, _, _ in passing}
    if len(keys) > 1:
        problems.append(f"the launches resolve to {len(keys)} cache keys, not one")
    n = None
    if compile_log is not None:
        names = {
            sp.kernel_name() for sp in specs.values() if not isinstance(sp, ValueError)
        }
        n = sum(1 for name in compile_log[n_compiles_before:] if name in names)
        if n > 1:
            problems.append(f"the dense kernel was compiled {n} times, not once")
    if len(passing) < 3:
        problems.append(f"only {len(passing)} passing launches (need 3 on one binary)")
    else:
        for what, pick in (
            ("batch", lambda s: s[0]),
            ("lengths", lambda s: (s[1], s[2])),
            ("head counts", lambda s: (s[3], s[4])),
        ):
            if len({pick(shp) for _, shp, _ in passing}) < 2:
                problems.append(f"the launches on one binary do not differ in {what}")
        if len({st for _, _, st in passing}) < 2:
            problems.append("the launches on one binary do not differ in strides")
    return CohortReport(
        cohort,
        rows,
        problems,
        served,
        time.perf_counter() - t0,
        n,
        backend.compiler_label() if hasattr(backend, "compiler_label") else "",
    )
