# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Case model, validity rules, input materialisation and oracle evaluation.

Host-only (numpy); ``to_torch`` imports torch lazily. The case table itself
lives in ``cases.py``.

Conventions
-----------
* Masks are a diagonal alignment plus ``left_bound`` / ``right_bound``
  (``-1`` = unbounded); ``causal_bool`` optionally selects the deprecated
  boolean form, which overrides bounds and alignment.
* ``length_mode``: ``fixed`` (every batch uses ``s_q``/``s_kv``), ``padded``
  (dense storage, per-batch counts ``q_lens``/``kv_lens``; ``s_q``/``s_kv`` are
  the maxima) or ``ragged`` (packed ``[T, H, D]`` tokens plus offsets).
* Fully masked rows (band, bias or padding) must produce ``O = 0`` and
  ``LSE = -inf``; rows past a batch's valid query count are padding and are
  never compared.
* LSE is the natural log of the softmax denominator.
* ``scale=None`` means the default ``1/sqrt(D)``.
* Bias is added after scaling; ``BiasSpec.shape`` is the broadcastable source
  shape and ``layout`` selects how the tensor handed to a kernel is strided.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided

from rocke.numeric.sdpa_reference import resolve_diagonal_band, sdpa_reference

CDNA_ARCHS = ("gfx942", "gfx950")
RDNA_ARCHS = ("gfx1151", "gfx1201", "gfx1250")
ALL_ARCHS = CDNA_ARCHS + RDNA_ARCHS
KNOWN_ARCHS = frozenset(ALL_ARCHS)

DTYPES = ("fp16", "bf16")
LAYOUTS = ("bhsd", "bshd", "strided_bhsd", "strided_bshd", "mixed", "packed_qkv", "thd")
LENGTH_MODES = ("fixed", "padded", "ragged")
DIAGONALS = ("top_left", "bottom_right")
BIAS_KINDS = ("dense", "mask_neginf", "neginf_rows")
BIAS_LAYOUTS = ("contiguous", "expanded", "row_padded")
TIERS = ("smoke", "full", "exhaustive")

BASE_SEED = 20240611

# Proposed comparison tolerances for O (abs, rel) and LSE (abs, rel). They are
# placeholders until an agreed tolerance replaces them.
TOLERANCE_O = {"fp16": (1e-2, 1e-2), "bf16": (2e-2, 2e-2)}
TOLERANCE_LSE = (1e-2, 1e-2)


@dataclass(frozen=True)
class BiasSpec:
    """Additive bias description.

    ``shape`` is the (right-aligned, broadcastable) source shape, rank 1..4.
    ``kind``: ``dense`` random values; ``mask_neginf`` -inf wherever
    ``(q + k) % 5 == 4`` and 0 elsewhere; ``neginf_rows`` whole rows
    ``q % 4 == 3`` are -inf, others random. The two -inf kinds need the last
    two dims to be the full ``(s_q, s_kv)`` extents. ``layout``: ``contiguous``;
    ``expanded`` (descriptor carries the full ``[B, H_q, S_q, S_kv]`` dims with
    stride 0 on broadcast axes); ``row_padded`` (row stride padded to a multiple
    of 8). ``dtype`` is ``fp32`` (the required case) or ``same`` as the query.
    """

    shape: tuple
    kind: str = "dense"
    layout: str = "contiguous"
    dtype: str = "fp32"


@dataclass(frozen=True)
class SdpaCase:
    id: str
    archs: tuple
    dtype: str
    b: int
    h_q: int
    h_k: int
    h_v: int
    s_q: int
    s_kv: int
    d: int
    layout: str = "bhsd"
    diagonal: str = "top_left"
    left_bound: int = -1
    right_bound: int = -1
    causal_bool: Optional[str] = None
    length_mode: str = "fixed"
    q_lens: Optional[tuple] = None
    kv_lens: Optional[tuple] = None
    scale: Optional[float] = None
    bias: Optional[BiasSpec] = None
    emit_lse: bool = False
    decode: bool = False
    fully_masked_rows: bool = False
    reqs: tuple = ()
    tags: frozenset = frozenset()

    @property
    def band(self):
        """Effective ``(left, right, top_left)`` after the causal-bool override."""
        return resolve_diagonal_band(
            left_bound=self.left_bound,
            right_bound=self.right_bound,
            top_left=self.diagonal == "top_left",
            causal=self.causal_bool == "top_left",
            causal_bottom_right=self.causal_bool == "bottom_right",
        )

    def lens(self):
        """Per-batch ``(q_lens, kv_lens)`` actually attended."""
        if self.length_mode == "fixed":
            return (self.s_q,) * self.b, (self.s_kv,) * self.b
        return tuple(self.q_lens), tuple(self.kv_lens)

    @property
    def resolved_scale(self):
        return 1.0 / math.sqrt(self.d) if self.scale is None else float(self.scale)

    @property
    def cost(self):
        """Rough oracle work: sum over batches of q_len * kv_len * heads * 2D."""
        lq, lk = self.lens()
        return sum(a * c for a, c in zip(lq, lk)) * self.h_q * 2 * self.d


# ---------------------------------------------------------------------------
# strides / layouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorDesc:
    """Addressing of one tensor: ``buffer`` key, element offset, dims, strides."""

    buffer: str
    offset: int
    dims: tuple
    strides: tuple


def _dense_strides(kind, B, H, S, D):
    """Element strides ``(B, H, S)``; the D stride is always 1."""
    if kind == "bhsd":
        return (H * S * D, S * D, D)
    if kind == "bshd":
        return (S * H * D, D, H * D)
    if kind == "strided_bhsd":
        sh = (S + 8) * D
        return (H * sh + 256, sh, D)
    if kind == "strided_bshd":
        ss = H * D + 64
        return (S * ss + 128, D, ss)
    raise ValueError(kind)


def _span(dims, strides):
    return 1 + sum((n - 1) * s for n, s in zip(dims, strides))


def layout_tensors(case):
    """Return ``(descs, buffer_sizes)`` for q, k, v, o of ``case``.

    ``o`` only has a descriptor and size; the consumer allocates it. For packed
    QKV the three inputs are views of one buffer keyed ``qkv``.
    """
    B, D = case.b, case.d
    heads = {"q": case.h_q, "k": case.h_k, "v": case.h_v, "o": case.h_q}
    seqs = {"q": case.s_q, "k": case.s_kv, "v": case.s_kv, "o": case.s_q}
    descs, sizes = {}, {}
    if case.layout == "thd":
        tq, tk = sum(case.q_lens), sum(case.kv_lens)
        toks = {"q": tq, "k": tk, "v": tk, "o": tq}
        for n in "qkvo":
            dims = (toks[n], heads[n], D)
            strides = (heads[n] * D, D, 1)
            descs[n] = TensorDesc(n, 0, dims, strides)
            sizes[n] = _span(dims, strides)
        return descs, sizes
    if case.layout == "packed_qkv":
        ht = case.h_q + case.h_k + case.h_v
        S = case.s_q
        st = (S * ht * D, D, ht * D, 1)
        offs = {"q": 0, "k": case.h_q * D, "v": (case.h_q + case.h_k) * D}
        for n in "qkv":
            descs[n] = TensorDesc("qkv", offs[n], (B, heads[n], S, D), st)
        sizes["qkv"] = B * S * ht * D
        st_o = _dense_strides("bshd", B, case.h_q, S, D) + (1,)
        descs["o"] = TensorDesc("o", 0, (B, case.h_q, S, D), st_o)
        sizes["o"] = _span(descs["o"].dims, st_o)
        return descs, sizes
    kinds = dict.fromkeys("qkvo", case.layout)
    if case.layout == "mixed":
        kinds = {"q": "bshd", "k": "bhsd", "v": "bhsd", "o": "bshd"}
    for n in "qkvo":
        st = _dense_strides(kinds[n], B, heads[n], seqs[n], D) + (1,)
        dims = (B, heads[n], seqs[n], D)
        descs[n] = TensorDesc(n, 0, dims, st)
        sizes[n] = _span(dims, st)
    return descs, sizes


def strides_overlap(dims, strides):
    """True if two distinct index tuples could address the same element."""
    extent = 1
    for s, n in sorted((s, n) for n, s in zip(dims, strides) if n > 1):
        if s < extent:
            return True
        extent += (n - 1) * s
    return False


def _bias_desc(case):
    """Descriptor and element count of the bias buffer for ``case``."""
    spec = case.bias
    src = (1,) * (4 - len(spec.shape)) + tuple(spec.shape)
    dense = [1] * 4
    acc = 1
    for i in (3, 2, 1, 0):
        dense[i] = acc
        acc *= src[i]
    if spec.layout == "expanded":
        full = (case.b, case.h_q, case.s_q, case.s_kv)
        strides = tuple(0 if src[i] == 1 else dense[i] for i in range(4))
        return TensorDesc("bias", 0, full, strides), acc
    if spec.layout == "row_padded":
        row = -(-src[3] // 8) * 8
        strides = (src[1] * src[2] * row, src[2] * row, row, 1)
        return TensorDesc("bias", 0, src, strides), src[0] * src[1] * src[2] * row
    return TensorDesc("bias", 0, src, tuple(dense)), acc


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------


def _bias_pattern_keep(spec, sq_idx, sk_idx):
    """Boolean keep matrix contributed by structural -inf bias kinds."""
    if spec is None or spec.kind == "dense":
        return None
    if spec.kind == "mask_neginf":
        return ((sq_idx[:, None] + sk_idx[None, :]) % 5) != 4
    return np.broadcast_to((sq_idx % 4 != 3)[:, None], (len(sq_idx), len(sk_idx)))


def has_fully_masked_rows(case):
    """Independent (non-oracle) computation of the fully-masked-row flag."""
    left, right, tl = case.band
    lq, lk = case.lens()
    for sq, skv in zip(lq, lk):
        qi = np.arange(sq)
        ki = np.arange(skv)
        off = 0 if tl else skv - sq
        m = np.ones((sq, skv), dtype=bool)
        if right >= 0:
            m &= ki[None, :] <= qi[:, None] + off + right
        if left >= 0:
            m &= ki[None, :] >= qi[:, None] + off - left
        pat = _bias_pattern_keep(case.bias, qi, ki)
        if pat is not None:
            m &= pat
        if not m.any(axis=1).all():
            return True
    return False


def validate_case(case):
    """Return a list of rule violations (empty means a valid graph)."""
    e = []
    if case.dtype not in DTYPES:
        e.append(f"dtype {case.dtype!r} not in {DTYPES}")
    if not case.archs or not set(case.archs) <= KNOWN_ARCHS:
        e.append(f"bad archs {case.archs!r}")
    if case.d == 256 and set(case.archs) & set(RDNA_ARCHS):
        e.append("d=256 is declared for CDNA only")
    if min(case.b, case.h_q, case.h_k, case.h_v, case.s_q, case.s_kv, case.d) < 1:
        return e + ["all extents must be positive"]
    if case.h_q % case.h_k or case.h_q % case.h_v:
        e.append("h_q must be a multiple of h_k and h_v")
    if case.d % 8:
        e.append("d must be a multiple of 8")
    if case.layout not in LAYOUTS:
        e.append(f"unknown layout {case.layout!r}")
    if case.diagonal not in DIAGONALS:
        e.append(f"unknown diagonal {case.diagonal!r}")
    if case.left_bound < -1 or case.right_bound < -1:
        e.append("bounds must be >= -1")
    if case.causal_bool not in (None, "top_left", "bottom_right"):
        e.append("bad causal_bool")
    if case.scale is not None and not (case.scale > 0 and math.isfinite(case.scale)):
        e.append("scale must be positive and finite")
    if case.length_mode not in LENGTH_MODES:
        return e + [f"unknown length_mode {case.length_mode!r}"]
    if case.length_mode == "fixed":
        if case.q_lens is not None or case.kv_lens is not None:
            e.append("fixed mode takes no per-batch lengths")
        if case.layout == "thd":
            e.append("thd layout needs ragged length mode")
    else:
        if case.q_lens is None or case.kv_lens is None:
            return e + ["padded/ragged need both q_lens and kv_lens"]
        if len(case.q_lens) != case.b or len(case.kv_lens) != case.b:
            e.append("one length per batch required")
        elif min(case.q_lens + case.kv_lens) < 1:
            e.append("lengths must be positive")
        elif max(case.q_lens) != case.s_q or max(case.kv_lens) != case.s_kv:
            e.append("s_q/s_kv must equal the longest per-batch length")
        if case.length_mode == "ragged" and case.layout != "thd":
            e.append("ragged needs thd layout")
        if case.length_mode == "padded" and case.layout == "thd":
            e.append("padded needs a dense layout")
    if case.layout == "packed_qkv" and (
        case.s_q != case.s_kv or case.length_mode == "ragged"
    ):
        e.append("packed qkv needs s_q == s_kv and a dense length mode")
    if not e:
        descs, _ = layout_tensors(case)
        for n, dsc in descs.items():
            if dsc.strides[-1] != 1:
                e.append(f"{n}: innermost stride must be 1")
            if strides_overlap(dsc.dims, dsc.strides):
                e.append(f"{n}: overlapping strides")
    if case.bias is not None:
        sp = case.bias
        if not 1 <= len(sp.shape) <= 4:
            e.append("bias rank must be 1..4")
        else:
            src = (1,) * (4 - len(sp.shape)) + tuple(sp.shape)
            if src[3] not in (case.s_kv, 1):
                e.append("bias last dim must be s_kv or 1")
            if src[2] not in (case.s_q, 1):
                e.append("bias second-to-last dim must be s_q or 1")
            if src[0] not in (case.b, 1) or src[1] not in (case.h_q, 1):
                e.append("bias batch/head dims must be B/H_q or 1")
            if sp.kind not in BIAS_KINDS:
                e.append(f"unknown bias kind {sp.kind!r}")
            elif sp.kind != "dense" and (src[2] != case.s_q or src[3] != case.s_kv):
                e.append("structural -inf bias kinds need full (s_q, s_kv) dims")
            if sp.layout not in BIAS_LAYOUTS:
                e.append(f"unknown bias layout {sp.layout!r}")
            if sp.dtype not in ("fp32", "same"):
                e.append("bias dtype must be fp32 or same")
    if not e:
        lq, _ = case.lens()
        if case.decode != all(n == 1 for n in lq):
            e.append("decode flag disagrees with query lengths")
        if case.fully_masked_rows != has_fully_masked_rows(case):
            e.append("fully_masked_rows flag disagrees with the mask")
    tiers = case.tags & set(TIERS)
    if not tiers:
        e.append("case has no tier tag")
    if "smoke" in tiers and "full" not in tiers:
        e.append("tier tags must be nested")
    if "full" in tiers and "exhaustive" not in tiers:
        e.append("tier tags must be nested")
    return e


# ---------------------------------------------------------------------------
# input materialisation
# ---------------------------------------------------------------------------


def case_seed(case, base=BASE_SEED):
    return (zlib.crc32(case.id.encode()) ^ base) & 0xFFFFFFFF


def round_to(x, dtype):
    """Round float values to fp16 / bf16 precision (result is float32)."""
    x = np.asarray(x, dtype=np.float32)
    if dtype == "fp16":
        return x.astype(np.float16).astype(np.float32)
    if dtype == "bf16":
        u = np.ascontiguousarray(x).view(np.uint32).astype(np.uint64)
        r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
        return np.where(np.isfinite(x), r.view(np.float32), x)
    raise ValueError(dtype)


@dataclass
class SdpaInputs:
    """Materialised problem: buffers, descriptors, lengths and logical values."""

    case: SdpaCase
    buffers: dict  # name -> flat float32 array already rounded to the case dtype
    descs: dict  # q, k, v, o -> TensorDesc
    sizes: dict
    bias_desc: Optional[TensorDesc]
    seq_len_q: Optional[np.ndarray]  # [B,1,1,1] int32 (padded mode)
    seq_len_kv: Optional[np.ndarray]
    q_offsets: Optional[np.ndarray]  # [B+1] token offsets (ragged mode)
    kv_offsets: Optional[np.ndarray]
    scale: float
    q: np.ndarray  # logical float64 values: dense [B,H,S,D] or ragged [T,H,D]
    k: np.ndarray
    v: np.ndarray
    bias: Optional[np.ndarray]  # source-shape float64


def _view(buf, desc):
    return as_strided(
        buf[desc.offset :],
        shape=desc.dims,
        strides=tuple(s * buf.itemsize for s in desc.strides),
        writeable=True,
    )


def _make_bias(spec, case, rng):
    src = tuple(spec.shape)
    full = (1,) * (4 - len(src)) + src
    if spec.kind == "dense":
        vals = rng.standard_normal(src).astype(np.float32) * 2.0
    else:
        qi = np.arange(full[2])[:, None]
        ki = np.arange(full[3])[None, :]
        if spec.kind == "mask_neginf":
            vals = np.zeros(full, dtype=np.float32)
            sel = ((qi + ki) % 5) == 4
        else:
            vals = rng.standard_normal(full).astype(np.float32)
            sel = np.broadcast_to((qi % 4) == 3, (full[2], full[3]))
        vals = np.where(np.broadcast_to(sel, vals.shape), -np.inf, vals).astype(
            np.float32
        )
        vals = vals.reshape(src)
    return round_to(vals, case.dtype) if spec.dtype == "same" else vals


def materialize(case, seed=None):
    """Build deterministic inputs for ``case`` (seed derived from its id).

    Storage outside the attended region (padding rows, stride gaps) holds finite
    garbage so a kernel that reads it is caught by the comparison. Buffers hold
    values already rounded to the case dtype.
    """
    rng = np.random.default_rng(case_seed(case) if seed is None else seed)
    descs, sizes = layout_tensors(case)
    buffers = {}
    for name, n in sizes.items():
        if name != "o":
            buffers[name] = round_to(rng.standard_normal(n) * 4.0, case.dtype)
    logical = {}
    for n in "qkv":
        dsc = descs[n]
        view = _view(buffers[dsc.buffer], dsc)
        view[...] = round_to(rng.standard_normal(dsc.dims), case.dtype)
        logical[n] = view.astype(np.float64)
    bias_src = bias_desc = None
    if case.bias is not None:
        bias_src = _make_bias(case.bias, case, rng)
        bias_desc, n_el = _bias_desc(case)
        buf = (rng.standard_normal(n_el) * 4.0).astype(np.float32)
        full = (1,) * (4 - bias_src.ndim) + bias_src.shape
        rows = bias_src.reshape(-1, full[3])
        if case.bias.layout == "row_padded":
            row = bias_desc.strides[2]
            buf.reshape(-1, row)[:, : full[3]] = rows
        else:
            buf[: rows.size] = rows.reshape(-1)
        buffers["bias"] = buf
    q_lens, kv_lens = case.lens()
    seq_q = seq_kv = qo = ko = None
    if case.length_mode == "padded":
        seq_q = np.array(q_lens, dtype=np.int32).reshape(-1, 1, 1, 1)
        seq_kv = np.array(kv_lens, dtype=np.int32).reshape(-1, 1, 1, 1)
    elif case.length_mode == "ragged":
        qo = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        ko = np.concatenate([[0], np.cumsum(kv_lens)]).astype(np.int32)
    return SdpaInputs(
        case=case,
        buffers=buffers,
        descs=descs,
        sizes=sizes,
        bias_desc=bias_desc,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        q_offsets=qo,
        kv_offsets=ko,
        scale=case.resolved_scale,
        q=logical["q"],
        k=logical["k"],
        v=logical["v"],
        bias=None if bias_src is None else bias_src.astype(np.float64),
    )


def lse_shape(case):
    if case.length_mode == "ragged":
        return (sum(case.q_lens), case.h_q)
    return (case.b, case.h_q, case.s_q)


def to_torch(inputs, device="cuda"):
    """Torch tensors (strided views of device buffers) for a kernel launch.

    Returns a dict with q, k, v, o (zero-filled), bias or None, the length
    tensors / offsets or None, the resolved scale and ``lse`` (zeros, or None
    when the case does not emit it). Imports torch lazily.
    """
    import torch

    case = inputs.case
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[case.dtype]
    dev = {
        n: torch.from_numpy(a).to(device=device, dtype=tdt)
        for n, a in inputs.buffers.items()
        if n != "bias"
    }
    out = {}
    for n in "qkv":
        dsc = inputs.descs[n]
        out[n] = torch.as_strided(dev[dsc.buffer], dsc.dims, dsc.strides, dsc.offset)
    od = inputs.descs["o"]
    obuf = torch.zeros(inputs.sizes["o"], dtype=tdt, device=device)
    out["o"] = torch.as_strided(obuf, od.dims, od.strides, od.offset)
    out["bias"] = None
    if inputs.bias_desc is not None:
        bdt = torch.float32 if case.bias.dtype == "fp32" else tdt
        bbuf = torch.from_numpy(inputs.buffers["bias"]).to(device=device, dtype=bdt)
        bd = inputs.bias_desc
        out["bias"] = torch.as_strided(bbuf, bd.dims, bd.strides, bd.offset)
    for name in ("seq_len_q", "seq_len_kv", "q_offsets", "kv_offsets"):
        a = getattr(inputs, name)
        out[name] = None if a is None else torch.from_numpy(a).to(device)
    out["scale"] = inputs.scale
    out["lse"] = (
        torch.zeros(lse_shape(case), dtype=torch.float32, device=device)
        if case.emit_lse
        else None
    )
    return out


# ---------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------


@dataclass
class SdpaReference:
    o: np.ndarray
    lse: np.ndarray
    valid_rows: np.ndarray  # bool, shaped like lse: True for real query rows


def evaluate(inputs):
    """Run the float64 oracle on materialised inputs."""
    case = inputs.case
    left, right, tl = case.band
    kw = dict(
        scale=inputs.scale,
        bias=inputs.bias,
        left_bound=left,
        right_bound=right,
        top_left=tl,
    )
    if case.length_mode == "ragged":
        o, lse = sdpa_reference(
            inputs.q,
            inputs.k,
            inputs.v,
            q_offsets=inputs.q_offsets,
            kv_offsets=inputs.kv_offsets,
            **kw,
        )
        valid = np.ones(lse.shape, dtype=bool)
    else:
        sl_q = sl_kv = None
        if case.length_mode == "padded":
            sl_q = inputs.seq_len_q.reshape(-1)
            sl_kv = inputs.seq_len_kv.reshape(-1)
        o, lse = sdpa_reference(
            inputs.q, inputs.k, inputs.v, seq_len_q=sl_q, seq_len_kv=sl_kv, **kw
        )
        lq, _ = case.lens()
        valid = np.arange(case.s_q)[None, None, :] < np.array(lq)[:, None, None]
        valid = np.broadcast_to(valid, lse.shape).copy()
    return SdpaReference(o=o, lse=lse, valid_rows=valid)


def compare(case, o, lse, ref, *, o_tol=None, lse_tol=None):
    """Compare kernel outputs to the oracle; returns a list of problems.

    ``o`` is shaped like the oracle ``O`` (``[B,H,S,D]`` or ``[T,H,D]``) and
    ``lse`` like the oracle LSE (None when the case does not emit it). Padding
    rows are ignored; no NaN/Inf may appear in valid rows of O; fully masked
    rows must match the oracle (zero output, -inf LSE).
    """
    probs = []
    ao, ro = o_tol or TOLERANCE_O[case.dtype]
    o = np.asarray(o, dtype=np.float64)
    if o.shape != ref.o.shape:
        return [f"O shape {o.shape} != {ref.o.shape}"]
    keep_rows = ref.valid_rows[..., None]
    og = np.where(keep_rows, o, 0.0)
    if not np.isfinite(og).all():
        probs.append("non-finite values in valid rows of O")
    else:
        want = np.where(keep_rows, ref.o, 0.0)
        err = np.abs(og - want)
        bad = err > ao + ro * np.abs(want)
        if bad.any():
            probs.append(
                f"O mismatch in {int(bad.sum())} elements (max abs err {err.max():.3g})"
            )
    if case.emit_lse:
        if lse is None:
            probs.append("LSE expected but not produced")
        else:
            lse = np.asarray(lse, dtype=np.float64)
            la, lr = lse_tol or TOLERANCE_LSE
            rows = ref.valid_rows
            if np.isnan(lse[rows]).any():
                probs.append("NaN in LSE")
            else:
                dead_w = np.isneginf(ref.lse) & rows
                dead_g = np.isneginf(lse) & rows
                if (dead_w != dead_g).any():
                    probs.append("fully-masked (-inf) LSE rows differ")
                live = rows & ~dead_w & ~dead_g
                if live.any():
                    diff = np.abs(lse[live] - ref.lse[live])
                    if (diff > la + lr * np.abs(ref.lse[live])).any():
                        probs.append("LSE mismatch")
    return probs
