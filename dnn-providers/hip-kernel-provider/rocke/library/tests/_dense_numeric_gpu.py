# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU helpers the per-arch dense numeric tests share, without torch.

* Operands: seeded random arrays stored in the kernel's element type and uploaded as
  :class:`~rocke.runtime.device_array.DeviceArray` (:func:`operand`, :func:`upload`,
  :func:`filled`, and :func:`guarded` with a trailing guard region), read back as
  float64 values (:func:`values`).
* Reference: :func:`dense_reference`, the float64 ``rocke.numeric.sdpa_reference``
  oracle in the runners' ``[B, S, H, D]`` layout, on exactly the stored values. It
  runs one head per worker thread and remembers each result by its inputs, so
  parametrized rows on the same seeded inputs compute it once.
* Comparisons: :func:`assert_close` and :func:`same_values` (equal elements). The
  error metric is ``rocke.numeric.compare.max_abs_error`` on :func:`values`; equal
  bit patterns are ``rocke.tests.ieee_compare.same_bits`` on
  ``rocke.runtime.device_array.host_values``.
* :class:`Alias`: a second view of another array's memory under another element
  type, for the launch checks that refuse overlapping arguments.
* fp8 K/V (:func:`fp8_kv`) and a paged cache (:func:`paged_cache`,
  :func:`run_dispatched_paged`).
"""

from __future__ import annotations

import functools
import hashlib
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
from rocke.core.storage import element_bytes
from rocke.numeric import storage
from rocke.numeric.fp8 import fp8e4m3_decode_table
from rocke.numeric.sdpa_reference import sdpa_reference
from rocke.runtime.device_array import (
    ArrayInfo,
    DeviceArray,
    array_info,
    dense_strides,
    host_values,
)

from ._dense_paged_stand_ins import pinned
from ._reference_threads import run_reference_pieces

#: The largest finite OCP e4m3 magnitude.
_FP8E4M3_MAX = float(np.nanmax(fp8e4m3_decode_table()))


def upload(host, dtype: str) -> DeviceArray:
    """Host values stored as ``dtype`` (:func:`rocke.numeric.storage.encode`) on the
    device."""
    return DeviceArray(storage.encode(host, dtype), dtype)


def operand(rng, shape, dtype: str, *, scale: float = 1.0) -> DeviceArray:
    """Standard normals times ``scale``, stored as ``dtype`` on the device."""
    return upload(rng.standard_normal(tuple(shape)) * scale, dtype)


def filled(shape, dtype: str, value: float = float("nan")) -> DeviceArray:
    """A ``dtype`` array of ``shape`` holding ``value`` everywhere; NaN by default, so
    an element a kernel never writes cannot pass as a stale zero."""
    return upload(np.full(tuple(shape), value), dtype)


def guarded(shape, dtype: str, fill: float, *, guard_elements: int = 16) -> DeviceArray:
    """A ``dtype`` array of ``shape`` filled with ``fill``, at the start of a buffer
    whose trailing whole rows (at least ``guard_elements`` elements) hold ``fill``
    too, so :func:`guard_intact` sees a write past the array's end."""
    shape = tuple(shape)
    row = int(np.prod(shape[1:], dtype=np.int64))
    guard_rows = -(-guard_elements // row)
    buffer = storage.encode(np.full((shape[0] + guard_rows, *shape[1:]), fill), dtype)
    return DeviceArray(buffer, dtype, shape=shape)


def guard_intact(array: DeviceArray, fill: float) -> bool:
    """Whether the guard rows past a :func:`guarded` array still hold ``fill``."""
    tail = storage.decode(array.read()[array.shape[0] :], array.dtype)
    return bool(np.all(tail == storage.round_to(fill, array.dtype)))


def values(array) -> np.ndarray:
    """The elements of a device array (or of a host numpy array) as float64."""
    if isinstance(array, np.ndarray):
        return array.astype(np.float64)
    return storage.decode(host_values(array), array_info(array).dtype).astype(
        np.float64
    )


def dense_reference(
    q,
    k,
    v,
    scale: float,
    *,
    causal: bool = True,
    right_bound: int = 0,
    window: int = 0,
    bottom_right: bool = False,
    sinks=None,
    q_offsets: Sequence[int] | None = None,
    kv_offsets: Sequence[int] | None = None,
):
    """Float64 ``(O, LSE)`` of a dense attention launch on exactly the stored values.

    ``q``/``k``/``v`` are ``[B, S, H, D]`` (or packed ``[T, H, D]`` with
    ``q_offsets`` / ``kv_offsets``) device or host arrays; GQA maps query head ``h``
    to KV head ``h // (Hq // Hkv)``. The band follows the dense spec's fields:
    ``causal`` keeps keys up to ``right_bound`` past the diagonal (``right_bound`` is
    ignored without it), ``window`` keeps the ``window`` keys ending at the diagonal
    (0 = unbounded), ``bottom_right`` moves the diagonal to ``s_kv - s_q``. ``sinks``
    (one logit per query head) joins every row's softmax with no value row.

    ``O`` comes back in the input layout; ``LSE`` (natural log) is ``[B, Hq, Sq, 1]``
    (``[T, Hq, 1]`` packed). A row with no visible key has ``O = 0`` and
    ``LSE = -inf``, or the sink logit with a sink.
    """
    band = {
        "left_bound": window - 1 if window else -1,
        "right_bound": right_bound if causal else -1,
        "top_left": not bottom_right,
    }
    stored = [_stored(x) for x in (q, k, v)]
    sink_values = None if sinks is None else values(sinks)
    key = _memo_key(stored, sink_values, scale, band, q_offsets, kv_offsets)
    if key not in _REFERENCE_MEMO:
        qv, kv, vv = (values(x) for x in (q, k, v))
        _remember(
            key, _reference(qv, kv, vv, scale, band, sink_values, q_offsets, kv_offsets)
        )
    _REFERENCE_MEMO.move_to_end(key)
    return _REFERENCE_MEMO[key]


#: Computed references by input digest, least recently used first:
#: parametrized rows and reruns on the same seeded inputs (both grids, every block
#: size of a paged cache) reuse one computation.
_REFERENCE_MEMO: OrderedDict = OrderedDict()

#: The memo's byte cap. The largest set of references the dense numeric tests reuse
#: is the six shapes every row of the gfx942 one-persistent-binary test runs, 540 MiB
#: of float64 O and LSE; 640 MiB holds that set with an 18% margin. A smaller cap
#: evicts the set between rows and doubles that file's reference time.
_REFERENCE_MEMO_BYTES = 640 << 20


def _stored(array):
    """``(host bits, element type)`` of a device array, or a host array as is."""
    if isinstance(array, np.ndarray):
        return array, array.dtype.str
    return host_values(array), array_info(array).dtype


def _memo_key(stored, sink_values, scale, band, q_offsets, kv_offsets):
    digest = hashlib.sha256()
    for array, dtype in stored:
        digest.update(repr((array.shape, dtype)).encode())
        digest.update(np.ascontiguousarray(array).data)
    if sink_values is not None:
        digest.update(sink_values.tobytes())
    offsets = tuple(
        None if x is None else tuple(int(i) for i in x) for x in (q_offsets, kv_offsets)
    )
    return digest.hexdigest(), float(scale), tuple(sorted(band.items())), offsets


def _remember(key, result):
    for array in result:
        array.setflags(write=False)  # shared by every caller of this key
    _REFERENCE_MEMO[key] = result
    total = sum(x.nbytes for pair in _REFERENCE_MEMO.values() for x in pair)
    while total > _REFERENCE_MEMO_BYTES and len(_REFERENCE_MEMO) > 1:
        _, dropped = _REFERENCE_MEMO.popitem(last=False)
        total -= sum(x.nbytes for x in dropped)


def _reference(q, k, v, scale, band, sink_values, q_offsets, kv_offsets):
    """``sdpa_reference`` of float64 ``[B, S, H, D]`` (or packed ``[T, H, D]``)
    values as ``(O, LSE[..., 1])`` in the input layout.

    One call per (batch, query head), with that head's K and V, on the reference
    threads (``_reference_threads``):
    ``sdpa_reference`` computes each (batch, head) on its own, so the split changes
    no value. K and V must have the same head count (the dense runners' layout), so
    one KV head index serves both."""
    packed = q_offsets is not None
    if not packed:  # [B, S, H, D] -> [B, H, S, D]; packed [T, H, D] keeps heads on 1
        q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
    num_q_heads, num_kv_heads = q.shape[1], k.shape[1]
    if v.shape[1] != num_kv_heads:
        raise ValueError(
            f"dense_reference failed - K has {num_kv_heads} heads and V "
            f"{v.shape[1]}, expected the same count"
        )
    group = num_q_heads // num_kv_heads
    batches = [slice(None)] if packed else [slice(b, b + 1) for b in range(q.shape[0])]
    o = np.zeros(q.shape[:-1] + v.shape[-1:])
    lse = np.full(q.shape[:-1], -np.inf)

    def one_head(batch, q_head):
        kv_head = q_head // group
        q_index = (batch, slice(q_head, q_head + 1))
        kv_index = (batch, slice(kv_head, kv_head + 1))
        o[q_index], lse[q_index] = sdpa_reference(
            q[q_index],
            k[kv_index],
            v[kv_index],
            scale=scale,
            sinks=None if sink_values is None else sink_values[q_head : q_head + 1],
            q_offsets=q_offsets,
            kv_offsets=kv_offsets,
            **band,
        )

    run_reference_pieces(
        functools.partial(one_head, batch, q_head)
        for batch in batches
        for q_head in range(num_q_heads)
    )
    if not packed:
        o = o.transpose(0, 2, 1, 3)
    return o, lse[..., None]


def first_tile_max(q, k, scale: float, block_m: int, block_n: int, *, causal: bool):
    """Each query head's largest scaled score over the first ``block_m x block_n``
    tile of batch 0 (top-left causal when ``causal``): the running max a sink seed
    meets first. ``[Hq]`` float64."""
    hq, hkv = q.shape[2], k.shape[2]
    qt = values(q)[0, :block_m].transpose(1, 0, 2)
    kt = np.repeat(values(k)[0, :block_n].transpose(1, 0, 2), hq // hkv, axis=0)
    scores = qt @ kt.transpose(0, 2, 1) * scale
    if causal:
        future = np.arange(block_n)[None, :] > np.arange(block_m)[:, None]
        scores = np.where(future, -np.inf, scores)
    return scores.max(axis=(1, 2))


def window64_analytic_lse(num_heads: int, sinks=None) -> np.ndarray:
    """Natural-log LSE of ``Sq = 256, Skv = 64``, top-left causal, window 64, with
    ``Q = 0`` (every visible score is 0): row ``q`` sees
    ``min(q + 1, 64) - max(q - 63, 0)`` keys (64 at row 63, 1 at row 126, none from
    row 127), each adding ``exp(0)``, plus ``exp(sink)`` with a sink.
    ``[1, num_heads, 256, 1]``; ``-inf`` where a row has no mass."""
    rows = np.arange(256)
    counts = np.clip(np.minimum(rows + 1, 64) - np.clip(rows - 63, 0, None), 0, None)
    assert (counts[63], counts[126], counts[127]) == (64, 1, 0)
    mass = np.broadcast_to(
        counts[None, None, :, None].astype(np.float64), (1, num_heads, 256, 1)
    )
    if sinks is not None:
        mass = mass + np.exp(values(sinks))[None, :, None, None]
    with np.errstate(divide="ignore"):
        return np.log(mass)


def assert_close(out, ref, *, atol: float, rtol: float, msg: str = "") -> None:
    """``|out - ref| <= atol + rtol * |ref|`` element-wise; a NaN fails."""
    np.testing.assert_allclose(
        values(out), values(ref), rtol=rtol, atol=atol, equal_nan=False, err_msg=msg
    )


def same_values(a, b) -> bool:
    """Same shape and equal elements (NaN equals nothing; +0 == -0)."""
    return bool(np.array_equal(values(a), values(b)))


class Alias:
    """``shape`` elements of another element type ``dtype`` laid over the start of
    ``base``'s memory, as torch's ``Tensor.view(dtype)`` reinterprets a storage: a
    second view of one allocation, which launch checks must refuse as overlapping.

    It exists because :meth:`DeviceArray.view` cannot reinterpret the element
    type: a view keeps its buffer's, and every overlap case pairs an fp32 LSE with
    a 16-bit or int32 array. So this stand-in only describes the memory: :meth:`values` returns ``read_values`` (what a check
    that reads the elements back sees), and it must never reach a launch.
    """

    def __init__(self, base, shape, dtype: str, *, read_values=None) -> None:
        under = array_info(base)
        self.shape = tuple(int(extent) for extent in shape)
        self.dtype = dtype
        self._info = ArrayInfo(
            shape=self.shape,
            strides=dense_strides(self.shape),
            dtype=dtype,
            element_size=element_bytes(dtype),
            data_ptr=under.data_ptr,
            device=under.device,
            backed_elements=under.backed_elements
            * under.element_size
            // element_bytes(dtype),
        )
        self._values = read_values

    def info(self) -> ArrayInfo:
        return self._info

    def values(self) -> np.ndarray:
        if self._values is None:
            raise ValueError("Alias.values failed - built without read_values")
        return np.asarray(self._values, dtype=storage.host_dtype(self.dtype))


def fp8_kv(kv_values, dtype: str):
    """Per-tensor OCP e4m3 quantization of K or V, as the fp8 dense kernels read it.

    Returns ``(codes, scale, dequantized)``: the e4m3 device array, the f32 scale
    the kernel multiplies each code by, and the dequantized values rounded to the
    compute ``dtype`` -- the K/V the kernel attends over, so a reference on them
    isolates kernel error from quantization loss.
    """
    fp32 = np.asarray(kv_values, dtype=np.float32)
    scale = np.float32(max(float(np.abs(fp32).max()), 1e-8)) / np.float32(_FP8E4M3_MAX)
    codes = storage.encode(
        np.clip(fp32 / scale, -_FP8E4M3_MAX, _FP8E4M3_MAX), "fp8e4m3"
    )
    dequantized = storage.round_to(storage.decode(codes, "fp8e4m3") * scale, dtype)
    return DeviceArray(codes, "fp8e4m3"), float(scale), dequantized


def paged_cache(rng, k, v, block_size):
    """Scatter one sequence's K/V (``[1, S, Hkv, D]`` device arrays) into pages at
    permuted ids of a cache twice the needed size whose unused pages are NaN, so a
    wrong page or in-page row reads a NaN page or the wrong keys rather than a near
    miss. Returns ``(k_cache, v_cache, block_tables, kv_lens)`` device arrays."""
    _, sk, hkv, d = k.shape
    npages = sk // block_size
    phys = rng.permutation(2 * npages)[:npages]
    caches = []
    for src in (k, v):
        cache = storage.encode(
            np.full((2 * npages, block_size, hkv, d), np.nan), src.dtype
        )
        cache[phys] = host_values(src)[0].reshape(npages, block_size, hkv, d)
        caches.append(DeviceArray(cache, src.dtype))
    block_tables = DeviceArray(phys.astype(np.int32).reshape(1, npages))
    kv_lens = DeviceArray(np.array([sk], dtype=np.int32))
    return (*caches, block_tables, kv_lens)


def run_dispatched_paged(arch, spec_id, *, window=1024):
    """Request -> dispatch -> bind -> launch of the paged spec ``spec_id`` on one
    fp16 D128 sequence, with the block count taken from the cache at bind.

    Returns ``(q, k, v, out)``; ``k``/``v`` are the contiguous arrays the cache was
    scattered from, for the caller's reference."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    sq, hq, hkv, d = 2048, 16, 4, 128
    request = AttentionRequest(
        batch=1,
        nhead_q=hq,
        nhead_k=hkv,
        seqlen_q=sq,
        seqlen_k=sq,
        hdim_q=d,
        hdim_v=d,
        arch=arch,
        mask_type=1,
        dtype="fp16",
        sliding_window=window,
        layout="paged",
        kv_block_size=16,
    )
    result = dispatch_attention(pinned(request, spec_id))
    rng = np.random.default_rng(0)
    q = operand(rng, (1, sq, hq, d), "fp16")
    k = operand(rng, (1, sq, hkv, d), "fp16")
    v = operand(rng, k.shape, "fp16")
    k_cache, v_cache, block_tables, kv_lens = paged_cache(rng, k, v, 16)
    out = filled(q.shape, "fp16")
    result.bind_torch(
        {
            "q": q,
            "k": k_cache,
            "v": v_cache,
            "out": out,
            "block_tables": block_tables,
            "kv_lens": kv_lens,
        }
    ).launch()
    return q, k, v, out
