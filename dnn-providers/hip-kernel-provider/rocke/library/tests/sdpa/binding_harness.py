# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Map ``cases.py`` harness arrays onto the SDPA bindings' tensor contract.

Shared by the CPU data-movement tests (torch CPU tensors) and the GPU numeric
tests (:class:`rocke.runtime.device_array.DeviceArray` s, and torch tensors for
a few cases, as the production torch callers pass): ``bind_tensors`` and
``outputs`` take either kind. Torch is imported lazily.
"""

from __future__ import annotations

import numpy as np
from rocke.numeric.storage import decode
from rocke.runtime.device_array import (
    DeviceArray,
    array_info,
    device_values,
    host_values,
)
from rocke.runtime.hip_module import Runtime
from rocke.tests.lanes import TICKET, skip_until

from tests.sdpa import helpers as H
from tests.sdpa.request_cases import make_tensors
from tests.sdpa.request_reference import sdpa_reference_from_request

__all__ = [
    "bind_and_check",
    "bind_and_check_request",
    "bind_tensors",
    "outputs",
    "strided_tensor",
    "torch_caller_cases",
]


def strided_tensor(desc, *, misaligned: bool = False):
    """A random FP16 CPU tensor with ``desc``'s dims and strides.

    ``misaligned`` starts it one element (2 bytes) into its allocation, so a
    contiguous view of it is not 16-byte aligned.
    """
    import torch

    total = 1 + sum((n - 1) * s for n, s in zip(desc.dims, desc.strides))
    buf = torch.randn(total + int(misaligned)).half()[int(misaligned) :]
    assert (buf.data_ptr() % 16 != 0) is misaligned
    return torch.as_strided(buf, desc.dims, desc.strides)


def bind_tensors(case, dev, req):
    """The binding's ``tensors`` for ``case``, from ``helpers.to_torch`` or
    ``helpers.to_device_arrays`` output.

    A ragged operand is its packed ``[T, H, D]`` buffer with token offsets
    (``request_from_case`` sets ``ragged_offset_multiplier = H * D``). Stats
    are allocated, zeroed, in the request's ``(B, H_q, S_q, 1)`` shape, of
    the same kind as Q.
    """
    tensors = {n: dev[n] for n in ("q", "k", "v", "o")}
    for key, src in (
        ("seq_len_q", "seq_len_q"),
        ("seq_len_kv", "seq_len_kv"),
        ("ragged_offset_q", "q_offsets"),
        ("ragged_offset_k", "kv_offsets"),
    ):
        if dev[src] is not None:
            tensors[key] = dev[src]
    if req.stats is not None:
        tensors["stats"] = device_values(
            np.zeros(req.stats.dims, np.float32), like=dev["q"]
        )
    return tensors


def _float64(array) -> np.ndarray:
    """``array``'s elements on the host as float64 (raw bf16 bits decoded)."""
    return decode(host_values(array), array_info(array).dtype).astype(np.float64)


def outputs(case, tensors, req):
    """O and LSE as host arrays shaped like the case oracle's."""
    o = _float64(tensors["o"])
    if req.stats is None:
        return o, None
    stats = _float64(tensors["stats"])[..., 0]
    if case.length_mode != "ragged":
        return o, stats
    q_lens, _ = case.lens()
    return o, np.concatenate([stats[b, :, :n].T for b, n in enumerate(q_lens)])


def torch_caller_cases(cases):
    """The cases the GPU numeric tests also run with torch callers: the first
    of each length mode, and the first that emits LSE."""
    picked = {}
    for case in cases:
        picked.setdefault(case.length_mode, case)
        if case.emit_lse:
            picked.setdefault("lse", case)
    return sorted(
        {case.id: case for case in picked.values()}.values(), key=lambda c: c.id
    )


def bind_and_check(bind, case, req, *, kind: str) -> list:
    """Bind ``case``'s arrays of ``kind`` (``"device_array"``, or ``"torch"``
    for torch GPU tensors: the caller has imported torch) with ``bind``,
    launch twice (the second launch reuses the bound buffers, as graph replay
    does) and return ``helpers.compare``'s mismatches with the oracle."""
    inputs = H.materialize(case)
    if kind == "torch":
        import torch

        if not torch.cuda.is_available():
            skip_until(TICKET, reason="the installed torch has no GPU")
        arrays = H.to_torch(inputs, device="cuda")
    else:
        arrays = H.to_device_arrays(inputs)
    tensors = bind_tensors(case, arrays, req)
    binding = bind(req, tensors)
    binding.launch()
    binding.launch()
    if kind == "torch":
        torch.cuda.synchronize()
    else:
        Runtime().sync()
    o, lse = outputs(case, tensors, req)
    return H.compare(case, o, lse, H.evaluate(inputs))


def _strided_device(values: np.ndarray, desc) -> DeviceArray:
    """Logical ``values`` in device memory laid out with ``desc``'s strides."""
    span = 1 + sum((n - 1) * s for n, s in zip(desc.dims, desc.strides))
    flat = np.zeros(span, values.dtype)
    np.lib.stride_tricks.as_strided(
        flat, desc.dims, [s * flat.itemsize for s in desc.strides]
    )[...] = values
    return DeviceArray(flat).view(desc.dims, desc.strides)


def _logical_o(req, o, q_offsets) -> np.ndarray:
    """O read back as ``(B, H_q, S_q, D_v)``; a ragged O is unpacked by the
    token offsets."""
    values = np.asarray(host_values(o), np.float64)
    if not req.o.ragged:
        return values
    batch, heads, extent, width = req.o.dims
    out = np.zeros((batch, heads, extent, width))
    flat = values.reshape(-1)
    for b in range(batch):
        start, end = int(q_offsets[b]), int(q_offsets[b + 1])
        rows = flat[start * heads * width : end * heads * width]
        out[b, :, : end - start] = rows.reshape(-1, heads, width).transpose(1, 0, 2)
    return out


def bind_and_check_request(bind, req, *, seed: int = 0, atol: float = 1e-2) -> None:
    """Bind a ``request_cases`` FP16 request on ``DeviceArray`` s built from
    :func:`make_tensors`, launch twice and compare O, and stats when the
    request has them, with :func:`sdpa_reference_from_request` on its valid
    rows.

    For requests the case table cannot express, such as seeded dropout: a
    by-value seed or offset is passed as an int, a device one as a
    one-element int64 array.
    """
    host = make_tensors(req, seed=seed)
    for name in ("q", "k", "v"):
        host[name] = host[name].astype(np.float16)
    tensors = {}
    for name in ("q", "k", "v"):
        desc = getattr(req, name)
        tensors[name] = (
            DeviceArray(host[name])
            if desc.ragged
            else _strided_device(host[name], desc)
        )
    if req.o.ragged:
        tokens = int(host["ragged_offset_q"][-1])
        tensors["o"] = DeviceArray(
            np.zeros(tokens * req.o.dims[1] * req.o.dims[3], np.float16)
        )
    else:
        tensors["o"] = _strided_device(np.zeros(req.o.dims, np.float16), req.o)
    for key in ("ragged_offset_q", "ragged_offset_k", "seq_len_q", "seq_len_kv"):
        if key in host:
            tensors[key] = DeviceArray(np.asarray(host[key], np.int32).reshape(-1))
    if req.stats is not None:
        tensors["stats"] = DeviceArray(np.zeros(req.stats.dims, np.float32))
    if req.dropout is not None:
        for name in ("seed", "offset"):
            key = f"dropout_{name}"
            by_value = getattr(req.dropout, name).by_value
            tensors[key] = (
                host[key] if by_value else DeviceArray(np.array([host[key]], np.int64))
            )
    binding = bind(req, tensors)
    binding.launch()
    binding.launch()
    Runtime().sync()
    want = sdpa_reference_from_request(req, host)
    valid = want["row_valid"]
    got_o = _logical_o(req, tensors["o"], host.get("ragged_offset_q"))
    np.testing.assert_allclose(
        got_o.transpose(0, 2, 1, 3)[valid],
        want["o"].transpose(0, 2, 1, 3)[valid],
        atol=atol,
        rtol=0,
    )
    if req.stats is not None:
        got = np.asarray(host_values(tensors["stats"]), np.float64)
        np.testing.assert_allclose(
            got.transpose(0, 2, 1, 3)[valid],
            want["stats"].transpose(0, 2, 1, 3)[valid],
            atol=atol,
            rtol=0,
        )
