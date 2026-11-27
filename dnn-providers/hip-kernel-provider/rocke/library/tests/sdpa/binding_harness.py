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
from rocke.runtime.device_array import array_info, device_values, host_values
from rocke.runtime.hip_module import Runtime
from rocke.tests.lanes import TICKET, skip_until

from tests.sdpa import helpers as H

__all__ = [
    "bind_and_check",
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
