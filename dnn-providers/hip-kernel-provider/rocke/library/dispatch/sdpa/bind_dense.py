# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SDPA admission and binding for the dense attention candidates.

Translates a :class:`~dispatch.sdpa.common.SdpaRequest` into the
``dispatch/attention/`` request the gfx942 / gfx950 dense candidates (grid,
persistent, persistent wide-DMA) already consume, and asks each of them
whether it serves it. The dense candidates are opt-in, so each is probed with
its own ``algorithm`` / ``spec_id`` pinned (:func:`opt_in_probe`). The existing
``dispatch/attention/`` candidates are not edited.

This module adds no shape, dtype or head-size rule of its own: "can it run"
is the candidates' own ``admits()``. What it adds is the guarantee that
nothing is admitted and then dropped: every feature in ``SDPA_FEATURES`` is
either carried (:data:`SDPA_DENSE_FEATURES`) or declined with a reason; a
scale outside the kernel's own range is declined, because an
``AttentionRequest`` does not carry the scale; and a candidate counts only
when the band in the spec it builds keeps the keys the request asks for.

:func:`sdpa_dense_bind` launches the first admitting candidate, in registry
order, through ``bind_dense_attention_torch``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from rocke.core.arch import known_arches
from rocke.dispatch.core import TorchBinding, opt_in_probe

from dispatch.attention import ATTENTION_ROUTE_REGISTRY, AttentionRequest
from dispatch.attention.candidate import (
    DENSE_ALGORITHM,
    DENSE_GRID_ALGORITHM,
    DENSE_PERSIST_ALGORITHM,
)
from dispatch.attention.common import Alignment, AttentionBand, _dense_kernel_module

from .bind_shared import (
    TORCH_DTYPES,
    Operand,
    asked_band,
    band_text,
    keeps_asked_keys,
    to_attention_request,
    value_rejects,
)

if TYPE_CHECKING:
    from rocke.dispatch.core import KernelCandidate

    from .common import SdpaRequest

__all__ = [
    "SDPA_DENSE_FEATURES",
    "sdpa_dense_admits",
    "sdpa_dense_bind",
    "sdpa_dense_candidates",
]

_DENSE_ALGORITHMS = frozenset(
    {DENSE_ALGORITHM, DENSE_GRID_ALGORITHM, DENSE_PERSIST_ALGORITHM}
)

# The binding's ``supports_features``: what the dense launch path carries.
# Layouts other than contiguous BSHD are carried by copying Q/K/V into, and O
# out of, contiguous ``[B, S, H, D]`` buffers.
SDPA_DENSE_FEATURES = frozenset(
    {
        "gqa",
        "layout_bshd",
        "layout_bhsd",
        "layout_strided",
        "kv_layout_differs",
        "o_layout_differs",
        "batch_stride",
        "d_stride",
        "unaligned",
        "causal_tl",
        "causal_br",
        "window",
        "window_br",
        "window_right",
        "lse",
        "mma_core_mode",
        "implementation_hint",
    }
)

# Every other feature, with the reason it is declined. Checked in this order,
# so the reported reason is deterministic.
_DECLINED = {
    "kv_heads_differ": "kv_heads_differ: the dense request has one nhead_k for K and V",
    "d_v_differs": "d_v_differs: the dense kernels take one head size for Q/K and V",
    "dtype_mixed": "dtype_mixed: the dense kernels read and write one dtype",
    "compute_dtype": "compute_dtype: the dense kernels accumulate in fp32 only",
    "scale_tensor": "scale_tensor: the dense kernels take only a scalar scale",
    "bias": "bias: no dense candidate declares additive-bias support",
    "bias_broadcast": "bias_broadcast: no dense candidate declares additive-bias support",
    "alibi": "alibi: no dense candidate declares alibi support",
    "sinks": "sinks: no reference oracle for sink tokens yet",
    "padded": "padded: the dense candidates take one sequence length per batch",
    "ragged": "ragged: the dense candidates take one sequence length per batch",
    "dropout": "dropout: the dense kernels have no dropout",
    "paged": "paged: the page-table format is not mapped yet (see SdpaPaged)",
    "block_mask": "block_mask: the dense kernels take no block mask",
    "fp8": "fp8: the dense kernels take no descale/scale tensors",
    "fp8_s_o": "fp8_s_o: the dense kernels take no softmax/output quantization",
    "max_sum_exp": "max_sum_exp: the dense kernels write only LSE, not max and sum-exp",
}


def _dense_candidates(arch: str) -> tuple[KernelCandidate, ...]:
    """The registry's dense candidates for ``arch``, in registry order."""
    return tuple(
        c
        for c in ATTENTION_ROUTE_REGISTRY.candidates()
        if c.algorithm in _DENSE_ALGORITHMS
        and c.capability is not None
        and arch in c.capability.arches
    )


def _spec_band(kernel_spec) -> AttentionBand:
    """The band a dense kernel spec computes: ``q - k < sliding_window`` and,
    when causal, ``k <= d + right_bound``, on the top-left diagonal unless the
    spec shifts plain causal bottom-right."""
    window = int(kernel_spec.sliding_window)
    causal = bool(kernel_spec.causal)
    return AttentionBand(
        left_bound=window if window > 0 else None,
        right_bound=int(getattr(kernel_spec, "right_bound", 0)) if causal else None,
        alignment=(
            Alignment.BOTTOM_RIGHT
            if getattr(kernel_spec, "causal_bottom_right", False)
            else Alignment.TOP_LEFT
        ),
    )


def _scale_reject(req: SdpaRequest) -> tuple[bool, str] | None:
    """The kernel's own scale range, where its module declares one."""
    module = _dense_kernel_module(req.arch)
    low = getattr(module, "_MIN_SCALE", None)
    high = getattr(module, "_MAX_SCALE", None)
    scale = float(req.scale)
    if not math.isfinite(scale):
        return False, f"scale: the dense kernels need a finite scale, got {scale}"
    if low is not None and high is not None and not low <= scale <= high:
        return False, (
            f"scale: the {req.arch} dense kernels take a scale in "
            f"[{low}, {high}], got {scale}"
        )
    return None


def _sdpa_dense_rejects(req: SdpaRequest) -> tuple[bool, str] | None:
    """Return ``(False, reason)`` if this request cannot be served, else ``None``.

    Everything except the candidates' own ``admits()`` and the band of the
    spec each builds; see :func:`_dense_served`.
    """
    if req.arch not in known_arches():
        return False, f"unknown arch {req.arch!r}"
    if not _dense_candidates(req.arch):
        return False, f"no dense candidate for arch {req.arch!r}"

    features = req.features()
    for feature, reason in _DECLINED.items():
        if feature in features:
            return False, reason

    rejected = value_rejects(req, "dense")
    if rejected is not None:
        return rejected
    return _scale_reject(req)


def _dense_served(
    req: SdpaRequest,
) -> tuple[tuple[tuple[KernelCandidate, AttentionRequest, Any], ...], str]:
    """``((candidate, probe, spec), ...)`` for the dense candidates that serve
    ``req``, in registry order, and the reason when there are none.

    A candidate serves it when its own ``admits()`` accepts the pinned probe
    and the spec it then builds keeps the keys the request asks for.
    """
    rejected = _sdpa_dense_rejects(req)
    if rejected is not None:
        return (), rejected[1]
    attn = to_attention_request(req)
    served = []
    first_reason = ""
    for candidate in _dense_candidates(req.arch):
        probe = opt_in_probe(attn, candidate)
        ok, why = candidate.admits(probe)
        if ok:
            spec = candidate.select_spec(probe)
            band = _spec_band(spec.kernel_spec)
            if keeps_asked_keys(req, band):
                served.append((candidate, probe, spec))
                continue
            why = (
                f"band: {candidate.name} computes {band_text(band)} for "
                f"requested {band_text(asked_band(req))}"
            )
        first_reason = first_reason or f"{candidate.name}: {why}"
    if served:
        return tuple(served), "ok"
    return (), f"no dense candidate serves the request ({first_reason})"


def sdpa_dense_admits(req: SdpaRequest) -> tuple[bool, str]:
    """Top-level SDPA admission check for the dense family.

    Returns ``(True, "ok")`` only when nothing in :func:`_sdpa_dense_rejects`
    applies and at least one dense candidate serves the request.
    """
    served, reason = _dense_served(req)
    return (True, "ok") if served else (False, reason)


def sdpa_dense_candidates(req: SdpaRequest) -> tuple[str, ...]:
    """Names of the dense candidates that serve ``req``, for tests and triage.

    In registry order (the first is the one :func:`sdpa_dense_bind`
    launches); empty whenever :func:`sdpa_dense_admits` would return ``False``.
    """
    served, _ = _dense_served(req)
    return tuple(c.name for c, _, _ in served)


# --- binding -------------------------------------------------------------------


class _DensePlan:
    """The dense ABI for one bound request: contiguous ``[B, S, H, D]`` Q/K/V/O
    and a contiguous FP32 ``[B, H_q, S_q, 1]`` LSE.

    An operand already in that form is used in place; any other is copied
    through a buffer allocated once, at bind time (Q/K/V in before each
    launch, O and LSE out after).
    """

    def __init__(self, req: SdpaRequest, tensors: Mapping[str, Any]) -> None:
        import torch

        dtype = getattr(torch, TORCH_DTYPES[req.dtype])
        device = tensors["q"].device
        for name in ("q", "k", "v", "o"):
            t = tensors[name]
            if t.dtype != dtype or t.device != device:
                raise ValueError(
                    f"{name}: expected {dtype} on {device}, got {t.dtype} on {t.device}"
                )
        self.operands = {
            name: Operand(name, getattr(req, name), tensors[name])
            for name in ("q", "k", "v", "o")
        }
        self.views = {}
        self.copied = set()
        for name, op in self.operands.items():
            bshd = op.tensor.transpose(1, 2)
            if bshd.is_contiguous() and bshd.data_ptr() % 16 == 0:
                self.views[name] = bshd
            else:
                self.views[name] = torch.empty(bshd.shape, dtype=dtype, device=device)
                self.copied.add(name)

        self.stats = None
        self.lse = None
        if req.stats is not None:
            self.stats = Operand("stats", req.stats, tensors["stats"]).tensor
            in_place = (
                self.stats.dtype == torch.float32
                and self.stats.device == device
                and self.stats.is_contiguous()
            )
            self.lse = (
                self.stats
                if in_place
                else torch.empty(self.stats.shape, dtype=torch.float32, device=device)
            )

    def kernel_tensors(self) -> dict[str, Any]:
        out = {
            "q": self.views["q"],
            "k": self.views["k"],
            "v": self.views["v"],
            "out": self.views["o"],
        }
        if self.lse is not None:
            out["lse"] = self.lse
        return out

    def pack(self) -> None:
        for name in ("q", "k", "v"):
            if name in self.copied:
                self.views[name].copy_(self.operands[name].tensor.transpose(1, 2))

    def unpack(self) -> None:
        if "o" in self.copied:
            self.operands["o"].tensor.copy_(self.views["o"].transpose(1, 2))
        if self.lse is not None and self.lse is not self.stats:
            self.stats.copy_(self.lse)


def sdpa_dense_bind(req: SdpaRequest, tensors: Mapping[str, Any]) -> TorchBinding:
    """Bind an admitted request and its device tensors to a dense launch.

    ``tensors`` holds ``q``, ``k``, ``v``, ``o`` (each its logical BHSD tensor
    with the request's strides) and ``stats`` when the request sets it. The
    launch runs the first dense candidate that serves the request, at its
    default spec, on Torch's current stream, with the request's scale.
    Raises ``ValueError`` with the admission reason for a declined request.
    """
    served, reason = _dense_served(req)
    if not served:
        raise ValueError(reason)

    import torch

    from dispatch.attention.bindings import bind_dense_attention_torch

    _, probe, spec = served[0]
    plan = _DensePlan(req, tensors)
    binding = bind_dense_attention_torch(
        probe, spec, plan.kernel_tensors(), scale=float(req.scale)
    )
    device = plan.views["q"].device

    def launch(**_kw: Any) -> None:
        plan.pack()
        binding.launch(stream=int(torch.cuda.current_stream(device).cuda_stream))
        plan.unpack()

    return TorchBinding(launch=launch, grid=binding.grid, block=binding.block)
