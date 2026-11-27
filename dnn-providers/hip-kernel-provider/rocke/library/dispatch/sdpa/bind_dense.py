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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from rocke.core.arch import known_arches
from rocke.dispatch.core import TorchBinding, opt_in_probe
from rocke.runtime.device_array import (
    array_info,
    copy_elements,
    device_workspace,
    strided_view,
)

from dispatch.attention import (
    ATTENTION_ROUTE_REGISTRY,
    DENSE_ALGORITHMS,
    AttentionRequest,
    dense_scale_range,
)
from dispatch.attention.common import Alignment, AttentionBand

from .bind_shared import (
    ELEMENT_TYPES,
    Operand,
    asked_band,
    band_text,
    declined_table,
    dropout_launch_inputs,
    keeps_asked_keys,
    launch_stream,
    nonfinite_scale_reject,
    require_one_device,
    to_attention_request,
    usable_in_place,
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

# Named by the module that registers them (``dispatch.attention.DENSE_ALGORITHMS``),
# rather than duplicated here as literals, so a renamed or added dense algorithm
# breaks this import instead of silently leaving the gate unaware of it.
_DENSE_ALGORITHMS = DENSE_ALGORITHMS

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
        "noncausal_window",
        "lse",
        "dropout",
        "mma_core_mode",
        "implementation_hint",
    }
)

# Every other feature, with the reason it is declined (shared template in
# bind_shared.py, so the unified and dense wording cannot drift apart; a
# dense-only feature like d_v_differs/padded/ragged keeps its own text in
# ``overrides`` since unified does not decline it at all -- it carries it
# instead). Checked in this order, so the reported reason is deterministic.
_DECLINED = declined_table(
    "dense",
    (
        "kv_heads_differ",
        "d_v_differs",
        "dtype_mixed",
        "compute_dtype",
        "scale_tensor",
        "bias",
        "bias_broadcast",
        "alibi",
        "sinks",
        "padded",
        "ragged",
        "dropout_mask",
        "dropout_scale",
        "rng_dump",
        "paged",
        "block_mask",
        "fp8",
        "fp8_s_o",
        "max_sum_exp",
    ),
    overrides={
        "d_v_differs": "d_v_differs: the dense kernels take one head size for Q/K and V",
        "padded": "padded: the dense candidates take one sequence length per batch",
        "ragged": "ragged: the dense candidates take one sequence length per batch",
    },
)


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
    """The dense kernels' own scale range."""
    rejected = nonfinite_scale_reject(req, "dense")
    if rejected is not None:
        return rejected
    scale = float(req.scale)
    low, high = dense_scale_range(req.arch)
    if not low <= scale <= high:
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
    through a buffer of the caller's kind allocated once, at bind time (Q/K/V
    in before each launch, O and LSE out after), queued on the launch stream.
    """

    def __init__(self, req: SdpaRequest, tensors: Mapping[str, Any]) -> None:
        dtype = ELEMENT_TYPES[req.dtype]
        like = tensors["q"]
        require_one_device(tensors)
        self.operands = {
            name: Operand(name, getattr(req, name), tensors[name])
            for name in ("q", "k", "v", "o")
        }
        self.views = {}
        self.copied = set()
        for name, op in self.operands.items():
            bshd = op.bshd()
            if usable_in_place(bshd):
                self.views[name] = strided_view(op.array, bshd.shape, bshd.strides)
            else:
                self.views[name] = device_workspace(bshd.shape, dtype, like=like)
                self.copied.add(name)

        self.stats = None
        self.lse = None
        if req.stats is not None:
            # ``Operand`` refuses stats whose dtype is not the request's, so
            # the copy back from the FP32 LSE never narrows silently.
            self.stats = Operand("stats", req.stats, tensors["stats"]).array
            stats = array_info(self.stats)
            in_place = stats.dtype == "fp32" and usable_in_place(stats)
            self.lse = (
                self.stats
                if in_place
                else device_workspace(stats.shape, "fp32", like=like)
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

    def pack(self, stream: int = 0) -> None:
        """Copy the caller's Q, K and V into the copied views, queued on
        ``stream``."""
        for name in ("q", "k", "v"):
            if name in self.copied:
                copy_elements(
                    array_info(self.views[name]),
                    self.operands[name].bshd(),
                    stream=stream,
                )

    def unpack(self, stream: int = 0) -> None:
        """Copy O and LSE back to the caller's arrays, queued on ``stream``."""
        if "o" in self.copied:
            copy_elements(
                self.operands["o"].bshd(), array_info(self.views["o"]), stream=stream
            )
        if self.lse is not None and self.lse is not self.stats:
            copy_elements(array_info(self.stats), array_info(self.lse), stream=stream)


def sdpa_dense_bind(req: SdpaRequest, tensors: Mapping[str, Any]) -> TorchBinding:
    """Bind an admitted request and its device arrays to a dense launch.

    ``tensors`` holds ``q``, ``k``, ``v``, ``o`` (each its logical BHSD array
    with the request's strides) and ``stats`` when the request sets it: torch
    tensors or :class:`rocke.runtime.device_array.DeviceArray` s. The launch
    runs the first dense candidate that serves the request, at its default
    spec, on the current stream (Torch's, or HIP's null stream without
    torch), with the request's scale. Every array's element type must be the
    one its request descriptor names. A dropout request also takes
    ``dropout_seed`` and ``dropout_offset``, read at every launch (see
    :func:`~dispatch.sdpa.bind_shared.dropout_launch_inputs`). Raises
    ``ValueError`` with the admission reason for a declined request.
    ``launch(stream=...)`` accepts only the current stream; see :func:`~dispatch.sdpa.bind_shared.launch_stream`.
    """
    served, reason = _dense_served(req)
    if not served:
        raise ValueError(reason)

    from dispatch.attention.bindings import bind_dense_attention_torch

    _, probe, spec = served[0]
    # Refuses missing or malformed seed/offset before anything is allocated.
    dropout_launch_inputs(req, tensors)
    plan = _DensePlan(req, tensors)
    binding = bind_dense_attention_torch(
        probe, spec, plan.kernel_tensors(), scale=float(req.scale)
    )
    device = tensors["q"].device

    def launch(*, stream: Any = None) -> None:
        """Pack, launch and unpack, all on the current stream."""
        launch_on = launch_stream(stream, device, "dense")
        dropout = dropout_launch_inputs(req, tensors)
        plan.pack(launch_on)
        binding.launch(stream=launch_on, **dropout)
        plan.unpack(launch_on)

    return TorchBinding(launch=launch, grid=binding.grid, block=binding.block)
