# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""vLLM IR providers ``hipdnn`` for ``rms_norm`` and ``fused_add_rms_norm``.

Enable them with the IR op priority, for example::

    vllm serve ... --ir-op-priority '{"rms_norm":["hipdnn","vllm_c","native"],"fused_add_rms_norm":["hipdnn","vllm_c","native"]}'
    vllm serve ... --ir-op-priority.rms_norm hipdnn,vllm_c,native --ir-op-priority.fused_add_rms_norm hipdnn,vllm_c,native
    LLM(..., ir_op_priority={"rms_norm": ["hipdnn", "vllm_c", "native"], ...})

vLLM appends the platform defaults to a user list, so ``["hipdnn"]`` alone
also works.

Dispatch. ``supports_args`` accepts a signature only if hipDNN builds its plan
(probed once per signature, for a two-row input) and the args are what the op
takes: weight present, ``variance_size`` None, rows contiguous within the last
dim, fp16/bf16/fp32. Otherwise vLLM moves to the next provider; that is counted
as ``(op, fallback, reason)``, or raises in strict mode. Under torch.compile,
vLLM's lowering pass calls ``supports_args`` once per graph node with fake
tensors. The rows (dim 0) may be symbolic there; they never enter the decision.

The impls call ``torch.ops.hipdnn_vllm.rms_norm_rows`` /
``fused_add_rms_norm_rows``. Those run the dynamic-row interim, and a decline
at run time falls back inside the op (see csrc/graphs/norm_rows.cpp).
``fused_add_rms_norm`` is an add plus the hipDNN rmsnorm (counted ``PARTIAL``)
unless an engine takes the fused add→rmsnorm graph.

Exact plans are pre-built for vLLM's cudagraph capture sizes the first time a
signature runs eagerly. :func:`exact_rows_for_process` supplies the sizes.
"""

import logging
import threading
import time

import torch
from torch import Tensor

from vllm import ir

from .. import _bootstrap, stats
from ..config import load_config
from ..declines import Decline, DeclineError, parse_reason

__all__ = ["PROVIDER", "exact_rows_for_process", "fused_add_rms_norm", "rms_norm"]

logger = logging.getLogger("hipdnn_vllm")

PROVIDER = "hipdnn"

_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_config = load_config()
_STRICT = _config.strict


def _platform_ok() -> bool:
    import importlib.util

    from vllm.platforms import current_platform

    return current_platform.is_rocm() and importlib.util.find_spec("hipdnn_vllm._C") is not None


_PLATFORM_OK = _platform_ok()
RMS_SUPPORTED = _PLATFORM_OK and _config.op_enabled("rms_norm")
FUSED_SUPPORTED = _PLATFORM_OK and _config.op_enabled("fused_add_rms_norm")


def exact_rows_for_process() -> list[int] | None:
    """Row counts that get exact plans: HIPDNN_VLLM_NORM_EXACT_ROWS, else the
    cudagraph capture sizes of vLLM's current config. ``None`` while no vLLM
    config is current (the op asks again for its next new signature)."""
    if _config.norm_exact_rows is not None:
        return list(_config.norm_exact_rows)
    from vllm.config import get_current_vllm_config_or_none
    from vllm.config.compilation import CUDAGraphMode

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None
    cc = vllm_config.compilation_config
    if cc.cudagraph_mode is None or cc.cudagraph_mode == CUDAGraphMode.NONE:
        return []
    return sorted(set(cc.cudagraph_capture_sizes or ()))


# ---------------------------------------------------------------------------
# supports_args

_loaded = False
_probe_lock = threading.Lock()
_probes: dict[tuple, str] = {}


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        start = time.perf_counter()
        _bootstrap.load()
        loaded = time.perf_counter()
        from .. import _C

        # Create this device's hipDNN handle now (it loads the engine plugins),
        # so its cost shows apart from the first plan build.
        _C.loaded_plugin_paths(torch.cuda.current_device())
        _loaded = True
        logger.info(
            "hipdnn_vllm: runtime loaded in %.2f s; hipDNN handle and engine plugins in %.2f s",
            loaded - start,
            time.perf_counter() - loaded,
        )


def _static_ints(values) -> tuple[int, ...] | None:
    out = []
    for v in values:
        if type(v) is not int:  # a SymInt: not decidable without guarding
            return None
        out.append(v)
    return tuple(out)


def _signature(x: Tensor, weight: Tensor, epsilon: float) -> tuple | str:
    """Everything but the row count, or a decline reason."""
    if x.dtype not in _NORM_DTYPES or weight.dtype not in _NORM_DTYPES:
        return "UNSUPPORTED_FEATURE:dtype"
    if x.dim() < 2:
        return "UNSUPPORTED_FEATURE:rank"
    if weight.dim() != 1 or x.device != weight.device:
        return "UNSUPPORTED_FEATURE:weight_layout"
    rest = _static_ints(x.shape[1:])
    strides = _static_ints(x.stride())
    wstride = _static_ints(weight.stride())
    if rest is None or strides is None or wstride is None:
        return "UNSUPPORTED_FEATURE:symbolic_layout"
    if strides[-1] != 1 or wstride[0] != 1 or weight.shape[0] != rest[-1]:
        return "UNSUPPORTED_FEATURE:strided_rows"
    return (x.device.index, x.dtype, weight.dtype, rest, strides, float(epsilon))


def _probe(sig: tuple) -> str:
    """``""`` when hipDNN builds the two-row plan for ``sig``, else the reason.
    Two rows, not one, so leading dims that do not collapse into one row
    stride decline here as they would at run time. Real (tiny) device tensors,
    built outside any fake mode: the lowering pass calls supports_args under
    FakeTensorMode."""
    with _probe_lock:
        hit = _probes.get(sig)
    if hit is not None:
        return hit
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    _ensure_loaded()
    from .. import _C

    device, dtype, wdtype, rest, strides, eps = sig
    start = time.perf_counter()
    with unset_fake_temporarily():
        dev = torch.device("cuda", device if device is not None else torch.cuda.current_device())
        x = torch.empty_strided((2, *rest), strides, dtype=dtype, device=dev)
        w = torch.empty(rest[-1], dtype=wdtype, device=dev)
        reason = _C.rms_norm_probe(x, w, eps)
    logger.info(
        "hipdnn_vllm: norm probe %s rows=(2, %s) strides=%s eps=%g: %s (%.2f s)",
        dtype,
        rest,
        strides,
        eps,
        reason or "accepted",
        time.perf_counter() - start,
    )
    with _probe_lock:
        _probes[sig] = reason
    return reason


def _dynamo_tracing(x: Tensor) -> bool:
    # When vLLM disables the IR torch wrap and Dynamo traces dispatch() itself,
    # the probe (a pybind call) cannot be traced. vLLM's default compile path
    # (Inductor + lowering pass) calls supports_args outside Dynamo with
    # FakeTensors and is not affected.
    from torch._subclasses.fake_tensor import FakeTensor

    return torch.compiler.is_compiling() and not isinstance(x, FakeTensor)


def _decide(op: str, reason: str) -> bool:
    if not reason:
        return True
    stats.record(op, stats.ROUTE_FALLBACK, reason)
    if _STRICT:
        d = parse_reason(reason)
        raise DeclineError(op, Decline(d.stage, d.detail, "vLLM IR supports_args"))
    return False


def _rms_args_ok(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> bool:
    if _dynamo_tracing(x):
        return False
    if weight is None:
        return _decide("rms_norm", "UNSUPPORTED_FEATURE:no_weight")
    if variance_size is not None:
        return _decide("rms_norm", "UNSUPPORTED_FEATURE:variance_size")
    sig = _signature(x, weight, epsilon)
    if isinstance(sig, str):
        return _decide("rms_norm", sig)
    return _decide("rms_norm", _probe(sig))


def _fused_args_ok(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    if _dynamo_tracing(x):
        return False
    op = "fused_add_rms_norm"
    if weight is None:
        return _decide(op, "UNSUPPORTED_FEATURE:no_weight")
    if variance_size is not None:
        return _decide(op, "UNSUPPORTED_FEATURE:variance_size")
    if x_residual.dtype != x.dtype or x_residual.dim() != x.dim():
        return _decide(op, "UNSUPPORTED_FEATURE:residual_dtype")
    rstrides = _static_ints(x_residual.stride())
    if rstrides is None or rstrides[-1] != 1:
        return _decide(op, "UNSUPPORTED_FEATURE:strided_rows")
    sig = _signature(x, weight, epsilon)
    if isinstance(sig, str):
        return _decide(op, sig)
    # The decomposed path normalizes the contiguous sum; the fused graph (when
    # an engine takes it) is decided inside the op on first use.
    device, dtype, wdtype, rest, _, eps = sig
    return _decide(op, _probe((device, dtype, wdtype, rest, _contiguous_strides(rest), eps)))


def _contiguous_strides(rest: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides of ``(rows, *rest)``."""
    inner = []
    acc = 1
    for d in reversed(rest):
        inner.append(acc)
        acc *= d
    return (acc, *reversed(inner))


# ---------------------------------------------------------------------------
# Implementations (schemas must equal vllm/ir/ops/layernorm.py)


@ir.ops.rms_norm.register_impl(PROVIDER, supported=RMS_SUPPORTED, supports_args=_rms_args_ok)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    _ensure_loaded()
    return torch.ops.hipdnn_vllm.rms_norm_rows(x, weight, epsilon, _STRICT)


@ir.ops.fused_add_rms_norm.register_impl(
    PROVIDER, supported=FUSED_SUPPORTED, supports_args=_fused_args_ok
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    _ensure_loaded()
    return torch.ops.hipdnn_vllm.fused_add_rms_norm_rows(x, x_residual, weight, epsilon, _STRICT)
