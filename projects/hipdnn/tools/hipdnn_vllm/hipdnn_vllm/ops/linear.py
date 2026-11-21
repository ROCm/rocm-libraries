# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unquantized linear through hipDNN. Default off.

``HIPDNN_VLLM_OPS`` containing ``linear`` makes :func:`register` call
:func:`install_patch`, which replaces the name ``dispatch_unquantized_gemm`` in
``vllm.model_executor.layers.linear``. ``UnquantizedLinearMethod.__init__`` binds
``self._gemm_impl = dispatch_unquantized_gemm(linear_backend)`` when each layer
is constructed, so the patch must be in place before the model is built. vLLM
runs general plugins in ``init_worker`` (``vllm/v1/worker/worker_base.py``)
before the worker class is even resolved, so that holds for every worker.
``linear_all`` also patches the same name in ``vocab_parallel_embedding`` (the
LM head, ``UnquantizedEmbeddingMethod.apply``) and in ``fused_moe.runner.
moe_runner`` (the fused router gate in ``MoERunner._forward_impl``); both look
the name up on every call.

The replacement returns :func:`hipdnn_unquantized_gemm` wherever the original
dispatch returns ``rocm_unquantized_gemm``. That function calls the custom op
``torch.ops.hipdnn_vllm.unquantized_gemm``, which is opaque to torch.compile and
decides the route per call: hipDNN (``_C.linear_try``, the same code as
``torch.ops.hipdnn_vllm.linear``) or, on a decline, the original
``torch.ops.vllm.rocm_unquantized_gemm``. Declines are counted in ``_C``; strict
mode raises :class:`~hipdnn_vllm.declines.DeclineError` instead of falling back.

Every vLLM symbol the patch relies on is checked before anything is changed;
a mismatch raises :class:`LinearPatchError`, so a vLLM bump fails loudly rather
than silently routing nothing. ``HIPDNN_VLLM_OPS`` is part of the plugin's vLLM
compile-cache factor (``hipdnn_vllm.COMPILE_FACTOR_NAME``, added by
:func:`hipdnn_vllm.register`), so compiled artifacts saved with and without the
patch never load into each other.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from types import ModuleType
from typing import Any, Callable

import torch

from .. import _bootstrap
from ..config import Config
from ..declines import Decline, DeclineError, parse_reason

__all__ = [
    "OP",
    "SITES",
    "LinearPatchError",
    "hipdnn_unquantized_gemm",
    "install_patch",
    "linear",
    "linear_probe",
    "patch_state",
    "register_routed_op",
    "uninstall_patch",
]

logger = logging.getLogger("hipdnn_vllm")

OP = "linear"
ROUTED_OP = "unquantized_gemm"
_NAME = "dispatch_unquantized_gemm"
_UTILS = "vllm.model_executor.layers.utils"

# site -> (module, (class, function) whose code must reference the patched name
# or the bound impl). "linear" is installed for `linear`; all three for
# `linear_all`.
SITES: dict[str, tuple[str, tuple[str, str]]] = {
    "linear": ("vllm.model_executor.layers.linear", ("UnquantizedLinearMethod", "__init__")),
    "lm_head": (
        "vllm.model_executor.layers.vocab_parallel_embedding",
        ("UnquantizedEmbeddingMethod", "apply"),
    ),
    "moe_router_gate": (
        "vllm.model_executor.layers.fused_moe.runner.moe_runner",
        ("MoERunner", "_forward_impl"),
    ),
}
_LINEAR_SITES = ("linear",)
_LINEAR_ALL_SITES = ("linear", "lm_head", "moe_router_gate")


class LinearPatchError(RuntimeError):
    """The pinned vLLM symbols the linear patch replaces are not what it expects."""


_lock = threading.Lock()
_C: ModuleType | None = None
_strict = False
_lib: torch.library.Library | None = None
# Set by install_patch: the original vLLM callables.
_orig_dispatch: Callable[..., Any] | None = None
_orig_gemm: Callable[..., Any] | None = None
_installed: dict[str, ModuleType] = {}


# --- direct op wrappers -------------------------------------------------------


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ weight.T + bias`` through hipDNN. Raises on a decline."""
    _bootstrap.load()
    return torch.ops.hipdnn_vllm.linear(x, weight, bias)


def linear_probe(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> tuple[str, str]:
    """``("", engine)`` when hipDNN runs this signature, else ``(reason, "")``.
    Builds and caches the plan without executing it."""
    return _c().linear_probe(x, weight, bias)


def _c() -> ModuleType:
    global _C
    if _C is None:
        _bootstrap.load()
        from .. import _C as c

        _C = c
    return _C


# --- routed op ------------------------------------------------------------------


def _routed_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm import envs

    if envs.VLLM_BATCH_INVARIANT:
        # rocm_unquantized_gemm switches to vLLM's batch-invariant kernels.
        return _fallback(x, weight, bias, Decline.feature("batch_invariant"), count=True)
    out, reason, message = _c().linear_try(x, weight, bias)
    if out is not None:
        return out
    d = parse_reason(reason)
    return _fallback(x, weight, bias, Decline(d.stage, d.detail, message), count=False)


def _fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    decline: Decline,
    count: bool,
) -> torch.Tensor:
    if count:
        from .. import stats

        stats.record(OP, stats.ROUTE_FALLBACK, decline.reason)
    if _strict:
        raise DeclineError(OP, decline)
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


def _routed_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def register_routed_op() -> None:
    """Define ``torch.ops.hipdnn_vllm.unquantized_gemm``. Idempotent; no device
    work and no ``_C`` load (the op's first call loads it)."""
    global _lib
    if _lib is not None:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    lib = torch.library.Library("hipdnn_vllm", "FRAGMENT")  # noqa: TOR901
    direct_register_custom_op(
        op_name=ROUTED_OP,
        op_func=_routed_impl,
        fake_impl=_routed_fake,
        target_lib=lib,
    )
    _lib = lib


def hipdnn_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in for vLLM's ``rocm_unquantized_gemm(layer, x, weight, bias)``."""
    if not x.is_cuda:
        # Same device rule as rocm_unquantized_gemm: the ops are HIP-only.
        return _orig_gemm(layer, x, weight, bias)
    return torch.ops.hipdnn_vllm.unquantized_gemm(x, weight, bias)


def hipdnn_dispatch_unquantized_gemm(linear_backend: str = "auto") -> Callable[..., torch.Tensor]:
    """Replacement for ``vllm.model_executor.layers.utils.dispatch_unquantized_gemm``."""
    impl = _orig_dispatch(linear_backend)
    return hipdnn_unquantized_gemm if impl is _orig_gemm else impl


# --- install --------------------------------------------------------------------


def _params(fn: Callable[..., Any]) -> list[tuple[str, Any]]:
    return [(p.name, p.default) for p in inspect.signature(fn).parameters.values()]


def _fail(msg: str) -> None:
    raise LinearPatchError(
        f"hipdnn_vllm linear patch: {msg}. The patch is pinned to vLLM "
        "0.32.1rc1.dev117+gdf3bd094a; remove 'linear'/'linear_all' from HIPDNN_VLLM_OPS "
        "or update hipdnn_vllm/ops/linear.py for this vLLM."
    )


def _check_utils(utils: ModuleType) -> tuple[Callable[..., Any], Callable[..., Any]]:
    dispatch = getattr(utils, _NAME, None)
    gemm = getattr(utils, "rocm_unquantized_gemm", None)
    if dispatch is None or gemm is None:
        _fail(f"{_UTILS} has no {_NAME} / rocm_unquantized_gemm")
    if _params(dispatch) != [("linear_backend", "auto")]:
        _fail(
            f"{_UTILS}.{_NAME} signature is {inspect.signature(dispatch)}, "
            "expected (linear_backend='auto')"
        )
    empty = inspect.Parameter.empty
    if _params(gemm) != [("layer", empty), ("x", empty), ("weight", empty), ("bias", None)]:
        _fail(
            f"rocm_unquantized_gemm signature is {inspect.signature(gemm)}, "
            "expected (layer, x, weight, bias=None)"
        )
    if dispatch("auto") is not gemm:
        _fail(f"{_NAME}('auto') does not return rocm_unquantized_gemm on this platform")
    if not hasattr(torch.ops.vllm, "rocm_unquantized_gemm"):
        _fail("torch.ops.vllm.rocm_unquantized_gemm is not registered")
    return dispatch, gemm


def _check_site(site: str, mod: ModuleType, dispatch: Callable[..., Any]) -> None:
    current = getattr(mod, _NAME, None)
    if current is None:
        _fail(f"{mod.__name__} has no module-level {_NAME}")
    if current is not dispatch and current is not hipdnn_dispatch_unquantized_gemm:
        _fail(f"{mod.__name__}.{_NAME} is {current!r}, not vLLM's (patched by something else?)")
    cls_name, fn_name = SITES[site][1]
    cls = getattr(mod, cls_name, None)
    fn = getattr(cls, fn_name, None) if cls is not None else None
    code = getattr(fn, "__code__", None)
    if code is None or _NAME not in code.co_names:
        _fail(f"{mod.__name__}.{cls_name}.{fn_name} no longer calls {_NAME}")
    if site == "linear":
        apply = getattr(cls, "apply", None)
        if apply is None or "_gemm_impl" not in apply.__code__.co_names:
            _fail(f"{mod.__name__}.{cls_name}.apply no longer calls self._gemm_impl")


def install_patch(config: Config) -> list[str]:
    """Install the patch for the sites ``config`` enables. Returns the patched
    sites. Idempotent. All checks run before any module is modified."""
    global _orig_dispatch, _orig_gemm, _strict
    if config.op_enabled("linear_all"):
        sites = _LINEAR_ALL_SITES
    elif config.op_enabled("linear"):
        sites = _LINEAR_SITES
    else:
        return []
    with _lock:
        utils = importlib.import_module(_UTILS)
        dispatch, gemm = _check_utils(utils)
        mods = {site: importlib.import_module(SITES[site][0]) for site in sites}
        for site, mod in mods.items():
            _check_site(site, mod, dispatch)
        register_routed_op()
        _orig_dispatch = dispatch
        _orig_gemm = gemm
        _strict = config.strict
        for site, mod in mods.items():
            setattr(mod, _NAME, hipdnn_dispatch_unquantized_gemm)
            _installed[site] = mod
    logger.info("hipdnn_vllm: unquantized linear patched at %s", ", ".join(sites))
    return list(sites)


def uninstall_patch() -> None:
    """Restore vLLM's dispatch at every patched site (tests)."""
    with _lock:
        for mod in _installed.values():
            if getattr(mod, _NAME, None) is hipdnn_dispatch_unquantized_gemm:
                setattr(mod, _NAME, _orig_dispatch)
        _installed.clear()


def patch_state() -> dict[str, bool]:
    """Whether each site's module currently holds the hipDNN dispatch. A site
    whose module was never imported reports False."""
    import sys

    out = {}
    for site, (mod_name, _) in SITES.items():
        mod = sys.modules.get(mod_name)
        out[site] = (
            mod is not None and getattr(mod, _NAME, None) is hipdnn_dispatch_unquantized_gemm
        )
    return out
