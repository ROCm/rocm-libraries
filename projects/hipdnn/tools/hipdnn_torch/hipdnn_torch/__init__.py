# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch -- inject hipDNN into PyTorch by monkeypatching torch functionals.

Public API::

    import hipdnn_torch
    hipdnn_torch.enable_logging()          # optional: see each native fallback
    hipdnn_torch.install()                 # patch linear/rmsnorm/sdpa/layernorm/silu/gelu/conv2d/conv3d
    ...                                    # run your model
    print(hipdnn_torch.report())           # per-shape census + why calls fell back
    hipdnn_torch.uninstall()

Every intercepted call the engine cannot serve falls back to real PyTorch,
transparently, and is logged with the reason -- so the census/report is the
actionable "what is hipDNN still missing" list that drives future kernel work.

Importing this package does not import torch or touch the GPU; that happens lazily
on :func:`install` (or the first :func:`provider_ready`), which is also where a
misconfigured environment surfaces a clear :class:`~hipdnn_torch.bootstrap.BootstrapError`.
"""

import logging

from .activation import GeluOverride, SiluOverride
from .bootstrap import BootstrapError, bootstrap, is_bootstrapped
from .conv import Conv2dFpropOverride, Conv3dFpropOverride
from .layernorm import LayerNormOverride
from .linear import LinearOverride
from .rmsnorm import RmsNormOverride
from .sdpa import SdpaOverride

log = logging.getLogger("hipdnn_torch")
log.addHandler(logging.NullHandler())  # library default: no output unless asked

# One override instance per op, keyed by the name used in install(ops=...).
_OVERRIDES = {
    "linear": LinearOverride(),
    "rmsnorm": RmsNormOverride(),
    "sdpa": SdpaOverride(),
    "layernorm": LayerNormOverride(),
    "silu": SiluOverride(),
    "gelu": GeluOverride(),
    "conv2d": Conv2dFpropOverride(),
    "conv3d": Conv3dFpropOverride(),
}

_ALL = tuple(_OVERRIDES)

__all__ = [
    "install",
    "uninstall",
    "reset",
    "tuning",
    "tuning_report",
    "report",
    "census",
    "enable_logging",
    "provider_ready",
    "overrides",
    "BootstrapError",
    "bootstrap",
    "is_bootstrapped",
]


def _selected(ops):
    if ops is None:
        return _ALL
    unknown = [o for o in ops if o not in _OVERRIDES]
    if unknown:
        raise KeyError(f"unknown op(s) {unknown}; choose from {list(_ALL)}")
    return tuple(ops)


def install(ops=_ALL) -> None:
    """Patch the selected functionals (default: all of ``linear``, ``rmsnorm``,
    ``sdpa``, ``layernorm``, ``silu``, ``gelu``, ``conv2d``, ``conv3d``). Triggers the one-time
    bootstrap; raises :class:`BootstrapError` if the provider/backend/frontend
    cannot be discovered."""
    for name in _selected(ops):
        _OVERRIDES[name].install()


def uninstall(ops=None) -> None:
    """Restore the real functionals (default: everything installed)."""
    for name in _selected(ops):
        _OVERRIDES[name].uninstall()


def reset(ops=None) -> None:
    """Clear the intercept census + fallback tally."""
    for name in _selected(ops):
        _OVERRIDES[name].reset()


def census(ops=None) -> dict:
    """Return ``{op: {census_key: {"aot", "native", ...}}}``."""
    return {name: _OVERRIDES[name].census() for name in _selected(ops)}


def report(ops=None) -> str:
    """Human-readable per-op census + ranked fallback reasons. Ops with no
    intercepted calls are omitted."""
    chunks = []
    for name in _selected(ops):
        ov = _OVERRIDES[name]
        if ov.census() or ov.installed:
            chunks.append(ov.format_report())
    return "\n\n".join(chunks) if chunks else "hipdnn_torch: nothing installed"


def tuning(ops=None) -> list:
    """Every exhaustive sweep that actually ran, as a list of records.

    A record carries the op, the winning engine, how many candidates were
    benchmarked, the best ``robust_time_ms``, the top of the measured ranking,
    and -- the field that matters -- ``outcome``: the
    ``AutotuneCacheWriteOutcome`` name saying whether the ranking was persisted.
    Anything other than ``WRITTEN``/``UNCHANGED`` means later runs will re-decide
    from the heuristic rather than inherit this measurement.
    """
    out = []
    for name in _selected(ops):
        out.extend(_OVERRIDES[name]._tune_log)
    return out


def tuning_report(ops=None) -> str:
    """Human-readable summary of :func:`tuning`; empty string when nothing swept."""
    records = tuning(ops)
    if not records:
        return ""
    lines = ["exhaustive sweeps (torch.backends.cudnn.benchmark / HIPDNN_TORCH_TUNE):"]
    for rec in records:
        best = rec["best_ms"]
        lines.append(
            f"  {rec['op']:32s} winner={rec['winner']:24s} "
            f"benchmarked={rec['benchmarked']}/{rec['candidates']}  "
            f"best={best:.4f} ms  cache={rec['outcome']}"
            if best is not None else
            f"  {rec['op']:32s} no candidate succeeded  cache={rec['outcome']}"
        )
        for engine, ms in rec["ranking"][:4]:
            lines.append(f"      {engine:38s} {ms:8.4f} ms")
    return "\n".join(lines)


def overrides() -> dict:
    """The live override instances, keyed by op name (for advanced use)."""
    return dict(_OVERRIDES)


def enable_logging(level=logging.INFO) -> None:
    """Attach a simple stream handler to the ``hipdnn_torch`` logger so native
    fallbacks are printed. Call once; safe to call again (won't double-attach)."""
    for h in log.handlers:
        # NullHandler is not a StreamHandler, so the import-time null handler is
        # skipped; only a real stream handler we added earlier counts.
        if isinstance(h, logging.StreamHandler):
            log.setLevel(level)
            return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[hipdnn_torch] %(message)s"))
    log.addHandler(handler)
    log.setLevel(level)


def provider_ready() -> bool:
    """True if the environment bootstraps and a HIP device is available. Never
    raises: any failure (missing torch, unloadable backend, a HIP init error, or
    a BootstrapError naming an unset env var) simply returns False."""
    try:
        return bootstrap().torch.cuda.is_available()
    except Exception:  # noqa: BLE001 -- a probe must never raise; report False
        return False
