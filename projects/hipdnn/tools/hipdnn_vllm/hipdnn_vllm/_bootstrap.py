# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Point ``_C`` at the bundled ``libhipdnn_backend``, then configure providers.

The wheel bundles its own hipDNN backend (``hipdnn_vllm/backend/<family>/``)
and every engine provider (``hipdnn_vllm/providers/<family>/``), built from the
hipDNN integration branch. torch's backend (``_rocm_sdk_libraries``) is an
older API that rejects paged graphs, so ``_C`` never uses it. ``_C`` is built
in the frontend's runtime-load mode and links no backend.

Order matters:

1. ``torch.cuda.init()`` brings HIP/HSA up before the backend probes devices.
2. ``_C`` is imported (registering ``torch.ops.hipdnn_vllm.*``) and told the
   bundled backend's directory before its first backend call; the frontend
   then ``dlopen``-s that file by explicit path, ``RTLD_LOCAL``. torch's copy
   may also be mapped (torch loads it on its own); ``_C`` never resolves
   symbols from it.
3. ``_C`` refuses to run if the backend it resolved differs from the one it
   was built with.
4. Engine plugin paths are set before the first handle (see :func:`plugin_plan`).

A source checkout without bundled files takes both from a staged hipDNN
install: ``HIPDNN_VLLM_HIPDNN_PREFIX`` (``lib/libhipdnn_backend.so``,
``lib/hipdnn_plugins/engines``).

Nothing here runs at import; :func:`load` is called by the first op use.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Config, load_config

logger = logging.getLogger("hipdnn_vllm")

__all__ = [
    "BootstrapError",
    "PluginPlan",
    "State",
    "bundled_backend_dir",
    "bundled_provider_dir",
    "build_info",
    "load",
    "mapped_backends",
    "plugin_plan",
]

_BACKEND_FILE = "libhipdnn_backend.so"

_PREFIX_ENV = "HIPDNN_VLLM_HIPDNN_PREFIX"
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


class BootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class PluginPlan:
    mode: str  # "additive" | "absolute" (resolved; never "auto")
    paths: tuple[str, ...]
    why: str


@dataclass
class State:
    backend_path: str
    backend_version: str
    plugin_plan: PluginPlan
    bundled_dir: str | None
    other_backends: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)


_lock = threading.Lock()
_state: State | None = None
# The error of a load() that failed after _C was pointed at a backend. That
# step cannot run twice in a process, so a retry re-raises the original error
# instead of a misleading "already resolved" one.
_failure: BaseException | None = None


def build_info() -> dict[str, Any]:
    """Contents of ``_build_info.json`` written by the wheel build (pins,
    family, provider commit). Empty for a source checkout."""
    path = os.path.join(_PKG_DIR, "_build_info.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _family_dir(kind: str) -> str | None:
    root = os.path.join(_PKG_DIR, kind)
    if not os.path.isdir(root):
        return None
    family = build_info().get("family")
    if family and os.path.isdir(os.path.join(root, family)):
        return os.path.join(root, family)
    subdirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if len(subdirs) > 1:
        raise BootstrapError(f"several bundled {kind} families in {root}: {subdirs}")
    return os.path.join(root, subdirs[0]) if subdirs else None


def _prefix() -> str | None:
    prefix = os.environ.get(_PREFIX_ENV)
    return os.path.abspath(prefix) if prefix else None


def bundled_backend_dir() -> str | None:
    """Directory holding the ``libhipdnn_backend.so`` ``_C`` loads: the wheel's
    ``backend/<family>/``, else ``$HIPDNN_VLLM_HIPDNN_PREFIX/lib``."""
    found = _family_dir("backend")
    if found is None and _prefix():
        found = os.path.join(_prefix(), "lib")
    return found


def bundled_provider_dir() -> str | None:
    """The wheel's ``providers/<family>/``, else
    ``$HIPDNN_VLLM_HIPDNN_PREFIX/lib/hipdnn_plugins/engines``."""
    found = _family_dir("providers")
    if found is None and _prefix():
        found = os.path.join(_prefix(), "lib", "hipdnn_plugins", "engines")
    return found


def _plugin_files(directory: str | None) -> list[str]:
    if not directory or not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.startswith("lib") and ".so" in name:
            out.append(path)
    return out


# The MIOpen provider serves only conv, and its handle initialization is costly
# and paid once per process, so it is loaded only when conv is enabled.
_MIOPEN_PLUGIN_PREFIX = "libmiopen"


def plugin_plan(config: Config, bundled_dir: str | None) -> PluginPlan:
    """Resolve HIPDNN_VLLM_PLUGIN_MODE / _PROVIDER_DIRS into backend arguments.

    The bundled backend only ever loads the bundled providers (built with it),
    never torch's stock providers. ``auto`` (default) and ``absolute`` load
    ABSOLUTELY: the bundled providers (the MIOpen provider only when ``conv``
    is enabled) plus HIPDNN_VLLM_PROVIDER_DIRS; ``absolute`` with
    HIPDNN_VLLM_PROVIDER_DIRS loads only those. ``additive`` passes the same
    list additively, on top of the backend's own default plugin directory.
    """
    bundled = [
        p
        for p in _plugin_files(bundled_dir)
        if config.op_enabled("conv") or not os.path.basename(p).startswith(_MIOPEN_PLUGIN_PREFIX)
    ]
    extra = list(config.provider_dirs)
    conv = "" if config.op_enabled("conv") else "; conv disabled, MIOpen provider left out"

    mode = config.plugin_mode
    if mode == "absolute" and extra:
        return PluginPlan("absolute", tuple(extra), "HIPDNN_VLLM_PROVIDER_DIRS (absolute)")
    if mode == "additive":
        return PluginPlan("additive", tuple(bundled + extra), "bundled providers, additive" + conv)
    return PluginPlan("absolute", tuple(bundled + extra), f"{mode}: bundled providers only (absolute){conv}")


def mapped_backends() -> list[str]:
    """Distinct real paths of every ``libhipdnn_backend`` mapped into this
    process (Linux ``/proc/self/maps``)."""
    maps = "/proc/self/maps"
    if not os.path.exists(maps):
        return []
    seen: dict[str, None] = {}
    with open(maps, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split(None, 5)
            if len(parts) == 6 and "libhipdnn_backend" in parts[5]:
                seen[os.path.realpath(parts[5].strip())] = None
    return list(seen)


def _bundled_backend_path() -> str:
    directory = bundled_backend_dir()
    path = os.path.join(directory, _BACKEND_FILE) if directory else None
    if not path or not os.path.isfile(path):
        raise BootstrapError(
            f"no bundled {_BACKEND_FILE}: this hipdnn-vllm install has no backend/<family>/ "
            f"directory and {_PREFIX_ENV} does not name a staged hipDNN install"
        )
    return os.path.realpath(path)


def load(config: Config | None = None) -> State:
    """Idempotent. Raises :class:`BootstrapError` (or the ``_C`` version-guard
    RuntimeError) with an actionable message on failure."""
    global _state, _failure
    with _lock:
        if _state is not None:
            return _state
        if _failure is not None:
            raise BootstrapError(
                f"an earlier hipdnn_vllm bootstrap in this process failed after binding the backend "
                f"and cannot be retried: {_failure}"
            ) from _failure
        config = config or load_config()

        import torch

        if not torch.cuda.is_available():
            raise BootstrapError("no HIP device visible to torch")
        torch.cuda.init()

        backend = _bundled_backend_path()

        from . import _C  # noqa: F401  (registers torch.ops.hipdnn_vllm.*)
        from . import fake_impls  # noqa: F401

        try:
            _state = _bind_and_configure(_C, config, backend)
        except BaseException as ex:
            _failure = ex
            raise
        return _state


def _bind_and_configure(_C: Any, config: Config, backend: str) -> State:
    import torch

    if not _C.set_backend_directory(os.path.dirname(backend)):
        raise BootstrapError(
            "_C resolved its hipDNN backend before the bootstrap could point it at the bundled copy"
        )
    _C.check_backend_version()

    # Our backend is mapped by explicit path; torch's may be mapped as well,
    # but every symbol _C resolves must come from ours.
    resolved = os.path.realpath(_C.backend_library_path())
    origin = _C.backend_symbol_origin("hipdnnCreate")
    mapped = mapped_backends()
    if resolved != backend or origin != backend or backend not in mapped:
        raise BootstrapError(
            f"_C must resolve only the bundled backend {backend}; it resolved {resolved}, "
            f"hipdnnCreate comes from {origin!r}, mapped copies: {mapped}"
        )

    bundled = bundled_provider_dir()
    plan = plugin_plan(config, bundled)
    if _C.runtime_created():
        raise BootstrapError("a hipdnn_vllm runtime exists before plugin configuration")
    _C.configure_plugins(plan.mode, list(plan.paths))
    _C.set_plan_cache_size(config.plan_cache_size)

    # Create the handle now and log its cost: provider initialization runs
    # inside the one backend call, so the time covers all loaded providers.
    device = torch.cuda.current_device()
    started = time.perf_counter()
    loaded_paths = _C.loaded_plugin_paths(device)
    logger.info(
        "hipdnn_vllm: hipDNN handle on device %d created in %.2f s (%s plan) with providers %s",
        device,
        time.perf_counter() - started,
        plan.mode,
        [os.path.basename(p) for p in loaded_paths],
    )

    from . import stats

    stats.start(config)
    return State(
        backend_path=backend,
        backend_version=_C.loaded_backend_version(),
        plugin_plan=plan,
        bundled_dir=bundled,
        other_backends=tuple(p for p in mapped if p != backend),
    )


def loaded() -> State | None:
    return _state
