# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipdnn_torch.bootstrap -- the single, env-parametrized provider/backend/frontend
init that every op override shares.

The overrides route ``torch.nn.functional`` calls onto a hipDNN engine plugin. To
do that, three things have to come up, in this exact order (the whole reason this
lives in one place):

  1. ``import torch`` and warm the HIP/HSA stack (``torch.zeros(1, device="cuda")``)
     BEFORE anything else touches the GPU -- otherwise the backend's device probe
     races an un-initialised HSA and aborts.
  2. ``dlopen`` **torch's own** bundled ``libhipdnn_backend.so`` with
     ``RTLD_GLOBAL`` so the frontend bindings bind to the exact backend that torch
     ships. Mixing a system/SDK backend with torch's frontend is the #1
     hard-to-debug failure (the "version-skew trap"); see the README
     "Environment setup" section.
  3. Import the frontend bindings, point them at the provider ``.so``
     (``set_engine_plugin_paths``), and open a ``Handle``.

Everything discoverable is parametrised by environment variable so nothing here is
tied to one machine:

  * ``HIPDNN_TORCH_PROVIDER_SO``  (REQUIRED) -- path to the built provider plugin,
    e.g. ``<build>/lib/hipdnn_plugins/engines/libhip_kernel_provider.so``.
  * ``HIPDNN_TORCH_ENGINE``       -- engine name to pin (default
    ``AOT_CATALOG_ENGINE``).
  * ``HIPDNN_TORCH_FRONTEND_DIR`` -- fallback path to a raw
    ``frontend_bindings/build`` dir, used only when the ``hipdnn-frontend`` wheel
    is not importable.
  * ``HIPDNN_TORCH_BACKEND_GLOB`` -- override the glob used to locate torch's
    bundled ``libhipdnn_backend.so`` (rarely needed).

Importing this module does NOT import torch or touch the GPU. All of that happens
lazily on the first :func:`bootstrap` call (which the overrides trigger from
``install()``), so ``import hipdnn_torch`` stays cheap and side-effect free and any
discovery failure surfaces as a clear :class:`BootstrapError` naming the env var to
set.
"""

import ctypes
import glob
import os
import sys

_ENGINE_NAME = os.environ.get("HIPDNN_TORCH_ENGINE", "AOT_CATALOG_ENGINE")


def _fnv1a64(s: str) -> int:
    """Signed-int64 FNV-1a of the engine name (matches the backend's
    EngineNames.hpp). The id->name registry in shipped bindings can predate a
    plugin engine and return '' for it, so we identify the engine by this hashed
    id in the ranked-engine list rather than by name."""
    h = 0xCBF29CE484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h - (1 << 64) if h >= (1 << 63) else h


class BootstrapError(RuntimeError):
    """Raised when torch's backend, the frontend bindings, or the provider ``.so``
    cannot be discovered. The message always names the env var to set."""


class State:
    """Everything the overrides need after a successful bootstrap. Treat as
    read-only."""

    __slots__ = ("torch", "hipdnn", "handle", "engine_id", "engine_name", "dtype_map")

    def __init__(self, torch, hipdnn, handle, engine_id, engine_name, dtype_map):
        self.torch = torch
        self.hipdnn = hipdnn
        self.handle = handle
        self.engine_id = engine_id
        self.engine_name = engine_name
        self.dtype_map = dtype_map


_state = None  # cached State after the first successful bootstrap()


def _provider_so() -> str:
    so = os.environ.get("HIPDNN_TORCH_PROVIDER_SO")
    if not so:
        raise BootstrapError(
            "HIPDNN_TORCH_PROVIDER_SO is not set. Point it at the built "
            "hip-kernel-provider plugin, e.g. "
            "<build>/lib/hipdnn_plugins/engines/libhip_kernel_provider.so"
        )
    so = os.path.expanduser(so)
    if not os.path.isfile(so):
        raise BootstrapError(f"HIPDNN_TORCH_PROVIDER_SO does not exist: {so}")
    return so


def _torch_backend_path(torch) -> str:
    site = os.path.dirname(os.path.dirname(torch.__file__))
    pattern = os.environ.get(
        "HIPDNN_TORCH_BACKEND_GLOB",
        os.path.join(site, "_rocm_sdk_libraries_*", "lib", "libhipdnn_backend.so"),
    )
    hits = glob.glob(pattern)
    if not hits:
        raise BootstrapError(
            f"Could not find torch's bundled libhipdnn_backend.so (looked for "
            f"{pattern!r}). Is this a ROCm build of torch? Override the search "
            "with HIPDNN_TORCH_BACKEND_GLOB."
        )
    return hits[0]


def _import_frontend():
    """Prefer the installed ``hipdnn-frontend`` wheel; fall back to a raw
    ``frontend_bindings/build`` dir named by HIPDNN_TORCH_FRONTEND_DIR."""
    # 1. The public wheel package (re-exports the compiled extension), then the
    #    raw compiled extension if it happens to be importable already.
    for name in ("hipdnn_frontend", "hipdnn_frontend_python"):
        try:
            return __import__(name)
        except ImportError:
            pass
    # 2. A raw build directory on sys.path.
    fe_dir = os.environ.get("HIPDNN_TORCH_FRONTEND_DIR")
    if fe_dir:
        fe_dir = os.path.expanduser(fe_dir)
        if fe_dir not in sys.path:
            sys.path.insert(0, fe_dir)
        for name in ("hipdnn_frontend", "hipdnn_frontend_python"):
            try:
                return __import__(name)
            except ImportError:
                pass
    raise BootstrapError(
        "hipDNN frontend bindings are not importable. Install the "
        "'hipdnn-frontend' wheel, or set HIPDNN_TORCH_FRONTEND_DIR to a "
        "frontend_bindings/build directory."
    )


def bootstrap() -> State:
    """Idempotent one-time init. Returns the cached :class:`State` on repeat
    calls. Raises :class:`BootstrapError` with an actionable message on any
    discovery failure."""
    global _state
    if _state is not None:
        return _state

    provider = _provider_so()  # validate the cheap, most-common miss first

    import torch  # deferred: importing this module must not require torch

    # (1) Bring torch's HIP/HSA stack up before dlopening the backend.
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

    # (2) dlopen torch's OWN hipdnn backend RTLD_GLOBAL (0x101 == RTLD_LAZY |
    #     RTLD_GLOBAL) so the frontend binds to the exact backend torch ships.
    ctypes.CDLL(_torch_backend_path(torch), mode=0x101)

    # (3) Frontend bindings -> provider plugin -> handle.
    hipdnn = _import_frontend()
    hipdnn.set_engine_plugin_paths([provider])
    handle = hipdnn.Handle()

    dtype_map = {
        torch.float32: hipdnn.DataType.FLOAT,
        torch.bfloat16: hipdnn.DataType.BFLOAT16,
        torch.float16: hipdnn.DataType.HALF,
    }

    _state = State(
        torch=torch,
        hipdnn=hipdnn,
        handle=handle,
        engine_id=_fnv1a64(_ENGINE_NAME),
        engine_name=_ENGINE_NAME,
        dtype_map=dtype_map,
    )
    return _state


def is_bootstrapped() -> bool:
    return _state is not None
