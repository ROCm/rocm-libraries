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

  * ``HIPDNN_TORCH_PROVIDER_SO``  (REQUIRED unless ``..._SOS`` is set) -- path to a
    single built provider plugin, e.g.
    ``<build>/lib/hipdnn_plugins/engines/libhip_kernel_provider.so``.
  * ``HIPDNN_TORCH_PROVIDER_SOS`` -- ``os.pathsep``/comma-separated list of provider
    plugin ``.so``s to co-load (multi-provider "enable everything" mode), e.g.
    ``libhip_kernel_provider.so:libhipblaslt_plugin.so:libmiopen_plugin.so``. Takes
    precedence over the single-path var. Duplicate engine ids across .so's are
    rejected by the backend, so load each provider at most once.
  * ``HIPDNN_TORCH_SELECT``       -- engine-selection policy (default ``default``):
      - ``default``: hand the graph to hipDNN and let it select across all loaded
        engines (``create_execution_plans([FALLBACK])``). This is hipDNN's own
        engine selection; it can be pointed at a rules file (via the backend's
        ``HIPDNN_HEUR_CONFIG_PATH``) that decides the engine per graph/shape/params.
        Census records the *winning* engine name per shape. No engine is named --
        this is the engine-agnostic default.
      - ``force``: pin ``HIPDNN_TORCH_ENGINE`` via ``create_execution_plan_ext``
        (bypasses selection). Deterministic attribution: census ``aot`` means "*that
        one* engine served it", which is what you want to validate or bench a single
        engine in isolation. Requires ``HIPDNN_TORCH_ENGINE`` (no default) and stops
        with a clear error if it is unset.
  * ``HIPDNN_TORCH_ENGINE``       -- engine name to pin, used only in ``force`` mode
    (no default; ignored under ``default``). Note this pins a *hipDNN engine*, not
    a kernel source: e.g. ``AOT_CATALOG_ENGINE`` serves whatever ahead-of-time
    kernels have been added to that engine's catalog -- today those happen to come
    from rocKE, but the engine is source-agnostic and could serve AOT kernels from
    anywhere. "Force the AOT engine" is not "force rocKE".
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

# torch dtype name -> hipDNN DataType enum name, by *exact* format correspondence
# (the torch name states the format). Resolved defensively at bootstrap() time with
# getattr on both sides, so an entry is added only when both the torch dtype and the
# hipDNN enum exist -- forward-compatible across torch/frontend versions. A dtype not
# in the resulting map is simply never passed through; a mapped-but-unserved dtype
# "just works" the day a kernel for it appears, with no injection change.
_DTYPE_NAME_MAP = {
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "float16": "HALF",
    "bfloat16": "BFLOAT16",
    "float8_e4m3fn": "FP8_E4M3",
    "float8_e5m2": "FP8_E5M2",
    "float8_e4m3fnuz": "FP8_E4M3_FNUZ",
    "float8_e5m2fnuz": "FP8_E5M2_FNUZ",
    "uint8": "UINT8",
    "int8": "INT8",
    "int32": "INT32",
    "int64": "INT64",
    "bool": "BOOLEAN",
}


def _build_dtype_map(torch, hipdnn) -> dict:
    """Every torch dtype that has a hipDNN ``DataType`` counterpart, mapped by exact
    name correspondence. Only pairs where *both* sides exist are included."""
    out = {}
    for torch_name, hipdnn_name in _DTYPE_NAME_MAP.items():
        tdt = getattr(torch, torch_name, None)
        hdt = getattr(hipdnn.DataType, hipdnn_name, None)
        if tdt is not None and hdt is not None:
            out[tdt] = hdt
    return out


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

    __slots__ = (
        "torch",
        "hipdnn",
        "handle",
        "engine_id",
        "engine_name",
        "dtype_map",
        "select_mode",
    )

    def __init__(
        self, torch, hipdnn, handle, engine_id, engine_name, dtype_map, select_mode
    ):
        self.torch = torch
        self.hipdnn = hipdnn
        self.handle = handle
        self.engine_id = engine_id
        self.engine_name = engine_name
        self.dtype_map = dtype_map
        self.select_mode = select_mode


_state = None  # cached State after the first successful bootstrap()


def _provider_sos() -> list:
    """Resolve the provider plugin ``.so`` list. ``HIPDNN_TORCH_PROVIDER_SOS``
    (os.pathsep/comma separated) wins if set; else the single
    ``HIPDNN_TORCH_PROVIDER_SO``. Every path must exist."""
    multi = os.environ.get("HIPDNN_TORCH_PROVIDER_SOS")
    if multi:
        raw = [p for chunk in multi.split(os.pathsep) for p in chunk.split(",")]
        sos = [os.path.expanduser(p.strip()) for p in raw if p.strip()]
    else:
        one = os.environ.get("HIPDNN_TORCH_PROVIDER_SO")
        if not one:
            raise BootstrapError(
                "Neither HIPDNN_TORCH_PROVIDER_SOS nor HIPDNN_TORCH_PROVIDER_SO is "
                "set. Point one at the built provider plugin(s), e.g. "
                "<build>/lib/hipdnn_plugins/engines/libhip_kernel_provider.so"
            )
        sos = [os.path.expanduser(one)]
    for so in sos:
        if not os.path.isfile(so):
            raise BootstrapError(f"provider plugin does not exist: {so}")
    return sos


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

    # Read the engine/select env vars *here*, not at import, so setting them after
    # ``import hipdnn_torch`` (but before the first intercepted call) is honored.
    select_mode = os.environ.get("HIPDNN_TORCH_SELECT", "default")
    if select_mode not in ("default", "force"):
        raise BootstrapError(
            f"HIPDNN_TORCH_SELECT={select_mode!r} is not a valid policy; use "
            "'default' (hipDNN selects across all loaded engines) or "
            "'force' (pin HIPDNN_TORCH_ENGINE)."
        )
    engine_name = os.environ.get("HIPDNN_TORCH_ENGINE")  # no default; force mode only
    if select_mode == "force" and not engine_name:
        raise BootstrapError(
            "HIPDNN_TORCH_SELECT=force requires HIPDNN_TORCH_ENGINE to name the "
            "engine to pin (there is no default). Set it, or use the 'default' "
            "policy to let hipDNN select across all loaded engines."
        )

    providers = _provider_sos()  # validate the cheap, most-common miss first

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
    hipdnn.set_engine_plugin_paths(providers)
    handle = hipdnn.Handle()

    dtype_map = _build_dtype_map(torch, hipdnn)

    _state = State(
        torch=torch,
        hipdnn=hipdnn,
        handle=handle,
        engine_id=_fnv1a64(engine_name) if engine_name else None,
        engine_name=engine_name,
        dtype_map=dtype_map,
        select_mode=select_mode,
    )
    return _state


def is_bootstrapped() -> bool:
    return _state is not None
