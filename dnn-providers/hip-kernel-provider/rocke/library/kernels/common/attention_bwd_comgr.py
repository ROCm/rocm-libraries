# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The production comgr of the backward tests, gates, sweep and bench, without
importing torch.

The production ``libamd_comgr`` is the one the ROCm PyTorch wheel bundles in
``<torch>/lib``: every timed process imports torch, and rocKE prefers that
library once torch is in the process. The compile-only processes (the resource
gates, the pre-build, the sweep's candidate compile, the IR checks) do not
need torch for anything else, and importing it costs seconds per process. They
take the same library here instead:

* :func:`torch_package_dir` finds the torch package with
  ``importlib.util.find_spec``, which reads the import system's metadata and
  does not run the package.
* :func:`torch_bundled_comgr` names ``<torch>/lib/libamd_comgr.so`` (the
  ``amd_comgr*.dll`` of a Windows wheel), as the platform's own lookup does
  for an imported torch.
* :func:`pin_production_comgr` exports that path as ``ROCKE_COMGR_LIB``, the
  explicit override the platform resolver honours before anything else. An
  explicit ``ROCKE_COMGR_LIB`` is kept; ``ROCKE_BWD_NO_COMGR_PIN=1`` opts out.
  The platform names the ROCm vintage of a library in a torch package's
  ``lib`` directory from that package's ``version.py`` without importing
  torch (``rocke.runtime.comgr.resolved_lib_rocm_version``), so the flavor
  autodetect, the flavor / comgr guard and the toolchain identity of a
  torch-free process follow the pinned library, in this process and in the
  fresh compile processes it starts (which inherit ``ROCKE_COMGR_LIB``).
* :func:`pin_comgr_cache` gives comgr's own code-object cache a directory of
  this process tree. comgr caches every compile action by default under
  ``$XDG_CACHE_HOME/comgr`` (else ``~/.cache/comgr``); on a network home that
  directory is shared by every job and holds a very large number of entries:
  the lookup, write and pruning of each action then cost far more than the
  compile, and an entry another job wrote can be returned (its key is the
  action's inputs, not the compiling process's history, which matters for
  ``iglp_opt(1)`` kernels; see ``rocke.runtime.comgr.build_hsaco_from_llvm_ir``).
  The directory is a fresh one under the machine's temporary directory
  (:func:`process_comgr_cache`), created by the first process that pins it,
  inherited by its children (pool workers, shards) through the environment
  and removed when that process exits. An explicit ``AMD_COMGR_CACHE_DIR``
  and a disabled cache (``AMD_COMGR_CACHE=0``) are kept.
* :func:`ensure_production_comgr` does both; it is idempotent and cheap.

With the pin the comgr identity (path, vintage, content digest) of a
torch-free process equals that of a process that imported torch, so both
share the HSACO cache and the gate's ``comgr_is_production`` reads the same
string. Nothing here imports torch or loads a library.
"""

from __future__ import annotations

import atexit
import functools
import getpass
import glob
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

__all__ = [
    "COMGR_CACHE_DIR_ENV",
    "COMGR_CACHE_ENV",
    "COMGR_LIB_ENV",
    "NO_PIN_ENV",
    "ensure_production_comgr",
    "pin_comgr_cache",
    "pin_production_comgr",
    "process_comgr_cache",
    "torch_bundled_comgr",
    "torch_package_dir",
]

COMGR_LIB_ENV = "ROCKE_COMGR_LIB"
COMGR_CACHE_ENV = "AMD_COMGR_CACHE"
COMGR_CACHE_DIR_ENV = "AMD_COMGR_CACHE_DIR"
NO_PIN_ENV = "ROCKE_BWD_NO_COMGR_PIN"
_IS_WINDOWS = sys.platform == "win32"


@functools.lru_cache(maxsize=None)
def torch_package_dir() -> Path | None:
    """Directory of the torch package, found without importing it."""
    mod = sys.modules.get("torch")
    origin = getattr(mod, "__file__", None) if mod is not None else None
    if origin is None:
        try:
            spec = importlib.util.find_spec("torch")
        except (ImportError, ValueError):
            return None
        if spec is None or not spec.submodule_search_locations:
            return None
        return Path(list(spec.submodule_search_locations)[0])
    return Path(origin).parent


def torch_bundled_comgr() -> str | None:
    """``<torch>/lib`` comgr library, as the platform's lookup names it for an
    imported torch (``runtime_coexistence._torch_bundled_lib``), or None."""
    pkg = torch_package_dir()
    if pkg is None:
        return None
    libdir = os.path.join(str(pkg), "lib")
    if _IS_WINDOWS:
        direct = os.path.join(libdir, "amd_comgr.dll")
        if os.path.exists(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(libdir, "amd_comgr*.dll")))
        return matches[0] if matches else None
    candidate = os.path.join(libdir, "libamd_comgr.so")
    return candidate if os.path.exists(candidate) else None


def pin_production_comgr(environ: dict | None = None) -> str | None:
    """Export torch's bundled comgr as ``ROCKE_COMGR_LIB``; returns the
    library in effect (an explicit setting wins), or None when there is none
    to pin or ``ROCKE_BWD_NO_COMGR_PIN`` is set."""
    env = os.environ if environ is None else environ
    if env.get(NO_PIN_ENV, "").strip() not in ("", "0"):
        return None
    if env.get(COMGR_LIB_ENV):
        return env[COMGR_LIB_ENV]
    lib = torch_bundled_comgr()
    if lib is not None:
        env[COMGR_LIB_ENV] = lib
    return lib


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name (a container): the uid
        return str(os.getuid()) if hasattr(os, "getuid") else "user"


@functools.lru_cache(maxsize=None)
def process_comgr_cache() -> str:
    """A new ``<temp>/rocke-comgr-cache-<user>-*`` directory owned by this
    process (``TMPDIR`` honoured), made once per process and removed when
    the process exits. Nothing another process tree compiled is in it."""
    path = tempfile.mkdtemp(prefix=f"rocke-comgr-cache-{_user()}-")
    atexit.register(shutil.rmtree, path, True)
    return path


def pin_comgr_cache(environ: dict | None = None) -> str | None:
    """Point comgr's code-object cache at :func:`process_comgr_cache` (see the
    module doc); returns the directory in effect, or None when the cache is
    disabled or ``ROCKE_BWD_NO_COMGR_PIN`` is set. A directory already in the
    environment (set explicitly, or inherited from the process that made it)
    is kept."""
    env = os.environ if environ is None else environ
    if env.get(NO_PIN_ENV, "").strip() not in ("", "0"):
        return None
    if env.get(COMGR_CACHE_ENV, "").strip() == "0":
        return None
    if env.get(COMGR_CACHE_DIR_ENV):
        return env[COMGR_CACHE_DIR_ENV]
    env[COMGR_CACHE_DIR_ENV] = process_comgr_cache()
    return env[COMGR_CACHE_DIR_ENV]


def ensure_production_comgr() -> str | None:
    """:func:`pin_production_comgr` and :func:`pin_comgr_cache`; returns the
    pinned library."""
    lib = pin_production_comgr()
    pin_comgr_cache()
    return lib
