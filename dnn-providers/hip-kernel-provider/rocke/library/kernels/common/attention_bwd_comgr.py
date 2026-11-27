# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Comgr's code-object cache for the attention backward tests, gates, sweep
and bench: one directory per process tree.

The compiler itself is whatever rocKE's discovery loads (``ROCKE_COMGR_LIB``
when set, else the platform's search; ``dsl_docs/reference/env_flags.md``,
"Compiler detection and provenance"); nothing here chooses it.

comgr caches every compile action by default under ``$XDG_CACHE_HOME/comgr``
(else ``~/.cache/comgr``). On a network home that directory is shared by every
job and holds a very large number of entries: the lookup, write and pruning of
each action then cost far more than the compile, and an entry another job wrote
can be returned (its key is the action's inputs, not the compiling process's
history, which matters for ``iglp_opt(1)`` kernels; see
``rocke.runtime.comgr.build_hsaco_from_llvm_ir``).

* :func:`process_comgr_cache` is a fresh directory under the machine's
  temporary directory, created by the first process that asks for it and
  removed when that process exits.
* :func:`pin_comgr_cache` exports it as ``AMD_COMGR_CACHE_DIR``, so the
  process's children (pool workers, shards) inherit it. An explicit
  ``AMD_COMGR_CACHE_DIR`` and a disabled cache (``AMD_COMGR_CACHE=0``) are
  kept; ``ROCKE_BWD_NO_COMGR_PIN=1`` opts out.

Nothing here imports torch or loads a library.
"""

from __future__ import annotations

import atexit
import functools
import getpass
import os
import shutil
import tempfile

__all__ = [
    "COMGR_CACHE_DIR_ENV",
    "COMGR_CACHE_ENV",
    "NO_PIN_ENV",
    "pin_comgr_cache",
    "process_comgr_cache",
]

COMGR_CACHE_ENV = "AMD_COMGR_CACHE"
COMGR_CACHE_DIR_ENV = "AMD_COMGR_CACHE_DIR"
NO_PIN_ENV = "ROCKE_BWD_NO_COMGR_PIN"


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
