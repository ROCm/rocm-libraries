# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Keep comgr's code-object cache off network filesystems.

HIP unbundles compressed code objects, and hiprtc compiles, through comgr,
which caches results under ``AMD_COMGR_CACHE_DIR`` (default
``$XDG_CACHE_HOME/comgr``, else ``~/.cache/comgr``). The cache prunes through
LLVM's ``pruneCache``, which refreshes its interval timestamp by truncating an
empty file. NFS leaves that file's mtime unchanged, so on an NFS home every
comgr action rescans and stats the whole cache directory, and every hit is read
back from NFS. The MIOpen provider's handle loads MIOpen's CK grouped-conv
library, which is hundreds of compressed bundles, one comgr action each, so its
first handle pays this hundreds of times.

Upstream comgr (ROCm/llvm-project#4247) disables the default directory when it
is not on a local filesystem. This module does the same check but keeps the
cache, moved to a private node-local directory, so later processes still hit
it. comgr reads its variables once per process, at its first action, and HIP's
first kernel launch already runs one, so :func:`configure` runs in
``register()``, before any device work.
"""

from __future__ import annotations

import ctypes
import logging
import os
import stat
import sys
import tempfile
from collections.abc import MutableMapping

from .config import ConfigError

logger = logging.getLogger("hipdnn_vllm")

__all__ = ["ENV", "NETWORK_FS_MAGIC", "configure", "default_cache_dir"]

ENV = "HIPDNN_VLLM_COMGR_CACHE"
_MODES = ("auto", "keep")
_COMGR_ENVS = ("AMD_COMGR_CACHE", "AMD_COMGR_CACHE_DIR")

# statfs(2) f_type of network and cluster filesystems.
NETWORK_FS_MAGIC = {
    0x6969: "nfs",
    0x517B: "smb",
    0xFF534D42: "cifs",
    0xFE534D42: "smb2",
    0x0BD00BD0: "lustre",
    0x47504653: "gpfs",
    0x00C36400: "ceph",
    0x19830326: "beegfs",
    0x5346414F: "afs",
    0x73757245: "coda",
    0x01021997: "9p",
}


def default_cache_dir(env: MutableMapping[str, str]) -> str:
    """comgr's default cache directory (LLVM ``sys::path::cache_directory`` + ``comgr``)."""
    xdg = env.get("XDG_CACHE_HOME", "")
    base = xdg if os.path.isabs(xdg) else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "comgr")


def _fs_magic(path: str) -> int | None:
    """statfs f_type of the filesystem holding ``path`` (or its nearest
    existing ancestor); None where statfs is unavailable."""
    if not sys.platform.startswith("linux"):
        return None
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent
    buf = ctypes.create_string_buffer(256)  # struct statfs is 120 bytes on x86_64
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.statfs(os.fsencode(path), buf) != 0:
        return None
    # f_type (__fsword_t) is the first member; magic numbers are 32-bit.
    return int.from_bytes(buf.raw[: ctypes.sizeof(ctypes.c_long)], sys.byteorder) & 0xFFFFFFFF


def _private_dir(root: str) -> str | None:
    """``<root>/hipdnn_vllm_comgr_<uid>``, created 0700; None unless it is a
    real directory owned by this user and closed to others."""
    path = os.path.join(root, f"hipdnn_vllm_comgr_{os.getuid()}")
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as e:
        logger.warning("hipdnn_vllm: cannot create comgr cache dir %s: %s", path, e)
        return None
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        logger.warning(
            "hipdnn_vllm: %s is not a private directory owned by uid %d; comgr cache left as is", path, os.getuid()
        )
        return None
    return path


def configure(env: MutableMapping[str, str] | None = None, tmp_root: str | None = None) -> str | None:
    """Point ``AMD_COMGR_CACHE_DIR`` at a node-local directory when comgr's
    default cache directory is on a network filesystem. Does nothing when the
    user set ``AMD_COMGR_CACHE`` or ``AMD_COMGR_CACHE_DIR``, or
    ``HIPDNN_VLLM_COMGR_CACHE=keep``. Returns the directory it set, else None."""
    env = os.environ if env is None else env
    mode = env.get(ENV, "auto").strip().lower()
    if mode not in _MODES:
        raise ConfigError(f"{ENV}={mode!r} is not valid; use one of {_MODES}")
    if mode == "keep" or any(name in env for name in _COMGR_ENVS):
        return None
    default = default_cache_dir(env)
    fs = NETWORK_FS_MAGIC.get(_fs_magic(default))
    if fs is None:
        return None
    root = tmp_root or tempfile.gettempdir()
    if NETWORK_FS_MAGIC.get(_fs_magic(root)) is not None:
        logger.info("hipdnn_vllm: comgr cache %s is on %s, but so is %s; left as is", default, fs, root)
        return None
    path = _private_dir(root)
    if path is None:
        return None
    env["AMD_COMGR_CACHE_DIR"] = path
    logger.info(
        "hipdnn_vllm: comgr cache %s is on %s; set AMD_COMGR_CACHE_DIR=%s (%s=keep to opt out)", default, fs, path, ENV
    )
    return path
