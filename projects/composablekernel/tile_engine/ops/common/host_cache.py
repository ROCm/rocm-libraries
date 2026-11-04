# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host data of verified search calls, computed once per problem and run.

A verified search call runs one kernel in a fresh worker process, which would
otherwise regenerate the problem's random inputs and recompute its fp32
reference on every call. A search run keeps them in a temp directory of its
own (``HostCache``, in the driver) that its workers read from, or write to on
a miss (``cached``, in the worker):

- an entry is a directory ``<dir>/<key>`` of ``.npy`` files and a
  ``meta.json``, written under a temp name and renamed into place, so a reader
  sees a complete entry or none. Workers that miss together each compute (as
  without the cache) and the first rename wins;
- ``key`` hashes the fields the op says decide the data (problem, dtype,
  layout, input seed, ...); ``meta.json`` repeats them, and a mismatch is a
  miss;
- an entry larger than ``max_entry_bytes``, or one that would take the
  directory past ``max_total_bytes`` or leave its filesystem less than
  ``MIN_FREE_FRACTION`` free, is not written: the worker uses what it
  computed, as without the cache. Concurrent writers can pass the total check
  together, so the directory may exceed it by one entry per writer;
- an error reading or writing an entry is a miss: the cache never changes a
  result, only how often it is computed.

``HostCache.close`` (also on interpreter exit) removes the directory. Importing
this module does not import numpy.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

FORMAT = 1  # part of every key: a change of the entry layout misses old entries
MIN_FREE_FRACTION = 0.1  # of the filesystem, left free by every write
GIB = 1 << 30
PREFIX = "ck_search_host_cache-"


def entry_key(fields: Mapping) -> str:
    """The entry name of ``fields`` (JSON data)."""
    text = json.dumps({"format": FORMAT, "fields": fields}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def usage(root) -> Tuple[int, int]:
    """(entries, bytes) under ``root``; temp entries count bytes only."""
    entries = size = 0
    for path, _dirs, files in os.walk(root):
        if "meta.json" in files and not Path(path).name.startswith(".tmp-"):
            entries += 1
        for f in files:
            try:
                size += os.stat(os.path.join(path, f)).st_size
            except OSError:
                pass
    return entries, size


class HostCache:
    """A search run's cache directory: a new temp directory under ``parent``
    (the system temp directory when None). ``spec`` is what a worker gets.
    ``max_total_bytes`` None means a quarter of the free space of that
    filesystem now."""

    def __init__(self, parent=None, max_entry_bytes: int = 4 * GIB,
                 max_total_bytes: Optional[int] = None):
        self.dir = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent))
        if max_total_bytes is None:
            max_total_bytes = shutil.disk_usage(self.dir).free // 4
        self.spec = {"dir": str(self.dir), "max_entry_bytes": int(max_entry_bytes),
                     "max_total_bytes": int(max_total_bytes)}
        atexit.register(self.close)

    def close(self) -> Tuple[int, int]:
        """Remove the directory; returns its (entries, bytes) before."""
        atexit.unregister(self.close)
        found = usage(self.dir)
        shutil.rmtree(self.dir, ignore_errors=True)
        return found

    def __enter__(self) -> "HostCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextlib.contextmanager
def run_cache(settings, log=print):
    """A ``HostCache`` for the run's workers, from ``settings`` (the search's
    ``HostCacheSettings``), removed on exit with one ``log`` line; None when
    ``max_total_gib`` is 0."""
    if settings.max_total_gib == 0:
        yield None
        return
    total = settings.max_total_gib
    cache = HostCache(settings.dir, int(settings.max_entry_gib * GIB),
                      None if total is None else int(total * GIB))
    try:
        yield cache
    finally:
        entries, size = cache.close()
        log(f"  Host cache: {entries} entries, {size / GIB:.2f} GiB, removed ({cache.dir})")


def cached(spec: Optional[Mapping], fields: Mapping,
           compute: Callable[[], Dict[str, object]]) -> Dict[str, object]:
    """``compute()``'s named numpy arrays, read from the entry of ``fields``
    in the cache ``spec`` (a ``HostCache.spec``) when it has one, else
    computed and, within the caps, published. ``spec`` None: ``compute()``."""
    if not spec:
        return compute()
    import numpy as np

    fields = json.loads(json.dumps(fields))  # as meta.json reads back
    path = Path(spec["dir"]) / entry_key(fields)
    try:
        meta = json.loads((path / "meta.json").read_text())
        if meta["fields"] == fields:
            return {n: np.load(path / f"{n}.npy", allow_pickle=False)
                    for n in meta["arrays"]}
    except Exception:  # noqa: BLE001 - no entry, or an unreadable one: compute
        pass
    data = compute()
    try:
        _publish(spec, path, fields, data)
    except Exception:  # noqa: BLE001 - the result does not need the entry
        pass
    return data


def _publish(spec: Mapping, path: Path, fields, data: Mapping) -> None:
    import numpy as np

    size = sum(a.nbytes for a in data.values())
    root = path.parent
    if size > spec["max_entry_bytes"] or usage(root)[1] + size > spec["max_total_bytes"]:
        return
    disk = shutil.disk_usage(root)
    if disk.free - size < MIN_FREE_FRACTION * disk.total:
        return
    tmp = root / f".tmp-{path.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    os.mkdir(tmp)  # fails once the run removed its directory
    try:
        for name, a in data.items():
            np.save(tmp / f"{name}.npy", a, allow_pickle=False)
        (tmp / "meta.json").write_text(json.dumps({"fields": fields,
                                                   "arrays": sorted(data)}))
        os.rename(tmp, path)  # atomic; fails when another worker was first
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # gone after a rename
