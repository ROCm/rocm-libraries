# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One HIP module per binary per run, instead of one per binary per shape.

An AOT cache holds shape-generic binaries: the same HSACO serves every shape
its capabilities cover. The sweeps nonetheless built a
:class:`~rocke.runtime.launcher.KernelLauncher` -- ``read_bytes`` plus
``hipModuleLoadData`` plus ``hipModuleGetFunction`` -- inside the loop over
candidates, inside the loop over cases, so a run of *C* cases over *K*
candidates loaded *C x K* modules of *K* distinct binaries. At a few hundred
microseconds each that is the single largest fixed cost per candidate in a
run that compiles nothing.

:class:`ModuleCache` keys launchers by the binary and the entry point, so a
blob is read and loaded once per run and every later shape reuses the loaded
code object.

Admission, not eviction
-----------------------
The budget is bytes of HSACO held, and when it is reached the cache stops
admitting rather than evicting. That is deliberate. Every case walks the same
candidate list in the same order, which is exactly the access pattern that
defeats LRU: with a capacity below the working set, each lookup evicts the
entry the next case will ask for and the hit rate is zero. Keeping a stable
prefix of the list instead gives a hit rate of ``budget / working set`` --
and never does worse than not caching at all. A cache that does not evict
also never has to prove no launch is in flight on a module it unloads.

The cost of a miss after the budget is reached is exactly the old behavior:
the binary is read and loaded for that candidate alone. :meth:`clear` drains
the device and unloads everything; a run that wants the memory back between
phases calls it, and process exit covers the rest.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: Default HSACO bytes to hold loaded. Conv binaries run from tens to a few
#: hundred kB, so this keeps a few thousand of them -- for most caches the
#: whole working set, and for the widest ones the part the probe reaches
#: first.
DEFAULT_BUDGET_BYTES = 512 * 1024 * 1024


def _signature_key(signature: Sequence[Mapping[str, Any]]) -> Optional[tuple]:
    """Hashable form of a launch signature, or ``None`` if it has no such form.

    The signature decides the launcher's packer, so two launchers over one
    binary and entry point are still different launchers when it differs (a
    conv kernel and the dgrad transform that shares its blob, say).
    """
    try:
        key = tuple(tuple(sorted(arg.items())) for arg in signature)
        hash(key)
    except TypeError:
        return None
    return key


class ModuleCache:
    """Loaded :class:`KernelLauncher`\\s by ``(binary, entry point, signature)``.

    ``budget_bytes`` of 0 disables the cache: every :meth:`launcher` call
    builds a fresh one, which is what the sweeps did before this existed.
    """

    def __init__(self, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> None:
        self._budget = max(0, int(budget_bytes))
        self._held: Dict[tuple, Any] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._full = False

    # -- query ------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        """HSACO bytes of the loaded modules."""
        return self._bytes

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def __len__(self) -> int:
        return len(self._held)

    def summary(self) -> str:
        """One line of accounting, for the end of a run."""
        total = self._hits + self._misses
        pct = 100.0 * self._hits / total if total else 0.0
        line = (
            f"module cache: {len(self._held)} module(s), "
            f"{self._bytes / 1024 / 1024:.1f} MiB, "
            f"{self._hits}/{total} reused ({pct:.0f}%)"
        )
        if self._full:
            line += (
                f" -- budget of {self._budget / 1024 / 1024:.0f} MiB reached, "
                f"the rest reload per use (--module-cache-bytes)"
            )
        return line

    # -- use --------------------------------------------------------------

    def launcher(
        self,
        *,
        hsaco_path: Path,
        kernel_name: str,
        signature: Sequence[Mapping[str, Any]],
    ) -> Any:
        """The launcher for this binary and entry point, loading it if new.

        Raises whatever :class:`KernelLauncher` raises on a binary the device
        rejects (``HipError``); callers already handle that per candidate.
        """
        from rocke.runtime.launcher import KernelLauncher

        key: Optional[Tuple] = None
        sig_key = _signature_key(signature)
        if self._budget and sig_key is not None:
            key = (os.fspath(hsaco_path), kernel_name, sig_key)
            held = self._held.get(key)
            if held is not None:
                self._hits += 1
                return held

        self._misses += 1
        hsaco = Path(hsaco_path).read_bytes()
        launcher = KernelLauncher(
            hsaco=hsaco, kernel_name=kernel_name, signature=signature
        )
        if key is not None:
            if self._bytes + len(hsaco) <= self._budget:
                self._held[key] = launcher
                self._bytes += len(hsaco)
            else:
                self._full = True
        return launcher

    def clear(self) -> None:
        """Drain the device, then unload every module held.

        The drain is why this is not called implicitly: the cache cannot know
        that the launches it handed out have finished, and unloading a module
        under a kernel in flight faults.
        """
        if not self._held:
            return
        from rocke.runtime import synchronize_and_release

        synchronize_and_release()
        for launcher in self._held.values():
            launcher.unload()
        self._held.clear()
        self._bytes = 0
        self._full = False


def add_module_cache_arg(parser, *, default: int = DEFAULT_BUDGET_BYTES) -> None:
    """Add ``--module-cache-bytes`` to an AOT sweep's parser."""
    parser.add_argument(
        "--module-cache-bytes",
        type=int,
        default=default,
        dest="module_cache_bytes",
        metavar="N",
        help=f"HSACO bytes to keep loaded across cases, so a shape-generic "
        f"binary is loaded once per run rather than once per shape "
        f"(default: {default // 1024 // 1024} MiB; 0 loads per use)",
    )
