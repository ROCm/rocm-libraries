# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A Solution stored as a compressed pickle plus a summary of what the build driver reads.

TensileCreateLibrary parses every logic file in worker processes and ships the
resulting solutions back to one parent, which keeps all of them until the
library files are written. A fully built ``Solution`` is a ``_state`` dict of
some 500 entries plus a private ``ProblemType``: about 140 KB of resident
memory each once unpickled, so a full architecture build (350k solutions)
costs the parent 50 GiB. The parent reads a dozen of those keys. Everything
else is needed only inside a worker: kernel code generation hydrates the
solution, and library writing hydrates it to compute names.

``CompactSolution`` holds the pickled ``Solution`` as a zlib-compressed blob
(a few KB) and a tuple of the values for ``SUMMARY_KEYS``. It implements the
``Mapping`` protocol over that summary plus an overlay for the keys the parent
writes (``codeObjectFile``, ``BaseName``, ``Kernel``); ``hydrate()`` rebuilds
the real ``Solution`` with the overlay applied. Reads of any other key fall
back to a transient hydration and are counted (``fallbackReads``) so that an
unexpected hot path shows up as a number rather than as a silent slowdown;
an object read that way repeatedly is promoted to a cached hydrated copy so
the cost stays bounded.

Hashing and equality mirror ``Solution`` exactly (full name, codeObjectFile,
DeviceNames) so deduplication through ``dict.fromkeys`` is unchanged.
"""

import pickle
import zlib
from collections.abc import Mapping

from .Naming import getKernelFileBase, getKernelNameMin, getKeyNoInternalArgs
from .Solution import Solution

# Keys TensileCreateLibrary's parent process reads from a kernel/solution
# without hydrating it. Keep in sync with Run.py / Toolchain/Assembly.py.
SUMMARY_KEYS = (
    "KernelLanguage",
    "ISA",
    "WavefrontSize",
    "CustomKernel",
    "CustomKernelName",
    "codeObjectFile",
    "DeviceNames",
    "PrefetchGlobalRead",
    "NonTemporalA",
    "NonTemporalB",
    "NonTemporalD",
    "AdaptiveGemmNTAB",
    "WaveSeparateGlobalReadA",
    "WaveSeparateGlobalReadB",
    "UnrollLoopSwapGlobalReadOrder",
    "DirectToVgprA",
    "DirectToVgprB",
    "SolutionIndex",
    "SolutionNameMin",
    "StinkyTofuRegisterAllocation",
    "CUOccupancy",
    "MathClocksUnrolledLoop",
    "LdsNumBytes",
    "LoopIters",
    "LdsOffsetBias",
    "_subtileUnitScaleVgpr",
)
_SUMMARY_INDEX = {k: i for i, k in enumerate(SUMMARY_KEYS)}
# Presence is a bitmask over SUMMARY_KEYS rather than a sentinel value: a
# sentinel object would not survive the pickle round trip from the parse
# worker with its identity intact.

# Attributes of Solution that are shared, process-wide objects. They are kept
# out of the blob (one copy per CompactSolution pickle instead of one per
# solution) and reattached on hydration.
_SHARED_ATTRS = ("assembler", "isaInfoMap")

COMPRESS_LEVEL = 3

# After this many fallback reads on one object it is promoted to a cached
# hydrated Solution: a loop that reads many unknown keys of one solution then
# costs one hydration, not one per key.
PROMOTE_AFTER = 4

fallbackReads = 0


class CompactSolution(Mapping):
    __slots__ = (
        "_blob",
        "_summary",
        "_present",
        "_overlay",
        "_hydrated",
        "_misses",
        "_nameFull",
        "srcName",
        "duplicate",
        "baseName",
        "kernelNameMin",
        "_baseDedupeKey",
        "helperNames",
        "assembler",
        "isaInfoMap",
        "__weakref__",
    )

    def __init__(
        self,
        blob,
        summary,
        present,
        nameFull,
        srcName,
        baseName,
        kernelNameMin,
        baseDedupeKey,
        helperNames,
        assembler,
        isaInfoMap,
    ):
        self._blob = blob
        self._summary = summary
        self._present = present
        self._overlay = {}
        self._hydrated = None
        self._misses = 0
        self._nameFull = nameFull
        self.srcName = srcName
        self.duplicate = False
        self.baseName = baseName
        self.kernelNameMin = kernelNameMin
        self._baseDedupeKey = baseDedupeKey
        self.helperNames = helperNames
        self.assembler = assembler
        self.isaInfoMap = isaInfoMap

    # ------------------------------------------------------------------ build
    @classmethod
    def fromSolution(
        cls, solution: Solution, splitGSU: bool = False, helperNameCallables=None
    ):
        """Compact ``solution``.

        ``helperNameCallables`` is ``KernelHelperNaming.kernelObjectNameCallables()``
        (passed in to avoid an import cycle); when given, the helper-kernel
        names each callable derives from the solution are precomputed so the
        parent can deduplicate helper kernels without hydrating.
        """
        if isinstance(solution, CompactSolution):
            return solution
        # Same object the kernel list will hold; getKernels() marks it a kernel.
        kernel = solution.getKernels()[0]
        nameFull = str(solution)
        baseName = getKernelFileBase(splitGSU, kernel)
        kernelNameMin = getKernelNameMin(kernel, splitGSU)
        # getKeyNoInternalArgs ends with codeObjectFile + str(DeviceNames); the
        # parent assigns codeObjectFile later, so keep the part before them and
        # let dedupeKey() append the live values.
        fullKey = getKeyNoInternalArgs(kernel, splitGSU)
        suffix = (solution._state.get("codeObjectFile") or "") + str(
            solution._state.get("DeviceNames", "")
        )
        # None if the key's layout is not the expected one; dedupeKey() then
        # recomputes from the hydrated solution instead of guessing.
        baseDedupeKey = (
            fullKey[: len(fullKey) - len(suffix)] if fullKey.endswith(suffix) else None
        )
        helperNames = None
        if helperNameCallables is not None:
            helperNames = [
                (helperType, callable(solution))
                for helperType, callable in helperNameCallables
            ]
        state = solution._state
        summary = tuple(state.get(k) for k in SUMMARY_KEYS)
        present = 0
        for i, k in enumerate(SUMMARY_KEYS):
            if k in state:
                present |= 1 << i
        payload = dict(solution.__dict__)
        shared = tuple(payload.pop(a, None) for a in _SHARED_ATTRS)
        blob = zlib.compress(
            pickle.dumps((type(solution), payload), pickle.HIGHEST_PROTOCOL),
            COMPRESS_LEVEL,
        )
        return cls(
            blob,
            summary,
            present,
            nameFull,
            solution.srcName,
            baseName,
            kernelNameMin,
            baseDedupeKey,
            helperNames,
            *shared
        )

    # ---------------------------------------------------------------- hydrate
    def hydrate(self) -> Solution:
        """Return the full Solution, with parent-side writes applied.

        A new object every call unless this instance was promoted; callers that
        need the result more than once should keep it.
        """
        if self._hydrated is not None:
            return self._hydrated
        return self._build()

    def _build(self) -> Solution:
        cls, payload = pickle.loads(zlib.decompress(self._blob))
        sol = cls.__new__(cls)
        sol.__dict__.update(payload)
        for attr, value in zip(_SHARED_ATTRS, (self.assembler, self.isaInfoMap)):
            setattr(sol, attr, value)
        if self._overlay:
            sol._state.update(self._overlay)
        sol.duplicate = self.duplicate
        return sol

    def _fallback(self):
        global fallbackReads
        fallbackReads += 1
        self._misses += 1
        if self._misses >= PROMOTE_AFTER:
            self._hydrated = self._build()
            return self._hydrated
        return self._build()

    # ---------------------------------------------------------------- mapping
    def __getitem__(self, key):
        overlay = self._overlay
        if key in overlay:
            return overlay[key]
        i = _SUMMARY_INDEX.get(key)
        if i is not None:
            if not (self._present >> i) & 1:
                raise KeyError(key)
            return self._summary[i]
        if self._hydrated is not None:
            return self._hydrated._state[key]
        return self._fallback()._state[key]

    def __setitem__(self, key, value):
        self._overlay[key] = value
        if self._hydrated is not None:
            self._hydrated._state[key] = value

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self):
        return iter(self._fallback()._state)

    def __len__(self):
        return len(self._fallback()._state)

    def keys(self):
        return list(self._fallback()._state.keys())

    # --------------------------------------------------------- Solution look
    def dedupeKey(self) -> str:
        """getKeyNoInternalArgs(self): masked name + codeObjectFile + DeviceNames, with current values."""
        if self._baseDedupeKey is None:
            return getKeyNoInternalArgs(self.hydrate(), False)
        return (
            self._baseDedupeKey
            + (self.get("codeObjectFile") or "")
            + str(self.get("DeviceNames", ""))
        )

    def getKernels(self):
        self["Kernel"] = True
        return [self]

    def __str__(self):
        return self._nameFull

    def __repr__(self):
        return self._nameFull

    def __hash__(self):
        deviceNames = str(self.get("DeviceNames", ""))
        return hash(self._nameFull + (self.get("codeObjectFile") or "") + deviceNames)

    def __eq__(self, other):
        if not isinstance(other, (CompactSolution, Solution)):
            return False
        if str(self) != str(other):
            return False
        return self.get("DeviceNames", None) == other.get("DeviceNames", None)

    def __ne__(self, other):
        return not self.__eq__(other)

    # ----------------------------------------------------------------- pickle
    def __getstate__(self):
        return (
            self._blob,
            self._summary,
            self._present,
            self._overlay,
            self._nameFull,
            self.srcName,
            self.duplicate,
            self.baseName,
            self.kernelNameMin,
            self._baseDedupeKey,
            self.helperNames,
            self.assembler,
            self.isaInfoMap,
        )

    def __setstate__(self, state):
        (
            self._blob,
            self._summary,
            self._present,
            self._overlay,
            self._nameFull,
            self.srcName,
            self.duplicate,
            self.baseName,
            self.kernelNameMin,
            self._baseDedupeKey,
            self.helperNames,
            self.assembler,
            self.isaInfoMap,
        ) = state
        self._hydrated = None
        self._misses = 0


def hydrateSolution(solution):
    """The full Solution for ``solution``, which may already be one."""
    if isinstance(solution, CompactSolution):
        return solution.hydrate()
    return solution


def kernelFileBase(splitGSU: bool, kernel) -> str:
    """getKernelFileBase without hydrating a CompactSolution."""
    if isinstance(kernel, CompactSolution):
        return kernel.baseName
    return getKernelFileBase(splitGSU, kernel)


def kernelDedupeKey(splitGSU: bool, kernel) -> str:
    """getKeyNoInternalArgs without hydrating a CompactSolution."""
    if isinstance(kernel, CompactSolution):
        return kernel.dedupeKey()
    return getKeyNoInternalArgs(kernel, splitGSU)


def compactLibrarySolutions(library, splitGSU: bool, helperNameCallables) -> None:
    """Replace every originalSolution under a MasterSolutionLibrary with a CompactSolution."""
    for sol in library.solutions.values():
        sol.originalSolution = CompactSolution.fromSolution(
            sol.originalSolution, splitGSU, helperNameCallables
        )
    for lazy in library.lazyLibraries.values():
        compactLibrarySolutions(lazy, splitGSU, helperNameCallables)
