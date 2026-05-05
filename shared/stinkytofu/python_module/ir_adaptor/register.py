# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.register``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/include/register.hpp``
and ``rocisa/rocisa/src/register.cpp`` (nanobind bindings).

This is a stateful first-fit register allocator. Tensile's KernelWriter relies
on the *exact* allocation behavior (e.g. AMDGPU ABI requires
``KernArgAddress == SGPR0``); an off-by-one would produce silently wrong asm.
The implementation mirrors the C++ algorithm 1:1, including the tail-grow
overflow path and the ``i += j`` skip-ahead trick in ``checkOutAligned``.

The pool is *index-based* — it knows nothing about register names. The
``name -> index`` mapping is a separate concern that lives in the
KernelWriter (``self.sgprs[name] = idx``).

Notes:
- ``Status`` is an ``IntEnum`` with values matching the C++ enum class
  (``Unavailable=0``, ``Available=1``, ``InUse=2``); KernelWriterAssembly
  reads ``spool[i].status == RegisterPool.Status.InUse`` directly.
- ``Register`` is the ``(status, tag)`` value object; ``tag`` is purely a
  debug hint (used by warnings / ``state()``), never a unique key.
- ``initTmps`` returns ``None`` for now; once ``Module``/``SMovB32``/
  ``VMovB32`` are implemented we can swap in a real Module tree.
"""

from __future__ import annotations

import enum as _stdenum
from copy import deepcopy
from typing import Dict, List, Optional, Tuple


_UNTAGGED = "_untagged_"
_UNTAGGED_ALIGNED = "_untagged_aligned_"


class RegisterPool:
    """First-fit, growable register allocator.

    Construct one per register class (Sgpr/Vgpr/Accvgpr) and call
    ``checkOutAligned`` / ``checkIn`` over its lifetime.

    The pool starts empty (size=0 by convention) and grows from the tail
    when ``checkOutAligned`` cannot satisfy a request and ``preventOverflow``
    is false.
    """

    class Status(_stdenum.IntEnum):
        Unavailable = 0
        Available = 1
        InUse = 2

    class Register:
        """Per-slot state record. ``tag`` is for debugging only."""

        __slots__ = ("status", "tag")

        def __init__(self, status: "RegisterPool.Status", tag: str) -> None:
            self.status = status
            self.tag = tag

        def __repr__(self) -> str:
            return f"Register(status={self.status.name}, tag={self.tag!r})"

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, RegisterPool.Register)
                and self.status == other.status
                and self.tag == other.tag
            )

        def __deepcopy__(self, memo: dict) -> "RegisterPool.Register":
            return RegisterPool.Register(self.status, self.tag)

    def __init__(
        self,
        size: int,
        type: "object",  # rocisa.enum.RegisterType (Sgpr/Vgpr/Accvgpr/mgpr)
        defaultPreventOverflow: bool,
        printRP: bool = False,
    ) -> None:
        self._printRP: bool = bool(printRP)
        self._type = type
        self._defaultPreventOverflow: bool = bool(defaultPreventOverflow)
        self._pool: List[RegisterPool.Register] = [
            RegisterPool.Register(RegisterPool.Status.Unavailable, "init")
            for _ in range(int(size))
        ]
        self._checkOutSize: Dict[int, int] = {}
        self._checkOutSizeTemp: Dict[int, Tuple[int, str]] = {}
        self._occupancyLimitSize: int = 0
        self._occupancyLimitMaxSize: int = 0

    def __deepcopy__(self, memo: dict) -> "RegisterPool":
        new = RegisterPool.__new__(RegisterPool)
        new._printRP = self._printRP
        new._type = self._type
        new._defaultPreventOverflow = self._defaultPreventOverflow
        new._pool = [deepcopy(r, memo) for r in self._pool]
        new._checkOutSize = dict(self._checkOutSize)
        new._checkOutSizeTemp = dict(self._checkOutSizeTemp)
        new._occupancyLimitSize = self._occupancyLimitSize
        new._occupancyLimitMaxSize = self._occupancyLimitMaxSize
        return new

    def setOccupancyLimit(self, maxSize: int, size: int) -> None:
        self._occupancyLimitSize = int(size)
        self._occupancyLimitMaxSize = int(maxSize)

    def resetOccupancyLimit(self) -> None:
        self._occupancyLimitSize = 0
        self._occupancyLimitMaxSize = 0

    def getPool(self) -> List["RegisterPool.Register"]:
        return list(self._pool)

    def addRange(self, start: int, stop: int, tag: str = "") -> str:
        self.add(start, stop - start + 1, tag)
        if start == stop:
            return str(start)
        return f"{start}-{stop}"

    def add(self, start: int, size: int, tag: str = "") -> None:
        start = int(start)
        size = int(size)
        if self._printRP:
            print(f"RP::add({start}..{start + size - 1} for '{tag}')")

        newSize = start + size
        oldSize = len(self._pool)
        if newSize > oldSize:
            self._pool.extend(
                RegisterPool.Register(RegisterPool.Status.Unavailable, tag)
                for _ in range(newSize - oldSize)
            )

        for i in range(start, start + size):
            s = self._pool[i].status
            if s == RegisterPool.Status.Unavailable:
                self._pool[i].status = RegisterPool.Status.Available
                self._pool[i].tag = tag
            elif s == RegisterPool.Status.Available:
                print(
                    f"Warning: RegisterPool::add({start},{start + size - 1}) "
                    f"pool[{i}]({self._pool[i].tag}) already available"
                )
            elif s == RegisterPool.Status.InUse:
                print(
                    f"Warning: RegisterPool::add({start},{start + size - 1}) "
                    f"pool[{i}]({self._pool[i].tag}) already in use"
                )
            else:
                raise RuntimeError("RegisterPool::add() invalid status")

        if self._printRP:
            print(self.state())

    def addFromCheckOut(self, start: int) -> None:
        start = int(start)
        if start in self._checkOutSize:
            size = self._checkOutSize[start]
            for i in range(start, start + size):
                if self._pool[i].status != RegisterPool.Status.InUse:
                    raise RuntimeError(
                        "RegisterPool::addFromCheckOut() is not in InUse state"
                    )
                self._pool[i].status = RegisterPool.Status.Available
            self._checkOutSizeTemp[start] = (size, self._pool[start].tag)
            del self._checkOutSize[start]
            if self._printRP:
                print(
                    f"RP::addFromCheckOut('{self._pool[start].tag}') "
                    f"@ {start} +{size}"
                )
        else:
            raise RuntimeError(
                "RegisterPool::addFromCheckOut() but it was never checked out"
            )

    def remove(self, start: int, size: int, tag: str = "") -> None:
        start = int(start)
        size = int(size)
        if self._printRP:
            print(f"RP::remove({start}..{start + size - 1}) for {tag}")

        newSize = start + size
        oldSize = len(self._pool)
        if newSize > oldSize:
            print(
                f"Warning: RegisterPool::remove({start},{start + size - 1}) "
                f"but poolSize={oldSize}"
            )

        for i in range(start, start + size):
            if i >= len(self._pool):
                continue
            s = self._pool[i].status
            if s == RegisterPool.Status.Available:
                self._pool[i].status = RegisterPool.Status.Unavailable
            elif s == RegisterPool.Status.Unavailable:
                print(
                    f"Warning: RegisterPool::remove({start},{start + size - 1}) "
                    f"pool[{i}]({self._pool[i].tag}) already unavailable"
                )
            elif s == RegisterPool.Status.InUse:
                print(
                    f"Warning: RegisterPool::remove({start},{start + size - 1}) "
                    f"pool[{i}]({self._pool[i].tag}) still in use"
                )
            else:
                raise RuntimeError("RegisterPool::remove() invalid status")

    def removeFromCheckOut(self, start: int) -> None:
        start = int(start)
        if start in self._checkOutSizeTemp:
            size, tag = self._checkOutSizeTemp[start]
            for i in range(start, start + size):
                if self._pool[i].status != RegisterPool.Status.Available:
                    raise RuntimeError(
                        "RegisterPool::removeFromCheckOut() is not in Available state"
                    )
                self._pool[i].status = RegisterPool.Status.InUse
                self._pool[i].tag = tag
            self._checkOutSize[start] = size
            del self._checkOutSizeTemp[start]
            if self._printRP:
                print(
                    f"RegisterPool::removeFromCheckOut('{self._pool[start].tag}') "
                    f"@ {start} +{size}"
                )
        else:
            raise RuntimeError(
                "RegisterPool::removeFromCheckOut() but it was never checked out"
            )

    def checkOut(
        self, size: int, tag: str = _UNTAGGED, preventOverflow: int = -1
    ) -> int:
        return self.checkOutAligned(size, 1, tag, preventOverflow)

    def checkOutAligned(
        self,
        size: int,
        alignment: int,
        tag: str = _UNTAGGED_ALIGNED,
        preventOverflow: int = -1,
    ) -> int:
        size = int(size)
        alignment = int(alignment)

        # Mirror the C++ behavior: ``preventOverflow`` is an int with -1 meaning
        # "use defaultPreventOverflow". Tensile passes ``False`` explicitly
        # (becomes 0) to permit growth.
        if int(preventOverflow) == -1:
            preventOverflow = int(self._defaultPreventOverflow)
        else:
            preventOverflow = int(bool(preventOverflow))

        if size == 0:
            raise ValueError("Size must be greater than 0")

        # First-fit scan with the C++ ``i += j`` skip-ahead optimization. We
        # use a manual ``while`` loop because Python's ``for ... in range()``
        # cannot mutate the iterator.
        found = -1
        i = 0
        n = len(self._pool)
        while i < n:
            if i % alignment != 0:
                i += 1
                continue
            if i + size > n:
                i += 1
                continue
            allAvailable = True
            for j in range(size):
                if self._pool[i + j].status != RegisterPool.Status.Available:
                    allAvailable = False
                    i += j  # skip ahead past the known-unavailable slot
                    break
            if allAvailable:
                found = i
                break
            i += 1

        if found != -1:
            for k in range(found, found + size):
                self._pool[k].status = RegisterPool.Status.InUse
                self._pool[k].tag = tag
            self._checkOutSize[found] = size
            if self._printRP:
                print(
                    f"RP::checkOut '{tag}' ({size},{alignment}) "
                    f"@ {found} avail={self.available()}"
                )
            return found

        # Overflow path: grow the pool from the tail.
        if preventOverflow:
            raise RuntimeError(
                "RegisterPool::checkOutAligned: register overflow prevented "
                "by preventOverflow flag"
            )

        start = len(self._pool)
        if start:
            # Walk backwards while the tail is Available, claiming the run.
            # NOTE: matches C++ ``for (i = size-1; i > 0; --i)`` — index 0 is
            # intentionally NOT visited; this is a faithful reproduction of
            # the rocisa quirk so downstream allocations stay byte-for-byte
            # compatible.
            i = len(self._pool) - 1
            while i > 0:
                if self._pool[i].status == RegisterPool.Status.Available:
                    self._pool[i].tag = tag
                    start = i
                    i -= 1
                    continue
                else:
                    break
            start = ((start + alignment - 1) // alignment) * alignment

        newSize = start + size
        oldSize = len(self._pool)
        if self._occupancyLimitSize > 0:
            if (
                newSize > self._occupancyLimitSize
                and newSize <= self._occupancyLimitMaxSize
            ):
                print(
                    f"newSize {newSize} OldSize {oldSize} "
                    f"Limit {self._occupancyLimitSize}"
                )
                if self._occupancyLimitSize < newSize:
                    raise RuntimeError(
                        "RegisterPool::checkOutAligned: occupancy limit exceeded"
                    )

        overflow = (newSize - oldSize) if newSize > oldSize else 0

        for k in range(start, len(self._pool)):
            self._pool[k].status = RegisterPool.Status.InUse
            self._pool[k].tag = tag

        for _ in range(overflow):
            if len(self._pool) < start:
                # Padding pushed in to satisfy alignment on the next grow.
                self._pool.append(
                    RegisterPool.Register(RegisterPool.Status.Available, tag)
                )
            else:
                self._pool.append(
                    RegisterPool.Register(RegisterPool.Status.InUse, tag)
                )

        self._checkOutSize[start] = size
        if self._printRP:
            print(
                f"RP::checkOut '{tag}' ({size},{alignment}) "
                f"@ {start} (overflow)"
            )
        return start

    def checkOutMulti(
        self, sizes: List[int], alignment: int, tags: List[str]
    ) -> List[int]:
        if len(sizes) != len(tags):
            raise ValueError("Sizes and tags must have the same length")

        totalSize = sum(int(s) for s in sizes)
        # ``checkOutAligned(..., preventOverflow=False)`` mirrors C++ which
        # passes literal ``false`` here; tag is intentionally empty.
        idx = self.checkOutAligned(totalSize, alignment, "", False)

        # Re-partition the lump-sum checkout into per-tag entries.
        if idx in self._checkOutSize:
            del self._checkOutSize[idx]
        idxVec: List[int] = []
        cur = idx
        for s, t in zip(sizes, tags):
            s = int(s)
            idxVec.append(cur)
            self._checkOutSize[cur] = s
            for k in range(cur, cur + s):
                self._pool[k].tag = t
            cur += s
        return idxVec

    def initTmps(
        self, initValue: int, start: int = 0, stop: int = -1
    ) -> Optional[object]:
        # Real implementation requires Module + SMovB32 / VMovB32 to be
        # functional. Returning ``None`` is intentional during Phase 2 — the
        # callers in KernelWriterAssembly do ``module.add(initTmps(...))``
        # and the surrounding ``Module`` shim swallows ``None`` gracefully.
        # TODO(Phase 4): build a real Module containing per-slot init insts.
        return None

    def checkIn(self, start: int) -> None:
        start = int(start)
        if start in self._checkOutSize:
            size = self._checkOutSize[start]
            for i in range(start, start + size):
                self._pool[i].status = RegisterPool.Status.Available
            del self._checkOutSize[start]
            if self._printRP:
                print(
                    f"RP::checkIn('{self._pool[start].tag}') @ {start} +{size}"
                )
        else:
            tag = self._pool[start].tag if start < len(self._pool) else "?"
            print(
                f"Warning: RegisterPool::checkIn('{tag}',{start}) "
                f"but it was never checked out"
            )

    def size(self) -> int:
        return len(self._pool)

    def available(self) -> int:
        return sum(
            1 for r in self._pool if r.status == RegisterPool.Status.Available
        )

    def availableBlock(self, blockSize: int, align: int) -> int:
        if blockSize == 0:
            blockSize = 1
        blocksAvail = 0
        consecAvailable = 0
        for i, reg in enumerate(self._pool):
            if reg.status == RegisterPool.Status.Available:
                if not (consecAvailable == 0 and i % align != 0):
                    consecAvailable += 1
            else:
                blocksAvail += consecAvailable // blockSize
                consecAvailable = 0
        blocksAvail += consecAvailable // blockSize
        return blocksAvail * blockSize

    def availableBlockMaxVgpr(
        self, maxVgpr: int, blockSize: int, align: int
    ) -> int:
        if blockSize == 0:
            blockSize = 1
        blocksAvail = 0
        consecAvailable = 0
        for i in range(maxVgpr):
            if i >= len(self._pool):
                if not (consecAvailable == 0 and i % align != 0):
                    consecAvailable += 1
            else:
                reg = self._pool[i]
                if reg.status == RegisterPool.Status.Available:
                    if not (consecAvailable == 0 and i % align != 0):
                        consecAvailable += 1
                else:
                    blocksAvail += consecAvailable // blockSize
                    consecAvailable = 0
        blocksAvail += consecAvailable // blockSize
        return blocksAvail * blockSize

    def availableBlockAtEnd(self) -> int:
        cnt = 0
        for r in reversed(self._pool):
            if r.status == RegisterPool.Status.Available:
                cnt += 1
            else:
                break
        return cnt

    def checkFinalState(self) -> None:
        for i, r in enumerate(self._pool):
            if r.status == RegisterPool.Status.InUse:
                if self._printRP:
                    print(self.state())
                raise RuntimeError(
                    f"RegisterPool::checkFinalState: temp ({i}, '{r.tag}') "
                    "was never checked in."
                )
        if self._printRP:
            print(f"total vgpr count: {self.size()}")

    def state(self) -> str:
        out = ""
        place_values = [1000, 100, 10, 1]
        for pv_idx in range(1, len(place_values)):
            place_value = place_values[pv_idx]
            prior_place_value = place_values[pv_idx - 1]
            if len(self._pool) >= place_value:
                pvs = ""
                for i in range(len(self._pool)):
                    if i % place_value == 0:
                        pvs += str((i % prior_place_value) // place_value)
                    else:
                        pvs += " "
                out += pvs + "\n"
        for r in self._pool:
            if r.status == RegisterPool.Status.Unavailable:
                out += "."
            elif r.status == RegisterPool.Status.Available:
                out += "|"
            elif r.status == RegisterPool.Status.InUse:
                out += "#"
        return out

    def growPool(
        self,
        rangeStart: int,
        rangeEnd: int,
        checkOutSize: int,
        comment: str = "",
    ) -> None:
        tl: List[int] = []
        if checkOutSize == 1:
            continuous = 0
            availList: List[int] = []
            for r in self._pool:
                if r.status != RegisterPool.Status.Available:
                    if continuous > 0:
                        availList.append(continuous)
                        continuous = 0
                else:
                    continuous += 1
            rangeTotal = rangeEnd - rangeStart
            for numGpr in availList:
                rangeTurn = numGpr // checkOutSize
                if rangeTurn > 0:
                    tl.append(self.checkOut(checkOutSize * rangeTurn, comment))
                    rangeTotal -= rangeTurn
            if rangeTotal > 0:
                tl.append(self.checkOut(checkOutSize * rangeTotal, comment))
        else:
            for _ in range(rangeStart, rangeEnd):
                tl.append(self.checkOut(checkOutSize, comment))
        for t in tl:
            self.checkIn(t)

    def appendPool(self, newSize: int) -> None:
        oldSize = len(self._pool)
        if newSize > oldSize:
            self._pool.extend(
                RegisterPool.Register(
                    RegisterPool.Status.Available, "append pool"
                )
                for _ in range(newSize - oldSize)
            )
