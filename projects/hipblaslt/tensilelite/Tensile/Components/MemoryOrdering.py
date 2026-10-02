# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Architecture-specific primitives for device-scope publication protocols."""

import abc

from rocisa.code import Module
from rocisa.enum import CacheScope
from rocisa.instruction import GlobalInv, GlobalWb, SWaitCnt, SWaitXCnt

from ..Component import Component


class DeviceMemoryOrdering(Component):
    """Order device-memory accesses used by cross-workgroup publication.

    Callers retain their publication/polling protocol and memory-access cache
    modifiers. In particular, gfx950 requires coherent workspace and flag
    accesses (glc+slc); its wait fences alone cannot make cached data coherent.
    Targets with HasInvWbDevFences use explicit device-scope writeback and
    invalidation instead. These primitives do not replace arbitrary waits,
    barriers, or the ordering required by an atomic operation.

    The pre-volatile VMEM drain has its own replay capability gates, independent
    of release/acquire selection. No scheduling policy or protocol registers
    are needed to select or emit any of these operations.
    """

    def __call__(self):
        assert(0)

    @staticmethod
    def _hasXcdSplitL2(writer) -> bool:
        """Preserve the gfx950 ISA fallback for older capability tables."""
        if writer.states.archCaps.get("HasXCDSplitL2"):
            return True
        ver = getattr(writer.states, "version", None)
        if ver is None:
            return False
        return tuple(ver)[:3] in ((9, 5, 0),)

    def preVolatileVmem(self, writer, comment="") -> Module:
        """Drain replay before a volatile/atomic VMEM access when required."""
        module = Module("StreamK pre-volatile VMEM drain")
        if writer.states.archCaps["RequiresXCntForVolatileVMEM"] or \
                writer.states.archCaps["EnableXnackReplay"]:
            module.add(SWaitXCnt(xcnt=0, comment=comment))
        return module

    @abc.abstractmethod
    def releaseFence(self, writer) -> Module:
        """Order prior workspace stores before publishing completion."""
        pass

    @abc.abstractmethod
    def acquireFence(self, writer) -> Module:
        """Prepare dependent device-scope reads in the publication protocol."""
        pass


class DeviceMemoryOrderingDefault(DeviceMemoryOrdering):
    """Wait for stores; coherent accesses need no explicit acquire fence."""

    archCaps = {"HasInvWbDevFences": False, "HasXCDSplitL2": False}

    @classmethod
    def matches(cls, writer, debug=False):
        caps = writer.states.archCaps
        if caps.get("HasInvWbDevFences", False):
            return False
        return not cls._hasXcdSplitL2(writer)

    def releaseFence(self, writer) -> Module:
        module = Module("StreamK release fence (default)")
        module.add(SWaitCnt(vscnt=0, comment="wait for data store"))
        return module

    def acquireFence(self, writer) -> Module:
        return Module("StreamK acquire fence (default, no-op)")


class DeviceMemoryOrderingGfx9Xcd(DeviceMemoryOrdering):
    """Wait fences for XCD-split L2 with coherent glc+slc accesses (gfx950)."""

    archCaps = {"HasInvWbDevFences": False, "HasXCDSplitL2": True}

    @classmethod
    def matches(cls, writer, debug=False):
        caps = writer.states.archCaps
        if caps.get("HasInvWbDevFences", False):
            return False
        return cls._hasXcdSplitL2(writer)

    def releaseFence(self, writer) -> Module:
        module = Module("StreamK release fence (gfx9 XCD)")
        module.add(SWaitCnt(vlcnt=0, vscnt=0,
            comment="release: wait for partials stores before flag"))
        return module

    def acquireFence(self, writer) -> Module:
        module = Module("StreamK acquire fence (gfx9 XCD)")
        module.add(SWaitCnt(vlcnt=0, vscnt=0,
            comment="acquire: drain before reading partials"))
        return module


class DeviceMemoryOrderingDevScopeFences(DeviceMemoryOrdering):
    """Explicit writeback/invalidation at device scope (for example gfx1250)."""

    archCaps = {"HasInvWbDevFences": True}

    def releaseFence(self, writer) -> Module:
        module = Module("StreamK release fence (dev-scope)")
        module.add(SWaitCnt(vlcnt=0,
            comment="release: drain in-flight loads before global_wb"))
        module.add(SWaitCnt(vscnt=0, comment="wait for data store"))
        module.add(GlobalWb(scope=CacheScope.SCOPE_DEV,
            comment="release: writeback partials to L2-coherent point"))
        module.add(SWaitCnt(vlcnt=0, vscnt=0,
            comment="release: wait for global_wb"))
        return module

    def acquireFence(self, writer) -> Module:
        # Drop stale device-scope lines before dependent flag or workspace reads.
        module = Module("StreamK acquire fence (dev-scope)")
        module.add(GlobalInv(scope=CacheScope.SCOPE_DEV,
            comment="acquire: invalidate before dependent dev-scope read"))
        module.add(SWaitCnt(vlcnt=0, comment="acquire: wait for global_inv"))
        return module
