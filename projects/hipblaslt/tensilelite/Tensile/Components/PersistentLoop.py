################################################################################
#
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from rocisa.code import Module, Label
from rocisa.container import vgpr, sgpr
from rocisa.instruction import VMovB32, SCmpGeU32
from ..Component import Component
import abc

class PersistentLoop(Component):
    """
    Persistent loop code.
    """
    def __call__(self):
        assert(0)

    @abc.abstractmethod
    def openPersistentLoop(self, writer, kernel):
        pass

    @abc.abstractmethod
    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        pass

    @abc.abstractmethod
    def recalcLocalReadAddressesAB(self, writer, kernel):
        pass

    @abc.abstractmethod
    def closePersistentLoop(self, writer, kernel):
        pass


class PersistentLoopOff(PersistentLoop):
    kernel = {"StreamK": 0}

    def openPersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop Off openPersistentLoop")
        return module

    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        module = Module("PersistentLoop Off recalcLocalWriteAddresses")
        return module

    def recalcLocalReadAddressesAB(self, writer, kernel):
        module = Module("PersistentLoop Off recalcLocalReadAddressesAB")
        return module

    def closePersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop Off closePersistentLoop")
        return module


class PersistentLoopOn(PersistentLoop):
    # Stream-K persistent loop

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["StreamK"] > 0

    def openPersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop On openPersistentLoop")

        # Label start of persistent loop
        module.addComment2("Persistent Loop Start")
        persistentLabel = Label(label="PersistentLoopStart", comment="")
        module.add(persistentLabel)

        # TODO remove?
        # kStr += inst("s_add_u32", sgpr("PersistentLoopIter"), sgpr("PersistentLoopIter"), hex(1), "Inc PersistentLoop Iter")     # Back-up: not needed now
        #kStr += str(Code.WaitCnt(self.version, 0,0,"wait for outstanding stores"))
        return module

    def recalcLocalWriteAddresses(self, writer, kernel, tc):
        module = Module("PersistentLoop On recalcLocalWriteAddresses")

        if getattr(writer, "oriLwa%s" % tc) is None:
            setattr(writer, "oriLwa%s" % tc, writer.vgprPool.checkOut(1, "OriLocalWriteddr%s" % tc))
            module.add(VMovB32(dst=vgpr(getattr(writer, "oriLwa%s" % tc)), src=vgpr("LocalWriteAddr%s" % tc), comment="back up LWA for persistent kernel + wider local read"))

        return module

    def recalcLocalReadAddressesAB(self, writer, kernel):
        module = Module("PersistentLoop On recalcLocalReadAddressesAB")

        needRecalc = writer.states.numReadsIterCoalescedA > 1 or writer.states.numReadsIterCoalescedB > 1
        # backup LocalReadAddr
        # LdsPad + LBSPP case, need to backup LocalReadAddr even if recalc is not done

        # The code below backs up LRA at the start of the tail loop and restores at the end of tail loop.
        # This would be needed if we want to support PersistentKernel in the future (without stream-k).
        # Backup is not needed for stream-k since LRA is recalculated at the start of the persistent loop for the new tile or partial tile.
        # Enabling the backup code would require fixing an interaction with the code that reallocates vgprs for the store code.

        # needBackupLRAddr = needRecalc or (kernel["LdsPadA"] and kernel["LdsBlockSizePerPadA"] or kernel["LdsPadB"] and kernel["LdsBlockSizePerPadB"]) and kernel["StreamK"] == 0

        # if needBackupLRAddr:
        #     # need to back-up the LRA before reCalculation for wider local read (when no wlr, no need to do this)
        #     if writer.oriLraA is None: # and not kernel["DirectToVgprA"]: # no local read code if DirectToVgpr is enabled
        #         writer.oriLraA = writer.vgprPool.checkOut(1, "OriLocalReadAddrA")
        #         module.add(VMovB32(dst=vgpr(writer.oriLraA), src=vgpr("LocalReadAddrA"), comment="back up LRA for persistent kernel + wider local read"))
        #     if writer.oriLraB is None: # and not kernel["DirectToVgprB"]: # no local read code if DirectToVgpr is enabled
        #         writer.oriLraB = writer.vgprPool.checkOut(1, "OriLocalReadAddrB")
        #         module.add(VMovB32(dst=vgpr(writer.oriLraB), src=vgpr("LocalReadAddrB"), comment="back up LRA for persistent kernel + wider local read"))

        return module

    def closePersistentLoop(self, writer, kernel):
        module = Module("PersistentLoop On closePersistentLoop")
        skCloseLoopLabel = Label("SK_CloseLoop", "")
        module.add(skCloseLoopLabel)
        # endIter = "StreamKIterEnd" if kernel["StreamK"] == 1 else "TotalIters"
        endIter = "TotalIters" if kernel["StreamK"] == 2 else "StreamKIterEnd"
        module.add(SCmpGeU32(src0=sgpr("StreamKIter"), src1=sgpr(endIter), comment="Check if done all StreamK iterations"))
        module.add(writer.longBranchScc0(Label("PersistentLoopStart", ""), posNeg=-1))
        return module
