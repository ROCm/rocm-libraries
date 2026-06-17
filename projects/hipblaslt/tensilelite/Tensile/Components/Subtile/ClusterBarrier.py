# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cluster-scope barrier handshake for the subtile mainloop.

Subtile-specific equivalent of the StinkyTofu InsertClusterBarrierPass (which
does not run at OptLevel 0 / ScheduleIterAlg=3). Free functions take the writer
explicitly to keep cluster logic in Subtile/ rather than KernelWriterAssembly.
"""

from __future__ import annotations

from rocisa.code import Label, Module, TextBlock
from rocisa.instruction import SBarrier, SBarrierSignalIsFirst, SCBranchSCC0


def subtileClusterBarrier(writer, kernel, label="") -> Module:
    # Balanced workgroup barrier + cluster signal/wait -3 per occurrence.
    mod = Module("subtile_cluster_barrier")
    # Workgroup barrier via isfirst: the first wave to arrive gets SCC=1 and so
    # is the single wave that signals the cluster barrier (one arrival per WG).
    skipPreSignal = Label(writer.labels.getUniqueNamePrefix("skipCBPreSignal"), "")
    mod.add(SBarrierSignalIsFirst(False, "workgroup barrier signal (isfirst)"))
    mod.add(SBarrier(True, True, False, "workgroup barrier wait"))
    mod.add(SCBranchSCC0(skipPreSignal.getLabelName(), "only the first-arriving wave signals the cluster"))
    mod.add(SBarrier(True, False, True, "cluster_barrier signal"))
    mod.add(skipPreSignal)
    mod.add(SBarrier(True, True, True, "cluster_barrier wait"))
    # SQTT trace marker tagging which loop section this barrier belongs to.
    base = {"PRELOOP": 0x00, "MAINLOOP": 0x10, "NGLL": 0x20,
            "NLL": 0x30, "TAILLOOP": 0x40}
    kind, _, suffix = label.partition("_C")
    loopId = (base.get(kind, 0xf0) + (int(suffix) if suffix.isdigit() else 0)) & 0xff
    mod.add(TextBlock(f"s_ttracedata_imm {0xc100 | loopId:#06x}\n"))
    return mod
