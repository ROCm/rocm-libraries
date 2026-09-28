# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check live LDS-read destinations across iterations of the gfx950 TN schedule."""

import re

import pytest
from rocisa.instruction import SWaitCnt

from Tensile.Common import IsaVersion
from Tensile.Common.DataType import DataType
from Tensile.Components.CustomSchedule import hasCustomSchedule


@pytest.mark.parametrize("dtype", ["H", "B"])
@pytest.mark.parametrize("code_path", [0, 1])
def test_tn_cms_lds_reads_complete_before_mfma(dtype, code_path):
    data_type = DataType(dtype)
    kernel = {
        "UseCustomMainLoopSchedule": True,
        "EnableMatrixInstruction": True,
        "ISA": IsaVersion(9, 5, 0),
        "ProblemType": {
            "DataType": data_type,
            "DataTypeA": data_type,
            "DataTypeB": data_type,
            "TransposeA": True,
            "TransposeB": False,
        },
        "MacroTile0": 256,
        "MacroTile1": 256,
        "DepthU": 64,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "DirectToLds": True,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 8,
        "LocalReadVectorWidth": 8,
        "MatrixInstruction": [16, 16, 32, 1],
        "MIWaveGroup": [2, 2],
        "LDSTrInst": False,
        "TransposeLDS": 1,
    }
    enabled, schedule = hasCustomSchedule(kernel)
    assert enabled
    assert schedule.numMfma == 128
    assert schedule.numCodePaths == 2

    # Each half-loop multiplies eight A fragments by eight B fragments.
    # LRA0/LRB0 fill register bank 1; LRA1/LRB1 refill bank 0 for the
    # next iteration. The first bank is ready on entry from the prologue.
    # Track LDS reads only: barriers and VMEM waits do not retire them.
    pending = []
    for iteration in range(3):
        loads = 0
        for mfma in range(schedule.numMfma):
            bank = mfma // 64
            for operand in [("A", bank, mfma % 8), ("B", bank, (mfma % 64) // 8)]:
                assert operand not in pending, (
                    f"iteration {iteration}, MFMA {mfma}: {operand} is still pending"
                )

            # customMainLoopSchedule emits the indexed MFMA first, then
            # the scheduled streams in this dictionary's insertion order.
            for stream, paths in schedule.optSchedule.items():
                slots = paths[min(code_path, len(paths) - 1)]
                for index, slot in enumerate(slots):
                    if slot != mfma:
                        continue
                    if stream == "SYNC":
                        wait = schedule.syncCode[index]
                        if isinstance(wait, SWaitCnt) and wait.dscnt >= 0:
                            pending = pending[-wait.dscnt :] if wait.dscnt else []
                    elif re.fullmatch(r"LR[AB][01]", stream):
                        pending.append((stream[2], 1 - int(stream[3]), index))
                        # gfx950 LGKM capacity: issuing the sixteenth
                        # operation proves completion of the oldest one.
                        pending = pending[-15:]
                        loads += 1
        assert loads == 32
