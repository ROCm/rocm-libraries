# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check LDS-read dependencies in the gfx950 NN/TT custom schedule."""

import re

import pytest
from rocisa.instruction import SWaitCnt

from Tensile.Common import IsaVersion
from Tensile.Common.DataType import DataType
from Tensile.Components.CustomSchedule import hasCustomSchedule


@pytest.mark.parametrize("dtype", ["H", "B"])
@pytest.mark.parametrize("layout", ["NN", "TT"])
@pytest.mark.parametrize("code_path", [0, 1])
def test_nn_cms_lds_reads_complete_before_use(dtype, layout, code_path):
    data_type = DataType(dtype)
    kernel = {
        "UseCustomMainLoopSchedule": True,
        "EnableMatrixInstruction": True,
        "ISA": IsaVersion(9, 5, 0),
        "ProblemType": {
            "DataType": data_type,
            "DataTypeA": data_type,
            "DataTypeB": data_type,
            "TransposeA": layout == "TT",
            "TransposeB": layout == "TT",
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

    # Each half-loop uses eight fragments per input. LR[A/B]0 fills bank 1;
    # LR[A/B]1 refills bank 0 for the next iteration. The prologue supplies
    # a ready bank 0. NN packs A and uses B directly; TT swaps those roles.
    direct_input = "B" if layout == "NN" else "A"
    # The emitter places shared streams before code-path-specific streams,
    # preserving insertion order within each group, after the indexed MFMA.
    streams = sorted(schedule.optSchedule.items(), key=lambda item: len(item[1]))
    pending = []
    for iteration in range(3):
        loads = packs = 0
        for mfma in range(schedule.numMfma):
            # The MFMA nest advances A in the inner loop and B in the outer.
            fragment = (mfma % 64) // 8 if layout == "NN" else mfma % 8
            operand = (direct_input, mfma // 64, fragment)
            assert (
                operand not in pending
            ), f"iteration {iteration}, MFMA {mfma}: {operand} is still pending"

            for stream, paths in streams:
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
                    elif re.fullmatch(r"Pack[AB][01]", stream):
                        # Each group of four permutes reads fragment pairs
                        # (0,1), (2,3), (4,5), (6,7), repeated for each word.
                        for fragment in (2 * (index % 4), 2 * (index % 4) + 1):
                            operand = (stream[4], 1 - int(stream[5]), fragment)
                            assert operand not in pending, (
                                f"iteration {iteration}, MFMA {mfma}, {stream}: "
                                f"{operand} is still pending"
                            )
                        packs += 1
        assert loads == 32
        assert packs == 64
