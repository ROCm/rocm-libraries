# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDM descriptor grouping (TDMFuse) and the wave partition it selects.

TDMFuse=0  {A,B} + {MXSA,MXSB}   default two-way parity
TDMFuse=1  {MXSA,A} + {MXSB,B}   crossed parity (tdmFusePaired)
TDMFuse=2  {A,MXSA,MXSB} + {B}   1/1/2 remainder split, NumWaves==4 (tdmFuseAMx)
"""


def tdmBothTensors(ks):
    """True when TDMInst moves both A and B (bits 0 and 1)."""
    tdmInst = ks.get("TDMInst", 0)
    return bool(tdmInst & 0x01) and bool(tdmInst & 0x02)


def _tdmFuseCanShareDescriptors(ks):
    """Guards shared by TDMFuse=1 and 2.

    TDMSplit is kept here even while Solution rejects it globally: writer
    predicates and unit tests still see TDMSplit=True kernels.
    """
    if not tdmBothTensors(ks):
        return False
    if ks.get("TDMSplit") or ks.get("UseSubtileImpl"):
        return False
    pt = ks.get("ProblemType") or {}
    return bool(pt.get("MXBlockA") and pt.get("MXBlockB"))


def tdmFuseAMx(ks):
    """TDMFuse=2: {A,MXSA,MXSB} share one set, B owns its own. NumWaves==4."""
    return (ks.get("TDMFuse") == 2
            and _tdmFuseCanShareDescriptors(ks)
            and ks.get("NumWaves", 1) == 4)


def tdmFusePaired(ks):
    """TDMFuse=1: {MXSA,A} and {MXSB,B}, crossed parity. NumWaves>1."""
    return (ks.get("TDMFuse") == 1
            and _tdmFuseCanShareDescriptors(ks)
            and ks.get("NumWaves", 1) > 1)


def tdmWavePartition(ks, tc):
    """(numComp, waves) for tensor `tc`.

    numComp is how many waves divide the tensor; waves is which wave indices
    actually move it. Component id is the index within `waves`.
    """
    numWaves = ks.get("NumWaves", 1)
    if tdmFuseAMx(ks):
        if tc == "A":
            return 2, (0, 1)
        if tc == "MXSA":
            return 1, (2,)
        if tc == "MXSB":
            return 1, (3,)
        if tc == "B":
            return numWaves, tuple(range(numWaves))
    if tdmFusePaired(ks):
        isEvenArm = tc in ("A", "MXSB")
        return numWaves // 2, tuple(
            w for w in range(numWaves) if (w % 2 == 0) == isEvenArm)
    numComp = numWaves // 2
    isAArm = tc.endswith("A")
    return numComp, tuple(w for w in range(numWaves) if (w % 2 == 0) == isAArm)


def tdmWaveComponents(ks, tc):
    """(numComp, compShift) for tensor `tc`.

    compShift is a right-shift on WaveIdx, or None when WaveIdx is not used:

      None  SMov  id, 0
      0     SMov  id, WaveIdx
      1     SLshr id, WaveIdx, 1

    None is the sentinel because 0 already means "shift by zero". TensorDataMover
    and KernelWriterAssembly both branch on these three values.
    """
    numComp, waves = tdmWavePartition(ks, tc)
    if numComp == 1:
        return numComp, None
    if waves == tuple(range(numComp)):
        return numComp, 0
    return numComp, 1
