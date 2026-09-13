# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDM wave arrangement (TDMCross).

TDMFuse decides which tensors share a descriptor set. This decides the order its
members are laid onto waves, which is the whole of what the parameter does.

  TDMCross=0  the shipped order, as TDM_GROUPS writes it
  TDMCross=1  every partitioned group after the first is reversed, so a wave is
              handed one large item and one small one rather than two of a kind
"""

# An int rather than a bool so a grouping with more than two partitioned groups
# can name further arrangements without renaming existing solutions. Only these
# two are implemented; arrangedGroups treats any nonzero as CROSSED.
TDM_CROSS_DEFAULT = 0
TDM_CROSS_CROSSED = 1


def tdmCross(ks):
    """The requested arrangement. Missing key means default."""
    return ks.get("TDMCross", TDM_CROSS_DEFAULT) or TDM_CROSS_DEFAULT


def crossGroups(groups):
    """Reverse every partitioned group after the first.

    With two partitioned groups this is the one non-trivial arrangement: it
    pairs the first member of one group with the last of the other. Relabelling
    every wave is not a distinct arrangement, which is why the first group is
    left alone.
    """
    crossed, seen = [], 0
    for group in groups:
        if len(group) > 1:
            seen += 1
            if seen > 1:
                group = tuple(reversed(group))
        crossed.append(group)
    return tuple(crossed)


def arrangedGroups(groups, cross):
    """`groups` in the member order `cross` selects."""
    return groups if cross == TDM_CROSS_DEFAULT else crossGroups(groups)


def tdmCrossRejectReason(ks):
    """Why this solution cannot take a non-default TDMCross, or None.

    Derived from group structure, never from a TDMFuse integer, so a new row
    inherits the right answer. Block counts do not gate crossing: it preserves
    total LDS exactly and only moves which wave fills which part.
    """
    # Imported here, not at module scope: TDMFuse imports this module for
    # the arrangement, so a module-level import back would be a cycle.
    from .TDMFuse import (liveGroups, partitionedGroups, tdmFusePaired,
                          tdmGrouping, tdmWaveSeparated)

    if tdmCross(ks) == TDM_CROSS_DEFAULT:
        return None
    numWaves = ks.get("NumWaves", 1)
    # Without wave separation the writer programs the default set whatever
    # TDMFuse and TDMCross asked for, so a crossed name would describe a kernel
    # byte-identical to its TDMCross=0 twin.
    if not tdmWaveSeparated(ks) or ks.get("UseSubtileImpl"):
        return ("TDMCross=%d rearranges which wave issues which member of a "
                "descriptor group, which needs wave-separated TDM -- both A and B "
                "moved over more than one wave, and no UseSubtileImpl; got "
                "enableTDMA=%s enableTDMB=%s NumWaves=%d UseSubtileImpl=%s"
                % (tdmCross(ks), bool(ks.get("enableTDMA")),
                   bool(ks.get("enableTDMB")), numWaves,
                   bool(ks.get("UseSubtileImpl"))))
    groups = partitionedGroups(ks)
    if len(groups) < 2:
        return ("TDMCross=%d crosses one descriptor group against another, and "
                "the %s grouping leaves %d group(s) split across waves (%s); with "
                "fewer than two there is nothing to cross"
                % (tdmCross(ks), tdmGrouping(ks).name, len(groups),
                   " + ".join("{%s}" % ",".join(g) for g in liveGroups(ks)) or "none"))
    if tdmFusePaired(ks):
        return ("TDMCross=%d over the paired grouping is not implemented: "
                "crossing asks for both data tensors on one parity and both "
                "scales on the other, and _tdmPairedParityOrder answers within "
                "its own argument pair, so initialisation and the tail program "
                "different descriptors at any K that leaves a tail"
                % tdmCross(ks))
    pgrA, pgrB = ks.get("PrefetchGlobalReadA"), ks.get("PrefetchGlobalReadB")
    if pgrA is not None and pgrB is not None and pgrA != pgrB:
        return ("TDMCross=%d with decoupled prefetch depths (PrefetchGlobalReadA=%s "
                "!= PrefetchGlobalReadB=%s) computes wrong results at every size: "
                "crossing moves a scale onto the wave carrying the other data "
                "tensor while the per-tensor depth is still applied by the "
                "pre-crossing pairing, so the two disagree about which scale "
                "belongs to which data tensor"
                % (tdmCross(ks), pgrA, pgrB))
    return None
