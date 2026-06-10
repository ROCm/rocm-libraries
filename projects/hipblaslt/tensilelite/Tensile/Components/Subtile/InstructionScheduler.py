# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction scheduler for subtile-based mainloop.

Interleaves non-MFMA instructions between MFMAs using a slot-based placer.

The slot-placement algorithm itself lives in C++
(``tensile_writer.subtile.instruction_scheduler``). ``instructionSchedule`` is a
thin adapter that converts the live rocisa emitted-module objects into the
data-only C++ model, runs the C++ scheduler, and rebuilds the rocisa ``Module``
in the returned emission order (applying the waitcnt vmcnt post-pass). There is
no Python slot-placement twin and no opt-in flag — the C++ path runs
unconditionally, mirroring SubtileGeometry / TileInfo.

``extractPathsFromBeforeDeps`` (the pure-Python before-link path decomposition)
remains here because it is also used by ``LogicalScheduler.print_emit_dep_order``
for diagnostics; the C++ scheduler has its own equivalent internally.
"""

from typing import List, Tuple

from tensile_writer.subtile import instruction_scheduler as _cppsched


def extractPathsFromBeforeDeps(emittedModules) -> Tuple[int, List[List[int]], List[List[int]]]:
    """Extract non-MFMA dependency paths using only EmittedModule.before links.

    Returns:
      (mfmaIdx, paths, preMfmaPaths)
      - mfmaIdx: index of the MFMA emitted module in emittedModules
      - paths: list of non-MFMA module-index paths to interleave between MFMAs
      - preMfmaPaths: paths that must be emitted before the first MFMA
        (reachable from the MFMA's before link)
    """
    idToIdx = {em.moduleId: i for i, em in enumerate(emittedModules)}
    n = len(emittedModules)

    mfmaModuleIds = [i for i, em in enumerate(emittedModules) if em.opType == "mfma"]
    assert len(mfmaModuleIds) == 1, "extractPathsFromBeforeDeps expects exactly one MFMA emitted module"
    mfmaIdx = mfmaModuleIds[0]
    nonMfmaIds = [i for i in range(n) if i != mfmaIdx]
    nonMfmaSet = set(nonMfmaIds)

    # Identify the non-MFMA module the MFMA depends on (if any).
    mfmaBefore = emittedModules[mfmaIdx].before
    preMfmaTarget = None
    if mfmaBefore is not None:
        bi = idToIdx.get(mfmaBefore)
        if bi is not None and bi in nonMfmaSet:
            preMfmaTarget = bi

    # Each non-MFMA module has at most one predecessor, and each predecessor
    # has at most one child, so paths are simple chains.
    pred: List[int] = [-1 for _ in range(n)]
    child: List[int] = [-1 for _ in range(n)]
    for i in nonMfmaIds:
        parent = -1
        b = emittedModules[i].before
        if b is not None:
            bi = idToIdx.get(b)
            if bi is not None and bi != i and bi in nonMfmaSet:
                parent = bi
        pred[i] = parent
        if parent != -1:
            assert child[parent] == -1, \
                f"extractPathsFromBeforeDeps expects unique child per predecessor, got {child[parent]} and {i} for {parent}"
            child[parent] = i

    def _findHead(mid: int) -> int:
        cur = mid
        seen = [False for _ in range(n)]
        while pred[cur] != -1 and not seen[cur]:
            seen[cur] = True
            cur = pred[cur]
        return cur

    def _walkFromHead(head: int, used: List[bool]) -> List[int]:
        order: List[int] = []
        localSeen = [False for _ in range(n)]
        cur = head
        while cur != -1 and not used[cur] and not localSeen[cur]:
            order.append(cur)
            localSeen[cur] = True
            cur = child[cur]
        return order

    used = [False for _ in range(n)]
    paths: List[List[int]] = []
    for mid in nonMfmaIds:
        if used[mid]:
            continue
        head = _findHead(mid)
        order = _walkFromHead(head, used)
        assert order, f"extractPathsFromBeforeDeps produced empty path for module {mid}"
        for i in order:
            used[i] = True
        paths.append(order)

    # Separate paths that the MFMA depends on (must go before first MFMA).
    preMfmaPaths: List[List[int]] = []
    regularPaths: List[List[int]] = []
    for path in paths:
        if preMfmaTarget is not None and preMfmaTarget in path:
            preMfmaPaths.append(path)
        else:
            regularPaths.append(path)

    return mfmaIdx, regularPaths, preMfmaPaths


def instructionSchedule(emittedModules):
    """Interleave non-MFMA instructions between MFMAs (slot-based placement).

    Thin adapter over the C++ slot-placement algorithm. Converts the live
    rocisa emitted-module objects to the data-only C++ model, runs the C++
    scheduler, and returns a rocisa ``Module`` in the resulting emission order
    with the waitcnt vmcnt post-pass applied.

    Rules enforced by the C++ algorithm:
      - MFMA order is preserved.
      - Between two adjacent MFMAs there are 2 placement slots.
      - At most one ds_read (LocalReadInstruction) per interval.
      - Before dependencies are respected at module order level.
      - Minimum distance between a ds_read and its waitcnt (hardcoded for now).
      - Module-internal instruction order is preserved.
      - An LR path containing a WAIT_GR is packed from the end backwards so the
        WAIT_GR happens as late as possible.
      - The GR path is spread across remaining valid slots so the GRs issue as
        early as possible.
    """
    return _cppsched.instructionSchedule(emittedModules)
