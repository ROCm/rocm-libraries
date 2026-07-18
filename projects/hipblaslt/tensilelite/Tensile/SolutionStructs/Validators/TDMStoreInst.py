################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Solution-time validation for the ``TDMStoreInst`` epilogue (gfx1250).

``TDMStoreInst`` replaces the per-element ``buffer_store`` to the final D output
with one whole-MacroTile ``tensor_store_from_lds`` (fp32 accumulator -> bf16
convert+pack -> M-contiguous LDS scratch -> TDM store, edge handled by the
descriptor ``tensor_dim`` clamp).

Only the FFM-validated envelope is accepted; every other config is rejected up
front so it is a clean rejection rather than a silently mis-generated kernel:

  Supported
    - gfx1250, bf16 DestDataType, single ComputeDataType, HighPrecisionAccumulate.
    - SourceSwap 0/1, UseSubtileImpl 0/1.
    - GlobalSplitU=1 direct-to-D, and GlobalSplitU=-1 (auto-GSU) when it resolves
      to the SingleBuffer direct-to-D algorithm; UseBeta 0/1.
    - StreamK in the non-atomic workspace (``_GlobalAccumulation == 'PartialsBuffer'``)
      mode with UseBeta=True: the fp32 partial-tile store to the workspace stays on
      buffer_store (StreamK partialsWriteBatch), and only the fixup-owner's final
      bf16->D store -- structurally identical to the GSU=1 case -- uses the TDM store.

  Rejected
    - non-gfx1250 / non-bf16 / non-single-ComputeDataType / non-HPA: the convert+pack
      assumptions do not hold.
    - StreamK atomic (``_GlobalAccumulation`` != 'PartialsBuffer'): the atomic reduction
      does not route the final D store through the TDM-gated globalWriteBatch.
    - StreamK with UseBeta=False: StreamK reuses the Beta SGPR for its partial-tile
      index alias, so the kernel does not assemble when UseBeta=False.
    - GlobalSplitU>1: workspace accumulation store, destination is not D.
    - StoreRemapVectorWidth: the TDM store branch is skipped (silently inactive).
    - UseE / UseBias / UseScaleAlphaVec / UseScaleAB / UseScaleCD / Activation:
      epilogue features unvalidated with the TDM store (several stage into LDS at
      offset 0 and collide with the TDM M-contiguous scratch).

Pure dict-in / bool-out: no client build, no GPU, no rocisa device code.
"""

from ..Utilities import reject


def validateTDMStoreInst(state: dict, printRejectionReason: bool = True) -> bool:
    """Reject unsupported ``TDMStoreInst`` solutions.

    Returns True if the solution is acceptable (including the ``TDMStoreInst=False``
    no-op case), or False after emitting a rejection (which also sets
    ``state["Valid"] = False``).  Requires ``state["_GlobalAccumulation"]`` to have
    been assigned already (StreamK/GSU gate on it).
    """
    if not state["TDMStoreInst"]:
        return True

    pt = state["ProblemType"]

    if tuple(state["ISA"])[:2] != (12, 5):
        reject(state, printRejectionReason, "TDMStoreInst requires gfx1250 (tensor_store_from_lds is gfx1250-only)")
        return False
    if not pt["DestDataType"].isBFloat16():
        reject(state, printRejectionReason, "TDMStoreInst currently supports only bf16 DestDataType (converted+packed store path)")
        return False
    if not pt["HighPrecisionAccumulate"]:
        reject(state, printRejectionReason, "TDMStoreInst requires HighPrecisionAccumulate (fp32->bf16 convert+pack store path)")
        return False
    if not pt["ComputeDataType"].isSingle():
        reject(state, printRejectionReason, "TDMStoreInst requires ComputeDataType=single (the convert+pack store path assumes an fp32 accumulator)")
        return False
    if state["StreamK"] != 0 and state["_GlobalAccumulation"] != 'PartialsBuffer':
        reject(state, printRejectionReason, "TDMStoreInst supports StreamK only in the non-atomic workspace (PartialsBuffer) mode; the atomic StreamK reduction path does not route the final D store through the TDM-gated globalWriteBatch")
        return False
    if state["StreamK"] != 0 and not pt.get("UseBeta", False):
        reject(state, printRejectionReason, "TDMStoreInst + StreamK requires UseBeta=True (StreamK reuses the Beta SGPR for its partial-tile index alias)")
        return False
    if state["GlobalSplitU"] > 1:
        reject(state, printRejectionReason, "TDMStoreInst does not yet support GlobalSplitU>1 (workspace accumulation store)")
        return False
    if state["StoreRemapVectorWidth"]:
        reject(state, printRejectionReason, "TDMStoreInst is incompatible with StoreRemapVectorWidth (TDM store would be silently inactive)")
        return False
    if pt.get("UseE", False):
        reject(state, printRejectionReason, "TDMStoreInst does not support UseE (auxiliary output)")
        return False
    if pt.get("UseBias", 0):
        reject(state, printRejectionReason, "TDMStoreInst does not support UseBias")
        return False
    if pt.get("UseScaleAlphaVec", 0):
        reject(state, printRejectionReason, "TDMStoreInst does not support UseScaleAlphaVec")
        return False
    if pt.get("UseScaleAB", ""):
        reject(state, printRejectionReason, "TDMStoreInst does not support UseScaleAB")
        return False
    if pt.get("UseScaleCD", False):
        reject(state, printRejectionReason, "TDMStoreInst does not support UseScaleCD")
        return False
    if pt.get("ActivationType", 'none') != 'none':
        reject(state, printRejectionReason, "TDMStoreInst does not support fused Activation")
        return False

    return True
