# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""DebugDisableInstrMask: comment out whole instruction classes in a kernel's assembly.

Timing-only debug aid (e.g. compute-only / memory-only cycle counts); the resulting
kernels compute wrong results. s_waitcnt instructions are kept, they just wait on less.
"""

import re

from ..ExecutionPolicy import requiresPartialReduction

# bit -> mnemonic prefixes
DEBUG_INSTR_CLASSES = {
    1:  ("v_wmma", "v_swmmac", "v_mfma", "v_smfmac"),             # matrix instructions
    2:  ("buffer_load", "global_load", "flat_load", "tensor_load"),  # global loads (incl. direct-to-LDS)
    4:  ("buffer_store", "global_store", "flat_store", "tensor_store"),  # global stores
    8:  ("ds_load", "ds_read"),                                      # LDS loads
    16: ("ds_store", "ds_write"),                                    # LDS stores
}
DEBUG_INSTR_MASK_ALL = sum(DEBUG_INSTR_CLASSES)
DEBUG_INSTR_GLOBAL_MEM = 2 | 4

_PREFIX = "// DebugDisableInstrMask: "


def applyDebugDisableInstrMask(code: str, mask: int) -> str:
  if not mask:
    return code
  if mask & ~DEBUG_INSTR_MASK_ALL:
    raise ValueError(f"DebugDisableInstrMask={mask} has unknown bits (valid: 0..{DEBUG_INSTR_MASK_ALL})")
  prefixes = tuple(p for bit, ps in DEBUG_INSTR_CLASSES.items() if mask & bit for p in ps)
  pattern = re.compile(r"^(\s*)((?:%s)\w*\b.*)$" % "|".join(prefixes), re.MULTILINE)
  return pattern.sub(lambda m: m.group(1) + _PREFIX + m.group(2), code)


def debugDisableInstrMaskConflict(state) -> str | None:
  """Rejection reason if the mask would hang the kernel, else None.

  The Stream-K partial-tile fixup spins on a ready flag in global memory,
  read/written with buffer_load/buffer_store on several arches.
  """
  if state.get("DebugDisableInstrMask", 0) & DEBUG_INSTR_GLOBAL_MEM and requiresPartialReduction(state):
    return ("DebugDisableInstrMask bits 2/4 (global loads/stores) would hang the "
            "Stream-K fixup flag handshake")
  return None
