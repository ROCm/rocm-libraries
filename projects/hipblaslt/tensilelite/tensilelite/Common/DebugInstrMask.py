# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""DebugDisableInstrMask: comment out whole instruction classes in a kernel's assembly.

Timing-only debug aid (e.g. compute-only / memory-only cycle counts); the resulting
kernels compute wrong results. s_waitcnt instructions are kept, they just wait on less.
"""

import re

# bit -> mnemonic prefixes
DEBUG_INSTR_CLASSES = {
    1:  ("v_wmma", "v_swmmac", "v_mfma", "v_smfmac"),             # matrix instructions
    2:  ("buffer_load", "global_load", "flat_load", "tensor_load"),  # global loads (incl. direct-to-LDS)
    4:  ("buffer_store", "global_store", "flat_store", "tensor_store"),  # global stores
    8:  ("ds_load", "ds_read"),                                      # LDS loads
    16: ("ds_store", "ds_write"),                                    # LDS stores
}
DEBUG_INSTR_MASK_ALL = sum(DEBUG_INSTR_CLASSES)

# Instructions whose comment carries this tag are never disabled: the kernel
# branches on the data they move, so removing them would hang it.
DEBUG_INSTR_KEEP_TAG = "[DDIM:keep]"

_PREFIX = "// DebugDisableInstrMask: "


def keepInstrComment(writer, comment: str = "") -> str:
  """Tag `comment` so the mask keeps the instruction. No-op with mask 0."""
  kernel = getattr(writer.states, "kernel", None) or {}
  if not kernel.get("DebugDisableInstrMask", 0):
    return comment
  return f"{comment} {DEBUG_INSTR_KEEP_TAG}" if comment else DEBUG_INSTR_KEEP_TAG


def applyDebugDisableInstrMask(code: str, mask: int) -> str:
  if not mask:
    return code
  if mask & ~DEBUG_INSTR_MASK_ALL:
    raise ValueError(f"DebugDisableInstrMask={mask} has unknown bits (valid: 0..{DEBUG_INSTR_MASK_ALL})")
  prefixes = tuple(p for bit, ps in DEBUG_INSTR_CLASSES.items() if mask & bit for p in ps)
  pattern = re.compile(r"^(\s*)((?:%s)\w*\b(?!.*%s).*)$" % ("|".join(prefixes), re.escape(DEBUG_INSTR_KEEP_TAG)),
                       re.MULTILINE)
  return pattern.sub(lambda m: m.group(1) + _PREFIX + m.group(2), code)
