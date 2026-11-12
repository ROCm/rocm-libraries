# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prologue software instruction prefetch for UseSubtileImpl kernels.

When ``SwInstructionPrefetch`` resolves to Relative (explicit ``1``, or Auto on
Stream-K / non-Abs targets), insert ``s_prefetch_inst_pc_rel`` at kernel entry
covering ``label_LoopBeginL`` .. the loop back-edge — one 4 KiB chunk per
prefetch (klen=31), matching gemm_wt opt_v7 / ``patch_iprefetch.py``.

PC-rel offsets must be immediates, so this is a two-pass rewrite on the emitted
assembly string (placeholder assemble → read PCs from objdump → rewrite). No
StinkyTofu involvement.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from Tensile.Common import print2, printWarning
from Tensile.Common.ValidParameters import (
    SW_INSTRUCTION_PREFETCH_OFF,
    normalizeSwInstructionPrefetch,
    resolveSwInstructionPrefetch,
)

# (31 + 1) * 128 B = 4 KiB, the hardware max for one s_prefetch_inst_pc_rel.
_KLEN = 31
_CHUNK = 4096


def findAsmEntryIndex(lines: Sequence[str]) -> Optional[int]:
  """Index of the first real ``s_``/``v_`` after ``label_ASM_Start`` (skip macros)."""
  try:
    start = next(i for i, l in enumerate(lines) if l.startswith("label_ASM_Start:"))
  except StopIteration:
    return None
  inMacro = False
  for i in range(start, len(lines)):
    l = lines[i].strip()
    if l.startswith(".macro"):
      inMacro = True
    elif l.startswith(".endm"):
      inMacro = False
    elif not inMacro and re.match(r"^[sv]_\w+", l):
      return i
  return None


def withPrefetchLines(lines: List[str], entry: int, offsets: Sequence[int]) -> List[str]:
  pf = [f"s_prefetch_inst_pc_rel 0x{o:x}, null, {_KLEN}  // SIP prologue: main-loop chunk {k}"
        for k, o in enumerate(offsets)]
  return lines[:entry] + pf + lines[entry:]


def loopCoverChunkCount(loopAt: int, backAt: int) -> int:
  """Number of 4 KiB prefeches needed to cover [loopAt, backAt+4)."""
  return max(1, (backAt + 4 - loopAt + (_CHUNK - 1)) // _CHUNK)


def pcRelOffsets(loopAt: int, pfAt: Sequence[int]) -> List[int]:
  """Offsets so prefetch *k* targets ``loopAt + k * 4096``."""
  return [loopAt + _CHUNK * k - pfAt[k] for k in range(len(pfAt))]


def parseLoopSpan(disasm: str):
  """Return ``(loopAt, backAt)`` from llvm-objdump ``-d`` text, or ``None``."""
  loopM = re.search(r"^([0-9a-f]+) <label_LoopBeginL>:", disasm, re.M)
  backMs = re.findall(r"s_cbranch_scc\d label_LoopBeginL\s+// ([0-9A-F]+):", disasm)
  if not loopM or not backMs:
    return None
  return int(loopM.group(1), 16), int(backMs[-1], 16)


def parsePrefetchPcs(disasm: str) -> List[int]:
  return [int(a, 16) for a in re.findall(r"s_prefetch_inst_pc_rel.*// ([0-9A-F]+):", disasm)]


def insertPrologueInstructionPrefetch(asm: str, kernel, assembler, isaVersion) -> str:
  """Rewrite ``asm`` with entry PC-rel prefeches covering the main loop, or return as-is."""
  mode = normalizeSwInstructionPrefetch(
      kernel.get("SwInstructionPrefetch", SW_INSTRUCTION_PREFETCH_OFF))
  if mode == SW_INSTRUCTION_PREFETCH_OFF:
    return asm

  # Relative (and Auto→Relative): prologue cover. Absolute is a different
  # encoding / SGPR protocol — not implemented on the subtile path.
  isGfx1250 = tuple(isaVersion) == (12, 5, 0)
  swpRel, swpAbs = resolveSwInstructionPrefetch(
      mode, isGfx1250, bool(kernel.get("StreamK", 0)),
      kernel["ProblemType"]["DataType"].isDouble())
  if not swpRel or swpAbs:
    return asm
  if "label_LoopBeginL:" not in asm:
    print2("[SIP prologue] skip: no label_LoopBeginL")
    return asm

  lines = asm.split("\n")
  entry = findAsmEntryIndex(lines)
  if entry is None:
    print2("[SIP prologue] skip: no entry instruction")
    return asm

  asmPath = Path(assembler.path)
  objdump = asmPath.parent / "llvm-objdump"
  if not objdump.is_file():
    objdump = Path(os.environ.get("LLVM_BIN", "/opt/rocm/lib/llvm/bin")) / "llvm-objdump"
  targetGfx = f"gfx{isaVersion[0]}{isaVersion[1]}{isaVersion[2]}"
  wave = int(kernel.get("WavefrontSize", 32))

  try:
    with tempfile.TemporaryDirectory(prefix="sip_prologue_") as td:
      tdPath = Path(td)
      srcPath = tdPath / "kernel.s"
      objPath = tdPath / "kernel.o"

      def assemble_and_disassemble(text_lines: List[str]) -> str:
        srcPath.write_text("\n".join(text_lines))
        assembler(targetGfx, wave, str(srcPath), str(objPath))
        return subprocess.run(
            [str(objdump), "-d", f"--mcpu={targetGfx}", str(objPath)],
            capture_output=True, text=True, check=True).stdout

      # Discover loop span with two placeholders, then size the burst.
      dis = assemble_and_disassemble(withPrefetchLines(lines, entry, [0, 0]))
      span = parseLoopSpan(dis)
      if span is None:
        print2("[SIP prologue] skip: could not locate LoopBeginL span in objdump")
        return asm
      loopAt, backAt = span
      need = loopCoverChunkCount(loopAt, backAt)
      dis = assemble_and_disassemble(withPrefetchLines(lines, entry, [0] * need))
      loopAt = parseLoopSpan(dis)[0]
      pfAt = parsePrefetchPcs(dis)
      if len(pfAt) != need:
        print2(f"[SIP prologue] skip: expected {need} prefetch sites, got {len(pfAt)}")
        return asm
      offsets = pcRelOffsets(loopAt, pfAt)
      out_lines = withPrefetchLines(lines, entry, offsets)
      dis2 = assemble_and_disassemble(out_lines)
      if parsePrefetchPcs(dis2) != pfAt:
        print2("[SIP prologue] skip: prefetch PCs moved after offset rewrite")
        return asm
      print2(f"[SIP prologue] inserted {need} s_prefetch_inst_pc_rel covering "
             f"loop 0x{loopAt:x}..0x{backAt:x} ({backAt + 4 - loopAt} B)")
      return "\n".join(out_lines)
  except Exception as e:
    printWarning(f"[SIP prologue] failed ({e}); emitting kernel without prologue prefetch")
    return asm
