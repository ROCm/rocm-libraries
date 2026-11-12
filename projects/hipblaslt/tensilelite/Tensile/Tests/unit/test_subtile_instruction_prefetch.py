# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
#   pytest Tensile/Tests/unit/test_subtile_instruction_prefetch.py -v

from Tensile.Components.Subtile.InstructionPrefetch import (
    findAsmEntryIndex,
    withPrefetchLines,
    loopCoverChunkCount,
    pcRelOffsets,
    parseLoopSpan,
    parsePrefetchPcs,
    insertPrologueInstructionPrefetch,
)
from Tensile.Common.ValidParameters import (
    SW_INSTRUCTION_PREFETCH_OFF,
    SW_INSTRUCTION_PREFETCH_RELATIVE,
)


def test_findAsmEntryIndex_skips_macro():
  lines = [
      "label_ASM_Start:  /// Main body",
      ".macro V_MAGIC_DIV vgprDstIdx:req",
      "    s_nop 0",
      "v_mul_hi_u32 v[0], 1, 2",
      ".endm",
      "s_load_b32 s15, s[0:1], 0",
  ]
  assert findAsmEntryIndex(lines) == 5


def test_withPrefetchLines_inserts_at_entry():
  lines = ["label_ASM_Start:", "s_load_b32 s0, s[0:1], 0", "s_endpgm"]
  out = withPrefetchLines(lines, 1, [0x100, 0x1100])
  assert out[1].startswith("s_prefetch_inst_pc_rel 0x100")
  assert out[2].startswith("s_prefetch_inst_pc_rel 0x1100")
  assert out[3] == "s_load_b32 s0, s[0:1], 0"


def test_loopCoverChunkCount():
  # 1 B .. 4096 B → 1 chunk; 4097 B → 2
  assert loopCoverChunkCount(0x1000, 0x1000) == 1  # 4 B span
  assert loopCoverChunkCount(0x1000, 0x1FFC) == 1  # exactly 4 KiB
  assert loopCoverChunkCount(0x1000, 0x2000) == 2


def test_pcRelOffsets():
  assert pcRelOffsets(0x1840, [0x0, 0x8]) == [0x1840, 0x2840 - 0x8]


def test_parseLoopSpan_and_prefetch_pcs():
  dis = """
0000000000001840 <label_LoopBeginL>:
	s_nop 0
	s_cbranch_scc0 label_LoopBeginL                            // 000000003B34:
	s_prefetch_inst_pc_rel 0x1850, null, 31                    // 000000000000:
	s_prefetch_inst_pc_rel 0x2848, null, 31                    // 000000000008:
"""
  assert parseLoopSpan(dis) == (0x1840, 0x3B34)
  assert parsePrefetchPcs(dis) == [0x0, 0x8]


class _DT:
  def isDouble(self):
    return False


def test_insert_off_is_noop():
  asm = "label_ASM_Start:\ns_nop 0\nlabel_LoopBeginL:\ns_endpgm\n"
  kernel = {"SwInstructionPrefetch": SW_INSTRUCTION_PREFETCH_OFF,
            "StreamK": 3, "ProblemType": {"DataType": _DT()}, "WavefrontSize": 32}
  assert insertPrologueInstructionPrefetch(asm, kernel, assembler=None, isaVersion=(12, 5, 0)) == asm


def test_insert_relative_without_loop_is_noop():
  asm = "label_ASM_Start:\ns_nop 0\ns_endpgm\n"
  kernel = {"SwInstructionPrefetch": SW_INSTRUCTION_PREFETCH_RELATIVE,
            "StreamK": 3, "ProblemType": {"DataType": _DT()}, "WavefrontSize": 32}
  assert insertPrologueInstructionPrefetch(asm, kernel, assembler=None, isaVersion=(12, 5, 0)) == asm
