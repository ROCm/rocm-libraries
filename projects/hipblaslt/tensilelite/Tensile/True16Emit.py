################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from rocisa.code import Module
from rocisa.container import SDWAModifiers, VOP3PModifiers
from rocisa.enum import SelectBit
from rocisa.instruction import VCvtF16toF32, VCvtF32toF16, VCvtPkFP8toF32, VCvtPkBF8toF32

# TODO(true16-roadmap):
# Phase 1 (current): keep changes in Tensile codegen control flow.
# - This file centralizes NoSDWA/SDWA branching so call sites can use semantic
#   selectors (low/high) without duplicating syntax details.
# - Scope is intentionally local to codegen for lower regression risk.
#
# Phase 2 (next): extend policy handling into rocisa adapters/helpers.
# - Move more syntax selection logic from Tensile into rocisa-facing helper APIs
#   while keeping emitted assembly behavior equivalent.
# - Validate with focused true16 yaml + existing regression suites.
#
# Phase 3 (broader): evaluate pushing policy into instruction layer.
# - Candidate design points:
#   (a) extend VCvtInstruction to derive true16/sdwa encoding from ISA caps, or
#   (b) add a unified SDWA/NoSDWA decision path close to instruction creation.
# - This stage has wider blast radius (all users of these instructions), so it
#   should be landed incrementally with stage-gated testing before full switch.
#
# Note: each phase expands impact scope; implement and verify incrementally.

def emitCvtF16toF32(module: Module, noSDWA: bool, dst, src, sel: int, comment: str = ""):
  if noSDWA:
    module.add(VCvtF16toF32(dst=dst, src=src, true16=[-1, -1, sel], comment=comment))
  else:
    src_sel = SelectBit.WORD_1 if sel else SelectBit.WORD_0
    module.add(VCvtF16toF32(dst=dst, src=src, sdwa=SDWAModifiers(src0_sel=src_sel), comment=comment))


def emitCvtF32toF16(module: Module, noSDWA: bool, dst, src, sel: int, comment: str = "", useSdwaLegacy: bool = True):
  if noSDWA:
    module.add(VCvtF32toF16(dst=dst, src=src, true16=[sel], comment=comment))
  elif useSdwaLegacy:
    dst_sel = SelectBit.WORD_1 if sel else SelectBit.WORD_0
    module.add(VCvtF32toF16(dst=dst, src=src, sdwa=SDWAModifiers(dst_sel=dst_sel), comment=comment))
  else:
    module.add(VCvtF32toF16(dst=dst, src=src, comment=comment))


def emitPkFp8ToF32(module: Module, noSDWA: bool, dst, src, sel: int, comment: str = ""):
  if noSDWA:
    module.add(VCvtPkFP8toF32(dst=dst, src=src, vop3=VOP3PModifiers(op_sel=[sel]), true16=[-1, -1, sel], comment=comment))
  else:
    src_sel = SelectBit.WORD_1 if sel else SelectBit.WORD_0
    module.add(VCvtPkFP8toF32(dst=dst, src=src, sdwa=SDWAModifiers(src0_sel=src_sel), comment=comment))


def emitPkBf8ToF32(module: Module, noSDWA: bool, dst, src, sel: int, comment: str = ""):
  if noSDWA:
    module.add(VCvtPkBF8toF32(dst=dst, src=src, vop3=VOP3PModifiers(op_sel=[sel]), true16=[-1, -1, sel], comment=comment))
  else:
    src_sel = SelectBit.WORD_1 if sel else SelectBit.WORD_0
    module.add(VCvtPkBF8toF32(dst=dst, src=src, sdwa=SDWAModifiers(src0_sel=src_sel), comment=comment))
