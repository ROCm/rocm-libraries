/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#include "base.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/common.hpp"

namespace rocisa
{
    // Performs a division using 'magic number' computed on host
    // Argument requirements:
    //   - dstIdx must be two consecutive registers ; on exit the lower one will contain the quotient.  The upper is used as a temp.
    //   - First parm is passed as an integer vgpr index ; remaining are vgpr or sgpr symbolic names
    //   - dstIdx+1 cannot be same as dividend.  dividend+0 can be same as dividend and this may be useful for chaining divides.
    std::shared_ptr<Macro> MacroVMagicDiv(int algo)
    {
        auto macro = Macro(
            "V_MAGIC_DIV",
            {"vgprDstIdx:req", "dividend:req", "magicNumber:req", "magicShift:req", "magicA:req"});
        if(algo == 1) // TODO: remove me
        {
            macro.addT<VMulHIU32>(vgpr("DstIdx+1", 1, true), "\\dividend", "\\magicNumber");
            macro.addT<VMulLOU32>(vgpr("DstIdx+0", 1, true), "\\dividend", "\\magicNumber");
            macro.addT<VLShiftRightB64>(
                vgpr("DstIdx", 2, true), "\\magicShift", vgpr("DstIdx", 2, true));
        }
        else if(algo == 2)
        {
            macro.addT<VMulHIU32>(vgpr("DstIdx+1", 1, true), "\\dividend", "\\magicNumber");
            macro.addT<VMulLOU32>(vgpr("DstIdx+0", 1, true), "\\dividend", "\\magicA");
            macro.addT<VAddU32>(
                vgpr("DstIdx+0", 1, true), vgpr("DstIdx+0", 1, true), vgpr("DstIdx+1", 1, true));
            macro.addT<VLShiftRightB32>(
                vgpr("DstIdx+0", 1, true), "\\magicShift", vgpr("DstIdx+0", 1, true));
        }
        return std::make_shared<Macro>(macro);
    }

    std::shared_ptr<Macro> PseudoRandomGenerator()
    {
        auto macro = Macro("PRND_GENERATOR",
                           {"vgprRand:req", "vgprAcc:req", "vgprTemp0:req", "vgprTemp1:req"});
        macro.addComment0("PRND_GENERATOR: vgprRand=RND(vgprAcc, sgprSeed, vgprTid)");

        // V Logic
        macro.addT<VAndB32>(
            vgpr("Temp0", 1, true), "0xFFFF", vgpr("Acc", 1, true), "vgprTemp0 = vgprAcc & 0xFFFF");
        macro.addT<VLShiftRightB32>(
            vgpr("Temp1", 1, true), 16, vgpr("Acc", 1, true), "vgprTemp1 = vgprAcc >> 16");
        macro.addT<VXorB32>(vgpr("Temp0", 1, true),
                            vgpr("Temp0", 1, true),
                            vgpr("Temp1", 1, true),
                            std::nullopt,
                            "VTemp0 = vgprTemp0 ^ vgprTemp1");
        macro.addT<VAndB32>(
            vgpr("Temp1", 1, true), vgpr("Temp0", 1, true), 31, "vgprTemp1 = vgprTemp0 & 31");
        macro.addT<VLShiftLeftB32>(
            vgpr("Temp1", 1, true), 11, vgpr("Temp1", 1, true), "vgprTemp1 = vgprTemp1 << 11");
        macro.addT<_VLShiftLeftOrB32>(vgpr("Temp0", 1, true),
                                      vgpr("Temp0", 1, true),
                                      5,
                                      vgpr("Temp1", 1, true),
                                      "vgprTemp0 = vgprTemp0 << 5 | vgprTemp1");
        macro.addT<VMulU32U24>(
            vgpr("Temp0", 1, true),
            "0x700149",
            vgpr("Temp0", 1, true),
            "VTemp0 = vgprTemp0 * 0x700149"); // mult lower 24 bits should be enough??
        macro.addT<VMulU32U24>(
            vgpr("Temp1", 1, true),
            229791,
            vgpr("Serial"),
            "VTemp1 = vTid * 229791"); // TODO: use index of C/D instead of local Tid
        macro.addT<VXorB32>(vgpr("Rand", 1, true),
                            "0x1337137",
                            vgpr("Temp0", 1, true),
                            std::nullopt,
                            "VRand = vgprTemp0 ^ 0x1337137");
        macro.addT<VXorB32>(vgpr("Rand", 1, true),
                            vgpr("Rand", 1, true),
                            vgpr("Temp1", 1, true),
                            std::nullopt,
                            "VRand = vgprRand ^ vgprTemp1");
        macro.addT<VXorB32>(vgpr("Rand", 1, true),
                            vgpr("Rand", 1, true),
                            sgpr("RNDSeed"),
                            std::nullopt,
                            "VRand = vgprRand ^ sSeed");

        // NOTE: Some ideas on validation:
        //     1. to test with existing validator: if we use integer initialization pattern and the output is <=16, it will work since no rounding for int up to 16.0 for fp8.
        //     2. We can use same RND (e.g., 0) in both reference and gpu kernel by commenting out following line.
        //     3. If we use 0xFFFFFFFF, cvt_sr will always round the value up. So, tests with existing validator may fail if we don't ensure this in reference kernel of Tensile host
        //     4. A better way to validate:
        //        Fix the value of RNDSeed from the caller, Save the output of this macro-function and compare it with quantization kernel's (TF-SIM's) output.

        return std::make_shared<Macro>(macro);
    }
} // namespace rocisa
